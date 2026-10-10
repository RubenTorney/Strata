# Plan — asynchronous adaptive tier in the arena mode (from Fable's review, 2026-10-10)

Line numbers: src/program/generate.cpp at v0.1.42 (61b3fb5d) unless noted.

## Why
Blocking tier, arena mode: each window creates a std::thread running adapt() (8153-8279) during commit+draft, joined at
11685; the next window waits for its copies at 11574 (apply_pending(true) -> cudaEventSynchronize(adapt_ev) 8132, then
res_upload = synchronous cudaMemcpy 96 KB 5988-5991). ArenaExpertSource::pinned() (expert_source.cpp:4608-4612): only a
30 GiB prefix is registered (layers 0-31); for layers 32-47 adapt_copy_h2d (cudaMemcpyAsync from pageable memory, 193) is
synchronous. Estimated hole ~3.2 ms/window (~40 swaps/window cold) out of 25.9 ms. Target: ~+11 % cold decode.
STRATA_ADAPT_LAG=2 halves the swap rate (adapt() returns early while `pending` is not empty, 8154) -> -8 %: not the fix.

## Step 0 — Instrumentation (do first)
In the serial --serve loop, time (a) the wait 11573-11574 and (b) the join 11685 into `dt_adapt_wait`, `dt_adapt_join`;
count swaps per round (swaps.size() in adapt(), captured int64_t); append "adapt wait %.2f join %.2f, %.1f swaps/window"
to the `strata decode timing` line (11714-11719, enabled by STRATA_DECODE_TIMING=1, 10853).

## Step 1 — Open the async tier to the arena mode (gates)
- 6505-6519: replace `!o.resident_cpu_experts ? "it needs the resident RAM mode"` by
  `!o.resident_cpu_experts && srcp != &arena_src ? ...`; add `: multi_gpu && srcp == &arena_src ? "not with a layer split in the arena mode"`.
- 6702: `if (o.adapt_async && !src.complement_ready() && srcp != &arena_src) adapt_async_off(...)`.
- 8332: activation -> `o.adapt_async && !drive.d.usage.empty() && ((src.complement_ready() && src.exchange_capacity() >= 2*a_cap) || srcp == &arena_src)`.
  Declare `const bool a_arena = srcp == &arena_src;` before. Log 8336: add "(arena: copy-in only)".

## Step 2 — State machine (8354-8604) for the arena
- a_pick 8385: `const bool x = !a_arena && src.has_resident(...) && !src.has_resident(...)` -> exchange=false in arena => a_back (8390) posts nothing.
- a_from 8409-8421: first line `if (a_arena) return arena_src.blob_stable(w.layer, w.in);` (pageable source blocks the job
  thread, not the host). blob() increments reads_ (non-atomic, expert_source.cpp:4628) from the job thread: add a
  `const uint8_t* ArenaExpertSource::blob_at(l,e) const` without the counter (expert_source.hpp ~908 / .cpp ~4623) and use it.
- AState::RamFence 8590: `if (a_arena) { astate = AState::Commit; continue; }` instead of posting commit_copies.
  AState::Commit 8597: `if (!a_arena) src.commit_flip();`.
- Keep res_upload 8559/8575.

## Step 3 — Mandatory fences (latent bug + port)
`auto adapt_quiesce = [&] { cudaDeviceSynchronize(); apply_pending(true); if (!adapt_tick(true)) a_err = true; };`
used in the three `quiesce` lambdas of kvg_ensure: 9717, 9967, 11592 (comment 6313-6314 requires swaps to land; today
only apply_pending(true) is called -> with --kv-grow an in-flight swap can write a slot lent to the K/V). Surface a_err as ERR at the next tick (8502 does).
Document at the top of block 8280 the invariant: serial loop, at the tick nothing in flight reads a slot (run ->
cudaStreamSynchronize(cs_) verify.cpp:2116; draft synchronous; commit does not read the expert cache); the job thread never
touches host_res / drive.d.usage (snapshots a_usage / a_res).

## Step 4 — One round per window (inline pick, ~40 lines)
STRATA_ADAPT_PICK_INLINE (default 1 in arena): in `case Idle`, run a_pick() on the main thread, then go straight into the
CopyBack logic (mark `out`, res_upload, ajob->post(a_copy_in), astate = SwapIn) in the same tick. Next tick SwapIn -> Idle ->
`continue` starts a new round (8601). If the inline pick costs > 0.5 ms (measure), keep it on the job.

## Step 5 — Reproducible "wait" variant (~10 lines)
STRATA_ADAPT_ASYNC_WAIT (default 1 in arena): in `case SwapIn`, call a_ready(true) (cudaEventSynchronize) instead of
a_ready(drain). ~0 cost when the round (<= 80 x 2 MB) lands before the window ends; makes the window index where an
expert becomes GPU-resident independent of timing -> reproducible run to run.

## Step 6 — Copy cap per tick (copy-engine contention, ~30 lines)
STRATA_ADAPT_ASYNC_BATCH (default 32): a_copy_in posts only the first B remaining swaps; SwapIn admits the landed ones and
posts the next batch; Idle only when aswaps is empty. Check "GPU-reach wait" does not grow vs the blocking tier.

## Step 7 (optional, only if measured useful) — pinned bounce buffers for layers 32-47.

## Step 8 — stats line 11967-11970: add "(arena, copy-in only, batch B, wait W)".

## Tests
1. Smoke: short prompt with --adapt-async 1: log "adaptive tier asynchronous (--adapt-async 1)" and rounds/experts swapped;
   no ERR, no "residency table changed during the window".
2. Determinism: (a) --adapt-swaps 0 async vs blocking -> identical tokens; (b) wait variant, greedy, two fresh engines -> identical;
   (c) vs blocking tier: first-divergence index / agreement (expected > 99 % over 256 tokens).
3. Cold A/B (done by the orchestrator): 12 distinct prompts, fresh engine, paired ratio; expect >= +8 %; GPU-reach wait must not grow > ~0.3 ms.
4. Stress: 63k prompt (lend + kvg_ensure grows) then 4-5 short turns; STOP mid-decode then an immediate next request.
