# Round 2 plan (Fable max review, 2026-10-11) — base: branch perf/async-adapt-arena

Profile (cold prompt, 21.4 ms/window): GPU-reach 13.0 ms = GDN 8.2 + QSA 3.3 + head 0.8; hyper-connection reads (hc 2 x 13.1 MB
BF16 per layer) run at 28 % of roofline = 4.6 ms/window (gr_down_staged_kernel on DOWN_BLOCKS+1 = 41 blocks of 256 threads,
fused_gr.cu:35, :908-915); ~37 kernels per GDN layer (no PDL on sm_86) = 4-5 ms of gaps. waitB grows with the async tier's
copies (73 swaps/window = 149 MB = 5.7 ms of link, waitB 12 ms vs 2 ms at 40 swaps). CPU misses: 105 us/expert, instruction
bound. PCIe share exists only for pinned layers: pcie_layer() = device_alias(layer,0) (expert_source.cpp:3370-3374,
4613-4626); the pinned prefix = 32 slices = layers 0-31, so layers 32-47 are 100 % CPU, while the 30 GiB pinned hold ~10 GiB of
GPU-resident blobs never read by DMA.

## P1 — Pin by need (worktree strata-p1, branch perf/pin-by-need) — 250-350 lines
1. src/core/pinned.cu: new PinnedArena constructor taking a list of byte ranges [(offset, length)]: cudaHostRegister each range
   (page-rounded; ranges are >= one blob apart, never overlapping) BEFORE any page is touched (comment :364-366), in the given order,
   stop at the first refusal; registered bytes become a list of ranges + bitmap; lock_resident (memory.cpp:23-58) only on the holes.
   The WDDM budget (~RAM/2 = 30 GiB, pinned.cu:309-336) is not raisable: we choose WHAT to pin.
2. src/core/expert_source.cpp ArenaExpertSource (:4420-4560, :4613-4626): set_pin_priority(ranked) = (layer, expert) pairs from the
   expert profile (loaded before the arena, log line ~10) that are NOT GPU-resident at start, by decreasing rank, merged into ranges
   with neighbours; pinned(l,e) by bitmap; device_alias(l,e) by a range table (device base + offset); pcie_layer(l) = at least one
   pinned range in the layer.
3. src/program/generate.cpp: pass the profile / initial host_res before arena_src.load; log "N ranges, X GiB, Y % of non-resident pinned".
   Gate: STRATA_PIN_BY_NEED=1 (default off) so the old prefix pinning stays byte-identical.
4. Tests (orchestrator runs them): --pcie-frac 0 --adapt-swaps 0 STRATA_ROUTE_TAIL_SKIP=0 -> output identical to old build;
   STRATA_DECODE_TIMING "PCIe x.xx per layer-window" rises, "CPU experts" falls; cold A/B; STRATA_PREFILL_TIMING at 14K/55K.

## P2 — Async tier without contention (worktree strata-p2, branch perf/async-batch) — 60-80 lines
1. generate.cpp:8376: default STRATA_ADAPT_ASYNC_BATCH = 16 in the arena (the SwapIn -> post-next split exists :8670-8675).
2. a_pick (:8415-8450): round cap = min(adapt_swaps, max(8, 4 x misses of the last window)) using the delta of drive.d.multi_misses;
   in Idle (:8587) no round if < 2 misses in the last window. Env STRATA_ADAPT_ASYNC_DEMAND=1 (default on in arena; =0 restores).
3. If pageable copies remain (layers 32-47 without P1), make STRATA_ADAPT_BOUNCE default on when the blob is !pinned.
4. Tests: GPU profile of the first cold prompt: waitB from 9-12 to < 4 ms/window at constant GPU-reach; swaps/window ~0 when hit > 97 %.

## P3 — Hyper-connection read (worktree strata-p3, branch perf/hc-read) — 60-200 lines
0. Env already there (orchestrator A/Bs): STRATA_GR_V3=1 (fused_gr.cu:1685-1748, 164 blocks, 2 kernels, different fp32 sum order),
   STRATA_HC_Q8=1 (changes rounding).
1. fused_gr.cu: variant gr_down_staged_kernel<..., 128 threads> (WARPS=4 -> DOWN_BLOCKS 80 + 1 inject), stage_htile parametrized by
   thread count, same chunk order per lane (lane + 32q) -> bitwise identical; launch_multi (:867-938) picks it when
   multiProcessorCount >= 64 (env STRATA_GR_128=1 to force, =0 to disable; default on if bitwise selftest passes);
   add kHcStaged4 to fused_gr_selftest (:1812) and gr_parity.
2. Measure: profile columns "hc0 down" / "hc-read1+router" (expected 0.64 -> ~0.35 ms, and -0.3 on hc1), then cold A/B.

## Not to do
Single-GPU 2-window pipelining (needs the committed GDN state of window K per layer: -7 % measured for --spec-split); tree drafts.
