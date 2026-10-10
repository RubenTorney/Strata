# Asynchronous adaptive tier in the arena mode — RTX 3090 + Ryzen 5 5600X, Windows 11

Branch `perf/async-adapt-arena` (on v0.1.42). Measured 2026-10-10/11 by the branch author on one PC.

## Machine

- RTX 3090 24 GB (driver 616.64, 350 W limit, **also drives the desktop**, 2560 px), PCIe 4.0 x16
- Ryzen 5 5600X (6 cores / 12 threads, AVX2, no iGPU), 2 x 32 GB DDR4-3600
- Windows 11 Pro 22631. The arena: 30 of ~47 GiB page-locked (`cudaHostRegister` refuses the rest), large pages refused (error 1450)
- Qwen3.8-Flash-Next GSQ-RCO IQ3_S, `--max-context 262144 --kv int8`, MTP drafter (`--spec 4 --mtp`), draft vocabulary `fr`

## Arms

| Arm | Engine | Settings |
|---|---|---|
| `base0138` | release 0.1.38 (CUDA 13.0) | the PC's config before any tuning: `--expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp --max-context 262144 --kv int8` |
| `rel` | release 0.1.42 (CUDA 13.0) | tuned: + `--vram-reserve-mib 1200 --kv-grow`, `draft_vocab fr`, learned expert profile, calibration (`--adapt-every 1 --adapt-swaps 80 --adapt-decay 0.97`) |
| `loc` | this branch, local build (CUDA 12.8, VS 2022 17.14, sm_86) | as `rel` |
| `async` | this branch | as `rel` + `--adapt-async 1` |
| `stack2` | this branch | as `async` + `--pool-tasks 36 --lookup-chain 2 --mtp-q4 head`, `STRATA_GDN_SPLIT=1 STRATA_DF_BRANCH=1 STRATA_TSUM=1 STRATA_ATTN_LANECELL=1 STRATA_IQ3S_MT1=1` |

## Protocol

`strata_ab2.py` (this folder) drives `serve/server.py` on port 8080. Every session is a fresh engine; 2 warm-up requests are
not counted. Temperature 0.7, top_p 0.8, 256 tokens max, thinking off. Decode tok/s is the engine's own
`strata serve: prompt ... generated in ... (X tok/s)` line.

- **Cold short prompts**: 12 distinct prompts (FR/EN, code, prose), each sent **once** per session with a fixed seed, so no prompt
  finds the expert cache already adapted to it (a prompt sent twice in a row runs ~25 % faster the second time: an A/B that
  always sends arm B second is not valid). 2 rounds, the arms' order reversed in round 2; per-prompt mean over the rounds,
  paired ratio against the reference arm.
- **Context sweep**: one session per arm and size group; at each size one prompt (unique nonce first, repository text) is read
  once, then answered 3 times with seeds 1001/1002/1003 on the cached prefix; decode = mean of the 3.

## Cold short prompts (12 prompts x 2 rounds, paired)

| Arm | Median tok/s | vs `base0138` | vs `rel` | Prompts faster than `rel` |
|---|---:|---:|---:|---:|
| `base0138` | 57.0 | 1.00 | 0.65 | 0/12 |
| `rel` | 88.8 | **x1.53** | 1.00 | — |
| `stack2` | 102.8 | **x1.73** | **x1.15** | 12/12 |

Separate runs, same protocol (`cold_release_local_async.jsonl`, `cold_async_stacks.jsonl`):

| Arm | Median tok/s | Paired ratio | Prompts faster |
|---|---:|---:|---:|
| `loc` vs `rel` | 92.8 vs 83.2 | x1.08 | 10/12 |
| `async` vs `loc` | 98.7 vs 92.8 | **x1.11** | 12/12 |
| `async` + `STRATA_ADAPT_BOUNCE=1 STRATA_PCIE_MIN1=1` vs `loc` | 99.1 | x1.10 | 12/12 (no gain over `async`: both left off) |
| `stack1` (`async` + GDN_SPLIT, DF_BRANCH, TSUM, ATTN_LANECELL) vs `async` | 100.9 vs 99.8 | x1.02 | 7/12 (noise) |
| `stack2` vs `async` | 104.9 vs 99.8 | x1.04 | 11/12 |

`STRATA_DECODE_TIMING=1`, one cold prompt: blocking tier 25.9 ms/window of which ~3.2 ms outside verify and draft (the
per-window adapt thread and its copies); `--adapt-async 1`: 21.6 ms/window, adapt wait 0.41 ms, join 0.

## Context sweep (prompt read once, decode = mean of 3 seeded answers)

| Prompt tokens | Prefill tok/s base / rel / stack2 | Time to first token base → stack2 | Decode tok/s base / rel / stack2 | stack2 vs base |
|---:|---|---|---|---:|
| 3,859 | 1360 / 1440 / 1433 | 2.8 s → 2.7 s | 72.9 / 101.6 / 104.3 | x1.43 |
| 14,123 | 2005 / 2097 / 2095 | 7.0 s → 6.7 s | 81.5 / 113.1 / 102.6 | x1.26 |
| 27,536 | 2018 / 2094 / 2105 | 13.6 s → 13.1 s | 81.9 / 109.5 / 101.7 | x1.24 |
| 55,633 | 2110 / 2201 / 2206 | 26.4 s → 25.2 s | 78.6 / 106.2 / 105.2 | x1.34 |
| 116,832 | 1960 / 2100 / 2101 | 59.6 s → 55.6 s | 71.1 / 94.1 / 95.4 | x1.34 |
| 214,795 | 1758 / 1996 / 1994 | 122.2 s → 107.7 s | 62.3 / 83.5 / 81.8 | x1.31 |
| 255,805 | 1663 / 1944 / 1942 | 153.8 s → 131.7 s | 67.8 / 87.4 / 84.9 | x1.25 |

With `--kv-grow` the K/V takes expert-cache slots as it grows: 0.46 GiB at 32K (108 slots), 1.57 GiB at 123K (432 slots),
2.84 GiB at 221K (1,376 slots, decode hit rate 90.8 %); the base arm reserves the whole 3.35 GiB at start.

**Reading:** the branch's gain is on cold starts (a new subject: +15 % over the tuned release, +73 % over the untuned
0.1.38). After a long prompt has been read, the prompt itself has already moved the experts it uses into VRAM (hit rate
97-99 %), the blocking tier has little left to swap, and `stack2` is level with the tuned release (within the 3-sample spread,
about ±10 %; the 7-10 % gap at 14-28K in this sweep does not reproduce, see the attribution run below). The prompt read is unchanged
(the branch does not touch the prefill).

### After a long prompt: which setting? (`context_attribution.jsonl`)

Same sweep at 14K and 28K, one session per arm, back to back: decode mean of 3 seeded answers.

| Arm | 14,120 tokens | 27,535 tokens |
|---|---:|---:|
| `rel` | 100.8 | 104.2 |
| `async` | 96.6 | 103.1 |
| `stack1` | 96.7 | 97.8 |
| `stack2` | 96.1 | 97.2 |

The first answer of every arm is the slowest (81-86 tok/s at 14K), the spread within an arm is ±10 %: the 7-10 % gap of
the first sweep does not reproduce beyond ~4-6 %, and no single setting stands out. After a long prompt the arms are level
within the noise.

## Robustness

- 63K-token prompt then 4 follow-up turns in the same conversation (`stress stack2`): K/V grown to 65,536 cells, answers coherent,
  decode 80-93 tok/s, no `ERR`.
- Greedy answers are not byte-identical between two fresh engines with `stack2` (0 of 6 prompts) — nor with the unpatched local
  build (1 of 6): the CPU/GPU split of the misses and the adaptive tier already depend on timing on this machine.

## Raw data

`cold_base_tuned_patched.jsonl`, `cold_release_local_async.jsonl`, `cold_async_stacks.jsonl`, `context_sweep.jsonl`,
`determinism.jsonl`; harness `strata_ab2.py` (`ab`, `ctx`, `det`, `stress`, `smoke`, `report`).

## Round 2 (2026-10-11): profile-guided changes

Profile (`STRATA_VERIFY_PROFILE=1`, cold prompt, `stack2`): 21.4 ms/window = GPU chain ~13 ms (GDN 8.2 + QSA 3.3 + head 0.8;
the hyper-connection reads run at ~28 % of roofline, ~37 kernels per GDN layer) + CPU misses 5-14 ms + draft 1.4 ms. Arms on
top of `stack2`, same cold protocol (12 prompts, paired vs `stack2`):

| Change | Rounds | Ratio vs `stack2` | Prompts faster | Kept |
|---|---:|---:|---:|---|
| `STRATA_GR_V3=1 STRATA_EXPERT_V2=1 STRATA_QFUSE=1 STRATA_MMVF_ROWS=1` | 2 | 0.93 | 3/12 | no |
| `STRATA_GR_128=1` (new: 128-thread staged hyper-connection read, bitwise self-test at start) | 1 | ~0.99 | — | no (off) |
| `+ STRATA_ADAPT_ASYNC_BATCH=16` | 1 | ~0.84 | — | no |
| `STRATA_ADAPT_ASYNC_DEMAND=1` (new: round cap from CPU misses) | — | inert (misses/window x4 > 80) | — | no (off) |
| `STRATA_PIN_BY_NEED=1 STRATA_ARENA_PIN_GIB=29` (new: page-aligned arena stride, register the profile's non-resident experts instead of the first 30 GiB: 28.97 GiB, 92 % of them, all 48 layers get a PCIe share) | 2 | 0.995 | 6/12 | with the next rows |
| pin by need + `STRATA_PCIE_MIN1=1` (PCIe share floor of 1 at >= 2 misses) | 2 | 1.017 | 7/12 | — |
| pin by need + `--pcie-frac 0.6` | 2 | 1.023 | 10/12 | — |
| pin by need + both (`p1both`) | 1 | 1.026 | 10/12 | **yes** |

`p1both` 63K prompt + 4 turns: prefill 2,310 tok/s (2,180 without), decode 85-99 tok/s, no `ERR`. Load speed unchanged
(6.7 GiB/s; the padded layout reads through the file cache). Raw: `round2_*.jsonl`.
