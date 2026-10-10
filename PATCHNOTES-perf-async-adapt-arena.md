# Patch notes — branch `perf/async-adapt-arena` (on Strata v0.1.42)

One PC: RTX 3090 24 GB (also drives the desktop, 350 W), Ryzen 5 5600X (6 cores, AVX2), 2 x 32 GB DDR4-3600, Windows 11.
Model Qwen3.8-Flash-Next GSQ-RCO **IQ3_S**, 262K context, int8 K/V, MTP drafter. All numbers below are measured on that PC;
method, raw data and harness: [bench/results/2026-10-11-rtx3090-5600x-async-arena](bench/results/2026-10-11-rtx3090-5600x-async-arena/README.md).

## Gains

Decode speed, short prompts on a new subject (12 distinct prompts, each sent once per fresh engine, 2 rounds in reversed
order, compared prompt by prompt; temperature 0.7, 256 tokens, thinking off):

| Version | Decode tok/s (median) | vs start |
|---|---:|---:|
| Start: engine 0.1.38, config before any tuning | 57.0 | — |
| Engine 0.1.42 + settings + calibration (config only) | 88.8 | **x1.53** |
| + this branch: asynchronous adaptive tier + upstream opt-in switches | 102.8 | **x1.73** |
| + pin by need + PCIe share | ~105 (+2-3 %) | **~x1.8** |

Context sweep (prompt read once, then 3 seeded answers on the cached prefix; start vs the branch without pin by need):

| Prompt tokens | Prompt read (time to first token) | Decode tok/s | Decode gain |
|---:|---|---|---:|
| 3,859 | 2.8 s → 2.7 s | 72.9 → 104.3 | x1.43 |
| 27,536 | 13.6 s → 13.1 s | 81.9 → 101.7 | x1.24 |
| 116,832 | 59.6 s → 55.6 s | 71.1 → 95.4 | x1.34 |
| 214,795 | 122.2 s → 107.7 s | 62.3 → 81.8 | x1.31 |
| **255,805** | **153.8 s → 131.7 s** | **67.8 → 84.9** | **x1.25** |

Pin by need adds +6 % to the prompt read (63K tokens: 2,180 → 2,310 tok/s).

## 1. Settings (no code)

| Setting | What it does | Effect |
|---|---|---|
| Engine 0.1.38 → 0.1.42 | upstream release (missed tail experts skipped, ...) | measured together with the next rows |
| `"draft_vocab": "fr"` | the draft head scores a smaller vocabulary subset (setup's `fr` list) | not measured alone |
| `--max-context 262144 --kv int8 --kv-grow` | K/V allocated as the context grows; the expert cache gets the rest | 6,853 → 8,289 expert slots |
| `--vram-reserve-mib 1200` | VRAM kept free for the desktop | free VRAM 246 → ~930 MiB, no WDDM paging |
| `"expert_profile_save": "expert-profile-learned.bin"` | the adaptive tier's learned profile is reused at the next start | faster warm-up |
| Calibration (`setup.py --calibrate`) | `--adapt-every 1 --adapt-swaps 80 --adapt-decay 0.97`, `--spec-min-p 0.5` | 88.5 → 100.7 tok/s on its own prompt |
| Upstream opt-in switches | `--pool-tasks 36 --lookup-chain 2 --mtp-q4 head`, `STRATA_GDN_SPLIT=1 STRATA_DF_BRANCH=1 STRATA_TSUM=1 STRATA_ATTN_LANECELL=1 STRATA_IQ3S_MT1=1` | +4.3 % (11/12 prompts faster) |
| `--pcie-frac 0.6` (with pin by need) | larger PCIe share of the missed experts | see section 2 |

Settings + engine update together: x1.53 (57.0 → 88.8). A local build of 0.1.42 (VS 2022 + CUDA 12.8, sm_86 only) is
already 8 % faster than the release (92.8 vs 83.2 in a separate run of the same protocol).

## 2. Code changes (this branch)

| Change | Switch | Effect |
|---|---|---|
| **Asynchronous adaptive tier in the arena mode**: the swaps of the expert cache (copy-in only: the arena holds every expert) run between verify windows on a job thread instead of a thread created and joined every window; the pick runs inline; a swapped-in expert becomes resident only once its copy has landed | `--adapt-async 1` (`STRATA_ADAPT_ASYNC_WAIT`, `_PICK_INLINE`, `_BATCH`, `_DEMAND`) | **+11 %** cold (25.9 → 21.6 ms per window; 12/12 prompts faster) |
| Fences: every K/V grow and the `VRAM` command wait for the round in flight; a failed round still drains its copies | (with `--adapt-async 1`) | fixes a latent slot overwrite with `--kv-grow` |
| **Pin by need**: page-aligned arena stride (+0.15 % RAM); page-lock the profile's non-GPU-resident experts by rank (29 GiB cap on Windows) instead of the first 30 GiB, so all 48 layers get a PCIe share | `STRATA_PIN_BY_NEED=1`, `STRATA_ARENA_PIN_GIB` | with the next row and `--pcie-frac 0.6`: **+2-3 %** cold, +6 % prompt read |
| PCIe share floor: one miss over PCIe as soon as a layer has two | `STRATA_PCIE_MIN1=1` | part of the row above |
| Timing: `adapt wait / join, swaps/window` in the `STRATA_DECODE_TIMING` line | — | diagnostics |
| 128-thread hyper-connection read (bitwise self-test at start) | `STRATA_GR_128=1` | neutral (left off) |
| Pinned bounce buffers for adaptive copies | `STRATA_ADAPT_BOUNCE=1` | neutral (left off) |

Everything is off by default: without the switches the engine behaves as v0.1.42. Reviewed (read-only) after each step.

## Recommended config

```json
"args": [ "...", "--expert-cache", "auto", "--prefill", "auto", "--spec", "4", "--mtp", "<mtp dir>",
          "--max-context", "262144", "--kv", "int8", "--vram-reserve-mib", "1200", "--kv-grow",
          "--spec-min-p", "0.5", "--adapt-every", "1", "--adapt-swaps", "80", "--adapt-decay", "0.97",
          "--adapt-async", "1", "--pool-tasks", "36", "--lookup-chain", "2", "--mtp-q4", "head", "--pcie-frac", "0.6" ],
"env": { "STRATA_GDN_SPLIT": "1", "STRATA_DF_BRANCH": "1", "STRATA_TSUM": "1", "STRATA_ATTN_LANECELL": "1",
         "STRATA_IQ3S_MT1": "1", "STRATA_PIN_BY_NEED": "1", "STRATA_ARENA_PIN_GIB": "29", "STRATA_PCIE_MIN1": "1" },
"draft_vocab": "fr", "expert_profile_save": "expert-profile-learned.bin"
```

Build: `tools\build-local-vs2022.bat` (Visual Studio 2022 Build Tools + CUDA 12.8; VS 2019 cannot compile the CUDA
code, which needs C++20). Point the config's `"exe"` at the built `strata.exe` and `"lib_dirs"` at the CUDA 12.8 `bin`.

## Tested, no gain (cold, paired)

`--spec-min-p 0.7`, `--spec 3` / `5`, `STRATA_ROUTE_TAIL_SKIP=6`, `--host-core last`, `--aux-cpus auto`,
`STRATA_SPEC_COUPLED=1`, `STRATA_ADAPT_LAG=2`, `--adapt-every 4 --adapt-swaps 32`, `--pcie-frac 0.9` alone,
`--pool-workers` 6 / 8 / 11 (SMT siblings: 74 and 54 tok/s at 8 / 11), `--resident-experts` (+ async, + rotate),
`STRATA_GR_V3 / EXPERT_V2 / QFUSE / MMVF_ROWS` (−7 %), `STRATA_ADAPT_ASYNC_BATCH=16` (−16 %). Pitfall: a prompt sent twice in
a row runs ~25 % faster the second time (the cache has adapted to it); an A/B that always sends B second is invalid.

## Limits

- One PC, Windows only. After a long prompt the branch is level with the tuned release (the prompt itself fills the cache).
- Greedy answers are not byte-identical between two fresh engines — already true for the unpatched build (CPU/GPU split
  and adaptive tier depend on timing). `STRATA_IQ3S_MT1` changes the CPU rounding of single-token rows; the other
  switches change speed or drafts only (drafts are always verified). No perplexity run yet.
- What still costs per window: ~13 ms of GPU chain (≈1,750 small kernels, no PDL on sm_86, WDDM) and 5-14 ms of CPU work
  for missed experts.
