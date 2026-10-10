# H-OBS-1 — Direct GPU-execution observability under async DSpark serving

**Date:** 2026-10-11 · **Status:** COMPLETE (all 18 main-run cells + nsys diagnostic)
**Preregistration:** `docs/hoobs1-prereg.md` (prospectively fixed before any measurement run; §5 windows, §6 verdict rule, amendment A1 unchanged by this document)
**Parent result:** Track B Step 3 (`docs/trackb-results.md`, prereg R1–R4) — **final and unchanged.** S1a NOT MET, S1b/S2/S3/S4 MET. This document does **not** reclassify S1a and adds no trials to the old CI. It tests a measurement/observability hypothesis only.

---

## Verdict (prereg §6 rule)

> ## **PARTIALLY CONFIRMED**
>
> The feared observability gap — that CPU `schedule()` cadence *underestimates* actual GPU step time by up to 5.9× and increasingly so under load — is **not supported**. Direct CUDA-event measurement shows cadence tracks per-step GPU execution time at a median ratio of **0.98–1.00× across every arm and concurrency**, and on pure-decode steps cadence falls below 0.8× GPU time in **≤1.5% of cases** (AR/DSpark ≤0.5%; CTRL-LIVE up to 1.5% at C=64) — there is no systematic underestimation.
>
> What the data *does* show:
> 1. **The R4 "gap" was an occupancy artifact (H-OBS-1b CONFIRMED).** Recomputing the throughput-derived step time with *actual* per-step running requests instead of configured C collapses the gap from **4.59× (C=64)** to **~1.2×**, and direct GPU measurement shows **~0.99×**. The whole-run accepted-tokens-per-request `L` is constant across C (2.71–2.75), so the R4 formula's error came entirely from using configured C in the numerator.
> 2. **A real, regime-limited discrepancy exists, but in the opposite direction.** At high occupancy the cadence has a *right tail* where it **exceeds** GPU time (CPU-side stalls / queue blocking), most pronounced at C=32 (top-tercile residual σ 0.5→21 ms; ~10% of high-occ steps >1.2×, max 3.3×). This is a CPU-overhead effect, not a GPU/CPU decoupling, and it does **not** grow monotonically with pressure in the underestimation direction (H-OBS-1a primary: **refuted as stated**).
>
> A corrected CPU-side signal already suffices; the direct GPU-execution signal is low-overhead and causally available but is **not required** to fix an underestimate that does not exist. **H-OBS-2 is therefore NOT pursued** (no exploitable observability limitation was found that would motivate a GPU-instrumented controller).

This is a *negative-leaning* result for the "observability gap" hypothesis and, if anything, **favors** the lightweight-decision-model thesis: the CPU cadence signal the Track B controller already uses is a good, causally-available proxy for per-step GPU cost. No general claim about all lightweight decision models is made or needed (directive §2).

---

## 1. What was measured

**Environment (unchanged from Track B):** Qwen3-4B + `dspark_qwen3_4b_block7` (K=7), 1× RTX 3090 (SM86) on GPU1, vLLM V2 runner, async scheduling on, ShareGPT-V3 60 prompts, greedy seed 0, `--ignore-eos`, `max_num_seqs=64`, `max_num_batched_tokens=4096`. Conditions C ∈ {8, 32, 64}.

**Instrumentation (env-gated, additive; does not modify the frozen Track B controller):** a CUDA-event pair per model step on the main stream, recorded **asynchronously** and queried **lazily** (non-blocking `event.query()`; no `torch.cuda.synchronize()` / `cudaDeviceSynchronize` in the hot path):
- `ev_step_start` at the start of `execute_model` (after CPU input prep, before the forward launch);
- `ev_step_end` at the end of `sample_tokens` (after `speculator.propose`, so it brackets the full GPU step = target forward + sampling + drafter);
- an inner pair brackets only the target forward (`gpu_fwd_ms`) to separate verification/target cost from drafter cost.

Per step, logged by unique `step_id` (JSONL): `t_cpu_sched` (→ CPU cadence), `gpu_step_ms`, `gpu_fwd_ms`, `running`, `waiting`, `scheduled_tokens`, decode/prefill split, `sd_enabled`, `in_flight`, `cadence_ms`. Full data dictionary in `vllm/v1/core/sched/obs_tracer.py`.

**Arms (all tracer-on for the comparison; otherwise identical):**
- **AR** (no speculator) — cadence vs GPU-forward baseline.
- **DSpark-K7 native** (SD always on) — the regime of interest.
- **CTRL-LIVE** (frozen Track B controller) — how its ON/OFF steps map onto direct GPU time.

**Grid:** arm × C × tracer ∈ {on, off} = **18 cells**, one repetition each (a characterization/validation experiment, not an n-driven CI test — prereg A1(5)). Fresh server per cell; port 8400; metrics scraper @0.5 s. Plus a separate non-counted **nsys diagnostic** (dspark, C=8) for independent GPU-execution validation.

**Data integrity:** all 18 cells marked `DONE`, zero invalids, all on the frozen worktree rev `8f266ba`. Collector totals across the 10 tracer-on cells + nsys: **6721 records created, 6701 written-complete, 0 null-gpu, 0 failed-elapsed, 0 dropped** — no silent sample selection. The 20 un-written records are the deferred tail flushed at shutdown (backstop), not drops.

---

## 2. Validation of the GPU measurement itself (directive §4)

### 2a. Independent nsys cross-check (exact step-ID correlation)
The tracer's `gpu_step_ms` was validated against an independent `nsys` CUDA trace under the **actual async serving config**, correlating by unique step ID (no time-gap heuristics): each executed step opens an NVTX range `hoobs1.step.<id>` on the engine thread spanning that step's CPU enqueue; kernel launches inside the window are attributed to the step via `correlationId`, and their device-side spans give the step's GPU execution span.

| Gate | Result |
|---|---|
| S1: every tracer record id has an NVTX anchor (anchors ≥ records) | **PASS** — 459 anchors ≥ 400 records; 0 records without an anchor |
| S3: per-step nsys kernel span vs `gpu_step_ms` (matched by step id) | **PASS** — 358 steps matched; median span/gpu_step = **0.733** (within [0.4, 1.1]); **0%** of steps below 0.25× (tracer is not measuring enqueue-only time) |

The ~0.73 decode-step ratio is the expected **lower bound**: under async serving, tail kernels (drafter / last forward) can still be executing when the CPU-side NVTX range closes, so nsys-per-launch-window spans are ≤ the tracer's event-bracketed window. Large prefill steps agree near-exactly (nsys 167.9 ms vs tracer 177.5 ms). The serving-path analyzer is committed at worktree `af1db18` (fixes an O(n²) bug that had made the diagnostic appear to hang for >1 h; it now runs in ~0.5 s).

### 2b. Event-bracketing sanity (§4 "confirm events bracket the intended work")
- **AR arm** (no speculator): `gpu_step_ms − gpu_fwd_ms` ≈ **0.95–0.97 ms** at all C — sampler + async D2H copy only, no drafter. ✓
- **CTRL-LIVE**: drafter cost (`step − fwd`) = **7.48–7.55 ms** when `sd_enabled=1` vs **1.03 ms** when `sd_enabled=0` (n_on≈173–215, n_off≈97–103) — the drafter appears and vanishes exactly with the controller's decision. ✓
- **DSpark**: `gpu_fwd_ms` scales with scheduled tokens — **22.9 ms** on pure-decode steps vs **120.2 ms** on prefill chunks (≥512 tokens). ✓

The events bracket the intended work: full step = target forward + sampling + drafter, and each component is separable.

---

## 3. Direct results — cadence vs direct GPU step time (steady-state window)

Steady-state window per prereg §5: exclude the first 10% of steps (ramp-up) and the tail where `running < 5`. Steps are temporally correlated, so these are **per-cell distributions + cross-cell pressure dependence**, not n-driven CIs. `gpu/cad` = mean(gpu_step)/mean(cadence).

| arm·C | n_ss | cadence p50 (ms) | gpu_step p50 (ms) | gpu_fwd p50 (ms) | runninḡ | **gpu/cad** | R²(cad~gpu) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ar_c8 | 1317 | 12.70 | 12.66 | 11.71 | 7.67 | **0.996** | 0.0001 |
| ar_c32 | 593 | 14.39 | 14.35 | 13.38 | 16.13 | **0.996** | 0.0007 |
| ar_c64 | 511 | 14.51 | 14.47 | 13.50 | 17.96 | **0.996** | 0.9814 |
| dspark_c8 | 490 | 20.90 | 20.80 | 15.34 | 7.63 | **0.998** | 0.0016 |
| dspark_c32 | 214 | 32.87 | 30.92 | 24.15 | 16.76 | **0.991** | 0.3957 |
| dspark_c64 | 202 | 31.26 | 30.96 | 23.50 | 16.87 | **0.983** | 0.9841 |
| ctrl_live_c8 | 484 | 20.95 | 20.86 | 15.40 | 7.70 | **0.998** | 0.0029 |
| ctrl_live_c32 | 318 | 21.98 | 21.76 | 16.65 | 20.05 | **1.000** | 0.1716 |
| ctrl_live_c64 | 270 | 22.18 | 22.05 | 16.74 | 22.55 | **1.000** | 0.8308 |

**Reading.** The median cadence-to-GPU ratio is **0.98–1.00 in every cell** — cadence is a near-unity estimator of per-step GPU time, at *every* concurrency and in *all three arms*. There is no 1.5×–5.9× underestimation anywhere. (R²(cad~gpu) is low at C=8 because both quantities are nearly constant there — little variance to correlate — and rises toward ~0.98 as step time becomes occupancy-driven; it is not a signal of poor tracking, which the ratio column already establishes.)

---

## 4. H-OBS-1a — does prediction error grow with pressure? (primary)

Direct test on **pure-decode steps** (prefill=0), global OLS fit `cadence ~ gpu_step` per cell, residual std by running tercile. If the primary hypothesis held, both the bias and the residual variance would *grow* with running — in the underestimation direction.

| cell | n (pure-decode) | R² | run≈6–7 σ / ratio | run≈13 σ / ratio | run≈28–31 σ / ratio | hi-occ(≥16): frac>1.2× / max |
|---|---:|---:|---|---|---|---|
| dspark_c8 | 445 | 0.004 | 3.06 ms / 0.999 | 16.1 ms / 1.003 | 16.8 ms / 1.003 | n=0 (C=8 never reaches high occ) |
| dspark_c32 | 200 | 0.608 | 0.51 ms / 1.002 | 2.78 ms / 1.002 | **20.9 ms / 1.005** | n=89: **10.1% / 3.28×** |
| dspark_c64 | 201 | 0.984 | 1.06 ms / 1.002 | 1.79 ms / 1.002 | 3.36 ms / 1.003 | n=87: **4.6% / 1.36×** |

**Findings.**
- The **median ratio stays ~1.00 at every tercile and every C** — no systematic bias in either direction, and certainly no underestimation (on pure-decode steps, cadence < 0.8× GPU time occurs in ≤1.5% of cases; on mixed-workload windows it can reach ~7% at low C, but that is a prefill-spike composition effect — see §3 note — not a decoupling).
- The only pressure-dependent effect is a **right-tail variance increase at high occupancy**: at C=32 the top running tercile's residual σ jumps 0.5→21 ms and ~10% of high-occ steps run >1.2× (up to 3.3×) *longer* than the GPU step. At C=64 the same effect is smaller (σ 1.1→3.4 ms; 4.6% >1.2×, max 1.36×). These are **CPU-side stalls** (scheduling/enqueue overhead + blocking on the prior step's future when the 2-deep queue is full), i.e. cadence *over*-estimates GPU time in the tail — the opposite of R4's claim.
- The tail does not grow monotonically with C (C=32 > C=64 here), so it is a regime/queueing artifact, not a pressure-scaling decoupling.

**H-OBS-1a as stated ("cadence underestimates GPU cost, error grows with batch pressure"): REFUTED.** The real effect is a bounded high-occupancy *over*-estimate from CPU stalls.

---

## 5. H-OBS-1b — occupancy-corrected R4 gap (the core retraction)

R4 (Track B prereg §11b) derived `true_step ≈ L·C_configured / throughput_out` and claimed a 1.5×–5.9× gap vs cadence. Recomputed here with the **same formula** but window-consistent whole-run `L`, then corrected to use **actual steady-state running** instead of configured C, alongside the direct GPU measurement:

| C | thr (tok/s) | cadencē (ms) | gpu_step̄ (ms) | runninḡ | L (whole-run) | R4 step (configured C) | **R4 gap ×** | corrected step (actual running) | **corrected gap ×** | **direct gpu/cad ×** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 763.2 | 23.9 | 23.8 | 7.63 | 2.749 | 28.8 | **1.21** | 27.5 | **1.15** | **1.00** |
| 32 | 986.6 | 39.2 | 38.8 | 16.76 | 2.744 | 89.0 | **2.27** | 46.6 | **1.19** | **0.99** |
| 64 | 970.7 | 38.9 | 38.5 | 16.87 | 2.708 | 178.5 | **4.59** | 47.1 | **1.21** | **0.99** |

**The gap collapses.** At C=64 the throughput-derived estimate is **4.59×** cadence with configured C, but **~1.2×** once actual running (≈17, not 64) is used — and the direct GPU measurement says **0.99×**. The decisive diagnostic: `L` (accepted output tokens per running-request per step) is **constant across C at 2.71–2.75**, so the R4 formula's error came *entirely* from putting configured C in the numerator while the actual occupancy was ~3.8× lower at C=64. This reproduces, with window-consistent bookkeeping, the same qualitative conclusion prereg §1 reached prospectively: **the 5.9× was an occupancy artifact, not a cadence/GPU decoupling.** (The absolute multipliers differ slightly from Track B's 1.5×/5.9× because of how `L` and the analysis window are computed; the collapse under correction is identical.)

**H-OBS-1b: CONFIRMED.**

---

## 6. Instrumentation overhead & async-overlap preservation (directive §4)

Δ output throughput, tracer **on** vs **off**, per arm:

| arm·C | on (tok/s) | off (tok/s) | Δ% |
|---|---:|---:|---:|
| ar_c8 / c32 / c64 | 462.9 / 876.4 / 985.5 | 463.3 / 874.8 / 981.5 | −0.10% / +0.17% / +0.41% |
| dspark_c8 / c32 / c64 | 763.2 / 986.6 / 970.7 | 762.7 / 990.6 / 1004.6 | +0.08% / −0.40% / **−3.37%** |
| ctrl_live_c8 / c32 / c64 | 758.8 / 974.7 / 1012.7 | 759.7 / 973.9 / 1037.8 | −0.12% / +0.08% / **−2.42%** |

- At **C=8 and C=32** the tracer is **≤0.5%** overhead in every arm — comfortably within the ~1–2% guideline.
- At **C=64**, two cells exceed it (dspark −3.37%, ctrl_live −2.42%). This is a real, reported cost, not hidden: at the largest batch the deferred event-completion bookkeeping lands just over budget. It does **not** invalidate the C=8/32 signal and does not block this verdict (H-OBS-2 is not pursued), but it means a GPU-instrumented controller would need to weigh ~2–3% overhead at peak load.

**Async overlap preserved.** `in_flight` (outstanding model executions ≈ batch-queue depth in use) stays **≥2 for 93–100%** of steady-state steps across all arms/C (e.g. ar_c64 100%, dspark_c32/c64 100%, ctrl_live_c32 93.4%). The lazy `event.query()` design did **not** force serialization — a naive sync-based tracer would have collapsed this toward ~1.

---

## 7. Answers to the directive's §2 required questions

- **What exactly does `schedule()` cadence measure?** Wall time between consecutive `schedule()` entries = (a) CPU scheduling/enqueue + (b) any blocking wait on a prior step's GPU completion when the 2-deep batch queue is full (`future.result()` → `copy_event.synchronize()`) + (c) idle gaps. Not a direct measurement of the current step's GPU time (prereg §2 source trace).
- **When does the CPU enqueue a model step?** Inside `execute_model(non_block=True)` after `schedule()`; it launches the GPU forward and returns while the GPU work is still running.
- **When does the corresponding GPU work actually start/finish?** Measured directly: `ev_step_start` at `execute_model` entry, `ev_step_end` at `sample_tokens` exit (brackets target forward + sampling + drafter). Cross-validated by nsys (§2a).
- **How many model steps are outstanding?** `in_flight` = 2 for ~93–100% of steady-state steps (`max_concurrent_batches = pp_size+1 = 2`).
- **Does the timing estimate use actual per-step scheduled requests/tokens?** R4 used *configured* C (the flaw); H-OBS-1b uses *actual* `running`; the direct signal uses neither — it is measured GPU time.
- **How do prefill and decode contribute?** Separated: `gpu_fwd_ms` scales with scheduled tokens (22.9 ms decode vs 120 ms prefill); `step − fwd` is the drafter cost; raw cadence conflates them, the direct signal does not.
- **Does the measurement distinguish GPU compute from queueing delay?** Yes — `gpu_step_ms` is pure GPU (CUDA-event bracketed); cadence includes queueing/blocking. Their difference (cadence − gpu_step) is ≈0 at median and slightly *positive* in the high-occ tail (CPU stalls).
- **Is the signal available early enough for a causal decision?** Yes — cadence is known at `schedule()` entry (before the decision); the prior step's direct GPU time is available lazily by the next `schedule()` (its event is already complete). Both are causally available.

---

## 8. Uncertainty and limitations

- **One repetition per cell.** This is a characterization/validation experiment (prereg A1(5)); conclusions rest on per-cell distributions and cross-cell pressure dependence, not step-level CIs. The cadence≈GPU result is robust because it holds at the median in *all nine* tracer-on cells simultaneously; the high-occ right-tail magnitudes (C=32 vs C=64) carry more cell-to-cell uncertainty than the near-unity medians.
- **Workload shape.** The 60-prompt ShareGPT batch produces a decaying occupancy profile (running falls from ~57 to single digits as prompts finish), so "C" creates a *peak*-occupancy difference, not sustained pressure. Steady-state windows were trimmed accordingly; the high-occ tail statistics use only `running ≥ 16` steps.
- **nsys span is a lower bound** on the event-bracketed window for decode steps (async tail kernels outlive the CPU-side NVTX range); the agreement gate accounts for this ([0.4, 1.1] band). Prefill steps agree near-exactly.
- **C=64 overhead** (~2–3%) is a real cost of the deferred-completion bookkeeping at peak batch; it is reported, not smoothed over.

---

## 9. H-OBS-2 disposition

Prereg §6: H-OBS-2 (a GPU-execution-aware signal improves SD on/off decisions) is pursued **only under CONFIRMED**, or PARTIALLY *with a justified low-overhead causal GPU signal*. The primary result shows the CPU cadence already tracks direct GPU step time at ~1.0× median with no exploitable underestimate, and the only discrepancy (high-occ CPU-stall tail) is a CPU-side effect that a corrected CPU-side signal handles. **There is no observability limitation that would motivate adding a GPU-instrumented controller**, so **H-OBS-2 is not pursued.** This is consistent with the PARTIALLY CONFIRMED verdict and with the hard-stop condition "direct measurements do not support an observability problem."

---

## 10. Implications for the Track B controller (no re-tune performed)

This experiment does **not** change the frozen Track B controller, its constants, or its S1a NOT MET verdict — those stand exactly as recorded. It *does* correct the interpretive context: the R4 "5.9× observability gap" that was cited as a limitation of the CPU-cadence signal is an occupancy artifact, and the cadence signal itself is a near-unity proxy for per-step GPU cost. The one actionable (out-of-scope-for-this-task) observation is that at high occupancy the cadence's right tail can *over*-state GPU cost by up to ~3× in isolated steps — relevant only if a future controller wants to de-bias its pressure estimate, and addressable with a CPU-side correction (e.g. using actual `running` or a short EMA), not with GPU instrumentation.

---

## 11. Provenance & reproducibility

- **Prereg:** `docs/hoobs1-prereg.md` (commit `e022cb6`), fixed before any measurement run; §5 windows, §6 verdict rule, amendment A1 unchanged by this document.
- **Instrumentation + serving-path code:** worktree `/home/junior1/_dev/dspark-native-validation`, branch `track-b-live-sd`. Main run executed on frozen rev **`8f266ba`**; the only change after the run started is the nsys analyzer (`af1db18`), which does not touch the serving path. Arm-resolution unit self-test passes on final code.
- **Raw artifacts (local, gitignored):** `results/hoobs1/main/<cell>/` for all 18 cells + `nsys_c8/` — server log, bench result, metrics JSONL, obs trace (`steps.jsonl`), collector stats, effective-config manifest; each cell's `config_manifest.json` records worktree rev `8f266ba`.
- **Analysis:** `experiments/hoobs1_analyze.py` (deterministic; consumes the 18 cells) → `results/hoobs1/main/analysis.json` + `analysis_table.md`. nsys serving-path validation: `scripts/hoobs1_nsys_analyze.py` → `nsys_c8/serving_validation.json`.
- **Track B result preserved verbatim:** `docs/trackb-results.md`, prereg R1–R4 (commit `68cef64`). Only the R4 gap *magnitude* is re-posed as a hypothesis here; no Track B verdict is altered.
