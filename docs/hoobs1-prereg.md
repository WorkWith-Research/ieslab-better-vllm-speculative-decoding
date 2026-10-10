# Preregistration — H-OBS-1: Direct GPU-execution observability under async DSpark serving

**Date:** 2026-10-10 · **Status:** PROSPECTIVE (fixed before any measurement run)
**Parent result:** Track B Step 3 (`docs/trackb-results.md`, prereg R1–R4, commit `68cef64`) — **final and unchanged.**
**Directive:** "Validate Async GPU Observability Before Further Controller Optimization" §2–§6.

---

## 0. Scope and what is NOT being done

This is a **bounded observability validation**, not a controller re-tune.

- The Track B verdict stands exactly as recorded: S1a NOT MET, S1b/S2/S3/S4 MET. It is **not**
  reclassified here and no trials are added to move the old CI.
- H-OBS-1 tests a **measurement/observability hypothesis only**. The secondary controller hypothesis
  (H-OBS-2) is gated: it is tested **only if** H-OBS-1 confirms a usable, low-overhead causal GPU signal.
- No learned controller, no joint scheduling, no change to model weights / workload / the frozen Track B
  controller constants. A negative or inconclusive result is a valid, reportable outcome (§6 hard stops).

## 1. What the prior (R4) claim actually established — and its flaw

R4 (Track B prereg §11b) claimed `schedule()` cadence underestimates true GPU step time by
**1.5× (C=8) → 5.9× (C=64)**. That number was **derived, not measured**: it used
`true_step_time ≈ L·C / throughput` with **configured concurrency C** in the numerator.

Two problems make this a hypothesis requiring direct validation (directive §2):

1. **Configured C ≠ actual running requests per step.** The Track B causal traces *already* record
   `occupancy = running/max_num_seqs` each step. Recomputing from those traces:

   | C | configured | mean actual running (whole run) | steady-state max | cadence (measured) |
   |---:|---:|---:|---:|---:|
   | 8  | 8  | **6.7**  | 8  | ~23 ms |
   | 32 | 32 | **19.1** | 32 | ~31 ms |
   | 64 | 64 | **18.8** | 59 | ~32 ms |

   The whole-run mean is dragged down by the run tail (occupancy decays from ~57 to ~2 at C=64 as the
   60 prompts finish). Using configured C in `L·C/throughput` therefore **overstates** the per-step token
   count and hence the "true step time." The 5.9× gap is not established.

2. **The async pipeline depth is only 2.** For the V2 runner at PP=1,
   `max_concurrent_batches = pp_size + 1 = 2` (`vllm/config/vllm.py:492`). In
   `step_with_batch_queue` (`vllm/v1/engine/core.py:519`), the CPU schedules a step, enqueues it
   non-blocking, and — once the 2-deep batch queue is full — **blocks on the prior step's future**
   (`future.result()` → `AsyncOutput.get_output()` → `copy_event.synchronize()`,
   `vllm/v1/worker/gpu/async_utils.py:49`). So in steady state the CPU cadence is gated by GPU completion,
   not free-running ahead. A 5.9× decoupling is therefore *inconsistent* with the control flow; the flat
   ~31 ms cadence at C=64 actually implies steps of order ~30 ms, not ~180 ms.

**Conclusion:** R4's numerical gap is **retracted as an established fact and re-posed as hypothesis
H-OBS-1 below.** The qualitative claim (cadence may not equal GPU execution time) remains plausible and is
what H-OBS-1 tests directly. This correction is recorded here, prospectively; it does not alter the Track B
verdict (which depended on throughput and causal correctness, not on the R4 gap magnitude).

## 2. Source-level trace: what `schedule()` cadence actually measures

TP=1 → `UniProcExecutor` (`vllm/v1/executor/uniproc_executor.py`). The engine core runs
`run_busy_loop` → `_process_engine_step` → `step_with_batch_queue` (async path, since DSpark forces the V2
runner and async scheduling stays enabled).

Per loop iteration:
1. `scheduler.schedule()` — **this is where the Track B controller's `evaluate()` runs**
   (`vllm/v1/core/sched/scheduler.py:448`); its "measured signal" is the wall time since the previous
   `schedule()` call (`tb_controller.py:129-139`).
2. `execute_model(scheduler_output, non_block=True)` → `collective_rpc(..., non_block=True)` → runs the V2
   `GPUModelRunner.execute_model` **in-process** (no separate worker process at TP=1). It does CPU input prep
   then launches the GPU forward (`gpu/model_runner.py:1297` cudagraph FULL / `:1328` eager / `:1323`
   PIECEWISE) and returns `None` after enqueuing — **the GPU work is still running**.
3. `sample_tokens(grammar_output, non_block=True)` → V2 `GPUModelRunner.sample_tokens`: runs the sampler on
   GPU, starts an **async D2H copy** of sampled tokens on a side stream (`async_utils.py:30-47`), then — if a
   speculator exists and SD is on — runs `speculator.propose()` (the drafter forward, **this is the GPU cost
   the controller decides about**), and returns an `AsyncOutput` handle.
4. The future is appended to the 2-deep `batch_queue`. If the queue is now full, the loop **blocks on the
   oldest future's `get_output()`**, which calls `copy_event.synchronize()` — i.e. it waits until that step's
   GPU work + D2H copy have completed.

So the controller's cadence signal = time between consecutive `schedule()` calls = a mix of
**(a)** CPU scheduling/enqueue time, **(b)** any blocking wait for a prior step's GPU completion (when the
2-deep queue is full), and **(c)** idle gaps when no requests are ready. It is **not** a direct measurement
of the current step's GPU execution time, and it conflates prefill and decode steps. H-OBS-1 measures each
component directly.

## 3. Hypotheses

**H-OBS-1 (primary).** Under asynchronous serving of native fixed-K DSpark, the CPU `schedule()` cadence is a
biased / poorly predictive estimator of actual per-step GPU execution cost, with the bias or prediction error
growing as batch pressure (actual running requests) increases.

- **H-OBS-1a (direction):** the sign and magnitude of `cadence − gpu_step_time` is not constant across
  C=8/32/64; specifically it does NOT remain a small fixed offset.
- **H-OBS-1b (occupancy confound):** after accounting for *actual* per-step running requests (not configured
  C), the R4 "gap" shrinks substantially — i.e. most of the claimed 5.9× was the configured-vs-actual
  occupancy error, not a cadence/GPU decoupling.

**H-OBS-2 (secondary, GATED on H-OBS-1).** A causally-available, GPU-execution-aware signal (direct CUDA-event
step timing) improves SD on/off decisions over the CPU-cadence signal, at acceptable measurement overhead.
*Not tested unless H-OBS-1 confirms a usable low-overhead causal signal.*

## 4. Measurement design (instrumentation-only first)

**Environment (unchanged from Track B):** Qwen3-4B + dspark_qwen3_4b_block7 (K=7), 1× RTX 3090 (SM86),
GPU1, vLLM V2 runner, async scheduling on, ShareGPT-V3 60 prompts, greedy seed 0, `--ignore-eos`,
`max_num_seqs=64`, `max_num_batched_tokens=4096`. Conditions C ∈ {8, 32, 64}.

**Instrumentation (new, env-gated, additive — does not modify the frozen Track B controller):**
a CUDA-event pair per model step on the main stream, recorded **asynchronously** and queried **lazily**
(non-blocking `event.query()`; no `torch.cuda.synchronize()` / `cudaDeviceSynchronize` in the hot path):

- `ev_step_start` recorded at the start of `execute_model` (after CPU input prep, before the forward launch).
- `ev_step_end` recorded at the end of `sample_tokens` (after `speculator.propose`, so it brackets the full
  GPU step = target forward + sampling + drafter). `elapsed_time(start, end)` = **direct per-step GPU time**.
- A second inner pair brackets only the target forward (execute_model) to separate *verification/target*
  cost from *drafter* cost — the quantity the on/off decision actually trades off.

**Per step, logged by unique step ID (JSONL), correlated:**
- `step_id` (monotonic), `t_cpu_sched` (monotonic at `schedule()` entry) → **CPU cadence**.
- `gpu_step_ms` (CUDA-event elapsed, full step), `gpu_fwd_ms` (target forward only), `gpu_draft_ms`
  (≈ step − fwd, the drafter cost).
- `running` (actual scheduled requests this step), `waiting`, `scheduled_tokens`,
  `decode_tokens` / `prefill_tokens` split (per-req `num_scheduled_tokens`: 1 ⇒ decode, >1 ⇒ prefill/chunked).
- `sd_enabled` (whether the drafter ran this step — from the frozen controller's decision), `in_flight`
  (number of prior steps whose GPU completion event is not yet done when this step enqueues = outstanding
  model executions ≈ batch-queue depth in use).
- `cadence_ms` (wall time since previous `schedule()`).

**Arms for the instrumentation-only comparison (all with the tracer on, otherwise identical):**
- **AR** (no speculator) — cadence vs GPU fwd baseline.
- **DSpark-K7 native** (SD always on) — the regime of interest.
- **CTRL-LIVE** (frozen Track B controller) — to see how its ON/OFF steps map onto direct GPU time.

**Validation of the GPU measurement itself (directive §4):**
- Cross-check `gpu_step_ms` against an independent profiler on a short controlled run
  (`torch.profiler` with CUDA activities, or `nsys` if available) for a fixed small batch; report agreement.
- Confirm events bracket the intended work: `gpu_fwd_ms` should scale with scheduled tokens and be ≈0-drafter
  on AR steps; `gpu_draft_ms` should vanish when `sd_enabled=0`.

**Overhead measurement (directive §4):** compare output throughput of each arm **with tracer ON vs OFF**
(native control). If the tracer perturbs throughput by more than ~1–2% or destroys async overlap
(e.g. forces serialization), it is **not** a viable lightweight online signal and H-OBS-2 is not pursued.

## 5. Analysis plan (fixed now)

For each C, over the **steady-state window** (exclude first 10% ramp-up and the tail where running < 5):
1. Report distributions (mean, p50, p90) of `cadence_ms` vs `gpu_step_ms` vs `gpu_fwd_ms`, and their ratio.
2. Regress `cadence_ms` on `gpu_step_ms` and on `running`; report R² and whether the residual grows with
   `running` (tests H-OBS-1a).
3. Recompute the R4 "gap" using **actual** per-step `running` instead of configured C; compare to the direct
   `gpu_step_ms` (tests H-OBS-1b — does the 5.9× collapse?).
4. Report instrumentation overhead (Δ throughput tracer-on vs -off) per arm.

## 6. Verdict rule (fixed now)

Produce exactly one of, with the evidence and uncertainty:
- **OBSERVABILITY GAP CONFIRMED** — cadence is a biased/poor estimator of GPU step time, error grows with
  actual batch pressure, AND the R4 gap does NOT collapse under occupancy-corrected accounting (i.e. it is a
  real decoupling, not an occupancy artifact), AND the direct signal is low-overhead and causally available.
- **PARTIALLY CONFIRMED** — cadence ≠ GPU time in some regimes but the discrepancy is explained by the
  configured-vs-actual occupancy confound (R4 gap collapses) and/or does not grow with pressure; a corrected
  CPU-side signal (using actual `running`) may suffice without GPU instrumentation.
- **NOT CONFIRMED** — cadence tracks GPU step time well once actual occupancy is accounted for; the R4 gap was
  an artifact; no exploitable observability limitation found on this path.
- **INCONCLUSIVE** — measurement could not be validated, overhead too high, or signal not causally available.

**H-OBS-2 is pursued only under CONFIRMED (or PARTIALLY with a justified low-overhead causal GPU signal).**

## 7. Hard stop conditions (§6 of directive)

Stop and report a blocker if: direct measurements do not support an observability problem; GPU timing cannot
be correlated causally with scheduling steps; the signal requires disruptive synchronization; measurement
overhead overwhelms any expected gain; or (later) a corrected signal does not beat noise. A negative result is
a valid result. No learned controller, no joint scheduling, before a measurable opportunity is established.

## 7A. Dated amendment A1 (2026-10-10) — manifest reconciliation, schema correction, pilot disclosure

This amendment is prospective and applies to the main run below; it does NOT alter §5 analysis windows,
§6 verdict rule, or the Track B result. It resolves three items flagged by the recovery directive (§3A/§3B/§4):

**(1) Manifest reconciliation (cell count).** The earlier draft launcher described 12 trials (3 arms + a 4th
`dspark_off` arm × C∈{8,32,64}). That conflated two axes: policy and instrumentation. Per §4 ("compare output
throughput of each arm with tracer ON vs OFF") the correct grid is **arm ∈ {ar, dspark, ctrl_live} × C ∈ {8,32,64}
× tracer ∈ {on, off} = 18 cells**, one repetition each (this is a characterization/validation experiment, not a
hypothesis test with an n-driven CI — see (5) below). The historical `dspark_off` label meant "tracer OFF" (SD
still ON); it maps to cell `dspark_c*_off` and is preserved as such. Arm resolution is table-driven in
`scripts/hoobs1_arms.py` (unit self-tested: AR has no drafter and no controller mode; dspark = native always-on;
ctrl_live = frozen live controller; tracer is an independent axis; unknown arms rejected). A separate, non-counted
diagnostic run under `nsys` (dspark, C=8) provides the independent GPU-execution trace for Gate-1 validation.

**(2) Measurement schema correction (§3B).** An earlier draft field named `prefill_tokens` was actually
`scheduled_tokens − draft_tokens` = **non-draft work**, which in a pure decode step equals `running` (one bonus
position per request) and is NOT prefill. Verified empirically on the C=8 pilot: pure-decode steps show
`scheduled − draft ≈ running` with zero prompt work, while one step showed `417` non-draft tokens (a genuine
prefill chunk). The schema now records a **genuine per-request prefill/decode split** using phase bookkeeping
(`min(scheduled_r, num_prompt_tokens − num_computed_tokens)`, read before the token counter advances), plus a new
`sched_ms` field (CPU work inside `schedule()`, distinct from the cadence gap and from GPU time). The full data
dictionary (definition/units/source/timing boundary/controller-availability per field) is in
`vllm/v1/core/sched/obs_tracer.py`. Raw historical fields are preserved; old pilot traces used the old meaning.
The frozen Track B controller's input semantics are UNCHANGED by this relabeling (it still receives the same
non-draft token count it always did).

**(3) Collector policy (§3C).** The event collector no longer sleeps in the serving hot path: records whose GPU
events are not yet complete are DEFERRED to a later flush pass; `final_flush` (atexit, one-time sync outside the
performance window) is the guaranteed backstop. Counts of created / written-complete / null-gpu / failed-elapsed /
dropped records are persisted to `collector_stats.json` so samples are never silently selected.

**(4) Pilot disclosure.** A C=8 instrumentation smoke (dspark arm, 60-prompt workload, tracer on) was inspected
before this amendment. It measured CPU cadence ≈19.9 ms and direct GPU step ≈19.7 ms (ratio ≈1.02), with
in_flight≈2 for ~94% of steps. This CONTRADICTS the R4 throughput-derived 1.5× gap at C=8 but does NOT settle
C=32/C=64 and does NOT replace validation of the final instrumentation. It is disclosed here as historical
evidence and is NOT treated as unseen confirmatory data; it is excluded from the §5 analysis (which uses only the
main-run cells below).

**(5) Statistical handling.** Each cell yields hundreds of per-step records that are TEMPORALLY CORRELATED
(consecutive steps share queue state); they are not independent experimental repetitions. The main run therefore
reports per-cell distributions and cross-cell pressure dependence (C=8→32→64) rather than an n-driven CI over
steps, consistent with §5–§6 (which compare regimes, not step-level CIs). No valid cell is added to move a bound.

**Frozen main-run manifest (18 cells), run order (round-robin over arms within each C to balance thermal drift):**
for C in {8,32,64}: for arm in {ar,dspark,ctrl_live}: for tracer in {on,off}. Fresh server per cell; GPU1; port 8400;
ShareGPT-V3 60 prompts, greedy seed 0, `--ignore-eos`, `num_warmups=0`; metrics scraper @0.5 s; per-cell artifacts
(server log, bench result, metrics JSONL, obs trace, collector stats, effective-config manifest) under
`results/hoobs1/main/<cell>/`. A cell is VALID iff: server startup-complete marker present in its OWN log, bench
exit 0 with a result file, (tracer-on) steps.jsonl non-empty and `collector_stats.records_dropped==0`, and the arm's
expected SD behavior observed (AR: zero draft tokens; dspark/ctrl_live: drafts scheduled). Invalid cells are
preserved, marked, and cannot support a §6 verdict.

## 8. Deliverables

A dated synthesis (`docs/hoobs1-observability.md`) containing: this prereg, the source-level trace (§2),
direct GPU-timing evidence per C, comparison against the (corrected) throughput-derived estimate,
instrumentation overhead, and the CONFIRMED / PARTIALLY / NOT / INCONCLUSIVE verdict with uncertainty. The
Track B result and R1–R4 are preserved verbatim; only the R4 gap magnitude is re-posed as a hypothesis here.
