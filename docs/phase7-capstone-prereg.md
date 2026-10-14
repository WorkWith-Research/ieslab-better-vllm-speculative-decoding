# Phase 7 (capstone) — Preregistration: request-aware SD scheduler from live serving state (load × traffic-class)

- **Date:** 2026-10-14 (KST)
- **Status:** SPEC / PREREGISTRATION. Design + success/falsification criteria fixed. Constants marked **[CAL]** are
  filled from a calibration/smoke step BEFORE any controller trial runs (append-only block). No controller trial may
  start until that block exists and the two open signal-design questions (§5) are resolved by smoke test.
- **This is the capstone** of the project: it turns the *validated* scheduler signal into a *working in-loop
  controller* and shows realized end-to-end serving gain — the original mission ("a lightweight decision model that
  sets SD from live serving state") made concrete.

## 0. What this builds on (the evidence chain)

| result | finding it licenses |
|---|---|
| Phase 4.6 (FALSIFIED as specified) | Per-request greedy K allocation fails at saturation (negative externality); the spec path has a fixed overhead only a scheduler-level enable/disable removes → control must be **workload-regime / batch level**, with a real SD-off state. |
| H-CLASS-1 / 1C (#28) | Speculation payoff is a function of **request class** (draftability), not cache alone; the A/B gap is content/population-driven. |
| Track A (#29, NOT SUPPORTED) | Per-request acceptance is **not** content-predictable from admission features → do NOT build a per-request predictor. The signal is **population/regime level**. |
| scheduler-signal-validation (`0b11bfc`) | The 2-axis rule "SD-on iff low-load OR high-reuse-class" matches the oracle on all 9 load×class cells (100% gain, 0 regret); each axis alone is strictly worse (load-only misses Bc48/64; class-only misses Ac16/Cc16). |
| H-CLASS-1D (`83b2eb3`) | The load axis holds **out of sample**: boundary in [c16, c32]; A load curve monotone-decreasing. |
| Track B (mechanism built) | Per-step SD on/off is **feasible** in the isolated v0.25.0 worktree: `sd_enabled` V2 patch, causal-verification logging (OFF step ⇒ 0 drafts), probe-based re-entry — 20 tests passing. Its *reactive TPOT-trend* trigger is what we replace. |

## 1. Hypothesis (H-CAP)

A controller that turns native DSpark speculation **on/off per scheduling step** as a function of **two measured
serving-state signals** — (i) a load-regime signal and (ii) a traffic-class (draftability) signal, combined as
`ON iff load_LOW OR class_HIGH-REUSE` — recovers most of fixed-K DSpark's loss at/above the crossover **and**
preserves its low-load benefit **and** beats each single-axis variant on a workload that exercises both axes, versus
AR and native fixed-K=7 DSpark on the identical runtime.

This is the end-to-end realization of the signal validated in §0: not just "the rule matches the payoff surface"
(same-data), but "a live controller acting on measured state achieves the predicted decisions and their gains."

## 2. Runtime and arms

- **Runtime:** Track B's isolated worktree (vLLM v0.25.0+cu126, branch `track-b-live-sd`), Qwen3-4B +
  `dspark_qwen3_4b_block7`, GPU exclusive, same server flags as Track B/Phase 6. Reuse the existing `sd_enabled`
  patch + causal-verification logging; **replace only the trigger** (Track B's TPOT-trend rule → the §3 2-axis rule).
- **Arms:**
  1. **AR** — no spec config (baseline).
  2. **DSpark-K7** — native fixed K=7 (the deployment default this controller improves on).
  3. **CTRL-ON** — patched, policy constant ON → quantifies patch/decision overhead (validity control).
  4. **CTRL-LOADONLY** — patched, `ON iff load_LOW` (class axis removed) → isolates the load axis end-to-end.
  5. **CTRL-CLASSONLY** — patched, `ON iff class_HIGH-REUSE` (load axis removed) → isolates the class axis end-to-end.
  6. **CTRL-LIVE** — patched, the full 2-axis rule (the hypothesis arm).

  Arms 4–5 are the capstone-specific addition: they let us test "both axes necessary" *in the live system*, not just
  on the static payoff surface. Expect CTRL-LOADONLY to lose at high-load high-reuse traffic and CTRL-CLASSONLY to
  lose at low-load diverse traffic — mirroring the scheduler-signal-validation regret table.

## 3. The controller (2-axis rule, measured inputs only)

**Decision per scheduling step:** `sd_enabled ← load_LOW OR class_HIGHREUSE`.

- **Load-regime signal (`load_LOW`):** a function of **measured serving state**, NOT a concurrency integer
  (directive §4 prohibition stands: no "if concurrency == N"). Candidate primary: measured **KV-cache usage fraction**
  and/or **running occupancy = len(running)/max_num_seqs** (both continuous, both in-process, both monotone in true
  resident load, and — critically — NOT the `schedule()`-cadence signal that Track B Defect 2 showed decouples from
  GPU step time under async scheduling). `load_LOW` iff below calibrated thresholds **[CAL]**. Thresholds are on
  *measured continuous state*, which the directive permits; only hardcoded concurrency cutoffs are prohibited.
- **Traffic-class signal (`class_HIGHREUSE`):** rolling-window (W) **draft-acceptance ratio** = accepted/draft tokens
  over recent steps — the direct population-level draftability measure H-CLASS-1 established (B ~0.61 vs A/C ~0.33).
  It is **load-invariant** (Phase 6 / Track B R1), so it cleanly carries the *class* axis without conflating with load.
  `class_HIGHREUSE` iff rolling ratio > θ_acc **[CAL]**. Because acceptance is only observable while SD is ON, the
  estimate is held with slow exponential decay across brief OFF periods and refreshed on probe re-entry (reuses Track
  B's probe mechanism). A prefix-cache-hit-rate term may be added as a secondary class proxy (available even when OFF)
  **only if** the smoke test shows acceptance-decay alone flaps; decided in §5, not assumed.
- **Hysteresis / anti-flap:** reuse Track B's cooldown + probe-based re-entry structure (T_off, probe steps, band).
  Cold start = ON (preserves low-load behavior by default).
- **What it is NOT:** no learned model, no per-request K, no variable-K (blocked — `docs/trackb-feasibility.md` B1–B3),
  no hardcoded concurrency threshold anywhere.

**Why these two signals specifically:** they are *orthogonal by construction* — acceptance ratio is load-invariant
(so it carries class, not load) and KV/occupancy is acceptance-independent (so it carries load, not class). This is
exactly the orthogonality that made the 2-axis rule beat single-axis on the payoff surface; implementing it with
orthogonal measured signals is what should reproduce that result live.

## 4. Workload — must exercise BOTH axes (the capstone-specific design)

A single homogeneous workload cannot test a 2-axis rule (Track B's C∈{8,32,64} fixed-mix only moves load). The
controller is validated on **mixed traffic** that varies both axes within one stream:

- **Class blocks:** alternate segments of (a) **diverse** prompts (A/C-like, from the ShareGPT-V3 distinct bank) and
  (b) **high-reuse** prompts (B-like, a small set repeated), so `class_HIGHREUSE` should be high in (b) and low in (a).
- **Load sweep:** run the mixed stream at three applied loads (low / mid / high, mapped to the frozen λ=C/28 points
  c16 / c48 / c96 as in H-CLASS-1, open-loop Poisson for clean load application) so `load_LOW` is true only at low.
- **Expected 2-axis behavior (the prediction being tested):** CTRL-LIVE turns SD **ON** in high-reuse blocks even at
  high load (class axis rescues it — the Bc48/Bc96 cells), and **OFF** in diverse blocks at mid/high load (load axis
  — the Ac48/Ac96 cells), while staying ON at low load regardless of class. CTRL-LOADONLY fails the first;
  CTRL-CLASSONLY fails the second.

## 5. Open signal-design questions — resolve by smoke test BEFORE freezing [CAL]

These are genuine design choices, not constants to guess. A single mixed-workload smoke run (one low + one high load,
both arms' signals logged) resolves them; the winning choice is frozen in an append-only [CAL] block:

1. **Load signal:** KV-usage-fraction vs running-occupancy vs a combination — pick whichever separates the c16/c48/c96
   operating points most cleanly on the smoke traces (and confirm it does NOT track `schedule()` cadence, per Defect 2).
2. **Class signal robustness across OFF periods:** is acceptance-with-decay stable enough to avoid flapping, or is a
   prefix-cache-hit-rate secondary proxy needed? Measure transition count in the smoke run; freeze accordingly.

(If either signal turns out to be non-discriminating on this workload — as Track B's acceptance-ratio trigger was for
its *reactive* use — that is a recorded finding bounding the claim, not a silent retune.)

## 6. Success / falsification criteria (fixed before runs)

Let `X̄_C ± CI` be mean output-throughput and 95% CI of arm X in cell C; δ = baseline run-to-run uncertainty.
≥3 valid repetitions per cell, fresh server per trial, counterbalanced order.

**SUCCESS requires ALL:**
- **S1 (crossover recovery):** at high-load diverse blocks, CTRL-LIVE ≥ AR within CI (lower bound > −2%) AND > DSpark-K7
  in ≥2/3 paired rounds with mean Δ>0.
- **S2 (low-load preservation):** at low load, CTRL-LIVE ≥ DSpark-K7 − δ (no meaningful loss of the benefit).
- **S3 (both axes necessary — capstone claim):** on the mixed workload, CTRL-LIVE beats **both** CTRL-LOADONLY and
  CTRL-CLASSONLY in aggregate realized gain (the live analog of the payoff-surface regret table). This is the
  distinguishing result: it shows the *combination* matters end-to-end, not just statically.
- **S4 (overhead):** CTRL-ON vs DSpark-K7 ≤ 1% relative at every load (decision/patch overhead negligible).
- **S5 (causal correctness):** Track B's causal-verification assertion passes in all valid trials (OFF step ⇒ 0 new
  drafts; ON decode step ⇒ K=7 drafts).

**FALSIFIED:** S1 or S3 fails beyond uncertainty after the single preregistered run set (no retune loop; one revision
permitted only on a harness bug, charter §13). **INCONCLUSIVE:** if baseline δ is too large to order arms — report
per-trial values and the uncertainty needed.

**Prohibited claims (even on success):** outperforming native adaptive verification (never run on this hardware);
generalization beyond Qwen3-4B / dspark_qwen3_4b_block7 / SM86; novelty of load-aware K per se (suspended claim list).

## 7. Calibration block [CAL] — to be appended from the §5 smoke step before any controller trial

(empty by design)

## 8. Deliverable & scope

- `docs/phase7-capstone-results.md`: the realized-gain table across load×class, per-axis ablation (S3), overhead
  (S4), causal verification (S5), and the verdict against §6.
- Controller code committed to the isolated worktree branch with its own commit; patch diff reproducible.
- Closes the project: converts the validated signal into a working scheduler and quantifies its end-to-end value.

**Honest scope:** validates the *on/off* 2-axis rule at fixed K=7 on one model/draft pair/hardware. It does not make
K itself adaptive (blocked), does not claim generalization, and its class signal is population-level by design (Track A
showed per-request is the wrong granularity).
