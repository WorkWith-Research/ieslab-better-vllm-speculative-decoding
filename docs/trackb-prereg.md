# Track B Step 2 — Preregistration: minimal live-state SD on/off controller vs native fixed-K DSpark

- **Date:** 2026-10-09 (KST)
- **Status:** PREREGISTERED (design + success/falsification criteria fixed). Calibration numbers marked **[CAL]** are filled from the Phase 6 reproduction analysis BEFORE any controller trial runs, in an append-only calibration block. No controller trial may start until that block exists.
- **Directive:** §4 Steps 2–3. Correction binding: **no hard-coded concurrency thresholds** ("if concurrency == 64: disable SD" is prohibited); the decision must be a function of *measured serving state*; decisions must be verified to affect the real execution path.

## 1. Hypothesis (H-TB)

A controller that turns native DSpark speculation **on/off per scheduling step** based on measured serving state (rolling draft-acceptance economics + load pressure) recovers most of fixed-K DSpark's loss at/above the crossover while preserving its low-load benefit, versus both native AR and native fixed-K=7 DSpark on the identical runtime.

## 2. Runtime and arms

- **Runtime:** identical to Phase 6 / Gate E: Qwen3-4B + `dspark_qwen3_4b_block7`, vLLM v0.25.0+cu126 (isolated worktree), GPU 1, same server flags, ShareGPT_V3, 60 prompts, greedy, seed 0.
- **Arms:**
  1. **AR** — no spec config (Phase 6 data reused if valid; else re-run).
  2. **DSpark-K7** — native fixed K=7 (Phase 6 data reused if valid; else re-run).
  3. **CTRL-ON** — patched runtime, policy = constant ON. Purpose: quantifies patch/decision overhead against the unpatched baseline (validity control).
  4. **CTRL-LIVE** — patched runtime + the live-state rule below (the hypothesis arm).
- Conditions: C ∈ {8, 32, 64} (same as Phase 6), **≥3 valid repetitions each**, fresh server per trial, counterbalanced order as in Phase 6.

## 3. The controller (simplest causal rule)

**State measured at the scheduler every step (all already computed there — no new sensors):**
- `r_window` = rolling-window acceptance ratio over the last **W = 2 s** of wall time: `accepted_tokens / draft_tokens` (from the existing spec-decode counters; window kept as a deque of per-step deltas).
- `load_pressure` = measured `(waiting_reqs, kv_cache_usage)` — used only for the hysteresis/anti-flap term, never as a standalone threshold.

**Action (per scheduling step):**
```
if sd is ON:
    if r_window < τ_off  → set sd OFF, until t + T_off      # cooldown
else:
    if (now > off_deadline) and r_window > τ_on → set sd ON   # hysteresis re-entry
```
- **τ_off, τ_on [CAL]:** calibrated from Phase 6 data as the acceptance ratio at which DSpark's marginal throughput gain per draft token crosses zero in each regime (empirical break-even, not a load threshold). Expected: τ_off < τ_on (hysteresis band) to prevent flapping.
- **T_off [CAL]:** initial value 1 s; fixed before runs; sensitivity (0.5 s / 2 s) reported as secondary, not tuned-to-positive.
- **Cold start:** sd = ON for the first W seconds of a trial (window empty ⇒ no evidence to turn off). This preserves low-load behavior by default.
- **What it is NOT:** no learned model (per directive), no per-request K, no variable K (blocked — `docs/trackb-feasibility.md` §2 B1–B3), no concurrency threshold anywhere in the rule.

**Why this state, briefly:** at high load DSpark's verification cost is paid for a degrading acceptance rate (Gate E: ~0.25 accepted/draft-token at C=64). The rolling acceptance ratio is the *measured* signal of exactly that economics; waiting/KV pressure only shapes re-entry timing. A pure batch-size rule would be the prohibited hard-coded form.

## 4. Implementation (minimal patch to isolated v0.25.0 build)

1. `SchedulerOutput`: add `sd_enabled: bool` (default True).
2. Scheduler step end: evaluate the rule from measured counters; when turning OFF, also discard pending `request.spec_token_ids` so no new drafts are scheduled next step (`scheduler.py:1948` path).
3. V2 model runner: gate `self.speculator.propose(...)` (`gpu/model_runner.py:~1459`) on `sd_enabled`; when false the step proceeds through the existing no-draft "common case" decode branch (verified present, §feasibility 1.6).
4. **Causal-verification logging (mandatory):** per step append `(t, sd_enabled, drafts_scheduled_this_step, accepted_this_step)` to a JSONL sidecar. Post-run assertion: **every OFF step has 0 new draft tokens scheduled; every ON decode step schedules K=7 drafts** (modulo the one-step lag documented in feasibility §3). If the assertion fails, the trial is INVALID (harness bug), not a null result.
5. Decision overhead measured as CTRL-ON vs DSpark-K7 throughput difference (same patch, constant policy) — reported explicitly, success requires it to be small (§6).

## 5. Validity rules

Same as Phase 6 §4 (completed=60, no OOM/crash, scraper ≥95% coverage, GPU exclusive), plus:
- causal-verification assertion passes (§4.4);
- OFF-fraction and ON-fraction both > 0 for CTRL-LIVE at C=32/64 (a policy that never flips is a degenerate implementation, reported as such — not silently accepted);
- patch diff is committed to the isolated worktree with its own commit (reproducible), applied only when GPU idle.

## 6. Success / falsification criteria (fixed before runs)

Let `X̄_C ± CI_C` be the mean output-throughput and 95% CI of arm X at condition C (n≥3). Let `δ_C = width(CI_DSpark-K7,C)` (run-to-run uncertainty of the baseline).

**SUCCESS** requires ALL of:
- **S1 (crossover recovery):** at C=64, `CTRL-LIVE > DSpark-K7` by more than `δ_64` (i.e., improvement beyond baseline measurement uncertainty); AND CTRL-LIVE ≥ AR within CI.
- **S2 (low-load preservation):** at C=8, `CTRL-LIVE ≥ DSpark-K7 − δ_8` (no meaningful loss of the low-load benefit).
- **S3 (overhead):** CTRL-ON vs DSpark-K7 throughput difference ≤ 1% relative at every C (decision/patch overhead negligible or explicitly quantified).
- **S4 (causal correctness):** §4.4 assertion passes in all valid trials.

**FALSIFIED:** S1 fails beyond uncertainty at C=64 after the single preregistered run set (no re-tuning loop; one revision of τ/T_off is permitted ONLY if a harness bug is found, per charter §13).
**INCONCLUSIVE:** δ_64 is so large that no arm ordering at C=64 is distinguishable — report per-trial values and state what uncertainty level would be needed.

**Prohibited claims (even on success):** outperforming native adaptive verification (never executed on this hardware); generalization beyond Qwen3-4B/dspark_qwen3_4b_block7/SM86; novelty of load-aware K per se (suspended claim list, addendum §2).

## 7. Calibration block [CAL] — to be appended from Phase 6 analysis before any controller trial

(empty by design)

---

## 8. Dated revision R1 (2026-10-09, post Phase 6 analysis — original §3 text preserved above)

**Trigger revision, motivated by measured evidence (`docs/phase6-reproduction-prereg.md` §8.5):**
Per-position acceptance rates are load-invariant across C=8/32/64 (0.72→…→0.03 at every concurrency; acceptance length ≈ 2.77). A rolling-acceptance-ratio trigger therefore has **no discriminating signal** on this workload and would never fire. The crossover is a verification-compute-amortization effect under GPU compute saturation, with TPOT crossing over AR already at C=32 (26.2 vs 21.4ms) — earlier than throughput.

**Revised state/action rule (replaces §3's primary trigger; hysteresis/cooldown structure unchanged):**
- **Primary state:** rolling-window (W = 2 s) measured **TPOT trend** (slope of per-step inter-token latency, from the scheduler's own step timing / spec stats — no external sensor), plus current **batch occupancy** (`len(running)` as a fraction of `max_num_seqs`) and **prefill share** of the last window (fraction of scheduled tokens that are non-draft/prefill).
- **Action:** OFF when TPOT trend is rising AND (occupancy high OR prefill share high) — i.e., verification cost is no longer amortizing under load; ON (re-entry) after cooldown when TPOT trend flattens/falls. τ values [CAL] from Phase 6 per-cell TPOT/throughput operating points; the rule contains **no concurrency threshold** (occupancy enters as a continuous fraction, and only in conjunction with the latency trend).
- Acceptance ratio retained as a **secondary guard only** (protects against a genuinely degrading drafter on other workloads; cannot be the sole trigger per §8.5 finding 3).

**Unchanged:** success/falsification criteria (§6), validity rules (§5), causal-verification logging (§4.4), arm set, conditions, repetition count, and the prohibition on tuning-to-positive (one revision only, charter §13). The [CAL] block is filled from Phase 6 numbers below before any controller trial:

### [CAL] (filled 2026-10-09 from Phase 6 results)

- Operating points (mean of 3 trials): C=8: AR TPOT 13.55ms / DSpark 9.04ms (SD beneficial); C=32: AR 21.43 / DSpark 26.18 (SD costs +4.75ms/step); C=64: AR 34.92 / DSpark 47.21 (+12.29ms/step).
- τ_off [CAL] = TPOT_trend slope such that marginal SD step cost > 0, anchored at the measured +4.75 ms/step penalty at C=32 (initial: trend > +1 ms per 0.5 s window sustained over W).
- τ_on [CAL] = re-entry when trend ≤ −1 ms per 0.5 s for W (hysteresis band ±2 ms/s around zero slope).
- T_off [CAL] = 1 s (sensitivity 0.5/2 s reported secondarily).
- Occupancy/prefill-share terms enter as continuous modifiers of the effective threshold, not as standalone cutoffs.

These are initial values fixed before controller trials; any post-hoc change beyond one charter §13 revision invalidates the run set.
