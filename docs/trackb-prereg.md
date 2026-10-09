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
