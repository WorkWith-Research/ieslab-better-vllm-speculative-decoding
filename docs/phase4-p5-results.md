# Phase 4.5 — DSpark-rule vs LDM in-loop (result)

**Question (supervisor):** "Are you using/comparing DSpark?" We reimplemented DSpark's *decision rule*
(batch-level global greedy over per-position prefix-survival with a hardware-profiled SPS(B) curve, objective
Θ=τ·SPS(B)) as an in-loop baseline on our vLLM prototype, and compared it against our acceptance-only LDM and
fixed-K baselines at the two load regimes Phase 4.2 identified. Full faithfulness notes + deviations (incl. the
pre-registered SD-off fallback) in `docs/related-work-dspark.md` §5 and PROGRESS.md.

**Setup:** Qwen2.5-7B-Instruct, 1× RTX 3090, vLLM 0.19.1 eager, mixed workload (Phase 4.2), C ∈ {32, 96} × 3 trials.
Arms: AR / K4 / K8 (reused from Phase 4.2 cells) + LDM (local EMA acceptance, measured cost c(k)) + DSpark-rule
(batch-greedy, SPS(B) table from Phase 4.2). All in-loop via sitecustomize monkey-patch; KMAX=8.

## Result — at saturation, neither adaptive policy reaches AR; DSpark-rule is the *worst* arm

| arm | C=32 tok/s | C=96 tok/s | gap vs AR @C=96 |
|---|---|---|---|
| AR | 1449.4 | **2457.6** | — |
| K8 (best fixed) | 1464.5 | 1947.6 | −20.8% |
| LDM | 1432.9 | 1960.1 | −20.2% |
| DSpark-rule | 1445.4 | **1853.1** | **−24.6%** |

- **C=32 (transition):** all arms within ~1% — SD is roughly neutral here, as Phase 4.2 predicted. No separation.
- **C=96 (compute-saturated):** AR wins by 20–25%. **DSpark-rule is the worst arm** — it loses even to fixed K8.
  LDM edges out both (best adaptive) but still sits ~20% below AR.

## Why DSpark-rule underperforms (the mechanism, from decision logs)

| arm | C=96 mean K | frac(K=0 = SD off) | K distribution shape |
|---|---|---|---|
| LDM | 3.88 | **1.2%** | bimodal: K=1 (54%) / K=8 (39%) |
| DSpark-rule | 2.77 | **29.5%** | broad: K=0 (29%), K=1 (22%), K=8 (17%) |

- **DSpark-rule does turn SD off for ~30% of requests** (its SPS(B) term penalizes large B), but keeps the rest
  at high K (mean 2.77). Its objective Θ=τ·SPS(B) *overestimates* the value of speculating: τ is computed from
  empirical acceptance J_r[j], which is high for this repetitive workload, so the greedy keeps extending even
  though the true marginal cost — step-time inflation in a compute-saturated batch — is steeper than the
  profiled SPS(B) curve implies. The profile was taken at *uniform* B=C(1+K); with per-request variable K the
  actual per-step B wanders and the interpolation misses the real opportunity cost. Net: −24.6%, worse than a
  dumb fixed K8.
- **LDM never commits to SD-off** (frac K=0 = 1.2%): its per-request value ratio ΣJ/c(k) stays above the AR
  baseline because acceptance is decent, so it keeps speculating at K=1/8 even though the *batch* would run
  faster in pure AR. It lacks any notion that the system is saturated.

## Interpretation — this is the finding the LDM idea needs

1. **The load-driven-K premise is real (Phase 4.2), but DSpark's actual decision rule does not capture it.** A
   batch-coupled objective over a *profiled* SPS(B) curve is not enough to reach the saturated-regime optimum;
   on our hardware/workload it is worse than fixed-K. So "just adopt DSpark" is **not** the answer — this is a
   direct, measured rebuttal to that framing (answering the supervisor's question with evidence).
2. **The missing ingredient is a live load-regime signal.** The winning policy at C=96 is "SD off" (AR), which
   requires *knowing* the system is compute-saturated — a serving-state quantity (measured steps/s, TPOT trend,
   or B vs the SPS(B) knee) that **neither** LDM (acceptance-only) **nor** DSpark-rule (profiled curve only)
   currently consumes. This is precisely the differentiator for our lightweight decision model: per-request
   acceptance × live serving state → K (including off).
3. **Honest limitation of the current LDM prototype:** it also does not win, because it too is acceptance-only.
   Phase 4.5 therefore *motivates* the next step rather than validating the final design: augment LDM with a load-regime
   input so it can commit to SD-off at saturation and large-K when memory-bound. That augmented policy is the
   actual candidate contribution; Phase 4.5 shows both existing baselines (fixed-K, DSpark-rule) leave 20–25% on the
   table at saturation that a load-aware policy should recover.

## Caveats
- ngram self-drafting on a repetitive mixed workload: acceptance is high and fairly uniform across requests, so
  per-request *acceptance* heterogeneity is muted here (it is large on SPEED-Bench, Phase 4.3). The load-regime effect
  dominates this experiment by design.
- SPS(B) table has only 13 points (from Phase 4.2 cells); a denser profile could change DSpark-rule's exact number but
  not the conclusion (its objective structurally over-credits high-K under saturation).
- Deterministic workload → trial variance is jitter-only; C=96 gaps are all ≥20%, far above noise.
