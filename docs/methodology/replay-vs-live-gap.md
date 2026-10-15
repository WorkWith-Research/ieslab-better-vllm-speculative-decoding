# Replay vs Live Gap (Priority B)

**Status:** active investigation. First live A/B (v1, single run) classified INCONCLUSIVE;
candidate explanations below, each with a planned isolation experiment.

## Observed discrepancy
- **Offline causal replay** (`experiments/ldm_eval.py`): measured cost model c(k) + per-request J
  profiles → best fixed-K value 1.730, oracle +12.1%, causal LDM recovers 83–89% of the gap
  (~+10% in the replay objective). Replay's argmax fixed-K = K=1 (its cost model penalizes deep K
  when J decays fast).
- **Live vLLM serving** (A/B v1, mixed workload): best *fixed* config was K=8 (793.6 tok/s) > K=4
  (773.9); LDM 785.7 — i.e. live ranking of fixed-Ks is the **opposite direction** at the top end
  (K=8 best live, K=1 best in replay), and adaptive-K ≈ best fixed-K (−1.0%, single run).

The two are not directly comparable numbers (different objectives/workloads), but the *ranking*
disagreement is a first-class problem: it means at least one of {cost model, J-profile stationarity,
batch coupling} is materially wrong in one of the two settings.

## Candidate explanations & isolation experiments

### E1 — Batch-level verification coupling (top candidate)
The replay scores each request independently: produced/cost with cost = c(k) per request-step.
Live, one target forward pass serves the **whole batch**; a high-K request's extra draft tokens are
verified almost for free when other requests already fill the pass, while an all-low-K batch wastes
the pass's capacity. Consequence: over-speculation is amortized (cheap), under-speculation loses its
benefit → live optimum shifts to higher K than the per-request replay predicts.
**Isolation:** sweep concurrency C ∈ {1, 4, 16, 32} for fixed K=2 vs K=8 on the mixed workload. If
the K=8 advantage *grows with batch size* (and disappears at C=1 where there is no coupling), E1 is
supported. At C=1, live per-request ranking should match the replay's ranking if the cost model is
right — a direct cross-check of c(k).

### E2 — Cost model measured on the wrong workload
c(k) was derived from Phase-1 **random-template** runs (high ngram acceptance, accept-len 1.96→7.74).
On the mixed workload, per-position acceptance is much lower (K=8: 0.66→0.31), so the *realized*
verify cost per produced token differs from c(k) as measured.
**Isolation:** re-derive c'(k) from A/B-grid runs themselves (throughput ÷ accept-length per config,
as in Phase 1) and check whether replay with c' reproduces the live fixed-K ranking. If yes, E2 is
the main cause and the oracle-gap number must be recomputed on-workload.

### E3 — Non-stationary / spiky acceptance breaks the online estimator (live-specific)
Replay feeds each request its *true* stationary J profile into the estimator; live ngram acceptance
is spiky with **lag-1 autocorrelation ≈ 0** (measured on A/B v1 decision log). A W=8 window then
cannot predict the next step's regime, so live LDM performance < replay LDM performance even with a
perfect cost model. This affects only the adaptive-K arm, not the fixed-K ranking.
**Isolation:** run LDM with a *reactive* (last-step) rule vs W=8 window on the same workload; also
replay the A/B v1 **recorded live accept sequences** through the replay harness (same estimator,
live data instead of synthetic-from-J). If replay-on-live-data ≈ live LDM, E3 is confirmed and the
replay-vs-live gap for the adaptive arm is explained by data non-stationarity.

### E4 — Ragged variable-K execution overhead (eager mode)
Variable K makes the batch ragged; eager-mode CPU/GPU sync costs may penalize the LDM arm beyond
the algorithmic effect.
**Isolation:** `ldm_log` config (K=8 server + controller logging, NO truncation) vs fixed K=8: same
server, same hooks minus truncation → difference = pure controller overhead (decision compute +
logging). If ldm_log ≈ k8 within noise, enforcement overhead is negligible and E4 rejected.

## Rejected so far
- *Truncation not taking effect* — validated: effective draft-length distribution matches decisions.
- *Client serial-sending artifact* — fixed (worker pool), re-measured.
- *Cold-start collapse* — fixed (optimistic cold start); verified differentiation high→6/med→3/low→1.

## Partial results (2026-10-07, CPU analysis of A/B v1 decision log)
**Live-data oracle gap under the per-request cost model:** best fixed K=**1** (1.690), oracle 1.781
(+5.4%). This **directly contradicts live serving**, where K=8 (793.6) beat K=4 (773.9). The
contradiction is the strongest evidence yet for **E1**: at concurrency 16, one target forward pass
serves the whole batch, so a request's extra draft tokens are verified almost for free — the
per-request cost model c(k) massively overstates the marginal cost of deep K in a busy batch. E1's
C-sweep (especially C=1, where per-request ranking should match the model) is now the decisive test.

**Observation-bias caveat (important):** the live decision log only records acceptance up to each
step's *effective* K — for steps drafted at K=1 we never see what positions 2–8 would have accepted.
So live J profiles are **truncated**, and any replay over them is a lower bound; in particular the
LDM-replay-on-live-sequences score (1.415, −16% vs fixed K=1) reflects this bias plus the cost model,
and must not be read as "the live controller underperformed its own replay". Only counterfactual
runs (same workload at different fixed Ks) give unbiased per-k values — which is exactly what the
Priority-A grid provides.

## Implications for the oracle calculation
Until E1–E2 are resolved, the replay oracle gap (~12%) is an **upper-bound estimate under a
per-request cost model**, not a live-serving guarantee. The honest current statement: per-request
profitable-K heterogeneity is real (Phase 2), a causal controller can exploit most of it *in replay*,
and live end-to-end benefit is **unproven** (A/B v1 INCONCLUSIVE). Priority A grid + E1/E2 isolation
are the deciding experiments.
