# Track-A results — content-based per-request acceptance predictor (Phase 5, Issue #29)

**Date:** 2026-10-13 · **Prereg:** `docs/tracka-prereg.md` (prospective; bands fixed before collection)
**Parent findings:** H-CLASS-1 + H-CLASS-1C — acceptance is content-driven (not cache) and load-invariant.

## 0. Verdict (per prereg §3 fixed bands)

> **H-TA1 NOT SUPPORTED.** Admission-time prompt features explain only ~5% of per-request DSpark
> acceptance variance (M2 5-fold CV R² = 0.055; length-only baseline B1 R² = 0.008). Per-request
> acceptance is **not** predictably determined by prompt content at the individual level, and a
> threshold policy on predicted acceptance has **no decision value** (H-TA3 Spearman ≈ 0.04).

| Hypothesis | Result | Key number |
|---|---|---|
| H-TA1 predictive power (M2 vs B1) | **NOT SUPPORTED** | M2 R² = 0.055, ΔR² vs B1 = +0.047 (bands: need ≥0.30 & ≥0.10) |
| H-TA2 cache invariance (per-request) | **INVARIENT** (weak test — see D1) | ΔMAE(M1→M2+cache) = 0.00027 < 0.003 |
| H-TA3 decision value (threshold policy) | **NOT SUPPORTED** | Spearman(pred_acc, payoff) = 0.041; beats both baselines at 0/3 loads |

## 1. Label collection (prereg §1)

550 prompts labeled at strict concurrency=1 via `spec_decode_num_{draft,accepted}_tokens_total`
counter deltas (exact per-request attribution, no vLLM patch). **set1 = 400/400 valid** (diverse
ShareGPT-V3, S/L ISL brackets; primary), **set2 = 150/150 valid** (shared-doc C bank; cache probe).
Zero client errors. Per-request acceptance: set1 mean 0.315, sd 0.070, range 0.14–0.62 — right at the
H-CLASS-1 A anchor (0.33), with large per-prompt variance (the signal a predictor must capture).

## 2. H-TA1 detail (5-fold CV on set1, prompt-disjoint folds)

| model | features | R² (ridge) | MAE | R² (GBT) |
|---|---|---:|---:|---:|
| B0 | (intercept) | −0.019 | 0.0519 | — |
| B1 | isl | 0.008 | 0.0508 | −0.082 |
| M1 | isl, unigram_entropy, bigram_rep, trigram_rep | 0.054 | 0.0492 | −0.121 |
| **M2** | M1 + prompt_ppl, cache_hit_rate | **0.055** | 0.0492 | −0.095 |

- The content features (entropy/repetition/perplexity) add a small but real signal over length alone
  (ΔR² ≈ +0.047 ridge), yet even the best model leaves ~95% of acceptance variance unexplained.
- GBT underfits (negative R²) — no nonlinearity to exploit; ridge is the better estimator here.
- Per-fold R² for M2: 0.066 / 0.067 / 0.113 / 0.045 / −0.013 — positive in 4/5 folds but far below bar.

## 3. H-TA2 (cache invariance) — weak, but consistent with C

Within set2 (all prompts share doc D), adding `cache_hit_rate` to M1 changes MAE by 0.00027 (< 0.003
band) → cache state is not an acceptance *quality* feature at the per-request level. **Caveat:** only
2/150 set2 rows had a non-null per-request hit rate (deviation D1 — the engine's prefix-cache counters
don't tick per request on this build), so the test has near-zero cache variance and is essentially
informative, not confirmatory. The *load-bearing* invariance evidence remains H-CLASS-1C's workload-level
result (hit rate 0→0.976 with acceptance flat at ~0.30–0.33).

## 4. H-TA3 (decision value) — the payoff isn't content-determined

On the existing H-CLASS-1 A+C paired per-request data (n = 2011, prompt-matched DSpark/AR TPOT):
- Spearman(**predicted** acceptance, SD payoff = 1−ratio) = **0.041**.
- Control: Spearman(**true ISL**, payoff) = **0.033** (c16 0.018 / c48 0.064 / c96 0.034). Even the
  ground-truth input length is ~uncorrelated with which requests benefit from SD.
- Threshold policy (SD-on iff pred ≥ θ*, θ* = median): beats both always-on and always-off at **0/3**
  loads. Mean payoff is positive at c16 (+0.116) but negative at c48/c96 (−0.165/−0.172) — the sign is a
  *load* property (reproduces H-CLASS-1 C), not a per-request one.

## 5. Interpretation and what it means for the scheduler

1. **Per-request acceptance is high-variance and weakly content-determined.** Within a diverse
   population, individual prompts' draft acceptance (sd ≈ 0.07 around ~0.32) is driven by factors our
   cheap admission-time features don't capture (likely fine-grained local token predictability / draft-model
   interaction), not by coarse statistics (length, entropy, repetition, perplexity).
2. **The A↔B gap was a *population* effect, not a per-request one.** H-CLASS-1C's 0.33→0.61 contrast was
   between two *distributions* of prompts (diverse mix vs 2 repeated turns). Within the diverse distribution,
   there is little exploitable spread — so "predict this request's acceptance" is the wrong framing.
3. **Revised scheduler signal (supersedes Track-A per-request prediction):** SD on/off and K should be set
   from **load regime + workload class**, not per-request content:
   - *Load* decides the sign (SD helps at low load, mildly hurts at mid/high — H-CLASS-1 C, reproduced here).
   - *Workload class / prompt distribution* (e.g., "repetitive high-reuse traffic" vs "diverse") sets the
     expected acceptance level — a coarse admission feature that *is* available and *was* decisive in B.
   A lightweight decision model should therefore ingest **serving state** (queue, running, load) + a coarse
   **traffic-class** label, not per-request perplexity.

## 6. Causal / scope limits

Labels from concurrency=1; transfer to serving rests on measured acceptance load-invariance. Predictor is for
the DSpark-K7 / Qwen3-4B pair only. H-TA2 under-powered (D1). All claims descriptive.

## 7. Provenance

- Labels: `results/tracka/labels_set1.jsonl` (400), `labels_set2.jsonl` (150) — local, gitignored.
- Collection: `experiments/tracka_collect.py`; training/eval: `experiments/tracka_train.py`.
- Report data: `results/tracka/model_report.json`, `model_table.md`. Deviation D1 in prereg §4b.
