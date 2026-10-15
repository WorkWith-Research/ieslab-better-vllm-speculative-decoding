# Track-A preregistration — content-based per-request acceptance predictor

**Date opened:** 2026-10-13 · **Status:** PROSPECTIVE (fixed before any label collection)
**Issue:** #29 · **Parent findings:** H-CLASS-1 (`docs/hclass1-results.md`) + H-CLASS-1C (`docs/hclass1c-results.md`)

## 0. Question

Can the per-request draft acceptance rate of native fixed-K=7 DSpark be predicted from
**admission-time prompt features** (content/draftability), and does that prediction have
**decision value** for per-request SD on/off?

H-CLASS-1C established: (i) acceptance is driven by prompt **content**, not cache state
(C~A: 0.30–0.32 vs B's 0.61 despite a genuinely cached shared prefix); (ii) acceptance is
**load-invariant** (flat across c16/48/96 in both A and C). Both facts license this design:
labels can be collected at concurrency=1 (clean per-request attribution), and the learned
content→acceptance map should transfer to serving under load.

## 1. Label collection (fixed now)

**Per-request label:** acceptance rate = Σ accepted draft tokens / Σ drafted tokens over all
decode steps of one request, obtained as **deltas of the engine's cumulative Prometheus counters**
(`vllm:spec_decode_num_draft_tokens`, `vllm:spec_decode_num_accepted_tokens`) queried once after
each request completes. At strict concurrency=1 (no overlapping requests) the delta equals the
request's own counts exactly — no worktree instrumentation required. Same method for
`prefix_cache_queries/hits` → per-request cache hit rate.

**Environment:** Qwen3-4B + `deepseek-ai/dspark_qwen3_4b_block7` K=7, 1× RTX 3090 (two parallel
servers on GPU0+GPU1 splitting the prompt set; ports 8403/8404), vLLM V2 async scheduling, util
0.90, max_num_seqs 64, prefix caching ON, `temperature=0`, `max_tokens=2048`, stop token 151645
(`<|im_end|>`), non-streaming with `prompt_logprobs=1` (needed for the perplexity feature; prefill
runs regardless, so this is free compute).

**Prompt sets (both from existing banks; seed 20261013):**
- **Set 1 — main content set (~400 prompts):** condition-A bank (diverse distinct ShareGPT-V3
  turns, ISL brackets: short ~50 tok / long ~2048 tok), each prompt used once → no prefix reuse
  (hit rate ≈ template-prefix only). Content features reflect the full prompt.
- **Set 2 — cache probe (~150 prompts):** condition-C bank (shared ~552-tok context doc + distinct
  query; S≈600 / L≈2600 tok), run sequentially so the shared prefix is cached after the first
  request → per-request hit rate varies (S: ~0.9, L: ~0.2). Provides within-set variance in cache
  state for H-TA2.

**Inclusion:** requests with realized OSL ≥ 10 tokens (shorter responses give noisy acceptance
estimates at K=7); excluded count reported. Target N ≈ 550; minimum viable N = 400.

## 2. Admission-time features (fixed now — no post-admission information)

Computed from the prompt token ids (+ prefill logprobs) only:
1. `isl` — prompt length in tokens.
2. `unigram_entropy` — Shannon entropy of the prompt's unigram token distribution (bits/token).
3. `bigram_rep` — fraction of bigrams that are a repeat within the prompt (redundancy).
4. `trigram_rep` — same for trigrams.
5. `prompt_ppl` — exp(mean logprob) over prompt positions 1..T−1 from `prompt_logprobs`
   (target-model perplexity of the prompt; "free" since prefill runs at admission anyway).
6. `cache_hit_rate` — per-request prefix-cache hit fraction (cost-side state).

**Model tiers:** B0 = intercept-only; B1 = {isl} (the H-OBS-3/H-C3 length framing);
M1 = {isl, unigram_entropy, bigram_rep, trigram_rep} (cheap tier: no model forward needed);
M2 = M1 + {prompt_ppl, cache_hit_rate} (full tier). Models: ridge (α by 5-fold CV) and
gradient-boosted trees. Primary estimator: **5-fold CV with prompt-disjoint folds** (no prompt or
its prefix-sharing neighbor appears in both train and test of a fold); secondary: single 70/30
prompt-disjoint holdout (seed 20261013).

## 3. Hypotheses and verdict bands (fixed now)

- **H-TA1 (predictive power),** primary = M2 5-fold CV on per-request acceptance:
  - **SUPPORTED:** R² ≥ 0.30 AND ΔR² vs B1 ≥ 0.10.
  - **PARTIAL:** R² ≥ 0.15 AND beats B1 (ΔR² > 0).
  - **NOT SUPPORTED:** otherwise.
- **H-TA2 (cache invariance — per-request confirmation of C):** within Set 2, adding
  `cache_hit_rate` to M1 changes 5-fold CV MAE by < 0.003 → cache state is NOT an acceptance
  (quality) feature. Otherwise report the magnitude and direction.
- **H-TA3 (decision value),** on existing H-CLASS-1 A+C paired data (per-request DSpark/AR TPOT
  ratio, prompt-disjoint from nothing — it's a separate population; n ≈ 1500): Spearman
  correlation between predicted acceptance and per-request SD payoff (1 − ratio) ≥ +0.25 AND a
  threshold policy (SD-on iff pred ≥ θ*, θ* chosen on the label set to maximize simulated payoff)
  beats BOTH always-on and always-off on mean regret at ≥ 2 of the 3 loads → **SUPPORTED**.
  Otherwise report the correlation and per-load regret table.

**Causal limits:** labels come from concurrency=1 runs; transfer to serving-under-load rests on
the measured load-invariance of acceptance (H-CLASS-1 §A/C). Realized OSL is an outcome, not a
feature. The predictor is descriptive of the DSpark-K7/Qwen3-4B pair; generalization across
draft models is out of scope.

## 4. Hard stops (abort, no re-tune)

>6 GPU-hours total collection; >5% client errors; counter-delta smoke test fails (labels not
attributable); N < 400 after exclusions; any post-hoc feature/model/window change after this file
is committed.

## 4b. Deviation D1 (recorded during collection, 2026-10-13) — per-request cache state unavailable

The engine's `vllm:prefix_cache_queries_total` / `_hits_total` counters do NOT update per request on
this vLLM build (frozen at their startup value across many requests in an isolated probe, while the
`spec_decode_num_{draft,accepted}_tokens_total` counters update correctly). Therefore per-request
`cache_hit_rate` cannot be attributed via counter deltas.

Consequences:
- **H-TA2 (cache invariance) is UNTESTABLE as written** and is reported as such, not forced. This is
  acceptable because H-CLASS-1C already established at the *workload* level that acceptance is flat
  across cache conditions (hit rate ~0 → 0.976 while per-request acceptance stays ~0.30–0.33). The
  per-request confirmation was a nice-to-have, not load-bearing.
- `cache_hit_rate` in M2 is set to 0 for all rows (no information); on SET 1 (all distinct prompts, no
  reuse) it carries no variance anyway, so **M2 ≡ M1 on the primary H-TA1 test**. The primary verdict
  therefore rests on M1's content features + prompt_ppl; this does not weaken H-TA1 (the question is
  whether *content* predicts acceptance).

The spec-decode draft/accept counters (the load-bearing label source) were verified correct in the smoke
test (per-request deltas match expected magnitudes; mean acceptance ≈ A anchor). No other deviation.

## 5. Provenance (filled after run)

- Collection: `experiments/tracka_collect.py` → `results/tracka/labels_{set1,set2}.jsonl` (local).
- Training/eval: `experiments/tracka_train.py` → `results/tracka/model_report.json`.
- Payoff join for H-TA3: from existing `results/hclass1/{ar,dspark}_{A,C}_c*/client.jsonl` +
  `prompt_manifest.jsonl`.
- Report: `docs/tracka-results.md`.
