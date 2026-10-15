# Phase-2 Results — Per-request acceptance heterogeneity & the dynamic-K oracle gap

**Question:** within one serving batch, do different requests actually want different
speculation lengths K — and how much throughput could a per-request (dynamic) K policy
recover over any single fixed-K? This is the oracle-gap evidence for the LDM idea.

## Method (new tooling, all in `experiments/`)
- **`spec_hook/sitecustomize.py`** — non-invasive, env-gated observation hook. vLLM 0.19.1
  computes per-request draft/accept counts in `Scheduler.make_spec_decoding_stats`
  (`scheduler.py:1370-1390`, called once per request per step with the `request_id`) but
  immediately aggregates them into batch counters. The hook (injected via `PYTHONPATH`,
  **no vLLM source modified**) tees out one line per (request, decode-step):
  `{t, req, K, acc}`. Activated only when `VLLM_SPEC_HOOK_OUT` is set.
- **`spectrum_client.py` / `sharegpt_client.py`** — Poisson clients. The ShareGPT client sends
  real diverse conversational turns → genuinely request-dependent ngram acceptance.
- **`oracle_gap.py`** — from per-request accept counts builds each request's joint prefix-
  acceptance profile `J[l]=P(acc≥l)` (measured once at Kmax; prefix acceptance is K-invariant),
  then reports: (1) natural-K heterogeneity, (2) truncation headroom, (3) cost-model-sensitive
  oracle gap.

**Key modeling caveat (stated honestly):** vLLM verifies all K drafted tokens in a *single*
target forward pass, so a verification step's wall-clock cost grows only sub-linearly with K and
is amortized across the batch. A naive FLOP model (`cost ∝ 1+K`) is therefore **wrong** — it
predicts K=1 always wins, which contradicts Phase 1's *measured* K=8>K=1 at high acceptance.
So we report (1) and (2), which are cost-model-free, and bracket the % gap with three cost curves.

## Finding 1 — Per-request natural-K heterogeneity is large on realistic workloads
"Natural K" = a request's mean accept length `L=1+Σ_{l≥1}J[l]` (how long its drafts stay valid).

| workload | n | min | p25 | median | p75 | max | **CV** |
|---|---|---|---|---|---|---|---|
| synthetic spectrum (r6) | 66 | 1.24 | 8.07 | 8.67 | 8.89 | 9.00 | 0.25 |
| **real ShareGPT (r6)** | 31 | **1.18** | 2.23 | 3.81 | 6.00 | **8.82** | **0.51** |

On real conversational text, requests in the *same batch* span natural-K **1.18 → 8.82**:
some are near-AR (want K≈1), others highly self-draftable (want K≈8). A single fixed K
therefore simultaneously **over-speculates** the short ones (wastes verify compute on rejected
tokens) and **under-speculates** the long ones (leaves accepted tokens on the table). This is the
concrete "problem" a per-request dynamic-K scheduler must solve. The mismatch is far larger on
realistic text (CV 0.51) than on synthetic random-token prompts (CV 0.25) — which is why it has to
be *measured* per request, not assumed.

## Finding 2 — Truncation headroom (cost-model-free floor on the oracle gap)
Fraction of a request's achievable acceptance potential left unused by capping at K:

| fixed K | ShareGPT potential left unused |
|---|---|
| K=1 | +48.9% |
| K=2 | +35.9% |
| K=3 | +27.3% |
| **K=4** | **+19.9%** |
| K=6 | +8.8% |
| K=8 | 0% (reference) |

At the near-optimal fixed K=4, ~20% of the batch's achievable acceptance is still being truncated
away from requests that would accept longer chains — a lower bound on what per-request assignment
could reclaim in accepted tokens.

## Finding 3 — Realizable throughput gap is cost-model-dependent (bracketed)
`score_r(k)=captured_r(k)/c(k)`; oracle = mean over requests of `max_k score_r(k)`.

| cost model c(K) | best fixed-K | per-request oracle gain |
|---|---|---|
| linear `1+K` (each draft token = full fwd pass) | K=1 | +0.0% *(model is wrong for vLLM)* |
| sqrt `1+√k` | K=8 | **+10.2%** |
| constant (all K in one step) | K=8 | +0.0% |

The realizable % gain sits between these extremes, governed by how cheaply the target model
verifies extra draft tokens (KV/attention over them). The **measured** aggregate anchor on ShareGPT
r6: K=4 = 1037.9 tok/s vs K=8 = 1043.2 tok/s (**+0.5%**) — i.e. at this load/model the *aggregate*
difference between a good and a great single fixed-K is small, **but** the per-request mismatch
(CV 0.51) means the win comes from *per-request assignment*, not from a better global K. It should
grow under higher load (batch composition matters more) and with a real draft model (EAGLE), whose
acceptance is lower and more variable than ngram's — see caveats.

## What this means for Phase 3 (LDM prototype)
- **Decision input confirmed measurable & per-request:** the hook gives each request's live
  acceptance → natural-K in real time, with zero vLLM source changes. This is exactly the state
  feature the LDM reads.
- **Objective is well-defined:** maximize accepted output tokens/verify-step under a per-request K,
  subject to SLO (TPOT) and KV/batch constraints — i.e. assign each request its natural-K (clipped
  to [1, Kmax]) instead of one global K.
- **The gap is real but modest at ngram+Qwen2.5-7B** (single-digit %). The LDM's value proposition
  strengthens with (a) a lower-variance-cost target runner, (b) higher load, and (c) a learned
  draft model — all noted as follow-ups.

## Caveats / limitations
1. **Single drafter (ngram).** ngram acceptance is high and somewhat uniform; EAGLE's lower, more
   variable acceptance should widen the per-request spread and the realizable gap (deferred —
   EAGLE-Qwen2 incompatible with vLLM 0.19.1).
2. **Cost curve not calibrated.** We bracket rather than pin the % gap; a full Phase-3 prototype
   that actually *assigns* per-request K and measures end-to-end throughput would close this.
3. **Per-request sample size** is limited by how often ngram drafts (hard prompts draft rarely);
   we keep requests with ≥10 drafting steps for stable J profiles.

## Reproduce
```
# real ShareGPT, per-request hook on, at K=8 (full profile) and K=4 (measured anchor)
RATE=6 K=8 EXP=P2s_K8_r6 CLIENT=sharegpt NUM_PROMPTS=200 ./experiments/phase2_mixed.sh
RATE=6 K=4 EXP=P2s_K4_r6 CLIENT=sharegpt NUM_PROMPTS=200 ./experiments/phase2_mixed.sh
.venv/bin/python experiments/oracle_gap.py results/P2s_K8_r6 --kmax 8
```
