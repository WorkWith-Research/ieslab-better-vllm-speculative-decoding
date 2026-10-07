# Phase-3 Results — A lightweight causal decision model recovers most of the dynamic-K oracle gap

**Question:** can a *realistic* Lightweight Decision Model (LDM) — one that only observes each
request's accept outcomes as they happen, not its full acceptance profile — actually capture the
per-request dynamic-K headroom identified in Phase 2? This is the first concrete quantification of
the LDM idea's value.

## Method (`experiments/ldm_eval.py`)
Two ingredients, both grounded in measured data (no assumed cost model):

1. **Measured per-step verification-cost curve c(k).** From Phase-1 we recorded, at each K and
   load, output throughput and mean accept length L. Decode steps/sec = throughput / L, so the
   relative verify-step cost is `c(k) ∝ 1/(steps/sec)` → measured:
   `c(1)=1.00, c(2)=1.54, c(3)=1.88, c(4)=2.21, …, c(8)=3.41`. **Sub-linear in K** (c(8)≈3.4, not 8),
   confirming vLLM verifies all K drafts in ~one target forward pass. This is a property of the
   *target model's* verification (same for any workload on this model+hardware), so it transfers.

2. **Causal LDM policy.** Replay each request's recorded accept sequence; at every step the policy
   picks k from an online estimate of the request's joint-acceptance profile (sliding window W over
   past accept outcomes), maximizing `estimated captured / c(k)`. It never sees the request's full
   profile — only what it has observed so far. Realized value uses the true per-step acceptance.

All three policies are scored with one consistent metric: **aggregate produced tokens (accepted
drafts + bonus) / aggregate verify-cost** over an equal horizon, so they're directly comparable and
the oracle is a valid upper bound.

## Result — ShareGPT per-request data, measured c(k)
| policy | value | gain vs best fixed-K | recovers of oracle gap |
|---|---|---|---|
| best **fixed** K (=1 here) | 1.730 | — | — |
| **causal LDM**, W=4 | 1.854 | +7.2% | 59% |
| **causal LDM**, W=8 | 1.904 | +10.1% | 83% |
| **causal LDM**, W=16 | 1.912 | +10.5% | 87% |
| **causal LDM**, W=32 | 1.916 | +10.8% | 89% |
| **oracle** (full knowledge) | 1.940 | +12.1% | 100% |

- The **oracle gap is real and ~12%** (per-request optimal K vs the best single fixed-K), under a
  *measured* cost model.
- A **lightweight causal LDM recovers 83–89% of it** with just a short observation window (W≈8–16).
  It converges to the oracle as W grows — i.e. the residual gap is estimation error, not a policy
  limitation. This is the honest number for the LDM idea: **~10% throughput from per-request K,
  obtainable with an online estimator that needs no perfect knowledge.**

## Why this matters / interpretation
- Phase 2 showed *that* requests want different K (CV 0.51 on real text). Phase 3 shows *how much a
  practical controller can get*: the bulk of the oracle gap, with a simple causal policy.
- The policy is deliberately minimal (estimate profile → argmax captured/cost). It is the skeleton
  of the LDM: replace the hand-rolled estimator with a learned one that also ingests queue/KV/GPU
  state and SLO constraints, and it generalizes to admission + chunk-size decisions.
- **Honest caveats:**
  1. c(k) was measured on the random workload (K=1..8); applying it to ShareGPT assumes the
     verification cost curve is workload-invariant (true for a fixed target model+hardware — cost is
     per verify-step, independent of acceptance). A ShareGPT-native c(k) would need throughput at
     K=1,2 there (only K=4,8 measured); noted as a refinement.
  2. The replay uses each request's *own* accept stream; a live system estimates from the same
     signal in real time (the `spec_hook` provides it). Batch-level coupling (one batch shares a
     verify step) is approximated by the per-step cost, not fully modeled — Phase-3-next.
  3. Single drafter (ngram) + single model. EAGLE's lower/varied acceptance should widen both the
     gap and the LDM's advantage.

## In-loop prototype (validated live) — `experiments/ldm_controller/sitecustomize.py`
The LDM now runs **inside the serving loop**, not just offline replay. Two monkey-patches on
`Scheduler` (no vLLM source edits, env-gated by `VLLM_LDM_OUT`):

- `make_spec_decoding_stats` — fires once per request per decode step with that request's `K` and
  `acc`; updates its sliding-window acceptance history and computes k\*.
- `update_from_output` — logs one line/step: `{t, n_active, decisions:[{req, K, k*, acc_last}]}`.

Because vLLM processes outputs *before* storing next-step drafts (engine/core.py:415→520), a K update
at step t takes effect at t+1 — genuinely causal. **Validated on a live smoke workload** (repetitive +
diverse prompts, ngram K=8): the controller emits per-request decisions every step and k\* tracks
observed acceptance — high-acceptance steps → k\*=4–6, low-acceptance → k\*=1. The decision function is
unit-verified (a request consistently accepting 4/8 drafts → k\*=4). This confirms the LDM's decision
signal (live per-request acceptance) is available and usable in real time with zero vLLM changes.

**What this prototype does:** with `VLLM_LDM_ENFORCE=1`, a third monkey-patch on
`Scheduler.update_draft_token_ids` truncates each request's stored drafts to its current k\*, so the
worker verifies a **different K per request**. Feasibility rests on a code-audit finding: vLLM's
`SpecDecodeMetadata` builder (gpu_model_runner.py:2586) already handles non-uniform per-request draft
counts (its docstring shows `num_draft_tokens: [3, 0, 2, 0, 1]`) and FlashAttn uses variable query
lengths — so no worker/CUDA-graph surgery is needed; only the scheduler-side truncation. Variable-K
decode is non-uniform → requires `enforce_eager` (CUDA graphs need uniform decode).

**Lesson learned (cold-start collapse):** a *pessimistic* cold start (k\*=1 before any observations)
is **self-fulfilling**: the request only ever drafts 1 token, so the LDM never observes acceptance at
position ≥2, its estimate stays `J[2..]=0`, and it is stuck at k\*=1 forever — the first end-to-end A/B
collapsed exactly this way (all 2043 decisions k\*=1; LDM lost to both fixed baselines). The fix is an
**optimistic cold start** (draft at KMAX until observations exist, then pull K down only when
genuinely poor); verified the policy then differentiates high→6 / medium→3 / low→1. This is a general
pitfall for any online per-request controller: the decision must remain *observable* — a policy that
restricts its own information channel to match its current belief will not converge to the optimum.

**End-to-end A/B** (`experiments/ldm_ab.sh`, sustained concurrent mixed workload, eager mode):
LDM(K=8+enforce) vs fixed-K=4 vs fixed-K=8 — results in `docs/phase3-results.md` §A/B once the run lands.

## Reproduce
```
# measured cost model from Phase-1 random K-sweep + ShareGPT per-request profiles
.venv/bin/python experiments/ldm_eval.py \
  --phase1 results/K1_r6 results/K2_r6 results/K4_r6 results/K8_r6 \
  --per-request results/P2s_K8_r6/oracle_per_request.jsonl --kmax 8 --window 8
```

## Next (Phase 3 → prototype)
- Implement the LDM *in the loop*: use `spec_hook`'s live per-request acceptance to set each
  request's K via the audit's injection points (`request.spec_len`, scheduler.py:521-536), and
  measure **end-to-end** throughput vs fixed-K — closing the replay→live gap.
- Extend the decision to admission + prefill chunk size (the full LDM objective) under an SLO.
