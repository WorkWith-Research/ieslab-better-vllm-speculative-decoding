# Related-Work Audit — Adaptive Speculative Decoding (2026-10-07)

Structured comparison per charter §9. Fields: decision variable / granularity / input signal /
system state used / objective / draft-side vs verification-side / batching-aware / queue-aware /
KV-aware / prefill-aware / SLO-aware / CUDA-graph compat / environment / eval scale.

## AdaEDL (arXiv 2410.18351)
- **Decision variable:** draft length, via early *stopping* of drafting.
- **Granularity:** per-token within a single draft sequence (token-level).
- **Input signal:** entropy of the drafter's own logits (lower bound on acceptance prob).
- **System state used:** none — drafter-local only.
- **Objective:** minimize wasted draft compute (per-sequence latency).
- **Side:** draft-side (stops the drafter early; verification still gets whatever was drafted).
- **Batching/queue/KV/prefill/SLO-aware:** NO to all. Single-request view.
- **CUDA-graph:** compatible (draft-side, no scheduler changes).
- **Env/scale:** offline single-stream benchmarks; not a serving system.
- **Overlap with us:** adaptive draft length, training-free. **Difference:** AdaEDL never looks at
  serving state or batching; it cannot express "this request should verify K=2 while that one gets
  K=8 in the same batch", and has no cost model of verification.

## BanditSpec (arXiv 2505.15141, ICML'25)
- **Decision variable:** SD hyperparameters (incl. draft length) via multi-armed bandit.
- **Granularity:** step-level, online; context = current generation context.
- **Input signal:** reward from acceptance outcomes of past steps.
- **System state used:** none beyond the sequence's own history.
- **Objective:** per-sequence speedup (acceptance-weighted).
- **Side:** draft-side / config-level.
- **Batching/queue/KV/prefill/SLO-aware:** NO to all (single-stream evaluation).
- **CUDA-graph:** compatible in principle (config changes between steps).
- **Env/scale:** single-stream, multiple models; not continuous batching.
- **Overlap with us:** online bandit over K from acceptance feedback — closest in *mechanism family*
  to our causal LDM replay. **Difference:** no serving/batching context, no measured verification
  cost structure, no per-request scheduling semantics. Our contribution space is the *serving-system*
  layer: per-request K as a scheduler decision with batch-level verification coupling, plus the
  replay-vs-live gap analysis they don't face (no batching in their eval).

## Others (from charter list; to be deepened when relevant)
- **DSDE / DSpark / Nightjar / TETRIS / SpecDec++ / DISCO:** draft-side or token-selection
  adaptations; none identified so far that treat per-request K as a *scheduling* decision coupled
  to chunked prefill / token budgets. (Verify individually before claiming.)

## Novelty statement (current, honest)
Adaptive-K *per se* is not novel (AdaEDL, BanditSpec). Our candidate contributions:
1. **Verification-side per-request K in continuous batching** with a *measured* sub-linear
   verification cost model — prior work adapts the drafter, we adapt what the scheduler verifies.
2. **Replay-vs-live gap analysis**: why per-request oracle gains don't automatically appear
   end-to-end (batch coupling, non-stationary acceptance) — an explanation contribution no
   single-stream adaptive-K paper can make.
3. **Speculation-aware scheduling** (chunked-prefill / token-budget interaction) — the intended
   next step; closest to a genuinely new systems problem.
