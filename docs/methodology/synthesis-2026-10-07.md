# Research Synthesis — 2026-10-07 (after live A/B v1)

Per the research charter §16. State of record: this file + `PROGRESS.md` + `docs/phase*-results.md`.

## What is now experimentally established
1. **Fixed-K is not uniformly optimal** (Phase 1, 32 runs, two workloads): random templates → K=8
   best at all loads; ShareGPT → interior optimum K=4 (K=4/K=8 crossover). The variable that moves
   the optimum is **draft acceptance rate**, not load.
2. **Per-request profitable-K heterogeneity is real** (Phase 2, per-request hook): ShareGPT
   natural-K spans 1.18–8.82, CV≈0.51 — requests in the same batch want K from ~1 to ~8.
3. **Verification cost is sub-linear in K** (Phase 1 + Phase 3 measured c(k)): vLLM verifies all K
   drafts in ~one target forward pass; c(8)≈3.4×c(1), not 8×.
4. **A causal online controller recovers most of the oracle gap IN REPLAY** (Phase 3,
   `ldm_eval.py`): measured c(k) + sliding-window acceptance estimator → 83–89% of the modeled
   oracle gap (replay objective), converging to the oracle as the window grows.
5. **Variable-K execution is feasible with zero vLLM source changes** (Phase 3 code audit + live
   validation): `SpecDecodeMetadata` already handles non-uniform per-request draft counts; only a
   scheduler-side truncation patch is needed; requires eager mode.

## What was falsified
- **F1: "pessimistic cold start is fine for online K control."** FALSE — self-fulfilling collapse
  (k*=1 forever, A/B v0). Fixed with optimistic cold start. General lesson: an online controller
  must keep its information channel open or it cannot converge.
- **F2 (partial): "the replay's ~+10% transfers to live end-to-end throughput."** NOT SUPPORTED by
  A/B v1 (LDM −1.0% vs best fixed-K, single run). Not yet falsified either — the Priority-A grid
  with 3 trials is the deciding experiment.

## Invalid experiments (preserved, not discarded)
- **v1 K-sweep** (Phase 1): invalid steady-state window definition → redone as v2.
- **A/B v0**: client sent requests serially (31 in 150s, single-stream 51 tok/s) → INVALID, fixed
  with a fixed worker pool.
- **A/B v0b (cold-start collapse run)**: LDM arm invalid (policy bug, all decisions k*=1) → the
  *measurement* is valid evidence of the collapse; the comparison is not.

## Strongest current paper claim (defensible today)
> "Per-request profitable speculation depth is highly heterogeneous in real workloads (CV≈0.51),
> verification cost is sub-linear in K, and a causal online controller that estimates each request's
> acceptance profile from observed outcomes recovers ~85% of the dynamic-K oracle gap in replay —
> while we identify the concrete systems-level reasons (batch-level verification coupling, spiky
> non-stationary acceptance) why this benefit does not automatically appear in end-to-end live
> serving, and the conditions under which it should."

This is an honest *explanation + partial mechanism* claim. The positive end-to-end claim awaits the
Priority-A grid.

## Weakest / still-unproven claim
- Live end-to-end benefit of adaptive-K (A/B v1 INCONCLUSIVE; grid pending).
- That the phenomenon generalizes beyond ngram drafting (Priority C — EAGLE/MTP reproduction).

## Strongest alternative explanation for current results
**ngram artifact**: ngram acceptance is spiky and context-repetition-driven (lag-1 autocorrelation
≈0); a model-based speculator (EAGLE) with smoother, more stationary per-request acceptance may
show both a larger oracle gap AND a cleaner adaptive-K win — or the opposite. The entire positive
story could be drafter-specific. Priority C is designed to settle this.

## Closest competitors / novelty risk
- **AdaEDL** (adaptive speculation length), **BanditSpec**, **SpecDec++**, **DSDE/DSpark/Nightjar**:
  several adapt K or draft-side behavior. Our potential differentiation: (a) *verification-side*
  per-request K in a continuous-batching engine with measured cost structure, (b) the
  replay-vs-live gap analysis as a systems contribution, (c) speculation-aware *scheduling*
  (chunked prefill / token budget interaction) which most prior work does not touch. **Risk:** if
  adaptive-K live benefit is ~0, the contribution shrinks to "heterogeneity exists + explanation of
  why exploiting it is hard" — still publishable as an analysis paper but weaker. The scheduler
  direction (charter §6) is where a stronger systems contribution could come from.

## Biggest novelty risk
Claiming "dynamic K in vLLM" without the scheduling interaction: prior adaptive-K work exists; the
novelty must come from the **speculation × continuous-batching/chunked-prefill interaction** or the
**SLO-goodput vs throughput-optimal K** distinction.

## Highest-information experiment next
1. **(running) Priority-A grid**: AR/K1/K2/K4/K8/LDM × 3 trials — decides SUPPORT/FALSIFY for H1
   with variability, and gives TTFT/TPOT tails (SLO-relevant).
2. **E1 batch-coupling sweep** (script ready): C∈{1,4,16,32} × K∈{2,8} — if K=8's advantage grows
   with batch size, the replay-vs-live ranking flip is explained and the oracle model must be
   batch-aware. This also predicts *where* adaptive-K should win (low concurrency).
3. **Priority C**: EAGLE feasibility check on Qwen2.5-7B (leptonai checkpoint was incompatible;
   re-check current registry / alternatives) to reproduce heterogeneity with a model-based drafter.

## If submitted today, a skeptical reviewer would first ask
1. "Your adaptive controller is within 1% of the best fixed K — what is the actual contribution?"
   (Answer pending grid + E1; if benefit ≈0, pivot the claim to the explanation/scheduling direction.)
2. "ngram drafting is a toy drafter — does any of this hold with EAGLE?" (Priority C.)
3. "Your cost model c(k) was measured on one workload and applied to another — show it's invariant."
   (E2 isolation: re-derive c'(k) per workload, recompute the oracle.)
