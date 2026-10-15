# DSpark baseline audit — synthesis (2026-10-09)

Answers to the seven questions in the research addendum (§8), based on primary-source review, the isolated
native validation (Gates A–E), and the proxy-validity audit. Evidence: `docs/dspark-native-audit.md`,
`docs/dspark-native-results.md`, `docs/dspark-proxy-validity.md`.

## Q1 — Why did we originally use a proxy instead of native DSpark?

Because at design time (Phase 4.5, ~2026-10-07) our pinned runtime was **vLLM 0.19.1 (released 2026-04-18)**,
which predates the first DSpark-capable release **v0.25.0 (2026-07-11)** by three months — and which we could
not upgrade in place because every DSpark-capable release (≥0.25) ships cu13-only wheels while our driver caps
at CUDA 12.9 (verified: import fails with `libcudart.so.13`). The proxy was therefore the only way to test a
DSpark-style *decision rule* on our stack at that time. It was a legitimate, documented choice — but the
related-work claim "no vLLM DSpark integration exists" (true for 0.19.1) became outdated upstream on
2026-08-12 (PR #47808) and is now corrected with a date in `docs/related-work-dspark.md`.

## Q2 — Which original assumptions were correct, outdated, or unsupported?

| Assumption | Verdict | Evidence |
|---|---|---|
| Batch-level argmax over expected tokens / step-cost is the right decision structure | **CORRECT** | Upstream `adaptive_verification.py` uses the same form: `(N + cumsum(survival)) / cost`. |
| Prefill tokens must be priced in the verification cost | **CORRECT** (we were wrong to omit them) | Upstream indexes the verify cost table by `non_draft_tokens + B`; our Phase 4.5 proxy omitted prefill entirely (F4). |
| Draft and verification costs should be modeled separately | **CORRECT** | Upstream keeps two tables (draft by #requests, verify by #tokens), profiled from dummy CUDA-graph steps. |
| "No vLLM DSpark integration exists" | **OUTDATED** | True for our 0.19.1 pin; false upstream since PR #47808 (2026-08-12). Dated correction added. |
| "DSpark ignores all prefill-related work" | **UNSUPPORTED / FALSE** | Upstream explicitly prices prefill in the budget rule. |
| "A static cost profile means the decision uses no live state" | **UNSUPPORTED** | Upstream combines startup-profiled tables with *live* confidence-head outputs each step; profiling ≠ stateless. |
| "Batch-level decisions / load-aware K are novel by themselves" | **NO LONGER TRUE as a claim** | Shipped upstream since v0.28.0 (adaptive verification). Our differentiator must be live serving-state inputs + cheap-drafter SD-off economics, tested on a matched runtime. |

## Q3 — Which native capabilities were actually executed on our hardware?

- **Executed:** fixed-K=7 DSpark end-to-end — real `Qwen3DSparkModel` drafter (5-layer semi-AR backbone +
  Markov head) loading and drafting natively on an RTX 3090 (SM86), with its own CUDA-graph capture, under
  concurrent heterogeneous serving (Gates A, B, D, E all PASS). Build: vLLM v0.25.0 from source against CUDA
  12.6 (cu126), isolated worktree/venv; research env untouched.
- **NOT executed:** adaptive verification — doubly blocked: (a) v0.25.0 explicitly does not wire the confidence
  head into inference ("not wired into inference yet; skip its weights", `qwen3_dspark.py:173`); (b) the
  adaptive path (v0.28+) needs cu13 wheels AND varlen decode CUDA graphs (FA3/SM90 or FlashInfer
  trtllm-gen/SM100), both unavailable on our driver/hardware. This is a definitive, source-verified blocker,
  not an environment failure.

## Q4 — Is the old proxy comparison valid, limited, or affected by a bug?

**Affected by bugs (F1 + F4, confirmed).** The Phase 4.5 "DSpark-rule" arm's decision map covered only ~6–7%
of running requests; the rest ran at the KMAX=8 default, and decided K=0 values dropped out and returned to
KMAX next step — so the pre-registered all-K=0 SD-off fallback was never effectively enforced. The SPS(B) cost
term was evaluated at B≈25 instead of the true batch ~96–130. Therefore:
- "DSpark-rule is the worst arm (−24.6%)" is **invalid as a statement about DSpark's decision rule**; it stands
  only as "our buggy proxy lost".
- Phase 4.5 interpretation #1 (the rule can't learn load-driven K) is **unsupported**; #2 (a live-load signal
  is needed) is **weakened/unproven** by that arm.
- The AR / fixed-K arms and the Phase 4.6/5.1 controllers are unaffected (they already had the persistence +
  true-batch fixes). A corrected rerun (P4.5R) is pre-registered in `docs/dspark-proxy-validity.md` §6.

## Q5 — What do the native baseline and our proposed mechanism each already do?

**Native fixed-K DSpark (runnable on our HW):** a learned drafter proposes 7 tokens; vLLM verifies all of them
every step at a *fixed* K regardless of load, acceptance, or prefill pressure. Measured Gate E: +61.5% vs AR at
C=8, +12.2% at C=32, **−4.5% at C=64** — it already captures the low-load win but pays full verification cost at
saturation (TPOT 8.9→30.2 ms; TTFT 135→418 ms).

**Native adaptive verification (upstream, not runnable on our HW):** same drafter, plus a per-step batch budget
`argmax_k (N + cumsum(survival)) / cost` using the trained confidence head and profiled draft/verify cost tables,
with prefill priced in — i.e., it already does *batch-level allocation that prices the externality of drafts on
the rest of the batch*, with live per-position confidence.

**Our proposed mechanism (LDM family):** decides K (and SD on/off) from **live serving state** — measured batch
occupancy B, steps/s trend, prefill mix, queue/KV pressure — rather than only drafter-side confidence + a static
profile. What it adds *beyond both* native paths: (a) inputs the native budget rule does not consume (queue depth,
KV utilization, prefill intensity, measured saturation regime), and (b) the SD-off/draft-pass economics for cheap
self-drafters, which DSpark's always-on learned drafter does not model.

## Q6 — What measured problem remains after the strongest feasible baseline?

After native fixed-K=7 DSpark (the strongest *runnable* native baseline on this hardware), the remaining
measured problems are:
1. **The saturation loss of fixed K**: −4.5% vs AR at C=64 on Qwen3-4B (TPOT +78%, TTFT +84%). A controller that
   commits to smaller K or SD-off at/above the crossover has a measured headroom of ~4.5 pts here — real but
   smaller than on Qwen2.5-7B/ngram, because the 4B drafter is cheaper and its acceptance (0.25) already limits
   the benefit.
2. **Prefill-heavy mixes**: Gate E used ShareGPT (~50-tok prompts). Phase 4.3 showed long-prefill workloads flip
   the K ranking at *lower* concurrency; fixed-K DSpark has no prefill-awareness in its (fixed) verification
   length, so the gap should widen there — not yet measured on the native stack.
3. **What adaptive verification does not cover even where it runs**: cheap-drafter SD-off economics and
   prefill-chunk co-allocation (the upstream rule allocates a *verification budget*; it does not disable the
   drafter or jointly choose chunk size).

## Q7 — What is the smallest experiment that could falsify our remaining claim?

On the matched Qwen3-4B / dspark_qwen3_4b_block7 / v0.25.0-cu126 runtime (Gate E harness, 3 trials/cell):
a live-serving-state controller that (i) measures B and steps/s per step, (ii) picks K ∈ {0..7} per batch with
the externality-priced argmax (Phase 5 rule), committing the whole batch to SD-off when it wins — must beat
**native fixed-K DSpark by more than trial noise at C=64** (and not lose the C=8 gain beyond noise). Falsified if
it lands within noise of fixed-K=7 at C=64: then live serving state adds nothing over a static K on this stack,
and the contribution narrows to the prefill-mix and SD-off-economics claims (tested separately with SPEED-Bench-2k
cells). This is the next pre-registration (Track B), after P4.5R (Track A) is scheduled.

---

**Bottom line:** the audit's goal — a valid baseline and a defensible contribution, not a negative DSpark result
— is met: we now have (a) a *native* fixed-K DSpark serving baseline measured on our hardware, (b) a precise,
source-verified statement of what adaptive verification is and why it cannot run here, (c) an honest accounting
of the proxy's bugs and what its numbers do/don't show, and (d) a narrowed contribution claim with a concrete
smallest falsification experiment.
