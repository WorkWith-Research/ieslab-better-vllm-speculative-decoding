# Phase 4.6 — Live-load-aware adaptive speculation (result: H-4.6 FALSIFIED as specified)

**Hypothesis (pre-registered 2026-10-08, PROGRESS.md):** a minimal CAUSAL controller combining per-request
acceptance with a LIVE load-regime signal can (i) use large K when speculation is cheap and (ii) commit to SD-off
(K=0) when the batch is compute-saturated — without being given the concurrency label C. Controller `ldm_load`,
final rule (rev #4, unified SPS curve):

    k*_r = argmax_{k∈{0..8}} [(N−1)(1+a_bar) + 1 + Σ_{l≤k} J_r[l]] · SPS(B_live + (k − d_r))

with N, B_live, d_r all MEASURED in-loop; no C anywhere. Pre-registered bounds: **C=96 ldm_load ≥ AR − noise
(≈0.3%) AND beats DSpark-rule; C=8 ldm_load ≥ +5% vs AR.** FALSIFY if it fails either bound (no retuning, charter §13).

**Setup:** Qwen2.5-7B-Instruct, 1× RTX 3090, vLLM 0.19.1 eager, ngram self-draft K≤8, mixed workload (Phase 4.2),
C ∈ {8, 32, 96} × 3 trials + force-0 diagnostics (VLLM_LDMLOAD_FORCE0=1). All arms verified same spec config
(`{'method': 'ngram', 'num_speculative_tokens': 8}`) and same harness.

## Result — server-side generation rate (steady window, t ≥ 20s)

| arm | C=8 tok/s (vs AR) | C=32 tok/s (vs AR) | C=96 tok/s (vs AR) |
|---|---|---|---|
| AR (spec disabled) | 387.0 | 1460.0 | 2557.2 |
| K4 (fixed) | 414 (+7.0%) | 1401 (−4.0%) | 1949 (−23.8%) |
| K8 (fixed, best fixed) | 430 (+11.1%) | 1437 (−1.6%) | 1937 (−24.3%) |
| LDM (acceptance-only, P4.5) | — | 1398 (−4.2%) | 1953 (−23.6%) |
| DSpark-rule (P4.5) | — | 1416 (−3.0%) | 1858 (−27.3%) |
| **ldm_load (rev-#4c)** | **417.0 (+7.8%)** | **1304.2 (−10.7%)** | **2235.2 (−12.6%)** |
| ldm_load force-0 (SD-off floor) | 354.4 (−8.4%) | 1336.9 (−8.5%) | 2437.6 (−4.7%) |

AR trial spreads (noise): C=8: 2.4, C=32: 7.9, C=96: 16.1 tok/s. ldm_load trial spread ≤ 28 tok/s at C=96
(2218.9 / 2246.8 / 2238.9) — the failure is not jitter.

**Classification (exactly as pre-registered):**
- **C=8: PASS** — +7.8% vs AR ≥ +5% bound; recovers ~70% of fixed-K8's +11.1% without ever knowing C. meanK*=4.1,
  frac(K=0)≈11%. The load signal correctly leaves speculation ON where it is cheap.
- **C=96: FAIL** — −12.6% vs AR; the bound was AR − noise ≈ 2551 (−0.3%). It *beats* DSpark-rule by +20.2% (the
  second clause passes), but the first clause fails → **H-4.6 is FALSIFIED as specified.** No post-hoc retuning.

## What the controller actually did at saturation (decision logs)

| C | mean K* | frac(K=0) | k* distribution | draft tok/s |
|---|---|---|---|---|
| 8 | 4.12 | 11.5% | near-uniform up to 8 | ~93 |
| 32 | 1.49 | 81.6% | **bimodal: 0 (81%) / 8 (18%)** | ~43 |
| 96 | 1.56 | 80.9% | **bimodal: 0 (80%) / 8 (20%)** | ~69 |

Enforcement verified from the logs: of 270,192 k*=0 decisions at C=96, only 0.5% show any scheduled draft
(the rest are first-step requests whose K is applied before their decision is logged). The signal is not stuck —
it *recognizes* saturation (80% of requests at K=0) and picks the right regime at C=8. What it cannot do is
**commit**: a ~20% minority stays permanently at K=8, and the batch oscillates meanK* 0↔5.4 over ~10s cycles
(sampled from the decision log). Two compounding causes:

1. **Missing externality (the core finding).** Each request's argmax prices its OWN accepted tokens against the
   batch's SPS, but not the slowdown its drafts impose on the other N−1 requests' verification — a negative
   externality that grows with B. At saturation the per-request greedy therefore over-admits speculation: every
   request independently reasons "my acceptance is high, K=8 is worth it," and the batch collectively pays for it.
   The force-0 diagnostic proves the optimum here IS full SD-off (2437.6 > 2235.2), yet no per-request decision
   rule reaches it. This is a property of per-request greedy allocation under shared-batch compute saturation, not
   of this particular value function — DSpark-rule fails the same way (broad K distribution, −27.3%) and
   acceptance-only LDM worse (−23.6%).
2. **The spec path has a fixed overhead no K-only controller can remove.** Force-0 (k*=0 for all requests, spec
   decode still ENABLED) sits at 2437.6 vs AR 2557.2 = **−4.7% at C=96** (−8.4/−8.5% at C=8/32). Even with zero
   drafts, running the spec-decode code path costs measurable throughput. Hence "ldm_load ≥ AR − noise" was
   structurally unreachable for any controller restricted to K∈{0..8} — reaching AR requires disabling the path
   itself, a scheduler-level action (Phase 5).

## Interpretation

1. **The live load signal WORKS; the per-request greedy ALLOCATION does not.** ldm_load is the best adaptive arm
   at every measured C: +7.8% at C=8 (beats fixed K4, near fixed K8), and at saturation it beats DSpark-rule by
   20 points and fixed-K8 by 12 — it recovers most of the gap that Phase 4.5 declared "left on the table." The
   regime information in live serving state is real and usable. What fails is the mechanism that turns it into a
   batch-optimal allocation: per-request myopic argmax under a shared compute budget.
2. **The remaining gap to AR at saturation has two distinct components** (now separated by measurement):
   ~8 points of *allocation* loss (per-request greedy vs the SD-off optimum) and ~4.7 points of *fixed spec-path
   overhead*. The first is a scheduling/allocation problem; the second requires scheduler-level control of the spec
   path itself. Both point to the same next step: **Phase 5 — speculation-aware scheduling** (batch-level K
   allocation with externality pricing, and/or dynamic enable/disable of the spec path), not another per-request
   controller revision.
3. **H-4.6 stands falsified as pre-registered.** The specified controller does not meet its C=96 bound; we report
   that, with the mechanism above explaining why, rather than retuning to win (charter §13).

## Method note — client vs server throughput metric

The client's `steady_throughput_tok_s` (completions-in-window × 256 tokens) is **open-loop biased**: workers still
in-flight at window end are excluded, and the exclusion scales with each arm's request latency. Measured effect:
AR/force-0/smokes all "coincidentally" read 2457.6 client-side while their true server rates differ (2557 / 2438 /
~2350). All cross-arm comparisons in this document use the vLLM `generation_tokens` counter rate (steady window
t ≥ 20s); `experiments/analyze_p46_server.py` reproduces the table. Phase 4.2/4.5 client-based tables remain valid
*within* their own arm sets but should be re-read with this bias in mind.

## Caveats
- ngram self-drafting on a repetitive mixed workload: acceptance is high and fairly uniform, so per-request
  acceptance heterogeneity is muted (it is large on SPEED-Bench, Phase 4.3). The saturation/externality effect
  dominates by design; the allocation failure should only be stronger with more heterogeneous acceptance.
- One static SPS(B) curve fitted to 19 points; a denser profile could shift ldm_load's exact C=96 number but not
  the classification (it is 12 points below its bound, and force-0 shows the attainable ceiling).
- Deterministic workload → trial variance is jitter-only (≤28 tok/s at C=96), far below the 322 tok/s failure gap.
