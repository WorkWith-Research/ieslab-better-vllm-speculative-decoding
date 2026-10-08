# DSpark — deep audit & comparison (P5)

**Purpose:** the supervisor asked whether DSpark is used/compared, and flagged that "DSpark changes K by GPU
load." This document pins down exactly what DSpark does, how load enters its decision, what is runnable in
our environment, and what remains unsolved after it. Sources: arXiv:2607.05147 (v1) full text (read 2026-10-06,
see `docs/paper-briefs.md` §3), DeepSpec repo, HF checkpoints — verified 2026-10-07.

## 1. What DSpark actually adapts

| Aspect | Detail |
|---|---|
| Decision variable | **Scheduled verification length** `l_r ∈ {0,…,γ}` per request |
| Draft block size γ | **Fixed** (7 offline; 5 in production "DSpark-5"). The drafter always runs the full γ-block; only *verification* is pruned. |
| Granularity | Per-request, per decoding step — but decided by a **batch-level global greedy** over all requests' candidate extensions |
| Confidence signal | Trained **confidence head** `c_k = σ(wᵀ[h_k; W_1[x_{k−1}]])` per draft position (prefix-survival prob), calibrated with Sequential Temperature Scaling → ECE ~1% |
| "Load" input | A **static, profiled throughput curve SPS(B)** (steps/sec vs verification batch size B), profiled once at init. Load enters *only* through the current total verification token count `B = Σ_r (1+l_r)` — i.e., it is a model of GPU compute load as a function of batch occupancy, **not** real-time queue depth / KV utilization / measured GPU state |
| Objective | Maximize expected system throughput `Θ = τ·SPS(B)` where `τ = Σ_r(1+Σ_{j≤l_r} a_{r,j})` (expected accepted tokens), solved by global greedy sort on prefix-survival `a_{r,j}` with early stop at Θ non-increase |
| Causality mechanism | In production the truncation capacity is set from confidence outputs **two steps earlier** (async gap = causal barrier) to stay compatible with continuous CUDA-graph replay + Zero-Overhead Scheduling |
| Engine | **DeepSeek-V4 proprietary engine only** (ZOS + CUDA graphs; flattened variable-length execution via marker-tensor sparse attention). No vLLM integration released. ⚠️ *Superseded 2026-10-09: upstream vLLM has had a native DSpark drafter + adaptive verification since PR #47808 (2026-08-12); see `docs/dspark-native-audit.md`.* |

### Correction to the common shorthand
"DSpark changes K by GPU load" is imprecise in two ways:
1. It adapts **verification length**, not draft length — draft-side cost (the parallel γ-block) is always paid,
   even for requests whose verification is pruned to l_r=0. This is their stated limitation (no difficulty-aware
   drafter early exit).
2. "GPU load" is a **profiled static curve SPS(B)**, not live GPU telemetry. Real-time serving state (queue, KV,
   prefill mix) does not enter the decision. Concurrency matters only insofar as it changes B and hence the SPS
   lookup — which is exactly the regime our P1/P2 experiments measure directly on real hardware.

## 2. Structured comparison (charter §10 table)

| Dimension | DSpark | Our current prototype | Candidate scheduler work |
|---|---|---|---|
| Decision variable | verification length l_r (draft γ fixed) | speculation length K (draft+verify, ngram self-drafting) | per-request K **and** prefill chunk / admission / SD on-off |
| Granularity | per-request, per step (global greedy) | per-request, per decode step (local EMA window W=8) | per-request + batch-level budget |
| Acceptance/confidence signal | trained confidence head + STS calibration (ECE~1%) | observed acceptance history (EMA over last W steps) | measured live acceptance (per-request hook) |
| Serving-state signal | profiled SPS(B) curve only | none (acceptance-only policy) | queue depth, KV util, batched tokens, GPU state, prefill mix |
| Batch awareness | yes — B = Σ(1+l_r) couples all requests in Θ | no (per-request local decision) | yes (token-budget co-allocation) |
| GPU/load awareness | indirect via SPS(B) | no | direct (measured saturation regimes) |
| Queue awareness | no | no | yes (admission) |
| KV awareness | no | no | yes |
| Prefill awareness | no (decode-only decision) | no | yes (chunk-size coupling — H-S2/H-S3) |
| SLO awareness | SLA used in *evaluation* only, not objective | no | goodput objective candidate |
| Objective | expected system throughput τ·SPS(B) | none explicit (greedy local value) | throughput or SLO-goodput (to be determined by §27 experiments) |
| CUDA-graph behavior | compatible via 2-step-async capacity + flattened execution | requires **eager** (variable K breaks uniform graph shapes); K-bucketing untested | K bucketing / padded-to-K_max design (code audit Option B) |
| Supported drafter | DSpark semi-AR backbone (trained), any confidence head | ngram self-drafting; draft_model (Qwen2.5-0.5B) prepared | drafter-agnostic |
| Evaluation workload | live production traffic (DeepSeek-V4); offline benchmarks for draft quality | mixed synthetic workload, ShareGPT; SPEED-Bench pending | SPEED-Bench + heterogeneous ISL/OSL |
| Evaluation concurrency | <200 (Flash), <150 (Pro) concurrent users | C=16 fixed so far; P1/P2 sweep 1→64 | full saturation range |
| Production integration | DeepSeek-V4 engine (proprietary) | vLLM 0.19.1 monkey-patch prototype | vLLM fork (patch surface in `docs/vllm-code-audit.md`) |

## 3. What DSpark already solves vs what remains

**Already solved by DSpark (do not claim as novelty):**
- Per-request verification-length adaptation from a calibrated per-position confidence signal.
- Batch-coupled objective (Θ = τ·SPS(B)) — i.e., the *idea* that one request's K has batch-level opportunity cost.
- Causality-safe dynamic length under CUDA graphs (2-step async).

**NOT solved by DSpark (our candidate contribution space):**
1. **Draft-side cost is fixed.** DSpark pays the full γ draft block always. For ngram/self-drafting or cheap
   drafters, adapting *draft* length K saves real work; for expensive model drafters it does not — so the
   economics of adaptive-K are drafter-dependent (testable with our Qwen2.5-0.5B draft_model arm).
2. **No real-time serving state.** SPS(B) is profiled offline. Queue depth, KV pressure, prefill/decode mix, and
   measured saturation regime do not enter the decision. If P1/P2 show the optimal K depends on *measured* load
   beyond what B captures (e.g., prefill interference), that gap is real.
3. **No prefill/chunked-prefill interaction.** DSpark's scheduler is decode-only. Whether high-K decode batches
   starve prefills under the batched-token budget (H-S2/H-S3) is unaddressed by any adaptive-SD paper we audited.
4. **Not runnable in our environment** — see below. So a *direct* DSpark baseline is impossible; the closest
   feasible proxy is implementing its **decision rule** (greedy over prefix-survival with an SPS(B) table
   profiled on our hardware) on top of our vLLM prototype, using observed acceptance as the confidence signal.
   This is a legitimate algorithmic baseline and is pre-registered below.

## 4. Runnability in our environment (documented per charter §10)

> **⚠️ Dated correction (2026-10-09, see `docs/dspark-native-audit.md`):** the table below is the
> original assessment made 2026-10-07 against our pinned vLLM 0.19.1 (released 2026-04-18). It is
> **superseded** for current upstream: native DSpark landed in vLLM via PR #47808 on **2026-08-12**
> (drafter from v0.25.0, adaptive verification from v0.28.0). The claim "No vLLM integration released" /
> "no `DSpark*` arch in the registry" was true only for vLLM 0.19.1 and is **false for upstream since
> Aug 2026**. On our SM86 hardware, adaptive verification is a definitive startup-time blocker (requires
> varlen decode CUDA graphs: FA3/SM90 or FlashInfer trtllm-gen/SM100); fixed-K DSpark drafter has no arch
> gate but needs a cu12-compatible vLLM build (our driver caps at CUDA 12.9; all DSpark-capable releases
> ship cu13-only wheels). See `docs/dspark-native-audit.md` §2 for the source-level evidence and the
> isolated-build attempt log.

| Component | Available? | Why / why not |
|---|---|---|
| DSpark drafter checkpoints | Partially | HF has DSpark checkpoints (DeepSpec repo, e.g. for Qwen3-4B/8B), but they are the **semi-AR backbone** architecture from their DeepSpec framework — vLLM 0.19.1's registry does not include it (verified: no `DSpark*` arch in `vllm/model_executor/models`). |
| DSpark scheduler | Not at all | The confidence-scheduled verification logic lives in the **DeepSeek-V4 engine**, which is proprietary and not open-sourced. No vLLM plugin exists. |
| Our vLLM 0.19.1 adaptive verification | No native support | vLLM has fixed `num_speculative_tokens` (code audit item 3: K captured at init, scalar everywhere). Our monkey-patch prototype provides the variable-K substrate DSpark's scheduler would need. |

**Conclusion:** run a **decision-rule reimplementation** as baseline (greedy top-K over prefix-survival with a
hardware-profiled SPS(B) table), NOT the full system. This is scientifically honest: it isolates *their*
algorithmic idea from *their* engine advantages.

## 5. Pre-registration — DSpark-rule baseline (to run after P2)

- **Question:** does DSpark's decision rule (batch-greedy over prefix-survival with profiled SPS(B)) beat our
  acceptance-only LDM and fixed-K on the same workload/load?
- **Setup:** profile SPS(B) on our hardware (B = total verification tokens, measured steps/sec for B ∈ {16..512}
  at fixed concurrency); implement greedy l_r selection in the existing `ldm_controller` hook (replace local EMA
  decision with batch-greedy using last-step observed acceptance as a_{r,j} proxy); compare vs LDM, best fixed-K,
  AR at C_med and C_sat from P1. Eager mode for all arms. 3 trials per arm.
- **Discriminating outcome:** if DSpark-rule ≥ LDM at both loads → our local policy is suboptimal; the
  contribution shifts to *serving-state inputs* (queue/KV/prefill) that SPS(B) lacks. If DSpark-rule ≈ fixed-K →
  batch-greedy adds nothing over per-request adaptation for ngram, and load-awareness must come from elsewhere.
- **Falsification of "DSpark already solves our problem":** any measured regime where the DSpark-rule baseline
  is outperformed by a policy using real-time serving state (queue/KV/prefill) that it does not use.
