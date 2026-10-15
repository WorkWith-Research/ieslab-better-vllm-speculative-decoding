# DSpark native support & runnability audit (2026-10-09)

**Scope:** establish what upstream vLLM's native DSpark integration actually is, which release it
landed in, and whether any of it can execute on our hardware (2× RTX 3090 / SM86, driver 575.57.08,
max CUDA 12.9). This supersedes the runnability table in `docs/related-work-dspark.md` §4, which is
now known to be wrong on its central claim.

**Primary sources (retrieved 2026-10-09):**
- vLLM blog "Adaptive Verification in vLLM: DSpark confidence-scheduled verification", 2026-08-14.
- vLLM source at `main` commit `994b8b7` (cloned to scratch, read-only) and tagged releases v0.24.0–v0.31.0.
- PR #47808 "[Spec Decode] DSpark confidence-scheduled verification", merged 2026-08-12 (merge commit `7f7a32c`).
- HF checkpoints `deepseek-ai/dspark_qwen3_4b_block7` (2.79 GB, public) + `Qwen/Qwen3-4B` (8.06 GB).
- DeepSpec repo (`deepseek-ai/DeepSpec`, authors' training/eval codebase).

## 1. What upstream vLLM actually implements

**Native DSpark is a first-class speculative method in vLLM, not a plugin.** `method="dspark"` in the
speculative config selects it; the drafter is a real learned model (semi-AR backbone + trained
confidence head), and there is an optional **adaptive verification** mode.

| Component | Where (vllm main @ 994b8b7) | What it does |
|---|---|---|
| Drafter model | `model_executor/models/qwen3_dspark.py` (`Qwen3DSparkForCausalLM`) | Loads the DSpark checkpoint; consumes target hidden states at `target_layer_ids`; emits a draft block + per-position confidence. |
| Speculator / sampling | `v1/worker/gpu/spec_decode/dspark/speculator.py` | Sequential Markov-head sampling; `compute_confidence()` → per-(req, position) survival probability. When the trained head is present it **replaces** the empirical acceptance estimator. |
| Adaptive verification | `v1/worker/gpu/spec_decode/adaptive_verification.py` (`AdaptiveVerificationManager`) | Per-step batch budget selection + on-device draft trimming (see §2). Gated by `enable_adaptive_verification`. |
| Config | `config/speculative.py` | `method="dspark"`, `num_speculative_tokens`, `enable_adaptive_verification`, `dspark_draft_topk`, target/draft pair validation. |

### 1a. The budget rule (the part that matters for our contribution)

`AdaptiveVerificationManager.get_num_tokens()` computes, per step:

```
survival[r,j] = cumprod over j of confidence_probs[r, :]      # prefix-survival, causal
scores        = all (r,j) slots with j < scheduled_drafts[r], sorted desc
N             = # requests that will actually SAMPLE this step   # chunked-prefill reqs contribute 0
num_tokens_to_estimated_accepted_tokens[k] = N + cumsum(scores)[:k]
costs[k]      = draft_cost_ms[num_reqs] + verify_cost_ms[T : T+k]   # T = non-draft tokens already scheduled
draft_budget  = argmax_k  num_tokens_to_estimated_accepted_tokens[k] / costs[k]
```

This is **the same objective form we independently arrived at in Phase 5 (H-5.1)** — a batch-level
argmax over expected-output-tokens divided by a profiled step-cost, with the externality carried by the
single cost term that grows with total scheduled tokens. The differences are in the *inputs*, not the
shape:

| Input | Upstream native | Our proxy (Phase 4.5) / LDM (4.6/5.1) |
|---|---|---|
| Confidence | **trained** head, calibrated (blog: ECE ~1%), per-position survival = cumprod | empirical acceptance over last W steps (J_r[j]) |
| Cost curve | profiled at startup from **dummy CUDA-graph steps**, two separate tables (draft by #reqs, verify by #tokens), graph-padded staircase + monotonic enforcement | one SPS(B) curve fit to Phase-4.2/4.6 measured points |
| Prefill in cost | **yes** — `T` = non-draft scheduled tokens indexes the verify table; chunked-prefill requests excluded from N | Phase 4.5 proxy: **no** (B was the decided subset only — F4 bug); Phase 5 LDM: decode-only B_eff |
| Causality | confidence copied to CPU on a side stream, budget reads it **one step stale** (double-buffered) | our decision uses last-step observed acceptance (also causal, one step) |
| Allocation | pick ONE batch budget B, then admit the globally top-B slots by survival rank (contiguous prefix per request), on-device via torch.compile/Triton | Phase 4.5: sequential marginal-gain greedy; Phase 5: uniform K_batch across decode requests |

**Implication for our contribution:** "batch-level token-budget allocation that prices the externality
of a request's drafts on the rest of the batch" is **already implemented and shipped upstream**. That
specific mechanism is no longer novel by itself. What remains open (and what our data actually probes)
is: (a) whether a *lightweight, drafter-agnostic* decision that consumes **live serving state** (queue,
KV, prefill mix, measured saturation regime — not just the profiled SPS(B)) beats native adaptive
verification in regimes where the static profile is stale, and (b) the draft-side cost / SD-off
question for cheap self-drafters, which native DSpark does not address (its drafter always runs the full
block).

## 2. Hardware gates on our box (SM86 / CUDA 12.9)

Two independent gates, both verified from source:

### Gate A — adaptive verification cannot run on SM86 (BLOCKER, definitive)

Adaptive verification trims drafts **on device**, so it replays decode CUDA graphs whose per-request
query lengths vary up to `num_speculative_tokens + 1`. That requires an attention backend reporting
`AttentionCGSupport.ALWAYS` for varlen decode:

- `flash_attn.py`: `_cudagraph_support = ALWAYS if get_flash_attn_version()==3 else UNIFORM_BATCH`.
  FA3 is selected only on SM90 (`device_capability.major==9`). **SM86 → FA2 → UNIFORM_BATCH.**
- `attn_utils.get_varlen_cudagraph_unsupported_backend()`: for a non-ALWAYS backend it needs
  `get_varlen_cudagraph_max_query_len() >= max_query_len`. FlashAttention **does not override** that
  method, so it inherits the base which returns `None` → bound is None → **rejected at startup**
  (`maybe_create_adaptive_verification_manager` raises ValueError).
- The only non-MLA backend with a real varlen-decode bound is FlashInfer
  (`flashinfer.py:1098`), and `_uses_trtllm_gen_varlen_decode` requires the **trtllm-gen** decode
  kernel, which `_get_flashinfer_trtllm_api_decode_kernel` returns only for `is_device_capability_family(100)`
  (Blackwell). SM86 → XQA/other → bound None → rejected.

So on our 3090s, `enable_adaptive_verification=true` is **rejected at server startup** by design; there
is no fallback to a slower path. This matches the blog's own Limitations section (FULL varlen decode
graphs require `AttentionCGSupport.ALWAYS`; otherwise "adaptive verification is rejected at startup").

### Gate B — fixed-K DSpark drafter: arch-agnostic, but needs a cu12-compatible vLLM build

The fixed-K path (`enable_adaptive_verification=false`, the default) has **no SM90+/Blackwell gate** in
`dspark/speculator.py` or `qwen3_dspark.py` (verified: no `device_capability`/`sm90` references). So the
*drafter* itself can run on SM86 — but only if we can get a DSpark-capable vLLM to import on our driver.

**The blocker is the CUDA runtime, not the architecture.** Every DSpark-capable release (v0.25.0 onward)
pins torch 2.11/2.13 and ships **only cu13 wheels** (`libcudart.so.13`); PyPI has a single wheel flavor
per release (no cu12 variant). Our driver caps at CUDA 12.9, so:

- **Attempt 1 (prebuilt wheel, v0.25.0 + torch 2.11.0+cu129):** install succeeds, but `import vllm`
  fails — `vllm/platforms/cuda.py:23 import vllm._C_stable_libtorch` →
  `ImportError: libcudart.so.13: cannot open shared object file`. The compiled extension links CUDA 13.
- **Attempt 2 (source build, CUDA 12.6 toolkit + isolated venv):** in progress. Progression of failures,
  each fixed and re-run:
  1. `GCC >= 11.3 required (found 9.4.0)` — system gcc too old for torch's C++20 headers.
  2. Bypassed with clang, but **clang 10** fails on torch C++20 intrinsics (`mwaitxintrin.h` / `x86intrin.h`).
  3. Installed GCC 13 via micromamba (isolated `/home/junior1/.local/gcc13env`, no host change). C++ files
     then compile, but **nvcc's host compiler was still g++9.4** → `.cu` files fail with `#error You need C++17`.
  4. Set `CMAKE_CUDA_HOST_COMPILER=<gcc13>` — build now compiles past all prior failure points (in progress).

If attempt 2 produces an importable vllm, the next gates are: load `dspark_qwen3_4b_block7` + `Qwen3-4B`,
confirm the **real DSpark checkpoint drafts** (not an ngram fallback), and observe scheduling/correctness.
Adaptive verification remains out of scope on this hardware regardless of build success — it is Gate A.

## 3. Milestone status (kept distinct, per addendum §4) — FINAL (2026-10-09)

| Milestone | Status on our hardware |
|---|---|
| Source support exists (upstream) | **Yes** — PR #47808 merged 2026-08-12; drafter in v0.25.0 (2026-07-11)+, adaptive verification in v0.28.0 (2026-08-26)+. |
| Model loads locally | **Yes** — Qwen3-4B + `dspark_qwen3_4b_block7` on SM86 via cu126 source build (isolated worktree). Gate A+B PASS: `Resolved architecture: Qwen3DSparkModel`, dspark CUDA graphs captured 45/45, real drafts with learned acceptance decay (no ngram fallback). |
| Adaptive verification active locally | **No — definitive blocker.** v0.25.0 explicitly skips the confidence head ("not wired into inference yet"); the adaptive path ships in v0.28+ which needs cu13 (driver caps at 12.9), AND varlen decode CUDA graphs need FA3/SM90 or FlashInfer trtllm-gen/SM100 (SM86 rejected at startup). |
| Serving benchmark completed | **Yes** — Gate E: native fixed-K=7 DSpark vs AR on Qwen3-4B, C∈{8,32,64}: +61.5% / +12.2% / −4.5%. Full table + raw results: `docs/dspark-native-results.md`. |

These are not interchangeable: "source support exists" (true) does not imply "adaptive verification is
active locally" (false here). An authors' offline DeepSpec evaluation could still validate the drafter's
draft quality, but it is **not** a substitute for a native continuous-batching serving baseline.

## 4. What this changes in the project narrative

1. The Phase 4.5 "DSpark-rule" arm was a **proxy**, and (per `docs/dspark-proxy-validity.md`) it had two
   confirmed implementation bugs (F1 decision-set completeness, F4 cost accounting). Its negative result
   cannot be cited as evidence about DSpark's algorithm — and now we know the real system is far more
   capable than the proxy (trained confidence head + separate draft/verify cost tables + prefill-aware
   budgeting).
2. "No vLLM DSpark integration exists" (related-work-dspark.md §4) is **false** as of 2026-08-12 and must
   be corrected with a date. It was defensible only for the pinned vLLM 0.19.1 (released 2026-04-18, which
   genuinely predates DSpark).
3. The novelty claim narrows: batch-level budget allocation that prices draft externality is upstream.
   Our candidate contribution must be stated against **native adaptive verification as the baseline**,
   and its differentiator (live serving-state inputs; cheap-drafter SD-off economics) must be tested on a
   matched target/drafter/runtime — not inferred from the buggy ngram proxy.

## 5. Next steps (smallest falsifiable experiment) — UPDATED after Gate A–E completed (2026-10-09)

Gate A–E were executed on the cu126 source build (`docs/dspark-native-results.md`). The strongest *runnable*
native baseline is fixed-K=7 DSpark, which already wins at low load and loses at C=64 (−4.5% vs AR). The
smallest experiment that could falsify our remaining claim: on this matched Qwen3-4B / dspark runtime, show
that a live-serving-state K decision (including SD-off) beats **native fixed-K DSpark** at/above the crossover
(C≈32–64) by more than trial noise — e.g. by tracking the measured load regime and committing to smaller K or
SD-off where fixed K=7 pays verification cost for a 0.25 acceptance rate. If it cannot, the contribution narrows
to what native DSpark (adaptive, on capable hardware) does not cover: cheap-drafter SD-off economics and
prefill-chunk co-allocation under mixed workloads.
