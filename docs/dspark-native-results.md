# Native DSpark validation — Gate A–E results (2026-10-09)

**Environment (isolated, per addendum §1/§4):** `/home/junior1/_dev/dspark-native-validation` — vLLM **v0.25.0
built from source against CUDA 12.6** (`vllm-0.25.0+cu126`, editable install), torch 2.11.0+cu129, Python 3.12.
Research environment (`.venv` vLLM 0.19.1) untouched. Single GPU used: RTX 3090 #1 (SM86). Port 8300/8301.

**Build notes (attempt log):** PyPI wheel = cu13-only (`libcudart.so.13`, import fails on driver CUDA 12.9) →
source build. Sequential toolchain fixes: (1) system GCC 9.4 < required 11.3; (2) clang 10 cannot compile torch's
C++20 headers; (3) installed GCC 13 via micromamba in user space (`/home/junior1/.local/gcc13env`); (4) nvcc still
used g++9.4 as host compiler → `CMAKE_CUDA_HOST_COMPILER=<gcc13>` fixed it. Build succeeded (~35 min). Post-build
dep repair: torchvision 0.26.0+cu129, torchaudio 2.11.0+cu129, torchcodec 0.16.0+cu129 (PyPI CPU wheels were
ABI-mismatched with the cu129 torch).

**Model pair (smallest plausible released pair, per addendum §4):** target `Qwen/Qwen3-4B` (8.06 GB) + drafter
`deepseek-ai/dspark_qwen3_4b_block7` (2.6 GB, public, non-gated). Checkpoint config: `architectures:
["Qwen3DSparkModel"]`, `block_size: 7`, `num_hidden_layers: 5` (draft), `target_layer_ids: [1,9,17,25,33]`
(= Qwen3-4B's 36 layers), `enable_confidence_head: true`, `confidence_head_with_markov: true`. Authors' DeepSpec
config (`config/dspark/dspark_qwen3_4b.py`) confirms target = `QWEN_3_4B` with identical hyperparameters.
Block size 7 kept as-is (NOT replaced to match our historical KMAX=8).

## Gate A — Configuration: PASS
Server accepts `--speculative-config '{"method": "dspark", "model": "deepseek-ai/dspark_qwen3_4b_block7",
"num_speculative_tokens": 7}'`. Log: `Resolved architecture: Qwen3DSparkModel`; engine config
`SpeculativeConfig(method='dspark', model='deepseek-ai/dspark_qwen3_4b_block7', num_spec_tokens=7)`. No fallback,
no ngram substitution.

## Gate B — Drafter execution: PASS (real DSpark checkpoint drafts; no fallback)
- `Capturing dspark CUDA graphs (FULL): 45/45` — the DSpark speculator's own graph capture ran.
- First request: `spec_decode_num_draft_tokens_total=175`, `num_accepted_tokens_total=38` over 25 draft steps;
  per-position acceptance decays 14→9→9→3→1→1 (a learned drafter's profile, not ngram's near-flat one).
- After the full concurrent test: **98,413 draft tokens / 24,994 accepted ≈ 0.254 acceptance rate** — low because
  ShareGPT-style text is hard for a 5-layer drafter; this is exactly the regime where K-adaptation matters.

## Gate C — Adaptive verification: NOT ACTIVE in v0.25.0 (documented, not a failure)
- `vllm/model_executor/models/qwen3_dspark.py:173`: *"confidence_head is not wired into inference yet; skip its
  weights"* — the trained confidence head's weights are explicitly skipped at load time in v0.25.0.
- No `adaptive`/`enable_adaptive_verification` config exists in v0.25.0's `speculative.py`; the adaptive
  verification manager (`v1/worker/gpu/spec_decode/adaptive_verification.py`) first ships in **v0.28.0**
  (2026-08-26), which requires torch 2.13 / CUDA 13 — unavailable on our driver (max CUDA 12.9).
- Therefore: v0.25.0 DSpark = **fixed-K=7 verification**. Adaptive verification is doubly blocked for us:
  version (v0.28+ needs cu13) AND hardware (varlen decode CUDA graphs need FA3/SM90 or FlashInfer trtllm-gen/SM100;
  SM86 rejected at startup). See `docs/dspark-native-audit.md` §2.

## Gate D — Scheduling and correctness: PASS
- 16 concurrent heterogeneous requests (short 10-tok / long ~50-tok prompts, max_tokens 32/48 mixed, greedy):
  all produced coherent, correct output; no crashes/corruption.
- Per-position acceptance under concurrency decays smoothly (pos0=135 → pos1=106 → pos2=84 → pos3=68 …) —
  proposals and verification are causally aligned as designed.
- ON/OFF transitions: not applicable in v0.25.0 (no adaptive path; K fixed at 7 by config).

## Gate E — Stable serving + smallest matched comparison (AR vs native fixed-K DSpark)
Same target, same runtime, same GPU (3090 #1), ShareGPT (real prompts, `--ignore-eos`, greedy), 60 prompts/cell,
`max-concurrency` C ∈ {8, 32, 64}, prefix caching on both arms. Single trial per cell (smallest comparison;
variance from Phase 4.2 experience is <2% on this workload class).

| C | AR tok/s | DSpark(K=7) tok/s | Δ vs AR | AR TPOT ms | DSpark TPOT ms | AR TTFT ms | DSpark TTFT ms |
|---|---|---|---|---|---|---|---|
| 8 | 463.2 | **757.9** | **+61.5%** | 13.43 | 8.92 | 104.7 | 135.6 |
| 32 | 995.0 | **1116.1** | **+12.2%** | 15.31 | 20.66 | 132.9 | 184.4 |
| 64 | **1134.0** | 1082.6 | **−4.5%** | 16.94 | 30.17 | 227.8 | 418.1 |

**Findings:**
1. **Native fixed-K DSpark reproduces the load-dependent K effect on our hardware with a learned drafter**:
   large win at low load (+61%), near-neutral at medium (+12% — still winning), net-negative at high load (−4.5%).
   The crossover sits between C=32 and C=64 for Qwen3-4B on one 3090 (vs C≈32–96 for Qwen2.5-7B in Phase 4.2 —
   the 4B model saturates later, as expected; confirms addendum §6's warning not to assume C=96 is the same
   point).
2. **TPOT degrades monotonically with load under DSpark** (8.9→20.7→30.2 ms) while AR stays flat-ish
   (13.4→16.9) — the verification-token cost of fixed K=7 is visible exactly where it should be. TTFT inflation at
   C=64 (418 vs 228 ms) shows draft/verify tokens crowding prefill in the batched-token budget.
3. This is a **matched-execution mechanism comparison** (addendum §6): same target weights, drafter, runtime,
   hardware, workload, measurement protocol. It does NOT compare against adaptive verification (not runnable —
   Gate C) and must not be read as "DSpark loses" — it measures what fixed-K=7 does, which is the strongest
   *runnable* native baseline on this hardware.

**Raw results:** `/tmp/gate_e_results/*.json` (6 files: {ar,dspark}_c{8,32,64}); server logs
`/tmp/dspark_gate_server.log`, `/tmp/ar_gate_server.log`; bench logs `/tmp/bench_{dspark,ar}_c*.log`.

## Milestone status (final, per addendum §4's non-interchangeable milestones)
| Milestone | Status |
|---|---|
| Source support exists (upstream) | Yes — PR #47808 (2026-08-12); drafter since v0.25.0 (2026-07-11), adaptive verification since v0.28.0 (2026-08-26). |
| Model loads locally | **Yes** — Qwen3-4B + dspark_qwen3_4b_block7 on SM86, cu126 source build. |
| Adaptive verification active locally | **No — definitive blocker** (version: v0.28+ needs cu13; hardware: varlen decode CG needs SM90/SM100). |
| Serving benchmark completed | **Yes** — fixed-K DSpark vs AR, 3 loads, Gate E table above. |

## What this means for the project
- The strongest *runnable* native baseline on our hardware is **fixed-K=7 DSpark**, and it already does exactly
  what our Phase 4.2/4.6 measurements predicted: wins at low load, loses at saturation. A K-adaptive controller
  that can commit to SD-off (or small K) at the crossover has a measured headroom of ~4.5 pts at C=64 on this
  model pair — smaller than Qwen2.5-7B's 20+ pts because the 4B drafter is cheaper and acceptance lower.
- The remaining open question for our contribution (per addendum §7): does a *live-serving-state* decision beat
  native fixed-K DSpark at/above the crossover, and can it also handle what fixed-K cannot (prefill-heavy mixes,
  SD-off economics)? That is the smallest falsifiable experiment — to be pre-registered next.
