# Phase 5.1 Results — Batch-level SD on/off + draft-pass skip (H-5.1)

**Verdict: FALSIFIED as specified.** The batch-level decision works exactly as designed (commits the whole
batch to SD-off at saturation, keeps speculation ON at low load), but the arm does not reach AR at C=96 — and
the draft-pass-skip mechanism costs ~3% rather than removing overhead. All numbers are server-side
`vllm:generation_tokens[counter]` rate over the steady window (t≥20s), Phase 4.6 methodology, all measured
same-day.

## Setup (identical to Phase 4.6)
Qwen2.5-7B-Instruct, 1× RTX 3090, vLLM 0.19.1 eager, ngram self-draft K≤8, MAX_MODEL_LEN=16384,
MAX_NUM_SEQS=128, CHUNK_TOKENS=2048, mixed workload (Phase 4.2), DUR=150s, WARMUP=20s, 3 trials per cell.

## Controller (`ldm_batch`, pre-registered in PROGRESS.md)
- **Batch-level decision:** `K_batch = argmax_{k∈0..8} SPS(B_eff + N_dec·k) · (1 + mean_l[k])` — the single
  SPS term prices the slowdown of ALL decode requests' extra scheduled tokens (the externality per-request
  greedy misses).
- **Draft-pass skip:** when `K_batch==0`, worker-side `propose_draft_token_ids` returns early (vLLM runs the
  ngram draft pass unconditionally whenever `speculative_config` is set — that was the hypothesized ~4.7% fixed overhead).
- Mechanical corrections M1/M2 (post-smoke OOM, decision rule shape unchanged): decode-only load signal
  (`B_eff`, `N_dec` over decode-phase requests) + uniform allocation (k\*=K_batch on ON steps, matching exactly
  what the argmax priced). See PROGRESS.md for the full M1/M2 rationale.

## Results (server-side gen/s, steady t≥20s)

| arm | C=8 | C=32 | C=96 |
|---|---|---|---|
| AR (no spec) — same-day control | 387.0 (p2) | 1460.0 (p2) | **2641.6** |
| K8 (best fixed) | 430 (+11.1%) | 1437 (−1.6%) | 1937 (−26.7%) |
| ldm_load (Phase 4.6) | 417.0 (+7.8%) | 1304.2 (−10.7%) | 2235.2 (−15.4% same-day AR) |
| **ldm_batch (H-5.1)** | **410.1 (+6.0%)** | 1302.3 (−10.8%) | **2422.2 (−8.3%)** |
| P4.6 force-0 floor (spec ON, k\*=0, no skip) — same-day control | — | — | 2431.9 (−7.9%) |
| **ldm_batch force-off (skip every step)** | — | — | **2355.2 (−10.9%)** |

## Pre-registered bounds → adjudication

| bound | requirement | measured | verdict |
|---|---|---|---|
| C=8 keeps the gain | ldm_batch ≥ +5% vs AR (≥406) | 410.1 (+6.0%) | **PASS** |
| C=96 reaches AR | ldm_batch ≥ AR − noise (≥2634) | 2422.2 (−8.3%) | **FAIL** |
| C=96 beats force-0 floor | ldm_batch > 2431.9 | 2422.2 (−0.4%) | **FAIL** |
| mechanism cost | force-off ≥ AR − 1% (≥2615) | 2355.2 (−10.9%) | **FAIL** |

Per charter §13: no post-hoc retuning. H-5.1 is FALSIFIED as specified.

## What the mechanism DID (the decision rule works)

Decision logs confirm the batch argmax behaves exactly as designed, even at the all-ones cold-start prior
(OFF periods schedule no drafts → `make_spec_decoding_stats` never fires → acceptance history stays at prior):

| regime | K_batch | interpretation |
|---|---|---|
| C=8 steady | 5 (OFF=0% of steps) | speculation ON, near-full draft length |
| C=32 steady | 1 (OFF=0%) | minimal drafts — externality priced in |
| C=96 steady | 0 (OFF≈99.3%) | **whole batch committed to SD-off** |
| force-off diagnostic | 0 (100%) | skip active every step, 0 drafts scheduled |

This is precisely the behavior Phase 4.6's per-request greedy could not produce (it oscillated with a ~20%
K=8 minority at C=96). The externality pricing in the batch argmax works: at B_eff≈96, `SPS(96+96k)` drops
faster than `(1+mean_l[k])` grows, so k=0 wins.

## Why it still fails (two independent gaps)

**Gap 1 — the decision is right but "SD-off" isn't free.** ldm_batch at C=96 (2422.2) ≈ P4.6 force-0 floor
(2431.9, −0.4%) < AR (2641.6). The batch controller achieves the SD-off optimum — but the SD-off state on a
spec-enabled server is itself ~8% below AR. Phase 4.6 already showed this floor is structurally unreachable by
any K-only controller; H-5.1's answer was to make the off-state free by skipping the draft pass. It didn't.

**Gap 2 — the draft-pass skip does NOT remove the floor (A/B-confirmed).** The force-off diagnostic (skip every
step, 0 drafts) measured **2351±5** (3 runs: 2354.3, 2355.2, 2344.7) vs the no-skip floor (spec path enabled,
k\*=0 for all) **2398±43** (3 runs: 2431.9, 2416.6, 2346.4; one P4.6 run drifted low mid-day). Same GPU, same day,
identical KV cache (117,264 tokens), 0 OOM, err=0 in every run. The skip costs ~0–2% (within the no-skip arm's
own spread) and recovers NONE of the ~9% floor gap to AR. Conclusion: the Phase 4.6 "fixed spec-path overhead"
does NOT live in the ngram draft kernel — skipping it at runtime neither helps nor meaningfully hurts, so the
floor must sit in the rest of the spec-enabled serving path (scheduler bookkeeping, rejection-sampler setup,
input-batch spec fields) that stays active whenever `speculative_config` is set.

## Interpretation for Phase 5

- The **decision rule is validated**: batch-level argmax prices the externality and commits to SD-off at
  saturation where per-request greedy oscillated. This is a real, reusable result (the mechanism works; the
  cost model it acts on was wrong).
- The **cost model was wrong in one way that matters**: "SD-off" has a ~9% fixed cost on a spec-enabled server
  that no K-allocation and no draft-pass skip can avoid. The lever must be upstream of the serving path —
  e.g., admitting/serving requests through a genuinely spec-free code path, or removing the spec-path
  bookkeeping itself — not in the drafter kernel.
- Next step: Phase 5.2 pre-registration around a spec-path-free off-state (or its measured cost), informed by
  profiling WHERE the ~9% floor lives (scheduler vs model-runner spec fields) before committing to a mechanism.

## Files
- Controller: `experiments/ldm_batch_controller/sitecustomize.py` (M1/M2/M3)
- Cell runner: `experiments/p51_cell.sh`; grid: `experiments/p51_grid.sh`; controls: `experiments/p51_controls.sh`
- Results: `results/p5_1/p51_ldmbatch_c{8,32,96}_tt{1,2,3}_mixed_{metrics,decisions}.jsonl`,
  `results/p5_1/p51_ldmbatch_c96_tf0_mixed_*` (force-off),
  `results/p5_1/p51ctl_AR_knone_c96_metrics.jsonl` (same-day AR),
  `results/p4_6/p46_ldmload_c96_ttf0r_mixed_metrics.jsonl` (same-day P4.6 force-0 floor)
