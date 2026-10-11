# Phase 5.2 — H-5.2 Pre-registration: locate and remove the SD-off floor

**Date written:** 2026-10-11 · **Status:** PROSPECTIVE (fixed before any run) · **Parent:** Issue #23 (Phase 5), supersedes the paused "H-5.2 pre-registration pending" item
**Scope decision (user, 2026-10-11):** proceed with H-5.2 as scoped — profile where the ~9% SD-off floor lives, then build a mechanism that removes it (spec-path-free off-state), on the Qwen2.5-7B / ngram / vLLM 0.19.1 stack.

---

## 1. Question

Phase 5.1 falsified H-5.1: a batch-level SD on/off decision + draft-pass skip commits the whole
batch to SD-off at saturation, but the off-state arm lands **~5–8% below AR** (C=96) and the
draft-pass skip recovered *none* of that gap. The residual is the **SD-off floor**: the cost of
running on a spec-decode-enabled server with drafts truncated to zero.

H-OBS-1 already established the GPU side is not it: an AR (no-spec) server's non-forward GPU cost
is only 0.95–0.97 ms/step (sampler + async D2H copy), and `RejectionSampler.parse_output` is
runtime-adaptive (takes the cheap `max_gen_len==1` path when no drafts). So a spec server at k\*=0
should cost ≈AR **on the GPU**. The floor must therefore be **CPU-side per-step bookkeeping that
stays active whenever `speculative_config` is set**, dominated by model-runner *static* spec fields:

- `is_spec_decode = bool(speculative_config)` (gpu_model_runner.py:622) — set at init, never cleared.
- `uniform_decode_query_len = 1 + num_spec_tokens` (gpu_model_runner.py:775) — attention metadata
  sized for K+1 query positions every decode step, even when k\*=0.
- per-step spec input-batch / draft-token-id buffers prepared whenever the config is present.

**H-5.2 hypothesis:** a *per-step* spec-path-free off-state (make these fields runtime-adaptive so
that when k\*=0 they collapse to their AR values) removes most of the SD-off floor, bringing the
off-state arm to within noise of AR at C=96.

## 2. Prior evidence this builds on (all committed)

- **Phase 5.1** (`docs/phase5-p1-results.md`): batch decision + uniform allocation works; draft-pass
  skip (early-return in `propose_draft_token_ids`) recovers **none** of the floor → the drafter
  kernel is not the floor. Off-state arm ≈ force-0 floor at C=96 (−5 to −8% vs AR).
- **H-OBS-1** (`docs/hoobs1-observability.md`): spec server non-forward GPU overhead at k\*=0 is
  ~0.95–0.97 ms/step ≈ AR; no exploitable GPU-side gap. nsys + CPU-cadence tooling exists and is
  validated (H-OBS-1 Gate-1 PASS).
- **P4.6 force-0 floor** (this stack, eager): C=32 = 1336.9 vs AR 1456.9 (**−8.2%**); C=96 =
  2346–2432 vs AR ~2557 (**−5 to −8.4%**). This is the floor to be removed.

## 3. Stack (frozen, identical to P4.6/Phase 5.1)

Qwen/Qwen2.5-7B-Instruct · ngram K=8 (KMAX=8) · MAX_MODEL_LEN=16384 · MAX_NUM_SEQS=128 ·
CHUNK_TOKENS=2048 · **ENFORCE_EAGER=1** (the floor was measured eager; CUDA-graph mode is out of
scope — `uniform_decode_query_len` is baked into graph capture there) · TP=1 · mixed workload
(Phase 4.2 `latency_client.py`) · server-side `vllm:generation_tokens` rate over steady t≥20s.

**Isolation:** all vLLM source edits happen in a dedicated git worktree (branch `p5-2-spec-free-off`),
never in the main repo's `.venv`. The off-state arm runs against the patched vLLM; AR and force-0
controls run against stock vLLM so the comparison is clean.

## 4. Arms

| Arm | Config | Purpose |
|---|---|---|
| **AR** | `SPEC_METHOD=none` | upper bound (no spec path at all) |
| **force-0** | ngram K=8 + force k\*=0 every step, stock vLLM | the floor baseline (re-measured fresh this run for same-session comparability; P4.6 value cited as cross-check) |
| **specfree-off** | ngram K=8 + force k\*=0 every step, **patched** vLLM (off-state mechanism §5) | does the mechanism remove the floor? |

AR baselines may be reused from `results/p2/` (C=96 mean ~2557.2; C=32 1456.9) for the headline
number, but a fresh AR cell is run in-session as an environment control.

## 5. Mechanism (what "spec-path-free off-state" means, concretely)

When the per-step decision is k\*=0 for the whole decode batch (the off-state), make the model
runner behave exactly as if `speculative_config` were None:

1. **`uniform_decode_query_len`:** use 1 (not `1+num_spec_tokens`) for that step's attention
   metadata, so query buffers are sized for AR decode, not K+1 positions.
2. **`is_spec_decode`:** the per-step attention-metadata / input-batch path must not take the
   spec branch (K+1 query layout, draft-token-id slots) when k\*=0.
3. **Per-step spec buffers:** skip preparing draft-token-id / `num_decode_draft_tokens` buffers and
   the D2H copy stream for that step (they are only needed to carry drafts back).

The drafter pass is already skipped in Phase 5.1's off-state; this mechanism adds the *static-field*
and *metadata-sizing* removals that Phase 5.1 did not touch. **Design constraint:** no change to any
token, acceptance, or sampling math — only the bookkeeping/metadata path. If a field cannot be made
runtime-adaptive without touching CUDA-graph capture (eager-only is fine), document it and fall back
to leaving that one field static; the verdict must reflect which fields were actually removed.

## 6. Pre-registered run plan, hard stops, falsification

**P1 — Profile (attribute the floor).** nsys + CPU-cadence on **AR vs force-0** at C=32 and C=96
(60–90 s each, single GPU). Attribute the wall-clock/step gap to: (a) model-runner static spec
fields / attention metadata sizing, (b) scheduler spec bookkeeping, (c) GPU step time, (d) other.
- **Gate G1:** ≥50% of the floor gap must be attributable to **(a) or (b)** (CPU-side spec bookkeeping)
  for the mechanism to be justified. If the gap is dominated by (c) GPU or an unattributable (d),
  H-5.2-as-specified is **falsified** — report where it actually lives and stop before building.

**P2 — Mechanism build.** Implement §5 in the worktree; unit-test that k\*=0 steps produce byte-identical
sampled tokens to AR (correctness invariant), then a C=32 smoke for OOM/crash.

**P3 — Adjudication run.** specfree-off vs force-0 vs AR, C∈{32,96} × 3 trials, DUR=150 s.
- **H-5.2 CONFIRMED:** specfree-off ≥ force-0 + (≥50% of the AR−force-0 gap) at C=96 **and** within
  ~2% trial noise of AR at C=96 **and** no regression below −2% vs force-0 at C=32.
- **PARTIALLY CONFIRMED:** specfree-off recovers some but <50% of the gap, or reaches AR at one C only.
- **FALSIFIED:** (i) G1 fails (floor not in CPU-side spec bookkeeping), OR (ii) specfree-off ≤ force-0
  + 20% of the gap at C=96 (mechanism doesn't remove it), OR (iii) correctness invariant broken.

**Hard stops (abort, no re-tune):** OOM or serving crash in any adjudication cell; the mechanism
changes sampled tokens vs AR on a k\*=0 step; total wall > 4 GPU-hours for P1+P3 combined.

## 7. What this does / does not change

- **Does:** directly test whether the SD-off floor is removable by making the spec path runtime-adaptive —
  the missing piece that falsified H-5.1 and blocks any batch-level (or per-request) SD on/off controller
  from reaching AR at saturation.
- **Does not:** re-litigate the decision rule (Phase 4.6/5.1/P4.5R settled it), touch native DSpark
  (Track B), or claim anything about a trained-drafter stack. If CONFIRMED, the off-state mechanism is a
  general patch to vLLM's spec path, applicable to any SD on/off controller; if FALSIFIED, the floor is
  located and documented, which bounds what any such controller can recover at saturation.

## 8. Provenance (filled after run)

- Worktree: `git worktree add /home/junior1/_dev/p52-spec-free-off -b p5-2-spec-free-off`
- Mechanism diff: <commit> · unit test: <path> · raw data: `results/p5_2/` (gitignored)
- Report: `docs/p52-results.md`
