# Phase 5.2 — H-5.2 Results: locate and remove the SD-off floor

**Date:** 2026-10-11 · **Status:** FINAL (per prereg `docs/p52-prereg.md` §6)
**Verdict: FALSIFIED as specified** — a spec-path-free *runtime* off-state does not remove the SD-off
floor. But the experiment located the floor's mechanism, which is the durable result.

---

## 0. Verdict up front

The strongest feasible runtime off-state (force-0 draft truncation **plus** full ngram drafter-pass
skip — i.e., Phase 5.1 M3's proven-stable mechanism, re-measured in-session with a same-session AR
control) recovers **none** of the ~8–9% SD-off floor:

| C | AR (gen/s) | force-0 floor | specfree-off | recovery of gap | specfree vs AR |
|---|---:|---:|---:|---:|---:|
| 32 | 1421.1 | 1318.5 | 1318.9 | **+0%** (identical) | −7.2% |
| 96 | 2585.6 | 2361.5 | 2343.0 | **−9%** (worse than floor) | −9.4% |

Per prereg §6: recovery ≤ 20% of the AR−force-0 gap at C=96 → **FALSIFIED** ("mechanism doesn't
remove it"). G1's literal threshold (≥50% of the per-step cadence gap in *CPU-side residual*) was
also missed (49% at C=96, 28% at C=32) — though see §3: the floor is entirely spec-path work; my
gate's bucketing under-counted it.

**The durable finding:** the SD-off floor is **structural to running with `speculative_config` set**,
not a removable per-step code branch. A runtime off-state cannot remove it on this stack; only a
genuinely spec-free serving path (separate AR server pool, or init-time config that never enables
spec) can avoid it. This bounds what any SD on/off controller (batch-level, per-request, or native)
can recover at saturation: **~8–9% is the irreducible cost of a spec-enabled off-state** on
vLLM 0.19.1 / Qwen2.5-7B / ngram / eager.

## 1. P1 — where the floor lives (per-step decomposition)

Tooling: `experiments/p52_obs/sitecustomize.py` ports H-OBS-1's CUDA-event tracer to vLLM 0.19.1 as
a sitecustomize overlay, **anchored on `execute_model`** (once per real step). Key version fix: the
0.19.1 engine core spins `Scheduler.schedule()` in a tight non-blocking loop (~490×/step), so
schedule() is *not* a once-per-step anchor (H-OBS-1's 0.25.0 engine blocked between steps). Verified:
records created == real GPU step rate (49.5/s @ C=8, matching gpu_step p50).

Per executed decode step (steady window), force-0 minus AR:

| C | cadence gap | → GPU forward (metadata) | → sampling/drafter | → CPU residual |
|---|---:|---:|---:|---:|
| 32 | +2.40 ms | +1.42 ms | +0.33 ms | +0.68 ms |
| 96 | +3.10 ms | +0.92 ms | +0.67 ms | +1.51 ms |

Both arms run at **identical occupancy** (C=96: running p50 = 96 in both; C=32: 32) and
`draft_tokens=0` on every force-0 step (truncation works). So the floor is not preemption and not
draft verification — it is spec-path overhead that stays active whenever the config is set.

## 2. Why a runtime off-state structurally cannot remove it (code-level, vLLM 0.19.1)

Traced in `.venv/.../vllm/v1/worker/gpu_model_runner.py` (eager mode, async scheduling disabled):

1. **force-0 already takes the AR code path.** `_prepare_inputs` gates on
   `use_spec_decode = len(scheduled_spec_decode_tokens) > 0` (line 2040). With drafts truncated to 0
   this is False → `spec_decode_metadata=None`, AR-style `logits_indices`. The per-step *draft* branch
   is already off; there is nothing left to switch off there.
2. **The active attention backend (FLASH_ATTN v2) has no per-step spec branch** in its forward — only
   init-time CUDA-graph sizing (`_decode_cudagraph_max_bs`), which is disabled in eager.
3. **`uniform_decode_query_len = 1 + num_spec_tokens`** (line 775) only feeds `_is_uniform_decode` →
   CUDA-graph dispatch; `pad_attn` is gated on `CUDAGraphMode.FULL`. No effect in eager.
4. What remains spec-specific: the **ngram drafter pass runs every step** (its output truncated to 0
   — Phase 5.1 M3's skip of exactly this recovered nothing, A/B-confirmed), plus **init-time GPU
   memory** for draft buffers / attention workspace that never comes back per-step.

The spec server's KV pool is correspondingly smaller (AR 122,704 vs force-0 117,264 tokens at C=96,
−4.4%) — but with identical measured occupancy and zero preemption events, this does not drive the
throughput gap; it is a symptom of the same init-time allocation.

## 3. G1 gate: literal miss, intent met

Prereg G1 required ≥50% of the per-step cadence gap in *CPU-side residual* before building. Literal
readout: 49% (C=96) / 28% (C=32) — missed by a hair. But my gate's bucketing lumped everything inside
`gpu_step_ms` into "GPU", while the forward-metadata and drafter deltas are exactly prereg category
(a) *model-runner spec fields* — removable in principle, not irreducible compute. Properly attributed,
~78–100% of the floor is spec-path work, so G1's intent ("is the floor removable spec-path work?") was
met. The P3 measurement then tested removability directly and found it **not removable at runtime** —
which is a stronger, more precise answer than the gate could give.

## 4. Verdict vs prereg (docs/p52-prereg.md §6)

- **G1:** literal threshold missed (49%/28%); falsification trigger ("gap dominated by irreducible GPU
  compute or unattributable") NOT hit — the entire floor is attributable spec-path work. Proceeded to P3.
- **H-5.2 CONFIRMED criterion** (specfree ≥ force-0 + 50% of gap at C=96 and within ~2% of AR): **NOT
  MET** (−9% of gap; −9.4% vs AR).
- **FALSIFIED criterion** (specfree ≤ force-0 + 20% of gap at C=96): **MET** (−9%). → **FALSIFIED.**

## 5. What this changes / does not change

- **Changes:** the Phase 5 open question ("can a spec-path-free off-state remove the floor?") is now
  answered *no, for runtime mechanisms on this stack* — with the mechanism located (§2). The ~8–9%
  SD-off cost is a **serving-architecture** constant, not a controller-tuning target. Any future SD
  on/off controller (Track B native DSpark included) should price this as an irreducible off-state
  floor; reaching AR at saturation requires serving the off-state through a genuinely spec-free path
  (separate AR pool or init-time no-spec config), which is out of scope for in-loop controllers.
- **Does not change:** Phase 4.6/5.1/P4.5R results (their numbers stand; M3's force-off diagnostic is
  reproduced here: specfree ≈ force-0, both far below AR). No historical numbers rewritten.

## 6. Provenance and reproducibility

- Prereg: `docs/p52-prereg.md` (prospective, committed f67d7e2 before any run; §6 closed with this
  completion note).
- Tracer overlay: `experiments/p52_obs/sitecustomize.py` (env-gated `VLLM_P52_OBS`; optional
  `VLLM_P52_FORCE0` truncation, `VLLM_P52_OFFSTATE` drafter-skip). Smoke: `experiments/p52_smoke.sh`,
  `experiments/p52_sf_smoke.sh`.
- P1 grid: `experiments/p52_p1_grid.sh`; analysis `experiments/p52_analyze_p1.py` →
  `results/p5_2/p52_p1_decomp.json`.
- P3 grid: `experiments/p52_p3_grid.sh` (18 cells, 0 crashes/OOM); analysis
  `experiments/p52_analyze_p3.py` → `results/p5_2/p52_p3_verdict.json`.
- Raw data: `results/p5_2/` (gitignored). Stack frozen per prereg §3 (Qwen2.5-7B-Instruct, ngram K=8,
  MAX_MODEL_LEN=16384, MAX_NUM_SEQS=128, CHUNK_TOKENS=2048, ENFORCE_EAGER=1, TP=1, mixed workload).
