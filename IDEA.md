A research notebook for an open question.

- Log experiments, results, and dead ends
- Cite sources inline
- Weekly synthesis of what I learned

---

# Open question

**How should request scheduling in LLM serving (Orca-style continuous batching,
chunked prefill) be redesigned when speculative decoding (SD) is enabled?**

Working idea: a *Lightweight Decision Model* that observes the serving state
(queue, per-request acceptance history, KV-cache occupancy, GPU load, SLOs) and
decides each iteration which requests to run, per-request speculation length K,
prefill chunk size, and SD on/off.

Advisor (2026-10-07): direction is right; dynamic-SD-config work exists in quantity
→ **first observe vLLM under load, find the concrete problems, quantify the effect
of reconfiguration**, then differentiate. See `docs/observation-plan.md`.

## Pointer index

| What | Where |
|---|---|
| Related work + gap analysis | `docs/literature-map.md` |
| Phase-1 experiment design | `docs/observation-plan.md` |
| vLLM code-level audit | `docs/vllm-code-audit.md` (TBD) |
| Running progress log | `PROGRESS.md` (+ GitHub project #1) |
| Experiment scripts | `experiments/` |
| Results | `results/<exp>/` |

## Environment

- 2× RTX 3090 (24 GB), driver 575.57.08 / CUDA 12.9, ~94 GB RAM
- `.venv`: Python 3.12, vLLM 0.31.0, torch 2.13.0+cu129
- Models: `Qwen/Qwen2.5-7B-Instruct` (target), `leptonai/EAGLE-Qwen2.5-7B-Instruct` (draft)

## Log

### 2026-10-07 — kickoff
- Literature scan done. Direct prior work on *serving-level* dynamic SD: Nightjar
  (batch-level bandit K + global on/off, vLLM), DSDE (per-seq KLD-variance SL + cap,
  vLLM fork), DSpark scheduler (per-request verify length from confidence head,
  DeepSeek-V4 production). Single-request adaptive-K space is crowded (DISCO,
  SpecDec++, AdaEDL, SVIP, PEARL, AdaSD) — none look at batch/system state.
- Gap we can own: **joint** decisions {admission, per-request K, chunk size, SD on/off}
  from a rich serving state, with prefill–decode and KV-cache coupling + SLO/goodput
  objective. vLLM upstream has no runtime dynamic-K (fixed at init, CUDA graphs).
- Plan: Phase 1 = observation runs in stock vLLM (fixed K sweep under Poisson load),
  Phase 2 = offline oracle-gap replay, Phase 3 = controller prototype.

