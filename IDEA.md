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


## Current evidence (2026-10-08)

- **Serving load changes the optimal speculation configuration.** On the short mixed workload, K=8 wins at C=8, is near-tied with AR at C=32, and AR wins by ~21% over K=8 at C=96.
- **Prefill pressure is an independent load axis.** At the same C=32, changing from short ISL (~50) to SPEED-Bench ~2k-token prefills flips the ranking from K=8≈AR to AR beating K=8 by ~13%.
- **Request-level effects remain heterogeneous.** SPEED-Bench categories show large TPOT differences between low- and high-entropy requests even when aggregate throughput is near-tied.
- **Neither tested adaptive baseline solves saturation.** At C=96, acceptance-only LDM remains ~20% below AR and the DSpark decision-rule reimplementation is ~25% below AR.
- **Immediate hypothesis:** profitable speculation requires both request-level draftability and a **live serving-state/load-regime signal**, including the ability to select K=0 (SD off).

Current Phase 4 status: **4.1✓ 4.2✓ 4.3✓ 4.5✓; 4.4 pending; 4.6 next.** Phase 5 is reserved for speculation-aware scheduler design after Phase 4 validation is complete.

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

