# Progress Log — [iESLAB] Better vLLM Speculative Decoding

GitHub Project: https://github.com/orgs/WorkWith-Research/projects/1 (projectV2 id `PVT_kwDOEyEOMs4Bl95-`)

## Project items (for API updates)

| Item ID | Issue | Title | Status |
|---|---|---|---|
| `PVTI_lADOEyEOMs4Bl95-zg-9nck` | #1 | Phase 0: Environment setup | In progress |
| `PVTI_lADOEyEOMs4Bl95-zg-9nc4` | #2 | Phase 1: Observation — fixed-K SD under load | Backlog |
| `PVTI_lADOEyEOMs4Bl95-zg-9neU` | #3 | Phase 2: Analysis — oracle gap for dynamic K | Backlog |
| `PVTI_lADOEyEOMs4Bl95-zg-9nfY` | #4 | Phase 3: Prototype — lightweight decision model | Backlog |

Status option IDs: Backlog=`f75ad846` Ready=`61e4505c` In progress=`47fc9ee4`
In review=`df73e18b` Done=`98236657`. Update via `scripts/set_item_status.sh <item-id> <option-id>`.

## Log

### 2026-10-07 (cont.) — Phase 0 → Phase 1 transition
- **EAGLE-Qwen2 incompatibility:** vLLM 0.19.1 registry has no `EagleQwen2ForCausalLM`;
  the leptonai EAGLE-Qwen2.5 draft is a raw single-layer `Qwen2ForCausalLM` head →
  `value_error` on serve. Real-EAGLE path needs Llama-3.x (HF-gated, no token) or an
  `Eagle*`-wrapped Qwen checkpoint we don't have. **Decision:** Phase 1 uses **ngram
  drafting + vLLM's synthetic rejection sampler** (`rejection_sample_method=synthetic`)
  on the open Qwen2.5-7B — isolates *scheduler* behavior from draft-model quality, which
  is exactly what Phase 1 studies. Real EAGLE numbers deferred to a later phase.
- **Synthetic acceptance semantics (verified):** `synthetic_acceptance_rate` = mean joint
  acceptance probability; vLLM converts it (via `compute_synthetic_rejection_sampler_params`)
  to base-rate + geometric decay. p=0.8→mean accept len≈4.2, 0.6→3.4, 0.35→2.4 tokens/step.
  Gives precise, reproducible, decoupled acceptance profiles. Confirmed live: smoke test
  observed mean accept len 3.74 (p=0.8).
- **Scraper gauge bug found + fixed:** vLLM V1 keeps a stale `model_name`-labeled gauge
  series at 0 that must not be summed with the live engine series; also `_created`
  timestamp gauges leak in. Rewrote `scrape_metrics.py` to keep gauges per-label-set (max)
  and drop `_created`. Verified: now captures running/waiting/kv_util/gpu correctly.
- **`request-rate=inf` behavior:** bench client sends all requests at t=0 but the engine
  trickles them in (observed waiting→20, running filling over ~12s). Finite rates = Poisson
  inter-arrivals rescaled to exactly N/rate s → clean open-loop steady state. Phase 1 uses
  finite rates {2,6,12,24} req/s.
- **Phase 1 sweep launched** (`experiments/phase1_sweep.sh`, 22 runs on 2 GPUs):
  - Batch A: acceptance {0.85,0.60,0.35} × load {2,6,12,24} + AR baseline (chunk=2048).
  - Batch B: chunked-prefill budget {512,2048,8192} × load {6,12,24}, AR + synth-0.60 SD.
  All K=4, ngram+synthetic, Qwen2.5-7B, in/out 512/384, max-len 8192.
- **Paper briefs done** → `docs/paper-briefs.md` (Nightjar/DSDE/DSpark + TETRIS/BanditSpec).
  Positioning confirmed: no prior work jointly decides {admission, per-request K, chunk size,
  SD on/off} from rich live serving state with an SLO/goodput objective. Our gap is real.
- vLLM code-audit subagent dispatched (lookahead-KV path, draft→next-step flow, dynamic-K
  patch surface) → `docs/vllm-code-audit.md` pending.
