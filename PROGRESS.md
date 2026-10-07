# Progress Log — [iESLAB] Better vLLM Speculative Decoding

GitHub Project: https://github.com/orgs/WorkWith-Research/projects/1 (projectV2 id `PVT_kwDOEyEOMs4Bl95-`)

## Project items (for API updates)

| Item ID | Issue | Title | Status |
|---|---|---|---|
| `PVTI_lADOEyEOMs4Bl95-zg-9nck` | #1 | Phase 0: Environment setup | Done |
| `PVTI_lADOEyEOMs4Bl95-zg-9nc4` | #2 | Phase 1: Observation — fixed-K SD under load | In progress |
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
- **⚠️ CRITICAL FINDING — synthetic rejection sampler is a no-op in vLLM 0.19.1:**
  `rejection_sample_method=synthetic` + `synthetic_acceptance_rate` are *accepted* by the config
  (and logged in `non-default args`) but **not applied** when drafting with ngram. Proof: three runs
  at rates 0.35/0.60/0.85 produced **byte-identical** total accepted tokens (8680), per-position
  acceptance (93.99/91.24/89.37%), and mean accept len (3.75). The worker's `RejectionSampler`
  branch (`rejection_sampler.py:560`) exists but is evidently not hit on this path. **Consequence:**
  the "vary acceptance rate" axis cannot use the built-in synthetic feature in this version.
  v1 sweep (34 runs) archived to `results/_v1_invalid/` — its SD runs are invalid; only the 4 AR
  baselines (`A_ar_r{2,6,12,24}`) remain usable.
- **Pivot to K-sweep (Phase 1 v2):** speculation length K is the user's core decision variable and
  does NOT depend on the broken synthetic feature. With real ngram acceptance (~3.75 tok/step at
  moderate load), validated that accept len scales cleanly with K (K=1→1.95, K=8→7.28) and
  throughput follows (740→801 tok/s @ r6). Hypothesis: optimum-K shifts with load → the oracle gap
  for dynamic-K scheduling. `phase1_sweep_v2.sh`: K∈{1,2,4,8} × rate∈{2,6,12,24}, real ngram,
  all knobs exported (fixed v1's non-exported-var bug that silently defaulted K=3, num_prompts=40).
- **Harness bug class fixed:** shell vars set at a script's top are NOT inherited by child scripts
  unless `export`ed. v1 sweep's `K`, `NUM_PROMPTS` were plain vars → children used defaults. All v2
  knobs are exported; `run_experiment.sh` now also accepts EXP as `$1`.

### 2026-10-07 (cont.) — Phase 1 v2 K-sweep result + v3 launch
- **v2 random-prompt K-sweep COMPLETE (16/16).** Full grid in `docs/phase1-results.md`.
  - Throughput: **K=8 wins at every load** (r2 756 / r6 962 / r12 1038 / r24 1049 tok/s);
    K=1 is 1.15–1.20× slower than K=8. SD@K=8 beats AR baseline by ~1.4–1.44× at matched load.
  - Accept len scales ~linearly with K (1.96→7.7) and is **flat across load** (draft_acc_rate
    0.95→0.82). GPU util ≈93–95%, KV-util max ≈0.20, zero preemptions at all loads.
- **⚠️ NULL RESULT — no interior optimum-K, no load-driven shift on random prompts.**
  ngram acceptance on templated random prompts is high (~0.9+) and nearly uniform across load, so
  there is no over-speculation penalty to create an interior optimum → larger K always wins. The
  predicted "optimum-K moves with load" pattern does not appear here. This reframes Phase 1:
  *whether* an optimum-K exists (and where it sits) is governed by how low + heterogeneous the
  per-request acceptance rate is — random prompts are an easy-draft case.
- **Weak load-dependent signal found:** p99-TPOT tail penalty grows with K (K=8 ≈72–77 ms vs
  K=1 ≈43–53 ms) → a tail-latency/SLO tradeoff, but small vs the throughput gain on this workload.
- **v3 ShareGPT K-sweep launched** to reach the low/varied-acceptance regime: smoke (K=8,r6) gave
  mean accept len ≈3.0 tok/step, draft_acc_rate ≈0.38 — ~2.5× lower than random → interior optimum-K
  expected. `phase1_sweep_v3_sharegpt.sh` + `DATASET_PATH` knob (vLLM bench serve does NOT
  auto-download ShareGPT; dataset at `data/`, git-ignored). Full grid running (~70 min).
