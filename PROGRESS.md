# Progress Log — [iESLAB] Better vLLM Speculative Decoding

GitHub Project: https://github.com/orgs/WorkWith-Research/projects/1 (projectV2 id `PVT_kwDOEyEOMs4Bl95-`)

## Project items (for API updates)

| Item ID | Issue | Title | Status |
|---|---|---|---|
| `PVTI_lADOEyEOMs4Bl95-zg-9nck` | #1 | Phase 0: Environment setup | Done |
| `PVTI_lADOEyEOMs4Bl95-zg-9nc4` | #2 | Phase 1: Observation — fixed-K SD under load | Done |
| `PVTI_lADOEyEOMs4Bl95-zg-9neU` | #3 | Phase 2: Analysis — oracle gap for dynamic K | In progress |
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

### 2026-10-07 (cont.) — Phase 1 v3 result: optimum-K is workload-dependent (HEADLINE)
- **v3 ShareGPT K-sweep COMPLETE (16/16).** draft_acc_rate @K=8 ≈ 0.34–0.41 (vs random 0.79–0.86).
- **★ The optimum-K flips with the acceptance regime, not primarily with load:**
  - random (high acc): **K=8 best at every load** (over-speculation penalty ≈ 0).
  - ShareGPT (low acc): **interior optimum K=4** at r2/r6/r12; K=8 only edges back in at r24.
  - Crossover is clean: K=8 vs K=4 = random +5.1/+2.9/+10.6/+7.5% (r2/6/12/24) vs sharegpt
    −1.6/−2.3/−5.1/+0.2%. Full table in `docs/phase1-results.md`.
- **Implication for LDM:** a fixed-K policy loses ~5–10% throughput by picking the wrong K for the
  workload's acceptance regime → concrete Phase-2 oracle-gap target. The decision input that matters
  most is **live draft acceptance rate**, which we now measure per-run (and can get per-request).
- **Caveats logged:** load-driven shift at fixed workload is weak (K=4→8 only r12→r24); margins
  modest + single ngram drafter; per-request heterogeneity not yet measured (Phase 2); real EAGLE
  drafts should sharpen the crossover (deferred — EAGLE-Qwen2 incompatible with vLLM 0.19.1).
- `analyze.py` now handles both families (`K{K}_r{rate}` random + `S_K{K}_r{rate}` sharegpt) and
  emits a cross-dataset optimum-K-shift comparison. All 32 runs → `results/summary.csv`.

### 2026-10-07 (cont.) — Phase 2: per-request heterogeneity + oracle gap
- **New tooling (all in `experiments/`):**
  - `spec_hook/sitecustomize.py` — non-invasive env-gated hook on
    `Scheduler.make_spec_decoding_stats` (vLLM computes per-request draft/accept at
    scheduler.py:1370-1390 but aggregates it away). Injected via PYTHONPATH, **no vLLM source
    modified**; tees `{t,req,K,acc}` per (request, decode-step) when `VLLM_SPEC_HOOK_OUT` set.
  - `spectrum_client.py` / `sharegpt_client.py` — Poisson clients (real ShareGPT = genuine
    per-request acceptance diversity). `phase2_mixed.sh` orchestrates server+hook+client.
  - `oracle_gap.py` — builds per-request joint-acceptance profile J[l], reports natural-K
    heterogeneity + truncation headroom + cost-model-bracketed oracle gap.
- **★ Finding 1 — per-request natural-K is highly heterogeneous on real text.** Natural-K
  (mean accept length) on ShareGPT r6 spans **1.18 → 8.82, CV=0.51** (vs synthetic 0.25): some
  requests in one batch want K≈1, others K≈8 → a single fixed-K over-speculates the short ones and
  under-speculates the long ones. This is the concrete problem dynamic-K solves; it must be
  *measured* per request (can't assume).
- **Finding 2 — truncation headroom:** at near-optimal fixed K=4, ~20% of the batch's achievable
  acceptance potential is still truncated from requests that would accept longer chains (cost-model-free floor).
- **Finding 3 — realizable gap is cost-model-dependent.** vLLM verifies all K drafts in ONE target
  forward pass (sub-linear cost), so a naive FLOP model (cost∝1+K) is wrong (predicts K=1 always,
  contradicts Phase-1 measured K=8>K=1). Bracket: linear +0% / sqrt +10.2% / constant +0%.
  Measured aggregate anchor ShareGPT r6: K=4 1037.9 vs K=8 1043.2 tok/s (+0.5%) — the win is from
  *per-request assignment*, not a better global K; should grow with load + a real draft model.
- **Phase-3 implication confirmed:** the LDM's decision input (live per-request acceptance →
  natural-K) is measurable in real time with zero vLLM source changes; objective = assign each
  request its natural-K (clipped to [1,Kmax]) under SLO/KV constraints. Full detail: `docs/phase2-results.md`.

### 2026-10-07 (cont.) — Phase 3: causal LDM recovers ~85% of the oracle gap
- **`experiments/ldm_eval.py`** — first concrete quantification of the LDM idea. Two measured
  ingredients (no assumed cost model):
  - **Measured per-step verify-cost c(k)** from Phase-1 (throughput÷accept-len → steps/sec):
    `c(1)=1.00 … c(8)=3.41` — **sub-linear** (vLLM verifies all K drafts in ~one target fwd pass).
  - **Causal LDM policy**: replay each request's accept stream; at each step pick k from an online
    sliding-window estimate of its joint-acceptance profile, maximizing est-captured/c(k). Never sees
    the full profile — only observed outcomes. All policies scored on one metric (produced tokens /
    verify-cost) so the oracle is a valid upper bound.
- **★ Result (ShareGPT per-request, measured c(k)):** best fixed-K value 1.730; **oracle +12.1%**;
  causal LDM recovers **59% (W=4) → 83% (W=8) → 89% (W=32)** of the oracle gap, i.e. **~10% throughput**
  from per-request K with a simple online estimator (no perfect knowledge). Residual = estimation error,
  not policy limitation — converges to oracle as W grows.
- **Interpretation:** Phase 2 proved requests want different K; Phase 3 proves a *practical* controller
  captures most of it. This is the honest value number for the LDM idea. The hand-rolled estimator is
  the skeleton: replace with a learned model ingesting queue/KV/GPU/SLO → generalizes to admission +
  chunk-size (full LDM objective).
- **Caveats logged:** c(k) measured on random workload, applied to ShareGPT (assumes workload-invariant
  verify cost — true for fixed target model); replay approximates batch coupling; single ngram drafter.
- Full detail: `docs/phase3-results.md`. **Next:** in-loop prototype (set per-request K live via the
  audit's injection points, measure end-to-end throughput vs fixed-K).

### 2026-10-07 (cont.) — Variable-K execution is FEASIBLE with zero vLLM source changes
Key code-audit finding that unblocks end-to-end measurement: **vLLM already executes variable
per-request K.** The `SpecDecodeMetadata` builder (gpu_model_runner.py:2586) explicitly handles
non-uniform draft counts — its docstring shows `num_draft_tokens: [3, 0, 2, 0, 1]` — and FlashAttn
uses `query_start_loc` (variable query lengths). The ngram proposer is CPU-based. So the worker can
verify a *different* K per request; the only missing piece is **truncating each request's drafts to
its LDM-chosen k\***, done in one monkey-patch on `Scheduler.update_draft_token_ids`.

- **`experiments/ldm_controller/sitecustomize.py`** third patch (gated by `VLLM_LDM_ENFORCE=1`):
  after drafts are stored, truncate `request.spec_token_ids[:k*]`. A k\* set at step t applies at t+1
  (outputs processed before next-step draft storage) — causal. **Validated live:** server survives
  variable-K decode on a mixed workload. Variable-K decode is non-uniform → requires `enforce_eager`
  (CUDA graphs need uniform decode); added an `ENFORCE_EAGER` knob to `run_server.sh`.
- **`experiments/mixed_workload_client.py`** — sustained Poisson mixed workload (50% templated
  high-ngram-acceptance + 50% diverse low-acceptance), reports output throughput.
- **`experiments/ldm_ab.sh`** — end-to-end A/B: LDM(K=8+enforce) vs fixed-K=4 vs fixed-K=8, same
  workload, eager mode. Running; results → next update.

### 2026-10-07 (cont.) — A/B v1 result: INCONCLUSIVE + pre-registration of Priority-A grid
**A/B v1 (single run, concurrent 16, mixed workload, eager):** LDM 785.7 / fixed K4 773.9 /
fixed K8 793.6 tok/s → **INCONCLUSIVE** (LDM between baselines, no variability estimate). NOT a
refutation, not a win. Decision-log analysis: k* bimodal {1: 4820, 8: 2159 steps}; high-acceptance
requests tracked well (k*=7.65), low-acceptance ones lag at k*=3.73 because ngram acceptance is
spiky (lag-1 autocorrelation ≈ 0) → W=8 window can't predict regime switches. Full analysis:
`docs/phase3-results.md` §A/B v1; candidate explanations for replay→live ranking disagreement in
**`docs/replay-vs-live-gap.md`** (E1 batch coupling, E2 cost-model workload mismatch, E3 estimator
non-stationarity, E4 ragged-eager overhead — each with an isolation experiment).

**Pre-registration — Priority A: rigorous live adaptive-K vs fixed-K grid**
- **Question:** does adaptive per-request K outperform the best fixed K in live serving?
- **Hypothesis H1:** adaptive K ≥ best fixed K (replay suggests up to ~+10%; live may be less).
  **Falsification condition:** adaptive K < best fixed K beyond run-to-run variability, OR
  best fixed K = AR (speculation net-negative on this workload).
- **Configs (all eager, same model/workload/concurrency):** `ar` (no SD), `k1`, `k2`, `k4`, `k8`,
  `ldm` (K=8 server + per-request truncation).
- **IV:** config. **DV:** output tok/s, request throughput, TTFT mean/p50/p95/p99, TPOT
  mean/p50/p95/p99, E2E latency percentiles, mean accepted spec length, acceptance rate, chosen-K
  distribution (ldm), GPU util, preemptions. **Controls:** Qwen2.5-7B-Instruct, ngram drafter,
  concurrency=16, mixed workload seed 1234, max_tokens=256, 150s window / 20s warmup excluded,
  same GPU/port/chunking, server restart between configs.
- **Trials:** 3 per config (fresh server each). Report mean ± std; classify SUPPORT/FALSIFY/
  INCONCLUSIVE using overlap of trial ranges.
- **VALID conditions:** all trials complete; steady window fully inside run; no OOM/preemption in
  logs; ldm truncation effective (decision log present); ar config has zero spec tokens.
- **INVALID if:** any trial <80% of expected completions, server restart mid-trial, or workload
  drift (different seed/prompt mix).

**Pre-registration — Priority B isolation E1 (batch coupling):** concurrency sweep C ∈ {1,4,16,32},
K=2 vs K=8, 2 trials each. **Discriminating outcome:** if K=8's advantage over K=2 grows with C and
vanishes at C=1 → batch-level verification coupling explains the live-vs-replay ranking flip. At
C=1 the per-request ranking should match replay's (cross-check of c(k)).

**Pre-registration — Priority C: model-based speculator reproduction (draft_model)**
- **Question:** is heterogeneous profitable speculation depth a general serving phenomenon, or an
  ngram artifact?
- **Setup:** target Qwen2.5-7B-Instruct + draft **Qwen2.5-0.5B-Instruct** (same family/vocab,
  `method=draft_model` — the officially supported model-based path in vLLM 0.19.1; EAGLE-Qwen
  checkpoints remain incompatible). K ∈ {2,4,8}, eager mode for comparability with the ngram grid.
- **Measurements:** (a) fixed-K sweep on the mixed workload (does an interior/best-K exist?);
  (b) per-request acceptance heterogeneity via `spec_hook` (natural-K span + CV vs ngram's 1.18–8.82,
  CV≈0.51); (c) adaptive-K live comparison if (a)+(b) show meaningful headroom.
- **Falsification of the general-phenomenon claim:** if draft-model per-request natural-K CV ≪ ngram's
  AND fixed-K ranking is flat → heterogeneity was an ngram artifact; pivot the story to
  drafter-dependent speculation depth.
- **VALID conditions:** draft model loads (no arch errors); acceptance length >1 on templated
  prompts (sanity: drafting actually works); same concurrency/window/warmup as Priority A.

### 2026-10-07 (cont.) — PA grid run #1 classified INVALID (metric bug) + re-run with fixed client
**Bug found during validation (charter Step 5):** `latency_client.py` counted **SSE chunks** as
tokens. vLLM emits one chunk per *decode step*, so speculative configs' token counts were
undercounted by ~their acceptance length (k1 ×~1.9, k4 ×~3.7, k8 ×~5+), while AR was unaffected
(1 token/chunk). Consequence: run #1's throughput "AR 756 > k1 655 ≈ k2 660 < k4 666" is **INVALID**
as a throughput ranking (it measured steps/s, not tokens/s — and made SD look *worse* than AR, the
opposite of Phase-1's established result). TPOT from run #1 is likewise invalid (divided by chunk
count). TTFT, GPU util, and chosen-K distributions remain valid.
**Fix:** client now counts real completion tokens via `stream_options.include_usage` (authoritative
`usage.completion_tokens` in the final chunk) with a text-length fallback; smoke-verified against
A/B v1's independently-measured 773.9 tok/s (K=4, C=16, eager) before re-running.
**Run #1 preserved:** `results/pa_grid_run1_invalid/` + `logs/pa_grid.log` (not overwritten).
**Re-run:** PA grid run #2 with the fixed client (same pre-registration; results supersede run #1's
throughput/TPOT only).
