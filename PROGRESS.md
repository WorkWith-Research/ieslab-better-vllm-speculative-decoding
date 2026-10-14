# Progress Log — [iESLAB] Better vLLM Speculative Decoding

GitHub Project: https://github.com/orgs/WorkWith-Research/projects/1 (projectV2 id `PVT_kwDOEyEOMs4Bl95-`)

## Project items (for API updates)

| Item ID | Issue | Title | Status |
|---|---|---|---|
| `PVTI_lADOEyEOMs4Bl95-zg-9nck` | #1 | Phase 0: Environment setup | Done |
| `PVTI_lADOEyEOMs4Bl95-zg-9nc4` | #2 | Phase 1: Observation — fixed-K SD under load | Done |
| `PVTI_lADOEyEOMs4Bl95-zg-9neU` | #3 | Phase 2: Analysis — oracle gap for dynamic K | Done |
| `PVTI_lADOEyEOMs4Bl95-zg-9nfY` | #4 | Phase 3: Prototype — per-request dynamic-K controller + live validation | Done |
| `PVTI_lADOEyEOMs4Bl95-zg_JZ9c` | #10 | Phase 4: Realistic-workload validation — load, SPEED-Bench, heterogeneous ISL/OSL, DSpark | Done |
| `PVTI_lADOEyEOMs4Bl95-zhAIokA` | #28 | H-CLASS-1: Request-class-dependent speculation economics under realistic heterogeneous serving | Done |
| `PVTI_lADOEyEOMs4Bl95-zhAcqK4` | #29 | Phase 5: Content-based per-request acceptance predictor (Track A) | Done |
| `PVTI_lADOEyEOMs4Bl95-zhApxh4` | #30 | Phase 7 (capstone): request-aware SD scheduler from live serving state (load × traffic-class) | In progress |

**Current Phase-4 checkpoint:** CLOSED 2026-10-14 (see entry below). All six sub-phases complete with final preregistered verdicts: 4.1 no capacity saturation at C≤64; 4.2 H-load CONFIRMED (optimal K flips with load); 4.3 SPEED-Bench long-prefill second load axis; 4.4 heterogeneous ISL×OSL FALSIFY; 4.5 DSpark-rule vs LDM; 4.6 live-load-aware adaptive speculation FALSIFIED as specified (mechanism characterized: per-request greedy allocation + fixed spec-path overhead).

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
tokens. vLLM does NOT emit one chunk per token — it batches several tokens per chunk (measured on a
K=4 verify run: mean 0.854 chunks/token, range 0.22–0.99; more bundling at higher K). So speculative
configs' token counts were undercounted (≈15% at K=4, more at K=8), while AR was unaffected (its
chunks are also sub-token-rate but the relative error pattern differs); run #1's SD configs all
clustered near ~660 "chunks/s" regardless of K — a step/flush-rate quantity, not tokens/s. The
throughput numbers and TPOT from run #1 are **INVALID** (TPOT biased high by ~the same ratio for SD
configs; throughput within ~15% but imprecise). Note: run #1's "AR 756 > k4 666" does NOT by itself
contradict Phase-1 (different workload: mixed vs random-template) — it becomes a real finding only
once the fixed client confirms it. TTFT, GPU util, and chosen-K distributions remain valid.
**Fix:** client now counts real completion tokens via `stream_options.include_usage` (authoritative
`usage.completion_tokens` in the final chunk) with a text-length fallback; smoke-verified against
A/B v1's independently-measured 773.9 tok/s (K=4, C=16, eager) before re-running.
**Run #1 preserved:** `results/pa_grid_run1_invalid/` + `logs/pa_grid.log` (not overwritten).
**Re-run:** PA grid run #2 with the fixed client (same pre-registration; results supersede run #1's
throughput/TPOT only).

### 2026-10-07 (cont.) — PA grid run #2 (fixed client): H1 FALSIFY + supervisor-feedback phase opened
**Result (C=16, mixed workload, eager, 3 trials each):** AR 735.2±14.8 | K1 730.6±3.2 | K2 762.1±0.0 |
K4 792.3±1.8 | **K8 802.8±1.8** | LDM 782.4±2.4 tok/s. GPU util ~44% (SD arms) / 48% (AR).
**Classification: FALSIFY** — LDM < best fixed K=8 by 20.4 tok/s, beyond trial uncertainty (max sd 2.4).
LDM k* distribution is bimodal (k*=1: ~69%, k*=8: ~31% of decisions) — the W=8 acceptance window pins
decisions to extremes; lowering K for low-acceptance requests cost batch throughput, it did not help.
**Key context:** at C=16 the GPU is NOT saturated (44%), so this measures algorithmic policy quality in a
moderate-load regime only. ngram on this mixed workload: K=1 ≈ AR (speculation net-neutral), gains grow
monotonically with K up to 8.
**Headline change:** the Phase-1 claim "optimal K is set by acceptance regime, not load" is DOWNGRADED to a
working hypothesis — load was never actually varied to saturation in any experiment so far.

**Supervisor feedback (2026-10-07) — now highest priority:** (1) is DSpark compared? (2) is the GPU load
sufficient? (3) realistic multi-request env? (4) heterogeneous prefill/decode sizes? (5) use ORCA/Sarathi/
SPEED-Bench workloads. Tracked as issue #10 / project item `PVTI_lADOEyEOMs4Bl95-zg_JZ9c` (In progress).
Discussion #11 posted with this result + plan.

**Pre-registration — Phase 4.1: serving saturation characterization (AR baseline, no SD)**
- **Question:** where does Qwen2.5-7B on 1x RTX 3090 (eager) transition from load-proportional to saturated
  serving? Which concurrency gives compute/capacity saturation?
- **Setup:** SPEC_METHOD=none, C ∈ {1,2,4,8,16,32,48,64}, 120s each (C≤8: 90s), warmup 20s, same mixed
  workload + seed as PA grid. MAX_NUM_SEQS=64. One trial per C (characterization, not hypothesis test).
- **Measurements:** steady throughput, TTFT/TPOT percentiles, running/waiting queue (scraper), KV-cache
  util max, GPU util, preemptions, achieved concurrency.
- **Regime definition (pre-specified):** LOW = throughput ∝ C and TPOT flat; TRANSITION = TPOT or TTFT
  growing >2× baseline while throughput still rising; SATURATED = throughput flat/declining AND waiting
  queue >0 sustained (or preemptions >0). Report the boundary concurrencies.
- **Falsification of "our experiments were unsaturated":** if C=16 already shows TPOT growth >2× vs C=4,
  then PA grid was in transition regime and K×load effects may have been partly visible — revisit.

**Pre-registration — Phase 4.2: fixed-K × load matrix (same workload as PA grid)**
- **Question:** on the SAME mixed workload, does the globally optimal fixed K shift across low / medium /
  saturated load? (Direct test of DSpark-style load-dependent K motivation.)
- **Setup:** K ∈ {none(AR),1,2,4,8} × C ∈ {C_low, C_med, C_sat} — concurrencies chosen from P1 results
  (pre-specified rule: C_low = lowest C with TPOT within 1.2× of C=1; C_med = midpoint of transition;
  C_sat = first saturated C). 3 trials per cell, eager, mixed workload seed 1234. 5×3×3 = 45 runs (~9h);
  if P1 shows saturation only at very high C, drop one K level (K=2) to fit time — decision recorded here
  before execution: prefer dropping K=2 over shortening trials.
- **Discriminating outcome:** SUPPORT for load-dependence = argmax_K changes between C_low and C_sat AND
  the gap between best-K at each load vs its worst-K exceeds trial uncertainty. FALSIFY = same K wins at
  all three loads (within uncertainty) → load is not a material variable for ngram on this workload, and
  DSpark-style load-driven K has no measured basis here.
- **VALID conditions:** per-cell achieved concurrency within ±15% of target; no OOM/preemption beyond the
  saturated cell's expected amount (recorded); steady-state window ≥60s.

### 2026-10-07 (cont.) — Phase 4.1 result: NO capacity saturation at C≤64 + GPU-util measurement correction
**Correction:** PA grid run #2's `gpu_util_mean` (~44%) is an ARTIFACT — the sampler omitted `--id`, so it
averaged in the idle GPU1. Raw logs show GPU0 at 92–100% kernel occupancy at C=16 (AR 100%, SD arms 92%).
PA run #2's throughput/latency/K-dist numbers remain valid; only its GPU column is not citable. Sampler fixed.
**Phase 4.1 sweep (AR, eager, C=1→64):** throughput perfectly linear in C (51→2621 tok/s; per-request 41–51 tok/s),
waiting queue = 0 at all C (MAX_NUM_SEQS=64), preemptions 0, KV ≤14%, TPOT stable 20→24ms (+20%), TTFT grows
linearly (prefill queuing does not hurt decode — chunked-prefill effect). **Conclusion: this workload
(ISL~50 tok, OSL 256) has no saturated regime up to C=64** — the PA grid ran in a compute-saturated but
capacity-unsaturated regime. Lesson: nvidia-smi GPU util (100%) ≠ serving saturation; queue depth + achieved
concurrency are the operative signals.
**Phase 4.1b extension running:** C=96/128 with MAX_NUM_SEQS=128. If still linear, saturation must be induced via longer
sequences (KV residency) → SPEED-Bench throughput split (1k–32k ISL) becomes the Phase 4.2 workload — long prefills
create real token-budget contention and queue pressure.

**Pre-registration — Phase 4.2: fixed-K × load matrix on a workload that can actually saturate**
- **Workload decision rule (pre-specified):** if P1b shows no saturation at C≤128 on the short mixed workload,
  Phase 4.2 uses **SPEED-Bench throughput_2k** (prompts padded/truncated to 2k ISL — official construction) so KV
  residency creates real capacity limits; OSL=256. If P1b DOES saturate the short workload, Phase 4.2 uses it.
  Recorded here before execution either way.
- **Cells:** K ∈ {none(AR), 1, 2, 4, 8} × C ∈ {C_low, C_med, C_sat} where (from Phase 4.1/4.1b): C_low = first C with
  per-request TPS within 10% of the C=1 value (expected ~8–16); C_med = midpoint between C_low and first
  saturated C; C_sat = first C with waiting_p50 > 0 sustained or throughput sub-linear (<90% of linear
  extrapolation). If no C_sat exists at MAX_NUM_SEQS=128, use C=128 as the highest-load cell and label it
  "highest feasible" (not saturated) — recorded deviation.
- **Trials:** 3 per cell, fresh server, eager mode all arms, mixed-seed fixed (1234), warmup 20s, duration 150s.
  5 K × 3 C × 3 = 45 runs ≈ 9h → run over ~2 GPU-hours per night; if time-constrained, drop K=2 first
  (pre-specified, not result-driven).
- **Hypothesis:** H-load: argmax_K(throughput) changes between C_low and C_sat. Mechanism candidate: at low
  load each decode step is memory-bound → extra verification tokens are nearly free → large K wins; under
  saturation the batched-token budget (2048) is contested by prefills + verification, so marginal K tokens
  displace prefill work and/or lengthen steps → smaller K or even AR wins.
- **Falsification:** same argmax_K at all three loads within trial uncertainty → load does not move the optimum
  for ngram on this hardware/workload; DSpark-style load-driven K has no measured basis here (important negative).
- **Discriminating vs PA grid:** PA grid was single-load (C=16, capacity-unsaturated). Phase 4.2 varies the regime.
- **VALID conditions:** achieved concurrency within ±15% of target (scraper running_p50); steady window ≥60s;
  no OOM beyond expected preemptions at C_sat (recorded per cell); K applied (server log num_speculative_tokens).

**PENDING (blocked, not yet posted):** Discussion #11 comment with the GPU-util correction + P1 saturation
table is drafted at scratch (`disc11_comment.md`); the GraphQL post hit an approval timeout and must not be
retried in the same turn. Post it on the next opportunity (content unchanged).

**Pre-registration — Phase 4.3: SPEED-Bench throughput_2k (real 2k-token prefills) K × load matrix**
- **Motivation:** supervisor feedback — requests with diverse prefill/decode sizes. The mixed workload has
  ISL~50 tok, so prefill never contends with verification for the 2048-token budget. SPEED-Bench throughput_2k
  (official construction: prompts padded/truncated to 2k tokens, 1536 prompts, 3 difficulty tiers) gives every
  request a ~2048-token prefill → under load, prefills and SD verification compete for the chunked-prefill budget.
- **Hypothesis (H-S2/H-S3):** at C_sat with long prefills, large K inflates TTFT disproportionately (verification
  tokens crowd out prefill chunks within the 2048 budget) while TPOT gains shrink → argmax_K shifts to smaller K
  or AR relative to the short-workload P2 result. Falsified if the K ranking at C=96 is unchanged by ISL.
- **Cells:** K ∈ {none,1,2,4,8} × C ∈ {32, 96} × 3 trials = 30 cells (dual-GPU, ~80 min). C=8 omitted: at 2k ISL,
  C=8 already has substantial prefill work; low-load regime covered by P2. If P2 shows argmax_K SHIFTING with load,
  add C=16 row (15 cells) to resolve the transition — decision made after P2 analysis, recorded here.
- **Metrics:** throughput + TTFT/TPOT/E2E percentiles per category and difficulty tier (SPEED-Bench client emits
  both), plus waiting/KV/GPU from scraper. Official SPEED-Bench metric = output tokens/sec; we report it plus the
  latency breakdown (theirs doesn't separate prefill/decode effects).
- **VALID conditions:** same as P2 (achieved concurrency ±15%, steady window ≥60s, K applied per server log).

**Phase 4.2 caveat (recorded during run):** the mixed workload is fully deterministic (fixed seed 1234 + greedy
decoding), so identical trials produce nearly identical token counts (verified: K=none C=8 trials t1/t2/t3 all
51200 steady tokens, 393.8 tok/s; raw per-request timestamps differ). Trial variance therefore measures
scheduling/queueing jitter only, not workload sampling noise. This is acceptable for ranking K (the workload
mix is identical across arms — the comparison is controlled), but the ±sd reported by analyze_p2.py understates
total uncertainty. SPEED-Bench cells (Phase 4.3) use 1536 distinct prompts → real per-trial variation. If P2 shows a
near-tie between adjacent K values at any C, that tie is NOT evidence of equivalence — resolve with P3 or a
seed-varied repeat before drawing conclusions.

### 2026-10-08 — Phase 4.2 grid COMPLETE (45/45) + H-load verdict: **SUPPORT**
**Result (`docs/p2-load-matrix-results.md`, pivot in `results/p2/summary.csv`):** argmax_K flips with load.

| C | argmax_K | best tok/s | SD(K=8) vs AR |
|---|---|---|---|
| 8 | K=8 | 434.5 | **+10.3%** |
| 32 | K=8 | 1464.5 | +1.0% |
| 96 | **none (AR)** | 2457.6 | **−20.8%** |

- At low/medium load larger K wins; at the highest load **speculation is net-negative** and the loss grows
  with K (K=1 −2.7% … K=8 −20.8% vs AR). The DSpark-style load-driven-K effect is confirmed on ngram.
- **Verdict: SUPPORT (H-load)** — argmax changes C_low→C_high AND best-vs-worst gap at each C (13–21%) ≫ trial
  sd (≤30.8 tok/s ≈ 1.6%). Falsification not triggered.
- **Recorded deviation:** C=96 shows waiting_max=0, KV ≤21%, no preemptions → it is the *highest feasible*
  load on this short-ISL workload, i.e. **compute/occupancy**-saturated, NOT capacity-saturated (no sustained
  queue). The flip is a compute-occupancy phenomenon; behavior under true capacity saturation (long prefills)
  is the Phase 4.3 question (SPEED-Bench throughput_2k, already pre-registered above).
- **Validity all met:** achieved concurrency within ±15% (running_p50 = 8/32/96 exact); K applied per server
  log (`num_speculative_tokens` none/1/2/4/8 verified); no OOM/preemption; steady window ≥60s.
- **Caveat:** deterministic workload → sd understates total uncertainty (jitter only). C=96 K=4 sd=30.8 is one
  outlier trial (1927.9/1904.2/1965.3); ranking unaffected.

**Next (per pre-registration, not auto-launched):** Phase 4.3 = SPEED-Bench throughput_2k K×C matrix to test whether the
argmax flip persists/strengthens under long-prefill capacity saturation. Awaiting explicit go-ahead before launch.

### 2026-10-08 — Phase 4.2 COMPLETE: H-load CONFIRMED (optimal K flips with load)
45/45 cells done (K∈{none,1,2,4,8} × C∈{8,32,96} × 3 trials, mixed workload, eager).
- C=8 (memory-bound): **K=8 best (+10.3% vs AR)**, monotone in K — verification tokens nearly free.
- C=32 (transition): K=8 best by only +1.0% (near noise floor) — SD advantage nearly gone.
- C=96 (compute-saturated): **AR beats every SD arm by 14–21%**; ranking INVERTED vs C=8.
Mechanism: acceptance rate load-invariant (0.63–0.85); steps/s falls ~38% (AR→K=8 at C=96) because each step
verifies B=C(1+K̄) tokens and step time grows with batched tokens in the saturated regime. TPOT 36.8→56.3ms.
SPS(B) profiled curve reconstructed (45.5 → 16.6 steps/s over B=32→130): results/p2/sps_table.jsonl.
Largest single effect in the project so far: at C=96, "best fixed-K" loses 21% to oracle-per-load K (=AR).
Caveats: deterministic workload (variance = jitter only), unexplained TTFT anomaly (SD arms LOWER TTFT than AR
at C=96 — flagged, not cited), ngram-specific magnitudes, eager mode. Full write-up: docs/phase4-p2-results.md.
**Phase 4.3 launched next (supervisor priority): SPEED-Bench throughput_2k (2k-token prefills) K × load matrix.**

**Pre-registration — Phase 4.5: DSpark-rule baseline vs LDM vs fixed-K (in-loop, same workload/load as Phase 4.2/4.3)**
- **Arms:** AR (K=0 all), fixed K=4, fixed K=8, **LDM** (acceptance-only, local EMA window W=8, measured
  cost curve c(k)), **DSpark-rule** (batch-level global greedy over empirical prefix-survival J_r[j] with the
  hardware-profiled SPS(B) table from P2; objective Θ=τ·SPS(B)). All in-loop via sitecustomize monkey-patch
  (run_server.sh CONTROLLER=ldm|dspark), eager mode, KMAX=8.
- **Faithfulness notes (documented deviations):** DSpark uses a trained confidence head (ECE~1%) — we use
  empirical per-position acceptance over the last W steps (causal). DSpark's SPS(B) is profiled offline — ours
  too (Phase 4.2 table), same spirit. DSpark's production system has no "SD off" outcome; P2 showed AR wins at C=96, so
  we EXTEND both policies with an explicit all-K=0 fallback: after the greedy/EMA decision, if Θ(K=0 for all)
  ≥ Θ(chosen allocation), take K=0 (recorded as a deviation, pre-specified). This is what makes "SD off"
  reachable and is exactly the regime P2 says matters.
- **Cells:** C ∈ {32, 96} × arm ∈ {AR, K4, K8, LDM, DSpark-rule} × 3 trials = 30 cells, dual-GPU, mixed workload
  (Phase 4.2 workload) first; if time permits repeat on SPEED-Bench 2k (Phase 4.3 workload). AR/K4/K8 at these C already exist
  from P2 — reuse those numbers, run only LDM + DSpark-rule (12 new cells ≈ 90 min dual-GPU).
- **Hypothesis:** H-D: DSpark-rule ≥ best fixed-K at C=32 (its SPS(B) term should down-weight K as B grows), and
  DSpark-rule < AR-oracle gap-closer at C=96 because per-request confidence keeps some requests at high K even
  when the batch is saturated. LDM vs DSpark-rule: if LDM ≥ DSpark-rule without any load signal, the local
  acceptance-only policy already suffices (weakens the serving-state motivation); if DSpark-rule > LDM, batch
  coupling matters and our contribution must add live serving state on top.
- **Falsification of "DSpark already solves it":** any arm using live serving state (queue/KV/prefill) beating
  DSpark-rule by > trial noise at C=96. Conversely, if DSpark-rule ≈ AR within noise at C=96 and ≥ everything
  else everywhere, our LDM idea has no measured edge on this hardware/workload — report that plainly.
- **VALID conditions:** controller active line in server log ([dspark]/[ldm] banner); decision JSONL non-empty;
  achieved concurrency within ±15% of target; K distribution from decision logs reported (not just throughput).

### 2026-10-08 — Phase 4.3 COMPLETE: SPEED-Bench 2k (long prefills) reveals a SECOND load axis + per-request structure
30/30 cells (K∈{none,1,2,4,8} × C∈{32,96} × 3 trials, SPEED-Bench throughput_2k, ISL p50=1986/max 3980, pre-tokenized).
- **C=32 (moderate load, heavy prefill stream): AR wins by up to +13% over K=8.** ~2k prefills arriving continuously →
  SD verification tokens crowd out prefill chunks in the 2048 budget → prefills finish slower → fewer reach decode.
  This is the H-S2/H-S3 prefill-interference effect. Contrast P2 (ISL~50) at SAME C=32: K=8 was +1.0% vs AR.
  Same concurrency, different ISL → OPPOSITE ranking.
- **C=96 (queue-saturated, TTFT p50 ~15s):** throughput pinned ~610 tok/s regardless of K; K=8 vs AR = +0.7% (null).
- **Per-category split (the LDM finding):** at C=96, K=8 cuts low_entropy TPOT by 57% (38 vs 88ms) but inflates
  high_entropy TPOT by +32–60% (117 vs 88ms). Aggregate hides it; a uniform-K policy is wrong for ≥half the traffic.
- **Two load axes:** C (P2) and ISL (P3) independently move argmax_K → a single "GPU load" scalar (DSpark's SPS(B))
  is insufficient; need both concurrency AND prefill-intensity signals. Full write-up: docs/phase4-p3-results.md.
**Next: Phase 4.5 — DSpark-rule vs LDM in-loop (directly answers supervisor's "are you using DSpark?").**

### 2026-10-08 — Phase 4.5 COMPLETE: DSpark-rule baseline vs LDM
12/12 cells (LDM + DSpark-rule arms; AR/K4/K8 reused from Phase 4.2), mixed workload, C∈{32,96}×3.
- **C=32:** all arms within ~1% (SD neutral at transition — as Phase 4.2 predicted).
- **C=96 (saturated): AR 2457.6 wins by 20–25%. DSpark-rule is the WORST arm (1853.1, −24.6%)** — worse than
  fixed K8 (1947.6). LDM (1960.1) best adaptive but still −20.2% vs AR.
- **Mechanism (decision logs):** DSpark-rule turns SD off for ~30% of requests (its SPS(B) term works) but keeps
  the rest at high K (mean 2.77) — its Θ=τ·SPS(B) objective OVER-credits speculation because τ uses empirical
  acceptance (high on this repetitive workload) while the profiled SPS(B) curve (taken at uniform B) underestimates
  the true step-time inflation from per-request variable K. LDM never commits to SD-off (frac K=0 = 1.2%) — it is
  acceptance-only and has no saturation notion.
- **Interpretation:** the load-driven-K premise is real (Phase 4.2) but DSpark's decision rule does NOT capture it;
  "just adopt DSpark" is measured-rebutted. The missing ingredient = a LIVE load-regime signal (measured steps/s,
  TPOT trend, or B vs SPS(B) knee) that neither baseline consumes → the actual LDM contribution must augment
  acceptance with serving state so it can commit to SD-off at saturation. Phase 4.5 motivates Phase 4.6 (live-load-aware adaptive speculation),
  which will test whether explicit live load awareness can recover the 20–25% gap. Full write-up: docs/phase4-p5-results.md.
**Phase 4 status: 4.1✓ 4.2✓ 4.3✓ 4.5✓; 4.4 heterogeneous ISL×OSL remains. Next controller experiment: Phase 4.6 live-load-aware adaptive speculation. Phase 5 is reserved for scheduler-level design after Phase 4 closes.**


### 2026-10-08 — Phase numbering / current-state normalization
- Canonical Phase-4 subphase names are **Phase 4.1–4.6**; older P1/P2/P3/P5 labels in historical logs and result paths are retained only for provenance/backward compatibility.
- **Complete:** Phase 4.1 saturation characterization; Phase 4.2 fixed-K × load; Phase 4.3 SPEED-Bench long-prefill; Phase 4.5 DSpark-rule vs LDM.
- **Outstanding:** Phase 4.4 heterogeneous ISL × OSL mixed workload.
- **Next controller experiment:** Phase 4.6 live-load-aware adaptive speculation (request acceptance + live serving-state/load regime → K including SD-off).
- **Phase 5 is not started.** It is reserved for speculation-aware scheduler design/implementation if Phase 4 shows a scheduler-level gap after the remaining validation.

### 2026-10-08 — Pre-registration — Phase 4.4: heterogeneous ISL×OSL mixed workload (Priority A)
**Supervisor ask:** "다양한 크기의 prefill/decode token을 처리하는 request들을 만들어 두고 실험" (ORCA/Sarathi-style classes).
**Workload `hetero`** — 4 request classes, equal weight (25% each), deterministic seed 1234:
| class | ISL (tok) | OSL cap (tok) | grounded pattern |
|---|---|---|---|
| SS | ~50 | 256 | chat turn (P2-mixed-like) |
| SL | ~50 | 1024 | short prompt / long generation (agent continuation) |
| LS | ~2048 | 256 | RAG: long context, short answer |
| LL | ~2048 | 1024 | document QA with long response |
Prompts generated by truncating/padding a fixed corpus to target ISL (±5%); OSL enforced via per-request max_tokens.
**Cells:** K ∈ {none(AR),1,2,4,8} × C ∈ {32,96} × 3 trials = 30 cells, DUR=150s, eager, same server config as Phase 4.2.
GPU split: GPU0 K∈{none,1,2}, GPU1 K∈{4,8}. Results → results/p4_4/ (tag p44_k{K}_c{C}_t{T}_hetero).
**Hypotheses + falsification conditions (fixed before measurement):**
- **H-4.4a (per-class interaction):** at C=96, argmax_K differs ACROSS classes — specifically SL (short-prefill/long-decode)
  tolerates larger K than LS/LL (long-prefill). FALSIFY if all four classes share the same argmax_K at C=96.
- **H-4.4b (ISL scales SD harm):** TPOT degradation of K=8 vs AR grows with ISL within a cell (SS ≤ SL < LS ≤ LL).
  FALSIFY if no monotone trend across classes at C=96.
- **Validity:** per-class metrics from client-side token accounting (TTFT/TPOT/throughput per class); ≥10 completed requests
  per class per cell; server waiting/KV/preemption logged; K verified server-side (spec counters).
- **Interpretation guard:** aggregate throughput may tie while per-class SLO harm diverges (Phase 4.3 lesson) — report both.

### 2026-10-08 — Pre-registration — Phase 4.6: live-load-aware adaptive speculation controller (Priority B)
**Hypothesis (H-4.6):** a minimal CAUSAL controller combining per-request acceptance with a LIVE load-regime signal can
(i) use large K when speculation is cheap and (ii) commit to SD-off (K=0) when the batch is compute-saturated — WITHOUT
being given the concurrency label C.
**Controller `ldm_load` (rule fixed BEFORE any run):**
- Live state: B_live = running requests + scheduled speculative tokens this step (measured in update_from_output;
  NO C anywhere in the controller code). S(B) = EMA (τ=5s) of achieved steps/s at current B.
- Per-request acceptance: J_r[l] = P(acc ≥ l) over last W=8 observed steps (causal).
- **Decision rule — REVISION #2 (pre-run, mechanical-flaw fix, charter §13):** the revision-#1 form
  J_r[k]·S(B_live) ≥ T is mechanically flawed: vLLM verifies all K drafts in ONE forward pass, so the step-rate
  cost of a request's K depends on the TOTAL batch size B, not per-position (the naive per-position marginal test
  double-counts cost and was shown offline to return k*=0 even at C=8). Corrected rule, per request r, each step:
    N = #running requests;  B_live = Σ scheduled tokens this step (both measured in update_from_output);
    SPS(B) = static profile fitted to Phase-4.2 measured points (AR path for k=0, SD path otherwise);
    k*_r = argmax_{k∈{0..8}}  N · (1 + Σ_{l≤k} J_r[l]) · SPS(B_live + (k − k_cur)·N),
  where k_cur is r's K last step and J_r[l] = P(acc≥l) over the request's last W=8 steps (causal; global-prior
  fallback for cold start, then k*=4 until first observation). No C anywhere in the code — only measured
  (N, B_live) + the static profile. If all requests have identical J this reduces to the batch-level argmax, so
  per-request and global decisions coincide on uniform workloads and diverge only via measured acceptance
  heterogeneity (the LDM differentiator).
- **Pre-run calibration (MANDATORY before launch, output committed):** run `experiments/p46_calibrate.py` on Phase 4.2
  uniform-K cells to tabulate the rule's implied K* at each (C, measured B, measured SPS, measured J). The rule is
  LAUNCHED AS-SPECIFIED regardless of calibration outcome; if calibration shows it cannot separate C=32 from C=96,
  that is recorded and the run proceeds to FALSIFY or SUPPORT accordingly (no post-hoc retuning; ≤1 further revision
  allowed only if the rule crashes/misfires mechanically, per charter §13).
**Revision #3 (pre-run, mechanical-flaw fix #2):** revision #2's value model used a uniform-batch delta
`(k-k_cur)*N` and an `a_bar` computed over ALL pending requests. Two flaws found during smoke testing:
(1) ngram self-drafting produces far fewer ACTUAL drafts than the instructed K on most steps (measured
mean d≈2.6/req at C=8), so the batch-size delta must use measured per-request draft count d_r, not k-k_cur;
(2) `a_bar` (batch mean accepted/request-step) included zero-draft requests and omitted the +1 bonus token
each drafted request emits every step — this drove a self-fulfilling K=0 collapse in the C=8 smoke test
(meanK 0.52 where fixed-K8 wins +10.3%). Corrected rule (rev #3): k*_r = argmax_k [(N-1)(1+a_bar) + 1 +
Σ_{l≤k}J_r[l]] · SPS_path(B_live + (k - d_r)), a_bar over DRAFTED requests only incl. the bonus token.
**INVALID cells:** 4 ldmload cells run under rev #2 (c8 t0/t1/t2, c32 t0) are quarantined in
results/p4_6/invalid_rev2/ and must not be reused. The SPS profile is extended with SD K=4/8 at C=96
(B≈110-130) so the cost model covers the saturation regime (initial profile only reached B≈86).

**Cells:** ldm_load × C ∈ {8,32,96} × 3 trials = 9 cells, mixed workload (Phase 4.2), DUR=150s, eager. Baselines reused:
AR/K4/K8/LDM/DSpark-rule from Phase 4.2/4.5 (no re-run). GPU1.
**Success (fixed):** at C=96 ldm_load ≥ AR − noise (noise = max trial spread of AR across trials, ≈0.3%) AND beats DSpark-rule;
at C=8 ldm_load ≥ +5% vs AR (recovers most of fixed-K8's +10.3%). **FALSIFY** if it fails either bound: then the load-regime
signal as specified does not carry enough information, and we report that (do NOT retune to win).
**Tuning budget:** ≤2 revisions total for this hypothesis (charter §13); each revision must be logged with rationale.
**Interpretation guard:** if ldm_load ≈ acceptance-only LDM at C=96 (i.e., the regime term never triggers SD-off), the
hypothesis is FALSIFIED as specified — a live SPS(B) ratio does not let this controller detect saturation.

### 2026-10-08 — Incident + Revision #4 — Phase 4.6 unified SPS curve; Phase 4.4 port-collision quarantine
**HARNESS INCIDENT (Phase 4.4, affects 13 cells):** killing orchestrator chains mid-run left ORPHANED `vllm serve`
processes holding ports 8100/8101 — vLLM 0.19.x spawns APIServer/EngineCore children that survive a bare
`kill <pid>` (the recorded pid is only the top-level process). Subsequent cells' own servers failed to bind, but
the health check passed against the orphan, so clients measured the WRONG server. Audit (startup-complete marker in
the cell's OWN log + PID match) invalidated 13/30 Phase-4.4 cells: all K=2 (6), K=1 c32 t0/t1/t2 + K=1 c96 t0/t1/t2 (6),
and knone c96 t2 (zero completions). Quarantined in results/p4_4/invalid_portcollision/. The earlier H-4.4a/b analysis
that used those columns is NOT a valid verdict. Fix: `experiments/kill_server.sh` (kills the whole process tree, waits
for the port to actually go quiet, SIGKILLs any remaining listener) wired into p44_cell.sh/p46_cell.sh, plus a startup
verification gate in both runners (abort cell as INVALID if our pid is dead or our log lacks "Application startup
complete"). All 13 cells re-running with the hardened harness.

**Revision #4 (pre-run, evidence-motivated per charter §13 — final revision for this hypothesis):** rev-#3 ran
cleanly on c32 t1/t2 and c96 ×3 after the orphan cleanup (archived in results/p4_6/rev3_two_regime/) but collapsed to
K=0 at ALL loads (meanK≈0.16 even at C=8, where fixed-K8 wins +10.3%). Offline reconstruction of the rev-#3 value
function from its OWN logged live state found the root cause: the TWO-regime SPS model double-penalized speculation.
The profile's "SD points" carry a small fixed per-step overhead of the spec-decode code path on top of their (larger) B,
so at equal B the SD curve sat 10–86% below the AR curve — an artifact of how uniform-K profile cells were constructed,
not a separate cost regime. In vLLM's continuous batch a k=0 request and a speculating request ride the SAME forward
pass; step time is set by TOTAL scheduled tokens B, and the spec-path overhead is constant across all k choices (the
server always runs with spec decode enabled), so it cancels in argmax over k. Rev-#4 rule (only change: SPS_path → ONE
unified curve fitted to all 19 measured profile points; everything else — measured N, B_live, d_r, drafted-only a_bar,
causal J_r, cold-start k*=4 — unchanged):
    k*_r = argmax_{k∈0..8} [(N-1)(1+a_bar) + 1 + Σ_{l≤k}J_r[l]] · SPS(B_live + (k − d_r))
Desk check on rev-#3's logged live state (no concurrency label anywhere): C=8 → k*=8 (robust to d_r=0..8),
C=32 → k*=0, C=96 (meanN=92.4, meanB_live=120.2) → k*=0 — matching Phase 4.2 ground truth at all three regimes.
**INVALID cells:** rev-#3 c8 ×3 + c32 t0 were run while an orphan held port 8100 (quarantined in
results/p4_6/invalid_rev3/); the clean rev-#3 runs (c32 t1/t2, c96 ×3) are archived in results/p4_6/rev3_two_regime/ —
both sets must not be reused in claims. Rev-#4 re-runs all 9 cells fresh. This is the FINAL revision for H-4.6: if rev-#4
fails its pre-registered bounds, the hypothesis is reported FALSIFIED as specified (charter §13).

**SPS profile complete:** 19 points — AR C=8..128 (B=8..128) + SD K=4/8 C=8..64 (B≈10..86) + SD K=4/8 C=96
(B≈124.2/130.5, SPS≈16.7/15.9; verified clean: own server startup=1, correct spec config). The two high-B SD points sit
slightly below the AR trend at equal B (the fixed overhead above) — consistent with a single underlying curve plus
constant overhead, which is exactly what rev-#4's unified fit absorbs.

**Phase 4.4 re-run plan (hardened harness):** K=1 ×6 + knone c96 t2 on GPU0; K=2 ×6 on GPU1 — then H-4.4a/b
re-analyzed on the full clean 30-cell set. Phase 4.6 rev-#4 cells (C=8/32/96 ×3) follow on the same GPUs after their
respective re-runs finish.

### 2026-10-08 — ENFORCEMENT BUG FOUND + FIXED — Phase 4.6 rev-#4b (mechanical fixes, rule unchanged)
The first rev-#4 cell run (9 cells, all startup-valid) was **INVALID** and quarantined in
`results/p4_6/rev4_unenforced/`. Two mechanical flaws found by auditing what was ACTUALLY scheduled
(`Kobs` in decision logs + spec-decode counters), not just what the controller decided:

1. **k*=0 was never enforced.** The per-step decision map was built only from DRAFTED requests
   (`make_spec_decoding_stats` never fires for zero-draft steps). A request set to k*=0 stopped being
   drafted, vanished from the map, and its next drafts fell back to the cold-start default (kstar=4) —
   i.e. "SD-off" oscillated into K≈4 one step later. Measured: C=96 chose k*=0 for 99.6% of steps yet
   still verified 415 draft tok/s (~3.7 drafts/req-step), reaching only ~2013 tok/s vs true AR's
   2457.6 (−18%) — the spec-decode path overhead was paid without any speculation benefit.
2. **Cold-start ceiling of K=4 made k*=8 unreachable.** The controller can never observe acceptance at
   positions >4, so its J estimate is capped and it can never choose k>4 — which cannot meet the
   pre-registered C=8 success bar (K8 = +10.3% vs AR; K4 = only +5.5%).

Fixes (both mechanical; the rev-#4 decision rule, unified SPS curve, and measured-state inputs are
UNCHANGED — this is not a new hypothesis revision per charter §13):
1. Decide for EVERY running request each step; decisions persist in `last_kstar`; requests without a
   live drafted history use their FROZEN acceptance history (`khist`), so k*=0 stays k*=0 unless the
   live load regime makes speculation profitable again (the regime term is re-evaluated every step).
2. Cold start KMAX instead of 4 (a pessimistic ceiling is self-fulfilling; same rationale as the Phase
   4.5 LDM cold-start fix). Transient cost at saturation bounded to each cell's first ~150 steps.

Smoke-verified before launch: k*=8 now reachable (C=8 steady meanK=4.52, distribution includes k=8;
throughput 438.9 vs AR 393.8 = +11.5%); debug log confirms spec_token_ids truncated to 0 for k*=0
requests (`sample(spec_len,kstar)` shows (0,0) entries). Rev-#4b re-runs all 9 cells (same names).

### 2026-10-08 — PLUMBING BUGS FOUND + FIXED — Phase 4.6 rev-#4c (mechanical fixes; rule byte-for-byte unchanged)
The rev-#4b run (9 cells, enforcement correct: draft/s 415→68 at C=96) is **INVALID** and quarantined in
`results/p4_6/rev4b_wrong_signal/`. Results were C=8 +6.1% / C=32 −9.1% / C=96 −9.0% vs AR, with the controller
oscillating at saturation (meanK swung 0↔3.6, fracK0 55–100%).

**Root cause (corrected after live instrumentation — see below):** the per-request draft count `d_r` used by the
value function's `(k − d_r)` delta carried **stale values from earlier steps** (the dict was never cleared), so a
request currently at k*=0 still showed its last drafted step's d≈8, inflating the value of further drafts. This is
what drove the oscillation.

**IMPORTANT CORRECTION to an earlier (wrong) narrative:** during diagnosis I hypothesized that B_live
(`total_num_scheduled_tokens`) EXCLUDED the spec drafts and re-derived a "708 tok/step" figure from those same stale
per-request counts. Live ground-truth instrumentation on vLLM 0.19.1 **falsified this**: `total_num_scheduled_tokens`
DOES include scheduled spec drafts (sampled step: N=96 with 8 drafts → total_sched=104; source: scheduler.py
num_new_tokens = num_tokens_with_spec + placeholders − computed). So rev-#4b's B_live was already the correct
verification-batch signal, and the "708 tok/step" number was an artifact of stale d values. The load-signal part of
the pre-registration was implemented correctly from the start; what broke was per-request state plumbing.

**Fixes (all mechanical; decision rule, SPS profile, action space unchanged — charter §13 misfire allowance):**
1. Per-request draft counts now live in a closure dict (`step_d`) populated by the stats hook and **cleared at the end
   of every update_from_output**, so d_r is strictly this-step (k*=0 requests see d_r=0). Instance-attribute handoff
   (`self._ll_pending`) was dropped: it was observed to read empty at decision time in this vLLM build, which is why
   two intermediate fix attempts (smoke cells ts1/ts2) silently no-op'd. Closure state is immune to that.
2. Enforcement (decide for every running request; persistent last_kstar; khist frozen history) and KMAX cold start
   from the rev-#4b fixes are retained.
3. Diagnostic env `VLLM_LDMLOAD_FORCE0` added: forces k*=0 for all requests, to measure the **SD-off floor** on a
   spec-enabled server (decomposes the C=96 gap into "controller still speculating" vs "fixed spec-path overhead").

Smoke cells ts1–ts5 (diagnostics only) are archived in `results/p4_6/smoke_diagnostics/`. rev-#4c re-runs all 9 cells
under the same names; classification against the pre-registered bounds follows.

**Interim measurements (while the grid runs):**
- **SD-off floor at C=96 (force k*=0 for all requests, spec path still enabled): 2437.6 gen/s vs AR 2557 = −4.7%.**
  Even with speculation fully OFF, a spec-enabled server sits ~4.7% below a spec-disabled one at saturation — the fixed
  cost of running the spec-decode code path (draft bookkeeping + synthetic rejection sampler) with zero drafts. No
  in-loop controller restricted to K∈{0..8} can eliminate it; reaching AR requires disabling the path itself
  (a scheduler-level action → Phase 5 territory). This makes the literal pre-registered bound "ldm_load ≥ AR − noise"
  structurally unreachable by a K-only controller, independent of how well the load signal works.
- **Clean-plumbing rev-#4c still oscillates at C=96** (smoke: meanK≈1.5, fracK0≈81%, gen/s≈2308 — WORSE than force-0's
  2437). The per-request greedy rule over-admits speculation at saturation because each request's decision to draft
  does not price the shared-batch slowdown its drafts impose on the other N−1 requests (an externality absent from the
  value function). Constant SD-off beats the adaptive rule in this regime. This is a mechanistic finding about
  per-request greedy allocation under compute saturation — exactly the "opportunity cost of speculative verification
  inside the scheduler" that motivates Phase 5 — not a further plumbing bug. Per charter §13 (revision budget), no more
  controller changes: the full 9-cell grid runs once with this harness and is classified as-is.
- **Client-metric caveat:** `steady_throughput_tok_s` = completions-in-window × 256 tokens; open-loop in-flight requests
  at window end are excluded, so it undercounts by an arm-dependent amount (AR/force-0/smokes all "coincidentally" read
  2457.6 while their true server rates differ: 2557 / 2437.6 / ~2350). Server-side generation-counter rate is the
  trustworthy cross-arm metric; `experiments/analyze_p46_server.py` reports both.

### 2026-10-08 — COMPLETE — Phase 4.4 heterogeneous ISL×OSL: H-4.4a FALSIFY, H-4.4b FALSIFY
All 30 cells re-ran clean on the hardened harness (own-server startup verified per cell; audit: 30/30 valid).
Full results in `docs/phase4-p44-results.md`. Headline numbers (steady-state aggregate tok/s, mean over trials):

| C | none (AR) | K=1 | K=2 | K=4 | K=8 | argmax |
|---|---|---|---|---|---|---|
| 32 | **1314.1** | 1022.0 | 1061.5 | 1156.5 | 1137.7 | K=none |
| 96 | **2009.9** | 1073.9 | 1228.7 | 1320.1 | 1307.4 | K=none |

- **H-4.4a FALSIFY:** all four ISL×OSL classes share argmax K=none at C=96 (and C=32) — class identity does
  not change the optimal K on this drafter/corpus.
- **H-4.4b FALSIFY:** TPOT p50 degradation of K8 vs AR at C=96 is large in every class (SS +64%, SL +39.5%,
  LS +67.5%, LL +42.4%) with no monotone ISL trend.
- Mechanism: ngram acceptance on ShareGPT-derived text decays steeply with K (acc_rate K=8: 0.38 at C=32,
  0.50 at C=96 — far below the Phase 4.2 mixed workload), so ΣJ never compensates the B-growth cost; K=1 is
  the worst arm (pays spec-path overhead for ~1 token). Same load-regime × draftability story as Phases 4.2/4.3,
  with class membership NOT a useful feature for K selection here. The per-class TPOT harm remains a real
  SLO-relevant effect.

### 2026-10-08 — COMPLETE — Phase 4.6 live-load-aware adaptive speculation: **H-4.6 FALSIFIED as specified** (C=8 PASS, C=96 FAIL)
Full results in `docs/phase4-p6-results.md`. rev-#4c grid complete (9 cells + force-0 diagnostics; enforcement
verified from decision logs: 99.5% of k*=0 decisions show zero scheduled drafts). Server-side generation rate
(steady t≥20s — the client metric is open-loop biased, see results doc):

| arm | C=8 (vs AR) | C=32 (vs AR) | C=96 (vs AR) |
|---|---|---|---|
| AR | 387.0 | 1460.0 | 2557.2 |
| K8 (best fixed) | 430 (+11.1%) | 1437 (−1.6%) | 1937 (−24.3%) |
| LDM (P4.5) | — | 1398 (−4.2%) | 1953 (−23.6%) |
| DSpark-rule (P4.5) | — | 1416 (−3.0%) | 1858 (−27.3%) |
| **ldm_load** | **417.0 (+7.8%) PASS** | **1304.2 (−10.7%)** | **2235.2 (−12.6%) FAIL** (bound ≈ AR−0.3%) |
| ldm_load force-0 (SD-off floor) | 354.4 (−8.4%) | 1336.9 (−8.5%) | 2437.6 (−4.7%) |

- **C=8 PASS** (+7.8% ≥ +5% bound; ~70% of K8's gain, without knowing C). **C=96 FAIL**: −12.6% vs the AR−noise
  bound; it does beat DSpark-rule by +20.2 pts (second clause passes) but the first fails → FALSIFIED as
  pre-registered, no retuning (charter §13).
- **The load signal WORKS; per-request greedy ALLOCATION does not.** At C=96 ldm_load puts 80% of requests at K=0
  (it recognizes saturation) but a ~20% minority stays permanently at K=8 and the batch oscillates (meanK* 0↔5.4,
  bimodal k* distribution). Each request's argmax prices its own accepted tokens against SPS(B) but NOT the
  slowdown its drafts impose on the other N−1 requests — a negative externality that grows with B. The force-0
  diagnostic proves the saturation optimum IS full SD-off (2437.6 > 2235.2), yet no per-request decision rule
  reaches it. DSpark-rule fails the same way (broad K dist, −27.3%); acceptance-only LDM worse (−23.6%).
- **The spec path has a fixed overhead no K-only controller can remove:** force-0 = −4.7% vs AR at C=96 even with
  zero drafts. So "ldm_load ≥ AR − noise" was structurally unreachable for any K∈{0..8} controller; closing the
  last ~5 pts requires disabling the spec path itself — a scheduler-level action.
- **Phase 5 is now motivated by measurement, not speculation:** the remaining saturation gap decomposes into
  ~8 pts of allocation loss (per-request greedy vs SD-off optimum) + ~4.7 pts fixed spec-path overhead → both
  point to batch-level K allocation with externality pricing and/or dynamic enable/disable of the spec path.

**Phase 4 status: 4.1✓ 4.2✓ 4.3✓ 4.4✓ 4.5✓ 4.6✓ (H-4.6 FALSIFIED, mechanism characterized). Phase 5 NOT STARTED
**Phase 5 status: NOT STARTED (reserved for speculation-aware scheduler design/implementation).**

### 2026-10-08 — Pre-registration — Phase 5: speculation-aware scheduler with batch-level SD on/off + draft-pass skip (H-5.1)

**Motivation (from Phase 4.6 measurement, not speculation):** the C=96 gap decomposes into
~8 pts of *allocation* loss (per-request greedy vs SD-off optimum) + ~4.7 pts of *fixed spec-path
overhead* (force-0 floor: a spec-enabled server with k*=0 for all requests still runs 2437.6 vs AR
2557.2). The second component is **structurally unreachable by any K-only controller** — it is the cost
of running the ngram GPU draft pass every step (scatter + matching kernel + CPU copy), which vLLM 0.19.1
runs unconditionally whenever `speculative_config` is set, even when zero drafts are scheduled (verified in
source: gpu_model_runner.py:4205-4249 gates the draft pass on `spec_config is not None`, NOT on whether any
drafts were scheduled). Closing it requires a *scheduler-level* action: dynamically disabling the spec path
itself, not just setting K=0.

**The externality (why per-request fails, Phase 4.6):** each request's argmax prices its OWN accepted tokens
against SPS(B) but NOT the slowdown its drafts impose on the other N−1 requests' verification — a negative
externality that grows with B. At saturation the per-request greedy over-admits (a ~20% minority stays at K=8,
batch oscillates). The fix is to make the decision at the **batch** level, where the total B and total slowdown
are visible.

**Hypothesis (H-5.1):** a speculation-aware scheduler that (i) makes a **batch-level SD on/off decision** from
live load state — committing the WHOLE batch to K=0 when the batch-level argmax says SD-off wins — and (ii)
when OFF, **skips the ngram draft pass entirely** (a new scheduler/worker-level action that removes the fixed
spec-path overhead) — reaches **AR throughput at saturation** while retaining the low-load gain.

**Controller `ldm_batch` (rule fixed BEFORE any run):**
- Per-request acceptance J_r[l] = P(acc ≥ l) over last W=8 steps (causal), exactly as Phase 4.6.
- Per-request k*_r computed EXACTLY as Phase 4.6 rev-#4c: `argmax_{k∈0..8} [(N−1)(1+a_bar) + 1 + Σ_{l≤k} J_r[l]] · SPS(B_live + (k − d_r))`
  (kept so the comparison to Phase 4.6 is clean — same acceptance signal, same SPS profile).
- **NEW batch-level decision:** `K_batch = argmax_{k∈0..8} SPS(B_live + N·k) · (1 + mean_l[k])` where
  `mean_l[k] = mean_r Σ_{l≤k} J_r[l]`. This prices the externality: all N requests' slowdown from N·k extra
  scheduled tokens is in the single SPS term. (The batch-level N factor cancels in argmax.)
- **NEW batch-level action:** if `K_batch == 0` → set `OFF=True`, override ALL per-request k*_r to 0, AND skip
  the ngram draft pass this step (worker-side: `GPUModelRunner.propose_draft_token_ids` returns early). vLLM's
  forward path is already AR when `scheduled_spec_decode_tokens` is empty (gpu_model_runner.py:2040); skipping
  the draft pass removes the residual per-step overhead. If `K_batch > 0` → `OFF=False`, use per-request k*_r
  as Phase 4.6, normal drafting.
- **No concurrency label C anywhere.** Only measured (N, B_live, d_r, a_bar, J_r) + static SPS profile.
- OFF is a module-level flag visible to both scheduler and worker (TP=1 → UniProcExecutor → same process).

**Why this is a NEW hypothesis (not a Phase 4.6 revision):** Phase 4.6's action space was K∈{0..8} per request,
decided independently. Phase 5 adds (a) a batch-level decision that can commit the whole batch to SD-off (fixing
the externality) and (b) a scheduler/worker-level action to disable the spec path itself (removing fixed overhead).
Different mechanism + different contribution; charter §13 revision budget resets for this new hypothesis.

**Pre-run correction (before any run):** an earlier draft of H-5.1 gated OFF on "ALL per-request k*_r == 0". That
gate is dead at C=96 given Phase 4.6's known oscillation (P(all 96 simultaneously k*=0) ≈ 0.81^96 ≈ 1e-12), so it
would falsify for a mechanical reason. Replaced with the batch-level argmax above, which is the actual externality
fix and can commit the batch to SD-off. No run used the flawed gate.

**Mechanical corrections M1+M2 (post-smoke s1/s2 OOM, charter §13 — decision rule shape UNCHANGED; the
argmax `SPS(B + N·k)·(1+mean_l[k])` is exactly as pre-registered above):**
- **Root cause of the s1/s2 OOM (diagnosed from decision logs + Phase 4.6 ldmload control run, which ran C=96
  clean for 150s @ 2449.7 tok/s → environment fine, controller at fault):** the batch argmax was CORRECT even at
  the all-ones cold-start prior (B_live=96 → K_batch=0, OFF ~98% of steps); the fatal step had K_batch=1 but the
  ON-path ALLOCATION fell back to Phase 4.6's per-request rule, which with an empty acceptance history (a_bar=0 —
  OFF periods schedule no drafts so `make_spec_decoding_stats` never fires) over-estimated J_r and scheduled
  sum_k*=728 on top of a prefill wave → ~864 sample positions → 846 MiB `get_token_bin_counts_and_mask` OOM.
  The allocation violated the cost model the batch decision had just priced (budget ~91 tokens).
- **M1 — decode-only load signal:** B_eff and N are computed over DECODE-phase requests only
  (`num_computed_tokens ≥ num_prompt_tokens`, tracked via cumulative `num_scheduled_tokens`), because
  speculation only acts on decode-phase requests and the SPS profile's B-axis is total scheduled positions.
  Verified: at steady state (B_live≈96) N_dec mean = 93.2/96; during prefill waves N_dec drops with the number
  of actually-decoding requests, which bounds total drafts to `N_dec·K_batch` ≤ ~104/step in s3 (vs fatal ~864).
- **M2 — allocation matches the priced model:** when ON at K_batch=k, k*=k for every DECODE request (uniform),
  k*=0 for prefill requests. This is exactly what `SPS(B_eff + N_dec·k)` priced and is the truest expression of
  H-5.1 (the batch decision IS the allocation; per-request greedy is what Phase 4.6 falsified).
- **No clamp on B in the argmax** (as pre-registered): clamping to the profile's max fitted B makes SPS constant
  for all k above the knee, so the argmax degenerates to "pick max k" exactly at high load — destroying the
  externality pricing (verified numerically: with clamp C96-steady → K_batch=8; without → K_batch=0).
  Extrapolating SPS(B)=A/(Bc+Cc·B^D) beyond the fitted range is monotone-decreasing and physically sensible.
- **Smoke s3 (C=96, 90s, post-M1/M2):** err=0, 0 OOM lines, OFF 98.9% of steps, max drafts/step=104, steady
  server-side gen/s ≈ 2394 (vs AR 2557.2). Mechanism fires as designed; full pre-registered trials below are
  the adjudication.

**Cells:** ldm_batch × C ∈ {8, 32, 96} × 3 trials = 9 cells, mixed workload (Phase 4.2), DUR=150s, eager.
Plus 1 diagnostic: `ldm_batch` with `VLLM_LDM_BATCH_FORCEOFF=1` at C=96 (forces OFF always → skips draft pass
always; measures the pure "spec path fully disabled at runtime" behavior — should ≈ AR if the skip is clean).
Baselines reused: AR / K4 / K8 / LDM / DSpark-rule / ldm_load / force-0 from Phase 4.2/4.5/4.6 (no re-run).

**Success (pre-registered):**
- **C=96:** ldm_batch ≥ AR − noise (≈0.3%, i.e., ≥2541 tok/s) AND beats force-0 (2437.6, proves the draft-pass
  skip removes the fixed overhead) AND beats ldm_load (2235.2).
- **C=8:** ldm_batch ≥ +5% vs AR (i.e., ≥406 tok/s) — the batch decision must keep speculation ON at low load.
- **Mechanism cost:** force-off diagnostic at C=96 ≥ AR − 1% (≥2532 tok/s) — confirms the skip is clean
  (runtime-disabled spec path ≈ spec-disabled server).
**FALSIFY** if it fails any bound: then the batch-level on/off + draft-pass-skip as specified does not reach AR at
saturation, and we report that (no post-hoc retuning, charter §13).

**Interpretation guard:** if ldm_batch ≈ ldm_load at C=96 (OFF never fires or doesn't help), the batch decision is
not committing to SD-off — FALSIFIED. If ldm_batch reaches AR at C=96 but loses the C=8 gain, the batch decision is
too aggressive (commits to SD-off when speculation should be on) — also FALSIFIED as specified.

**Correctness note:** skipping the draft pass leaves the ngram drafter's token-history buffer stale during OFF
periods; on resume, acceptance may dip for a few steps (stale context) but output stays LOSSLESS by construction
(speculative decoding verifies every draft against the target model — bad/stale drafts are rejected, never produce
wrong tokens). The main forward pass is independent of the drafter's buffers. Verified empirically in smoke tests
(output validity + no crash + acceptance recovers on resume).

**Implementation:** new controller `experiments/ldm_batch_controller/sitecustomize.py` (extends Phase 4.6's with
the batch argmax + OFF flag + worker-side propose-skip hook). `VLLM_LDM_BATCH_FORCEOFF=1` forces OFF always.
Enforcement verified from decision logs: when OFF=True, draft/s ≈ 0 in metrics and no crash/corruption.

**RESULT (2026-10-08, full grid + same-day controls): H-5.1 FALSIFIED as specified.** Full write-up:
`docs/phase5-p1-results.md`. Server-side gen/s (steady t≥20s), all same-day:

| arm | C=8 | C=32 | C=96 |
|---|---|---|---|
| AR (same-day control) | 387.0 (p2) | 1460.0 (p2) | **2641.6** |
| ldm_load (P4.6) | 417.0 (+7.8%) | 1304.2 | 2235.2 |
| **ldm_batch (H-5.1)** | **410.1 (+6.0%) PASS** | 1302.3 (−10.8%) | **2422.2 (−8.3% FAIL)** |
| P4.6 force-0 floor (no skip, same-day) | — | — | 2431.9 (−7.9%) |
| ldm_batch force-off (skip every step) | — | — | **2355.2 (−10.9% FAIL)** |

Bounds: C=8 ≥+5% → PASS; C=96 ≥AR−noise (≥2634) → FAIL (2422.2); C=96 beats force-0 floor → FAIL (2422.2 < 2431.9,
by 0.4%); mechanism-cost force-off ≥AR−1% → FAIL (2355.2). Per §13: no retuning; FALSIFIED as specified.

**Two findings inside the falsification:**
1. **The batch decision rule WORKS.** Decision logs: C=8 → K_batch≈5 (OFF 0%), C=32 → K_batch≈1, C=96 → K_batch=0
   (OFF 99.3%), force-off → 100% OFF / 0 drafts. Even at the all-ones cold-start prior (OFF periods starve
   `make_spec_decoding_stats` so acceptance never populates at C=96), the externality-priced argmax commits the
   whole batch to SD-off at saturation — exactly what per-request greedy could not do (it oscillated with a ~20%
   K=8 minority). The mechanism is validated; it simply acts on a cost model that under-prices "SD-off".
2. **The draft-pass skip does NOT remove the floor (A/B-confirmed, 3 runs each, same GPU/day).** force-off
   (skip) = 2351±5 vs no-skip floor (spec ON, k*=0) = 2398±43: the skip is ~neutral (−2%, within the no-skip
   arm's spread) and recovers NONE of the ~9% gap to AR. The Phase 4.6 "fixed spec-path overhead" therefore does
   NOT live in the ngram draft kernel; it sits in the rest of the spec-enabled serving path (scheduler
   bookkeeping, rejection-sampler setup, input-batch spec fields) that stays active whenever `speculative_config`
   is set. Identical KV cache (117,264 tok), 0 OOM, err=0 in all runs.

**Phase 5 status: H-5.1 FALSIFIED (mechanism validated, cost model wrong). Next:** Phase 5.2 — profile WHERE the
~9% SD-off floor lives (scheduler vs model-runner spec fields) and pre-register a mechanism that removes it
upstream (spec-path-free off-state), since no K-allocation or drafter-kernel skip can.

### 2026-10-09 — Research addendum: native DSpark baseline validation + proxy-validity audit (Phase 5.2 deprioritized)
A research addendum re-prioritized the project: before further scheduler work, verify the DSpark baseline and
related-work assumptions against primary sources, in an isolated environment that does not touch the running
setup. Phase 5.2 is paused until this audit completes. Findings (full docs below):

**1. Native DSpark exists in upstream vLLM — our "no integration" premise is outdated.** PR #47808
("DSpark confidence-scheduled verification") merged **2026-08-12**; blog 2026-08-14. Release boundary verified
from tagged sources: DSpark drafter (`qwen3_dspark`) first in **v0.25.0**, adaptive verification
(`adaptive_verification.py`) first in **v0.28.0** (404-checked v0.24–v0.27). Our pin vLLM 0.19.1 (released
2026-04-18) genuinely predates it, so the proxy was a legitimate choice at design time — but the related-work
claim needed a dated correction (added to `docs/related-work-dspark.md` §4 + table, and `docs/phase4-p5-results.md`).
Upstream's budget rule is the **same argmax form** as our H-5.1 (`(N + cumsum(survival)) / cost`), with a trained
confidence head (cumprod prefix-survival), separate draft/verify cost tables profiled from dummy CUDA-graph steps,
and prefill priced via the non-draft token count — so "batch-level allocation pricing draft externality" is **no
longer novel by itself**; our differentiator must be live serving-state inputs + cheap-drafter SD-off economics.

**2. Hardware gates (SM86 / driver CUDA 12.9).** Adaptive verification: **definitive blocker** — it requires varlen
decode CUDA graphs (`AttentionCGSupport.ALWAYS`): FA3 needs SM90, FlashInfer trtllm-gen needs SM100; FA2 on SM86 is
`UNIFORM_BATCH` with no bound → rejected at startup by design. Fixed-K DSpark drafter: **no arch gate** in source —
runnable IF we get a cu12-compatible vLLM build. But every DSpark-capable release (≥0.25) ships cu13-only wheels
(torch 2.11/2.13, `libcudart.so.13`), and our driver caps at CUDA 12.9 → **Gate A fails on prebuilt wheels**
(reproduced: import error). Isolated attempt (separate clone `/home/junior1/_dev/dspark-native-validation`, own venv,
no changes to the research env): source build of v0.25.0 against CUDA 12.6 — fixed four sequential toolchain blockers
(GCC 9.4 gate → clang 10 C++20 failure → installed GCC 13 via micromamba in user space → nvcc host-compiler still
g++9.4, fixed with `CMAKE_CUDA_HOST_COMPILER`); build now compiling cleanly (in progress). Checkpoints are public and
fit a 3090 (`deepseek-ai/dspark_qwen3_4b_block7` 2.79 GB + Qwen3-4B 8.06 GB).

**3. Proxy validity audit — two confirmed implementation bugs in the Phase 4.5 DSpark-rule arm.**
(F1) Decision map covered only requests that drafted last step (~6–7% of running at C=96); all others enforced at the
KMAX=8 default; a decided K=0 dropped out and returned to KMAX next drafted step (317 observed instances) — the
pre-registered all-K=0 SD-off fallback was **never effectively enforced**. (F4) SPS(B) evaluated at the decided
subset's token count (~25), not the true batch (~96–130) → cost term could not price saturation. The "DSpark-rule is
the worst arm" result therefore stands only as a statement about our buggy proxy, **not** about DSpark's decision rule;
interpretation #1 of Phase 4.5 (rule can't learn load-driven K) is unsupported, #2 (live-load signal needed) weakened.
Phase 4.6/5.1 controllers are unaffected (they already fixed this bug class — rev-#4b "decide for every running
request" + persistent kstar). Corrected rerun **P4.5R** pre-registered in `docs/dspark-proxy-validity.md` (Track A,
deferred until the native audit closes).

**Docs:** `docs/dspark-native-audit.md` (new — support/runnability evidence), `docs/dspark-proxy-validity.md` (new —
F1/F4 + P4.5R preregistration), dated corrections appended to `docs/related-work-dspark.md`, `docs/phase4-p5-results.md`.
Historical measurements preserved; no raw results touched. **Next:** finish the isolated source build → Gates A–D on
fixed-K DSpark (adaptive verification documented as SM86 blocker) → smallest matched AR-vs-fixedK-DSpark comparison →
answer the 7 synthesis questions. Standing instructions: this entry + docs will be pushed at a safe checkpoint and
tracked in GitHub Project / Discussion.

### 2026-10-09 (cont.) — Native DSpark Gates A–E COMPLETE + synthesis (audit closed)
The isolated source build succeeded (`vllm-0.25.0+cu126`, CUDA 12.6 + GCC 13 via micromamba; post-build dep repair:
torchvision/torchaudio/torchcodec cu129 wheels). All five gates executed on Qwen3-4B + `dspark_qwen3_4b_block7`
(2.6 GB, public; block_size=7 kept as-is) on GPU1, port 8300/8301:

- **Gate A (config): PASS** — `Resolved architecture: Qwen3DSparkModel`, engine config `method='dspark', num_spec_tokens=7`.
- **Gate B (drafter): PASS** — dspark CUDA graphs captured 45/45; real drafts with learned acceptance decay
  (14→9→9→3→1→1 per position); full run: 98,413 draft / 24,994 accepted tokens (≈0.25 rate on ShareGPT text). No ngram fallback.
- **Gate C (adaptive verification): NOT ACTIVE in v0.25.0** — `qwen3_dspark.py:173`: "confidence_head is not wired into
  inference yet; skip its weights". Adaptive path ships v0.28+ (needs cu13) AND needs varlen decode CG (SM90/SM100).
  Definitive, source-verified blocker on our stack.
- **Gate D (scheduling/correctness): PASS** — 16 concurrent heterogeneous requests, coherent output, causally aligned
  per-position acceptance under load.
- **Gate E (stable serving + smallest matched comparison): COMPLETE.** Same target/runtime/GPU/workload (ShareGPT,
  greedy, 60 prompts), C∈{8,32,64}:

| C | AR tok/s | DSpark(K=7) tok/s | Δ vs AR | TPOT ms (AR→DSpark) | TTFT ms (AR→DSpark) |
|---|---|---|---|---|---|
| 8 | 463.2 | **757.9** | **+61.5%** | 13.4 → 8.9 | 105 → 136 |
| 32 | 995.0 | **1116.1** | **+12.2%** | 15.3 → 20.7 | 133 → 184 |
| 64 | **1134.0** | 1082.6 | **−4.5%** | 16.9 → 30.2 | 228 → 418 |

Native fixed-K DSpark reproduces the load-dependent K effect with a learned drafter: wins at low load, net-negative
at C=64 (crossover between 32 and 64 on Qwen3-4B/3090 — later than Qwen2.5-7B's, as expected for a smaller model).
Raw results + server logs archived in `results/gate_e_native_dspark/`.

**Synthesis (7 addendum questions answered):** `docs/dspark-audit-synthesis.md`. Headlines: proxy was legitimate at
design time (0.19.1 predates v0.25.0) but the "no integration" claim is dated-corrected; our batch-argmax + prefill
pricing assumptions were CORRECT (upstream matches); "novel by itself" claims are retired; the proxy's Phase 4.5
DSpark-rule numbers stand only as a statement about our buggy implementation (F1/F4). **Remaining measured problem:**
fixed-K DSpark's saturation loss (−4.5% at C=64) + prefill-heavy mixes (not yet tested natively). **Smallest
falsification experiment (next, Track B):** live-serving-state K/SD-off controller vs native fixed-K DSpark on this
matched runtime, 3 trials/cell — must beat fixed-K=7 by > noise at C=64 without losing the C=8 gain.

**Audit status: COMPLETE.** Phase 5.2 (scheduler expansion) remains deprioritized per addendum; next work = P4.5R
(Track A, deferred) and the Track B pre-registration above.

---

## 2026-10-09 (cont. 2) — New directive received: crossover reproduction + Track B live-state controller

Directive "Native DSpark Baseline and Next Research Milestone" (checkpoint `8491ec0`). §2 state sync confirmed
local=remote, no post-checkpoint work by others. Two deliverables completed this session:

**Phase 6 — crossover reproduction (prereg + full run + analysis).** `docs/phase6-reproduction-prereg.md`
(§8 = append-only results). 18 valid trials (3/cell × {AR, DSpark K=7} × C∈{8,32,64}), fresh server per trial,
counterbalanced order, seed 0. One harness incident (my shell-quoting bug killed all first-pass DSpark servers;
no data lost, logs preserved `*.failed1`, fix smoke-verified before relaunch).

| C | AR tok/s | DSpark tok/s | Δ vs AR | Welch 95% CI |
|---|---|---|---|---|
| 8 | 459.6 ± 4.7 | **765.6 ± 1.0** | **+66.6%** | [+65.1, +68.1] |
| 32 | 872.3 ± 7.8 | **986.2 ± 33.3** | **+13.1%** | [+7.7, +18.4] |
| 64 | **983.4 ± 3.0** | 961.3 ± 25.9 | **−2.2%** | [−6.1, +1.6] |

Preregistered decision rule → **REVISE branch**: sign pattern (+,+,-) reproduced (stronger at low load than Gate E),
but the C=64 deficit's CI includes 0 at n=3 (directional: DSpark<AR in 3/3 paired rounds, every trial). Operating
regime: C=64 = GPU-compute-saturated (util 100%), NOT KV-saturated (≤0.2%, zero preemptions). **Key mechanistic
finding:** per-position acceptance is load-invariant (0.72→…→0.03 at all C; acc_len ≈ 2.77) — the crossover is a
verification-compute-amortization effect, and TPOT crosses over AR already at C=32 (26.2 vs 21.4ms), earlier than
throughput. Dated premise revision recorded in §8.6; Gate E's −4.5% single-trial figure is NOT claimed significant.

**Track B — feasibility audit + controller prereg.** `docs/trackb-feasibility.md`: v0.25.0 has a native dynamic-K
mechanism (`num_speculative_tokens_per_batch_size` → per-step `num_spec_tokens_to_schedule`) but it is **V1-runner-only**;
DSpark forces the V2 GPU runner where that field is dead (grep-verified), draft width is fixed at init with FULL CGs,
rejection sampler is fixed-K → **per-iteration variable K blocked (B1 worker / B2 verification / B3 channel)**.
Feasible surface: **per-step binary SD on/off** via ~10-line patch (gate `speculator.propose()` in V2 runner; K=0 steps
take the existing no-draft decode path), with per-step causal-verification logging. `docs/trackb-prereg.md`: H-TB,
arms AR / DSpark-K7 / CTRL-ON (overhead control) / CTRL-LIVE, C∈{8,32,64} × ≥3 reps; success S1–S4 + falsification
criteria fixed pre-run. **Revision R1 (dated, post-analysis):** acceptance-ratio trigger has no signal on this workload
(load-invariant) → primary trigger revised to measured TPOT trend + batch occupancy + prefill share (no concurrency
thresholds); [CAL] block filled from Phase 6 operating points before any controller trial.

**Track B Step 3a — IMPLEMENTED + unit-verified (2026-10-09, GPU idle).** The per-step SD on/off patch is implemented in
the isolated worktree `/home/junior1/_dev/dspark-native-validation` on branch `track-b-live-sd` (commit 82255c3): a new
`vllm/v1/core/sched/tb_controller.py` (measured-state decision, no concurrency thresholds) + minimal hooks in
`scheduler.py` / `output.py` / `gpu/model_runner.py` / `spec_decode/utils.py`. OFF steps skip `speculator.propose()`
entirely and clear pending drafts (0 new drafts scheduled; decode runs as plain AR, 1 token/req); ON steps are
byte-identical to upstream. CUDA-graph safety verified: OFF-step shapes fall back to eager via `CudaGraphManager.dispatch`
(no recapture). **Revision R2 (S1 restated to S1a parity + S1b directional — original S1 unmeasurable at n=3)** and
**R3 (final controller design fixed before any trial)** are recorded in `docs/trackb-prereg.md`. R3 corrected two defects
that implementing R1's "TPOT-trend slope" exposed: (1) unit mismatch — the per-step measured cost must be compared to an
L-scaled AR threshold (`threshold(N)=2.77·est_ar_ms(N)`), else it would fire OFF at C=8 and kill the low-load benefit;
(2) circular re-entry — re-entry is now **probe-based** (turn SD on for 3 steps after cooldown, commit only if the measured
DSpark cost < threshold). Final rule classifies Phase 6 as C=8 ON / C=32 ON / C=64 OFF. **17 deterministic tests pass**
(`tests/v1/core/test_tb_controller.py`), including the REAL `Scheduler.schedule()` causal assertion (OFF ⇒ 0 drafts +
1 token/req; ON ⇒ K=7 drafts/req) and probe commit/revert/cooldown; existing vLLM spec-decode scheduler tests still pass
(no regression).

**Next:** the Track B comparison run set — AR / DSpark-K7 / CTRL-ON / CTRL-LIVE × C∈{8,32,64} × ≥3 valid reps (fresh
server per trial, GPU1, counterbalanced), then analysis against S1a/S1b/S2/S3/S4. Track A (P4.5R) stays deferred per
directive §5; supervisor's heterogeneous prefill/decode characterization queued after Track B.




### 2026-10-10/11 — H-OBS-1 closed (PARTIALLY CONFIRMED) + H-OBS-3 COMPLETE (H2 SUPPORTED, H3 NOT SUPPORTED, H1 INCONCLUSIVE)

**H-OBS-1 (Issue #26, Done):** async GPU observability validation of the obs tracer. Verdict
PARTIALLY CONFIRMED — `gpu_step_ms` = target forward + sampling + drafter cleanly separable via
`gpu_fwd_ms`; nsys cross-check PASS (S1 anchor count / S3 aggregate busy gates); ≤0.5% overhead at C≤32.
Full report: `docs/hoobs1-observability.md`. H-OBS-2 not pursued per its verdict.

**H-OBS-3 (Issue #27, Done) — heterogeneous prefill/decode characterization of native fixed-K DSpark.**
Prereg `docs/hoobs3-prereg.md` (prospective, fixed pre-run); results `docs/hoobs3-results.md` @ 35fb6aa.
4 cells arm{ar,dspark} × C{32,96}, hetero client (SS/SL/LS/LL ISL×OSL classes, natural EOS, seed 20261011),
180 s each — all DONE-VALID first attempt, hard stops audited clean (in_flight≥2 ≥0.995, 100% gpu_step_ms).

- **H1 phase: INCONCLUSIVE.** Acceptance flat where labels are clean (c32: decode 0.6221 vs prefill 0.5931,
  gap −0.029 < 0.03 bar, n=68 < 100 guard); c96 phase labels invalid under async admission + prefix caching.
  Scope caveat: workload is cache-dominated — client reuses 4 canonical prompts, **97.3% prefix-cache hit rate**
  in both dspark cells → prefill-heavy steps only 3.2% at C=32. Null does not extend to diverse-prompt serving.
- **H2 class: SUPPORTED (strong).** DSpark/AR TPOT ratio within each C: c32 0.552–0.798 (range 35.9%),
  c96 0.745–1.166 (range 43.8%). Long-output classes gain most; SS crosses to negative under saturation.
- **H3 price: NOT SUPPORTED.** Drafter share flat vs pressure: dspark 0.214→0.209, ar 0.047→0.036
  (Δ per +16 running: −0.25pp / −0.55pp; bar was >+1pp). Absolute drafter ms scales 16.4→31.2 with batch but
  target forward scales faster. Net SD throughput gain: +30.9% (C=32) → +12.5% (C=96).

**Opportunity surface for Track A:** class/OSL-aware admission = strong signal; phase features = none on this
workload; pressure enters via predicted payoff, not a growing cost term. Harness: `experiments/hetero_client_qwen3.py`
+ `experiments/hoobs3_analyze.py` (epoch-aligned metrics↔steps via new additive `t_wall`/`t_epoch` fields);
launcher `scripts/run_hoobs3.sh` + tracer field in worktree @ 333b37b.

**Next:** Track A P4.5R corrected-proxy rerun remains the queued item (when it doesn't obstruct higher-priority work).

### 2026-10-11 — P4.5R corrected-proxy rerun COMPLETE (Track A; hypothesis CONFIRMED, falsification not triggered)

Corrected DSpark-rule proxy (`experiments/dsparkr_controller/`, commit 886a300): same decision rule
as Phase 4.5, three mechanical fixes only per `docs/dspark-proxy-validity.md` §6 — F1 decide every
running request + persistent k* (K=0 survives zero-draft steps); F4 B = total_num_scheduled_tokens
(true batch) with monotone-decreasing extrapolation beyond the profile range; F2 censoring-aware J.
Unit tests pass (`experiments/test_dsparkr.py`); C=32 smoke verified fixes live before the grid.

6/6 cells valid (C∈{32,96} × 3 trials, server-side gen/s steady t≥20s, baselines reused from P2):
- **Fixes confirmed working:** decided/N = 1.000 (was ~6–7%); B_live = 33.2 / 99.6 at C=32/96
  (was ~7–25). The rule now prices saturation.
- **C=96:** corrected proxy −4.5% vs AR (old buggy: −27.2%) — between fixed-K8 (−24.2%) and AR,
  committed to SD-off (frac K=0 = 0.994). **+22.7pp recovered** → Phase 4.5's "worst arm" ranking
  confirmed as an implementation-bug artifact, not a DSpark-rule failure.
- **C=32:** −7.8% vs AR; within 0.5% of the P4.6 force-0 SD-off floor (1336.9) — the rule finds no
  profitable speculation regime on this ngram stack, so it pays the floor and turns SD off.
- **Verdict:** hypothesis CONFIRMED (between K8 and AR at C=96); falsification NOT triggered
  (−4.5% outside ~2% trial noise). Rule is faithful given correct inputs; residual gap = SD-off
  floor on an ngram-drafter stack (proxy-level limitation, ngram scope stated). Full report:
  `docs/p45r-results.md`; prereg §6 closed with a dated completion note.

**Next:** Track A P4.5R is now DONE — the queued item list is clear; remaining open project work is
Phase 5 (speculation-aware scheduler, Issue #23) if/when prioritized.

### 2026-10-11 — Phase 5.2 (H-5.2) COMPLETE: FALSIFIED as specified; floor located + root-caused

Restarted Phase 5 per user directive. H-5.2 = "profile where the ~9% SD-off floor lives, then build a
mechanism that removes it (spec-path-free off-state)." Prereg `docs/p52-prereg.md` (committed f67d7e2
before any run). **Verdict: FALSIFIED as specified** — a runtime spec-path-free off-state does not
remove the floor; but the experiment located its mechanism, which is the durable result.

- **P1 (profile):** ported H-OBS-1's CUDA-event tracer to vLLM 0.19.1 as a sitecustomize overlay
  (`experiments/p52_obs/`), anchored on `execute_model` (once/real-step) — key version fix: the 0.19.1
  engine spins `schedule()` ~490×/step in a tight non-blocking loop, so schedule() is not a
  once-per-step anchor (H-OBS-1's 0.25.0 blocked between steps). Floor decomposes per decode step into
  forward-metadata sizing + sampling/drafter + CPU residual; both arms run at identical occupancy and
  draft_tokens=0 on every force-0 step. G1 literal threshold (≥50% CPU-side residual) missed (49%/28%)
  but its falsification trigger was not hit — the floor is entirely spec-path work.
- **P3 (adjudication):** 18/18 cells valid, 0 crashes/OOM. specfree-off (force-0 truncation + full
  drafter-pass skip = Phase 5.1 M3's mechanism) recovers NONE of the floor: C=96 −9% of the AR−force-0
  gap (worse than force-0), C=32 identical to force-0; both ~8–9% below AR. FALSIFIED criterion met
  (recovery ≤ 20% at C=96).
- **Root cause (code-level, vLLM 0.19.1 eager):** the floor is structural to running with
  `speculative_config` set — force-0 already takes the AR code path in `_prepare_inputs`
  (`use_spec_decode=False` when drafts are zeroed); FLASH_ATTN v2 has no per-step spec branch; the
  remaining cost is the ngram drafter pass running every step (M3's skip of exactly this recovered
  nothing, A/B-confirmed) + init-time GPU memory that never returns per-step. So no runtime off-state
  can remove it. Durable result: the ~8–9% SD-off floor is a serving-architecture constant that any SD
  on/off controller (incl. native Track B) must price as irreducible at saturation; reaching AR requires
  a genuinely spec-free serving path (separate AR pool / init-time no-spec), out of scope for in-loop
  controllers.

**Phase 5 is now closed** (H-5.1 falsified, H-5.2 falsified with mechanism located). Full report:
`docs/p52-results.md`; prereg §6 closed with a dated completion note. Tooling committed (P1 @6ec6288,
P3 @ed4d252); this entry + results doc + prereg close = final commit.

### 2026-10-12 — New directive: request-aware serving scheduling; H-CLASS-1 opened (Issue #28)
Directive "From Speculative Decoding Characterization to Request-Aware Serving Scheduling" received at
checkpoint `3fb1273`. State sync: local == remote, no post-checkpoint work, GPUs free. All 10
directive-listed docs read.

**Deliverable 1 — Phase 5.2 scope note (`docs/p52-scope-note.md`, @055cf36):** dated claim-scope
correction per directive §9. The tested P3 mechanism was force-0 truncation + drafter-pass skip (M3),
NOT the prereg §5 metadata/static-field modifications — which were verified no-ops in eager mode before
P3, so zero §5 fields were removed (prereg's own fallback clause satisfied). FALSIFIED verdict preserved
verbatim; scope boundary fixed: the ~8–9% floor constant holds for vLLM 0.19.1/ngram/eager only and must
not be transferred to native DSpark/vLLM 0.25.0 without measurement.

**Deliverable 2 — H-CLASS-1 preregistration (`docs/hclass1-prereg.md`, @055cf36, Issue #28 created +
project item `PVTI_...zhAIokA` In progress).** Step-1 audit of the H-OBS-3 workload (client code + raw
records) found three confounds: (1) 4 canonical prompts at 97.3% prefix-cache hit rate — prompt identity/
content/cache fully confounded with class; (2) **realized OSL == configured max_tokens for 100% of
requests in every class/cell** (frac_at_max = 1.000, zero natural-EOS terminations despite the "NATURAL
EOS" docstring — synthetic turn-concatenation prompts have no stopping point), so H-OBS-3's class effect
is on (ISL, configured decode horizon), not natural output length; (3) all-at-once open-loop decay makes
C=96 non-steady. De-confounded design: diverse distinct real ShareGPT-V3 prompts per class (seeded,
disjoint, identity-paired across arms), natural EOS on real turns, Poisson arrivals at calibration-frozen
rates; cache condition A (diverse + cache OFF, primary) and B (H-OBS-3's 4 canonical prompts + cache ON,
control); AR-only calibration → frozen P_low/P_med/P_high before the confirmatory grid; 36-cell grid
(arm{AR, DSpark-K7} × condition{A,B} × pressure{3} × 3 trials), trial-level inference. H-C1 replication
bar: cross-class TPOT-ratio range >15% in both conditions at ≥1 common pressure.

**Next:** harness build + smoke validation (diverse-prompt Poisson client, cache on/off modes, per-request
identity pairing), then calibration → freeze §5b → confirmatory grid.

### 2026-10-13 — H-CLASS-1 COMPLETE: H-OBS-3's class effect does not survive de-confounding (Issue #28)
Full grid run + analyzed; **41/41 DONE-VALID** (5 cal + 36 confirmatory; 5 cells re-run once each for the
transient CUDA-driver-init startup race, all passed). Report: `docs/hclass1-results.md`.

- **★ H-C1 FALSIFIED.** The cross-class DSpark/AR TPOT-ratio spread that H-OBS-3 reported (35.9–43.8%) is
  an artifact of its confounded workload. On the de-confounded diverse-prompt, cache-off condition A the
  range collapses to **0.57% / 0.48% / 9.69%** (c16/c48/c96) — none reach the 15% bar. The spread only
  reappears in condition B, which reproduces H-OBS-3's exact config (2 repeated prompts, 97–98% cache hit):
  37.0 / 41.6 / 42.2%. No common pressure has both conditions >15%, so the replication conjunction fails.
- **H-C2 VIOLATED at c48/c96.** The *sign* of SD's per-request payoff flips across cache conditions: on A,
  fixed-K=7 DSpark **hurts** TPOT (ratio 1.16–1.27) at mid/high load; on B the long-horizon classes still
  **benefit** (LL ratio 0.67/0.75). At c16 all signs agree (SD helps everywhere). SD's payoff sign is not a
  stable property of request class — it depends on the serving regime (cache/reuse) as much as the class.
- **H-C3 NOT SUPPORTED — acceptance dominates.** Descriptive OLS (n=18): β_acceptance = **−1.644** vs
  β_logOSL = −0.117, β_logISL = −0.021. Draft **acceptance rate**, not realized output length, carries the
  per-request SD payoff. Acceptance nearly doubles from A (~0.33) to B (~0.61) purely via cache reuse making
  repeated prompts' drafts more predictable — drafter share stays flat (~20%).
- **Revised Track-A signal:** predict an *acceptance proxy* (content/draftability + cache-hit state) at
  admission to set per-request K / SD on/off; treat cache state as a first-class feature (it flips the sign).
  This supersedes H-OBS-3's "predict output length" framing. Condition C (diverse prompts *with* cache) is
  the deferred follow-up that would split the joint diversity+cache effect.
- **Tooling fixes this phase:** `parse_cell()` regex `\d`→`[0-9]` (invalid in bash POSIX ERE; latent whole-grid
  bug); §5b lambda table frozen into `lookup_lambda`; counterbalanced arm-alternating cell order; **deviation
  D3** — condition-A validity now gates on applied load (issued arrivals, no dropped queue) + a service-aware
  completion floor (`SAT_REQ_RATE=0.35`), replacing the arrival-rate floor that over-predicted ~4× under
  saturation and would have false-rejected every P_high cell; analyzer metric-key fix (`..._d` not
  `...[counter]_d`) so acceptance + cache-hit rates populate.

**H-CLASS-1 is now closed** (H-C1 falsified, H-C2 violated at mid/high load, H-C3 acceptance-dominant). Full
report: `docs/hclass1-results.md`; prereg deviations D1/D2/D3 in `docs/hclass1-prereg.md`. Issue #28 → Done.

### 2026-10-13 — H-CLASS-1C follow-up: diverse queries + shared cached context (prefix caching ON)
- **Motivation:** H-C2's violation flagged a joint diversity+cache effect; condition C splits it. Fixed question
  (prereg `docs/hclass1c-prereg.md`, written before any cell ran): does a diverse-query workload with a *shared
  cached context document* sit closer to A (content-driven acceptance gap) or B (reuse-state effect)? Since prefix
  caching changes no logits, the A↔B acceptance gap must be content, reuse-state, or both.
- **Workload:** one shared ~552-tok ShareGPT context doc as a system message (byte-identical prefix across all
  2000 bank entries, LCP=546 tok verified) + distinct diverse queries; S=doc+short (~600), L=doc+long (~2600);
  uniform cap 2048, stop at `<|im_end|>`. Prefix caching ON (the one difference from A). `build_hclass1_c_bank.py`
  → `data/hclass1_c_bank.jsonl` (local).
- **Run:** 18 cells (arm {AR, DSpark-K7} × c{16,48,96} × 3 trials), counterbalanced, frozen §5b λ, GPU0.
  **18/18 DONE-VALID, zero invalids** (no re-runs needed). Wall ≈ 1.5 h for the 17-cell grid.
- **Condition sanity:** prefix-cache hit rate 0.327/0.230/0.230 (c16/c48/c96) — between A (~0) and B (0.976);
  >5% validity guard passed everywhere. Hit rate falls with concurrency (prefix eviction), yet acceptance is flat.
- **VERDICT: C≈A (content-driven gap).** Trial-mean dspark acceptance: c16=0.304, c48=0.303, c96=0.315 — within
  0.029 of A's 0.332 at every pressure, ~0.30 below B's 0.611. Cross-class TPOT-ratio range 7.58/5.67/10.73% (all
  <15%), mirroring A (0.57/0.48/9.69%) and contrasting B (37.0/41.6/42.2%). Prereg bands → `near_a` true,
  `near_b` false. **Caching a shared prefix does not lift acceptance or restore the class spread.**
- **Interpretation:** H-OBS-3's "class effect" and the ~0.33→0.61 acceptance jump come from B's *specific repeated
  prompts being more draftable*, confounded with repetition — not a reusable cache state. C's acceptance even sits
  slightly below A's (extra context length / longer answers), reinforcing that content, not cache, drives it.
- **Refined Track-A signal (supersedes the H-CLASS-1 note):** the admission-time predictor should target an
  *acceptance proxy from prompt content/draftability* (perplexity/entropy features of query+context); treat cache
  state as a *cost* feature (TTFT/prefill savings), not a *quality* feature. The pressure-dependent sign flip of SD's
  payoff (helps c16, mildly hurts c48/c96) reproduces across A and C — that part is real and content-independent.
- **Causal limit (prereg §5):** within C the shared document and "longer richer prompt" are not separable; a
  cache-ON diverse *no-shared-doc* control would split them (not run, out of scope). Regression claims descriptive.
- **Harness:** `experiments/hclass1_client.py` (COND=C open-loop path), `hclass1_run.sh` (cache ON for C + >5%
  hit-rate validity guard; `HCLASS1_CONDS`/`HCLASS1_C_LEVELS`), `hclass1_analyze.py` (`SCHEME["C"]`, `condition_c`
  verdict block). Committed `8d964e2`. Full report: `docs/hclass1c-results.md`.

### 2026-10-13 — Phase 5 / Track A (Issue #29): content-based per-request acceptance predictor
- **Question:** can per-request DSpark-K7 draft acceptance be predicted from admission-time prompt
  features, and does that prediction have decision value for per-request SD on/off?
- **Method (prereg `docs/tracka-prereg.md`, bands fixed first):** 550 prompts labeled at concurrency=1 via
  `spec_decode_num_{draft,accepted}_tokens_total` counter deltas (exact per-request attribution, NO vLLM patch —
  licensed by H-CLASS-1C's measured acceptance load-invariance). set1 = 400/400 valid (diverse; primary),
  set2 = 150/150 valid (shared-doc cache probe). Features: isl, unigram entropy, bigram/trigram repetition,
  prompt perplexity (free via `prompt_logprobs`), cache hit rate. Models B0/B1/M1/M2 (ridge + GBT), 5-fold CV
  prompt-disjoint folds.
- **Deviation D1:** this vLLM build's `prefix_cache_{queries,hits}_total` counters don't tick per request
  (frozen at startup; isolated probe) → per-request cache_hit_rate unattributable (2/150 non-null in set2).
  H-TA2 under-powered (reported, not forced); spec-decode counters (the label source) verified exact.
- **VERDICT: H-TA1 NOT SUPPORTED.** M2 5-fold CV R² = 0.055 (B1 length-only 0.008; ΔR² +0.047 < the 0.10 bar).
  Content features add a small real signal over length, but ~95% of per-request acceptance variance is unexplained
  by cheap admission-time statistics. Per-prompt acceptance: mean 0.315, sd 0.070, range 0.14–0.62.
- **H-TA2 INVARIENT (weak):** ΔMAE(M1→M2+cache) = 0.00027 < 0.003 — cache not a quality feature; consistent with
  H-CLASS-1C's workload-level invariance (the load-bearing evidence).
- **H-TA3 NOT SUPPORTED.** On the existing H-CLASS-1 A+C paired per-request data (n=2011): Spearman(pred_acc,
  SD payoff) = 0.041; control Spearman(true ISL, payoff) = 0.033 — even ground-truth length is ~uncorrelated with
  which requests benefit. Threshold policy beats always-on/off at 0/3 loads. Mean payoff is a LOAD property
  (+0.116 @c16, −0.165/−0.172 @c48/c96), not a per-request one.
- **Interpretation (revises the Track-A signal twice over):** (i) the A↔B acceptance gap was a *population*
  effect (two prompt distributions), not per-request spread — "predict this request's acceptance" is the wrong
  framing; (ii) within diverse traffic, individual acceptance is high-variance and driven by fine-grained local
  predictability our coarse features miss. **Revised scheduler signal: set SD on/off + K from LOAD REGIME (sign)
  + coarse TRAFFIC CLASS / prompt-distribution (expected level), not per-request perplexity.** A lightweight
  decision model should ingest serving state (queue/running/load) + a traffic-class label.
- **Deliverables:** `docs/tracka-results.md`; harness `experiments/tracka_collect.py` + `tracka_train.py`
  (committed `ed3209f`); labels local under `results/tracka/`. Issue #29 → Done.

### 2026-10-13 — Scheduler-signal validation: load-regime × traffic-class SD on/off policy
- **Question:** does the two-axis signal implied by Track A's negative result (load regime = sign, traffic class =
  level) actually beat single-axis policies on the measured payoff surface? Pure analysis of existing H-CLASS-1
  A/B/C data — no new GPU runs.
- **Measured payoff surface** (per-request SD payoff = mean(1−DSparkTPOT/ARTPOT), pid-matched, over t=1,2,3):

  | cond \ load | c16 | c48 | c96 |
  |---|---:|---:|---:|
  | A diverse | +0.122 | −0.157 | −0.156 |
  | B high-reuse | +0.477 | +0.121 | +0.022 |
  | C shared-doc | +0.109 | −0.172 | −0.186 |

  Two independent load-bearing axes: **load** flips A/C from + (c16) to − (c48/c96); **traffic class** keeps B
  positive at every load (acceptance ~0.61 vs ~0.33 rescues DSpark even at c96).
- **VERDICT: SUPPORTED.** The rule "SD-on iff low-load OR high-reuse-class" matches the oracle on **all 9 cells** —
  captures 100% of achievable gain, zero regret (request-weighted, n=5389). Every single-axis alternative is worse:
  class-only 90.3% (misses Ac16/Cc16), load-only 67.0% (misses Bc48/Bc96), always-on 51.3%, always-off 0%. Both axes
  are necessary — each covers a different half of the surface; only their disjunction is complete. Confirms the
  scheduler should ingest serving state (load) + a coarse traffic-class label, not per-request content.
- **Honest scope limits:** (i) same-data validation — internal-consistency/design confirmation, NOT a held-out
  generalization test (needs new cells: c32/c64, more classes, other draft/target pairs); (ii) Bc96 is a marginal
  boundary cell (+0.022, n1530) — matches oracle but with ~no margin; treat |payoff|<~0.03 as "indifferent";
  (iii) payoff at fixed K=7 — structure expected to hold if K becomes a control input, but the exact boundary should
  be re-checked.
- **Deliverable:** `docs/scheduler-signal-validation.md`. No new project item (design-validation analysis, not a
  preregistered experiment); logged here + in the doc.

### 2026-10-14 — H-CLASS-1D: held-out load-boundary validation of the scheduler signal
- **Question:** does the 2-axis rule's load axis hold OUT OF SAMPLE? Probe the A/C sign-flip interval at two
  intermediate pressures (c32 = boundary midpoint, c64 = deep-mid control), condition A. Prereg
  `docs/hclass1d-prereg.md` frozen first; same λ=C/28 mapping extended to c32→1.1429, c64→2.2857 (documented
  extension of the frozen table, not re-calibration).
- **Result:** all 4 cells DONE-VALID (c32 in 2 trials since t1 sat just outside the ±0.03 ambiguity band).
  Held-out payoffs: **A-c32 = −0.041** (t1 −0.037 / t2 −0.045, n=208), **A-c64 = −0.126** (n=120). Full A load
  curve now monotone-decreasing: c16 +0.122 → c32 −0.041 → c48 −0.157 → c64 −0.126 → c96 −0.157.
- **VERDICT: SUPPORTED** (prereg §5: p32 ≤ 0 and p64 < 0). The A/C load boundary lies in [c16, c32]; the rule's
  "low-load = {c16}" is correct — no threshold update. The load axis generalizes out of sample, upgrading the
  prior same-data design-validation to a held-out result. Boundary has a real margin (−0.041, not near-zero).
- **Scope limits:** condition A only (B positive at all loads, C mirrors A); c64 single-trial (control, far from
  0); fixed K=7 (validates existence/sign of the load flip, not an absolute threshold).
- **Incidents:** HF API rate-limit on first launch → relaunched offline (HF_HUB_OFFLINE=1, models cached); one
  transient CUDA-driver wedge (orphaned EngineCore procs from failed boots) recovered by killing orphans, no GPU
  reset. Both lessons recorded in the vllm-serving-experiments skill.
- **Deliverable:** `docs/hclass1d-results.md`. No new project item (held-out confirmation of an existing finding).

### 2026-10-14 — Phase 4 CLOSED (Issue #10 → Done)
All six sub-phases had final preregistered verdicts; this entry consolidates and closes the phase. No new runs.

| sub-phase | result | one-line finding |
|---|---|---|
| 4.1 saturation | no saturation at C≤64 | GPU util ≠ serving saturation; throughput linear in concurrency, zero wait queue (docs/phase4-p41-results.md) |
| 4.2 fixed-K × load | **H-load CONFIRMED** | optimal K flips with load; best fixed-K is workload- and load-dependent (docs/phase4-p42-results.md) |
| 4.3 SPEED-Bench long-prefill | second load axis | long prefills induce a distinct regime + per-request structure (docs/phase4-p43-results.md) |
| 4.4 heterogeneous ISL×OSL | H-4.4a/b FALSIFY | mixed ISL×OSL does not change the K-optimum story (docs/phase4-p44-results.md) |
| 4.5 DSpark-rule vs LDM | complete | rule baselines characterized; motivates live-load-aware control (docs/phase4-p45-results.md) |
| 4.6 live-load-aware adaptive spec | **H-4.6 FALSIFIED as specified** | load signal works; per-request greedy allocation fails at C=96 (negative externality) + fixed spec-path overhead no K-only controller can remove (docs/phase4-p6-results.md) |

**Phase 4 synthesis:** (i) optimum speculation is a function of *serving state* (load regime) and *workload class*,
not a constant — establishing the design space for the scheduler. (ii) Per-request greedy K allocation cannot
reach the saturation optimum because each request's argmax ignores the slowdown its drafts impose on other
requests (externality), and the spec path carries a fixed overhead only a scheduler-level enable/disable can
remove. Both point to **batch/workload-regime-level control** — exactly the signal later validated in H-CLASS-1 /
scheduler-signal-validation / H-CLASS-1D. Phase 4 therefore closes as the *measurement foundation* for the
capstone (Phase 6: load×class live SD controller). Issue #10 → Done; project item updated.

### 2026-10-14 — Phase 7 (capstone) SPEC: request-aware SD scheduler from live serving state
- **Goal:** turn the *validated* 2-axis signal into a *working in-loop controller* and show realized end-to-end
  serving gain — the original mission made concrete. Prereg: `docs/phase7-capstone-prereg.md` (design + success/falsification
  frozen; [CAL] block filled from a smoke step before any controller trial).
- **Key discovery:** Track B already built the live per-step SD on/off *mechanism* in the isolated v0.25.0 worktree
  (`sd_enabled` V2 patch, causal-verification logging OFF⇒0 drafts, probe-based re-entry; 20 tests passing). The capstone
  **reuses that mechanism and swaps only the trigger** — Track B's reactive TPOT-trend rule → the validated 2-axis rule.
- **Controller:** `sd_enabled ← load_LOW OR class_HIGHREUSE` per scheduling step, measured inputs only (no hardcoded
  concurrency): load signal = KV-usage/occupancy (continuous; deliberately NOT the schedule()-cadence signal Track B
  Defect 2 showed decouples from GPU time under async scheduling); class signal = rolling draft-acceptance ratio
  (load-invariant → carries class, not load). Orthogonal by construction — the same orthogonality that beat single-axis
  on the payoff surface. Hysteresis/probe/cooldown reused from Track B; cold start ON.
- **Workload (capstone-specific):** MIXED traffic alternating diverse (A/C-like) and high-reuse (B-like) blocks across
  three applied loads (c16/c48/c96, frozen λ=C/28) — a homogeneous workload cannot test a 2-axis rule. Prediction:
  CTRL-LIVE ON in high-reuse-at-high-load (class rescues) and OFF in diverse-at-mid/high (load), the live analog of the
  payoff-surface cells.
- **Arms:** AR / DSpark-K7 / CTRL-ON (overhead) / CTRL-LOADONLY / CTRL-CLASSONLY / CTRL-LIVE. The two single-axis arms
  are the capstone addition — S3 tests "both axes necessary" END-TO-END, not just statically.
- **Success (frozen):** S1 crossover recovery (≥AR within CI + >DSpark-K7 2/3 rounds at high-load diverse); S2 low-load
  preservation; S3 CTRL-LIVE beats BOTH single-axis arms in aggregate realized gain; S4 overhead ≤1%; S5 causal
  verification passes. Falsified if S1 or S3 fail beyond uncertainty (one revision only on harness bug).
- **Open design questions (resolve by smoke test before freezing [CAL]):** (1) which load signal separates c16/c48/c96
  cleanest without tracking schedule() cadence; (2) whether acceptance-with-decay across OFF periods is stable enough or
  needs a prefix-cache-hit-rate secondary proxy. Recorded as findings if non-discriminating, not silent retunes.
- **Status:** SPEC only — no runs yet. Next: smoke step to fill [CAL], then the comparison run set.

### 2026-10-14 (cont.) — Phase 7 [CAL] block filled offline (both signal questions resolved)
Both capstone design questions resolved from EXISTING H-CLASS-1 data (analysis.json + per-cell metrics.jsonl + obs
traces) — stronger than a new smoke run. Frozen in `docs/phase7-capstone-prereg.md` §7:
- **[CAL-1] Load signal = `kv_cache_usage_perc`** (primary; occupancy secondary). Steady medians c16/c48/c96 =
  0.303/0.929/0.983; waiting FAILS (c48=0). Hysteresis θ_low_on=0.55 / θ_low_off=0.75 (mid-gap, ~2× margin each side).
- **[CAL-2] Class signal = rolling draft-acceptance ratio, θ_acc=0.45** (A/C max 0.335 ≪ 0.45 ≪ B min 0.599; load-invariant
  by construction -> carries class not load).
- **Remaining open (needs live controller):** mixed-stream bimodality + flapping across OFF periods — confirmed by the
  first CTRL-LIVE smoke trial before the full run set (if it flaps, add prefix-cache-hit-rate secondary proxy then).
Next: implement the 2-axis trigger in the Track B worktree (swap TPOT-trend rule -> load_LOW OR class_HIGHREUSE),
unit-verify causal path, mixed-stream smoke, then the 6-arm comparison run set.

### 2026-10-14 (cont.) — Phase 7 mixed-stream SMOKE DONE → NEGATIVE for the reactive class axis (S3 falsified in this form)
Implemented the capstone 2-axis controller (`phase7_controller.py` in dspark-native-validation worktree, branch
track-b-live-sd, commit 6ff5b0a; `sd_enabled <- load_LOW OR class_HIGHREUSE`, reuses Track B cooldown+probe re-entry).
24 new unit/integration tests pass (44 total incl. 20 existing Track B); S5 causal assertion verified offline
(OFF ⇒ 0 drafts scheduled, ON ⇒ K=7 drafts). Ran the live mixed-stream smoke (`experiments/phase7_smoke.sh`, one DSpark-K7
`live` server per scenario, prefix caching ON):

| scenario | kv p50 | SD-on % | acc_ratio global max | steps > θ_acc(0.45) | on/off transitions | causal |
|---|---:|---:|---:|---:|---:|---|
| HI-MIX (c96, mixed) | 0.987 | 1% | 0.295 | 0 / 379 | 13 / 3100 | ✓ |
| LO-MIX (c10, mixed) | 0.231 | 100% | 0.555 | 166 / 1196 | 0 / 1200 | ✓ |

**Finding:** at high load the reactive class signal NEVER crosses θ_acc in any stretch (global max 0.295), even though
H-CLASS-1 measured steady-state B@c96 acceptance at **0.617** (same traffic, same load, always-on). The controller stays
OFF ~99% of the time and the class axis is non-functional exactly where it must work. Root cause: OFF periods cool the
prefix cache + 3-step probes too short/noisy → self-reinforcing OFF lock. **Wiring verified correct** (LO-MIX acceptance
matches H-CLASS-1 c16 range). **Consequence:** S3 ("both axes necessary") is FALSIFIED in this reactive form — CTRL-LIVE ≡
CTRL-LOADONLY on the mixed workload. The load axis works perfectly (no flapping, 13 transitions/3100 steps); S5 passes.

**Design lesson (not a retune):** a class signal from SD's own acceptances can't be measured while SD is off — the state it
must predict. Class needs an SD-independent proxy (prefix-cache hit rate). Reframes claim to "class must be inferred from
SD-independent state." The full 6-arm run set is NOT launched (would only formally confirm S3's failure) pending a
class-signal redesign decision.

Next: decide class-signal redesign (prefix-cache-hit-rate proxy vs reframe capstone as load-axis-only + documented
negative on reactive class axis). `docs/phase7-capstone-prereg.md` [CAL-2b] records the finding; new harness
`experiments/phase7_smoke.sh` + `experiments/phase7_mixed_client.py` + shared `experiments/hclass1_prompts.py`.

### 2026-10-14 (cont.) — Phase 7 class axis REBUILT on prefix-cache hit-rate + CONFIRMED (S3 now testable)
Per user decision, rebuilt the class axis on an SD-independent proxy (prefix-cache admission hit-rate) after [CAL-2b]
showed the reactive acceptance signal fails at high load. Worktree commit 2ab37e7:
- **Premise check** (c96 mixed, SD off): reusable B prefixes SURVIVE eviction from 96 diverse A requests; per-request
  admission hit-rate bimodal (623 @~0 / 659 @~1); 2s windowed mean tracks the traffic mix. -> redesign is viable.
- **Controller rebuilt**: class signal = rolling mean admission prefix-hit-rate, theta_hit=0.45; both axes now directly
  observable with SD off -> dropped Track B cooldown+probe (simple OFF->ON cooldown flap guard; load axis bypasses it).
  scheduler.py feeds prev-step admitted-request hit-rate (accumulated in waiting loop); env-gated admission trace added.
- **Confirmation smoke** (DSpark-K7 live, same scenarios): HI-MIX c96 SD-on **1% -> 24%** (class axis now fires at high
  load for reusable traffic), LO-MIX c10 100% ON (load axis). Causal OFF=>0 drafts PASSES both. S5 ok.

Result: the exact [CAL-2b] failure is fixed — class axis works at high load. S3 (both axes necessary) now testable:
CTRL-LIVE turns SD on for reusable traffic at high load where CTRL-LOADONLY stays off. Next: run the full 6-arm S3
comparison set to measure end-to-end realized gain + confirm CTRL-LIVE beats both single-axis ablations in aggregate.
docs/phase7-capstone-prereg.md [CAL-2c]; new experiments/phase7_premise_check.sh; smoke harness field fixed (hit_rate).
