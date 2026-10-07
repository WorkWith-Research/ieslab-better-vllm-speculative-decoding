# Progress Log — [iESLAB] Better vLLM Speculative Decoding

GitHub Project: https://github.com/orgs/WorkWith-Research/projects/1 (projectV2 id `PVT_kwDOEyEOMs4Bl95-`)

## Project items (for API updates)

| Item ID | Issue | Title | Status |
|---|---|---|---|
| `PVTI_lADOEyEOMs4Bl95-zg-9nck` | #1 | Phase 0: Environment setup | Done |
| `PVTI_lADOEyEOMs4Bl95-zg-9nc4` | #2 | Phase 1: Observation — fixed-K SD under load | Done |
| `PVTI_lADOEyEOMs4Bl95-zg-9neU` | #3 | Phase 2: Analysis — oracle gap for dynamic K | Done |
| `PVTI_lADOEyEOMs4Bl95-zg-9nfY` | #4 | Phase 3: Prototype — per-request dynamic-K controller + live validation | Done |
| `PVTI_lADOEyEOMs4Bl95-zg_JZ9c` | #10 | Phase 4: Realistic-workload validation — load, SPEED-Bench, heterogeneous ISL/OSL, DSpark | In progress |

**Current Phase-4 checkpoint:** Phase 4.1 saturation characterization, Phase 4.2 fixed-K × load, Phase 4.3 SPEED-Bench long-prefill validation, and Phase 4.5 DSpark-rule vs LDM are complete. Phase 4.4 heterogeneous ISL × OSL remains outstanding; Phase 4.6 live-load-aware adaptive speculation is the next controller experiment.

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
**Cells:** ldm_load × C ∈ {8,32,96} × 3 trials = 9 cells
**Cells:** ldm_load × C ∈ {8,32,96} × 3 trials = 9 cells, mixed workload (Phase 4.2), DUR=150s, eager. Baselines reused:
AR/K4/K8/LDM/DSpark-rule from Phase 4.2/4.5 (no re-run). GPU1.
**Success (fixed):** at C=96 ldm_load ≥ AR − noise (noise = max trial spread of AR across trials, ≈0.3%) AND beats DSpark-rule;
at C=8 ldm_load ≥ +5% vs AR (recovers most of fixed-K8's +10.3%). **FALSIFY** if it fails either bound: then the load-regime
signal as specified does not carry enough information, and we report that (do NOT retune to win).
**Tuning budget:** ≤2 revisions total for this hypothesis (charter §13); each revision must be logged with rationale.
**Interpretation guard:** if ldm_load ≈ acceptance-only LDM at C=96 (i.e., the regime term never triggers SD-off), the
hypothesis is FALSIFIED as specified — a live SPS(B) ratio does not let this controller detect saturation.
