# Phase 6 — Preregistration: Reproduction of the native fixed-K DSpark crossover

- **Date:** 2026-10-09 (KST)
- **Status:** PREREGISTERED — no trial results were seen when this document was finalized.
- **Directive:** "Hermes Research Directive — Native DSpark Baseline and Next Research Milestone", §3 (highest priority: reproduce the native DSpark crossover).
- **Supersedes nothing.** Gate E (`docs/dspark-native-results.md`) remains the exploratory single-trial record; this experiment exists to test whether its sign pattern is real.

## 1. Question

Is the load-dependent crossover of native fixed-K=7 DSpark versus AR (Gate E: +61.5% / +12.2% / −4.5% at C=8/32/64, single trial each) reproducible across ≥3 independent repetitions per cell, and is the C=64 deficit larger than run-to-run uncertainty?

## 2. Design (fixed before any run)

| Factor | Value |
|---|---|
| Target model | `Qwen/Qwen3-4B` (unchanged weights) |
| Drafter | `deepseek-ai/dspark_qwen3_4b_block7`, K=7 fixed |
| Runtime | vLLM v0.25.0+cu126 source build, isolated env `/home/junior1/_dev/dspark-native-validation` (torch 2.11.0+cu129) |
| GPU | RTX 3090 #1 (SM86), `CUDA_VISIBLE_DEVICES=1`, TP=1 — same GPU as Gate E |
| Server config | Identical to `/tmp/run_dspark_gate.sh` / AR twin: `--gpu-memory-utilization 0.90 --max-num-seqs 64 --max-num-batched-tokens 4096 --enable-prefix-caching`; DSpark arm adds `--speculative-config '{"method":"dspark","model":"deepseek-ai/dspark_qwen3_4b_block7","num_speculative_tokens":7}'` |
| Workload | ShareGPT_V3 (repo `data/ShareGPT_V3_unfiltered_cleaned_split.json`), 60 prompts, greedy (`--temperature 0 --ignore-eos`), infinite request rate, `--seed 0` (bench default; made explicit) |
| Concurrency cells | C ∈ {8, 32, 64} |
| Arms | AR (no spec config) and DSpark fixed K=7 |
| Repetitions | **3 independent trials per (arm, cell)** → 18 bench runs total |

**Trial independence.** vLLM 0.25.0 exposes no prefix-cache reset endpoint (verified in source), so each trial uses a **fresh server start** (clean prefix cache, clean spec counters). This also resets CUDA-graph warm state identically across arms.

**Ordering (counterbalanced, fixed):** three rounds; within a round the C-block order varies and arm order alternates:

- Round 1: C=8 (AR→DSpark), C=32 (AR→DSpark), C=64 (AR→DSpark)
- Round 2: C=64 (DSpark→AR), C=32 (DSpark→AR), C=8 (DSpark→AR)
- Round 3: C=32 (AR→DSpark), C=64 (AR→DSpark), C=8 (AR→DSpark)

Each (arm, cell) appears exactly once per round; each arm leads in 1 of 3 rounds… (AR leads in rounds 1 and 3 for a given cell is not uniform — the schedule above gives DSpark-first at C=64 only in round 2 and AR-first in rounds 1, 3; any residual order bias is addressed by reporting per-round values alongside pooled means.)

**Warmup protocol:** identical to Gate E (bench default `--num-warmups 0`); the first requests of each trial are part of the measurement, exactly as in Gate E.

## 3. Metrics

From `vllm bench serve --save-result` JSON per run:
- `output_throughput` (server-side generated tok/s — **primary**), `request_throughput`, duration
- TTFT / TPOT / ITL / E2E mean + median, p50/p98 percentiles
- `completed`, `failed`, `max_concurrent_requests` (actual concurrency), spec counters (`spec_decode_draft_tokens`, `spec_decode_accepted_tokens`, acceptance rate/length)

From the 0.5 s `/metrics` + nvidia-smi scraper (`experiments/scrape_metrics.py`) per run:
- gauges: `num_requests_running`, `num_requests_waiting`, `kv_cache_usage_perc`, preemptions
- counters (deltas): generation tokens, spec draft/accepted tokens
- GPU: utilization %, memory MiB, power W

## 4. Validity rules (a trial is INVALID and excluded with reason if)

1. `completed < 60` or `failed > 0`;
2. server log shows OOM, crash-restart, or CUDA-graph capture failure;
3. scraper coverage < 95% of bench duration;
4. any other GPU process was running on GPU 1 during the trial (checked before start and after end).

Invalid trials are **not** replaced silently: if a cell ends with < 3 valid trials it is reported as such and the analysis uses what remains, flagged.

## 5. Analysis plan (fixed)

- Per (arm, cell): mean ± 95% CI (t-based, n=3) of `output_throughput`; report all individual trials.
- Δ(C) = DSpark throughput / AR throughput − 1 per cell; 95% CI via Welch two-sample on the 6 values.
- **Crossover reproducible** iff (a) sign pattern is (+, +, −) at C=8/32/64 as in Gate E, AND (b) Δ(C=64) 95% CI excludes 0 (i.e., the deficit is larger than run-to-run uncertainty). If (a) holds but (b) fails → "crossover sign pattern reproduced; C=64 deficit not statistically distinguishable from zero at n=3".
- **Operating-regime characterization** (no assumption that C=64 = saturation): per cell report median/95th-pct of running requests, waiting requests, KV-cache usage %, GPU util %, and preemption counts. Saturation claims must be grounded in these numbers.

## 6. Decision rule for Track B (fixed)

- **GO** — crossover reproducible as defined above: proceed to Track B Step 1 (feasibility audit of per-iteration K adaptation).
- **REVISE** — sign pattern breaks, or C=64 deficit is within noise: before any controller work, re-examine the premise and write a dated revision of this prereg stating what was measured and which condition actually shows a reproducible DSpark-vs-AR gap (if any). No threshold tuning.

## 7. Preservation

- Raw per-run JSON + scraper JSONL + server logs → `results/gate_e_repro/` (local, gitignored per repo convention).
- This document is updated with an **append-only** results section after analysis; the preregistered sections above are not edited.

---

## 8. RESULTS (appended 2026-10-09, post-run — prereg sections untouched)

### 8.1 Execution record (transparency)

- First pass of the orchestrator lost **all DSpark trials** to a shell-quoting bug in my harness (`--speculative-config` JSON unquoted → argparse parse error at server start). No DSpark data was produced; failure logs preserved as `*.failed1`. Five AR trials from that pass completed validly and were kept.
- Fix (bash array quoting) verified by a standalone smoke test (server healthy + real draft generation: 56 draft / 18 accepted tokens) before relaunch. Remaining 13 trials then ran clean.
- **All 18 final trials valid** under §4 rules: completed=60, failed=0 in every trial; no OOM/crash (server logs clean); scraper coverage full active window; GPU exclusive (memory verified free between trials).
- Order deviation from §2: round-1 DSpark trials ran last (after the fix), not in their preregistered position. Per-round values are reported below as required by §2's own caveat.

### 8.2 Primary result — output throughput (tok/s), mean ± 95% CI (t, n=3)

| C | AR | DSpark K=7 | Δ vs AR | Welch 95% CI on Δ |
|---|---|---|---|---|
| 8 | 459.6 ± 4.7 | **765.6 ± 1.0** | **+66.6%** | [+65.1%, +68.1%] |
| 32 | 872.3 ± 7.8 | **986.2 ± 33.3** | **+13.1%** | [+7.7%, +18.4%] |
| 64 | **983.4 ± 3.0** | 961.3 ± 25.9 | **−2.2%** | [−6.1%, +1.6%] |

Individual trials (tok/s): AR C=8: 462.8/458.1/457.8 · DSpark C=8: 765.2/766.3/765.3 · AR C=32: 874.7/875.2/866.9 · DSpark C=32: 1003.4/990.6/964.7 · AR C=64: 984.7/981.4/984.0 · DSpark C=64: 943.6/969.8/970.6.

Paired sign by round: DSpark<AR at **3/3 rounds for C=64**, 0/3 for C=8 and C=32 (every one of the six C=64 DSpark trials is below every AR trial).

### 8.3 Decision rule applied (§6)

- (a) sign pattern (+, +, −): **REPRODUCED** — and stronger than Gate E's single trial at low load (+66.6% vs +61.5%).
- (b) Δ(C=64) 95% CI excludes 0: **NOT MET** — [−6.1%, +1.6%] includes zero at n=3.
- **Verdict per the preregistered rule: "crossover sign pattern reproduced; C=64 deficit not statistically distinguishable from zero at n=3" → REVISE branch.** The Gate E −4.5% single-trial figure is NOT confirmed as a significant deficit; the reproducible, uncertainty-exceeding statements are: DSpark wins decisively at C=8 and C=32, and at C=64 it is **directionally** ≤ AR (3/3 paired) but within noise.

### 8.4 Operating-regime characterization (§5 — no saturation assumed)

Active-window scraper aggregates (median across trials):

| C | arm | running med/p95 | waiting med/max | KV usage med/p95 | GPU util | preemptions |
|---|---|---|---|---|---|---|
| 8 | AR/DSpark | 8 / 8 | 0 / 0 | ~0% / ≤0.1% | 100% | 0 |
| 32 | AR | 14 / 32 | 0 / 11 | ~0.1% / 0.1% | 100% | 0 |
| 32 | DSpark | 27.5 / 32 | 0 / 1 | ~0.1% / 0.1% | 100% | 0 |
| 64 | AR | 14 / 44 | 0 / 21 | ~0.1% / 0.1% | 100% | 0 |
| 64 | DSpark | 28 / 50 | 0 / 30 | ~0.1% / 0.2% | 100% | 0 |

**C=64 is GPU-compute-saturated (util 100%) but NOT KV-saturated (usage ≤0.2%, zero preemptions).** DSpark keeps a larger in-flight batch (running p95 50 vs 44) and pays for it: TPOT 47.2ms vs AR 34.9ms (+35%), TTFT 1358 vs 1263ms, e2el 6374 vs 5224ms. The regime is "compute-bound with queue buildup", not memory pressure.

### 8.5 Mechanistic finding (the important one)

Per-position acceptance rates are **load-invariant** across C=8/32/64 (mean of trials):
`0.72 → 0.45 → 0.27 → 0.16 → 0.09 → 0.05 → 0.03` at every concurrency; acceptance length ≈ 2.77 everywhere.

Consequences:
1. The crossover is **not** a draft-quality phenomenon — the drafter's learned profile does not degrade under load on this workload. It is a **verification-compute-amortization** phenomenon: at compute saturation, verifying K=7 positions costs ~47ms/step regardless of the fact that only ~2.8 positions are accepted.
2. **TPOT crosses over earlier than throughput.** DSpark beats AR on TPOT at C=8 (9.0 vs 13.6ms) but *loses* at C=32 (26.2 vs 21.4ms) and C=64 (47.2 vs 34.9ms), while throughput only crosses between 32 and 64. Per-token latency is the sharper, more reproducible signal of the regime boundary.
3. For a live-state controller this implies: an acceptance-rate trigger will **not** fire (the signal is flat); the decision must key on **measured verification cost / TPOT trend / compute saturation** — which is exactly what the Track B prereg's state list already includes (TPOT trend, batch occupancy, prefill share), and it strengthens the case for dropping raw acceptance ratio as the primary trigger.

### 8.6 Premise revision (dated correction per directive §3: "If the crossover is not reproducible, revise the experimental premise before developing a new controller")

- **Revised premise:** On this runtime/workload, native fixed-K=7 DSpark's advantage over AR degrades monotonically with compute load and turns into a small, directionally-consistent but uncertainty-burdened deficit by C=64. The exploitable gap for a live-state controller is the **C≈32–64 band**, where (i) TPOT is already worse than AR (measured), (ii) acceptance length stays ~2.8 (so K=7 pays for ~4 unamortized positions per step), and (iii) GPU is compute-saturated so verification cost is not free.
- **Track B GO/REVISE resolution:** proceed to Track B Step 3 implementation, but with the preregistered controller's trigger revised per §8.5 before calibration: primary state = measured TPOT trend + batch occupancy + prefill share (rolling window); acceptance ratio retained as a secondary guard only. The success criteria in `docs/trackb-prereg.md` §6 are unchanged and still binding; the [CAL] block must be filled from this analysis before any controller trial.
- **Not claimed:** statistical significance of the C=64 deficit (CI includes 0 at n=3); generalization beyond this model pair/workload/hardware.

### 8.7 Data locations

`results/gate_e_repro/`: 18 bench JSONs, 18 scraper JSONLs, server logs (failed first-pass dspark logs as `*.failed1`), `status.log`. Raw data local only (gitignored).
