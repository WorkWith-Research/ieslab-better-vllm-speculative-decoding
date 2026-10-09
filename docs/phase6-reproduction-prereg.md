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
