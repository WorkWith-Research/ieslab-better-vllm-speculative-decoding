# Observation Plan — "Look at vLLM first" (advisor's ask)

Goal: before designing any controller, *measure* where fixed-K speculative decoding
misbehaves under multi-request load in vLLM, and quantify the headroom a dynamic
reconfiguration policy could capture.

## Models & hardware

- Target: `Qwen/Qwen2.5-7B-Instruct` (fp16/bf16, ~15 GB) on GPU 0
- Draft: EAGLE via `leptonai/EAGLE-Qwen2.5-7B-Instruct` (if vLLM accepts the
  checkpoint; fallback = ngram drafter or synthetic acceptance for scheduler studies)
- Second 3090 (GPU 1): spare / second engine for A-B comparisons if needed
- Everything single-GPU first; TP=2 only if 7B+draft doesn't fit comfortably.

## Workloads

Synthetic Poisson arrivals + realistic traces, three regimes:

1. **W1 low-load** (memory-bound): concurrency 1–8, arrival rate ≤ 4 req/s
   → SD should win big; baseline sanity.
2. **W2 mid-load** (crossover zone): concurrency 8–32, Poisson arrivals tuned so the
   system sits near saturation. This is where fixed-K is expected to hurt most.
3. **W3 high-load / bursty** (compute-bound + queueing): sustained overload (arrival
   > service) and a 2× burst step; tests preemption, KV pressure, TTFT SLO violations.

Workload mix: ShareGPT prompts (realistic length distribution) + GSM8K-style math
prompts (long outputs, low acceptance for small drafts) to create *heterogeneous*
acceptance rates within a batch — the core assumption of our idea.

## Configurations (the "reconfiguration" axes we study)

| Axis | Values |
|---|---|
| SD method | AR (K=0), EAGLE K ∈ {1,3,5}, ngram K=4 (fallback), synthetic-acceptance K=5 with per-position rates {0.9,0.8,0.7,0.6,0.5} and {0.7,...,0.3} |
| Chunked prefill budget `max_num_batched_tokens` | 512 / 2048 / 8192 / off |
| Concurrency cap `max_num_seqs` | 16 / 32 / 64 |

## Metrics (all logged to per-run JSONL)

Per-step (from vLLM Prometheus, scraped at ~10 Hz):
- queue depth, running batch size, KV cache usage %, num preemptions, prefix-cache hits
- spec-decode: drafts, draft tokens, accepted tokens, acceptance rate, per-position
  acceptance (where exposed), mean accepted length τ

Per-request:
- TTFT, TPOT, ITL p50/p99, total latency, output tokens, (for math mix) correctness
- *derived*: effective verify overhead = draft_tokens/accepted_tokens

Aggregate:
- throughput (output tok/s), goodput under SLOs (TTFT ≤ 1s, TPOT ≤ 50ms — tentative),
  p99 latency, preemption count, time in overload

## Key analyses

1. **Crossover curve**: throughput & TPOT vs arrival rate for each K → find the load
   where SD(K) < AR. (Answers Q1; mirrors Nightjar Fig. 2 but on current vLLM.)
2. **Acceptance heterogeneity**: distribution of per-request τ across a run;
   correlation with prompt length, output position, task type. Quantify how much of
   the batch is "wasting" K at any step (E[τ] ≪ K+1 for some requests while others
   saturate). (Q2)
3. **Oracle gap**: replay logged per-step acceptance to compute what a clairvoyant
   per-request K policy would achieve (no implementation needed — offline counterfactual
   from the same traces). This bounds the value of *any* dynamic controller, including
   ours. (Q3)
4. **Prefill/decode interference**: TTFT vs TPOT tradeoff across chunk budgets under
   SD; does a larger prefill budget steal verify steps? (Q4)

## Deliverables

- `results/exp01_*/` … raw JSONL + metrics CSVs
- `docs/findings-phase1.md` — the "problem statement with numbers" we take to the advisor
- one figure set: crossover curves, acceptance-heterogeneity heatmap, oracle-gap bars

## Risks / mitigations

- EAGLE checkpoint incompatible with vLLM 0.31 → fall back to ngram (still real draft
  cost) + synthetic mode for controlled studies; or train a tiny EAGLE head later.
- CUDA graphs break per-step K changes → Phase 1 only *observes* fixed-K configs
  (no runtime mutation); dynamic controller comes in Phase 3 with eager/piecewise graphs.
- 7B on 24 GB leaves little KV headroom for high concurrency → use `--max-model-len`
  8k, gpu-mem-util 0.90; if KV-starved, drop to Qwen2.5-3B (drafts scale down too).
