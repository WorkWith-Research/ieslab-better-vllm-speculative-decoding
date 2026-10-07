# Phase-1 Results — K-sweep (speculation length × load)

Model: Qwen/Qwen2.5-7B-Instruct · drafter: ngram self-drafting · 2×RTX3090, GPU0
Server: max-model-len 8192, max-num-seqs 32, max-num-batched-tokens 2048, prefix-caching on
Workload: Poisson arrivals, 150 prompts/run. **v2 = random templated prompts** (in≈512/out≈384);
**v3 = ShareGPT** (realistic conversational text). K ∈ {1,2,4,8} × rate ∈ {2,6,12,24} req/s.

## v2 (random prompts) — COMPLETE, 16/16 runs

### Output throughput (tok/s)
| rate | K=1 | K=2 | K=4 | K=8 | best-K |
|---|---|---|---|---|---|
| r2  | 714 | 731 | 719 | **756** | K=8 |
| r6  | 879 | 839 | 935 | **962** | K=8 |
| r12 | 891 | 864 | 938 | **1038** | K=8 |
| r24 | 869 | 844 | 975 | **1049** | K=8 |

### Mean accept length (tokens/step) — scales ~linearly with K, flat across load
| rate | K=1 | K=2 | K=4 | K=8 |
|---|---|---|---|---|
| r2  | 1.96 | 2.89 | 4.67 | 7.74 |
| r6  | 1.94 | 2.86 | 4.55 | 7.22 |
| r12 | 1.96 | 2.90 | 4.59 | 7.56 |
| r24 | 1.96 | 2.89 | 4.57 | 7.45 |

draft_acc_rate (per-position): K=1≈0.95 → K=8≈0.82. GPU util ≈93–95% throughout;
KV-cache util max ≈0.20; no preemptions at any load.

### Finding: **no interior optimum-K, no load-driven shift** — on this workload
ngram acceptance on templated random prompts is **high (~0.9+) and nearly uniform across
load**. With near-perfect drafts, longer K keeps paying off (each extra drafted token is
almost always accepted), so **K=8 wins at every load level**. The "optimum-K moves with
load" pattern the theory predicts does **not** appear here — there is no over-speculation
penalty to create an interior optimum.

Load-dependent signal that *does* exist: **p99-TPOT tail penalty grows with K**
(K=8 ≈ 72–77 ms vs K=1 ≈ 43–53 ms) — a tail-latency/SLO tradeoff, but weak against the
throughput gain on this workload.

### Interpretation
The result is workload-bound: **where the optimum-K sits (or whether one exists at all)
is governed by how low and how heterogeneous the per-request acceptance rate is.** Random
templated prompts are an easy draft case → no shift. To observe the predicted dynamic-K
benefit we need a regime with lower, more varied acceptance — which ShareGPT's diverse
conversational text provides (see v3 below).

## v3 (ShareGPT) — RUNNING
Motivation: realistic diverse prompts should give **lower and more heterogeneous** ngram
acceptance. Smoke check (K=8, r6): mean accept len ≈ **3.0 tok/step**, draft_acc_rate ≈ 0.38
(vs K=8 random: 7.5 / 0.82) — i.e. a ~2.5× lower acceptance regime where an interior
optimum-K and load-driven shift are expected to emerge.

Full grid pending; results will be appended here with the same pivots + optimum-K-shift table.

## AR baseline (no speculative decoding), random prompts
| rate | out_tok/s | mean_tpot_ms | SD K=8 speedup |
|---|---|---|---|
| r2  | 546 | 25.3 | 1.38× |
| r6  | 695 | 27.2 | 1.38× |
| r12 | 725 | 26.6 | 1.43× |
| r24 | 730 | 26.5 | 1.44× |

(see `results/A_ar_r*/`). SD at K=8 beats AR by ~1.4–1.44× on output throughput at matched
load — the headroom a scheduler must not waste by mis-allocating K.

## How to reproduce
```
./experiments/phase1_sweep_v2.sh            # random (v2)
./experiments/phase1_sweep_v3_sharegpt.sh   # ShareGPT (v3)
.venv/bin/python experiments/analyze.py     # pivots + optimum-K shift -> results/summary.csv
```
