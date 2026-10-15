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

## v3 (ShareGPT) — COMPLETE, 16/16 runs
Realistic diverse conversational prompts → **lower, more heterogeneous** ngram acceptance:
draft_acc_rate @K=8 ≈ 0.34–0.41 (vs random 0.79–0.86); mean accept len @K=8 ≈ 3.8 (vs 7.5).

### Output throughput (tok/s)
| rate | K=1 | K=2 | K=4 | K=8 | best-K |
|---|---|---|---|---|---|
| r2  | 400 | 398 | **400** | 394 | K=4 |
| r6  | 657 | 660 | **680** | 664 | K=4 |
| r12 | 664 | 677 | **730** | 693 | K=4 |
| r24 | 676 | 706 | 728 | **729** | K=8 |

### Draft acceptance rate (per-position) — the regime driver
| rate | K=1 | K=2 | K=4 | K=8 |
|---|---|---|---|---|
| r2  | 0.72 | 0.69 | 0.58 | 0.41 |
| r6  | 0.73 | 0.56 | 0.47 | 0.36 |
| r12 | 0.74 | 0.56 | 0.56 | 0.34 |
| r24 | 0.79 | 0.74 | 0.56 | 0.35 |

## ★ Headline: the optimum-K is workload-dependent (K=4/K=8 crossover)

Comparing the two regimes at matched load — which K maximizes output throughput:

| rate | random (high acc) | sharegpt (low acc) | K=8 vs K=4 |
|---|---|---|---|
| r2  | **K=8** | **K=4** | random +5.1% / sharegpt −1.6% |
| r6  | **K=8** | **K=4** | +2.9% / −2.3% |
| r12 | **K=8** | **K=4** | +10.6% / −5.1% |
| r24 | **K=8** | K=8 (tie) | +7.5% / +0.2% |

**The single variable that moves the optimum-K is the draft acceptance rate, not the load.**
- High acceptance (random): every extra drafted token is mostly accepted → longer K always wins;
  over-speculation penalty ≈ 0 → **K=8 best at all loads**.
- Low acceptance (ShareGPT): drafts fail early, so long chains waste verification compute on
  rejected tokens → an **interior optimum appears (K=4)**; only at the highest load (r24) does K=8
  edge back in (batching amortizes the draft cost).

This is exactly the headroom a dynamic scheduler should capture: **a fixed-K policy loses up to
~5–10% throughput** by picking the wrong K for the workload's acceptance regime. A Lightweight
Decision Model that reads live per-request acceptance (which we now measure) and sets K per-request
can recover it — this is the Phase-2 oracle-gap target.

### Caveats / what's NOT yet shown
1. **Load-driven shift at fixed workload is weak.** Within ShareGPT, best-K moves only K=4→K=8
   between r12 and r24 (a 2-step edge effect), not a clean monotonic shift. The dominant axis is
   *acceptance regime*, with load as a secondary modifier.
2. **Margins are modest** (2–10%) and on a single model/drafter (ngram). ngram acceptance is
   itself load-insensitive here, so we vary the regime via the dataset, not via load alone.
3. **Per-request heterogeneity** (the LDM's real target) is not yet measured — these are
   per-run aggregates. Phase 2 needs per-request accept-len distribution under mixed workloads to
   show that *within* one batch, requests want different K.
4. Single drafter (ngram). Real EAGLE drafts have lower + more variable acceptance → the crossover
   should be sharper; deferred to a later phase (EAGLE-Qwen2 incompatible with vLLM 0.19.1).

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
