# Phase 4.4 — Heterogeneous ISL × OSL mixed workload (results)

**Status: COMPLETE — H-4.4a FALSIFY, H-4.4b FALSIFY (as preregistered).**

## Setup (preregistered in PROGRESS.md)
Workload: 4 request classes, equal weight, round-robin per worker, seed 1234, prompts from the
ShareGPT V3 corpus (externally tokenized, sent as id arrays):

| class | ISL | OSL (max_tokens) | intent |
|---|---|---|---|
| SS | ~50 | 256 | chat turn |
| SL | ~50 | 1024 | short prompt / long generation |
| LS | ~2048 | 256 | RAG: long context, short answer |
| LL | ~2048 | 1024 | document QA, long response |

Grid: K ∈ {none,1,2,4,8} × C ∈ {32,96} × 3 trials = 30 cells, DUR=150s, eager, Qwen2.5-7B-Instruct,
ngram self-drafting (K>0), chunked prefill 2048. Steady-state window t≥20s.

**Harness note:** the first K=1/K=2 pass was invalidated by a port-collision incident (orphaned
`vllm serve` children answered the health endpoint for later cells; 13/30 cells measured the wrong
server — see PROGRESS.md 2026-10-08 incident entry, quarantined in `invalid_portcollision/`).
All 30 cells below are from the hardened re-run (own-server startup verified per cell).

## Results — steady-state aggregate tok/s (mean over trials)

| C | none (AR) | K=1 | K=2 | K=4 | K=8 | argmax |
|---|---|---|---|---|---|---|
| 32 | **1314.1** | 1022.0 | 1061.5 | 1156.5 | 1137.7 | K=none |
| 96 | **2009.9** | 1073.9 | 1228.7 | 1320.1 | 1307.4 | K=none |

AR wins at both loads by 12–46%. The SD arms are non-monotone in K: K=1 is the *worst* arm
(it pays the spec-decode path overhead for almost no accepted tokens), and K=2/4/8 cluster within
~10% of each other.

## Per-class steady tok/s (mean over trials)

C=32:

| class | none | K=1 | K=2 | K=4 | K=8 |
|---|---|---|---|---|---|
| SS | **126.0** | 105.0 | 105.7 | 115.5 | 120.1 |
| SL | **524.6** | 384.9 | 396.5 | 441.6 | 419.8 |
| LS | **139.8** | 104.4 | 110.3 | 116.2 | 115.5 |
| LL | **523.6** | 427.7 | 449.1 | 483.2 | 482.3 |

C=96:

| class | none | K=1 | K=2 | K=4 | K=8 |
|---|---|---|---|---|---|
| SS | **200.2** | 94.5 | 125.4 | 137.8 | 125.4 |
| SL | **806.9** | 377.9 | 462.7 | 511.1 | 499.2 |
| LS | **196.9** | 94.5 | 110.3 | 137.2 | 129.3 |
| LL | **806.0** | 507.0 | 530.3 | 534.0 | 553.5 |

## Hypothesis verdicts

**H-4.4a (per-class argmax_K diverges at C=96): FALSIFY.** All four classes share
argmax = K=none at C=96 (and at C=32). ISL×OSL class identity does not, by itself, change the
optimal speculation configuration on this corpus/drafter.

**H-4.4b (TPOT degradation of K=8 vs AR grows monotonically with ISL): FALSIFY.** TPOT p50
degradation at C=96: SS +64.0%, SL +39.5%, LS +67.5%, LL +42.4% — large in every class, no
monotone ISL trend. Speculation harms long-decode and short-decode requests alike here.

## Mechanism (consistent with Phases 4.2/4.3)

Measured acceptance (accepted/drafted tokens, steady window):

| C | K=1 | K=2 | K=4 | K=8 |
|---|---|---|---|---|
| 32 | 0.782 | 0.705 | 0.614 | 0.384 |
| 96 | 0.795 | 0.708 | 0.664 | 0.502 |

- ngram self-drafting acceptance decays steeply with K on this corpus (ShareGPT text is far less
  repetitive than the Phase 4.2 mixed workload). The expected gain per step, ΣJ, stays small even
  at K=8, while the verification batch B = C·(1+d) grows with d (actual drafts ≈ 5–6/req),
  pushing steps/s down. The gain never compensates the cost at either load.
- Prefill pressure shows up where Phase 4.3 predicted — as throughput loss and TPOT degradation
  under mixed long-prefill traffic — but it does *not* produce a class-specific optimal K: the
  binding constraint is draftability × load regime, not request class per se.

## Interpretation (guarded)

On ngram self-drafting / Qwen2.5-7B / RTX 3090 with this ShareGPT-derived heterogeneous workload,
AR dominates every SD arm at C=32 and C=96 in aggregate and in all four ISL×OSL classes; the
K=1 arm is worst of all. This does not mean request-level state is useless — it means class
membership (ISL/OSL) is not the right feature for K selection on this drafter/corpus; the features
that moved optimal K in Phase 4.2 were load regime and draftability, which are exactly what the
Phase 4.6 controller consumes. The per-class TPOT harm (40–68% at C=96) is a real SLO-relevant
effect even though it does not change the argmax.

## Files
- Raw: `results/p4_4/p44_k{K}_c{C}_t{T}_hetero_{client.out,metrics.jsonl,scrape.out}` (30 cells)
- Invalid first pass: `results/p4_4/invalid_portcollision/` (13 cells + README)
- Analysis: `experiments/analyze_p44.py` → `results/p4_4/summary.json`
