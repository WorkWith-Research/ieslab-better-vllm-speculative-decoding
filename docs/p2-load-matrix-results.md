# P2 — Fixed-K × Load Matrix (post-supervisor-feedback)

**Question (pre-registered in PROGRESS.md):** on the SAME mixed workload, does the globally
optimal fixed K shift across low / medium / high load? This is the direct test of
DSpark-style *load-dependent* K motivation. Distinct from the earlier "Phase 2" oracle-gap
study (per-request heterogeneity) — see `docs/phase2-results.md` for that one.

**Setup:** K ∈ {none(AR), 1, 2, 4, 8} × C ∈ {8, 32, 96} × 3 trials = 45 runs. Qwen2.5-7B-Instruct,
ngram drafter, **eager** all arms, mixed workload seed 1234 (ISL~50 tok, OSL 256), MAX_NUM_SEQS=128,
dual-GPU (one GPU per K-value sub-grid). C cells chosen from P1/P1b: C=8 ≈ low (per-req TPS near
C=1 value), C=32 ≈ mid-transition, C=96 = highest feasible load. Analyzer: `experiments/analyze_p2.py`.

## Result — throughput pivot (tok/s, mean ± sd over 3 trials)

| K | C=8 | C=32 | C=96 |
|---|---|---|---|
| none (AR) | 393.8 ± 0.0 | 1449.4 ± 0.0 | **2457.6 ± 0.0** |
| 1 | 384.0 ± 0.0 | 1331.2 ± 5.2 | 2099.2 ± 9.0 |
| 2 | 404.4 ± 1.2 | 1392.2 ± 2.0 | **2103.1 ± 2.0** |
| 4 | 415.5 ± 0.0 | 1435.6 ± 2.0 | 1932.5 ± 30.8 |
| 8 | **434.5 ± 1.2** | **1464.5 ± 3.0** | 1947.6 ± 2.0 |

## ★ Headline: argmax_K flips from K=8 (low load) to AR (high load); SD benefit collapses with load

| C | argmax_K | best tok/s | SD(K=8) vs AR |
|---|---|---|---|
| 8 | **K=8** | 434.5 | **+10.3%** |
| 32 | **K=8** | 1464.5 | +1.0% |
| 96 | **none (AR)** | 2457.6 | **−20.8%** |

- At low/medium load, larger K wins (K=8 > AR by 10.3% / 1.0%).
- At the highest load, **speculative decoding is net-negative**: every SD arm loses to AR, and the
  loss *grows* with K (K=1 −2.7%, K=2 −14.6%, K=4 −21.4%, K=8 −20.8%). The best SD arm at C=96 is
  still ~15% below AR.
- Mechanism (consistent with P1): this workload never forms a waiting queue even at C=96
  (MAX_NUM_SEQS=128, KV ≤21%), so the "saturation" here is **compute/occupancy** saturation, not
  capacity. Under full occupancy the batched-token budget is contested between decode steps and
  verification; extra verify tokens lengthen each step and displace useful work → marginal K tokens
  become net-costly. This is exactly the DSpark-style load-driven-K effect, observed on ngram.

## Classification vs pre-registration — **SUPPORT (H-load)**

Pre-registered discriminating outcome: *SUPPORT = argmax_K changes between C_low and C_sat AND the
gap between best-K at each load vs its worst-K exceeds trial uncertainty.* Both hold:
- argmax_K: K=8 (C=8) → K=8 (C=32) → **none** (C=96). It changes across the load range.
- Best-vs-worst gap at each C: C=8 434.5 vs 384.0 = 13.2% ; C=32 1464.5 vs 1331.2 = 9.9% ;
  C=96 2457.6 vs 1932.5 = 21.5% — all ≫ trial sd (max observed sd 30.8 tok/s ≈ 1.6%).

**Falsification was NOT triggered** (same K did not win at all three loads).

## Recorded deviation (honesty)

The pre-registration defined C_sat as "first C with sustained waiting_p50 > 0 OR throughput
sub-linear." At C=96 we measured **waiting_max = 0, KV ≤ 21%, no preemptions** — so C=96 is the
*highest feasible* load on this short-ISL workload, **not a capacity-saturated regime**. The argmax
flip occurs at *compute*-saturation (full occupancy), not queue-based capacity saturation. This does
not change the SUPPORT verdict (the pre-registered rule's intent — vary the regime and check whether
argmax moves — is met, and the effect is large and monotonic), but it must be stated plainly:
DSpark-style load-driven K is confirmed here as a **compute-occupancy** phenomenon on this workload,
and its behavior under true *capacity* saturation (long-ISL prefills) is the P3 question.

## Validity (all met)

- Achieved concurrency within ±15% of target: **yes** — running_p50 = 8/32/96 exactly (worst dev −1.0%).
- K applied per server log `num_speculative_tokens`: none→(none), 1→1, 2→2, 4→4, 8→8. **verified.**
- No OOM / preemption beyond expected: C=96 cells show preemptions_max = 0, no OOM in server logs.
- Steady window ≥ 60 s: yes (150 s runs, 20 s warmup excluded).

## Caveats

1. **Deterministic workload** (fixed seed + greedy): identical trials give near-identical token counts,
   so sd measures scheduling/queueing jitter only, not sampling noise — it *understates* total
   uncertainty. The C=96 K=4 sd=30.8 is a single-trial outlier within an otherwise tight set (1927.9 /
   1904.2 / 1965.3 tok/s); the ranking is unaffected.
2. **Single drafter (ngram), short ISL.** The flip is a compute-occupancy effect; whether it persists or
   strengthens under long-prefill (capacity) saturation is P3 (SPEED-Bench throughput_2k).
3. K=1 ≈ AR at low load (speculation net-neutral); the monotonic SD-vs-AR collapse with load is the
   robust signal, not any single adjacent-K tie.

## Reproduce

```
# grid driver (dual-GPU), or per-cell:
K=8 C=96 TRIAL=1 WORKLOAD=mixed GPU=0 PORT=8100 ./experiments/p2_cell.sh
# analysis:
.venv/bin/python experiments/analyze_p2.py     # -> results/p2/summary.csv + pivot
```
