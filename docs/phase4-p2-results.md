# Phase 4 — P2: fixed-K × load matrix (result)

**Question:** does the optimal speculation length K depend on serving load? (Supervisor feedback: DSpark
changes K by GPU load; our earlier experiments never actually saturated the system.)
**Setup:** Qwen2.5-7B-Instruct, 1× RTX 3090, vLLM 0.19.1 eager, mixed workload (ISL~50 tok, OSL 256),
K ∈ {none(AR),1,2,4,8} × C ∈ {8,32,96} × 3 trials, fresh server per cell. Load regimes from P1/P1b:
C=8 memory-bound, C=32 transition, C=96 compute-saturated (throughput plateau, TPOT +48%).

## Result — the optimal K flips with load

| K | C=8 tok/s | C=32 tok/s | C=96 tok/s |
|---|---|---|---|
| none (AR) | 393.8 | 1449.4 | **2457.6** ← argmax |
| 1 | 384.0 | 1331.2 | 2099.2 |
| 2 | 404.4 | 1392.2 | 2103.1 |
| 4 | 415.5 | 1435.6 | 1932.5 |
| 8 | **434.5** ← argmax | **1464.5** ← argmax (+1.0% vs AR) | 1947.6 (−21%) |

- **C=8:** K=8 beats AR by +10.3%; monotone in K. Memory-bound steps: verification tokens are nearly free.
- **C=32:** K=8 still best but only +1.0% vs AR — the SD advantage is nearly gone at the transition.
- **C=96 (saturated):** **AR wins by 14–21%** over every SD arm; larger K is worse (K=4: −21.4%, K=8: −20.8%).
  The ranking is *inverted* relative to C=8.

**H-load CONFIRMED.** This directly supports the premise of load-driven K (DSpark's motivation) and shows the
effect on our hardware/workload/drafter goes all the way to "SD off" at saturation.

## Mechanism (from cumulative spec-decode counters, steady window)

| K | C | steps/s | verified tok/step B | acc rate | per-req tok/s | TPOT p50 ms |
|---|---|---|---|---|---|---|
| none | 96 | 26.6 | 96 | — | 76.8 | 36.8 |
| 1 | 96 | 19.0 | 109.6 | 0.84 | 65.6 | 50.0 |
| 2 | 96 | 18.7 | 116.1 | 0.76 | 65.7 | 50.2 |
| 4 | 96 | 17.0 | 123.7 | 0.68 | 60.4 | 54.9 |
| 8 | 96 | 16.6 | 129.6 | 0.63 | 60.9 | 56.3 |

- Acceptance rate is **load-invariant** (0.63–0.85, stable across C) — the drafter sees the same text.
- What changes: steps/sec falls ~38% from AR to K=8 at C=96 because each step verifies B=C(1+K̄) tokens;
  in the compute-saturated regime step time grows with total batched tokens, so per-request rate collapses.
- At C=8 the same B inflation costs nothing (steps are memory-bound): steps/s stays ~42–43 and K=8 wins.

## SPS(B) profiled curve (DSpark's "load model", reconstructed on our hardware)

| B (verified tok/step) | 9–11 | 32 | 37–44 | 96 | 110–130 |
|---|---|---|---|---|---|
| steps/s | ~42–43.5 | 45.5 (AR C=32) | ~36 | 26.6 (AR C=96) | 16.6–19.0 |

SPS(B) is a strongly decreasing, non-linear curve — the input DSpark's greedy uses. Table:
`results/p2/sps_table.jsonl` (derived via exact counter identities S=(gen−acc)/C, B=C+prop/S).

## Caveats (recorded, not swept under)

1. **Deterministic workload:** fixed seed + greedy → trial variance measures scheduling jitter only (±0–3%).
   Rankings are robust (gaps ≥1%, mostly ≥5%), but a near-tie is NOT evidence of equivalence. C=32's K=8 vs AR
   gap (+1.0%) sits near the noise floor — treat as "K≈AR at transition", not "K=8 clearly wins".
2. **TTFT anomaly (unexplained, do not cite):** SD arms show *lower* TTFT than AR at C=96 (e.g. K=8 137ms vs
   AR 218ms p50) while their TPOT is worse and throughput lower. Direction is consistent across all SD arms but
   the causal mechanism is not identified. Flagged for follow-up; excluded from any claim here.
3. **ngram self-drafting specific:** acceptance depends on repetitive content (templates). The *direction* of the
   load effect should hold for any drafter (it comes from verification compute), but magnitudes are ngram-specific.
4. **Eager mode** (variable-K requirement); CUDA-graph overhead not in these numbers.

## Consequences for the LDM idea

- A policy that reads **load regime** (e.g. achieved steps/s, TPOT trend, or B vs SPS(B) knee) and sets
  K=large→small→off as load rises would capture most of the gap between "best fixed-K" and "oracle per-load K":
  at C=96 that gap is **21%** (AR 2457.6 vs best SD arm). This is the largest single effect measured in this
  project so far — bigger than any acceptance-heterogeneity effect from Phase 2/3.
- The DSpark-rule baseline (batch-greedy over prefix-survival with this SPS(B) table, no live serving state)
  can at best recover *part* of it: its objective uses B only through SPS(B), and per-request confidence may
  keep some requests at high K even when the batch is saturated. The discriminating test (P5) is whether a
  policy with **live serving state** beats DSpark-rule by more than trial noise at C=96.
