# Serving-load characterization (Qwen2.5-7B-Instruct, 1× RTX 3090, vLLM 0.19.1, eager)

**Question (P1):** where does this model/hardware transition from load-proportional to saturated serving, and by
which signals? AR baseline (no SD), mixed workload (ISL~50 tok, OSL 256, seed 1234), `--max-num-batched-tokens
2048`, warmup 20s excluded.

## Data (C = 1 → 128; P1 + P1b)

| C | out tok/s | per-req tok/s | TTFT p50/p95 (ms) | TPOT p50/p95 (ms) | running p50 | waiting p50/max | KV max | GPU util |
|---|---|---|---|---|---|---|---|---|
| 1 | 51.2 | 51.2 | 42/44 | 20.0/20.1 | 1 | 0/0 | 0.002 | ~100%* |
| 4 | 204.8 | 51.2 | 50/62 | 20.3/20.3 | 4 | 0/0 | 0.009 | 100% |
| 8 | 380.3 | 47.5 | 56/75 | 20.7/20.7 | 8 | 0/0 | 0.018 | 100% |
| 16 | 737.3 | 46.1 | 74/94 | 21.4/22.3 | 16 | 0/0 | 0.036 | 100% |
| 32 | 1392.6 | 43.5 | 106/136 | 21.7/24.4 | 32 | 0/0 | 0.071 | 100% |
| 48 | 1966.1 | 41.0 | 128/189 | 23.0/25.6 | 48 | 0/0 | 0.104 | 100% |
| 64 | 2621.4 | 41.0 | 154/192 | 23.9/24.1 | 64 | 0/0 | 0.142 | 100% |
| 96 | 2646.6 | **27.6** | 209/273 | **35.4/36.0** | 96 | 0/0 | 0.213 | 100% |
| 128 | 3024.7 | **23.6** | 288/373 | **38.8/39.2** | 128 | 0/0 | 0.284 | 100% |

\* nvidia-smi kernel occupancy; see measurement note below.

## Regime determination (pre-specified rules from PROGRESS.md)

- **LOW / memory-bound:** C ≤ 8. Throughput exactly ∝ C (per-request flat at ~51 tok/s), TPOT flat (~20ms),
  TTFT ≈ constant. Decode steps are latency-bound, not compute-bound.
- **TRANSITION:** C = 16–48. Throughput still ~linear but per-request TPS starts falling (46→41), TPOT drifts
  up (+15%), TTFT grows with queueing of prefills.
- **SATURATED / compute-bound:** C ≥ 96. **Throughput plateaus** (2621 → 2647 from C=64→96, i.e. +32 requests
  buy +1% tokens/s; linear extrapolation would predict ~3930) while **per-request TPS collapses** (41 → 27.6,
  −33%) and **TPOT inflates +48%** (23.9 → 35.4ms p50). At C=128: per-req 23.6 tok/s (−43% vs C=64), TPOT
  38.8ms (+63%).

**Boundary: saturation begins between C=64 and C=96.** The operative signals are *throughput sub-linearity* +
*TPOT growth*, NOT queue depth: waiting stayed 0 at all C (MAX_NUM_SEQS=128 ≥ C) and KV usage stayed ≤28% —
this is **compute saturation** (each decode step now processes up to 9×C verification/decode tokens; step wall
time grows, per-request rate falls), not capacity/KV saturation.

## Measurement notes & lessons

1. **nvidia-smi GPU util is not a load signal here**: it reads ~100% from C=1 (kernel occupancy of small
   batches still saturates the SMs briefly each step). Use achieved concurrency, waiting queue, per-request TPS,
   and TPOT growth instead.
2. **PA-grid GPU bug (corrected 2026-10-07):** the old sampler omitted `--id`, averaging in the idle second GPU
   → "44%" readings were halved. Raw logs confirm GPU0 at 92–100% at C=16. PA run #2's `gpu_util_mean` column
   is not citable; all other PA metrics valid.
3. **Prefill queuing does not hurt decode** (TTFT grows ~linearly while TPOT stays flat until compute
   saturation) — chunked prefill (budget 2048) isolates the two, as designed in Sarathi-style schedulers. This
   is the baseline behavior that SD verification work could perturb (H-S2/H-S3 territory).

## Consequences for P2 (K × load matrix)

- **C_low = 8** (per-req within 10% of C=1), **C_med = 32**, **C_sat = 96** — per the pre-registered selection
  rules.
- Workload decision rule (pre-registered): P1b DID saturate the short mixed workload (compute regime) → P2 uses
  it. Extension to SPEED-Bench 2k ISL (real prefill contention + longer residency) follows as P3/P4, same harness.
- **Hypothesis sharpened:** at C=96 each decode step verifies up to (1+K)×96 tokens; K=8 → ~864 verification
  tokens/step vs 96 for AR. If step time grows super-linearly in total batched tokens, the marginal cost of K
  rises with load and argmax_K should shift toward smaller K at C_sat. Falsified if argmax_K stays K=8 (or AR)
  across all three loads within trial uncertainty.
