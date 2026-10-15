#!/usr/bin/env bash
# Phase 4.4 grid: K x C on the heterogeneous ISLxOSL workload, one GPU per K-group.
# Env: KS="none 1 2" CONC="32 96" TRIALS="0 1 2" GPU=0 PORT=8100 DUR=150 WARMUP=20 SEED=1234
set -uo pipefail
cd "$(dirname "$0")/.."
KS="${KS:?}"; CONC="${CONC:-32 96}"; TRIALS="${TRIALS:-0 1 2}"
GPU="${GPU:-0}"; PORT="${PORT:-8100}"; DUR="${DUR:-150}"; WARMUP="${WARMUP:-20}"; SEED="${SEED:-1234}"
echo "=== P4.4 grid start: KS=[$KS] CONC=[$CONC] GPU=$GPU PORT=$PORT ==="
for K in $KS; do
  for C in $CONC; do
    for T in $TRIALS; do
      echo "--- dispatching p44 K=$K C=$C T=$T (gpu$GPU port$PORT) ---"
      K="$K" C="$C" TRIAL="$T" GPU="$GPU" PORT="$PORT" DUR="$DUR" WARMUP="$WARMUP" SEED="$SEED" \
        ./experiments/p44_cell.sh >> "logs/p44_gpu${GPU}.cell.log" 2>&1
    done
  done
done
echo "=== P4.4 grid (gpu$GPU) done ==="
