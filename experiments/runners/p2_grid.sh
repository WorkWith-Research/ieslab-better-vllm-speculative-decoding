#!/usr/bin/env bash
# P2: K x load matrix, 45 cells split across both GPUs (pre-registered in PROGRESS.md).
# Split is by K (each GPU owns a disjoint set of K values; all C and trials per cell):
#   GPU A: K=none,1    GPU B: K=2,4,8
# Env: WORKLOAD=mixed|speedb ISL=2k CONC="8 32 128" KS="none 1" GPUIDX=0 PORT=8100 DUR=150
set -uo pipefail
cd "$(dirname "$0")/.."
WORKLOAD="${WORKLOAD:-speedb}"; ISL="${ISL:-2k}"
CONC="${CONC:-8 32 128}"
KS="${KS:?}"
GPUIDX="${GPUIDX:?}"; PORT="${PORT:-8100}"
DUR="${DUR:-150}"
mkdir -p results/p2

for K in $KS; do
  for C in $CONC; do
    for TRIAL in 1 2 3; do
      echo "[gpu$GPUIDX] dispatching K=$K C=$C trial=$TRIAL"
      K="$K" C="$C" TRIAL="$TRIAL" WORKLOAD="$WORKLOAD" ISL="$ISL" \
        GPU="$GPUIDX" PORT="$PORT" DUR="$DUR" ./experiments/p2_cell.sh \
        || echo "[gpu$GPUIDX] CELL FAILED K=$K C=$C t=$TRIAL"
    done
  done
done
echo "=== P2 gpu$GPUIDX done ==="
