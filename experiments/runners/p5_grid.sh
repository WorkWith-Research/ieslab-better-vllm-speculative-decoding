#!/usr/bin/env bash
# P5 grid: in-loop LDM + DSpark-rule arms (AR/K4/K8 baselines reused from P2/P3 cells).
# Split by arm across GPUs: GPU A runs ldm, GPU B runs dspark (each C x 3 trials).
# Env: WORKLOAD=mixed|speedb ISL=2k CONC="32 96" ARM=ldm|dspark GPUIDX=0 PORT=8100 DUR=150
set -uo pipefail
cd "$(dirname "$0")/.."
WORKLOAD="${WORKLOAD:-mixed}"; ISL="${ISL:-2k}"
CONC="${CONC:-32 96}"
ARM="${ARM:?}"
GPUIDX="${GPUIDX:?}"; PORT="${PORT:-8100}"
DUR="${DUR:-150}"
mkdir -p results/p5

for C in $CONC; do
  for TRIAL in 1 2 3; do
    echo "[gpu$GPUIDX] dispatching arm=$ARM C=$C trial=$TRIAL"
    ARM="$ARM" C="$C" TRIAL="$TRIAL" WORKLOAD="$WORKLOAD" ISL="$ISL" \
      GPU="$GPUIDX" PORT="$PORT" DUR="$DUR" ./experiments/p5_cell.sh \
      || echo "[gpu$GPUIDX] CELL FAILED arm=$ARM C=$C t=$TRIAL"
  done
done
echo "=== P5 gpu$GPUIDX arm=$ARM done ==="
