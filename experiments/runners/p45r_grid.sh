#!/usr/bin/env bash
# P4.5R full grid: C in {32,96} x 3 trials, corrected DSpark-rule proxy (dsparkr).
set -uo pipefail
cd "$(dirname "$0")/.."
for C in 32 96; do
  for T in 1 2 3; do
    echo "########## P4.5R cell C=$C trial=$T ##########"
    C=$C TRIAL=$T GPU=1 PORT=8101 DUR=150 WARMUP=20 bash experiments/p45r_cell.sh
  done
done
echo "=== P4.5R grid complete ==="
