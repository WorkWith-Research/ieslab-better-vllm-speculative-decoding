#!/usr/bin/env bash
# Phase 5.1 full pre-registered grid (H-5.1), one GPU per invocation.
# Usage: p51_grid.sh <GPU> <PORT> <cell-spec ...>
#   cell-spec: "C8" | "C32" | "C96" | "FOFF"  (FOFF = force-off diagnostic at C=96)
# Trials t1/t2/t3 are generated for C* cells; FOFF runs once.
set -uo pipefail
cd "$(dirname "$0")/.."
GPU="$1"; PORT="$2"; shift 2
for spec in "$@"; do
  case "$spec" in
    FOFF)
      C=96 TRIAL=f0 GPU=$GPU PORT=$PORT DUR=150 WARMUP=20 FORCEOFF=1 ./experiments/p51_cell.sh ;;
    *)
      C="${spec#C}"
      for t in 1 2 3; do
        C=$C TRIAL=t$t GPU=$GPU PORT=$PORT DUR=150 WARMUP=20 FORCEOFF=0 ./experiments/p51_cell.sh
      done ;;
  esac
done
echo "=== GRID DONE on GPU $GPU ==="
