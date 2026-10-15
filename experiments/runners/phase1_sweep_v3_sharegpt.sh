#!/usr/bin/env bash
# Phase-1 v3: K x load on SHAREGPT (realistic, diverse prompts).
# Motivation: on templated random prompts, ngram acceptance is ~0.9+ and uniform,
# so larger K monotonically wins (no interior optimum, no load-driven shift).
# ShareGPT's diverse conversational text should give lower + more heterogeneous
# ngram acceptance -> may reveal the predicted optimum-K shift with load.
set -uo pipefail
cd "$(dirname "$0")/.."

export GPU="${GPU:-0}" PORT="${PORT:-8100}"
export MODEL="Qwen/Qwen2.5-7B-Instruct"
export SPEC_METHOD="ngram"
export NUM_PROMPTS="${NUM_PROMPTS:-150}"
export WARMUP_SECS="${WARMUP_SECS:-10}"
export MAX_MODEL_LEN=8192
export DATASET=sharegpt
export DATASET_PATH="$(cd "$(dirname "$0")/.." && pwd)/data/ShareGPT_V3_unfiltered_cleaned_split.json"
export CHUNK_TOKENS=2048

echo "=== Phase-1 v3 ShareGPT K-sweep start $(date) ==="
for K in 1 2 4 8; do
  for r in 2 6 12 24; do
    export K REQUEST_RATE=$r
    EXP="S_K${K}_r${r}"
    echo ">>> $EXP (K=$K rate=$r sharegpt)"
    ./experiments/run_experiment.sh "$EXP" || echo "[FAIL] $EXP rc=$?"
  done
done
echo "=== Phase-1 v3 ShareGPT K-sweep complete $(date) ==="
