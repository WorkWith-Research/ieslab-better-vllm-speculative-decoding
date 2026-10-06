#!/usr/bin/env bash
# Phase-1 sweep v2: SPECULATION-LENGTH (K) x LOAD, real ngram acceptance.
# (v1 "vary synthetic acceptance" was invalidated: vLLM 0.19.1 does not apply
#  rejection_sample_method=synthetic with ngram — accepted tokens were identical
#  across rates. See PROGRESS.md.)
#
# K is the user's core decision variable. With real ngram acceptance (~3.75 tok/step
# on this workload), we expect: small K under-speculates, large K over-drafts
# (wastes verify compute), optimum in between — and the optimum should shift with
# load. That shift IS the oracle gap for dynamic-K scheduling.
#
# All knobs are EXPORTED so they reach run_server.sh / run_experiment.sh children.
set -uo pipefail
cd "$(dirname "$0")/.."

export GPU="${GPU:-0}" PORT="${PORT:-8100}"
export MODEL="Qwen/Qwen2.5-7B-Instruct"
export SPEC_METHOD="ngram"
export NUM_PROMPTS="${NUM_PROMPTS:-150}"
export WARMUP_SECS="${WARMUP_SECS:-10}"
export MAX_MODEL_LEN=8192
export DATASET=random IN_LEN=512 OUT_LEN=384
export CHUNK_TOKENS=2048

echo "=== Phase-1 v2 K-sweep start $(date) ==="
# K x load grid (sequential on one GPU to avoid memory contention)
for K in 1 2 4 8; do
  for r in 2 6 12 24; do
    export K REQUEST_RATE=$r
    EXP="K${K}_r${r}"
    echo ">>> $EXP (K=$K rate=$r)"
    ./experiments/run_experiment.sh "$EXP" || echo "[FAIL] $EXP rc=$?"
  done
done
