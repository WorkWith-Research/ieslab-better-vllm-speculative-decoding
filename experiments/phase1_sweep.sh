#!/usr/bin/env bash
# Phase-1 observation sweep. Two independent sub-batches run in parallel on the
# two GPUs (no cross-talk; each engine owns one GPU + its own port).
#   Batch A (main K/acceptance x load sweep)  -> GPU $GPU_A, PORT $PORT_A
#   Batch B (chunked-prefill interaction)      -> GPU $GPU_B, PORT $PORT_B
#
# All runs use ngram drafting + SYNTHETIC acceptance so we isolate the
# SCHEDULER's behavior from draft-model quality (Phase-1 goal). K fixed at 4.
set -uo pipefail
cd "$(dirname "$0")/.."

GPU_A="${GPU_A:-0}"; PORT_A="${PORT_A:-8100}"
GPU_B="${GPU_B:-1}"; PORT_B="${PORT_B:-8101}"
MODEL="Qwen/Qwen2.5-7B-Instruct"
K=4
NUM_PROMPTS="${NUM_PROMPTS:-150}"
WARMUP_SECS="${WARMUP_SECS:-10}"

run() { # $1=gpu $2=port rest...
  local gpu="$1" port="$2"; shift 2
  EXP="$1" GPU=$gpu PORT=$port MODEL=$MODEL K=$K NUM_PROMPTS=$NUM_PROMPTS \
  WARMUP_SECS=$WARMUP_SECS ./experiments/run_experiment.sh "$@"
}

echo "=== Phase-1 sweep start $(date) ==="

# ---- Batch A: acceptance-rate x load (chunk fixed at 2048) ------------------
batch_a() {
  local gpu=$GPU_A port=$PORT_A
  for rate_cfg in "synth_0.85:0.85" "synth_0.60:0.60" "synth_0.35:0.35"; do
    local name="${rate_cfg%%:*}"; local syn="${rate_cfg##*:}"
    for r in 2 6 12 24; do
      EXP="A_${name}_r${r}" GPU=$gpu PORT=$port SPEC_METHOD=ngram SYNTH_RATE=$syn \
      REQUEST_RATE=$r DATASET=random IN_LEN=512 OUT_LEN=384 MAX_MODEL_LEN=8192 \
      ./experiments/run_experiment.sh "A_${name}_r${r}" || echo "[A] ${name}_r${r} FAILED rc=$?"
    done
  done
  # AR baseline (no SD) across the same loads
  for r in 2 6 12 24; do
    EXP="A_ar_r${r}" GPU=$gpu PORT=$port SPEC_METHOD=none REQUEST_RATE=$r DATASET=random \
    IN_LEN=512 OUT_LEN=384 MAX_MODEL_LEN=8192 \
    ./experiments/run_experiment.sh "A_ar_r${r}" || echo "[A] ar_r${r} FAILED rc=$?"
  done
  echo "=== Batch A done $(date) ==="
}

# ---- Batch B: chunked-prefill budget x load (AR + mid-acceptance SD) --------
batch_b() {
  local gpu=$GPU_B port=$PORT_B
  for r in 6 12 24; do
    for chunk in 512 2048 8192; do
      EXP="B_ar_r${r}_c${chunk}" GPU=$gpu PORT=$port SPEC_METHOD=none REQUEST_RATE=$r \
      DATASET=random IN_LEN=512 OUT_LEN=384 MAX_MODEL_LEN=8192 CHUNK_TOKENS=$chunk \
      ./experiments/run_experiment.sh "B_ar_r${r}_c${chunk}" || echo "[B] ar_r${r}_c${chunk} FAILED rc=$?"
      EXP="B_synth0.60_r${r}_c${chunk}" GPU=$gpu PORT=$port SPEC_METHOD=ngram SYNTH_RATE=0.60 \
      REQUEST_RATE=$r DATASET=random IN_LEN=512 OUT_LEN=384 MAX_MODEL_LEN=8192 CHUNK_TOKENS=$chunk \
      ./experiments/run_experiment.sh "B_synth0.60_r${r}_c${chunk}" || echo "[B] synth0.60_r${r}_c${chunk} FAILED rc=$?"
    done
  done
  echo "=== Batch B done $(date) ==="
}

batch_a & PA=$!
batch_b & PB=$!
wait $PA; wait $PB
echo "=== Phase-1 sweep complete $(date) ==="
