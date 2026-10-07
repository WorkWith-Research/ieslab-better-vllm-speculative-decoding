#!/usr/bin/env bash
# Launch a vLLM server for SD observation experiments.
# Usage: run_server.sh <exp-name> [extra vllm serve args...]
# Env knobs (with defaults):
#   GPU=0  PORT=8100  MODEL=Qwen/Qwen2.5-7B-Instruct
#   SPEC_METHOD=eagle  SPEC_MODEL=<path>  K=3
#   MAX_MODEL_LEN=8192  MAX_NUM_SEQS=32  CHUNK_TOKENS=2048  MEM_UTIL=0.90
set -euo pipefail
EXP="$1"; shift || true
cd "$(dirname "$0")/.."
mkdir -p results/$EXP logs

GPU="${GPU:-0}"
PORT="${PORT:-8100}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
SPEC_METHOD="${SPEC_METHOD:-eagle}"
SPEC_MODEL="${SPEC_MODEL:-/home/junior1/.cache/huggingface/hub/models--leptonai--EAGLE-Qwen2.5-7B-Instruct/snapshots/a3479b6591d42613c622f6eb72c82208199abee2}"
K="${K:-3}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
CHUNK_TOKENS="${CHUNK_TOKENS:-2048}"
MEM_UTIL="${MEM_UTIL:-0.90}"

SPEC_ARGS=()
if [ "$SPEC_METHOD" != "none" ]; then
  if [ -n "${SYNTH_RATE:-}" ]; then
    # Controlled acceptance profile (NOTE: vLLM 0.19.1 does not actually apply
    # synthetic with ngram — see PROGRESS.md; kept for future vLLM versions).
    SPEC_ARGS=(--speculative-config "{\"method\": \"$SPEC_METHOD\", \"model\": \"ngram\", \"num_speculative_tokens\": $K, \"rejection_sample_method\": \"synthetic\", \"synthetic_acceptance_rate\": $SYNTH_RATE}")
  elif [ "$SPEC_METHOD" = "ngram" ] || [ "$SPEC_METHOD" = "suffix" ]; then
    # Self-drafting methods: no external draft model needed.
    SPEC_ARGS=(--speculative-config "{\"method\": \"$SPEC_METHOD\", \"num_speculative_tokens\": $K}")
  else
    SPEC_ARGS=(--speculative-config "{\"method\": \"$SPEC_METHOD\", \"model\": \"$SPEC_MODEL\", \"num_speculative_tokens\": $K}")
  fi
fi

CUDA_VISIBLE_DEVICES=$GPU .venv/bin/vllm serve "$MODEL" \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$CHUNK_TOKENS" \
  --gpu-memory-utilization "$MEM_UTIL" \
  --enable-prefix-caching \
  ${ENFORCE_EAGER:+--enforce-eager} \
  "${SPEC_ARGS[@]}" \
  "$@" > "logs/$EXP.server.log" 2>&1 &
echo $! > "results/$EXP/server.pid"
echo "server pid $(cat results/$EXP/server.pid), port $PORT, exp=$EXP (log: logs/$EXP.server.log)"
