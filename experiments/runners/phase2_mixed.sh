#!/usr/bin/env bash
# Phase-2: one mixed-workload run with the per-request spec-decode hook active.
# Run at Kmax (default 8) so each request's full joint-acceptance profile J[0..8]
# is captured; oracle_gap.py then derives optimal-K + gap for all k<=Kmax.
set -uo pipefail
cd "$(dirname "$0")/.."

export GPU="${GPU:-0}" PORT="${PORT:-8100}"
export MODEL="Qwen/Qwen2.5-7B-Instruct"
export SPEC_METHOD="ngram"
export K="${K:-8}"
export MAX_MODEL_LEN=8192 CHUNK_TOKENS=2048

# per-request observation hook (non-invasive, env-gated)
export PYTHONPATH="$PWD/experiments/spec_hook"
EXP="${EXP:-P2_K${K}_r${RATE}}"
export VLLM_SPEC_HOOK_OUT="results/$EXP/spec_hook.jsonl"

NUM_PROMPTS="${NUM_PROMPTS:-150}" RATE="${RATE:-6}" EASY_FRACTION="${EASY_FRACTION:-0.5}" MAXTOK="${MAXTOK:-256}"
CLIENT="${CLIENT:-spectrum}"   # spectrum | sharegpt

mkdir -p results/$EXP
: > "results/$EXP/spec_hook.jsonl"

echo ">>> $EXP (K=$K rate=$RATE client=$CLIENT n=$NUM_PROMPTS) [hook on]"
./experiments/run_server.sh "$EXP" &
for i in $(seq 1 240); do
  curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" | grep -q 200 && break
  sleep 5
done
sleep "${WARMUP_SECS:-10}"

if [ "$CLIENT" = "sharegpt" ]; then
  .venv/bin/python experiments/sharegpt_client.py \
    --port "$PORT" --model "$MODEL" --num-prompts "$NUM_PROMPTS" --rate "$RATE" \
    --max-tokens "$MAXTOK" --out "results/$EXP/client.jsonl" --seed "${SEED:-0}"
else
  .venv/bin/python experiments/spectrum_client.py \
    --port "$PORT" --model "$MODEL" --num-prompts "$NUM_PROMPTS" --rate "$RATE" \
    --max-tokens "$MAXTOK" --out "results/$EXP/client.jsonl" --seed "${SEED:-0}"
fi

kill -TERM $(cat results/$EXP/server.pid) 2>/dev/null
for i in $(seq 1 30); do kill -0 $(cat results/$EXP/server.pid) 2>/dev/null || break; sleep 2; done
kill -9 $(cat results/$EXP/server.pid) 2>/dev/null

echo ">>> $EXP done: $(wc -l < results/$EXP/spec_hook.jsonl) hook lines, $(wc -l < results/$EXP/client.jsonl) client recs"
