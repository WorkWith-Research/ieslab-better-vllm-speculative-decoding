#!/usr/bin/env bash
# Run one observation experiment end-to-end:
#   server (config from env) + metrics scraper + vllm bench serve (Poisson load)
# Usage:
#   EXP=e01_smoke K=3 SPEC_METHOD=eagle \
#   NUM_PROMPTS=40 REQUEST_RATE=inf DATASET=random IN_LEN=256 OUT_LEN=256 \
#   ./run_experiment.sh
set -uo pipefail
cd "$(dirname "$0")/.."

# EXP may be passed as $1 or via env; prefer $1.
if [ $# -ge 1 ] && [ -n "${1:-}" ]; then EXP="$1"; shift; fi
EXP="${EXP:?need EXP name (as \$1 or env)}"
PORT="${PORT:-8100}"
GPU="${GPU:-0}"
NUM_PROMPTS="${NUM_PROMPTS:-40}"
REQUEST_RATE="${REQUEST_RATE:-inf}"   # inf = all at once (closed loop, max concurrency)
DATASET="${DATASET:-random}"          # random | sharegpt
IN_LEN="${IN_LEN:-256}"
OUT_LEN="${OUT_LEN:-256}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-0}"  # 0 = unlimited
WARMUP_SECS="${WARMUP_SECS:-20}"

mkdir -p results/$EXP logs

# --- server ---
GPU=$GPU PORT=$PORT ./experiments/run_server.sh "$EXP" &
SERVER_ORCH=$!

echo "[exp $EXP] waiting for server health on :$PORT ..."
for i in $(seq 1 240); do
  if curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" | grep -q 200; then
    echo "[exp $EXP] server healthy after ${i}x5s"; break
  fi
  if ! kill -0 $(cat results/$EXP/server.pid) 2>/dev/null; then
    echo "[exp $EXP] SERVER DIED — see logs/$EXP.server.log"; tail -30 logs/$EXP.server.log; exit 1
  fi
  sleep 5
done

# --- metrics scraper (background) ---
.venv/bin/python experiments/scrape_metrics.py --port $PORT --gpu $GPU \
  --out results/$EXP/metrics.jsonl --interval 0.5 > results/$EXP/scraper_summary.json 2>&1 &
SCRAPER=$!

# --- warmup: let CUDA graphs / caches settle, counters start from ~steady state ---
echo "[exp $EXP] warmup ${WARMUP_SECS}s"
sleep "$WARMUP_SECS"

# --- benchmark ---
BENCH_ARGS=(--backend openai --host 127.0.0.1 --port $PORT
  --model Qwen/Qwen2.5-7B-Instruct
  --dataset-name $DATASET
  --num-prompts $NUM_PROMPTS
  --request-rate $REQUEST_RATE
  --metric-percentiles 50,90,99
  --percentile-metrics ttft,tpot,e2el
  --save-result --result-dir results/$EXP --result-filename $EXP.bench.json)
if [ "$DATASET" = "random" ]; then
  BENCH_ARGS+=(--random-input-len $IN_LEN --random-output-len $OUT_LEN --random-range-ratio 0.3)
fi
if [ "$MAX_CONCURRENCY" != "0" ]; then
  BENCH_ARGS+=(--max-concurrency $MAX_CONCURRENCY)
fi

.venv/bin/vllm bench serve "${BENCH_ARGS[@]}" > results/$EXP/bench.log 2>&1
BENCH_RC=$?
echo "[exp $EXP] bench rc=$BENCH_RC"

# --- teardown ---
kill -TERM $(cat results/$EXP/server.pid) 2>/dev/null
for i in $(seq 1 30); do
  kill -0 $(cat results/$EXP/server.pid) 2>/dev/null || break
  sleep 2
done
kill -9 $(cat results/$EXP/server.pid) 2>/dev/null
wait $SCRAPER 2>/dev/null

echo "[exp $EXP] done. artifacts in results/$EXP/"
exit $BENCH_RC
