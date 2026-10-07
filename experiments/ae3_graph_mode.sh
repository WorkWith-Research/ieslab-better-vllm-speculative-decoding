#!/usr/bin/env bash
# Skeptic pass AE3: does the fixed-K ranking hold in CUDA-graph mode (vs eager)?
# LDM cannot run here (variable-K needs eager); only the fixed arms + AR.
# If the K ranking flips between eager and graph, the "best fixed K" baseline is
# execution-mode-dependent and all live claims must state the mode explicitly.
set -uo pipefail
cd "$(dirname "$0")/.."
DUR=120; CONCURRENCY=16; WARMUP=20

run_one() {
  local name="$1"; local spec="$2"; local K="${3:-none}"
  mkdir -p "results/$name"
  echo "=== $name (spec=$spec K=$K, GRAPH mode) ==="
  GPU=0 PORT=8100 MODEL=Qwen/Qwen2.5-7B-Instruct SPEC_METHOD="$spec" K="$K" \
    MAX_MODEL_LEN=8192 CHUNK_TOKENS=2048 \
    ./experiments/run_server.sh "$name"
  local up=0
  for i in $(seq 1 60); do
    if curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8100/health | grep -q 200; then up=1; break; fi
    sleep 5
  done
  [ "$up" = "1" ] || { echo "SERVER FAILED TO START $name"; tail -20 "logs/$name.server.log"; return 1; }
  .venv/bin/python experiments/mixed_workload_client.py 8100 "$DUR" "$CONCURRENCY" "$WARMUP" | tee "results/$name/client.out"
  sleep 3
  kill $(cat "results/$name/server.pid") 2>/dev/null; wait 2>/dev/null
  sleep 6
}

run_one ae3_ar   none  none
run_one ae3_k1   ngram 1
run_one ae3_k4   ngram 4
run_one ae3_k8   ngram 8
echo "=== AE3 graph-mode ranking check done ==="
