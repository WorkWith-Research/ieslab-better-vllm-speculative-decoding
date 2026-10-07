#!/usr/bin/env bash
# P2 single cell: fixed-K x concurrency on a chosen workload (pre-registered in PROGRESS.md).
# Env: K (none|1|2|4|8) C TRIAL WORKLOAD=mixed|speedb ISL=2k DIFF=all GPU=0 PORT=8100 DUR=150 WARMUP=20
set -uo pipefail
cd "$(dirname "$0")/.."
K="${K:?}"; C="${C:?}"; TRIAL="${TRIAL:?}"
WORKLOAD="${WORKLOAD:-mixed}"; DIFF="${DIFF:-all}"
ISL=""
[ "$WORKLOAD" = "speedb" ] && ISL="${ISL:-2k}"
GPU="${GPU:-0}"; PORT="${PORT:-8100}"; DUR="${DUR:-150}"; WARMUP="${WARMUP:-20}"
mkdir -p results/p2

tag="p2_k${K}_c${C}_t${TRIAL}_${WORKLOAD}${ISL:+_${ISL}}"
SPEC_M="none"; [ "$K" != "none" ] && SPEC_M="ngram"
echo "=== $tag (K=$K C=$C workload=$WORKLOAD isl=$ISL diff=$DIFF) ==="
GPU=$GPU PORT=$PORT MODEL=Qwen/Qwen2.5-7B-Instruct SPEC_METHOD="$SPEC_M" K="$K" \
  MAX_MODEL_LEN=16384 MAX_NUM_SEQS=128 CHUNK_TOKENS=2048 ENFORCE_EAGER=1 \
  ./experiments/run_server.sh "$tag"
up=0
for i in $(seq 1 60); do
  if curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" | grep -q 200; then up=1; break; fi
  sleep 5
done
[ "$up" = "1" ] || { echo "SERVER FAILED $tag"; kill $(cat "results/$tag/server.pid") 2>/dev/null; exit 1; }

.venv/bin/python experiments/scrape_metrics.py --port "$PORT" --gpu "$GPU" \
  --out "results/p2/C${C}_k${K}_t${TRIAL}_metrics.jsonl" --interval 0.5 > "results/p2/C${C}_k${K}_t${TRIAL}_scrape.out" 2>&1 &
SPID=$!

if [ "$WORKLOAD" = "speedb" ]; then
  .venv/bin/python experiments/speed_bench_client.py "$PORT" "data/speed_bench/throughput_${ISL}" \
    "$DUR" "$C" "$WARMUP" "/tmp/${tag}.jsonl" 256 | tee "results/p2/${tag}_client.out"
else
  .venv/bin/python experiments/latency_client.py "$PORT" "$DUR" "$C" "$WARMUP" "/tmp/${tag}.jsonl" 256 \
    | tee "results/p2/${tag}_client.out"
fi

kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
sleep 3
kill $(cat "results/$tag/server.pid") 2>/dev/null; wait 2>/dev/null
sleep 5
echo "=== $tag done ==="
