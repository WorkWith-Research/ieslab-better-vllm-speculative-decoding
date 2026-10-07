#!/usr/bin/env bash
# P5 single arm cell: in-loop controller (ldm | dspark) vs fixed baselines, same workload/load as P2/P3.
# Env: ARM=ldm|dspark C WORKLOAD=mixed|speedb ISL=2k TRIAL GPU PORT DUR WARMUP
# Reuses P2's AR/K4/K8 cells (no re-run). Only LDM and DSpark-rule arms are executed here.
set -uo pipefail
cd "$(dirname "$0")/.."
ARM="${ARM:?}"; C="${C:?}"; TRIAL="${TRIAL:?}"
WORKLOAD="${WORKLOAD:-mixed}"; DIFF="${DIFF:-all}"
ISL=""
[ "$WORKLOAD" = "speedb" ] && ISL="${ISL:-2k}"
GPU="${GPU:-0}"; PORT="${PORT:-8100}"; DUR="${DUR:-150}"; WARMUP="${WARMUP:-20}"
mkdir -p results/p5

tag="p5_${ARM}_c${C}_t${TRIAL}_${WORKLOAD}${ISL:+_${ISL}}"
OUT="results/p5/${tag}_decisions.jsonl"
: > "$OUT"

# Controller env (server process inherits via run_server.sh PYTHONPATH injection).
# NOTE: run_server.sh maps CONTROLLER=<name> -> experiments/<name>/, so pass the
# DIRECTORY name (ldm_controller / dspark_controller), not a short alias.
case "$ARM" in
  ldm)
    export VLLM_LDM_OUT="$OUT" VLLM_LDM_ENFORCE=1 VLLM_LDM_KMAX=8 VLLM_LDM_WINDOW=8
    CONTROLLER_NAME=ldm_controller ;;
  dspark)
    export VLLM_DSPARK_OUT="$OUT" VLLM_DSPARK_KMAX=8 VLLM_DSPARK_WINDOW=8 \
           VLLM_SPS_TABLE="results/p2/sps_table.jsonl"
    CONTROLLER_NAME=dspark_controller ;;
  *) echo "unknown ARM $ARM"; exit 1 ;;
esac

echo "=== $tag (arm=$ARM C=$C workload=$WORKLOAD) ==="
CONTROLLER="$CONTROLLER_NAME" GPU=$GPU PORT=$PORT MODEL=Qwen/Qwen2.5-7B-Instruct SPEC_METHOD=ngram K=8 \
  MAX_MODEL_LEN=16384 MAX_NUM_SEQS=128 CHUNK_TOKENS=2048 ENFORCE_EAGER=1 \
  ./experiments/run_server.sh "$tag"
up=0
for i in $(seq 1 60); do
  if curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" | grep -q 200; then up=1; break; fi
  sleep 5
done
[ "$up" = "1" ] || { echo "SERVER FAILED $tag"; kill $(cat "results/$tag/server.pid") 2>/dev/null; exit 1; }
# controller must have installed (banner in server log). EngineCore may start AFTER the
# API server reports healthy, so poll for the banner up to 120s.
act=0
for i in $(seq 1 24); do
  if grep -qaE "\[(ldm|dspark)\] .* active" "logs/$tag.server.log"; then act=1; break; fi
  sleep 5
done
[ "$act" = "1" ] || { echo "CONTROLLER NOT ACTIVE $tag — aborting cell"; kill $(cat "results/$tag/server.pid") 2>/dev/null; exit 1; }

.venv/bin/python experiments/scrape_metrics.py --port "$PORT" --gpu "$GPU" \
  --out "results/p5/${tag}_metrics.jsonl" --interval 0.5 > "results/p5/${tag}_scrape.out" 2>&1 &
SPID=$!

if [ "$WORKLOAD" = "speedb" ]; then
  .venv/bin/python experiments/speed_bench_client.py "$PORT" "data/speed_bench/throughput_${ISL}" \
    "$DUR" "$C" "$WARMUP" "/tmp/${tag}.jsonl" "${MAXTOK:-512}" | tee "results/p5/${tag}_client.out"
else
  .venv/bin/python experiments/latency_client.py "$PORT" "$DUR" "$C" "$WARMUP" "/tmp/${tag}.jsonl" 256 \
    | tee "results/p5/${tag}_client.out"
fi

kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
sleep 3
kill $(cat "results/$tag/server.pid") 2>/dev/null; wait 2>/dev/null
sleep 5
echo "=== $tag done ==="
