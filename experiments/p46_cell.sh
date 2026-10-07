#!/usr/bin/env bash
# Phase 4.6 single cell: live-load-aware LDM (ldmload) arm, mixed workload.
# Pre-registered in PROGRESS.md. Env: C TRIAL GPU PORT DUR WARMUP
set -uo pipefail
cd "$(dirname "$0")/.."
C="${C:?}"; TRIAL="${TRIAL:?}"
GPU="${GPU:-1}"; PORT="${PORT:-8101}"; DUR="${DUR:-150}"; WARMUP="${WARMUP:-20}"
mkdir -p results/p4_6

tag="p46_ldmload_c${C}_t${TRIAL}_mixed"
OUT="results/p4_6/${tag}_decisions.jsonl"
: > "$OUT"

export VLLM_LDMLOAD_OUT="$OUT" VLLM_LDMLOAD_KMAX=8 VLLM_LDMLOAD_WINDOW=8 \
       VLLM_LDMLOAD_PROFILE="results/p4_6/sps_profile.jsonl"

echo "=== $tag (arm=ldmload C=$C) ==="
CONTROLLER=ldmload_controller GPU=$GPU PORT=$PORT MODEL=Qwen/Qwen2.5-7B-Instruct SPEC_METHOD=ngram K=8 \
  MAX_MODEL_LEN=16384 MAX_NUM_SEQS=128 CHUNK_TOKENS=2048 ENFORCE_EAGER=1 \
  ./experiments/run_server.sh "$tag"
up=0
for i in $(seq 1 60); do
  if curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" | grep -q 200; then up=1; break; fi
  sleep 5
done
[ "$up" = "1" ] || { echo "SERVER FAILED $tag"; kill $(cat "results/$tag/server.pid") 2>/dev/null; exit 1; }
act=0
for i in $(seq 1 24); do
  if grep -qaE "\[ldmload\] .* active" "logs/$tag.server.log"; then act=1; break; fi
  sleep 5
done
[ "$act" = "1" ] || { echo "CONTROLLER NOT ACTIVE $tag — aborting cell"; kill $(cat "results/$tag/server.pid") 2>/dev/null; exit 1; }

.venv/bin/python experiments/scrape_metrics.py --port "$PORT" --gpu "$GPU" \
  --out "results/p4_6/${tag}_metrics.jsonl" --interval 0.5 > "results/p4_6/${tag}_scrape.out" 2>&1 &
SPID=$!

.venv/bin/python experiments/latency_client.py "$PORT" "$DUR" "$C" "$WARMUP" "/tmp/${tag}.jsonl" 256 \
  | tee "results/p4_6/${tag}_client.out"

kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
sleep 3
kill $(cat "results/$tag/server.pid") 2>/dev/null; wait 2>/dev/null
sleep 5
echo "=== $tag done ==="
