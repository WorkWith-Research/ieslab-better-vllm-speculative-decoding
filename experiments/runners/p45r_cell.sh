#!/usr/bin/env bash
# P4.5R single cell: CORRECTED DSpark-rule proxy (dsparkr), mixed workload.
# Pre-registered in docs/dspark-proxy-validity.md §6. Env: C TRIAL GPU PORT DUR WARMUP
set -uo pipefail
cd "$(dirname "$0")/.."
C="${C:?}"; TRIAL="${TRIAL:?}"
GPU="${GPU:-1}"; PORT="${PORT:-8101}"; DUR="${DUR:-150}"; WARMUP="${WARMUP:-20}"
mkdir -p results/p4_5r

tag="p45r_dsparkr_c${C}_t${TRIAL}_mixed"
OUT="results/p4_5r/${tag}_decisions.jsonl"
: > "$OUT"

export VLLM_DSPARKR_OUT="$OUT" VLLM_DSPARKR_KMAX=8 VLLM_DSPARKR_WINDOW=8 \
       VLLM_SPS_TABLE="results/p2/sps_table.jsonl"

echo "=== $tag (arm=dsparkr C=$C) ==="
CONTROLLER=dsparkr_controller GPU=$GPU PORT=$PORT MODEL=Qwen/Qwen2.5-7B-Instruct SPEC_METHOD=ngram K=8 \
  MAX_MODEL_LEN=16384 MAX_NUM_SEQS=128 CHUNK_TOKENS=2048 ENFORCE_EAGER=1 \
  ./experiments/run_server.sh "$tag"
up=0
for i in $(seq 1 60); do
  if curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" | grep -q 200; then up=1; break; fi
  sleep 5
done
[ "$up" = "1" ] || { echo "SERVER FAILED $tag"; kill $(cat "results/$tag/server.pid") 2>/dev/null; exit 1; }
# HARDENING (p46_cell.sh pattern): verify OUR pid is alive and OUR log shows startup.
SPID_SERVER=$(cat "results/$tag/server.pid")
sv_ok=0
for i in $(seq 1 24); do
  if kill -0 "$SPID_SERVER" 2>/dev/null && grep -qa "Application startup complete" "logs/$tag.server.log"; then sv_ok=1; break; fi
  sleep 5
done
[ "$sv_ok" = "1" ] || { echo "SERVER NOT OURS/NOT UP $tag (pid $SPID_SERVER) — aborting cell as INVALID"; kill $(cat "results/$tag/server.pid") 2>/dev/null; exit 1; }
act=0
for i in $(seq 1 24); do
  if grep -qaE "\[dsparkR\] .* active" "logs/$tag.server.log"; then act=1; break; fi
  sleep 5
done
[ "$act" = "1" ] || { echo "CONTROLLER NOT ACTIVE $tag — aborting cell"; kill $(cat "results/$tag/server.pid") 2>/dev/null; exit 1; }

.venv/bin/python experiments/scrape_metrics.py --port "$PORT" --gpu "$GPU" \
  --out "results/p4_5r/${tag}_metrics.jsonl" --interval 0.5 > "results/p4_5r/${tag}_scrape.out" 2>&1 &
SPID=$!

.venv/bin/python experiments/latency_client.py "$PORT" "$DUR" "$C" "$WARMUP" "/tmp/${tag}.jsonl" 256 \
  | tee "results/p4_5r/${tag}_client.out"

kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
sleep 3
kill $(cat "results/$tag/server.pid") 2>/dev/null; wait 2>/dev/null
sleep 5
echo "=== $tag done ==="
