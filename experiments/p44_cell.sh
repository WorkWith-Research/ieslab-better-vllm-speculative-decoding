#!/usr/bin/env bash
# Phase 4.4 single cell: fixed-K x concurrency on the heterogeneous ISLxOSL workload (hetero).
# Pre-registered in PROGRESS.md (H-4.4a/b). Env: K (none|1|2|4|8) C TRIAL GPU PORT DUR WARMUP SEED
set -uo pipefail
cd "$(dirname "$0")/.."
K="${K:?}"; C="${C:?}"; TRIAL="${TRIAL:?}"
GPU="${GPU:-0}"; PORT="${PORT:-8100}"; DUR="${DUR:-150}"; WARMUP="${WARMUP:-20}"; SEED="${SEED:-1234}"
mkdir -p results/p4_4

tag="p44_k${K}_c${C}_t${TRIAL}_hetero"
SPEC_M="none"; [ "$K" != "none" ] && SPEC_M="ngram"
echo "=== $tag (K=$K C=$C hetero) ==="
GPU=$GPU PORT=$PORT MODEL=Qwen/Qwen2.5-7B-Instruct SPEC_METHOD="$SPEC_M" K="$K" \
  MAX_MODEL_LEN=16384 MAX_NUM_SEQS=128 CHUNK_TOKENS=2048 ENFORCE_EAGER=1 \
  ./experiments/run_server.sh "$tag"
up=0
for i in $(seq 1 60); do
  if curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" | grep -q 200; then up=1; break; fi
  sleep 5
done
[ "$up" = "1" ] || { echo "SERVER FAILED $tag"; kill $(cat "results/$tag/server.pid") 2>/dev/null; exit 1; }
# HARDENING (post-Phase4.4 incident): the health endpoint can be answered by an ORPHANED
# server still holding the port while THIS cell's server died at bind time. Verify (a) our
# pid is alive and (b) OUR log shows startup completion — otherwise abort the cell as INVALID.
SPID_SERVER=$(cat "results/$tag/server.pid")
sv_ok=0
for i in $(seq 1 24); do
  if kill -0 "$SPID_SERVER" 2>/dev/null && grep -qa "Application startup complete" "logs/$tag.server.log"; then sv_ok=1; break; fi
  sleep 5
done
[ "$sv_ok" = "1" ] || { echo "SERVER NOT OURS/NOT UP $tag (pid $SPID_SERVER) — aborting cell as INVALID"; kill "$SPID_SERVER" 2>/dev/null; exit 1; }

.venv/bin/python experiments/scrape_metrics.py --port "$PORT" --gpu "$GPU" \
  --out "results/p4_4/${tag}_metrics.jsonl" --interval 0.5 > "results/p4_4/${tag}_scrape.out" 2>&1 &
SPID=$!

.venv/bin/python experiments/hetero_client.py "$PORT" "$DUR" "$C" "$WARMUP" "/tmp/${tag}.jsonl" "$SEED" \
  | tee "results/p4_4/${tag}_client.out"

kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
sleep 3
./experiments/kill_server.sh "results/$tag/server.pid" "$PORT"
echo "=== $tag done ==="
