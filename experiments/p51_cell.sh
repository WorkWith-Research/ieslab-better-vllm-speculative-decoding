#!/usr/bin/env bash
# Phase 5 single cell: batch-level SD on/off + draft-pass-skip (ldm_batch) arm, mixed workload.
# Pre-registered in PROGRESS.md (H-5.1). Env: C TRIAL GPU PORT DUR WARMUP FORCEOFF(0/1)
set -uo pipefail
cd "$(dirname "$0")/.."
C="${C:?}"; TRIAL="${TRIAL:?}"
GPU="${GPU:-1}"; PORT="${PORT:-8101}"; DUR="${DUR:-150}"; WARMUP="${WARMUP:-20}"
FORCEOFF="${FORCEOFF:-0}"
mkdir -p results/p5_1

tag="p51_ldmbatch_c${C}_t${TRIAL}_mixed"
[ "$FORCEOFF" = "1" ] && tag="p51_ldmbatch_c${C}_tf0_mixed"
OUT="results/p5_1/${tag}_decisions.jsonl"
: > "$OUT"

export VLLM_LDM_BATCH_OUT="$OUT" VLLM_LDM_BATCH_KMAX=8 VLLM_LDM_BATCH_WINDOW=8 \
       VLLM_LDM_BATCH_PROFILE="results/p4_6/sps_profile.jsonl"
[ "$FORCEOFF" = "1" ] && export VLLM_LDM_BATCH_FORCEOFF=1

echo "=== $tag (arm=ldmbatch C=$C forceoff=$FORCEOFF) ==="
CONTROLLER=ldm_batch_controller GPU=$GPU PORT=$PORT MODEL=Qwen/Qwen2.5-7B-Instruct SPEC_METHOD=ngram K=8 \
  MAX_MODEL_LEN=16384 MAX_NUM_SEQS=128 CHUNK_TOKENS=2048 ENFORCE_EAGER=1 \
  ./experiments/run_server.sh "$tag"
up=0
for i in $(seq 1 60); do
  if curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" | grep -q 200; then up=1; break; fi
  sleep 5
done
[ "$up" = "1" ] || { echo "SERVER FAILED $tag"; ./experiments/kill_server.sh "results/$tag/server.pid" "$PORT"; exit 1; }
# HARDENING (post-Phase4.4 port-collision incident): verify OUR pid is alive and OUR log shows startup.
SPID_SERVER=$(cat "results/$tag/server.pid")
sv_ok=0
for i in $(seq 1 24); do
  if kill -0 "$SPID_SERVER" 2>/dev/null && grep -qa "Application startup complete" "logs/$tag.server.log"; then sv_ok=1; break; fi
  sleep 5
done
[ "$sv_ok" = "1" ] || { echo "SERVER NOT OURS/NOT UP $tag (pid $SPID_SERVER) — aborting cell as INVALID"; ./experiments/kill_server.sh "results/$tag/server.pid" "$PORT"; exit 1; }
act=0
for i in $(seq 1 24); do
  if grep -qaE "\[ldm_batch\] .* active" "logs/$tag.server.log"; then act=1; break; fi
  sleep 5
done
[ "$act" = "1" ] || { echo "CONTROLLER NOT ACTIVE $tag — aborting cell"; kill $(cat "results/$tag/server.pid") 2>/dev/null; exit 1; }

.venv/bin/python experiments/scrape_metrics.py --port "$PORT" --gpu "$GPU" \
  --out "results/p5_1/${tag}_metrics.jsonl" --interval 0.5 > "results/p5_1/${tag}_scrape.out" 2>&1 &
SPID=$!

.venv/bin/python experiments/latency_client.py "$PORT" "$DUR" "$C" "$WARMUP" "/tmp/${tag}.jsonl" 256 \
  | tee "results/p5_1/${tag}_client.out"

kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
sleep 3
./experiments/kill_server.sh "results/$tag/server.pid" "$PORT"
echo "=== $tag done ==="
