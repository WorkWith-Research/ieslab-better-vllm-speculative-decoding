#!/usr/bin/env bash
# Apples-to-apples control for the Phase 5.1 force-off anomaly (H-5.1 adjudication).
# Re-runs, TODAY under identical conditions (DUR=150 WARMUP=20, same client/workload), the two
# reference points whose original numbers came from different days:
#   GPU $1 port 8101: Phase 4.6 ldmload FORCE0 at C=96 (spec path ENABLED, k*=0 for all, NO skip)
#                     -> reproduces the "2437.6 SD-off floor"
#   GPU $2 port 8100: plain AR (no spec) at C=96
#                     -> reproduces the "2557.2 AR" reference
set -uo pipefail
cd "$(dirname "$0")/.."

GPU_A="$1"; GPU_B="$2"

(
  # --- P4.6 force-0 floor (C=96) on GPU_A ---
  export VLLM_LDMLOAD_FORCE0=1
  C=96 TRIAL=tf0r GPU=$GPU_A PORT=8101 DUR=150 WARMUP=20 ./experiments/p46_cell.sh
  unset VLLM_LDMLOAD_FORCE0
) > logs/p51ctl_p46force0_c96.out 2>&1 &
PID_A=$!

(
  # --- AR C=96 on GPU_B (plain run_server, no controller, no spec) ---
  cd "$(dirname "$0")/.."
  tag="p51ctl_AR_knone_c96"
  mkdir -p results/p5_1
  export ENFORCE_EAGER=1   # match all other arms (eager mode)
  CONTROLLER="" GPU=$GPU_B PORT=8100 MODEL=Qwen/Qwen2.5-7B-Instruct SPEC_METHOD=none K= \
    MAX_MODEL_LEN=16384 MAX_NUM_SEQS=128 CHUNK_TOKENS=2048 ENFORCE_EAGER=1 \
    ./experiments/run_server.sh "$tag" > "results/p5_1/${tag}_server.out" 2>&1 &
  SPID=$!
  echo $SPID > "results/$tag/server.pid"
  up=0
  for i in $(seq 1 60); do
    curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8100/health | grep -q 200 && { up=1; break; }
    sleep 5
  done
  [ "$up" = "1" ] || { echo "AR SERVER FAILED"; kill $SPID 2>/dev/null; exit 1; }
  .venv/bin/python experiments/scrape_metrics.py --port 8100 --gpu $GPU_B \
    --out "results/p5_1/${tag}_metrics.jsonl" --interval 0.5 > "results/p5_1/${tag}_scrape.out" 2>&1 &
  SPID_SC=$!
  .venv/bin/python experiments/latency_client.py 8100 150 96 20 /tmp/${tag}.jsonl 256 \
    | tee "results/p5_1/${tag}_client.out"
  kill $SPID_SC 2>/dev/null; wait $SPID_SC 2>/dev/null
  sleep 3
  ./experiments/kill_server.sh "results/$tag/server.pid" 8100
  echo "=== AR C96 done ==="
) > logs/p51ctl_ar_c96.out 2>&1 &
PID_B=$!

wait $PID_A; wait $PID_B
echo "=== CONTROLS DONE ==="
