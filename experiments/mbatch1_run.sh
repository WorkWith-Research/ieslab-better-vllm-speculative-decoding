#!/bin/bash
# M-BATCH-1 — B-class payoff under live MIXED batches (prereg docs/mbatch1-prereg.md).
# Mechanism confirmation for the Phase 7 falsification: does class-B's DSpark/AR TPOT
# payoff, measured on HOMOGENEOUS cells (+0.106 c48 / +0.006 c96 pid-matched), survive
# inside a live MIXED stream? No controller involved — arms are exactly S3's crossover pair:
#   ar     = no speculative config
#   dspark = native DSpark-K7 always-on (VLLM_TB_MODE=native)
# Workload: phase7_mixed_client.py PHASE=MIX (identical to S3), 150 s/cell, seed 20261013.
# Cells: {ar,dspark} x {c48,c96} x t{1,2,3} = 12. Fresh server per cell. GPU0.
set -u
ENV=/home/junior1/_dev/dspark-native-validation
REPO=/home/junior1/_dev/better-vllm-speculative-decoding
OUT=${MB1_OUT:-$REPO/results/mbatch1}
PORT=8402
GPU=${MB1_GPU:-0}
PY=$ENV/.venv/bin/python
DURATION=${MB1_DUR:-150}
BLOCKS=8
SEED=20261013
mkdir -p "$OUT"

# Frozen H-CLASS-1 §5b pressures (same relationship lambda=C/28 as the homogeneous cells):
# c48 -> 1.7143 req/s, c96 -> 3.4286 req/s. Workers = C (H-CLASS-1 cond-A concurrency).
LAMBDA_OF() { case "$1" in 48) echo "1.7143";; 96) echo "3.4286";; *) echo "";; esac; }

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

gpu_free() {
  local m; m=$(nvidia-smi --id=$GPU --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [ "${m:-9999}" -lt 100 ]
}

start_server() { # $1=cell $2=arm (ar|dspark)
  local cell="$1" arm="$2"
  local dir="$OUT/$cell"
  unset VLLM_TB_MODE VLLM_OBS_TRACE VLLM_TB_TRACE 2>/dev/null || true
  export CUDA_VISIBLE_DEVICES=$GPU
  cd "$ENV"
  if [ "$arm" = "ar" ]; then
    setsid .venv/bin/python -m vllm.entrypoints.openai.api_server \
      --model Qwen/Qwen3-4B --served-model-name qwen3-4b --port $PORT \
      --tensor-parallel-size 1 --gpu-memory-utilization 0.90 \
      --max-num-seqs 64 --max-num-batched-tokens 4096 --enable-prefix-caching \
      > "$dir/server.log" 2>&1 < /dev/null &
  else
    export VLLM_TB_MODE=native
    setsid .venv/bin/python -m vllm.entrypoints.openai.api_server \
      --model Qwen/Qwen3-4B --served-model-name qwen3-4b --port $PORT \
      --tensor-parallel-size 1 --gpu-memory-utilization 0.90 \
      --max-num-seqs 64 --max-num-batched-tokens 4096 --enable-prefix-caching \
      --speculative-config '{"method": "dspark", "model": "deepseek-ai/dspark_qwen3_4b_block7", "num_speculative_tokens": 7}' \
      > "$dir/server.log" 2>&1 < /dev/null &
  fi
  SRV_PID=$!
}

wait_health() { # $1=cell
  local i
  for i in $(seq 1 240); do
    if curl -s -o /dev/null -w "%{http_code}" --max-time 2 http://localhost:$PORT/health | grep -q 200; then
      if grep -q "Application startup complete" "$OUT/$1/server.log" 2>/dev/null && kill -0 $SRV_PID 2>/dev/null; then
        return 0
      fi
    fi
    sleep 2
  done
  return 1
}

stop_server() {
  kill -- -"$SRV_PID" 2>/dev/null || kill "$SRV_PID" 2>/dev/null
  for i in $(seq 1 60); do kill -0 "$SRV_PID" 2>/dev/null || break; sleep 1; done
  kill -9 -- -"$SRV_PID" 2>/dev/null; kill -9 "$SRV_PID" 2>/dev/null
  for i in $(seq 1 60); do gpu_free && return 0; sleep 1; done
  log "WARN: GPU$GPU still busy after stop (mem=$(nvidia-smi --id=$GPU --query-gpu=memory.used --format=csv,noheader,nounits))"
}

# Prereg §6 validity gates.
validate_cell() { # $1=cell $2=arm
  local cell="$1" arm="$2" dir="$OUT/$1" reasons=()
  grep -q "Application startup complete" "$dir/server.log" 2>/dev/null || reasons+=("no startup-complete")
  [ -f "$dir/client.jsonl" ] || { log "INVALID $cell: ${reasons[*]}"; return 1; }
  local okc errc bA bB
  read okc errc bA bB < <($PY - "$dir/client.jsonl" <<'EOF'
import json, sys
recs = [json.loads(l) for l in open(sys.argv[1])]
ok = [r for r in recs if "tpot_ms" in r]
err = len(recs) - len(ok)
bA = sum(1 for r in ok if r["cls"].startswith("A"))
bB = sum(1 for r in ok if r["cls"].startswith("B"))
print(len(ok), err, bA, bB)
EOF
)
  [ "${okc:-0}" -ge 30 ] || reasons+=("too few completions ($okc)")
  $PY -c "import sys;sys.exit(0 if int('${errc:-0}')*10 <= max(1,int('${okc:-0}')) else 1)" \
    || reasons+=("error flood ($errc/$okc)")
  # B-completion floor: prereg §6 requires >=20 completed B-class requests per cell (data-quality
  # floor, not power — dspark serves slower under saturation; power is the pooled-pairs requirement).
  [ "${bB:-0}" -ge 20 ] || reasons+=("too few B completions ($bB < 20)")
  if [ "$arm" = "dspark" ]; then
    grep -qiE "speculat|draft" "$dir/server.log" >/dev/null || reasons+=("no spec/draft evidence in server log")
  fi
  if [ ${#reasons[@]} -eq 0 ]; then return 0; else log "INVALID $cell: ${reasons[*]}"; return 1; fi
}

run_cell() { # $1=arm $2=conc $3=trial
  local arm="$1" conc="$2" trial="$3"
  local cell="${arm}_c${conc}_t${trial}"
  local dir="$OUT/$cell"
  local lam; lam=$(LAMBDA_OF "$conc")
  [ -n "$lam" ] || { log "FATAL: no frozen lambda for c$conc"; return 1; }
  if [ -f "$dir/DONE" ] || [ -f "$dir/INVALID" ]; then log "SKIP $cell (already done)"; return 0; fi
  gpu_free || { log "FAIL $cell: GPU$GPU not free at start"; mkdir -p "$dir"; touch "$dir/INVALID"; return 1; }
  log "START $cell ($arm, c$conc, lambda=$lam, ${DURATION}s)"
  mkdir -p "$dir"
  local attempt
  for attempt in 1 2; do
    start_server "$cell" "$arm"
    if wait_health "$cell"; then break; fi
    log "WARN $cell: server unhealthy (attempt $attempt)"; tail -3 "$dir/server.log"
    stop_server
    [ $attempt -eq 2 ] && { log "FAIL $cell: not healthy after retry"; touch "$dir/INVALID"; return 1; }
    sleep 10
  done
  cd "$ENV" && timeout $((DURATION + 180)) .venv/bin/python "$REPO/experiments/mbatch1_client.py" \
    $PORT $DURATION $conc 10 "$dir/client.jsonl" $SEED $lam "$dir/prompt_manifest.jsonl" > "$dir/client.log" 2>&1
  local rc=$?
  log "  $cell client rc=$rc"
  sleep 3; stop_server; sleep 8
  if [ $rc -eq 0 ] && validate_cell "$cell" "$arm"; then
    touch "$dir/DONE"; log "DONE-VALID $cell"
  else
    touch "$dir/INVALID"; log "INVALID $cell (rc=$rc)"
  fi
}

# Counterbalanced: alternate arms within each (load, trial) block. Cell args override:
#   mbatch1_run.sh [arm_cC_tT ...]  e.g. "dspark_c96_t1" (for smoke / resume).
if [ $# -gt 0 ]; then
  for cell in "$@"; do
    if [[ $cell =~ ^(ar|dspark)_c([0-9]+)_t([0-9]+)$ ]]; then
      run_cell "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" "${BASH_REMATCH[3]}"
    else
      log "FATAL: unparseable cell label '$cell' (want arm_cC_tT)"
      exit 1
    fi
  done
else
  for conc in 48 96; do
    for t in 1 2 3; do
      run_cell ar     "$conc" "$t"
      run_cell dspark "$conc" "$t"
    done
  done
fi
log "=== M-BATCH-1 complete: $(ls $OUT/*/DONE 2>/dev/null | wc -l) valid, $(ls $OUT/*/INVALID 2>/dev/null | wc -l) invalid ==="
