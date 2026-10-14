#!/bin/bash
# Phase 7 capstone — S3 comparison run set (prereg docs/phase7-capstone-prereg.md §2/§4).
#
# Tests "both axes necessary" in the LIVE system: does CTRL-LIVE (2-axis) beat both
# single-axis ablations (CTRL-LOADONLY / CTRL-CLASSONLY) in aggregate on a mixed
# diverse+high-reuse stream at high load, where the class axis rescues SD for reusable
# traffic that the load-only arm turns off?
#
#   Arms (6): DSpark-K7 (native fixed K=7), CTRL-ON (constant ON = overhead control),
#             CTRL-LOADONLY (ON iff load_LOW), CTRL-CLASSONLY (ON iff class_HIGHREUSE),
#             CTRL-LIVE (full 2-axis rule). [CTRL-OFF omitted: zero drafts, no info.]
#   Loads (2): c48 (mid), c96 (high) — the cells where axes differentiate. c16 (low)
#              is covered by the confirmation smoke (all arms ON there by design).
#   Reps (2), counterbalanced arm order per load, shared workload seed 20261013.
#
# Server: DSpark block7 K=7, prefix caching ON, max-num-seqs 64, GPU0. Per-cell causal
# trace via VLLM_TB_TRACE. Incremental per-cell save -> partial data survives a crash.
set -u
ENV=/home/junior1/_dev/dspark-native-validation
REPO=/home/junior1/_dev/better-vllm-speculative-decoding
OUT=${P7_OUT:-/tmp/phase7_s3}
PORT=8402
GPU=${P7_GPU:-0}
PY=$ENV/.venv/bin/python
DURATION=${P7_DUR:-45}
WARMUP=10
BLOCKS=8
SEED=20261013
mkdir -p "$OUT"

gpu_free() {
  local m; m=$(nvidia-smi --id=$GPU --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [ "${m:-9999}" -lt 100 ]
}
gpu_free || { echo "FATAL: GPU$GPU not free"; exit 1; }

start_server() { # $1=name $2=mode   (mode in dspark|on|loadonly|classonly|live)
  local name="$1" mode="$2" dir="$OUT/$name"
  unset VLLM_TB_MODE VLLM_OBS_TRACE 2>/dev/null || true
  export VLLM_TB_MODE=$mode
  export VLLM_TB_TRACE="$dir/trace.jsonl"; rm -f "$VLLM_TB_TRACE"
  export CUDA_VISIBLE_DEVICES=$GPU
  cd "$ENV"
  setsid .venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B --served-model-name qwen3-4b --port $PORT \
    --tensor-parallel-size 1 --gpu-memory-utilization 0.90 \
    --max-num-seqs 64 --max-num-batched-tokens 4096 --enable-prefix-caching \
    --speculative-config '{"method": "dspark", "model": "deepseek-ai/dspark_qwen3_4b_block7", "num_speculative_tokens": 7}' \
    > "$dir/server.log" 2>&1 < /dev/null &
  SRV_PID=$!
}

wait_health() { # $1=name
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
  echo "WARN: GPU$GPU still busy after stop"
}

run_cell() { # $1=arm_name $2=mode $3=conc $4=rep
  local cell="$1" mode="$2" conc="$3" rep="$4" dir="$OUT/$cell"
  mkdir -p "$dir"
  for attempt in 1 2; do
    start_server "$cell" "$mode"
    if wait_health "$cell"; then break; fi
    echo "  $cell server unhealthy (attempt $attempt)"; tail -5 "$dir/server.log"
    stop_server
    [ $attempt -eq 2 ] && { echo "SKIP: $cell not healthy after retry"; return 1; }
    sleep 10
  done
  cd "$ENV" && timeout $((DURATION + 90)) .venv/bin/python "$REPO/experiments/phase7_mixed_client.py" \
    $PORT $DURATION $conc $BLOCKS "$dir/client.jsonl" $SEED MIX > "$dir/client.log" 2>&1
  local rc=$?
  echo "  $cell client rc=$rc"
  sleep 3; stop_server; sleep 8
  # incremental per-cell save
  echo "{\"arm\":\"$cell\",\"conc\":$conc,\"rep\":$rep,\"client_rc\":$rc,\"ts\":\"$(date -Is)\"}" >> "$OUT/cells.jsonl"
}

# Counterbalanced arm order per load (same workload seed -> identical block sequence).
# Arm -> VLLM_TB_MODE: dspark->native (zero-overhead always-ON = true DSpark-K7 baseline,
# no trace); on->on (constant ON = patched overhead control); loadonly/classonly/live ->
# Phase7Controller ablations + full 2-axis rule.
ARM_TO_MODE=(live dspark on classonly loadonly)
MODE_OF() { case "$1" in live) echo live;; dspark) echo native;; on) echo on;; classonly) echo classonly;; loadonly) echo loadonly;; esac; }
run_load() { # $1=conc $2=label
  local conc="$1" label="$2" arm mode
  echo "=== LOAD c$conc ($label): $(date -Is) ==="
  for rep in 0 1; do
    for arm in "${ARM_TO_MODE[@]}"; do
      mode=$(MODE_OF "$arm")
      run_cell "${label}_${arm}_r${rep}" "$mode" "$conc" "$rep"
    done
  done
}

echo "=== S3 RUN START $(date -Is) GPU$GPU dur=${DURATION}s seed=$SEED ==="
run_load 48 mid
run_load 96 high
echo "S3 DONE $(date -Is)"
