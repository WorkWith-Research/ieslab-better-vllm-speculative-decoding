#!/usr/bin/env bash
# End-to-end A/B: LDM (adaptive per-request K) vs fixed-K, same mixed workload,
# all eager mode (variable-K decode is non-uniform -> needs enforce_eager).
# Configs:
#   ldm      : ngram K=8 + LDM controller ENFORCE (truncates each req to its k*)
#   fixed_k4 : ngram K=4, no controller
#   fixed_k8 : ngram K=8, no controller  (upper bound = what LDM starts from)
set -uo pipefail
cd "$(dirname "$0")/.."
DUR=150; RATE=6; WARMUP=20

run_one() {
  local name="$1"; shift
  local enforce="$1"; shift
  local K="$1"; shift
  rm -f "/tmp/ab_${name}.jsonl"
  mkdir -p "results/$name"
  if [ "$enforce" = "1" ]; then
    export PYTHONPATH="$PWD/experiments/ldm_controller" VLLM_LDM_OUT="/tmp/ab_${name}.jsonl" VLLM_LDM_ENFORCE=1
  else
    unset PYTHONPATH VLLM_LDM_OUT VLLM_LDM_ENFORCE
  fi
  echo "=== $name (K=$K enforce=$enforce) ==="
  GPU=0 PORT=8100 MODEL=Qwen/Qwen2.5-7B-Instruct SPEC_METHOD=ngram K="$K" \
    MAX_MODEL_LEN=8192 CHUNK_TOKENS=2048 ENFORCE_EAGER=1 \
    ./experiments/run_server.sh "$name"
  for i in $(seq 1 60); do curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8100/health | grep -q 200 && break; sleep 5; done
  .venv/bin/python experiments/mixed_workload_client.py 8100 "$DUR" "$RATE" "$WARMUP"
  # capture final spec-decode stats from the server log (acceptance length)
  sleep 3
  kill $(cat "results/$name/server.pid") 2>/dev/null; wait 2>/dev/null
  sleep 6
}

run_one ldm      1 8
run_one fixed_k4 0 4
run_one fixed_k8 0 8
echo "=== all configs done ==="
