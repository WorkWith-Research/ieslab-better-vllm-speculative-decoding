#!/usr/bin/env bash
# Priority C: model-based speculator (draft_model) reproduction of the heterogeneity
# phenomenon. Target Qwen2.5-7B-Instruct + draft Qwen2.5-0.5B-Instruct, K in {2,4,8},
# eager mode, mixed workload. Part (a): fixed-K sweep throughput. Part (b): per-request
# acceptance heterogeneity via spec_hook (natural-K span + CV).
set -uo pipefail
cd "$(dirname "$0")/.."
DUR=150; CONCURRENCY=16; WARMUP=20
DRAFT="${DRAFT_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"

run_one() {
  local K="$1"; local hook="$2"
  local tag="pc_dm_k${K}${hook:+_hook}"
  rm -f "/tmp/pc_${tag}.jsonl"
  mkdir -p "results/$tag"
  if [ -n "${hook:-}" ]; then
    export PYTHONPATH="$PWD/experiments/spec_hook" VLLM_SPEC_HOOK_OUT="/tmp/pc_${tag}.jsonl"
  else
    unset PYTHONPATH VLLM_SPEC_HOOK_OUT
  fi
  echo "=== $tag (draft_model K=$K hook=${hook:-off}) ==="
  GPU=0 PORT=8100 MODEL=Qwen/Qwen2.5-7B-Instruct SPEC_METHOD=draft_model DRAFT_MODEL="$DRAFT" \
    K="$K" MAX_MODEL_LEN=8192 CHUNK_TOKENS=2048 ENFORCE_EAGER=1 \
    ./experiments/run_server.sh "$tag"
  local up=0
  for i in $(seq 1 90); do
    if curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8100/health | grep -q 200; then up=1; break; fi
    sleep 5
  done
  [ "$up" = "1" ] || { echo "SERVER FAILED TO START $tag"; tail -20 "logs/$tag.server.log"; return 1; }
  .venv/bin/python experiments/mixed_workload_client.py 8100 "$DUR" "$CONCURRENCY" "$WARMUP" | tee "results/$tag/client.out"
  sleep 3
  kill $(cat "results/$tag/server.pid") 2>/dev/null; wait 2>/dev/null
  sleep 6
}

# (a) fixed-K sweep, no hook
run_one 2 ""
run_one 4 ""
run_one 8 ""
# (b) per-request heterogeneity at K=8 (full profile), with hook
run_one 8 "_hook"
echo "=== Priority C draft-model runs done ==="
