#!/usr/bin/env bash
# Skeptic pass AE2 screen: does the LDM-vs-fixed-K result hold on REAL ShareGPT text
# (vs the templated mixed workload)? 1 trial each, LDM vs fixed K=8 (best ngram config).
set -uo pipefail
cd "$(dirname "$0")/.."
NUMP=600; RATE=6

run_one() {
  local name="$1"; local enforce="$2"; local K="${3:-8}"
  rm -f "/tmp/ae2_${name}.jsonl"
  mkdir -p "results/$name"
  if [ "$enforce" = "1" ]; then
    export PYTHONPATH="$PWD/experiments/ldm_controller" VLLM_LDM_OUT="/tmp/ae2_${name}.jsonl" VLLM_LDM_ENFORCE=1
  else
    unset PYTHONPATH VLLM_LDM_OUT VLLM_LDM_ENFORCE
  fi
  echo "=== $name (enforce=$enforce K=$K, ShareGPT real text) ==="
  GPU=0 PORT=8100 MODEL=Qwen/Qwen2.5-7B-Instruct SPEC_METHOD=ngram K="$K" \
    MAX_MODEL_LEN=8192 CHUNK_TOKENS=2048 ENFORCE_EAGER=1 \
    ./experiments/run_server.sh "$name"
  local up=0
  for i in $(seq 1 60); do
    if curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8100/health | grep -q 200; then up=1; break; fi
    sleep 5
  done
  [ "$up" = "1" ] || { echo "SERVER FAILED TO START $name"; return 1; }
  .venv/bin/python experiments/sharegpt_client.py --port 8100 --num-prompts "$NUMP" \
    --rate "$RATE" --max-tokens 256 --out "results/$name/client.jsonl" | tee "results/$name/client.out"
  # throughput from client log: sum(out_tokens)/elapsed
  .venv/bin/python - <<PYEOF
import json, time
recs=[json.loads(l) for l in open("results/$name/client.jsonl")]
ok=[r for r in recs if r.get("status")==200]
tok=sum(r.get("out_tokens",0) for r in ok)
t0=min(r["t0"] for r in ok); t1=max(r["t1"] for r in ok)
print(f"AE2 $name: ok={len(ok)} out_tokens={tok} elapsed={t1-t0:.0f}s throughput_tok_s={tok/(t1-t0):.1f}")
PYEOF
  sleep 3
  kill $(cat "results/$name/server.pid") 2>/dev/null; wait 2>/dev/null
  sleep 6
}

run_one ae2_ldm     1 8
run_one ae2_fixedk4 0 4
run_one ae2_fixedk8 0 8
echo "=== AE2 ShareGPT screen done ==="
