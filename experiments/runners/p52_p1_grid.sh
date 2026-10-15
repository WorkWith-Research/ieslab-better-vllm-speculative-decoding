#!/usr/bin/env bash
# P5.2 H-5.2 P1 profiling grid: AR vs force-0 (ngram K=8) at C in {32,96}, each ~90s.
# Tracer overlay (p52_obs) on for all cells. Output: results/p5_2/p52_{arm}_c{C}_obs/steps.jsonl
set -uo pipefail
cd "$(dirname "$0")/.."
GPU="${GPU:-0}"; PORT="${PORT:-8411}"
DUR="${DUR:-90}"; C_LIST="${C_LIST:-32 96}"

cell() { # arm specmethod k c tag force0
  local arm="$1" sm="$2" k="$3" c="$4" tag="$5" f0="${6:-}"
  echo "=== P1 $tag (arm=$arm C=$c) ==="
  GPU=$GPU PORT=$PORT MODEL=Qwen/Qwen2.5-7B-Instruct SPEC_METHOD="$sm" K="${k:-8}" \
    MAX_MODEL_LEN=16384 MAX_NUM_SEQS=128 CHUNK_TOKENS=2048 ENFORCE_EAGER=1 \
    CONTROLLER=p52_obs VLLM_P52_OBS="results/p5_2/${tag}_obs" VLLM_P52_FORCE0="${f0}" \
    ./experiments/run_server.sh "p52_${tag}" >/dev/null 2>&1 &
  local SPID=$!
  local up=0
  for i in $(seq 1 90); do
    if curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" | grep -q 200; then up=1; break; fi
    sleep 5
  done
  if [ "$up" != "1" ]; then echo "SERVER FAILED $tag"; kill $(cat "results/p52_${tag}/server.pid") 2>/dev/null; return 1; fi
  .venv/bin/python experiments/latency_client.py "$PORT" "$DUR" "$c" 15 "/tmp/p52_${tag}.jsonl" 256 >/dev/null 2>&1
  sleep 3
  kill $(cat "results/p52_${tag}/server.pid") 2>/dev/null; wait $SPID 2>/dev/null
  sleep 5
  local f="results/p5_2/${tag}_obs/steps.jsonl"
  if [ -f "$f" ]; then
    .venv/bin/python - "$f" <<'EOF'
import json, sys
rows=[json.loads(l) for l in open(sys.argv[1])]
wg=[r for r in rows if r.get("gpu_step_ms") is not None]
print(f"  {sys.argv[1]}: n={len(rows)} with_gpu={len(wg)} " + (f"gpu_p50={sorted(r['gpu_step_ms'] for r in wg)[len(wg)//2]:.2f}ms" if wg else "NO GPU"))
EOF
  else
    echo "  NO steps.jsonl for $tag"; return 1
  fi
  sleep 4
}

for C in $C_LIST; do
  cell AR none "" "$C" p52_ar_c${C} ""
  cell FORCE0 ngram 8 "$C" p52_f0_c${C} 1
done
echo "P1 GRID DONE"
