#!/usr/bin/env bash
# P5.2 P1 smoke: verify the p52_obs overlay works on both arms (AR + force-0 ngram) at C=8.
set -uo pipefail
cd "$(dirname "$0")/.."
GPU="${GPU:-0}"; PORT="${PORT:-8410}"

smoke() { # arm specmethod k tag force0
  local arm="$1" sm="$2" k="$3" tag="$4" f0="${5:-}"
  echo "=== smoke $tag (arm=$arm) ==="
  GPU=$GPU PORT=$PORT MODEL=Qwen/Qwen2.5-7B-Instruct SPEC_METHOD="$sm" K="${k:-8}" \
    MAX_MODEL_LEN=16384 MAX_NUM_SEQS=128 CHUNK_TOKENS=2048 ENFORCE_EAGER=1 \
    CONTROLLER=p52_obs VLLM_P52_OBS="results/p5_2/smoke_${tag}_obs" VLLM_P52_FORCE0="${f0}" \
    ./experiments/run_server.sh "p52_smoke_$tag" >/dev/null 2>&1 &
  local SPID=$!
  # wait for health (also lets model load)
  local up=0
  for i in $(seq 1 90); do
    if curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" | grep -q 200; then up=1; break; fi
    sleep 5
  done
  if [ "$up" != "1" ]; then echo "SERVER FAILED $tag"; kill $(cat "results/p52_smoke_$tag/server.pid") 2>/dev/null; return 1; fi
  .venv/bin/python experiments/latency_client.py "$PORT" 30 8 15 "/tmp/p52_smoke_${tag}.jsonl" 256 >/dev/null 2>&1
  sleep 3
  kill $(cat "results/p52_smoke_$tag/server.pid") 2>/dev/null; wait $SPID 2>/dev/null
  sleep 4
  local f="results/p5_2/smoke_${tag}_obs/steps.jsonl"
  if [ -f "$f" ]; then
    .venv/bin/python - "$f" <<'EOF'
import json, sys
rows=[json.loads(l) for l in open(sys.argv[1])]
wg=[r for r in rows if r.get("gpu_step_ms") is not None]
cad=[r["cadence_ms"] for r in wg if r.get("cadence_ms") is not None]
res=[r["residual_ms"] for r in wg if r.get("residual_ms") is not None]
def p50(x): x=sorted(x); return x[len(x)//2] if x else float('nan')
print(f"  {sys.argv[1]}: n={len(rows)} with_gpu={len(wg)} "
      f"gpu_p50={p50([r['gpu_step_ms'] for r in wg]):.2f}ms "
      f"cad_p50={p50(cad):.2f}ms resid_p50={p50(res):.2f}ms")
EOF
  else
    echo "  NO steps.jsonl for $tag"; return 1
  fi
  sleep 5
}

smoke AR none "" ar_c8 ""
smoke FORCE0 ngram 8 f0_c8 1
echo "SMOKE DONE"
