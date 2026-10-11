#!/usr/bin/env bash
# P5.2 specfree-off single-cell smoke (C=96, 40s): confirm no crash with OFFSTATE+FORCE0+tracer.
set -uo pipefail
cd "$(dirname "$0")/.."
GPU="${GPU:-0}"; PORT="${PORT:-8413}"
rm -rf results/p52_p52_sf_smoke* logs/p52_p52_sf_smoke* results/p5_2/sf_smoke_obs 2>/dev/null
GPU=$GPU PORT=$PORT MODEL=Qwen/Qwen2.5-7B-Instruct SPEC_METHOD=ngram K=8 \
  MAX_MODEL_LEN=16384 MAX_NUM_SEQS=128 CHUNK_TOKENS=2048 ENFORCE_EAGER=1 \
  CONTROLLER=p52_obs VLLM_P52_OBS="results/p5_2/sf_smoke_obs" VLLM_P52_FORCE0=1 VLLM_P52_OFFSTATE=1 \
  ./experiments/run_server.sh "p52_sf_smoke" >/dev/null 2>&1 &
SPID=$!
up=0; for i in $(seq 1 90); do curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health | grep -q 200 && { up=1; break; }; sleep 5; done
if [ "$up" != "1" ]; then echo "SERVER FAILED"; kill $(cat results/p52_sf_smoke/server.pid) 2>/dev/null; exit 1; fi
.venv/bin/python experiments/latency_client.py $PORT 40 96 15 /tmp/sf_smoke.jsonl 256 >/dev/null 2>&1
sleep 3
ERR=$(grep -icE "traceback|OOM|CUDA error|RuntimeError" logs/p52_p52_sf_smoke.server.log 2>/dev/null)
kill $(cat results/p52_sf_smoke/server.pid) 2>/dev/null; wait $SPID 2>/dev/null; sleep 4
echo "specfree-off smoke: server_errors=$ERR"
grep -iE "p52_obs|deferred install" logs/p52_p52_sf_smoke.server.log | head -3
.venv/bin/python - <<'EOF'
import json, os
f = 'results/p5_2/sf_smoke_obs/steps.jsonl'
if os.path.exists(f):
    rows = [json.loads(l) for l in open(f)]
    wg = [r for r in rows if r.get('gpu_step_ms') is not None]
    if wg:
        print(f'steps: n={len(rows)} with_gpu={len(wg)} gpu_p50={sorted(r["gpu_step_ms"] for r in wg)[len(wg)//2]:.2f}ms')
    else:
        print('no gpu records')
else:
    print('no steps.jsonl')
EOF
