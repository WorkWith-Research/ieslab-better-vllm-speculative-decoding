#!/usr/bin/env bash
# P5.2 H-5.2 P3 adjudication grid: specfree-off vs force-0 vs AR, C in {32,96} x 3 trials, DUR=150s.
#   AR            : SPEC_METHOD=none                                   (upper bound)
#   force-0       : ngram K=8 + FORCE0 truncation                      (the floor; drafter runs, output discarded)
#   specfree-off  : ngram K=8 + FORCE0 truncation + OFFSTATE drafter-skip (maximal runtime off-state = M3 mechanism)
# Server-side gen/s over steady t>=20s, identical harness to P4.6 / Phase 5.1.
set -uo pipefail
cd "$(dirname "$0")/.."
GPU="${GPU:-0}"; PORT="${PORT:-8412}"
DUR="${DUR:-150}"; C_LIST="${C_LIST:-32 96}"; TRIALS="${TRIALS:-3}"

cell() { # arm specmethod k tag force0 offstate
  local arm="$1" sm="$2" k="$3" tag="$4" f0="${5:-}" off="${6:-}"
  echo "=== P3 $tag (arm=$arm) ==="
  GPU=$GPU PORT=$PORT MODEL=Qwen/Qwen2.5-7B-Instruct SPEC_METHOD="$sm" K="${k:-8}" \
    MAX_MODEL_LEN=16384 MAX_NUM_SEQS=128 CHUNK_TOKENS=2048 ENFORCE_EAGER=1 \
    CONTROLLER=p52_obs VLLM_P52_OBS="results/p5_2/${tag}_obs" VLLM_P52_FORCE0="${f0}" VLLM_P52_OFFSTATE="${off}" \
    ./experiments/run_server.sh "p52_${tag}" >/dev/null 2>&1 &
  local SPID=$!
  local up=0
  for i in $(seq 1 90); do
    if curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" | grep -q 200; then up=1; break; fi
    sleep 5
  done
  if [ "$up" != "1" ]; then echo "SERVER FAILED $tag"; kill $(cat "results/p52_${tag}/server.pid") 2>/dev/null; return 1; fi
  .venv/bin/python experiments/scrape_metrics.py --port "$PORT" --gpu "$GPU" \
    --out "results/p5_2/${tag}_metrics.jsonl" --interval 0.5 >/dev/null 2>&1 &
  local MPID=$!
  .venv/bin/python experiments/latency_client.py "$PORT" "$DUR" "${C_TAG}" 15 "/tmp/p52_${tag}.jsonl" 256 >/dev/null 2>&1
  kill $MPID 2>/dev/null; wait $MPID 2>/dev/null
  sleep 3
  kill $(cat "results/p52_${tag}/server.pid") 2>/dev/null; wait $SPID 2>/dev/null
  sleep 5
  # server-side gen/s steady t>=20s
  .venv/bin/python - "results/p5_2/${tag}_metrics.jsonl" <<'EOF'
import json, sys
fn=sys.argv[1]
rows=[json.loads(l) for l in open(fn)]
mr=[r for r in rows if r.get("t",0)>=20]
if len(mr)<10: print(f"  {fn}: SHORT"); sys.exit()
a,b=mr[0],mr[-1]; dt=max(1e-9,b["t"]-a["t"])
c=lambda r,k:(r.get(k,0) or 0)
gen=(c(b,"vllm:generation_tokens[counter]")-c(a,"vllm:generation_tokens[counter]"))/dt
print(f"  {fn}: gen/s={gen:.1f}")
EOF
  sleep 4
}

for C in $C_LIST; do
  for T in $(seq 0 $((TRIALS-1))); do
    C_TAG=$C
    cell AR none ""        p52_ar_c${C}_t${T}     ""   ""
    cell FORCE0 ngram 8    p52_f0_c${C}_t${T}     1    ""
    cell SPECFREE ngram 8  p52_sf_c${C}_t${T}     1    1
  done
done
echo "P3 GRID DONE"
