#!/usr/bin/env bash
# Priority B — E1 isolation: does K=8's advantage over K=2 grow with concurrency?
# (batch-level verification coupling hypothesis). C in {1,4,16,32}, 2 trials each.
# All eager, mixed workload, same seed. Output -> results/e1_coupling/trials.jsonl
set -uo pipefail
cd "$(dirname "$0")/.."
DUR=90; WARMUP=15; TRIALS=2
OUTDIR=results/e1_coupling
mkdir -p "$OUTDIR"

run_trial() {
  local C="$1"; local K="$2"; local trial="$3"
  local tag="e1_c${C}_k${K}_t${trial}"
  mkdir -p "results/$tag"
  echo "=== $tag (C=$C K=$K) ==="
  GPU=0 PORT=8100 MODEL=Qwen/Qwen2.5-7B-Instruct SPEC_METHOD=ngram K="$K" \
    MAX_MODEL_LEN=8192 CHUNK_TOKENS=2048 ENFORCE_EAGER=1 \
    ./experiments/run_server.sh "$tag"
  local up=0
  for i in $(seq 1 60); do
    if curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8100/health | grep -q 200; then up=1; break; fi
    sleep 5
  done
  [ "$up" = "1" ] || { echo "SERVER FAILED TO START $tag"; return 1; }
  .venv/bin/python experiments/mixed_workload_client.py 8100 "$DUR" "$C" "$WARMUP" | tee "results/$tag/client.out"
  sleep 3
  kill $(cat "results/$tag/server.pid") 2>/dev/null; wait 2>/dev/null
  sleep 6
}

for C in 1 4 16 32; do
  for K in 2 8; do
    for t in 1 2; do
      run_trial "$C" "$K" "$t"
    done
  done
done

.venv/bin/python - <<'PYEOF'
import json, re, glob, os, statistics as st
rows=[]
for p in sorted(glob.glob("results/e1_*/client.out")):
    tag=os.path.basename(os.path.dirname(p))
    m=re.match(r"e1_c(\d+)_k(\d+)_t(\d+)",tag)
    if not m: continue
    txt=open(p).read()
    mm=re.search(r"steady_throughput_tok_s=([\d.]+)",txt)
    rows.append({"C":int(m.group(1)),"K":int(m.group(2)),"trial":int(m.group(3)),
                 "tok_s":float(mm.group(1)) if mm else None})
with open("results/e1_coupling/trials.jsonl","w") as f:
    for r in rows: f.write(json.dumps(r)+"\n")
# summary: per (C,K) mean and K8-K2 delta
from collections import defaultdict
agg=defaultdict(list)
for r in rows: agg[(r["C"],r["K"])].append(r["tok_s"])
print("C | K=2 mean | K=8 mean | K8-K2")
for C in [1,4,16,32]:
    k2=[x for x in agg.get((C,2),[]) if x]; k8=[x for x in agg.get((C,8),[]) if x]
    if k2 and k8:
        print(f"{C:3d} | {st.mean(k2):8.1f} | {st.mean(k8):8.1f} | {st.mean(k8)-st.mean(k2):+.1f}")
PYEOF
echo "=== E1 coupling sweep done ==="
