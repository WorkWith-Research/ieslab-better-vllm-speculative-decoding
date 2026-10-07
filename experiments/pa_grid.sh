#!/usr/bin/env bash
# Priority A: rigorous live adaptive-K vs fixed-K grid (pre-registered in PROGRESS.md).
# Configs: ar, k1, k2, k4, k8, ldm  — all eager, concurrency=16, mixed workload, seed 1234.
# 3 trials per config, fresh server each trial. Per-trial JSON -> results/pa_grid/trials.jsonl
set -uo pipefail
cd "$(dirname "$0")/.."
DUR=150; CONCURRENCY=16; WARMUP=20; TRIALS=3
OUTDIR=results/pa_grid
mkdir -p "$OUTDIR"
: > "$OUTDIR/trials.jsonl"

run_trial() {
  local name="$1"; local trial="$2"; shift 2
  local enforce="${1:-0}"; local K="${2:-none}"
  local tag="${name}_t${trial}"
  rm -f "/tmp/pa_${tag}.jsonl" "/tmp/pa_${tag}_gpu.log"
  mkdir -p "results/$tag"

  if [ "$enforce" = "1" ]; then
    export PYTHONPATH="$PWD/experiments/ldm_controller" VLLM_LDM_OUT="/tmp/pa_${tag}_dec.jsonl" VLLM_LDM_ENFORCE=1
  else
    unset PYTHONPATH VLLM_LDM_OUT VLLM_LDM_ENFORCE
  fi

  local spec_method="none"; [ "$K" != "none" ] && spec_method="ngram"
  echo "=== $tag (spec=$spec_method K=$K enforce=$enforce) ==="
  GPU=0 PORT=8100 MODEL=Qwen/Qwen2.5-7B-Instruct SPEC_METHOD="$spec_method" K="$K" \
    MAX_MODEL_LEN=8192 CHUNK_TOKENS=2048 ENFORCE_EAGER=1 \
    ./experiments/run_server.sh "$tag"
  local up=0
  for i in $(seq 1 60); do
    if curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8100/health | grep -q 200; then up=1; break; fi
    sleep 5
  done
  [ "$up" = "1" ] || { echo "SERVER FAILED TO START $tag"; return 1; }

  # GPU utilization sampler (5s) during the trial
  ( for i in $(seq 1 $((DUR/5 + 2))); do nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader >> "/tmp/pa_${tag}_gpu.log"; sleep 5; done ) &
  local gpid=$!

  .venv/bin/python experiments/latency_client.py 8100 "$DUR" "$CONCURRENCY" "$WARMUP" "/tmp/pa_${tag}.jsonl" 2>&1 | tee "results/$tag/client.out"
  kill $gpid 2>/dev/null

  sleep 3
  kill $(cat "results/$tag/server.pid") 2>/dev/null; wait 2>/dev/null
  sleep 6
}

run_trial ar   1 0 none
run_trial ar   2 0 none
run_trial ar   3 0 none
run_trial k1   1 0 1
run_trial k1   2 0 1
run_trial k1   3 0 1
run_trial k2   1 0 2
run_trial k2   2 0 2
run_trial k2   3 0 2
run_trial k4   1 0 4
run_trial k4   2 0 4
run_trial k4   3 0 4
run_trial k8   1 0 8
run_trial k8   2 0 8
run_trial k8   3 0 8
run_trial ldm  1 1 8
run_trial ldm  2 1 8
run_trial ldm  3 1 8

# aggregate per-trial summaries
.venv/bin/python - <<'PYEOF'
import json, glob, os, statistics as st
def pct(xs, p):
    if not xs: return None
    xs = sorted(xs); k = (len(xs)-1)*p/100
    f = int(k); c = min(f+1, len(xs)-1)
    return xs[f] + (xs[c]-xs[f])*(k-f)
rows = []
for cfg in ["ar","k1","k2","k4","k8","ldm"]:
    for t in [1,2,3]:
        tag = f"{cfg}_t{t}"
        p = f"/tmp/pa_{tag}.jsonl"
        if not os.path.exists(p):
            rows.append({"config":cfg,"trial":t,"error":"missing"}); continue
        recs = [json.loads(l) for l in open(p)]
        ok = [r for r in recs if "ttft_ms" in r]
        st_ = [r for r in ok if 20 <= r["t"] <= 150]
        tok = sum(r["ntok"] for r in st_)
        window = max(1e-9, min(150.0, 150.0) - 20)
        thr = tok/window if st_ else None
        gpus = []
        gp = f"/tmp/pa_{tag}_gpu.log"
        if os.path.exists(gp):
            for l in open(gp):
                try: gpus.append(int(l.split(",")[0].strip()))
                except Exception: pass
        dec = f"/tmp/pa_{tag}_dec.jsonl"
        kdist = None
        if cfg=="ldm" and os.path.exists(dec):
            from collections import Counter
            c = Counter()
            for l in open(dec):
                for d in json.loads(l)["decisions"]: c[d["kstar"]] += 1
            kdist = dict(sorted(c.items()))
        rows.append({
            "config":cfg,"trial":t,
            "throughput_tok_s":round(thr,1) if thr else None,
            "completions_steady":len(st_),"completions_total":len(ok),
            "err_count":len(recs)-len(ok),
            "ttft_ms":{"mean":round(st.mean([r["ttft_ms"] for r in st_]),1) if st_ else None,
                       "p50":round(pct([r["ttft_ms"] for r in st_],50),1) if st_ else None,
                       "p95":round(pct([r["ttft_ms"] for r in st_],95),1) if st_ else None,
                       "p99":round(pct([r["ttft_ms"] for r in st_],99),1) if st_ else None},
            "tpot_ms":{"mean":round(st.mean([r["tpot_ms"] for r in st_]),2) if st_ else None,
                       "p50":round(pct([r["tpot_ms"] for r in st_],50),2) if st_ else None,
                       "p95":round(pct([r["tpot_ms"] for r in st_],95),2) if st_ else None,
                       "p99":round(pct([r["tpot_ms"] for r in st_],99),2) if st_ else None},
            "e2e_s":{"mean":round(st.mean([r["e2e_s"] for r in st_]),2) if st_ else None,
                     "p95":round(pct([r["e2e_s"] for r in st_],95),2) if st_ else None},
            "gpu_util_mean":round(st.mean(gpus),1) if gpus else None,
            "kstar_dist":kdist,
        })
with open("results/pa_grid/trials.jsonl","w") as f:
    for r in rows: f.write(json.dumps(r)+"\n")
print(json.dumps(rows, indent=1))
PYEOF
echo "=== Priority-A grid done ==="
