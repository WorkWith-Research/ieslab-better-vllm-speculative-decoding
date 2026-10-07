#!/usr/bin/env bash
# P1: serving saturation characterization (pre-registered in PROGRESS.md).
# AR baseline (no SD), eager mode (matches P2 execution conditions), mixed workload seed 1234.
# C sweep with scraper (queue/KV/GPU) + streaming client (TTFT/TPOT/E2E/throughput).
set -uo pipefail
cd "$(dirname "$0")/.."
OUTDIR=results/p1_sat
mkdir -p "$OUTDIR"

DUR_SHORT=90; DUR_LONG=120; WARMUP=20
CS="1 2 4 8 16 32 48 64"

for C in $CS; do
  DUR=$DUR_SHORT; [ "$C" -ge 16 ] && DUR=$DUR_LONG
  tag="p1sat_c${C}"
  echo "=== $tag (C=$C dur=${DUR}s) ==="
  GPU=0 PORT=8100 MODEL=Qwen/Qwen2.5-7B-Instruct SPEC_METHOD=none K=none \
    MAX_MODEL_LEN=8192 MAX_NUM_SEQS=64 CHUNK_TOKENS=2048 ENFORCE_EAGER=1 \
    ./experiments/run_server.sh "$tag"
  up=0
  for i in $(seq 1 60); do
    if curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8100/health | grep -q 200; then up=1; break; fi
    sleep 5
  done
  [ "$up" = "1" ] || { echo "SERVER FAILED $tag"; kill $(cat "results/$tag/server.pid") 2>/dev/null; continue; }

  .venv/bin/python experiments/scrape_metrics.py --port 8100 --gpu 0 \
    --out "$OUTDIR/C${C}_metrics.jsonl" --interval 0.5 > "$OUTDIR/C${C}_scrape.out" 2>&1 &
  SPID=$!

  .venv/bin/python experiments/latency_client.py 8100 "$DUR" "$C" "$WARMUP" "/tmp/p1sat_c${C}.jsonl" 256 \
    | tee "$OUTDIR/C${C}_client.out"

  kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
  sleep 3
  kill $(cat "results/$tag/server.pid") 2>/dev/null; wait 2>/dev/null
  sleep 6
done

# aggregate
.venv/bin/python - "$OUTDIR" <<'PYEOF'
import json, sys, os, statistics as st
outdir = sys.argv[1]
def pct(xs, p):
    if not xs: return None
    xs = sorted(xs); k = (len(xs)-1)*p/100
    f = int(k); c = min(f+1, len(xs)-1)
    return xs[f] + (xs[c]-xs[f])*(k-f)
rows = []
for C in [1,2,4,8,16,32,48,64]:
    p = f"/tmp/p1sat_c{C}.jsonl"
    mp = os.path.join(outdir, f"C{C}_metrics.jsonl")
    if not os.path.exists(p):
        rows.append({"C": C, "error": "missing client data"}); continue
    recs = [json.loads(l) for l in open(p)]
    ok = [r for r in recs if "ttft_ms" in r]
    st_ = [r for r in ok if 20 <= r["t"] <= (90 if C < 16 else 120)]
    tok = sum(r["ntok"] for r in st_)
    thr = tok / max(1e-9, min(90.0 if C < 16 else 120.0, 90.0 if C < 16 else 120.0) - 20) if st_ else None
    m = [json.loads(l) for l in open(mp)] if os.path.exists(mp) else []
    def g(k): return [r[k] for r in m if k in r]
    run_g = g("vllm:num_requests_running[gauge]")
    wait_g = g("vllm:num_requests_waiting[gauge]")
    kv_g = g("vllm:kv_cache_usage_perc[gauge]")
    gpu_g = g("gpu_util")
    rows.append({
        "C": C,
        "throughput_tok_s": round(thr,1) if thr else None,
        "completions_steady": len(st_),
        "ttft_ms": {"p50": round(pct([r["ttft_ms"] for r in st_],50),1),
                    "p95": round(pct([r["ttft_ms"] for r in st_],95),1)},
        "tpot_ms": {"p50": round(pct([r["tpot_ms"] for r in st_],50),2),
                    "p95": round(pct([r["tpot_ms"] for r in st_],95),2)},
        "e2e_s_p95": round(pct([r["e2e_s"] for r in st_],95),2),
        "running_p50": st.median(run_g) if run_g else None,
        "waiting_p50": st.median(wait_g) if wait_g else None,
        "waiting_max": max(wait_g) if wait_g else 0,
        "kv_util_max": max(kv_g) if kv_g else None,
        "gpu_util_p50": round(st.median(gpu_g),1) if gpu_g else None,
    })
with open(os.path.join(outdir,"summary.jsonl"),"w") as f:
    for r in rows: f.write(json.dumps(r)+"\n")
print(f"{'C':>3} {'tok/s':>8} {'TTFT p50/p95':>14} {'TPOT p50/p95':>15} {'run_p50':>8} {'wait_p50/max':>13} {'KVmax':>6} {'GPU%':>5}")
for r in rows:
    if "error" in r: print(f"{r['C']:>3} ERROR {r['error']}"); continue
    print(f"{r['C']:>3} {r['throughput_tok_s'] or 0:>8.1f} "
          f"{r['ttft_ms']['p50'] or 0:>7.0f}/{r['ttft_ms']['p95'] or 0:<6.0f} "
          f"{r['tpot_ms']['p50'] or 0:>7.1f}/{r['tpot_ms']['p95'] or 0:<7.1f} "
          f"{r['running_p50'] or 0:>8.0f} {r['waiting_p50'] or 0:>6.0f}/{r['waiting_max'] or 0:<6.0f} "
          f"{(r['kv_util_max'] or 0):>6.3f} {r['gpu_util_p50'] or 0:>5.1f}")
PYEOF
echo "=== P1 saturation sweep done ==="
