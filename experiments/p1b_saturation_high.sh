#!/usr/bin/env bash
# P1b: extend saturation sweep past MAX_NUM_SEQS=64 (raise to 128) to find true saturation.
set -uo pipefail
cd "$(dirname "$0")/.."
OUTDIR=results/p1_sat
mkdir -p "$OUTDIR"

DUR=150; WARMUP=20
CS="96 128"

for C in $CS; do
  tag="p1sat_c${C}"
  echo "=== $tag (C=$C dur=${DUR}s) ==="
  GPU=0 PORT=8100 MODEL=Qwen/Qwen2.5-7B-Instruct SPEC_METHOD=none K=none \
    MAX_MODEL_LEN=8192 MAX_NUM_SEQS=128 CHUNK_TOKENS=2048 ENFORCE_EAGER=1 \
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

# merge into summary (append new rows, keep old)
.venv/bin/python - "$OUTDIR" <<'PYEOF'
import json, sys, os, statistics as st
outdir = sys.argv[1]
def pct(xs, p):
    if not xs: return None
    xs = sorted(xs); k = (len(xs)-1)*p/100
    f = int(k); c = min(f+1, len(xs)-1)
    return xs[f] + (xs[c]-xs[f])*(k-f)
rows = []
for C in [96, 128]:
    p = f"/tmp/p1sat_c{C}.jsonl"
    mp = os.path.join(outdir, f"C{C}_metrics.jsonl")
    if not os.path.exists(p):
        rows.append({"C": C, "error": "missing"}); continue
    recs = [json.loads(l) for l in open(p)]
    ok = [r for r in recs if "ttft_ms" in r]
    st_ = [r for r in ok if 20 <= r["t"] <= 150]
    tok = sum(r["ntok"] for r in st_)
    thr = tok / max(1e-9, 150.0 - 20) if st_ else None
    m = [json.loads(l) for l in open(mp)] if os.path.exists(mp) else []
    def g(k): return [r[k] for r in m if k in r]
    run_g = g("vllm:num_requests_running[gauge]"); wait_g = g("vllm:num_requests_waiting[gauge]")
    kv_g = g("vllm:kv_cache_usage_perc[gauge]"); gpu_g = g("gpu_util")
    rows.append({"C": C, "throughput_tok_s": round(thr,1) if thr else None,
        "completions_steady": len(st_),
        "ttft_ms": {"p50": round(pct([r["ttft_ms"] for r in st_],50),1), "p95": round(pct([r["ttft_ms"] for r in st_],95),1)},
        "tpot_ms": {"p50": round(pct([r["tpot_ms"] for r in st_],50),2), "p95": round(pct([r["tpot_ms"] for r in st_],95),2)},
        "e2e_s_p95": round(pct([r["e2e_s"] for r in st_],95),2),
        "running_p50": st.median(run_g) if run_g else None,
        "waiting_p50": st.median(wait_g) if wait_g else None,
        "waiting_max": max(wait_g) if wait_g else 0,
        "kv_util_max": max(kv_g) if kv_g else None,
        "gpu_util_p50": round(st.median(gpu_g),1) if gpu_g else None})
# merge with existing summary
sp = os.path.join(outdir, "summary.jsonl")
old = {}
if os.path.exists(sp):
    for l in open(sp):
        r = json.loads(l); old[r["C"]] = r
for r in rows: old[r["C"]] = r
with open(sp, "w") as f:
    for C in sorted(old): f.write(json.dumps(old[C]) + "\n")
print(f"{'C':>4} {'tok/s':>8} {'per-req':>8} {'TTFT p50/p95':>14} {'TPOT p50/p95':>15} {'run_p50':>8} {'wait p50/max':>13} {'KVmax':>6}")
prev = None
for C in sorted(old):
    r = old[C]
    if "error" in r: print(f"{C:>4} ERROR"); continue
    perreq = (r["throughput_tok_s"] or 0)/C
    print(f"{C:>4} {r['throughput_tok_s'] or 0:>8.1f} {perreq:>8.2f} "
          f"{r['ttft_ms']['p50'] or 0:>7.0f}/{r['ttft_ms']['p95'] or 0:<6.0f} "
          f"{r['tpot_ms']['p50'] or 0:>7.1f}/{r['tpot_ms']['p95'] or 0:<7.1f} "
          f"{r['running_p50'] or 0:>8.0f} {r['waiting_p50'] or 0:>6.0f}/{r['waiting_max'] or 0:<6.0f} "
          f"{(r['kv_util_max'] or 0):>6.3f}")
PYEOF
echo "=== P1b done ==="
