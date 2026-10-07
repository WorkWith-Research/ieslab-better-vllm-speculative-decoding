#!/usr/bin/env bash
# Phase 4.6 prerequisite: fresh AR-only SPS(B) profile (live-load-aware controller input).
# SPS(B) = steps/s of an AR decode batch of size B. Profiled on GPU $GPU, one cell per C.
# Output: results/p4_6/sps_profile.jsonl  lines: {"C":..,"B":..,"SPS":..,"tok_s":..}
set -uo pipefail
cd "$(dirname "$0")/.."
GPU="${GPU:-0}"; PORT="${PORT:-8100}"
OUT="results/p4_6/sps_profile.jsonl"
mkdir -p results/p4_6
: > "$OUT"
for C in 8 16 32 48 64 96 128; do
  tag="p46_prof_c${C}"
  echo "=== $tag (AR profile, C=$C) ==="
  GPU=$GPU PORT=$PORT MODEL=Qwen/Qwen2.5-7B-Instruct SPEC_METHOD=none K="" \
    MAX_MODEL_LEN=16384 MAX_NUM_SEQS=128 CHUNK_TOKENS=2048 ENFORCE_EAGER=1 \
    ./experiments/run_server.sh "$tag"
  up=0
  for i in $(seq 1 60); do
    if curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" | grep -q 200; then up=1; break; fi
    sleep 5
  done
  [ "$up" = "1" ] || { echo "SERVER FAILED $tag"; kill $(cat "results/$tag/server.pid") 2>/dev/null; continue; }
  .venv/bin/python experiments/scrape_metrics.py --port "$PORT" --gpu "$GPU" \
    --out "results/p4_6/${tag}_metrics.jsonl" --interval 0.5 > "results/p4_6/${tag}_scrape.out" 2>&1 &
  SPID=$!
  .venv/bin/python experiments/latency_client.py "$PORT" 75 "$C" 15 "/tmp/${tag}.jsonl" 256 \
    | tee "results/p4_6/${tag}_client.out"
  kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
  sleep 3
  kill $(cat "results/$tag/server.pid") 2>/dev/null; wait 2>/dev/null
  sleep 5
done
# derive SPS(B) from steady throughput: SPS = tok_s / C (AR: exactly 1 token/req/step)
.venv/bin/python - <<'EOF'
import json, re
out = []
for C in [8, 16, 32, 48, 64, 96, 128]:
    p = f"results/p4_6/p46_prof_c{C}_client.out"
    try:
        last = [l for l in open(p) if "steady_throughput_tok_s=" in l][-1]
        tok_s = float(re.search(r"steady_throughput_tok_s=([\d.]+)", last).group(1))
    except Exception as e:
        print(f"WARN c{C}: {e}")
        continue
    out.append({"C": C, "B": float(C), "SPS": round(tok_s / C, 3), "tok_s": tok_s})
with open("results/p4_6/sps_profile.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
print("profile points:", len(out))
for r in out:
    print(f"  B={r['B']:>5.0f}  SPS={r['SPS']:.2f}  tok_s={r['tok_s']:.1f}")
EOF
