#!/bin/bash
# H-CLASS-1 harness smoke (prereg §4 validation, pre-confirmatory):
#   S1: AR server + condition A (diverse prompts, cache OFF), C=8, lambda=0.3, 60 s
#       -> verifies: distinct prompts, natural EOS (realized OSL < max in >=2 classes),
#          Poisson arrivals, tracer records, zero draft tokens (AR).
#   S2: AR server + condition B (canonical 4 prompts, cache ON), C=8, 60 s
#       -> verifies: canonical construction loads, high prefix-cache hit rate,
#          realized OSL == max_tokens (reproduces H-OBS-3 degenerate behavior).
# GPU1, port 8402. Manual smoke — not part of the prereg cell grid.
set -u
ENV=/home/junior1/_dev/dspark-native-validation
REPO=/home/junior1/_dev/better-vllm-speculative-decoding
OUT=/tmp/hclass1_smoke
PORT=8402
GPU=0   # DEVIATION from prereg s3 (GPU1): GPU1 occupied by external job (junior2 vLLM serve, pid 2859851) at smoke time; refuse-to-kill rule applies
PY=$ENV/.venv/bin/python
rm -rf "$OUT"; mkdir -p "$OUT"

gpu_free() {
  local m; m=$(nvidia-smi --id=$GPU --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [ "${m:-9999}" -lt 100 ]
}
gpu_free || { echo "FATAL: GPU$GPU not free"; exit 1; }

start_server() { # $1=name $2=cache_arg
  local name="${1:-}" cache_arg="${2:-}"
  local dir="$OUT/$name"
  unset VLLM_TB_MODE VLLM_OBS_TRACE 2>/dev/null || true
  export VLLM_OBS_TRACE="$dir/obs"
  mkdir -p "$dir/obs"; rm -f "$dir/obs/steps.jsonl"
  export CUDA_VISIBLE_DEVICES=$GPU
  cd "$ENV"
  setsid .venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B --served-model-name qwen3-4b --port $PORT \
    --tensor-parallel-size 1 --gpu-memory-utilization 0.90 \
    --max-num-seqs 64 --max-num-batched-tokens 4096 "$cache_arg" \
    > "$dir/server.log" 2>&1 < /dev/null &
  SRV_PID=$!
}

wait_health() { # $1=name
  local i
  for i in $(seq 1 240); do
    if curl -s -o /dev/null -w "%{http_code}" --max-time 2 http://localhost:$PORT/health | grep -q 200; then
      if grep -q "Application startup complete" "$OUT/$1/server.log" 2>/dev/null && kill -0 $SRV_PID 2>/dev/null; then
        return 0
      fi
    fi
    sleep 2
  done
  return 1
}

stop_server() {
  kill -- -"$SRV_PID" 2>/dev/null || kill "$SRV_PID" 2>/dev/null
  for i in $(seq 1 60); do kill -0 "$SRV_PID" 2>/dev/null || break; sleep 1; done
  kill -9 -- -"$SRV_PID" 2>/dev/null; kill -9 "$SRV_PID" 2>/dev/null
  for i in $(seq 1 60); do gpu_free && return 0; sleep 1; done
  echo "WARN: GPU$GPU still busy after stop"
}

# ---- S1: condition A (diverse, cache OFF) ----
echo "=== S1: AR + condA (diverse, cache OFF), C=8, lambda=0.3, 60s ==="
for attempt in 1 2; do
  start_server s1_ar_a --no-enable-prefix-caching
  if wait_health s1_ar_a; then break; fi
  echo "S1 server unhealthy (attempt $attempt)"; tail -5 "$OUT/s1_ar_a/server.log"
  stop_server
  [ $attempt -eq 2 ] && { echo "FAIL: S1 server not healthy after retry"; exit 1; }
  sleep 10   # settle delay: transient CUDA-init race seen after a rapid restart
done
mkdir -p "$OUT/s1_ar_a"
cd "$ENV" && timeout 300 .venv/bin/python "$REPO/experiments/hclass1_client.py" \
    $PORT 60 8 15 "$OUT/s1_ar_a/client.jsonl" 20261013 A 0.3 "$OUT/s1_ar_a/prompt_manifest.jsonl" > "$OUT/s1_ar_a/client.log" 2>&1
echo "S1 client rc=$?"
sleep 3; stop_server; sleep 10   # settle delay before next server start (transient CUDA-init race)

# ---- S2: condition B (canonical, cache ON) ----
echo "=== S2: AR + condB (canonical-4, cache ON), C=8, 60s ==="
for attempt in 1 2; do
  start_server s2_ar_b --enable-prefix-caching
  if wait_health s2_ar_b; then break; fi
  echo "S2 server unhealthy (attempt $attempt)"; tail -5 "$OUT/s2_ar_b/server.log"
  stop_server
  [ $attempt -eq 2 ] && { echo "FAIL: S2 server not healthy after retry"; exit 1; }
  sleep 10   # settle delay: transient CUDA-init race seen after a rapid restart
done
mkdir -p "$OUT/s2_ar_b"
cd "$ENV" && timeout 300 .venv/bin/python "$REPO/experiments/hclass1_client.py" \
    $PORT 60 8 15 "$OUT/s2_ar_b/client.jsonl" 20261013 B 0.3 > "$OUT/s2_ar_b/client.log" 2>&1
echo "S2 client rc=$?"
sleep 3; stop_server

# ---- summary ----
$PY - <<'EOF'
import json, os
for name in ("s1_ar_a", "s2_ar_b"):
    p = f"/tmp/hclass1_smoke/{name}/client.jsonl"
    if not os.path.exists(p):
        print(f"{name}: NO client.jsonl"); continue
    recs = [json.loads(l) for l in open(p)]
    ok = [r for r in recs if "ttft_ms" in r]
    err = len(recs) - len(ok)
    pids = {r.get("pid") for r in ok}
    print(f"\n=== {name}: ok={len(ok)} err={err} distinct_pids={len(pids)} ===")
    from collections import defaultdict
    by = defaultdict(list)
    for r in ok:
        by[r["cls"]].append(r["ntok"])
    for c, mx in (("SS",256),("SL",1024),("LS",256),("LL",1024)):
        v = sorted(by.get(c, []))
        if not v: print(f"  {c}: n=0"); continue
        fam = sum(1 for x in v if x >= mx)/len(v)
        print(f"  {c}: n={len(v)} ntok p10/p50/p90={v[len(v)//10]}/{v[len(v)//2]}/{v[9*len(v)//10]} frac_at_max={fam:.3f}")
    # obs trace
    sp = f"/tmp/hclass1_smoke/{name}/obs/steps.jsonl"
    if os.path.exists(sp):
        st = [json.loads(l) for l in open(sp)]
        pop = sum(1 for r in st if r.get("gpu_step_ms") is not None)
        draft = sum(r.get("draft_tokens",0) for r in st)
        print(f"  obs: steps={len(st)} gpu_populated={100*pop/max(1,len(st)):.1f}% draft_tokens={draft}")
EOF
echo "SMOKE DONE"
