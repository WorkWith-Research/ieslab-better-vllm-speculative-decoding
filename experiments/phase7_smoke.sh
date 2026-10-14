#!/bin/bash
# Phase 7 capstone — MIXED-stream smoke (prereg docs/phase7-capstone-prereg.md §5 / [CAL-2]).
#
# The one open [CAL] question that needs the live controller: does the class signal
# (rolling draft-acceptance ratio) stay BIMODAL in a single mixed stream alternating
# diverse + high-reuse blocks, and does the 2-axis controller FLAP across brief OFF
# periods? This smoke answers it with two DSpark `live` runs on the SAME workload family:
#
#   HI-MIX : CONCURRENCY=96 (high load), PHASE=MIX -> kv~0.95; expect class axis to
#            oscillate (diverse blocks low-acc, reuse blocks high-acc) and the controller
#            mostly OFF with periodic re-entry probes.
#   LO-MIX : CONCURRENCY=10 (low load),  PHASE=MIX -> kv~0.30; expect load axis to keep
#            SD ON regardless of class (the low-load rescue).
#
# Server: DSpark block7 K=7, prefix caching ON (required for the high-reuse signal),
#   max-num-seqs 64. VLLM_TB_MODE=live + VLLM_TB_TRACE capture the per-step causal trace.
# GPU0 (free at smoke time). Manual smoke — not part of the confirmatory cell grid.
set -u
ENV=/home/junior1/_dev/dspark-native-validation
REPO=/home/junior1/_dev/better-vllm-speculative-decoding
OUT=/tmp/phase7_smoke
PORT=8402
GPU=${P7_GPU:-0}
PY=$ENV/.venv/bin/python
rm -rf "$OUT"; mkdir -p "$OUT"

gpu_free() {
  local m; m=$(nvidia-smi --id=$GPU --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [ "${m:-9999}" -lt 100 ]
}
gpu_free || { echo "FATAL: GPU$GPU not free"; exit 1; }

start_server() { # $1=name
  local name="$1" dir="$OUT/$1"
  unset VLLM_TB_MODE VLLM_OBS_TRACE 2>/dev/null || true
  export VLLM_TB_MODE=live
  export VLLM_TB_TRACE="$dir/trace.jsonl"; rm -f "$VLLM_TB_TRACE"
  export VLLM_OBS_TRACE="$dir/obs"; mkdir -p "$dir/obs"; rm -f "$dir/obs/steps.jsonl"
  export CUDA_VISIBLE_DEVICES=$GPU
  cd "$ENV"
  setsid .venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B --served-model-name qwen3-4b --port $PORT \
    --tensor-parallel-size 1 --gpu-memory-utilization 0.90 \
    --max-num-seqs 64 --max-num-batched-tokens 4096 --enable-prefix-caching \
    --speculative-config '{"method": "dspark", "model": "deepseek-ai/dspark_qwen3_4b_block7", "num_speculative_tokens": 7}' \
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

run_scenario() { # $1=name $2=concurrency $3=phase $4=duration
  local name="$1" conc="$2" phase="$3" dur="$4"
  local dir="$OUT/$name"
  for attempt in 1 2; do
    start_server "$name"
    if wait_health "$name"; then break; fi
    echo "$name server unhealthy (attempt $attempt)"; tail -5 "$dir/server.log"
    stop_server
    [ $attempt -eq 2 ] && { echo "FAIL: $name server not healthy after retry"; exit 1; }
    sleep 10
  done
  cd "$ENV" && timeout $((dur + 90)) .venv/bin/python "$REPO/experiments/phase7_mixed_client.py" \
    $PORT $dur $conc 8 "$dir/client.jsonl" 20261013 "$phase" > "$dir/client.log" 2>&1
  echo "$name client rc=$?"
  sleep 3; stop_server; sleep 10
}

echo "=== HI-MIX: conc=96 (high load), MIX, 45s ==="
run_scenario hi_mix 96 MIX 45
echo "=== LO-MIX: conc=10 (low load), MIX, 45s ==="
run_scenario lo_mix 10 MIX 45

# ---- analysis: class-signal bimodality + flapping from the causal trace ----
$PY - <<'EOF'
import json, os
def load(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []
for name in ("hi_mix", "lo_mix"):
    tr = load(f"/tmp/phase7_smoke/{name}/trace.jsonl")
    cl = load(f"/tmp/phase7_smoke/{name}/client.jsonl")
    okc = [r for r in cl if "ttft_ms" in r]
    print(f"\n=== {name}: trace_steps={len(tr)} client_ok={len(okc)} err={len(cl)-len(okc)} ===")
    if not tr:
        print("  NO TRACE"); continue
    # kv usage + sd_enabled summary
    kvs = [r["kv_usage"] for r in tr]
    on_steps = sum(1 for r in tr if r["sd_enabled"])
    off_steps = len(tr) - on_steps
    import statistics as st
    print(f"  kv_usage p50={st.median(kvs):.3f} min={min(kvs):.3f} max={max(kvs):.3f}")
    print(f"  sd_enabled: ON={on_steps} OFF={off_steps} ({100*on_steps/len(tr):.0f}% on)")
    # state distribution (on/off/probe)
    from collections import Counter
    sc = Counter(r["state"] for r in tr)
    print(f"  state dist: {dict(sc)}")
    # flapping: count on<->off transitions (ignore probe as a transient on-state)
    trans = 0
    prev = None
    for r in tr:
        cur = "on" if r["sd_enabled"] else "off"
        if prev is not None and cur != prev:
            trans += 1
        prev = cur
    print(f"  on/off transitions (flapping proxy) = {trans} over {len(tr)} steps")
    # class signal bimodality: acc_ratio distribution, split by whether the step's
    # dominant traffic was reuse vs diverse is not directly in trace; instead show the
    # acc_ratio spread and how often it crosses theta_acc=0.45.
    accs = [r["acc_ratio"] for r in tr if r["acc_ratio"] is not None]
    if accs:
        lo = sum(1 for a in accs if a <= 0.45)
        hi = sum(1 for a in accs if a > 0.45)
        print(f"  acc_ratio non-null={len(accs)} p50={st.median(accs):.3f} "
              f"min={min(accs):.3f} max={max(accs):.3f} | <=0.45:{lo} >0.45:{hi}")
    # causal assertion: OFF steps must have 0 drafts scheduled; ON/probe decode steps K drafts
    bad_off = sum(1 for r in tr if not r["sd_enabled"] and r["drafts_scheduled_this_step"] > 0)
    print(f"  CAUSAL check: OFF steps with >0 drafts = {bad_off} (must be 0)")
EOF
echo "SMOKE DONE"
