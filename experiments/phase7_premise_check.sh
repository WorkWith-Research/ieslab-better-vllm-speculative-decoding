#!/bin/bash
# Phase 7 — PREMISE CHECK for the SD-independent class proxy (prefix-cache hit rate).
#
# The reactive acceptance-signal class axis failed at high load (OFF periods can't measure
# it). We're rebuilding it on prefix-cache hit-rate, which is measured from KV blocks at
# admission (independent of speculative decoding -> available with SD off). But there's a
# deeper risk: under c96 mixed traffic, do the reusable B prefixes SURVIVE eviction pressure
# from 96 diverse A requests? If not, the reuse advantage itself vanishes at high load and
# "load-axis-only" is genuinely correct (not just a measurement artifact).
#
# This diagnostic runs ONE scenario: c96 mixed traffic, SD OFF (pure AR -> clean isolation
# of the cache-thrashing question, no draft overhead), prefix caching ON, per-request
# admission hit-rate captured via VLLM_P7_ADMISSION_TRACE. Then it checks whether the
# admission hit-rate distribution is BIMODAL (B requests ~1.0, A requests ~0.0) or collapsed
# to ~0 (cache thrashing kills B too).
#
# GPU0. Manual diagnostic — not part of the confirmatory grid.
set -u
ENV=/home/junior1/_dev/dspark-native-validation
REPO=/home/junior1/_dev/better-vllm-speculative-decoding
OUT=/tmp/phase7_premise
PORT=8402
GPU=${P7_GPU:-0}
PY=$ENV/.venv/bin/python
rm -rf "$OUT"; mkdir -p "$OUT"

gpu_free() {
  local m; m=$(nvidia-smi --id=$GPU --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [ "${m:-9999}" -lt 100 ]
}
gpu_free || { echo "FATAL: GPU$GPU not free"; exit 1; }

export VLLM_TB_MODE=off          # SD off: pure AR, isolates the cache question
export VLLM_P7_ADMISSION_TRACE="$OUT/admission.jsonl"; rm -f "$VLLM_P7_ADMISSION_TRACE"
unset VLLM_OBS_TRACE 2>/dev/null || true
export CUDA_VISIBLE_DEVICES=$GPU
cd "$ENV"
setsid .venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B --served-model-name qwen3-4b --port $PORT \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.90 \
  --max-num-seqs 64 --max-num-batched-tokens 4096 --enable-prefix-caching \
  > "$OUT/server.log" 2>&1 < /dev/null &
SRV_PID=$!

for i in $(seq 1 240); do
  if curl -s -o /dev/null -w "%{http_code}" --max-time 2 http://localhost:$PORT/health | grep -q 200 \
     && grep -q "Application startup complete" "$OUT/server.log" 2>/dev/null && kill -0 $SRV_PID 2>/dev/null; then
    break
  fi
  sleep 2
done

cd "$ENV" && timeout 150 .venv/bin/python "$REPO/experiments/phase7_mixed_client.py" \
  $PORT 60 96 8 "$OUT/client.jsonl" 20261013 MIX > "$OUT/client.log" 2>&1
echo "client rc=$?"

kill -- -"$SRV_PID" 2>/dev/null; kill "$SRV_PID" 2>/dev/null
for i in $(seq 1 40); do kill -0 "$SRV_PID" 2>/dev/null || break; sleep 1; done
kill -9 -- -"$SRV_PID" 2>/dev/null

$PY - <<'EOF'
import json, os, statistics as st
adm=[json.loads(l) for l in open("/tmp/phase7_premise/admission.jsonl")] if os.path.exists("/tmp/phase7_premise/admission.jsonl") else []
print(f"admission records: {len(adm)}")
if not adm:
    print("NO ADMISSION DATA"); raise SystemExit
print("sample req_ids:", [a["req_id"][:24] for a in adm[:3]])
hr=[a["hit_rate"] for a in adm]
print(f"hit_rate p50={st.median(hr):.3f} p90={sorted(hr)[int(0.9*len(hr))]:.3f} max={max(hr):.3f}")
hi=sum(1 for h in hr if h>0.8); lo=sum(1 for h in hr if h<0.2); mid=len(hr)-hi-lo
print(f"  >0.8 (reusable prefix survived): {hi}   <0.2 (diverse/evicted): {lo}   mid: {mid}")
# Timeline of the HIGH-hit fraction per 10s bucket — does reusable-reuse persist or decay?
t0=adm[0]["t"]
buckets={}
for a in adm:
    b=int((a["t"]-t0)//10)
    buckets.setdefault(b,[]).append(a["hit_rate"])
print("  10s-bucket high-hit(>0.8) fraction + p50:")
for b in sorted(buckets):
    v=buckets[b]
    frac=sum(1 for h in v if h>0.8)/len(v)
    print(f"    t{b*10:>3}-{(b+1)*10:>3}s: high-hit={frac:.2f}  p50={st.median(v):.2f}  max={max(v):.2f}  n={len(v)}")
EOF
echo "PREMISE CHECK DONE"
