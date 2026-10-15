#!/bin/bash
# H-OBS-1 Gate-1 serving-path validation under nsys (bounded diagnostic).
# The standalone torch.profiler check validates TIMER PLACEMENT in-process. This
# adds the directive-required "bounded check of the ACTUAL serving path": an
# async multi-process server run captured by nsys, so we confirm the CUDA-event
# tracer's per-step windows line up with real GPU kernel execution under the
# production (async) configuration — not just the in-process one.
#
# This is a DIAGNOSTIC run: it is NOT one of the 18 main-run cells and its
# throughput numbers are not used for any overhead/verdict claim (nsys perturbs
# timing). Its only job is to produce an independent GPU-execution trace we can
# correlate with the tracer by step count + kernel-burst spans.
set -u
OUT=${1:?usage: hoobs1_profile_check.sh <out_dir>}
ENV=/home/junior1/_dev/dspark-native-validation
REPO=/home/junior1/_dev/better-vllm-speculative-decoding
PORT=8401
GPU=1
DATASET=$REPO/data/ShareGPT_V3_unfiltered_cleaned_split.json
mkdir -p "$OUT"

export CUDA_VISIBLE_DEVICES=$GPU PORT=$PORT
unset VLLM_TB_MODE VLLM_OBS_TRACE VLLM_TB_TRACE 2>/dev/null || true
export VLLM_TB_MODE=native          # dspark arm (the regime of interest)
export VLLM_OBS_TRACE="$OUT/obs"    # tracer on, so we have per-step windows to compare
export VLLM_OBS_NVTX=1              # NVTX step anchors for the trace
rm -f "$OUT"/nsys*.sqlite "$OUT"/nsys*.nsys-rep

cd "$ENV"
# nsys profile the api_server. --cuda-graph-trace=node gives per-kernel timestamps
# even for CUDA-graph replay (DSpark uses FULL graphs). Trace only CUDA + NVTX.
nsys profile -t cuda,nvtx --cuda-graph-trace=node \
  -o "$OUT/nsys" --force-overwrite true \
  .venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B --served-model-name qwen3-4b --port $PORT \
    --tensor-parallel-size 1 --gpu-memory-utilization 0.90 \
    --max-num-seqs 64 --max-num-batched-tokens 4096 --enable-prefix-caching \
    --speculative-config '{"method": "dspark", "model": "deepseek-ai/dspark_qwen3_4b_block7", "num_speculative_tokens": 7}' \
    > "$OUT/server.log" 2>&1 &
SRV_PID=$!

ok=0
for i in $(seq 1 210); do
  if curl -s -o /dev/null -w "%{http_code}" --max-time 2 http://localhost:$PORT/health | grep -q 200 \
     && grep -q "Application startup complete" "$OUT/server.log" 2>/dev/null; then ok=1; break; fi
  sleep 2
done
if [ "$ok" != "1" ]; then echo "nsys check: server not healthy"; tail -20 "$OUT/server.log"; kill $SRV_PID 2>/dev/null; exit 1; fi

# Short steady-state load (30 prompts, C=8) so the trace stays small.
cd "$ENV"
timeout 400 .venv/bin/vllm bench serve --backend openai-chat --base-url http://localhost:$PORT \
  --endpoint /v1/chat/completions --model qwen3-4b --tokenizer Qwen/Qwen3-4B \
  --dataset-name sharegpt --dataset-path "$DATASET" \
  --num-prompts 30 --max-concurrency 8 --ignore-eos --temperature 0 \
  --seed 0 --num-warmups 0 --label nsys_c8 > "$OUT/bench.log" 2>&1

sleep 3   # drain tracer atexit flush
kill $SRV_PID 2>/dev/null
for i in $(seq 1 60); do kill -0 $SRV_PID 2>/dev/null || break; sleep 1; done
kill -9 $SRV_PID 2>/dev/null

# Export to sqlite and run the correlation analysis.
nsys export --type=sqlite -o "$OUT/nsys.sqlite" "$OUT/nsys.nsys-rep" > "$OUT/export.log" 2>&1
.venv/bin/python "$ENV/scripts/hoobs1_nsys_analyze.py" --out "$OUT" || { echo "nsys analyze failed"; }
