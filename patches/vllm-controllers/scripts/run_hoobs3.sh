#!/bin/bash
# H-OBS-3 run set (prereg docs/hoobs3-prereg.md §3): heterogeneous prefill/decode
# characterization of native fixed-K DSpark. 6 cells = arm{ar,dspark} x C{32,96},
# tracer ON in all cells, fresh server each. Heterogeneous client (4 ISL/OSL
# classes, natural EOS), 180 s per cell, port 8401, GPU1.
#
# Usage: run_hoobs3.sh [--dry-run] [cell ...]     # default: all 6 cells in frozen order
set -u
ENV=/home/junior1/_dev/dspark-native-validation
REPO=/home/junior1/_dev/better-vllm-speculative-decoding
OUT=$REPO/results/hoobs3/hetero
PORT=8401
GPU=1
PY=$ENV/.venv/bin/python
MANIFEST=$OUT/manifest.jsonl
LOCK=/tmp/hoobs3_gpu1.lock
DURATION=180
WARMUP=20
SEED=20261011

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$OUT/status.log"; }

# ---- resource lock (same sentinel pattern as run_hoobs1.sh) ----
acquire_lock() {
  if [ -e "$LOCK" ]; then
    local old; old=$(cat "$LOCK" 2>/dev/null)
    if [ "${old%% *}" != "0" ] && kill -0 "${old%% *}" 2>/dev/null; then
      log "FATAL: GPU lock held by live pid ${old%% *} (another run) — refusing to start"
      exit 1
    fi
    rm -f "$LOCK"
  fi
  echo "0 $(date '+%s')" > "$LOCK"
}

gpu_free() {
  local m; m=$(nvidia-smi --id=$GPU --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [ "${m:-9999}" -lt 100 ]
}

write_manifest() { # $1=cell $2=arm $3=C
  local cell=$1 arm=$2 C=$3
  $PY - "$OUT" "$cell" "$arm" "$C" <<'EOF'
import json, os, sys, subprocess
out, cell, arm, C = sys.argv[1:5]
d = os.path.join(out, cell); os.makedirs(d, exist_ok=True)
def rev(path):
    try:
        return subprocess.run(["git","-C",path,"rev-parse","HEAD"],capture_output=True,text=True).stdout.strip()
    except Exception:
        return None
m = {
  "cell": cell, "arm": arm, "C": int(C), "tracer": "on",
  "created_utc": __import__("datetime").datetime.utcnow().isoformat()+"Z",
  "worktree_rev": rev("/home/junior1/_dev/dspark-native-validation"),
  "repo_rev": rev("/home/junior1/_dev/better-vllm-speculative-decoding"),
  "server_cmd": ["vllm.entrypoints.openai.api_server","--model","Qwen/Qwen3-4B",
                  "--served-model-name","qwen3-4b","--port","8401","--tensor-parallel-size","1",
                  "--gpu-memory-utilization","0.90","--max-num-seqs","64",
                  "--max-num-batched-tokens","4096","--enable-prefix-caching"] + (
                    ["--speculative-config",'{"method": "dspark", "model": "deepseek-ai/dspark_qwen3_4b_block7", "num_speculative_tokens": 7}'] if arm != "ar" else []),
  "env": {"VLLM_TB_MODE": ({"ar": None, "dspark": "native"}[arm]),
           "VLLM_OBS_TRACE": f"{out}/{cell}/obs"},
  "workload": {"client":"hetero_client_qwen3.py","classes":{"SS":[50,256],"SL":[50,1024],
               "LS":[2048,256],"LL":[2048,1024]},"duration_s":180,"warmup_s":20,
               "concurrency":int(C),"seed":20261011,"greedy":True,"ignore_eos":False},
  "sd_expected": ({"ar":"off","dspark":"on"}[arm]),
}
p = os.path.join(d, "config_manifest.json")
json.dump(m, open(p,"w"), indent=2)
print("manifest:", p)
EOF
}

start_server() { # $1=cell $2=arm -> sets SRV_PID
  local cell=$1 arm=$2 dir="$OUT/$cell"
  unset VLLM_TB_MODE VLLM_OBS_TRACE VLLM_TB_TRACE VLLM_LDM_BATCH_FORCEOFF VLLM_LDMLOAD_FORCE0 2>/dev/null || true
  export VLLM_OBS_TRACE="$dir/obs"
  mkdir -p "$dir/obs"; rm -f "$dir/obs/steps.jsonl" "$dir/obs/collector_stats.json"
  case $arm in
    dspark) export VLLM_TB_MODE=native ;;
    ar)     : ;;
  esac
  local spec_args=()
  [ "$arm" != "ar" ] && spec_args=(--speculative-config '{"method": "dspark", "model": "deepseek-ai/dspark_qwen3_4b_block7", "num_speculative_tokens": 7}')
  export CUDA_VISIBLE_DEVICES=$GPU PORT=$PORT
  cd "$ENV"
  setsid .venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B --served-model-name qwen3-4b --port $PORT \
    --tensor-parallel-size 1 --gpu-memory-utilization 0.90 \
    --max-num-seqs 64 --max-num-batched-tokens 4096 --enable-prefix-caching \
    "${spec_args[@]}" > "$dir/server.log" 2>&1 < /dev/null &
  SRV_PID=$!
  echo "$SRV_PID $(date '+%s')" > "$dir/srv_pid"
}

wait_health() { # $1=cell; verifies OUR server (startup marker in own log + pid alive)
  local cell=$1 i
  for i in $(seq 1 210); do
    if curl -s -o /dev/null -w "%{http_code}" --max-time 2 http://localhost:$PORT/health | grep -q 200; then
      if grep -q "Application startup complete" "$OUT/$cell/server.log" 2>/dev/null && kill -0 $SRV_PID 2>/dev/null; then
        return 0
      fi
    fi
    sleep 2
  done
  return 1
}

stop_server() {
  local pid=$SRV_PID
  kill -- -"$pid" 2>/dev/null || kill "$pid" 2>/dev/null
  for i in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -9 -- -"$pid" 2>/dev/null; kill -9 "$pid" 2>/dev/null
  for i in $(seq 1 60); do gpu_free && return 0; sleep 1; done
  log "WARN: GPU$GPU still busy after stop (mem=$(nvidia-smi --id=$GPU --query-gpu=memory.used --format=csv,noheader,nounits))"
}

validate_cell() { # $1=cell $2=arm -> 0 valid, 1 invalid
  local cell=$1 arm=$2 dir="$OUT/$cell"
  local reasons=()
  grep -q "Application startup complete" "$dir/server.log" 2>/dev/null || reasons+=("no startup-complete in own log")
  [ -f "$dir/client.jsonl" ] || { reasons+=("missing client records"); return 1; }
  # client: >=200 completed requests over the 180 s window, no error flood
  local okc errc
  okc=$($PY -c "import json;r=[json.loads(l) for l in open('$dir/client.jsonl')];print(sum(1 for x in r if 'ttft_ms' in x))")
  errc=$($PY -c "import json;r=[json.loads(l) for l in open('$dir/client.jsonl')];print(sum(1 for x in r if 'err' in x))")
  [ "${okc:-0}" -ge 200 ] || reasons+=("client ok-requests too low ($okc < 200)")
  $PY -c "import sys;sys.exit(0 if int('$errc')*10 <= max(1,int('${okc:-0}')+int('$errc')) else 1)" \
    || reasons+=("client error flood ($errc errors vs $okc ok)")
  # obs trace: non-empty, >=95% gpu-populated
  local nsteps=0 gpufrac
  [ -f "$dir/obs/steps.jsonl" ] && nsteps=$(wc -l < "$dir/obs/steps.jsonl")
  [ "${nsteps:-0}" -gt 100 ] || reasons+=("obs trace missing/too short (${nsteps:-0} steps)")
  if [ -f "$dir/obs/steps.jsonl" ]; then
    gpufrac=$($PY -c "
import json
recs=[json.loads(l) for l in open('$dir/obs/steps.jsonl')]
pop=sum(1 for r in recs if r.get('gpu_step_ms') is not None)
print(round(100.0*pop/max(1,len(recs)),1))")
    $PY -c "import sys;sys.exit(0 if float('$gpufrac')>=95.0 else 1)" \
      || reasons+=("only ${gpufrac}% of steps have gpu_step_ms populated")
  fi
  # collector dropped == 0 when stats file present
  if [ -f "$dir/obs/collector_stats.json" ]; then
    local dropped; dropped=$($PY -c "import json;print(json.load(open('$dir/obs/collector_stats.json'))['records_dropped'])")
    [ "$dropped" = "0" ] || reasons+=("collector dropped $dropped records")
  fi
  # arm-specific SD behavior from the trace
  if [ -f "$dir/obs/steps.jsonl" ]; then
    local sdsum; sdsum=$($PY -c "
import json
recs=[json.loads(l) for l in open('$dir/obs/steps.jsonl')]
print(sum(r['draft_tokens'] for r in recs))")
    case $arm in
      ar) [ "$sdsum" = "0" ] || reasons+=("AR arm scheduled $sdsum draft tokens (expected 0)") ;;
      dspark) [ "$sdsum" -gt 1000 ] || reasons+=("dspark arm scheduled only $sdsum draft tokens (expected SD active)") ;;
    esac
  fi
  if [ ${#reasons[@]} -eq 0 ]; then return 0; else log "INVALID $cell: ${reasons[*]}"; return 1; fi
}

trial() { # $1=arm $2=C
  local arm=$1 C=$2 cell="${arm}_c${C}" dir="$OUT/$cell"
  if [ -f "$dir/DONE" ] || [ -f "$dir/INVALID" ]; then log "SKIP $cell (already done)"; return 0; fi
  gpu_free || { log "FAIL $cell: GPU$GPU not free at start"; mkdir -p "$dir"; touch "$dir/INVALID"; return 1; }
  log "START $cell"
  mkdir -p "$dir"
  write_manifest "$cell" "$arm" "$C"
  start_server "$cell" "$arm"
  if ! wait_health "$cell"; then
    log "FAIL $cell: server not healthy/verified in 420s"; tail -15 "$dir/server.log" | tee -a "$OUT/status.log"
    stop_server; touch "$dir/INVALID"; return 1
  fi
  ( cd "$ENV" && .venv/bin/python "$REPO/experiments/scrape_metrics.py" \
      --port $PORT --gpu $GPU --interval 0.5 --out "$dir/metrics.jsonl" ) \
      > "$dir/scraper.log" 2>&1 &
  local SC_PID=$!
  sleep 3
  cd "$ENV"
  timeout $((DURATION + 300)) .venv/bin/python "$REPO/experiments/hetero_client_qwen3.py" \
      $PORT $DURATION $C $WARMUP "$dir/client.jsonl" $SEED > "$dir/client.log" 2>&1
  local CRC=$?
  kill $SC_PID 2>/dev/null; wait $SC_PID 2>/dev/null
  sleep 3   # let atexit final_flush drain the last steps' events (documented in prereg)
  stop_server
  if [ $CRC -ne 0 ]; then log "FAIL $cell: client rc=$CRC"; touch "$dir/INVALID"; return 1; fi
  if validate_cell "$cell" "$arm"; then
    touch "$dir/DONE"; log "DONE-VALID $cell"
  else
    touch "$dir/INVALID"
  fi
}

# ---------------- main ----------------
mkdir -p "$OUT"
touch "$MANIFEST"
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
if [ $DRY -eq 1 ]; then shift; fi

# frozen order (prereg §3): C-major, AR before DSpark within each C — 4 cells
CELLS="ar_c32 dspark_c32 ar_c96 dspark_c96"
if [ $# -gt 0 ]; then CELLS="$*"; fi

parse_cell() { # sets P_ARM P_C
  local cell=$1
  if [[ $cell =~ ^(.+)_c([0-9]+)$ ]]; then
    P_ARM="${BASH_REMATCH[1]}"; P_C="${BASH_REMATCH[2]}"
    return 0
  fi
  return 1
}

log "=== H-OBS-3 run set: $(echo $CELLS | wc -w) cells (hetero, natural EOS, tracer on) ==="
for cell in $CELLS; do
  parse_cell "$cell" || { log "FATAL: unparseable cell label '$cell'"; exit 1; }
  arm=$P_ARM; C=$P_C
  case $arm in ar|dspark) ;; *) log "FATAL: unknown arm $arm"; exit 1;; esac
  if [ $DRY -eq 1 ]; then
    tbmode="<unset>"; [ "$arm" = "dspark" ] && tbmode=native
    echo "[dry-run] $cell -> TB_MODE=$tbmode OBS_TRACE=ON C=$C"
    continue
  fi
  acquire_lock
  trial "$arm" "$C"
done
log "=== H-OBS-3 run set complete: $(ls $OUT/*/DONE 2>/dev/null | wc -l) valid, $(ls $OUT/*/INVALID 2>/dev/null | wc -l) invalid ==="
