#!/bin/bash
# H-CLASS-1 run set (prereg docs/hclass1-prereg.md §4-§7).
#   Calibration cells:  cal_ar_<C>      = AR, condition A, Poisson lambda=<C>/28 req/s (90 s)
#   Confirmatory cells: <arm>_<cond>_c<C>_t<T>  arm in {ar,dspark}, cond in {A,B}, T in {1,2,3}
# Condition A: diverse distinct prompts, --no-enable-prefix-caching.
# Condition B: H-OBS-3 canonical 4 prompts (client-side), --enable-prefix-caching.
# Port 8402, GPU1 (H-OBS-3 convention). Fresh server per cell.
#
# Usage: hclass1_run.sh [--dry-run] [cell ...]
set -u
ENV=/home/junior1/_dev/dspark-native-validation
REPO=/home/junior1/_dev/better-vllm-speculative-decoding
OUT=$REPO/results/hclass1
PORT=8402
GPU=${HCLASS1_GPU:-0}   # prereg s3 says GPU1; GPU1 is occupied by an external job (junior2 vllm serve, pid 2859851) so we run on the free GPU0. Override with HCLASS1_GPU=1 if GPU1 frees up.
PY=$ENV/.venv/bin/python
LOCK=/tmp/hclass1_gpu$GPU.lock
DURATION=150
WARMUP=20
SEED=20261013

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$OUT/status.log"; }

acquire_lock() {
  if [ -e "$LOCK" ]; then
    local old; old=$(cat "$LOCK" 2>/dev/null)
    if [ "${old%% *}" != "0" ] && kill -0 "${old%% *}" 2>/dev/null; then
      log "FATAL: GPU lock held by live pid ${old%% *} (another run) — refusing to start"; exit 1
    fi
    rm -f "$LOCK"
  fi
  echo "0 $(date '+%s')" > "$LOCK"
}

gpu_free() {
  local m; m=$(nvidia-smi --id=$GPU --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [ "${m:-9999}" -lt 100 ]
}

# ---- cell parsing: cal_ar_32 | ar_A_c32_t1 ---------------------------------
parse_cell() { # sets P_ARM P_COND P_C P_TRIAL P_LAMBDA P_DUR
  local cell=$1
  if [[ $cell =~ ^cal_(ar)_(\d+)$ ]]; then
    P_ARM="ar"; P_COND="A"; P_C="${BASH_REMATCH[2]}"; P_TRIAL=0
    P_LAMBDA=$(python3 -c "print(round(${BASH_REMATCH[2]}/28.0, 4))")
    P_DUR=90; return 0
  fi
  if [[ $cell =~ ^(ar|dspark)_([AB])_c(\d+)_t([123])$ ]]; then
    P_ARM="${BASH_REMATCH[1]}"; P_COND="${BASH_REMATCH[2]}"
    P_C="${BASH_REMATCH[3]}"; P_TRIAL="${BASH_REMATCH[4]}"
    # lambda per pressure: filled from frozen s5b table (see LAMBDA_TABLE below)
    P_LAMBDA=$(lookup_lambda "$P_C"); P_DUR=$DURATION; return 0
  fi
  return 1
}

# Frozen lambda table (prereg §5b). Filled after calibration; until then these are
# the calibration candidates used for confirmatory cells only AFTER s5b is frozen.
lookup_lambda() { # $1=C -> lambda req/s
  case "$1" in
    8)  echo "0.29" ;;   # ~ C/28: low occupancy, no queue (P_low candidate)
    32) echo "1.14" ;;   # P_med candidate
    96) echo "3.43" ;;   # P_high candidate (sustained queue at max_num_seqs=64)
    *)  echo "0.57" ;;
  esac
}

write_manifest() { # $1=cell
  local cell="${1:-}"
  local dir="$OUT/$cell"
  $PY - "$OUT" "$cell" "$P_ARM" "$P_COND" "$P_C" "$P_TRIAL" "$P_LAMBDA" "$P_DUR" <<'EOF'
import json, os, sys, subprocess
out, cell, arm, cond, C, trial, lam, dur = sys.argv[1:9]
def rev(path):
    try:
        return subprocess.run(["git","-C",path,"rev-parse","HEAD"],capture_output=True,text=True).stdout.strip()
    except Exception:
        return None
spec = [] if arm == "ar" else [
  "--speculative-config",
  '{"method": "dspark", "model": "deepseek-ai/dspark_qwen3_4b_block7", "num_speculative_tokens": 7}']
cache_on = (cond == "B")
m = {
  "cell": cell, "arm": arm, "condition": cond, "C": int(C), "trial": int(trial),
  "lambda_req_s": float(lam), "duration_s": int(dur), "warmup_s": 20, "seed": 20261013,
  "created_utc": __import__("datetime").datetime.utcnow().isoformat()+"Z",
  "worktree_rev": rev("/home/junior1/_dev/dspark-native-validation"),
  "repo_rev": rev("/home/junior1/_dev/better-vllm-speculative-decoding"),
  "server_cmd": ["vllm.entrypoints.openai.api_server","--model","Qwen/Qwen3-4B",
                  "--served-model-name","qwen3-4b","--port","8402","--tensor-parallel-size","1",
                  "--gpu-memory-utilization","0.90","--max-num-seqs","64",
                  "--max-num-batched-tokens","4096"] + (["--enable-prefix-caching"] if cache_on else ["--no-enable-prefix-caching"]) + spec,
  "env": {"VLLM_TB_MODE": (None if arm == "ar" else "native"),
           "VLLM_OBS_TRACE": f"{out}/{cell}/obs"},
  "workload": {"client":"hclass1_client.py","cond":cond,"classes":{"SS":[50,256],
               "SL":[50,1024],"LS":[2048,256],"LL":[2048,1024]},
               "arrival":"poisson" if cond=="A" else "back-to-back-rr",
               "lambda_req_s": float(lam), "greedy": True, "ignore_eos": False},
  "sd_expected": ("off" if arm == "ar" else "on"),
}
json.dump(m, open(os.path.join(out, cell, "config_manifest.json"),"w"), indent=2)
print("manifest:", os.path.join(out, cell, "config_manifest.json"))
EOF
}

start_server() { # $1=cell -> sets SRV_PID
  local cell="${1:-}"
  local dir="$OUT/$cell"
  unset VLLM_TB_MODE VLLM_OBS_TRACE VLLM_TB_TRACE VLLM_LDM_BATCH_FORCEOFF VLLM_LDMLOAD_FORCE0 2>/dev/null || true
  export VLLM_OBS_TRACE="$dir/obs"
  mkdir -p "$dir/obs"; rm -f "$dir/obs/steps.jsonl" "$dir/obs/collector_stats.json"
  [ "$P_ARM" = "dspark" ] && export VLLM_TB_MODE=native
  local spec_args=() cache_arg
  [ "$P_ARM" = "dspark" ] && spec_args=(--speculative-config '{"method": "dspark", "model": "deepseek-ai/dspark_qwen3_4b_block7", "num_speculative_tokens": 7}')
  if [ "$P_COND" = "B" ]; then cache_arg=--enable-prefix-caching; else cache_arg=--no-enable-prefix-caching; fi
  export CUDA_VISIBLE_DEVICES=$GPU PORT=$PORT
  cd "$ENV"
  setsid .venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B --served-model-name qwen3-4b --port $PORT \
    --tensor-parallel-size 1 --gpu-memory-utilization 0.90 \
    --max-num-seqs 64 --max-num-batched-tokens 4096 "$cache_arg" \
    "${spec_args[@]}" > "$dir/server.log" 2>&1 < /dev/null &
  SRV_PID=$!
  echo "$SRV_PID $(date '+%s')" > "$dir/srv_pid"
}

wait_health() {
  local i
  for i in $(seq 1 240); do
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

validate_cell() { # $1=cell -> 0 valid, 1 invalid
  local cell="${1:-}"
  local dir="$OUT/$cell" reasons=()
  grep -q "Application startup complete" "$dir/server.log" 2>/dev/null || reasons+=("no startup-complete in own log")
  [ -f "$dir/client.jsonl" ] || { reasons+=("missing client records"); return 1; }
  local okc errc minreq
  okc=$($PY -c "import json;r=[json.loads(l) for l in open('$dir/client.jsonl')];print(sum(1 for x in r if 'ttft_ms' in x))")
  errc=$($PY -c "import json;r=[json.loads(l) for l in open('$dir/client.jsonl')];print(sum(1 for x in r if 'err' in x))")
  minreq=40; [ "$P_DUR" = "150" ] && minreq=60
  [ "${okc:-0}" -ge $minreq ] || reasons+=("client ok-requests too low ($okc < $minreq)")
  $PY -c "import sys;ok,err=int('${okc:-0}'),int('$errc');sys.exit(0 if err*10 <= max(1,ok+err) else 1)" \
    || reasons+=("client error flood ($errc errors vs $okc ok)")
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
  if [ -f "$dir/obs/collector_stats.json" ]; then
    local dropped; dropped=$($PY -c "import json;print(json.load(open('$dir/obs/collector_stats.json'))['records_dropped'])")
    [ "$dropped" = "0" ] || reasons+=("collector dropped $dropped records")
  fi
  if [ -f "$dir/obs/steps.jsonl" ]; then
    local sdsum; sdsum=$($PY -c "
import json
recs=[json.loads(l) for l in open('$dir/obs/steps.jsonl')]
print(sum(r['draft_tokens'] for r in recs))")
    case $P_ARM in
      ar) [ "$sdsum" = "0" ] || reasons+=("AR arm scheduled $sdsum draft tokens (expected 0)") ;;
      dspark) [ "$sdsum" -gt 1000 ] || reasons+=("dspark arm scheduled only $sdsum draft tokens (expected SD active)") ;;
    esac
  fi
  # async overlap preserved
  if [ -f "$dir/obs/steps.jsonl" ]; then
    local of; of=$($PY -c "
import json
recs=[json.loads(l) for l in open('$dir/obs/steps.jsonl')]
hi=sum(1 for r in recs if r.get('in_flight',0)>=2)
print(round(hi/max(1,len(recs)),3))")
    $PY -c "import sys;sys.exit(0 if float('$of')>=0.9 else 1)" \
      || reasons+=("async overlap broken (in_flight>=2 fraction $of < 0.9)")
  fi
  if [ ${#reasons[@]} -eq 0 ]; then return 0; else log "INVALID $cell: ${reasons[*]}"; return 1; fi
}

trial() { # $1=cell
  local cell="${1:-}"
  local dir="$OUT/$cell"
  if [ -f "$dir/DONE" ] || [ -f "$dir/INVALID" ]; then log "SKIP $cell (already done)"; return 0; fi
  gpu_free || { log "FAIL $cell: GPU$GPU not free at start"; mkdir -p "$dir"; touch "$dir/INVALID"; return 1; }
  log "START $cell (arm=$P_ARM cond=$P_COND C=$P_C lambda=$P_LAMBDA dur=$P_DUR)"
  mkdir -p "$dir"
  write_manifest "$cell"
  local attempt
  for attempt in 1 2; do
    start_server "$cell"
    if wait_health "$cell"; then break; fi
    log "WARN $cell: server unhealthy (attempt $attempt) — transient CUDA-init race seen after rapid restarts"
    tail -5 "$dir/server.log" | tee -a "$OUT/status.log"
    stop_server
    [ $attempt -eq 2 ] && { log "FAIL $cell: server not healthy/verified after retry"; touch "$dir/INVALID"; return 1; }
    sleep 10
  done
  ( cd "$ENV" && .venv/bin/python "$REPO/experiments/scrape_metrics.py" \
      --port $PORT --gpu $GPU --interval 0.5 --out "$dir/metrics.jsonl" ) > "$dir/scraper.log" 2>&1 &
  local SC_PID=$!
  sleep 3
  cd "$ENV"
  timeout $((P_DUR + 300)) .venv/bin/python "$REPO/experiments/hclass1_client.py" \
      $PORT $P_DUR $P_C $WARMUP "$dir/client.jsonl" $SEED $P_COND $P_LAMBDA \
      "$dir/prompt_manifest.jsonl" > "$dir/client.log" 2>&1
  local CRC=$?
  kill $SC_PID 2>/dev/null; wait $SC_PID 2>/dev/null
  sleep 3   # let atexit final_flush drain the last steps' events
  stop_server
  if [ $CRC -ne 0 ]; then log "FAIL $cell: client rc=$CRC"; touch "$dir/INVALID"; return 1; fi
  if validate_cell "$cell"; then
    touch "$dir/DONE"; log "DONE-VALID $cell"
  else
    touch "$dir/INVALID"
  fi
  sleep 10   # settle delay after teardown: transient CUDA-init race seen when the next server starts immediately
}

# ---------------- main ----------------
mkdir -p "$OUT"
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
if [ $DRY -eq 1 ]; then shift; fi

CAL_CELLS="cal_ar_8 cal_ar_16 cal_ar_32 cal_ar_48 cal_ar_96"
CONFIRM_CELLS=""
for cond in A B; do
  for arm in ar dspark; do
    for C in 8 32 96; do
      for t in 1 2 3; do CONFIRM_CELLS="$CONFIRM_CELLS ${arm}_${cond}_c${C}_t${t}"; done
    done
  done
done

CELLS="${CAL_CELLS} $CONFIRM_CELLS"
if [ $# -gt 0 ]; then CELLS="$*"; fi

log "=== H-CLASS-1 run set: $(echo $CELLS | wc -w) cells requested ==="
for cell in $CELLS; do
  parse_cell "$cell" || { log "FATAL: unparseable cell label '$cell'"; exit 1; }
  if [ $DRY -eq 1 ]; then
    cache="off"; [ "$P_COND" = "B" ] && cache="on"
    echo "[dry-run] $cell -> arm=$P_ARM cond=$P_COND C=$P_C lambda=$P_LAMBDA dur=${P_DUR}s cache=$cache"
    continue
  fi
  acquire_lock
  trial "$cell"
done
log "=== H-CLASS-1 run set complete: $(ls $OUT/*/DONE 2>/dev/null | wc -l) valid, $(ls $OUT/*/INVALID 2>/dev/null | wc -l) invalid ==="
