#!/bin/bash
# H-OBS-1 main run set (prereg docs/hoobs1-prereg.md §7A amendment A1).
# 18 cells = arm{ar,dspark,ctrl_live} x C{8,32,64} x tracer{on,off}, fresh server each.
# + 1 non-counted nsys diagnostic (dspark C=8) for Gate-1 independent-trace validation.
#
# Arm resolution is table-driven in scripts/hoobs1_arms.py (unit self-tested).
# This launcher shells out to it so the SAME logic that is tested launches servers.
# Usage:
#   run_hoobs1.sh [--dry-run] [cell ...]     # default: all 18 cells in frozen order
# A cell = <arm>_c<C>_<tracer>, e.g. dspark_c32_on.
set -u
ENV=/home/junior1/_dev/dspark-native-validation
REPO=/home/junior1/_dev/better-vllm-speculative-decoding
OUT=$REPO/results/hoobs1/main
PORT=8400
GPU=1
DATASET=$REPO/data/ShareGPT_V3_unfiltered_cleaned_split.json
PY=$ENV/.venv/bin/python
MANIFEST=$OUT/manifest.jsonl
LOCK=/tmp/hoobs1_gpu1.lock

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$OUT/status.log"; }

# ---- resource lock: one performance experiment per reserved GPU (directive §9) ----
# The lock guards against a CONCURRENT run grabbing the same GPU; it must NOT trip
# on this launcher's own sequential cells. We therefore record a sentinel owner
# ("0" = no real PID), not $$, so our own cell loop never looks like an external
# holder. A genuinely concurrent second invocation would be detected by gpu_free
# (the first run holds ~24GB) and by the per-cell startup wait, in addition to this
# lock. Stale locks from a dead process are reclaimed.
acquire_lock() {
  if [ -e "$LOCK" ]; then
    local old; old=$(cat "$LOCK" 2>/dev/null)
    # "0" is our own sentinel -> reclaimable (we're the only writer). A real PID
    # that is still alive means another run owns the GPU -> refuse.
    if [ "${old%% *}" != "0" ] && kill -0 "${old%% *}" 2>/dev/null; then
      log "FATAL: GPU lock held by live pid ${old%% *} (another run) — refusing to start"
      exit 1
    fi
    rm -f "$LOCK"   # stale or our own sentinel
  fi
  echo "0 $(date '+%s')" > "$LOCK"
}

gpu_free() {
  local m; m=$(nvidia-smi --id=$GPU --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [ "${m:-9999}" -lt 100 ]
}

# ---- resolve one cell's effective config via the table-driven arm module ----
resolve_cell() { # $1=arm $2=tracer -> sets CFG_* and RESOLVED_CMD
  local arm=$1 tracer=$2
  local cfg; cfg=$($PY "$ENV/scripts/hoobs1_arms.py" "$arm" "$tracer") || return 1
  CLI_SPEC=$($PY -c "import json,sys; c=json.loads(sys.stdin.read()); print(' '.join(c['cli_args']))" <<<"$cfg")
  TB_MODE=$($PY -c "import json,sys; c=json.loads(sys.stdin.read()); print(c['env_set'].get('VLLM_TB_MODE',''))" <<<"$cfg")
  SD_EXPECTED=$($PY -c "import json,sys; c=json.loads(sys.stdin.read()); print(c['sd_expected'])" <<<"$cfg")
}

# ---- per-cell effective-config manifest (directive §3A) ----
write_manifest() { # $1=cell $2=arm $3=C $4=tracer
  local cell=$1 arm=$2 C=$3 tracer=$4
  $PY - "$OUT" "$cell" "$arm" "$C" "$tracer" <<'EOF'
import json, os, sys, hashlib
out, cell, arm, C, tracer = sys.argv[1:6]
d = os.path.join(out, cell); os.makedirs(d, exist_ok=True)
def rev(path):
    try:
        import subprocess
        return subprocess.run(["git","-C",path,"rev-parse","HEAD"],capture_output=True,text=True).stdout.strip()
    except Exception:
        return None
m = {
  "cell": cell, "arm": arm, "C": int(C), "tracer": tracer,
  "created_utc": __import__("datetime").datetime.utcnow().isoformat()+"Z",
  "worktree_rev": rev("/home/junior1/_dev/dspark-native-validation"),
  "repo_rev": rev("/home/junior1/_dev/better-vllm-speculative-decoding"),
  "server_cmd": ["vllm.entrypoints.openai.api_server","--model","Qwen/Qwen3-4B",
                  "--served-model-name","qwen3-4b","--port","8400","--tensor-parallel-size","1",
                  "--gpu-memory-utilization","0.90","--max-num-seqs","64",
                  "--max-num-batched-tokens","4096","--enable-prefix-caching"] + (
                    ["--speculative-config",'{"method": "dspark", "model": "deepseek-ai/dspark_qwen3_4b_block7", "num_speculative_tokens": 7}'] if arm != "ar" else []),
  "env": {"VLLM_TB_MODE": ({"ar":None,"dspark":"native","ctrl_live":"live"}[arm]),
           "VLLM_OBS_TRACE": (f"{out}/{cell}/obs" if tracer=="on" else None)},
  "workload": {"dataset":"ShareGPT-V3","num_prompts":60,"max_concurrency":int(C),
               "greedy":True,"seed":0,"ignore_eos":True,"num_warmups":0},
  "sd_expected": ({"ar":"off","dspark":"on","ctrl_live":"on-off"}[arm]),
}
p = os.path.join(d, "config_manifest.json")
json.dump(m, open(p,"w"), indent=2)
print("manifest:", p)
EOF
}

start_server() { # $1=cell $2=arm $3=tracer -> sets SRV_PID
  local cell=$1 arm=$2 tracer=$3
  local dir="$OUT/$cell"
  # Clean subprocess environment (directive §3A): clear conflicting inherited settings.
  unset VLLM_TB_MODE VLLM_OBS_TRACE VLLM_TB_TRACE VLLM_LDM_BATCH_FORCEOFF VLLM_LDMLOAD_FORCE0 2>/dev/null || true
  if [ "$tracer" = "on" ]; then
    export VLLM_OBS_TRACE="$dir/obs"
    mkdir -p "$dir/obs"; rm -f "$dir/obs/steps.jsonl" "$dir/obs/collector_stats.json"
  fi
  case $arm in
    dspark)    export VLLM_TB_MODE=native ;;
    ctrl_live) export VLLM_TB_MODE=live ;;
    ar)        : ;;   # no controller mode at all
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
  echo "$SRV_PID $(date '+%s')" > "$dir/srv_pid"   # process ownership record
}

wait_health() { # $1=cell; verifies OUR server, not an orphan (Phase-4.4 lesson)
  local cell=$1 i
  for i in $(seq 1 210); do
    if curl -s -o /dev/null -w "%{http_code}" --max-time 2 http://localhost:$PORT/health | grep -q 200; then
      # startup-complete marker must be in OUR log AND our pid alive
      if grep -q "Application startup complete" "$OUT/$cell/server.log" 2>/dev/null && kill -0 $SRV_PID 2>/dev/null; then
        return 0
      fi
    fi
    sleep 2
  done
  return 1
}

stop_server() {
  # Kill the whole process group (setsid), verify port goes quiet.
  local pid=$SRV_PID
  kill -- -"$pid" 2>/dev/null || kill "$pid" 2>/dev/null
  for i in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -9 -- -"$pid" 2>/dev/null; kill -9 "$pid" 2>/dev/null
  # wait for GPU to actually be free before the next cell
  for i in $(seq 1 60); do gpu_free && return 0; sleep 1; done
  log "WARN: GPU$GPU still busy after stop (mem=$(nvidia-smi --id=$GPU --query-gpu=memory.used --format=csv,noheader,nounits))"
}

validate_cell() { # $1=cell $2=arm $3=tracer -> 0 valid, 1 invalid
  local cell=$1 arm=$2 tracer=$3 dir="$OUT/$cell"
  local reasons=()
  grep -q "Application startup complete" "$dir/server.log" 2>/dev/null || reasons+=("no startup-complete in own log")
  [ -f "$dir/bench_result.json" ] || reasons+=("missing bench result")
  if [ "$tracer" = "on" ]; then
    local nsteps=0 dropped=1 gpufrac
    [ -f "$dir/obs/steps.jsonl" ] && nsteps=$(wc -l < "$dir/obs/steps.jsonl")
    [ "${nsteps:-0}" -gt 50 ] || reasons+=("obs trace missing/too short (${nsteps:-0} steps)")
    # Collector stats file must exist (proves the tracer ran). In vLLM V1 abort-mode
    # shutdown final_flush may not run, so records_dropped is only meaningful when it
    # did; the authoritative integrity check is that gpu_step_ms is actually populated
    # on a healthy fraction of steps (pending tail records lost to abort are few and
    # fall in the excluded ramp/tail windows).
    if [ -f "$dir/obs/collector_stats.json" ]; then
      dropped=$($PY -c "import json;print(json.load(open('$dir/obs/collector_stats.json'))['records_dropped'])")
      [ "$dropped" = "0" ] || reasons+=("collector dropped $dropped records (final_flush ran)")
    else
      log "NOTE $cell: collector_stats.json absent (abort-mode shutdown); validating via gpu-populated fraction"
    fi
    if [ -f "$dir/obs/steps.jsonl" ]; then
      gpufrac=$($PY -c "
import json
recs=[json.loads(l) for l in open('$dir/obs/steps.jsonl')]
pop=sum(1 for r in recs if r.get('gpu_step_ms') is not None)
print(round(100.0*pop/max(1,len(recs)),1))")
      # >=95% of steps must carry a real gpu_step_ms (tail-loss to abort is the only
      # expected shortfall, and it's small).
      $PY -c "import sys;sys.exit(0 if float('$gpufrac')>=95.0 else 1)" \
        || reasons+=("only ${gpufrac}% of steps have gpu_step_ms populated")
    fi
  fi
  # arm-specific SD behavior from the trace (or metrics for tracer-off)
  if [ "$tracer" = "on" ] && [ -f "$dir/obs/steps.jsonl" ]; then
    local sdsum; sdsum=$($PY -c "
import json
recs=[json.loads(l) for l in open('$dir/obs/steps.jsonl')]
print(sum(r['draft_tokens'] for r in recs))")
    case $arm in
      ar) [ "$sdsum" = "0" ] || reasons+=("AR arm scheduled $sdsum draft tokens (expected 0)") ;;
      dspark|ctrl_live) [ "$sdsum" -gt 1000 ] || reasons+=("$arm arm scheduled only $sdsum draft tokens (expected SD active)") ;;
    esac
  fi
  if [ ${#reasons[@]} -eq 0 ]; then return 0; else log "INVALID $cell: ${reasons[*]}"; return 1; fi
}

trial() { # $1=arm $2=C $3=tracer
  local arm=$1 C=$2 tracer=$3 cell="${arm}_c${C}_${tracer}"
  local dir="$OUT/$cell"
  if [ -f "$dir/DONE" ] || [ -f "$dir/INVALID" ]; then log "SKIP $cell (already done)"; return 0; fi
  gpu_free || { log "FAIL $cell: GPU$GPU not free at start"; touch "$dir/INVALID"; return 1; }
  log "START $cell"
  mkdir -p "$dir"
  write_manifest "$cell" "$arm" "$C" "$tracer"
  start_server "$cell" "$arm" "$tracer"
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
  timeout 500 .venv/bin/vllm bench serve --backend openai-chat --base-url http://localhost:$PORT \
    --endpoint /v1/chat/completions --model qwen3-4b --tokenizer Qwen/Qwen3-4B \
    --dataset-name sharegpt --dataset-path "$DATASET" \
    --num-prompts 60 --max-concurrency $C --ignore-eos --temperature 0 \
    --seed 0 --num-warmups 0 \
    --percentile-metrics ttft,tpot,itl,e2el --label "$cell" --save-result \
    --result-filename bench_result.json --result-dir "$dir" \
    > "$dir/bench.log" 2>&1
  local BRC=$?
  kill $SC_PID 2>/dev/null; wait $SC_PID 2>/dev/null
  sleep 3   # let atexit final_flush drain the last steps' events (documented in prereg)
  stop_server
  if [ $BRC -ne 0 ]; then log "FAIL $cell: bench rc=$BRC"; touch "$dir/INVALID"; return 1; fi
  if validate_cell "$cell" "$arm" "$tracer"; then
    touch "$dir/DONE"; log "DONE-VALID $cell"
  else
    touch "$dir/INVALID"; log "DONE-INVALID $cell (artifacts preserved)"
  fi
}

# ---------------- main ----------------
mkdir -p "$OUT"
touch "$MANIFEST"
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
if [ $DRY -eq 1 ]; then shift; fi

# frozen order (prereg §7A): C-major, arms interleaved, tracer on/off adjacent
CELLS=""
for C in 8 32 64; do for arm in ar dspark ctrl_live; do for tr in on off; do CELLS="$CELLS ${arm}_c${C}_${tr}"; done; done; done
if [ $# -gt 0 ]; then CELLS="$*"; fi

# parse a cell label <arm>_c<C>_<tracer>; arm may contain underscores (ctrl_live)
parse_cell() { # sets P_ARM P_C P_TR ; returns 1 on bad label
  local cell=$1
  if [[ $cell =~ ^(.+)_c([0-9]+)_(on|off)$ ]]; then
    P_ARM="${BASH_REMATCH[1]}"; P_C="${BASH_REMATCH[2]}"; P_TR="${BASH_REMATCH[3]}"
    return 0
  fi
  return 1
}

log "=== H-OBS-1 main run set: $(echo $CELLS | wc -w) cells ==="
for cell in $CELLS; do
  parse_cell "$cell" || { log "FATAL: unparseable cell label '$cell' (want <arm>_c<C>_<tracer>)"; exit 1; }
  arm=$P_ARM; C=$P_C; tracer=$P_TR
  resolve_cell "$arm" "$tracer" || { log "FATAL: unknown cell $cell"; exit 1; }
  if [ $DRY -eq 1 ]; then
    echo "[dry-run] $cell -> TB_MODE=${TB_MODE:-<unset>} OBS_TRACE=$([ "$tracer" = on ] && echo ON || echo OFF) spec_args=[$CLI_SPEC]"
    continue
  fi
  acquire_lock
  trial "$arm" "$C" "$tracer"
done

# ---- Gate-1 independent-trace diagnostic (nsys, dspark C=8) — non-counted ----
if [ $DRY -eq 0 ] && [ ! -f "$OUT/nsys_c8/DONE" ]; then
  log "START nsys_c8 (Gate-1 independent GPU trace)"
  bash "$ENV/scripts/hoobs1_profile_check.sh" "$OUT/nsys_c8" && touch "$OUT/nsys_c8/DONE" || { log "nsys diagnostic incomplete (see log)"; }
fi
log "=== H-OBS-1 run set complete: $(ls $OUT/*/DONE 2>/dev/null | wc -l) valid, $(ls $OUT/*/INVALID 2>/dev/null | wc -l) invalid ==="
