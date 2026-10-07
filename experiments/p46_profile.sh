#!/usr/bin/env bash
# Phase 4.6 prerequisite: fresh SPS(B) profile for BOTH paths (AR and SD), dense in B.
#   AR path: SPEC_METHOD=none, C in {8,16,32,48,64,96,128}          -> B = C
#   SD path: SPEC_METHOD=ngram K=4/8, C in {8,16,32,48,64}          -> B = C + prop/s / SPS (measured)
# Output: results/p4_6/sps_profile.jsonl  lines: {"path":"AR|SD","K":..,"C":..,"B":..,"SPS":..,"tok_s":..}
set -uo pipefail
cd "$(dirname "$0")/.."
GPU="${GPU:-0}"; PORT="${PORT:-8100}"
OUT="results/p4_6/sps_profile.jsonl"
mkdir -p results/p4_6
: > "$OUT"

run_cell() { # path SPECMETHOD K C
  local path="$1" SM="$2" K="$3" C="$4"
  local ktag="${K:-none}"
  local tag="p46_prof_${path}_k${ktag}_c${C}"
  echo "=== $tag (path=$path K=$K C=$C) ==="
  GPU=$GPU PORT=$PORT MODEL=Qwen/Qwen2.5-7B-Instruct SPEC_METHOD="$SM" K="$K" \
    MAX_MODEL_LEN=16384 MAX_NUM_SEQS=128 CHUNK_TOKENS=2048 ENFORCE_EAGER=1 \
    ./experiments/run_server.sh "$tag"
  local up=0
  for i in $(seq 1 60); do
    if curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" | grep -q 200; then up=1; break; fi
    sleep 5
  done
  [ "$up" = "1" ] || { echo "SERVER FAILED $tag"; kill $(cat "results/$tag/server.pid") 2>/dev/null; return 1; }
  .venv/bin/python experiments/scrape_metrics.py --port "$PORT" --gpu "$GPU" \
    --out "results/p4_6/${tag}_metrics.jsonl" --interval 0.5 > "results/p4_6/${tag}_scrape.out" 2>&1 &
  local SPID=$!
  .venv/bin/python experiments/latency_client.py "$PORT" 90 "$C" 15 "/tmp/${tag}.jsonl" 256 \
    | tee "results/p4_6/${tag}_client.out"
  kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
  sleep 3
  kill $(cat "results/$tag/server.pid") 2>/dev/null; wait 2>/dev/null
  sleep 5
}

for C in 8 16 32 48 64 96 128; do run_cell AR none "" "$C"; done
for K in 4 8; do for C in 8 16 32 48 64; do run_cell SD ngram "$K" "$C"; done; done

# derive SPS(B) per cell from cumulative counters (same identities as profile_sps.py)
.venv/bin/python - <<'EOF'
import json, os
def c(r,k): return r.get(k,0) or 0
out=[]
for path,K,CS in (("AR","none",[8,16,32,48,64,96,128]),("SD","4",[8,16,32,48,64]),("SD","8",[8,16,32,48,64])):
    for C in CS:
        p=f"results/p4_6/p46_prof_{path}_k{K}_c{C}_metrics.jsonl"
        if not os.path.exists(p): print("WARN missing",p); continue
        rows=[json.loads(l) for l in open(p)]
        mr=[r for r in rows if r.get("t",0)>=15]
        if len(mr)<10: print("WARN short window",p); continue
        a,b=mr[0],mr[-1]; dt=max(1e-9,b["t"]-a["t"])
        gen=(c(b,"vllm:generation_tokens[counter]")-c(a,"vllm:generation_tokens[counter]"))/dt
        if path=="AR":
            B=float(C); sps=gen/C
        else:
            acc=(c(b,"vllm:spec_decode_num_accepted_tokens[counter]")-c(a,"vllm:spec_decode_num_accepted_tokens[counter]"))/dt
            prop=(c(b,"vllm:spec_decode_num_draft_tokens[counter]")-c(a,"vllm:spec_decode_num_draft_tokens[counter]"))/dt
            sps=(gen-acc)/C
            B=C+prop/sps if sps>0 else None
        if not B or sps<=0: print("WARN bad",p); continue
        out.append({"path":path,"K":K,"C":C,"B":round(B,2),"SPS":round(sps,3),"tok_s":round(gen,1)})
with open("results/p4_6/sps_profile.jsonl","w") as f:
    for r in out: f.write(json.dumps(r)+"\n")
print(f"{'path':>3} {'K':>4} {'C':>4} {'B':>7} {'SPS':>8} {'tok_s':>8}")
for r in sorted(out,key=lambda x:(x['path'],x['C'])):
    print(f"{r['path']:>3} {r['K']:>4} {r['C']:>4} {r['B']:>7.1f} {r['SPS']:>8.2f} {r['tok_s']:>8.1f}")
print("wrote", len(out), "points")
EOF
