"""SPEED-Bench client: fixed-concurrency pool over parquet splits, streaming completions.

Loads nvidia/SPEED-Bench parquet (qualitative or throughput_<isl>), sends prompts with a
fixed worker pool, and records per-request: category/sub_category/difficulty, TTFT, TPOT,
E2E, completion tokens (authoritative via usage). One JSONL line per request.

Usage: speed_bench_client.py PORT SPLIT_DIR DURATION CONCURRENCY WARMUP OUT_JSONL [MAXTOK]
  SPLIT_DIR e.g. data/speed_bench/qualitative or data/speed_bench/throughput_2k
"""
import json, random, sys, threading, time, glob, os
import httpx
import pyarrow.parquet as pq

PORT = int(sys.argv[1])
SPLIT_DIR = sys.argv[2]
DURATION = float(sys.argv[3]) if len(sys.argv) > 3 else 150.0
CONCURRENCY = int(sys.argv[4]) if len(sys.argv) > 4 else 16
WARMUP = float(sys.argv[5]) if len(sys.argv) > 5 else 20.0
OUT = sys.argv[6] if len(sys.argv) > 6 else "/tmp/sb.jsonl"
MAXTOK = int(sys.argv[7]) if len(sys.argv) > 7 else 256

base = f"http://127.0.0.1:{PORT}/v1/completions"
MODEL = "Qwen/Qwen2.5-7B-Instruct"

def load_prompts(split_dir, seed=1234):
    rows = []
    for f in sorted(glob.glob(os.path.join(split_dir, "*.parquet"))):
        t = pq.read_table(f).to_pylist()
        rows.extend(t)
    random.Random(seed).shuffle(rows)
    prompts = []
    for r in rows:
        # single-turn entries: first turn; multiturn: join turns as one prompt (deviation, documented)
        turns = r.get("turns") or []
        if not turns:
            continue
        p = "\n".join(turns) if len(turns) > 1 else turns[0]
        prompts.append({"prompt": p, "category": r.get("category"),
                        "sub_category": r.get("sub_category"),
                        "difficulty": r.get("difficulty")})
    return prompts

PROMPTS = load_prompts(SPLIT_DIR)
lock = threading.Lock()
idx = [0]
recs = []
stop_at = [0.0]
t0 = [0.0]

def stream_one(client, item):
    ts = time.monotonic()
    ttft = None
    ntok = 0
    text_len = 0
    try:
        with client.stream("POST", base, json={
                "model": MODEL, "prompt": item["prompt"],
                "max_tokens": MAXTOK, "temperature": 0, "stream": True,
                "stream_options": {"include_usage": True}}) as r:
            for line in r.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    d = json.loads(payload)
                    ch = (d.get("choices") or [{}])[0]
                    piece = ch.get("text") or (ch.get("delta") or {}).get("content")
                    if piece:
                        if ttft is None:
                            ttft = time.monotonic() - ts
                        text_len += len(piece)
                    u = d.get("usage")
                    if u and u.get("completion_tokens"):
                        ntok = u["completion_tokens"]
                except Exception:
                    pass
    except Exception as e:
        with lock:
            recs.append({"t": time.monotonic() - t0[0], "err": str(e)[:80], **{k: item[k] for k in ("category",)}})
        return
    if ntok == 0 and text_len > 0:
        ntok = max(1, round(text_len / 4))
    te = time.monotonic() - ts
    if ttft is None or ntok < 2:
        with lock:
            recs.append({"t": time.monotonic() - t0[0], "err": "no-tokens", "ntok": ntok,
                         "category": item["category"]})
        return
    tpot = (te - ttft) / (ntok - 1)
    with lock:
        recs.append({"t": time.monotonic() - t0[0], "ttft_ms": ttft * 1e3,
                     "tpot_ms": tpot * 1e3, "e2e_s": te, "ntok": ntok,
                     "category": item["category"], "sub_category": item["sub_category"],
                     "difficulty": item["difficulty"]})

def worker():
    client = httpx.Client(timeout=900)
    while time.monotonic() < stop_at[0]:
        with lock:
            if idx[0] >= len(PROMPTS):
                break
            item = PROMPTS[idx[0]]
            idx[0] += 1
        stream_one(client, item)

t0[0] = time.monotonic()
stop_at[0] = t0[0] + DURATION
threads = [threading.Thread(target=worker) for _ in range(CONCURRENCY)]
for th in threads: th.start()
for th in threads: th.join(timeout=900)

with open(OUT, "w") as f:
    for r in recs:
        f.write(json.dumps(r) + "\n")

ok = [r for r in recs if "ttft_ms" in r]
st = [r for r in ok if WARMUP <= r["t"] <= DURATION]
tok = sum(r["ntok"] for r in st)
window = max(1e-9, min(DURATION, time.monotonic() - t0[0]) - WARMUP)
print(f"split={os.path.basename(SPLIT_DIR)} concurrency={CONCURRENCY} ok={len(ok)} "
      f"err={len(recs)-len(ok)} steady_out_tokens={tok} steady_throughput_tok_s={tok/window:.1f}")
