"""Phase 4.4 heterogeneous ISL x OSL client (pre-registered in PROGRESS.md).

Four request classes, equal weight (round-robin per worker), deterministic seed:
  SS: ISL ~50    tok, max_tokens 256   (chat turn)
  SL: ISL ~50    tok, max_tokens 1024  (short prompt / long generation)
  LS: ISL ~2048  tok, max_tokens 256   (RAG: long context, short answer)
  LL: ISL ~2048  tok, max_tokens 1024  (document QA, long response)

Prompts are built from the ShareGPT V3 corpus and tokenized EXTERNALLY with the
served model's tokenizer (add_special_tokens=False), sent as integer id arrays so
the engine receives exact ISL (same methodology as speed_bench_client.py).

Per-request record: class, ttft_ms, tpot_ms, e2e_s, ntok, isl. Summary prints
aggregate + per-class steady-state throughput and latency percentiles.

Usage: hetero_client.py PORT DURATION CONCURRENCY WARMUP OUT_JSONL [SEED]
"""
import json, random, sys, threading, time
from collections import defaultdict

import httpx
from transformers import AutoTokenizer

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8100
DURATION = float(sys.argv[2]) if len(sys.argv) > 2 else 150.0
CONCURRENCY = int(sys.argv[3]) if len(sys.argv) > 3 else 16
WARMUP = float(sys.argv[4]) if len(sys.argv) > 4 else 20.0
OUT = sys.argv[5] if len(sys.argv) > 5 else "/tmp/hetero.jsonl"
SEED = int(sys.argv[6]) if len(sys.argv) > 6 else 1234

MODEL = "Qwen/Qwen2.5-7B-Instruct"
CORPUS = "data/ShareGPT_V3_unfiltered_cleaned_split.json"
CLASSES = {
    "SS": {"isl": 50, "maxtok": 256},
    "SL": {"isl": 50, "maxtok": 1024},
    "LS": {"isl": 2048, "maxtok": 256},
    "LL": {"isl": 2048, "maxtok": 1024},
}
ORDER = ["SS", "SL", "LS", "LL"]

random.seed(SEED)

def load_turns(path):
    data = json.load(open(path))
    turns = []
    for conv in data:
        msgs = conv.get("conversations") or []
        for m in msgs:
            v = (m.get("value") or "").strip()
            if 40 < len(v) < 20000:
                turns.append(v)
    return turns

TURNS = load_turns(CORPUS)
random.shuffle(TURNS)
TOK = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

_cache = {}
def make_prompt(isl_target):
    """Concatenate shuffled turns until token count >= target; truncate to target."""
    if isl_target in _cache:
        return _cache[isl_target]
    ids = []
    used = set()
    i = random.randrange(len(TURNS))
    while len(ids) < isl_target and len(used) < 64:
        j = (i + len(used)) % len(TURNS)
        if j in used:
            continue
        used.add(j)
        ids.extend(TOK.encode(TURNS[j], add_special_tokens=False))
    ids = ids[:isl_target]
    _cache[isl_target] = ids
    return ids

PROMPTS = {c: make_prompt(cfg["isl"]) for c, cfg in CLASSES.items()}

base = f"http://127.0.0.1:{PORT}/v1/completions"
lock = threading.Lock()
recs = []
stop_at = [0.0]
t0 = [0.0]

def stream_one(client, cls):
    cfg = CLASSES[cls]
    ts = time.monotonic()
    ttft = None
    ntok = 0
    text_len = 0
    try:
        with client.stream("POST", base, json={
                "model": MODEL, "prompt": PROMPTS[cls],
                "max_tokens": cfg["maxtok"], "temperature": 0, "stream": True,
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
            recs.append({"t": time.monotonic() - t0[0], "cls": cls, "err": str(e)[:80]})
        return
    if ntok == 0 and text_len > 0:
        ntok = max(1, round(text_len / 4))
    te = time.monotonic() - ts
    if ttft is None or ntok < 2:
        with lock:
            recs.append({"t": time.monotonic() - t0[0], "cls": cls, "err": "no-tokens",
                         "ntok": ntok})
        return
    tpot = (te - ttft) / (ntok - 1)
    with lock:
        recs.append({"t": time.monotonic() - t0[0], "cls": cls,
                     "ttft_ms": ttft * 1e3, "tpot_ms": tpot * 1e3, "e2e_s": te,
                     "ntok": ntok, "isl": len(PROMPTS[cls])})

def worker(wid):
    client = httpx.Client(timeout=900)
    step = 0
    while time.monotonic() < stop_at[0]:
        cls = ORDER[(wid + step) % 4]   # round-robin keeps the 25% class mix
        stream_one(client, cls)
        step += 1

t0[0] = time.monotonic()
stop_at[0] = t0[0] + DURATION
threads = [threading.Thread(target=worker, args=(i,)) for i in range(CONCURRENCY)]
for th in threads:
    th.start()
for th in threads:
    th.join(timeout=900)

with open(OUT, "w") as f:
    for r in recs:
        f.write(json.dumps(r) + "\n")

ok = [r for r in recs if "ttft_ms" in r]
st = [r for r in ok if WARMUP <= r["t"] <= DURATION]
window = max(1e-9, min(DURATION, time.monotonic() - t0[0]) - WARMUP)
tok = sum(r["ntok"] for r in st)
print(f"concurrency={CONCURRENCY} ok={len(ok)} err={len(recs)-len(ok)} "
      f"steady_out_tokens={tok} steady_throughput_tok_s={tok/window:.1f}")

def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1)))))
    return xs[k]

print(f"{'class':>5} {'n':>5} {'tok/s':>8} {'TTFT p50/p95 ms':>16} {'TPOT p50/p95 ms':>16} {'ntok p50':>9}")
for c in ORDER:
    rc = [r for r in st if r["cls"] == c]
    if not rc:
        print(f"{c:>5} {0:>5}")
        continue
    ctok = sum(r["ntok"] for r in rc) / window
    print(f"{c:>5} {len(rc):>5} {ctok:>8.1f} "
          f"{pct([r['ttft_ms'] for r in rc], 50):>7.1f}/{pct([r['ttft_ms'] for r in rc], 95):<7.1f} "
          f"{pct([r['tpot_ms'] for r in rc], 50):>7.1f}/{pct([r['tpot_ms'] for r in rc], 95):<7.1f} "
          f"{pct([r['ntok'] for r in rc], 50):>9.0f}")
