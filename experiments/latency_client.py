"""Priority-A latency client: fixed-concurrency worker pool, STREAMING completions so we
measure per-request TTFT (first token), TPOT ((e2e-ttft)/(n-1)), and E2E latency, plus
steady-state output throughput. Records one JSONL line per completed request.

Usage: latency_client.py PORT DURATION CONCURRENCY WARMUP OUT_JSONL [MAXTOK]
"""
import json, random, threading, time, sys
import httpx

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8100
DURATION = float(sys.argv[2]) if len(sys.argv) > 2 else 150.0
CONCURRENCY = int(sys.argv[3]) if len(sys.argv) > 3 else 16
WARMUP = float(sys.argv[4]) if len(sys.argv) > 4 else 20.0
OUT = sys.argv[5] if len(sys.argv) > 5 else "/tmp/lat.jsonl"
MAXTOK = int(sys.argv[6]) if len(sys.argv) > 6 else 256

base = f"http://127.0.0.1:{PORT}/v1/completions"
TEMPLATES = [
    "List the first 40 prime numbers in ascending order, one per line, then list them again.",
    "Write out the alphabet from A to Z ten times in a row, each letter separated by a space.",
    "Repeat the following sentence exactly 25 times: The quick brown fox jumps over the lazy dog near the river bank.",
    "Count up from 1 to 60 in increments of one, writing each number on its own line twice.",
]
LOW = [
    "Explain the tradeoffs between centralized and distributed databases for a fintech startup.",
    "Write a thoughtful analysis of how remote work has changed urban real estate markets.",
    "Describe the history of the printing press and its impact on religious reform in Europe.",
    "Discuss the ethics of algorithmic pricing in e-commerce, with concrete examples.",
]
random.seed(1234)

def pick():
    return random.choice(TEMPLATES) if random.random() < 0.5 else random.choice(LOW)

lock = threading.Lock()
recs = []
stop_at = [0.0]
t0 = [0.0]

def stream_one(client, prompt):
    ts = time.monotonic()
    ttft = None
    ntok = 0          # actual completion tokens (from usage in final chunk)
    nchunks = 0       # SSE chunks carrying text (for diagnostics only)
    text_len = 0      # fallback token estimate if the stream omits usage
    try:
        with client.stream("POST", base, json={
                "model": "Qwen/Qwen2.5-7B-Instruct", "prompt": prompt,
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
                    # /v1/completions streams carry choices[0].text; chat carries delta.content
                    piece = ch.get("text") or (ch.get("delta") or {}).get("content")
                    if piece:
                        if ttft is None:
                            ttft = time.monotonic() - ts
                        nchunks += 1
                        text_len += len(piece)
                    u = d.get("usage")
                    if u and u.get("completion_tokens"):
                        ntok = u["completion_tokens"]   # authoritative count
                except Exception:
                    pass
    except Exception as e:
        with lock:
            recs.append({"t": time.monotonic() - t0[0], "err": str(e)[:80]})
        return
    if ntok == 0 and text_len > 0:
        # stream omitted usage -> estimate tokens from text length (~4 chars/token)
        ntok = max(1, round(text_len / 4))
    te = time.monotonic() - ts
    if ttft is None or ntok < 2:
        with lock:
            recs.append({"t": time.monotonic() - t0[0], "err": "no-tokens",
                         "ntok": ntok, "nchunks": nchunks})
        return
    tpot = (te - ttft) / (ntok - 1)
    with lock:
        recs.append({"t": time.monotonic() - t0[0], "ttft_ms": ttft * 1e3,
                     "tpot_ms": tpot * 1e3, "e2e_s": te, "ntok": ntok,
                     "nchunks": nchunks})

def worker():
    client = httpx.Client(timeout=900)
    while time.monotonic() < stop_at[0]:
        stream_one(client, pick())

t0[0] = time.monotonic()
stop_at[0] = t0[0] + DURATION
threads = [threading.Thread(target=worker) for _ in range(CONCURRENCY)]
for th in threads: th.start()
for th in threads: th.join(timeout=900)

with open(OUT, "w") as f:
    for r in recs:
        f.write(json.dumps(r) + "\n")

# steady-state summary (completions ending in [t0+WARMUP, t0+DURATION])
ok = [r for r in recs if "ttft_ms" in r]
st = [r for r in ok if WARMUP <= r["t"] <= DURATION]
tok = sum(r["ntok"] for r in st)
window = max(1e-9, min(DURATION, time.monotonic() - t0[0]) - WARMUP)
print(f"concurrency={CONCURRENCY} ok={len(ok)} err={len(recs)-len(ok)} "
      f"steady_out_tokens={tok} steady_throughput_tok_s={tok/window:.1f}")
