"""Sustained CONCURRENT mixed workload for a fixed duration, steady-state throughput
(warmup excluded). A fixed pool of CONCURRENCY workers each loops: send a prompt, wait
for completion, immediately send the next -> exactly CONCURRENCY requests in flight at
all times (no burst-then-drain, no serial sending).

HIGH ngram-acceptance = templated prompts that induce self-repetitive output;
LOW = diverse free-form text. 50/50 mix by default."""
import random, threading, time, sys
import httpx

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8100
DURATION = float(sys.argv[2]) if len(sys.argv) > 2 else 150.0   # total seconds
CONCURRENCY = int(sys.argv[3]) if len(sys.argv) > 3 else 16     # requests in flight
WARMUP = float(sys.argv[4]) if len(sys.argv) > 4 else 20.0      # excluded from throughput
MAXTOK = 256

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
tok_done = []          # (t, tokens) completion events
stop_at = [0.0]

def worker():
    client = httpx.Client(timeout=900)
    while time.monotonic() < stop_at[0]:
        try:
            r = client.post(base, json={"model": "Qwen/Qwen2.5-7B-Instruct",
                                        "prompt": pick(), "max_tokens": MAXTOK, "temperature": 0})
            try:
                u = r.json()["usage"]["completion_tokens"]
            except Exception:
                u = 0
        except Exception:
            u = 0
        with lock:
            tok_done.append((time.monotonic(), u))

t0 = time.monotonic()
stop_at[0] = t0 + DURATION
threads = [threading.Thread(target=worker) for _ in range(CONCURRENCY)]
for th in threads: th.start()
for th in threads: th.join(timeout=900)

w_lo = t0 + WARMUP
w_hi = t0 + DURATION
tok = sum(u for t, u in tok_done if w_lo <= t <= w_hi)
window = max(1e-9, min(w_hi, time.monotonic()) - w_lo)
print(f"concurrency={CONCURRENCY} completions={len(tok_done)} "
      f"steady_window={window:.0f}s steady_out_tokens={tok} "
      f"steady_throughput_tok_s={tok/window:.1f}")
