"""Sustained Poisson mixed workload for a fixed duration, with STEADY-STATE throughput
measurement (warmup excluded). HIGH ngram-acceptance = templated prompts that induce
self-repetitive output; LOW = diverse free-form text. Concurrency is bounded by a
max-in-flight cap so the server stays busy for the whole window (no burst-then-drain)."""
import random, threading, time, sys
import httpx

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8100
DURATION = float(sys.argv[2]) if len(sys.argv) > 2 else 150.0   # total seconds
RATE = float(sys.argv[3]) if len(sys.argv) > 3 else 6.0         # arrivals/sec
WARMUP = float(sys.argv[4]) if len(sys.argv) > 4 else 20.0      # excluded from throughput
MAXTOK = 256
MAX_INFLIGHT = 96

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
inflight = [0]
tok_done = []          # (t, tokens) completion events
sent = [0]

def send(prompt):
    with lock:
        inflight[0] += 1
    try:
        with httpx.Client(timeout=900) as c:
            r = c.post(base, json={"model": "Qwen/Qwen2.5-7B-Instruct", "prompt": prompt,
                                   "max_tokens": MAXTOK, "temperature": 0})
            try:
                u = r.json()["usage"]["completion_tokens"]
            except Exception:
                u = 0
        with lock:
            tok_done.append((time.monotonic(), u))
    except Exception:
        pass
    finally:
        with lock:
            inflight[0] -= 1

t0 = time.monotonic()
stop_at = t0 + DURATION
while time.monotonic() < stop_at:
    with lock:
        busy = inflight[0] >= MAX_INFLIGHT
    if not busy:
        send(pick())
        sent[0] += 1
    time.sleep(random.expovariate(RATE) / max(1.0, RATE / 6.0))  # ~Poisson at RATE/s

# let in-flight finish (bounded)
deadline = time.monotonic() + 300
while time.monotonic() < deadline:
    with lock:
        if inflight[0] == 0:
            break
    time.sleep(1)

# steady-state throughput: completions whose END time is in [t0+WARMUP, t0+DURATION]
w_lo = t0 + WARMUP
w_hi = t0 + DURATION
tok = sum(u for t, u in tok_done if w_lo <= t <= w_hi)
window = max(1e-9, min(w_hi, time.monotonic()) - w_lo)
print(f"sent={sent[0]} completions={len(tok_done)} "
      f"steady_window={window:.0f}s steady_out_tokens={tok} "
      f"steady_throughput_tok_s={tok/window:.1f}")
