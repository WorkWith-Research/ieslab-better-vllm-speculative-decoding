"""Sustained Poisson mixed workload: HIGH ngram-acceptance (templated) + LOW
(diverse ShareGPT-style) prompts, for a fixed duration. Used to A/B LDM (adaptive K)
vs fixed-K end-to-end. High-acceptance class = templated prompts that induce
self-repetitive output (ngram matches context -> high acceptance). Low class =
diverse free-form text (low ngram acceptance)."""
import json, random, threading, time, sys
import httpx

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8100
DURATION = float(sys.argv[2]) if len(sys.argv) > 2 else 120.0   # seconds
RATE = float(sys.argv[3]) if len(sys.argv) > 3 else 6.0         # arrivals/sec
HIGH_FRAC = 0.5
MAXTOK = 256

base = f"http://127.0.0.1:{PORT}/v1/completions"
# HIGH-acceptance: templated, self-repetitive output (ngram-friendly)
TEMPLATES = [
    "List the first 40 prime numbers in ascending order, one per line, then list them again.",
    "Write out the alphabet from A to Z ten times in a row, each letter separated by a space.",
    "Repeat the following sentence exactly 25 times: The quick brown fox jumps over the lazy dog near the river bank.",
    "Count up from 1 to 60 in increments of one, writing each number on its own line twice.",
]
# LOW-acceptance: diverse free-form (ngram-unfriendly)
LOW = [
    "Explain the tradeoffs between centralized and distributed databases for a fintech startup.",
    "Write a thoughtful analysis of how remote work has changed urban real estate markets.",
    "Describe the history of the printing press and its impact on religious reform in Europe.",
    "Discuss the ethics of algorithmic pricing in e-commerce, with concrete examples.",
]
random.seed(1234)

def pick():
    if random.random() < HIGH_FRAC:
        return random.choice(TEMPLATES)
    return random.choice(LOW)

done = 0
sent = 0
tok_lock = threading.Lock()
total_tokens = [0]
stop_at = time.monotonic() + DURATION

def send(prompt):
    global done, sent
    try:
        with httpx.Client(timeout=600) as c:
            r = c.post(base, json={"model": "Qwen/Qwen2.5-7B-Instruct", "prompt": prompt,
                                   "max_tokens": MAXTOK, "temperature": 0})
            try:
                u = r.json()["usage"]["completion_tokens"]
                with tok_lock:
                    total_tokens[0] += u
            except Exception:
                pass
    except Exception:
        pass
    finally:
        done += 1

threads = []
t0 = time.monotonic()
while time.monotonic() < stop_at and len(threads) < 64:
    # Poisson-ish inter-arrival
    time.sleep(random.expovariate(RATE))
    th = threading.Thread(target=send, args=(pick(),))
    th.start()
    threads.append(th)
    sent += 1
for th in threads:
    th.join(timeout=600)
elapsed = time.monotonic() - t0
print(f"sent={sent} completed={done} duration~{elapsed:.0f}s "
      f"out_tokens={total_tokens[0]} throughput_tok_s={total_tokens[0]/max(elapsed,1e-9):.1f}")
