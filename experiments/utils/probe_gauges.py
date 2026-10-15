#!/usr/bin/env python3
"""Probe vLLM /metrics scheduler gauges under real concurrent load.
Starts an AR (no-SD) Qwen2.5-7B server on :8199, fires 8 concurrent streaming
requests, samples /metrics a few times mid-load, prints raw gauge values."""
import json, subprocess, time, urllib.request, threading

PORT = 8199
BASE = f"http://127.0.0.1:{PORT}"

def get(path, timeout=5):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return r.read().decode()

def wait_health():
    for _ in range(240):
        try:
            if get("/health", 3).strip() != "":
                return True
        except Exception:
            pass
        time.sleep(5)
    return False

def one_request(i, results):
    body = json.dumps({
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [{"role": "user", "content": f"Write a detailed paragraph {i} about speculative decoding in LLM serving."}],
        "max_tokens": 300, "stream": True,
    }).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            n = 0
            for line in r:
                if b"choices" in line:
                    n += 1
        results[i] = ("ok", round(time.time() - t0, 1), n)
    except Exception as e:
        results[i] = ("err", str(e)[:80], 0)

def read_gauges(tag):
    try:
        text = get("/metrics")
    except Exception as e:
        print(f"[{tag}] metrics fetch failed: {e}")
        return
    lines = [l for l in text.splitlines()
             if not l.startswith("#") and ("num_requests_running" in l or "num_requests_waiting" in l or "kv_cache_usage_perc" in l)]
    print(f"[{tag}] " + " | ".join(l.strip()[:70] for l in lines))

def main():
    log = open("logs/probe.server.log", "w")
    proc = subprocess.Popen(
        [".venv/bin/vllm", "serve", "Qwen/Qwen2.5-7B-Instruct",
         "--port", str(PORT), "--max-model-len", "4096",
         "--gpu-memory-utilization", "0.85"],
        stdout=log, stderr=subprocess.STDOUT)
    try:
        if not wait_health():
            print("SERVER FAILED TO START"); return
        print("server healthy; firing 8 concurrent requests...")
        results = {}
        threads = [threading.Thread(target=one_request, args=(i, results)) for i in range(8)]
        # sample gauges at staggered times while load is active
        for t, tag in [(2, "t+2s"), (6, "t+6s"), (10, "t+10s")]:
            time.sleep(t)
            read_gauges(tag)
        for th in threads: th.start()
        read_gauges("t+14s")
        for th in threads: th.join()
        print("request results:", json.dumps(results))
    finally:
        proc.terminate()
        try: proc.wait(timeout=20)
        except Exception: proc.kill()

if __name__ == "__main__":
    main()
