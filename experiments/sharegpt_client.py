#!/usr/bin/env python3
"""Real-ShareGPT Poisson client for Phase-2 (genuine per-request acceptance diversity).

Samples diverse human turns from ShareGPT and sends them at a fixed Poisson rate.
Unlike synthetic random-token prompts, real conversational text produces genuinely
request-dependent ngram self-draft acceptance -> measurable per-request K mismatch.

Usage:
  .venv/bin/python experiments/sharegpt_client.py --port 8100 --num-prompts 200 \
    --rate 6 --max-tokens 256 --out results/P2s_K4_r6/client.jsonl
"""
import argparse, json, os, random, threading, time
from concurrent.futures import ThreadPoolExecutor

import httpx


def load_prompts(path, n, rng):
    data = json.load(open(path))
    cands = []
    for conv in data:
        msgs = conv.get("conversations", [])
        if len(msgs) >= 2 and msgs[0].get("from") == "human":
            txt = msgs[0]["value"].strip()
            if 80 <= len(txt) <= 900:
                cands.append(txt)
    rng.shuffle(cands)
    return [cands[i % len(cands)] for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--num-prompts", type=int, default=200)
    ap.add_argument("--rate", type=float, default=6.0)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--sharegpt", default="data/ShareGPT_V3_unfiltered_cleaned_split.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    prompts = load_prompts(a.sharegpt, a.num_prompts, rng)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    logf = open(a.out, "w")
    lock = threading.Lock()
    base = f"http://{a.host}:{a.port}/v1/completions"

    def send(idx, prompt):
        t0 = time.monotonic()
        rec = {"idx": idx, "t0": round(t0, 3), "prompt_len": len(prompt)}
        try:
            with httpx.Client(timeout=600) as c:
                r = c.post(base, json={"model": a.model, "prompt": prompt,
                                       "max_tokens": a.max_tokens, "temperature": 0})
                rec["status"] = r.status_code
                if r.status_code == 200:
                    j = r.json()
                    rec["id"] = j.get("id")
                    rec["out_tokens"] = j.get("usage", {}).get("completion_tokens")
                else:
                    rec["error"] = r.text[:200]
        except Exception as e:
            rec["error"] = str(e)[:200]
        rec["t1"] = round(time.monotonic(), 3)
        with lock:
            logf.write(json.dumps(rec) + "\n"); logf.flush()

    t_start = time.monotonic()
    with ThreadPoolExecutor(max_workers=64) as ex:
        for idx, p in enumerate(prompts):
            ex.submit(send, idx, p)
            if idx < a.num_prompts - 1:
                time.sleep(rng.expovariate(a.rate))

    elapsed = time.monotonic() - t_start
    logf.close()
    print(f"[sharegpt_client] {a.num_prompts} prompts @ rate={a.rate} req/s "
          f"in {elapsed:.1f}s -> {a.out}")


if __name__ == "__main__":
    main()
