#!/usr/bin/env python3
"""Mixed-workload Poisson client for Phase-2 oracle-gap analysis.

Sends a stream of requests at a fixed Poisson rate, mixing two prompt classes:
  - "easy": repetitive small-vocabulary text -> high ngram self-draft acceptance
  - "hard": diverse ShareGPT conversational text -> low ngram acceptance

Each request's completion id, class, timing and output length are logged to a
JSONL file so it can be joined with the per-request spec-decode hook output
(vllm scheduler make_spec_decoding_stats, see experiments/spec_hook/).

Usage:
  .venv/bin/python experiments/mixed_client.py \
    --port 8100 --num-prompts 150 --rate 6 --easy-fraction 0.5 \
    --max-tokens 256 --out results/P2_K4/client.jsonl \
    [--sharegpt data/ShareGPT_V3_unfiltered_cleaned_split.json] [--seed 0]
"""
import argparse, json, os, random, threading, time
from concurrent.futures import ThreadPoolExecutor

import httpx


def load_hard_prompts(path, n, rng):
    """Sample diverse human turns from ShareGPT as 'hard' (low-acceptance) prompts."""
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


def make_easy_prompts(tokenizer, n, vocab_size=200, seq_len=300, seed=0):
    """Repetitive small-vocabulary prompts -> high ngram self-draft acceptance.

    Mimics vllm bench serve's random dataset: sample token ids from a small pool
    so n-grams repeat and the ngram proposer finds context matches.
    """
    all_ids = list(range(tokenizer.vocab_size))
    rng = random.Random(seed)
    pool = rng.sample(all_ids, vocab_size)
    out = []
    for _ in range(n):
        ids = [rng.choice(pool) for _ in range(seq_len)]
        out.append(tokenizer.decode(ids, skip_special_tokens=True).strip())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--num-prompts", type=int, default=150)
    ap.add_argument("--rate", type=float, default=6.0)
    ap.add_argument("--easy-fraction", type=float, default=0.5)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sharegpt", default="data/ShareGPT_V3_unfiltered_cleaned_split.json")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    n_easy = int(round(a.num_prompts * a.easy_fraction))
    n_hard = a.num_prompts - n_easy

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    easy = make_easy_prompts(tok, n_easy, seed=a.seed)
    hard = load_hard_prompts(a.sharegpt, n_hard, rng) if os.path.exists(a.sharegpt) else \
           ["Tell me about " + w for w in ["science", "history", "music"] * (n_hard // 3 + 1)]

    # interleave classes deterministically by seed
    order = [("easy", p) for p in easy] + [("hard", p) for p in hard]
    rng.shuffle(order)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    logf = open(a.out, "w")
    lock = threading.Lock()
    base = f"http://{a.host}:{a.port}/v1/completions"

    def send(idx, cls, prompt):
        t0 = time.monotonic()
        rec = {"idx": idx, "class": cls, "t0": round(t0, 3), "prompt_len": len(prompt)}
        try:
            with httpx.Client(timeout=600) as c:
                r = c.post(base, json={"model": a.model, "prompt": prompt,
                                       "max_tokens": a.max_tokens, "temperature": 0})
                rec["status"] = r.status_code
                if r.status_code == 200:
                    j = r.json()
                    rec["id"] = j.get("id")
                    u = j.get("usage", {})
                    rec["out_tokens"] = u.get("completion_tokens")
                else:
                    rec["error"] = r.text[:200]
        except Exception as e:
            rec["error"] = str(e)[:200]
        rec["t1"] = round(time.monotonic(), 3)
        with lock:
            logf.write(json.dumps(rec) + "\n"); logf.flush()

    t_start = time.monotonic()
    with ThreadPoolExecutor(max_workers=64) as ex:
        for idx, (cls, prompt) in enumerate(order):
            ex.submit(send, idx, cls, prompt)
            if idx < a.num_prompts - 1:
                time.sleep(rng.expovariate(a.rate))  # Poisson inter-arrival

    elapsed = time.monotonic() - t_start
    logf.close()
    print(f"[mixed_client] {a.num_prompts} prompts ({n_easy} easy / {n_hard} hard) "
          f"@ rate={a.rate} req/s in {elapsed:.1f}s -> {a.out}")


if __name__ == "__main__":
    main()
