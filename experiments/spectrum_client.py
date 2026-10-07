#!/usr/bin/env python3
"""Spectrum workload client for Phase-2 oracle-gap analysis.

Generates requests whose ngram self-draft acceptance spans a controlled range by
varying the fraction of output tokens drawn from a small repeating "core" pool
(repetition rate p). Low p -> diverse text -> low draft acceptance (short K*);
high p -> repetitive text -> high acceptance (long K*). All requests draft
frequently enough to be measurable, so per-request J profiles are well-defined.

Each request's completion id + its designed repetition rate are logged for joining
with the per-request spec-decode hook (experiments/spec_hook/).

Usage:
  .venv/bin/python experiments/spectrum_client.py --port 8100 --num-prompts 150 \
    --rate 6 --max-tokens 256 --out results/P2_K8_r6/client.jsonl [--seed 0]
"""
import argparse, json, os, random, threading, time
from concurrent.futures import ThreadPoolExecutor

import httpx


def make_prompt(tokenizer, p, core_len=40, tail_len=120, seed=0):
    """Prompt = a short repetitive 'core' (drives high n-gram match) + a diverse tail.

    Repetition rate p controls how much of the *generation* will be self-draftable:
    we instruct the model to continue the core pattern; with temperature 0 and a
    strongly periodic prefix, the model tends to keep emitting core-like tokens.
    """
    rng = random.Random(seed)
    vocab = tokenizer.vocab_size
    # small core pool -> highly repeatable n-grams
    core_pool = rng.sample(range(vocab), 24)
    # diverse tail pool
    tail_pool = rng.sample(range(vocab), 512)
    # build a periodic core sequence (repeats of the 24-token motif)
    motif = core_pool[:12]
    core_ids = motif * (core_len // len(motif))
    tail_ids = [rng.choice(tail_pool) for _ in range(tail_len)]
    # mix: p fraction of the *leading* tokens are core-like, rest diverse
    n_core = int((core_len + tail_len) * p)
    ids = (core_ids * 3)[:n_core] + tail_ids[: max(0, (core_len + tail_len) - n_core)]
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--num-prompts", type=int, default=150)
    ap.add_argument("--rate", type=float, default=6.0)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", required=True)
    # repetition-rate grid: low..high acceptance
    ap.add_argument("--p-values", default="0.30,0.45,0.60,0.75,0.90")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    pvals = [float(x) for x in a.p_values.split(",")]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    logf = open(a.out, "w")
    lock = threading.Lock()
    base = f"http://{a.host}:{a.port}/v1/completions"

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)

    # build request list: cycle through p-values so each acceptance level is represented
    reqs = []
    for i in range(a.num_prompts):
        p = pvals[i % len(pvals)]
        prompt = make_prompt(tok, p, seed=a.seed + i)
        reqs.append({"idx": i, "p": p, "prompt": prompt})

    def send(rec):
        t0 = time.monotonic()
        out = {"idx": rec["idx"], "p": rec["p"], "t0": round(t0, 3),
               "prompt_len": len(rec["prompt"])}
        try:
            with httpx.Client(timeout=600) as c:
                r = c.post(base, json={"model": a.model, "prompt": rec["prompt"],
                                       "max_tokens": a.max_tokens, "temperature": 0})
                out["status"] = r.status_code
                if r.status_code == 200:
                    j = r.json()
                    out["id"] = j.get("id")
                    out["out_tokens"] = j.get("usage", {}).get("completion_tokens")
                else:
                    out["error"] = r.text[:200]
        except Exception as e:
            out["error"] = str(e)[:200]
        out["t1"] = round(time.monotonic(), 3)
        with lock:
            logf.write(json.dumps(out) + "\n"); logf.flush()

    t_start = time.monotonic()
    with ThreadPoolExecutor(max_workers=64) as ex:
        for rec in reqs:
            ex.submit(send, rec)
            if rec["idx"] < a.num_prompts - 1:
                time.sleep(rng.expovariate(a.rate))

    elapsed = time.monotonic() - t_start
    logf.close()
    print(f"[spectrum_client] {a.num_prompts} prompts (p in {pvals}) "
          f"@ rate={a.rate} req/s in {elapsed:.1f}s -> {a.out}")


if __name__ == "__main__":
    main()
