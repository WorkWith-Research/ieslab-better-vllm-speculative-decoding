#!/usr/bin/env python3
"""Track-A label collection (prereg docs/tracka-prereg.md §1).

Per-request DSpark acceptance labels at strict concurrency=1, obtained as deltas of the
engine's cumulative Prometheus counters (vllm:spec_decode_num_draft_tokens /
_num_accepted_tokens; vllm:prefix_cache_queries/_hits) queried before/after each request.
At concurrency=1 with no overlapping requests the delta equals the request's own counts
exactly — NO worktree instrumentation required.

Admission-time features are computed here from the prompt token ids (+ prefill logprobs):
  isl, unigram_entropy (bits/token), bigram_rep, trigram_rep (repeat fractions),
  prompt_ppl = exp(-mean prompt logprob) via prompt_logprobs=1 (free: prefill runs anyway).

Usage: tracka_collect.py <port> <job.json>
  job.json = [ {"bank": "<bank.jsonl>", "indices": [i, ...], "out": "<labels.jsonl>"}, ... ]
Each job entry is processed sequentially against the server at <port>. Idempotent per
record: if <out> already contains a record for (bank basename, bank_idx) it is skipped.
"""
import json, math, os, random, sys, time
from collections import Counter
import httpx

PORT = int(sys.argv[1])
JOBS = json.load(open(sys.argv[2]))
BASE = f"http://localhost:{PORT}"
MODEL = "qwen3-4b"
STOP = [151645]          # < |im_end|>
MAXTOK = 2048
COUNTERS = ("vllm:spec_decode_num_draft_tokens_total", "vllm:spec_decode_num_accepted_tokens_total",
            "vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total")

def get_counters():
    r = httpx.get(BASE + "/metrics", timeout=15)
    out = {}
    for line in r.text.splitlines():
        if line.startswith("#"):
            continue
        name = line.split("{")[0]
        if name in COUNTERS:
            try:
                out[name] = float(line.rsplit(" ", 1)[1])
            except (ValueError, IndexError):
                pass
    return out

def features(ids):
    n = len(ids)
    c = Counter(ids)
    H = -sum((v / n) * math.log2(v / n) for v in c.values())
    bg = Counter(zip(ids[:-1], ids[1:])) if n > 1 else {}
    bigram_rep = sum(v - 1 for v in bg.values() if v > 1) / max(1, n - 1)
    tg = Counter(zip(ids[:-2], ids[1:-1], ids[2:])) if n > 2 else {}
    trigram_rep = sum(v - 1 for v in tg.values() if v > 1) / max(1, n - 2)
    return H, bigram_rep, trigram_rep

def prompt_ppl(ch, ids):
    """exp(-mean logprob of the realized prompt tokens), from prompt_logprobs.
    vLLM keys each position's dict by STRING token id; pl[k] corresponds to ids[k]."""
    pl = ch.get("prompt_logprobs") or []
    lps = []
    for k in range(1, min(len(pl), len(ids))):
        entry = pl[k]
        if not isinstance(entry, dict):
            continue
        lp = entry.get(str(ids[k]))
        if isinstance(lp, dict):
            lp = lp.get("logprob")
        if lp is not None and math.isfinite(lp):
            lps.append(lp)
    if len(lps) < 10:
        return None
    return round(math.exp(-sum(lps) / len(lps)), 4)

client = httpx.Client(timeout=900)
for job in JOBS:
    bank_path = job["bank"]
    bank = [json.loads(l) for l in open(bank_path)]
    bname = os.path.basename(bank_path)
    out_path = job["out"]
    # resume: skip (bank, idx) already recorded
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            try:
                r = json.loads(l)
                if r.get("bank") == bname and "err" not in r:
                    done.add(r["bank_idx"])
            except json.JSONDecodeError:
                pass
    todo = [i for i in job["indices"] if i not in done]
    print(f"[{bname}] total={len(job['indices'])} done={len(done)} todo={len(todo)}", flush=True)
    n_ok = 0
    with open(out_path, "a") as f:
        for k, i in enumerate(todo):
            e = bank[i]
            ids = e["ids"]
            c0 = get_counters()
            t0 = time.monotonic()
            try:
                r = client.post(BASE + "/v1/completions", json={
                    "model": MODEL, "prompt": ids, "max_tokens": MAXTOK, "temperature": 0,
                    "stop_token_ids": STOP, "prompt_logprobs": 1})
            except Exception as ex:
                f.write(json.dumps({"bank": bname, "bank_idx": i, "cls": e["cls"],
                                    "err": f"request: {str(ex)[:120]}"}) + "\n")
                f.flush()
                continue
            dt = time.monotonic() - t0
            if r.status_code != 200:
                f.write(json.dumps({"bank": bname, "bank_idx": i, "cls": e["cls"],
                                    "err": f"http {r.status_code}: {r.text[:160]}"}) + "\n")
                f.flush()
                continue
            d = r.json()
            ch = (d.get("choices") or [{}])[0]
            ntok = (d.get("usage") or {}).get("completion_tokens") or 0
            # small settle so the engine's last-step counters are visible, then delta
            time.sleep(0.5)
            c1 = get_counters()
            if c1.get("vllm:spec_decode_num_draft_tokens_total", 0) == c0.get("vllm:spec_decode_num_draft_tokens_total", 0):
                time.sleep(1.0)
                c1 = get_counters()
            draft = max(0.0, c1.get("vllm:spec_decode_num_draft_tokens_total", 0) - c0.get("vllm:spec_decode_num_draft_tokens_total", 0))
            accept = max(0.0, c1.get("vllm:spec_decode_num_accepted_tokens_total", 0) - c0.get("vllm:spec_decode_num_accepted_tokens_total", 0))
            q = max(0.0, c1.get("vllm:prefix_cache_queries_total", 0) - c0.get("vllm:prefix_cache_queries_total", 0))
            h = max(0.0, c1.get("vllm:prefix_cache_hits_total", 0) - c0.get("vllm:prefix_cache_hits_total", 0))
            H, bgr, tgr = features(ids)
            rec = {
                "bank": bname, "bank_idx": i, "cls": e["cls"], "isl": len(ids), "ntok": ntok,
                "draft": draft, "accept": accept,
                "acc_rate": round(accept / draft, 5) if draft > 0 else None,
                "cache_hit_rate": round(h / q, 5) if q > 0 else None,
                "unigram_entropy": round(H, 4), "bigram_rep": round(bgr, 5), "trigram_rep": round(tgr, 5),
                "prompt_ppl": prompt_ppl(ch, ids),
                "wall_s": round(dt, 1),
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()
            n_ok += 1
            if (k + 1) % 25 == 0:
                print(f"  [{bname}] {k+1}/{len(todo)} ok acc={rec['acc_rate']} hit={rec['cache_hit_rate']} "
                      f"ntok={ntok} ppl={rec['prompt_ppl']} ({dt:.0f}s)", flush=True)
    print(f"[{bname}] DONE ok={n_ok}", flush=True)
