#!/usr/bin/env python3
"""Phase 7 capstone — MIXED-stream client (prereg docs/phase7-capstone-prereg.md §5).

The existing hclass1_client.py only drives a SINGLE condition (A diverse, or B
high-reuse) per run. The live controller, however, sees a *single mixed stream* where
diverse and high-reuse requests interleave in time. This driver reproduces exactly that:
it alternates fixed-length BLOCKS — A blocks (open-loop Poisson over the diverse prompt
bank, unique prompts -> low acceptance) and B blocks (workers round-robin the canonical
high-reuse prompts back-to-back -> high prefix-cache hit + high acceptance) — for a
fixed total duration.

Reuses hclass1_client's exact prompt construction (imports it) so the A diverse prompts
and B canonical prompts are byte-identical to H-CLASS-1 / H-OBS-3. Prefix caching must be
ON server-side for the B high-reuse signal to manifest.

Usage:
  phase7_mixed_client.py PORT DURATION CONCURRENCY BLOCK_S OUT_JSONL SEED [PHASE]
    PHASE in {A, MIX} (default MIX). A = diverse blocks only (high-load class control).
    MIX = alternating A/B blocks.
Per-request record: pid, cls (A<cls> / B<cls>), block, t, ttft_ms, tpot_ms, e2e_s, ntok, isl.
"""
import json, math, random, sys, threading, time
import queue as _queue

import httpx

REPO = "/home/junior1/_dev/better-vllm-speculative-decoding"
sys.path.insert(0, f"{REPO}/experiments")
import hclass1_prompts as hc   # shared prompt construction (no side effects)

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8402
DURATION = float(sys.argv[2]) if len(sys.argv) > 2 else 50.0
CONCURRENCY = int(sys.argv[3]) if len(sys.argv) > 3 else 96
BLOCK_S = float(sys.argv[4]) if len(sys.argv) > 4 else 8.0
OUT = sys.argv[5] if len(sys.argv) > 5 else "/tmp/phase7_mixed/client.jsonl"
SEED = int(sys.argv[6]) if len(sys.argv) > 6 else 20261013
PHASE = sys.argv[7] if len(sys.argv) > 7 else "MIX"

MODEL = hc.MODEL
EOS_TOKEN_ID = hc.EOS_TOKEN_ID
CLASSES_A = hc.CLASSES_A          # {S:{isl,maxtok}, L:{...}}
ORDER_A = hc.ORDER_A              # [S, L]
CLASSES_B = hc.CLASSES_B          # {SS,SL,LS,LL}
ORDER_B = hc.ORDER_B
CANON = hc.CANON                  # canonical high-reuse prompts (H-OBS-3 exact)

# Diverse A bank: pre-tokenized chat-templated unique prompts (same as hclass1_client cond A).
bank = {c: list(v) for c, v in hc.BANK_A.items()}
for c in ORDER_A:
    assert bank[c], f"diverse bank missing class {c}"

base = f"http://127.0.0.1:{PORT}/v1/completions"
lock = threading.Lock()
recs = []
t0 = [0.0]
stop_at = [0.0]

def stream_one(client, cls, ids, pid, use_stop):
    cfg = CLASSES_A[cls] if cls in CLASSES_A else CLASSES_B[cls]
    ts = time.monotonic()
    ttft, ntok, text_len = None, 0, 0
    payload = {"model": MODEL, "prompt": ids, "max_tokens": cfg["maxtok"],
               "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    if use_stop:
        payload["stop_token_ids"] = [EOS_TOKEN_ID]
    try:
        with client.stream("POST", base, json=payload) as r:
            for line in r.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                p = line[5:].strip()
                if p == "[DONE]:":
                    break
                if p == "[DONE]":
                    break
                try:
                    d = json.loads(p)
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
            recs.append({"t": time.monotonic() - t0[0], "cls": cls, "pid": pid,
                         "err": str(e)[:80]})
        return
    if ntok == 0 and text_len > 0:
        ntok = max(1, round(text_len / 4))
    te = time.monotonic() - ts
    if ttft is None or ntok < 2:
        with lock:
            recs.append({"t": time.monotonic() - t0[0], "cls": cls, "pid": pid,
                         "err": "no-tokens", "ntok": ntok})
        return
    tpot = (te - ttft) / (ntok - 1)
    with lock:
        recs.append({"t": time.monotonic() - t0[0], "cls": cls, "pid": pid,
                     "ttft_ms": ttft * 1e3, "tpot_ms": tpot * 1e3, "e2e_s": te,
                     "ntok": ntok, "isl": len(ids)})

def worker_a(wid, q):
    client = httpx.Client(timeout=900)
    while time.monotonic() < stop_at[0] + 5:
        try:
            item = q.get(timeout=1.0)
        except _queue.Empty:
            continue
        cls, ids, pid = item
        stream_one(client, cls, ids, pid, use_stop=True)

def worker_b(wid):
    """High-reuse B block worker: round-robin the 4 canonical class slots back-to-back
    (H-OBS-3 pattern). Active only during B blocks (controlled by a shared flag)."""
    client = httpx.Client(timeout=900)
    step = 0
    while time.monotonic() < stop_at[0] + 5:
        if not in_b_block[0]:
            time.sleep(0.05)
            continue
        cls = ORDER_B[(wid + step) % len(ORDER_B)]
        stream_one(client, cls, CANON[cls], f"B{SEED}_{cls}_w{wid}s{step}", use_stop=False)
        step += 1

# Shared block-phase flag: workers B act only while True.
in_b_block = [False]

def dispatcher(q):
    """Drive the A/B block schedule on a wall clock. Even blocks = A (diverse): keep the
    queue backlogged with unique diverse prompts so all workers stay busy -> high load,
    low acceptance. Odd blocks = B (high-reuse, only when PHASE==MIX): set in_b_block so
    the B workers hammer canonical prompts back-to-back -> high load, high acceptance."""
    rng = random.Random(SEED)
    order = {c: list(range(len(bank[c]))) for c in ORDER_A}
    for c in ORDER_A:
        rng.shuffle(order[c])
    cursor = {c: 0 for c in ORDER_A}
    k = 0
    block_idx = 0
    BACKLOG = CONCURRENCY * 4   # keep this many diverse prompts queued during A blocks
    while time.monotonic() < stop_at[0] - BLOCK_S * 0.5:
        is_b = (PHASE == "MIX") and (block_idx % 2 == 1)
        block_end = time.monotonic() + BLOCK_S
        in_b_block[0] = is_b
        if not is_b:
            # A block: keep the queue backlogged with unique diverse prompts.
            while time.monotonic() < block_end:
                if q.qsize() >= BACKLOG:
                    time.sleep(0.02)
                    continue
                cls = ORDER_A[k % len(ORDER_A)]
                ids = bank[cls][order[cls][cursor[cls] % len(bank[cls])]]
                cursor[cls] += 1
                q.put((cls, ids, f"A{SEED}_{k:06d}"))
                k += 1
        else:
            # B block: hold while B workers generate high-reuse traffic back-to-back.
            while time.monotonic() < block_end:
                time.sleep(0.05)
        in_b_block[0] = False
        block_idx += 1
    print(f"dispatcher: ~{k} diverse prompts issued, blocks={block_idx}", flush=True)

t0[0] = time.monotonic()
stop_at[0] = t0[0] + DURATION
q = _queue.Queue(maxsize=4096)
th_d = threading.Thread(target=dispatcher, args=(q,), daemon=True)
ths = [threading.Thread(target=worker_a, args=(i, q)) for i in range(CONCURRENCY)]
if PHASE == "MIX":
    ths += [threading.Thread(target=worker_b, args=(i,)) for i in range(CONCURRENCY)]
th_d.start()
for th in ths:
    th.start()

time.sleep(max(0.0, stop_at[0] - time.monotonic()))
in_b_block[0] = False
for th in ths:
    th.join(timeout=120)

with open(OUT, "w") as f:
    for r in recs:
        f.write(json.dumps(r) + "\n")

ok = [r for r in recs if "ttft_ms" in r]
def pct(xs, p):
    if not xs: return float("nan")
    xs = sorted(xs); k = max(0, min(len(xs)-1, int(round(p/100*(len(xs)-1)))))
    return xs[k]
print(f"phase={PHASE} concurrency={CONCURRENCY} ok={len(ok)} err={len(recs)-len(ok)}")
for grp in ("A", "B"):
    rg = [r for r in ok if r["cls"].startswith(grp)]
    if not rg: continue
    print(f"  {grp}: n={len(rg)} ttft p50={pct([r['ttft_ms'] for r in rg],50):.0f}ms "
          f"tpot p50={pct([r['tpot_ms'] for r in rg],50):.1f}ms ntok p50={pct([r['ntok'] for r in rg],50):.0f}")
