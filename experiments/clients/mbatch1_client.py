#!/usr/bin/env python3
"""M-BATCH-1 — open-loop MIXED-stream client (prereg docs/mbatch1-prereg.md §3).

Why not reuse phase7_mixed_client.py: S3's closed-loop B block starves B of
completion slots (all workers held on canonical prompts while A backlog drains)
— S3 raw data shows ZERO completed B-class requests in any 45 s cell, so per-
request B payoff is unmeasurable there. Open-loop Poisson arrivals with a shared
seeded stream give both arms an identical request sequence (pid = global arrival
index) -> clean pid-pairing for the payoff, and B requests complete while A
backlog drains — exactly the mixed-batch regime under test.

Stream: each request k (Poisson at LAMBDA req/s) is class A (diverse unique
prompt from BANK_A, stop-token chat format) with prob 0.5, else class B (one of
the 4 canonical high-reuse prompts round-robin over arrival index). Prefix
caching must be ON server-side for the B signal.

Per-request record: pid, cls (A{S,L} / B{SS,SL,LS,LL}), t, ttft_ms, tpot_ms,
e2e_s, ntok, isl, in_flight_at_arrival.

Usage:
  mbatch1_client.py PORT DURATION CONCURRENCY WARMUP OUT_JSONL SEED LAMBDA [MANIFEST]
"""
import json, math, random, sys, threading, time

import httpx

REPO = "/home/junior1/_dev/better-vllm-speculative-decoding"
sys.path.insert(0, f"{REPO}/experiments")
import hclass1_prompts as hc

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8402
DURATION = float(sys.argv[2]) if len(sys.argv) > 2 else 150.0
CONCURRENCY = int(sys.argv[3]) if len(sys.argv) > 3 else 96
WARMUP = float(sys.argv[4]) if len(sys.argv) > 4 else 10.0
OUT = sys.argv[5] if len(sys.argv) > 5 else "/tmp/mbatch1/client.jsonl"
SEED = int(sys.argv[6]) if len(sys.argv) > 6 else 20261013
LAMBDA = float(sys.argv[7]) if len(sys.argv) > 7 else 1.7
MANIFEST = sys.argv[8] if len(sys.argv) > 8 else None

MODEL = hc.MODEL
EOS_TOKEN_ID = hc.EOS_TOKEN_ID
CLASSES_A, ORDER_A = hc.CLASSES_A, hc.ORDER_A
CLASSES_B, ORDER_B = hc.CLASSES_B, hc.ORDER_B
CANON = hc.CANON
bank = {c: list(v) for c, v in hc.BANK_A.items()}
for c in ORDER_A:
    assert bank[c], f"diverse bank missing class {c}"

base = f"http://127.0.0.1:{PORT}/v1/completions"
lock = threading.Lock()
recs = []
in_flight = [0]
t0 = [0.0]
stop_at = [0.0]

def stream_one(client, cls, ids, pid):
    cfg = CLASSES_A[cls] if cls in CLASSES_A else CLASSES_B[cls]
    ts = time.monotonic()
    with lock:
        in_flight[0] += 1
        infl_at_arr = in_flight[0]
    ttft, ntok, text_len = None, 0, 0
    payload = {"model": MODEL, "prompt": ids, "max_tokens": cfg["maxtok"],
               "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    if cls in CLASSES_A:
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
            in_flight[0] -= 1
            recs.append({"t": time.monotonic() - t0[0], "cls": cls, "pid": pid,
                         "err": str(e)[:80], "in_flight_at_arrival": infl_at_arr})
        return
    if ntok == 0 and text_len > 0:
        ntok = max(1, round(text_len / 4))
    te = time.monotonic() - ts
    with lock:
        in_flight[0] -= 1
    if ttft is None or ntok < 2:
        with lock:
            recs.append({"t": time.monotonic() - t0[0], "cls": cls, "pid": pid,
                         "err": "no-tokens", "ntok": ntok})
        return
    tpot = (te - ttft) / (ntok - 1)
    with lock:
        recs.append({"t": time.monotonic() - t0[0], "cls": cls, "pid": pid,
                     "ttft_ms": ttft * 1e3, "tpot_ms": tpot * 1e3, "e2e_s": te,
                     "ntok": ntok, "isl": len(ids), "in_flight_at_arrival": infl_at_arr})

def worker(wid):
    client = httpx.Client(timeout=900)
    while True:
        try:
            item = q.get(timeout=1.0)
        except Exception:
            if time.monotonic() > stop_at[0] + 5 and q.empty():
                break
            continue
        cls, ids, pid = item
        stream_one(client, cls, ids, pid)

def dispatcher(q):
    """Open-loop Poisson arrivals (hclass1 cond-A construction: t += -log(1-U)/lambda),
    stop issuing after DURATION*0.95; bounded queue with counted drops (backpressure).
    Class A (diverse unique prompt) with prob 0.5 else B (canonical, round-robin over
    the global arrival index k — seeded, so both arms draw the identical sequence)."""
    rng = random.Random(SEED)
    order = {c: list(range(len(bank[c]))) for c in ORDER_A}
    for c in ORDER_A:
        rng.shuffle(order[c])
    cursor = {c: 0 for c in ORDER_A}
    k = 0
    dropped = 0
    t = 0.0
    while t < DURATION * 0.95:
        t += -math.log(max(1e-9, 1.0 - rng.random())) / LAMBDA
        if q.qsize() >= QMAX:
            dropped += 1
            k += 1
            continue
        if rng.random() < 0.5:
            cls = ORDER_A[k % len(ORDER_A)]
            ids = bank[cls][order[cls][cursor[cls] % len(bank[cls])]]
            cursor[cls] += 1
        else:
            cls = ORDER_B[k % len(ORDER_B)]
            ids = CANON[cls]
        pid = f"MIX{SEED}_{k:06d}"
        k += 1
        q.put((cls, ids, pid))
        target = t0[0] + t
        now = time.monotonic()
        if target > now:
            time.sleep(target - now)
    print(f"dispatcher: issued={k} dropped_full_queue={dropped}", flush=True)

t0[0] = time.monotonic()
stop_at[0] = t0[0] + DURATION
import queue as _queue
QMAX = 2048   # bounded arrival queue (backpressure proxy), same as hclass1_client cond A
q = _queue.Queue(maxsize=QMAX)
th_d = threading.Thread(target=dispatcher, args=(q,), daemon=True)
ths = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(CONCURRENCY)]
th_d.start()
for th in ths:
    th.start()

time.sleep(max(0.0, stop_at[0] - time.monotonic()))
TAIL_GRACE = 10.0
for _ in range(50):
    if not any(th.is_alive() for th in ths):
        break
    time.sleep(TAIL_GRACE / 50)

with open(OUT, "w") as f:
    for r in recs:
        f.write(json.dumps(r) + "\n")
if MANIFEST:
    with open(MANIFEST, "w") as f:
        for r in recs:
            if "pid" in r:
                f.write(json.dumps({"pid": r["pid"], "cls": r["cls"]}) + "\n")

ok = [r for r in recs if "ttft_ms" in r]
def pct(xs, p):
    if not xs: return float("nan")
    xs = sorted(xs); kx = max(0, min(len(xs)-1, int(round(p/100*(len(xs)-1)))))
    return xs[kx]
print(f"ok={len(ok)} err={len(recs)-len(ok)} warmup_s={WARMUP}")
BCLS = {"SS", "SL", "LS", "LL"}   # B-class names have no "B" prefix (hclass1 convention)
for grp, pred in (("A", lambda c: c not in BCLS), ("B", lambda c: c in BCLS)):
    rg = [r for r in ok if pred(r["cls"])]
    if not rg: continue
    print(f"  {grp}: n={len(rg)} ttft p50={pct([r['ttft_ms'] for r in rg],50):.0f}ms "
          f"tpot p50={pct([r['tpot_ms'] for r in rg],50):.1f}ms ntok p50={pct([r['ntok'] for r in rg],50):.0f}")
af = [r["in_flight_at_arrival"] for r in ok]
print(f"  in-flight at arrival: mean={sum(af)/max(1,len(af)):.1f} p90={pct(af,90):.0f} max={max(af) if af else 0}")
