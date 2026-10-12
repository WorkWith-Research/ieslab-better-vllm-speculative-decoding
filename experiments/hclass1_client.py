#!/usr/bin/env python3
"""H-CLASS-1 heterogeneous client (prereg docs/hclass1-prereg.md §4).

Two conditions:
  A (primary): DIVERSE distinct real ShareGPT-V3 turns, one prompt per request
               (no in-run reuse), Poisson arrivals at rate LAMBDA req/s.
  B (control): H-OBS-3's canonical 4 prompts verbatim (same seed 20261011, same
               turn-concatenation construction -> byte-identical prompts),
               back-to-back per worker thread (H-OBS-3 arrival pattern).

Classes (ISL exact by token count; max_tokens is a CAP, realized OSL measured):
  SS: ISL 50,    maxtok 256   SL: ISL 50,    maxtok 1024
  LS: ISL 2048,  maxtok 256   LL: ISL 2048,  maxtok 1024

Prompts tokenized EXTERNALLY with the served model's tokenizer
(add_special_tokens=False), sent as integer-id arrays. temperature=0,
streaming, NATURAL EOS (no ignore-eos), stream_options.include_usage for
authoritative completion_tokens.

Per-request record: pid, cls, t (arrival offset s), ttft_ms, tpot_ms, e2e_s,
ntok (realized OSL), isl.

Usage:
  hclass1_client.py PORT DURATION CONCURRENCY WARMUP OUT_JSONL SEED COND LAMBDA [MANIFEST_JSONL]
    COND in {A,B}; LAMBDA = Poisson rate req/s (COND A only; B ignores it).
"""
import json, math, random, sys, threading, time
from collections import defaultdict

import httpx
from transformers import AutoTokenizer

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8402
DURATION = float(sys.argv[2]) if len(sys.argv) > 2 else 150.0
CONCURRENCY = int(sys.argv[3]) if len(sys.argv) > 3 else 32
WARMUP = float(sys.argv[4]) if len(sys.argv) > 4 else 20.0
OUT = sys.argv[5] if len(sys.argv) > 5 else "/tmp/hclass1.jsonl"
SEED = int(sys.argv[6]) if len(sys.argv) > 6 else 20261013
COND = sys.argv[7] if len(sys.argv) > 7 else "A"
LAMBDA = float(sys.argv[8]) if len(sys.argv) > 8 else 4.0
MANIFEST = sys.argv[9] if len(sys.argv) > 9 else None

MODEL = "qwen3-4b"
TOK_MODEL = "Qwen/Qwen3-4B"
CORPUS = "/home/junior1/_dev/better-vllm-speculative-decoding/data/ShareGPT_V3_unfiltered_cleaned_split.json"
CLASSES = {
    "SS": {"isl": 50, "maxtok": 256},
    "SL": {"isl": 50, "maxtok": 1024},
    "LS": {"isl": 2048, "maxtok": 256},
    "LL": {"isl": 2048, "maxtok": 1024},
}
ORDER = ["SS", "SL", "LS", "LL"]

TOK = AutoTokenizer.from_pretrained(TOK_MODEL, trust_remote_code=True)

def load_turns(path):
    data = json.load(open(path))
    turns = []
    for conv in data:
        msgs = conv.get("conversations") or []
        for m in msgs:
            if m.get("from") == "human":
                v = (m.get("value") or "").strip()
                if len(v) > 20:
                    turns.append(v)
    return turns

TURNS = load_turns(CORPUS)
print(f"loaded {len(TURNS)} user turns", flush=True)

def tok_len(s):
    return len(TOK.encode(s, add_special_tokens=False))

# ---- pre-tokenize pools by char-length buckets (fast filter), exact-truncate at draw time ----
SHORT_CANDS = [t for t in TURNS if 150 <= len(t) < 4000]      # brackets ~50 tok
LONG_CANDS = [t for t in TURNS if len(t) >= 6000]             # brackets ~2048 tok
print(f"pools: short={len(SHORT_CANDS)} long={len(LONG_CANDS)}", flush=True)

# ---- chat-template construction (condition A only; deviation D1, prereg s4) ----
# Raw-completion smoke showed frac_at_max ~1.0 for all classes: Qwen3-4B on a bare
# token continuation never emits EOS (no conversational end signal), so realized OSL
# == max_tokens and the "long-output class" axis degenerates to the configured cap.
# Condition A therefore serves prompts through the model's chat template (user turn +
# assistant header) — the realistic serving format for an instruct model — with
# stop_token_ids=[151645] (< |im_end|> = Qwen3 eos_token, vLLM drops it from output),
# so the model answers and can terminate naturally. Realized ISL is recorded per
# request; target ISL met within a bracket window (content truncation never cuts the
# template suffix). Condition B stays raw completions (H-OBS-3 parity).
def _templated_ids(turn):
    return TOK.encode(TOK.apply_chat_template(
        [{"role": "user", "content": turn}], add_generation_prompt=True, tokenize=False),
        add_special_tokens=False)

EOS_TOKEN_ID = 151645   # < |im_end|> == Qwen3-4B eos_token (verified via tokenizer)

def make_diverse(isl_target, rng):
    """Pick one real turn whose templated total brackets isl_target; return ids.
    Truncation (fallback only) shaves content tokens, never the template suffix.
    Bracket window: [target-8, target+16] so SS/SL stay genuinely distinct (a 34-token
    prompt would otherwise be accepted for a 50-target and collapse the two classes)."""
    pool = SHORT_CANDS if isl_target <= 64 else LONG_CANDS
    lo_b, hi_b = isl_target - 8, isl_target + 16
    for _ in range(64):
        t = pool[rng.randrange(len(pool))]
        ids = _templated_ids(t)
        if lo_b <= len(ids) <= hi_b:
            return ids
    # fallback: truncate the drawn turn's content so the templated total fits exactly
    t = pool[rng.randrange(len(pool))]
    lo, hi = 16, len(t)
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        ids = _templated_ids(t[:mid])
        if len(ids) <= isl_target:
            best, lo = ids, mid + 1
        else:
            hi = mid - 1
    return best if best else _templated_ids(t[:64])

# ---- condition B: reproduce H-OBS-3 canonical prompts EXACTLY (seed 20261011) ----
def make_canonical(isl_target):
    """H-OBS-3 construction: shuffled TURNS (seed 20261011), start at randrange,
    concatenate sequential turns until >= target, truncate to target."""
    rng = random.Random(20261011)
    tcopy = list(TURNS)
    rng.shuffle(tcopy)
    ids = []
    used = set()
    i = rng.randrange(len(tcopy))
    while len(ids) < isl_target and len(used) < 64:
        j = (i + len(used)) % len(tcopy)
        if j in used:
            continue
        used.add(j)
        ids.extend(TOK.encode(tcopy[j], add_special_tokens=False))
    return ids[:isl_target]

CANON = {c: make_canonical(cfg["isl"]) for c, cfg in CLASSES.items()}
print("canonical (cond B) prompt lens:", {c: len(v) for c, v in CANON.items()}, flush=True)

# ---- build the request stream ----------------------------------------------
rng = random.Random(SEED)
requests = []   # list of (cls, ids, pid)
if COND == "A":
    n_est = int(LAMBDA * DURATION * 1.6) + CONCURRENCY * 4
    used_pids = set()
    for k in range(n_est):
        cls = ORDER[k % 4]
        cfg = CLASSES[cls]
        ids = make_diverse(cfg["isl"], rng)
        pid = f"A{SEED}_{k:06d}"
        requests.append((cls, ids, pid))
elif COND == "B":
    # H-OBS-3 pattern: each worker round-robins the 4 canonical prompts back-to-back.
    # Represent as an infinite logical stream; workers draw by global step index.
    pass
else:
    raise SystemExit(f"unknown COND {COND}")

base = f"http://127.0.0.1:{PORT}/v1/completions"
lock = threading.Lock()
recs = []
stop_at = [0.0]
t0 = [0.0]

def stream_one(client, cls, ids, pid):
    cfg = CLASSES[cls]
    ts = time.monotonic()
    ttft = None
    ntok = 0
    text_len = 0
    payload = {
        "model": MODEL, "prompt": ids,
        "max_tokens": cfg["maxtok"], "temperature": 0, "stream": True,
        "stream_options": {"include_usage": True}}
    if COND == "A":
        # chat-templated prompts: stop at the assistant-turn terminator so a natural
        # answer end is honored (vLLM drops the stop token from output).
        payload["stop_token_ids"] = [EOS_TOKEN_ID]
    try:
        with client.stream("POST", base, json=payload) as r:
            for line in r.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    d = json.loads(payload)
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
    while time.monotonic() < stop_at[0]:
        try:
            item = q.get(timeout=1.0)
        except _queue.Empty:
            continue
        cls, ids, pid = item
        stream_one(client, cls, ids, pid)

def worker_b(wid):
    client = httpx.Client(timeout=900)
    step = 0
    while time.monotonic() < stop_at[0]:
        cls = ORDER[(wid + step) % 4]   # H-OBS-3 round-robin, back-to-back
        stream_one(client, cls, CANON[cls], f"B{SEED}_{cls}_w{wid}s{step}")
        step += 1

def dispatcher(q):
    """Poisson arrivals: interarrival ~ Exp(lambda); stop issuing after DURATION*0.95.
    Bounded queue (backpressure proxy): when qsize >= QMAX the arrival is DROPPED and
    counted — a real client would see 429/timeout; deterministic under fixed seed."""
    t = 0.0
    k = 0
    dropped = 0
    while t < DURATION * 0.95 and k < len(requests):
        t += -math.log(max(1e-9, 1.0 - rng.random())) / LAMBDA
        if q.qsize() >= QMAX:
            dropped += 1
            k += 1
            continue
        q.put(requests[k])
        k += 1
        target = t0[0] + t
        now = time.monotonic()
        if target > now:
            time.sleep(target - now)
    print(f"dispatcher: issued={k} dropped_full_queue={dropped}", flush=True)

import queue as _queue

QMAX = 2048   # bounded arrival queue (backpressure proxy); drops counted by dispatcher

t0[0] = time.monotonic()
stop_at[0] = t0[0] + DURATION

if COND == "A":
    q = _queue.Queue(maxsize=QMAX)
    th_d = threading.Thread(target=dispatcher, args=(q,), daemon=True)
    ths = [threading.Thread(target=worker_a, args=(i, q)) for i in range(CONCURRENCY)]
    th_d.start()
    for th in ths:
        th.start()
else:
    ths = [threading.Thread(target=worker_b, args=(i,)) for i in range(CONCURRENCY)]
    for th in ths:
        th.start()

# wait until stop time, then join workers so in-flight requests finish and are recorded
time.sleep(max(0.0, stop_at[0] - time.monotonic()))
for th in ths:
    th.join(timeout=120)

with open(OUT, "w") as f:
    for r in recs:
        f.write(json.dumps(r) + "\n")

if MANIFEST and COND == "A":
    with open(MANIFEST, "w") as f:
        seen = {}
        for cls, ids, pid in requests:
            if pid not in seen:
                seen[pid] = (cls, list(ids))
        for pid, (cls, ids) in seen.items():
            f.write(json.dumps({"pid": pid, "cls": cls, "ids": ids}) + "\n")

ok = [r for r in recs if "ttft_ms" in r]
st = [r for r in ok if WARMUP <= r["t"] <= DURATION]
window = max(1e-9, min(DURATION, time.monotonic() - t0[0]) - WARMUP)
tok = sum(r["ntok"] for r in st)
print(f"cond={COND} concurrency={CONCURRENCY} ok={len(ok)} err={len(recs)-len(ok)} "
      f"steady_out_tokens={tok} steady_throughput_tok_s={tok/window:.1f}")

def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1)))))
    return xs[k]

print(f"{'class':>5} {'n':>5} {'tok/s':>8} {'TTFT p50/p95 ms':>16} {'TPOT p50/p95 ms':>16} "
      f"{'ntok p50':>9} {'frac_at_max':>12}")
for c in ORDER:
    rc = [r for r in st if r["cls"] == c]
    if not rc:
        print(f"{c:>5} {0:>5}")
        continue
    ctok = sum(r["ntok"] for r in rc) / window
    mx = CLASSES[c]["maxtok"]
    fam = sum(1 for r in rc if r["ntok"] >= mx) / len(rc)
    print(f"{c:>5} {len(rc):>5} {ctok:>8.1f} "
          f"{pct([r['ttft_ms'] for r in rc], 50):>7.1f}/{pct([r['ttft_ms'] for r in rc], 95):<7.1f} "
          f"{pct([r['tpot_ms'] for r in rc], 50):>7.1f}/{pct([r['tpot_ms'] for r in rc], 95):<7.1f} "
          f"{pct([r['ntok'] for r in rc], 50):>9.0f} {fam:>12.3f}")
