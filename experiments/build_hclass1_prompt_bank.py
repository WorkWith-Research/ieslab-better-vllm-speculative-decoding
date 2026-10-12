#!/usr/bin/env python3
"""Build data/hclass1_prompt_bank.jsonl — pre-templated diverse prompts for H-CLASS-1
condition A (prereg §4, deviations D1+D2).

On-the-fly template encoding at client startup is too slow for the 2048-ISL class
(~15 full encodes per prompt via binary search; ~700+ prompts needed per run). This
builder does it once, offline, with exact control:

  S: 10000 distinct real ShareGPT-V3 turns, chat-templated, total token length in [42, 66]
     (direct hit only — no truncation; verified ~33% of the short pool hits this window).
  L: 10000 distinct pre-measured long turns (hclass1_longpool.jsonl), content-truncated
     so the templated total lands in [2040, 2064] (binary search on the FULL encoding:
     truncation only removes content tokens, never the template suffix).

Output line: {"cls": "S"|"L", "idx": int, "ids": [...], "len": N}
Deterministic (seed 20261013); idempotent — skips if the file exists and is complete.

The client loads this bank, shuffles it with its own run seed, and cycles through it:
every request in a run gets a unique prompt id (bank >> requests/run), and AR vs DSpark
arms share the same seeded assignment so requests pair by identity."""
import json, os, random, sys
from transformers import AutoTokenizer

REPO = "/home/junior1/_dev/better-vllm-speculative-decoding"
CORPUS = f"{REPO}/data/ShareGPT_V3_unfiltered_cleaned_split.json"
LONGPOOL = f"{REPO}/data/hclass1_longpool.jsonl"
OUT = f"{REPO}/data/hclass1_prompt_bank.jsonl"
N_S, N_L = 10000, 10000
SEED = 20261013
S_LO, S_HI = 42, 66        # target ISL 50, bracket [t-8, t+16]
L_LO, L_HI = 2040, 2064    # target ISL 2048, bracket [t-8, t+16]

if os.path.exists(OUT):
    n_s = n_l = 0
    for l in open(OUT):
        e = json.loads(l)
        if e["cls"] == "S": n_s += 1
        else: n_l += 1
    if n_s >= N_S and n_l >= N_L:
        print(f"exists: {OUT} (S={n_s} L={n_l}); skipping")
        sys.exit(0)

TOK = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)

def tids(turn):
    return TOK.encode(TOK.apply_chat_template(
        [{"role": "user", "content": turn}], add_generation_prompt=True, tokenize=False),
        add_special_tokens=False)

data = json.load(open(CORPUS))
turns = []
for conv in data:
    for m in (conv.get("conversations") or []):
        if m.get("from") == "human":
            v = (m.get("value") or "").strip()
            if len(v) > 20:
                turns.append(v)

rng = random.Random(SEED)

# ---- S class: direct hits in [42, 66] from the short pool ----
short_pool = [t for t in turns if 150 <= len(t) < 4000]
s_bank, seen_s = [], set()
attempts = 0
while len(s_bank) < N_S and attempts < N_S * 60:
    t = short_pool[rng.randrange(len(short_pool))]
    if id(t) in seen_s:      # distinct turns only (id() stable within this process)
        continue
    attempts += 1
    ids = tids(t)
    if S_LO <= len(ids) <= S_HI and t not in seen_s:
        seen_s.add(t)
        s_bank.append({"cls": "S", "idx": len(s_bank), "ids": ids, "len": len(ids)})
if len(s_bank) < N_S:
    raise SystemExit(f"S bank short: {len(s_bank)} < {N_S} after {attempts} attempts")

# ---- L class: binary-search content truncation into [2040, 2064] ----
long_pool = []
for l in open(LONGPOOL):
    e = json.loads(l)
    if 8192 >= e["tok_len"] >= L_HI:      # can reach the window by shaving content
        long_pool.append(e["text"])
print(f"long pool usable: {len(long_pool)} turns", flush=True)

l_bank, seen_l = [], set()
for k in range(N_L):
    t = long_pool[k % len(long_pool)]     # deterministic sweep; bank >> runs so no reuse-in-run
    full = tids(t)                        # one full encode per prompt
    if L_LO <= len(full) <= L_HI:
        best = full
    else:
        lo, hi, best = 16, len(t), None
        while lo <= hi:
            mid = (lo + hi) // 2
            ids = tids(t[:mid])
            if len(ids) <= L_HI:
                best, lo = ids, mid + 1
            else:
                hi = mid - 1
        if best is None or not (L_LO <= len(best) <= L_HI):
            continue
    l_bank.append({"cls": "L", "idx": len(l_bank), "ids": best, "len": len(best)})
    if (k + 1) % 500 == 0:
        print(f"  L {k+1}/{N_L} kept={len(l_bank)}", flush=True)
if len(l_bank) < N_L:
    raise SystemExit(f"L bank short: {len(l_bank)} < {N_L}")

with open(OUT, "w") as f:
    for e in s_bank + l_bank:
        f.write(json.dumps(e) + "\n")
import statistics
sl = sorted(e["len"] for e in s_bank); ll = sorted(e["len"] for e in l_bank)
print(f"DONE: S={len(s_bank)} (p50 {sl[len(sl)//2]}) L={len(l_bank)} (p50 {ll[len(ll)//2]}) -> {OUT}")
