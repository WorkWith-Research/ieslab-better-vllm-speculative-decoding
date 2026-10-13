#!/usr/bin/env python3
"""Build data/hclass1_c_bank.jsonl — pre-templated prompts for H-CLASS-1 condition C
(prereg docs/hclass1c-prereg.md §2): DIVERSE distinct queries + ONE shared context
document (system message) that every request caches, prefix caching ON.

Each prompt = apply_chat_template([{system: D}, {user: Q}], add_generation_prompt=True),
tokenized add_special_tokens=False. D is identical for every entry -> the token PREFIX
(template opening + system block + user header) is byte-identical across requests and is
what the prefix cache hits. Q is a distinct real ShareGPT-V3 turn (no in-run reuse).

Classes (ISL = shared doc D + distinct query Q; uniform output cap 2048 as in cond A):
  S: D(~512 tok) + short query (~50 tok)   -> total ~600
  L: D(~512 tok) + long  query (~2048 tok) -> total ~2570

Output line: {"cls":"S"|"L","idx":int,"ids":[...],"len":N}
Deterministic (seed 20261013); idempotent — skips if the file exists and is complete.
"""
import json, os, random, sys
from transformers import AutoTokenizer

REPO = "/home/junior1/_dev/better-vllm-speculative-decoding"
CORPUS = f"{REPO}/data/ShareGPT_V3_unfiltered_cleaned_split.json"
LONGPOOL = f"{REPO}/data/hclass1_longpool.jsonl"
OUT = f"{REPO}/data/hclass1_c_bank.jsonl"
N_S, N_L = 1000, 1000
SEED = 20261013
DOC_TARGET = 512          # shared context doc content target (tokens)

TOK = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)

def full_ids(d_text, q_text):
    return TOK.encode(TOK.apply_chat_template(
        [{"role": "system", "content": d_text}, {"role": "user", "content": q_text}],
        add_generation_prompt=True, tokenize=False), add_special_tokens=False)

data = json.load(open(CORPUS))
turns = []
for conv in data:
    for m in (conv.get("conversations") or []):
        if m.get("from") == "human":
            v = (m.get("value") or "").strip()
            if len(v) > 20:
                turns.append(v)

rng = random.Random(SEED)

# ---- shared context document D: one real turn, content-truncated so the SHARED PREFIX
#      (full_ids(D, "x")) lands near DOC_TARGET+overhead. Identical across all entries. ----
long_pool_all = [e["text"] for e in map(json.loads, open(LONGPOOL)) if 8192 >= e["tok_len"] >= 600]
d_src = long_pool_all[0]
lo, hi, D_text = 16, len(d_src), d_src[:400]
# binary search content length so full_ids(D,"x") is in [DOC_TARGET, DOC_TARGET+40]
target_lo, target_hi = DOC_TARGET, DOC_TARGET + 40
best = None
while lo <= hi:
    mid = (lo + hi) // 2
    L = len(full_ids(d_src[:mid], "x"))
    if L <= target_hi:
        best, D_text = mid, d_src[:mid]; lo = mid + 1
    else:
        hi = mid - 1
PFX = len(full_ids(D_text, "x"))          # shared prefix length (incl. ~1-tok sentinel query)
print(f"shared doc D: content={len(D_text)} chars -> shared prefix PFX={PFX} tok", flush=True)

# ---- S class: distinct short queries, total in [PFX+40, PFX+70] (query ~50 tok) ----
short_pool = [t for t in turns if 150 <= len(t) < 4000]
s_lo, s_hi = PFX + 40, PFX + 70
s_bank, seen_s = [], set()
attempts = 0
while len(s_bank) < N_S and attempts < N_S * 60:
    t = short_pool[rng.randrange(len(short_pool))]
    if t in seen_s:
        continue
    attempts += 1
    ids = full_ids(D_text, t)
    if s_lo <= len(ids) <= s_hi:
        seen_s.add(t)
        s_bank.append({"cls": "S", "idx": len(s_bank), "ids": ids, "len": len(ids)})
if len(s_bank) < N_S:
    raise SystemExit(f"S bank short: {len(s_bank)} < {N_S} after {attempts} attempts")

# ---- L class: distinct long queries, content-truncated so total in [PFX+2035, PFX+2070] ----
long_pool = [e["text"] for e in map(json.loads, open(LONGPOOL)) if 8192 >= e["tok_len"]]
l_lo, l_hi = PFX + 2035, PFX + 2070
l_bank, seen_l = [], set()
for k in range(N_L * 3):
    t = long_pool[k % len(long_pool)]
    if t in seen_l:
        continue
    full = full_ids(D_text, t)
    if l_lo <= len(full) <= l_hi:
        best = full
    else:
        lo2, hi2, cand = 16, len(t), None
        while lo2 <= hi2:
            mid = (lo2 + hi2) // 2
            ids = full_ids(D_text, t[:mid])
            if len(ids) <= l_hi:
                cand, lo2 = ids, mid + 1
            else:
                hi2 = mid - 1
        best = cand
    if best is None or not (l_lo <= len(best) <= l_hi):
        continue
    seen_l.add(t)
    l_bank.append({"cls": "L", "idx": len(l_bank), "ids": best, "len": len(best)})
    if len(l_bank) >= N_L:
        break
    if (k + 1) % 500 == 0:
        print(f"  L {k+1}/{N_L*3} kept={len(l_bank)}", flush=True)
if len(l_bank) < N_L:
    raise SystemExit(f"L bank short: {len(l_bank)} < {N_L}")

# ---- verify the shared prefix is actually identical across entries (cache precondition) ----
minlen = min(len(e["ids"]) for e in s_bank[:50] + l_bank[:50])
ref = s_bank[0]["ids"][:minlen]
ok = all(tuple(e["ids"][:minlen]) == tuple(ref) for e in s_bank[:50] + l_bank[:50])
print(f"shared-prefix identical across 100 sampled entries (first {minlen} tok): {ok}", flush=True)

with open(OUT, "w") as f:
    for e in s_bank + l_bank:
        f.write(json.dumps(e) + "\n")
import statistics
sl = sorted(e["len"] for e in s_bank); ll = sorted(e["len"] for e in l_bank)
print(f"DONE: S={len(s_bank)} (p50 {sl[len(sl)//2]}) L={len(l_bank)} (p50 {ll[len(ll)//2]}) PFX={PFX} -> {OUT}")
