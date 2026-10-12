#!/usr/bin/env python3
"""Build data/hclass1_longpool.jsonl: real ShareGPT-V3 user turns whose CHAT-TEMPLATED
token length >= 2064 (so a 2048-ISL class can be met by content truncation without
dropping below the bracket low bound). Char-length filters are unreliable here
(6k chars often < 2048 templated tokens; verified: only 39.6% of char>=5000 turns qualify).

Output line: {"text": ..., "tok_len": N}   (tok_len = len(templated ids), add_special_tokens=False)
Idempotent: skips if the file already exists and is non-empty."""
import json, os, sys
from transformers import AutoTokenizer

REPO = "/home/junior1/_dev/better-vllm-speculative-decoding"
CORPUS = f"{REPO}/data/ShareGPT_V3_unfiltered_cleaned_split.json"
OUT = f"{REPO}/data/hclass1_longpool.jsonl"
MIN_TOK = 2064

if os.path.exists(OUT) and os.path.getsize(OUT) > 0:
    n = sum(1 for _ in open(OUT))
    print(f"exists: {OUT} ({n} entries); skipping")
    sys.exit(0)

TOK = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
data = json.load(open(CORPUS))
turns = []
for conv in data:
    for m in (conv.get("conversations") or []):
        if m.get("from") == "human":
            v = (m.get("value") or "").strip()
            if len(v) > 20:
                turns.append(v)

def tlen(t):
    return len(TOK.encode(TOK.apply_chat_template(
        [{"role": "user", "content": t}], add_generation_prompt=True, tokenize=False),
        add_special_tokens=False))

# char filter is a fast pre-filter only; exact measurement below decides membership
cands = [t for t in turns if len(t) >= 4000]
print(f"candidates (char>=4000): {len(cands)}", flush=True)
kept = 0
with open(OUT, "w") as f:
    for i, t in enumerate(cands):
        L = tlen(t)
        if L >= MIN_TOK:
            f.write(json.dumps({"text": t, "tok_len": L}) + "\n")
            kept += 1
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(cands)} measured, kept={kept}", flush=True)
print(f"DONE: kept {kept} turns with templated tok_len >= {MIN_TOK}")
