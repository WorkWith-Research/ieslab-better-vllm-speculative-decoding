#!/usr/bin/env python3
"""Phase 7 / H-CLASS-1 — shared prompt construction (importable, no side effects).

Extracted from hclass1_client.py so the Phase 7 mixed-stream client can reuse the
EXACT same diverse-A prompt bank and canonical-B high-reuse prompts without executing
hclass1_client's main body. Single source of truth: hclass1_client.py imports these too.

Provides: TOK (tokenizer), TURNS, SHORT_CANDS, LONG_CANDS, CLASSES_A/B/C, ORDER_*,
CANON (canonical H-OBS-3 high-reuse prompts), EOS_TOKEN_ID, and the diverse-A prompt bank.
"""
import json, random

from transformers import AutoTokenizer

MODEL = "qwen3-4b"
TOK_MODEL = "Qwen/Qwen3-4B"
REPO = "/home/junior1/_dev/better-vllm-speculative-decoding"
CORPUS = f"{REPO}/data/ShareGPT_V3_unfiltered_cleaned_split.json"
LONGPOOL = f"{REPO}/data/hclass1_longpool.jsonl"
BANK_A_PATH = f"{REPO}/data/hclass1_prompt_bank.jsonl"

# Class schemes (ISL exact by token count; max_tokens is a CAP, realized OSL measured).
CLASSES_A = {"S": {"isl": 50, "maxtok": 2048}, "L": {"isl": 2048, "maxtok": 2048}}
ORDER_A = ["S", "L"]
CLASSES_B = {
    "SS": {"isl": 50, "maxtok": 256}, "SL": {"isl": 50, "maxtok": 1024},
    "LS": {"isl": 2048, "maxtok": 256}, "LL": {"isl": 2048, "maxtok": 1024},
}
ORDER_B = ["SS", "SL", "LS", "LL"]
CLASSES_C = {"S": {"isl": 600, "maxtok": 2048}, "L": {"isl": 2600, "maxtok": 2048}}
ORDER_C = ["S", "L"]

EOS_TOKEN_ID = 151645   # <|im_end|> == Qwen3-4B eos_token (verified via tokenizer)

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

# Pre-tokenized pools by char-length buckets (fast filter).
SHORT_CANDS = [t for t in TURNS if 150 <= len(t) < 4000]      # brackets ~50 tok
LONG_CANDS = [e["text"] for e in map(json.loads, open(LONGPOOL)) if e["tok_len"] <= 8192]


def load_turns_hoobs3(path):
    """H-OBS-3's EXACT turn filter (hetero_client_qwen3.py): 40 < len(value) < 20000."""
    data = json.load(open(path))
    turns = []
    for conv in data:
        msgs = conv.get("conversations") or []
        for m in msgs:
            if m.get("from") == "human":
                v = (m.get("value") or "").strip()
                if 40 < len(v) < 20000:
                    turns.append(v)
    return turns


def build_canonical():
    """H-OBS-3's exact construction sequence, including its per-ISL _cache:
    random.seed(20261011); TURNS shuffled ONCE; then make_prompt runs for each class in
    CLASSES order BUT is cached by isl_target — so only SS (ISL 50) and LS (ISL 2048)
    actually draw from the RNG; SL/LL are cache hits that consume NO random numbers.
    Replicating the cache keeps every prompt byte-identical to H-OBS-3's."""
    rng = random.Random(20261011)
    tcopy = load_turns_hoobs3(CORPUS)
    rng.shuffle(tcopy)
    cache = {}
    out = {}
    for c in ORDER_B:
        isl_t = CLASSES_B[c]["isl"]
        if isl_t in cache:
            out[c] = cache[isl_t]
            continue
        ids, used, i = [], set(), rng.randrange(len(tcopy))
        while len(ids) < isl_t and len(used) < 64:
            j = (i + len(used)) % len(tcopy)
            if j in used:
                continue
            used.add(j)
            ids.extend(TOK.encode(tcopy[j], add_special_tokens=False))
        ids = ids[:isl_t]
        cache[isl_t] = ids
        out[c] = ids
    return out


# Canonical prompts are ALWAYS the H-OBS-3 4-class scheme.
CANON = build_canonical()

# Diverse-A prompt bank: pre-tokenized chat-templated unique prompts (built by
# experiments/build_hclass1_prompt_bank.py). {cls: [ids, ...]}.
BANK_A = {c: [] for c in ORDER_A}
for _l in open(BANK_A_PATH):
    _e = json.loads(_l)
    if _e["cls"] in BANK_A:
        BANK_A[_e["cls"]].append(_e["ids"])
