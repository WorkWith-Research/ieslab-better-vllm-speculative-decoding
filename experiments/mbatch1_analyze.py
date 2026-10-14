#!/usr/bin/env python3
"""M-BATCH-1 analysis (prereg docs/mbatch1-prereg.md §4).

Primary: per-class TPOT payoff for B requests under the MIXED stream =
mean over pid-matched pairs of (1 - DSparkTPOT/ARTPOT), pooling valid reps per load,
request-level paired bootstrap (2000 resamples). Compare vs homogeneous H-CLASS-1 anchors.
Secondary: A-class payoff (control), B payoff vs in_flight_at_arrival bins, and
server-side context (KV usage, prefix hit rate, gen throughput) from server.log.

Usage: mbatch1_analyze.py [OUT_DIR]   (default results/mbatch1)
"""
import json, os, random, re, sys

REPO = "/home/junior1/_dev/better-vllm-speculative-decoding"
OUT = sys.argv[1] if len(sys.argv) > 1 else f"{REPO}/results/mbatch1"
HCLASS1 = f"{REPO}/results/hclass1"
WARMUP_S = 10.0
rng = random.Random(20261013)

def load_cell(cell):
    """pid -> record for a valid (DONE) cell; None if not done."""
    d = os.path.join(OUT, cell)
    if not os.path.exists(os.path.join(d, "DONE")):
        return None
    recs = [json.loads(l) for l in open(os.path.join(d, "client.jsonl"))]
    out = {}
    for r in recs:
        if "tpot_ms" in r and WARMUP_S <= r["t"] <= 150.0:
            out[r["pid"]] = r
    return out

def homogeneous_anchor(cond, C):
    """pid -> tpot from H-CLASS-1 cond cells (all 3 trials pooled)."""
    d = {}
    for arm in ("ar", "dspark"):
        dd = {}
        for t in (1, 2, 3):
            f = os.path.join(HCLASS1, f"{arm}_{cond}_c{C}_t{t}", "client.jsonl")
            if not os.path.exists(f):
                continue
            for l in open(f):
                r = json.loads(l)
                if "tpot_ms" in r:
                    dd[r["pid"]] = r["tpot_ms"]
        d[arm] = dd
    return d

def paired_payoff(ar, ds, pids=None, n_boot=2000):
    pairs = []
    for p in (pids if pids is not None else ar):
        if p in ds and ar[p] > 0:
            pairs.append(1.0 - ds[p] / ar[p])
    if len(pairs) < 5:
        return None
    n = len(pairs)
    mean = sum(pairs) / n
    boots = []
    for _ in range(n_boot):
        s = [pairs[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(s) / n)
    boots.sort()
    return {"n": n, "payoff": round(mean, 4),
            "ci95": [round(boots[int(0.025 * n_boot)], 4), round(boots[int(0.975 * n_boot)], 4)]}

def server_ctx(cell):
    f = os.path.join(OUT, cell, "server.log")
    if not os.path.exists(f):
        return {}
    txt = open(f, errors="ignore").read()
    gen = [float(m) for m in re.findall(r"Avg generation throughput: ([\d.]+)", txt)]
    kv = [float(m) for m in re.findall(r"KV cache usage: ([\d.]+)%", txt)]
    hit = [float(m) for m in re.findall(r"Prefix cache hit rate: ([\d.]+)%", txt)]
    def tail_mean(xs):
        xs = xs[-20:] if len(xs) >= 5 else xs
        return round(sum(xs) / len(xs), 3) if xs else None
    return {"gen_tok_s": tail_mean(gen), "kv_pct": tail_mean(kv), "hit_pct": tail_mean(hit)}

report = []
ap = report.append
ap("# M-BATCH-1 — B-class payoff under live MIXED batches")
ap("")
ap("Prereg: `docs/mbatch1-prereg.md`. Payoff = mean over pid-matched pairs of (1 - DSparkTPOT/ARTPOT); "
   ">0 ⇒ DSpark faster. Pooled across valid reps per load; bootstrap 2000.")
ap("")

anchors = {}
for C in (48, 96):
    h = homogeneous_anchor("B", C)
    a = paired_payoff(h["ar"], h["dspark"])
    anchors[C] = a
    if a:
        ap(f"Homogeneous anchor B c{C} (H-CLASS-1): payoff={a['payoff']:+.4f} n={a['n']} CI95={a['ci95']}")
    else:
        ap(f"Homogeneous anchor B c{C} (H-CLASS-1): NOT AVAILABLE")
ap("")

results = {}
for C in (48, 96):
    ar_all, ds_all = {}, {}
    valid = {"ar": 0, "dspark": 0}
    for t in (1, 2, 3):
        a = load_cell(f"ar_c{C}_t{t}")
        d = load_cell(f"dspark_c{C}_t{t}")
        if a: valid["ar"] += 1; ar_all.update(a)
        if d: valid["dspark"] += 1; ds_all.update(d)
    if not ar_all or not ds_all:
        ap(f"## c{C}: NO VALID CELLS (ar={valid['ar']} dspark={valid['dspark']})")
        continue
    # primary: B pooled + per class  (B-class names have no "B" prefix — hclass1 convention)
    BCLS = {"SS", "SL", "LS", "LL"}
    b_pids = [p for p in ar_all if p.startswith("MIX")]
    def is_b(p):
        r = ar_all.get(p) or ds_all.get(p)
        return bool(r) and r["cls"] in BCLS
    rows = []
    pb = paired_payoff({p: ar_all[p]["tpot_ms"] for p in b_pids if is_b(p)},
                       {p: ds_all[p]["tpot_ms"] for p in b_pids if is_b(p)})
    rows.append(("B (pooled)", pb))
    for cls in ("SS", "SL", "LS", "LL"):
        pids = [p for p in b_pids if (ar_all.get(p) or ds_all.get(p))["cls"] == cls]
        rows.append((cls, paired_payoff({p: ar_all[p]["tpot_ms"] for p in pids},
                                        {p: ds_all[p]["tpot_ms"] for p in pids})))
    pa = paired_payoff({p: ar_all[p]["tpot_ms"] for p in b_pids if not is_b(p)},
                       {p: ds_all[p]["tpot_ms"] for p in b_pids if not is_b(p)})
    rows.append(("A (control)", pa))

    ap(f"## c{C} (valid reps ar={valid['ar']} dspark={valid['dspark']})")
    ap("")
    ap("| class | n pairs | payoff | CI95 | vs homogeneous B anchor |")
    ap("|---|---:|---:|---|---|")
    for name, r in rows:
        if not r:
            ap(f"| {name} | 0 | — | — | — |")
            continue
        ci = f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]"
        ref = ""
        anc = anchors.get(C)
        if name == "B (pooled)" and anc:
            ref = f"anchor {anc['payoff']:+.4f}"
        ap(f"| {name} | {r['n']} | {r['payoff']:+.4f} | {ci} | {ref} |")
    ap("")
    # secondary: B payoff vs in-flight-at-arrival bins (queue depth proxy)
    pairs = []
    for p in b_pids:
        if is_b(p) and p in ds_all and ar_all.get(p, {}).get("tpot_ms", 0) > 0:
            pairs.append((ar_all[p].get("in_flight_at_arrival", 0), 1.0 - ds_all[p]["tpot_ms"] / ar_all["tpot_ms"]))
    if len(pairs) >= 20:
        bins = [(0, 5), (5, 15), (15, 40), (40, 10**9)]
        ap("B payoff vs in-flight-at-arrival (mixed-batch depth proxy):")
        ap("")
        ap("| in-flight at arrival | n | mean payoff |")
        ap("|---|---:|---:|")
        for lo, hi in bins:
            bp = [v for qd, v in pairs if lo <= qd < hi]
            if bp:
                ap(f"| [{lo}, {hi if hi < 10**9 else '∞'}) | {len(bp)} | {sum(bp)/len(bp):+.4f} |")
        ap("")
    # server context
    ctx = []
    for arm in ("ar", "dspark"):
        for t in (1, 2, 3):
            c = server_ctx(f"{arm}_c{C}_t{t}")
            if c:
                ctx.append((f"{arm} t{t}", c))
    ap("Server context (tail means): " + "; ".join(
        f"{n}: gen={c['gen_tok_s']} tok/s, KV={c['kv_pct']}%, hit={c['hit_pct']}%" for n, c in ctx))
    ap("")
    results[C] = {"B": pb, "A": pa}

# decision (prereg §5)
ap("## Decision (prereg §5)")
ap("")
for C, r in results.items():
    b = r.get("B")
    if not b:
        ap(f"c{C}: underpowered — no paired B payoff.")
        continue
    lo, hi = b["ci95"]
    a_anc = anchors.get(C, {}).get("payoff")
    if hi <= 0.0:
        verdict = "H1 CONFIRMED at c%d (B payoff CI entirely ≤ 0)" % C
    elif lo > 0 and a_anc is not None and b["payoff"] < a_anc - max(0.05, abs(b["ci95"][1] - b["ci95"][0]) / 2):
        verdict = "H2 (headroom exists) at c%d — B payoff > 0 but below homogeneous anchor" % C
    elif lo > 0:
        verdict = "H2 (headroom exists) at c%d — B payoff > 0 within CI under mixing" % C
    else:
        verdict = "c%d: CI straddles 0 — marginal; compare magnitude vs homogeneous anchor (%s)" % (C, a_anc)
    ap(f"- {verdict}")
ap("")

out_md = os.path.join(OUT, "report.md")
os.makedirs(OUT, exist_ok=True)
open(out_md, "w").write("\n".join(report))
print("\n".join(report))
print(f"\nwrote {out_md}")
