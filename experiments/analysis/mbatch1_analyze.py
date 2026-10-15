#!/usr/bin/env python3
"""M-BATCH-1 analysis (prereg docs/mbatch1-prereg.md §4).

Primary: per-class TPOT payoff for B requests under the MIXED stream =
mean over pid-matched pairs of (1 - DSparkTPOT/ARTPOT), paired WITHIN each rep
(pid repeats across reps by construction — same seed) and pooled across reps,
request-level paired bootstrap (2000 resamples). Compare vs homogeneous H-CLASS-1
anchors (same within-trial pairing convention as docs/scheduler-signal-validation.md).
Secondary: A-class payoff (control), B payoff vs in_flight_at_arrival bins, and
server-side context (KV usage, prefix hit rate, gen throughput) from server.log.

Usage: mbatch1_analyze.py [OUT_DIR]   (default results/mbatch1)
"""
import json, os, random, re, sys

REPO = "/home/junior1/_dev/better-vllm-speculative-decoding"
OUT = sys.argv[1] if len(sys.argv) > 1 else f"{REPO}/results/mbatch1"
HCLASS1 = f"{REPO}/results/hclass1"
WARMUP_S = 10.0
BCLS = {"SS", "SL", "LS", "LL"}   # B-class names have no "B" prefix (hclass1 convention)
rng = random.Random(20261013)

def load_cell(cell, warmup=None):
    """pid -> record for a valid cell; None if not done. Pids are unique WITHIN a
    cell (one arrival index per request); they repeat across reps by design."""
    d = os.path.join(OUT, cell)
    if not os.path.exists(os.path.join(d, "DONE")):
        return None
    recs = [json.loads(l) for l in open(os.path.join(d, "client.jsonl"))]
    out = {}
    for r in recs:
        if "tpot_ms" not in r:
            continue
        t_lo = WARMUP_S if warmup is None else warmup
        if t_lo <= r["t"] <= 150.0:
            out[r["pid"]] = r
    return out

def paired_payoff(pairs, n_boot=2000):
    """pairs: list of (ar_tpot, ds_tpot) with ar_tpot > 0."""
    vals = [1.0 - d / a for a, d in pairs if a > 0]
    if len(vals) < 5:
        return None
    n = len(vals)
    mean = sum(vals) / n
    boots = []
    for _ in range(n_boot):
        s = [vals[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(s) / n)
    boots.sort()
    return {"n": n, "payoff": round(mean, 4),
            "ci95": [round(boots[int(0.025 * n_boot)], 4), round(boots[int(0.975 * n_boot)], 4)]}

def homogeneous_anchor(cond, C):
    """Within-trial pid pairing across the 3 H-CLASS-1 trials, pooled (the convention
    used by docs/scheduler-signal-validation.md)."""
    pairs = []
    for t in (1, 2, 3):
        fa = os.path.join(HCLASS1, f"ar_{cond}_c{C}_t{t}", "client.jsonl")
        fd = os.path.join(HCLASS1, f"dspark_{cond}_c{C}_t{t}", "client.jsonl")
        if not (os.path.exists(fa) and os.path.exists(fd)):
            continue
        a = {json.loads(l)["pid"]: json.loads(l) for l in open(fa)}
        d = {json.loads(l)["pid"]: json.loads(l) for l in open(fd)}
        for p in a:
            if p in d and "tpot_ms" in a[p] and "tpot_ms" in d[p]:
                pairs.append((a[p]["tpot_ms"], d[p]["tpot_ms"]))
    return paired_payoff(pairs)

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
   ">0 ⇒ DSpark faster. Paired within rep, pooled across reps; bootstrap 2000.")
ap("")

anchors = {}
for C in (48, 96):
    a = homogeneous_anchor("B", C)
    anchors[C] = a
    if a:
        ap(f"Homogeneous anchor B c{C} (H-CLASS-1): payoff={a['payoff']:+.4f} n={a['n']} CI95={a['ci95']}")
    else:
        ap(f"Homogeneous anchor B c{C} (H-CLASS-1): NOT AVAILABLE")
ap("")

results = {}
for C in (48, 96):
    valid = {"ar": 0, "dspark": 0}
    # per-rep pairing: pid repeats across reps, so pair within each rep then pool
    by_cls = {"B": [], "A": []}            # cls group -> list of (ar_tpot, ds_tpot)
    b_qdepth = []                           # (in_flight_at_arrival, payoff) for B pairs
    for t in (1, 2, 3):
        a = load_cell(f"ar_c{C}_t{t}")
        d = load_cell(f"dspark_c{C}_t{t}")
        if not a or not d:
            continue
        if a: valid["ar"] += 1
        if d: valid["dspark"] += 1
        for p, ra in a.items():
            rd = d.get(p)
            if not rd:
                continue
            grp = "B" if ra["cls"] in BCLS else "A"
            by_cls[grp].append((ra["tpot_ms"], rd["tpot_ms"]))
            if grp == "B":
                b_qdepth.append((ra.get("in_flight_at_arrival", 0),
                                 1.0 - rd["tpot_ms"] / ra["tpot_ms"] if ra["tpot_ms"] > 0 else None))
    if not by_cls["B"]:
        ap(f"## c{C}: NO PAIRED CELLS (ar={valid['ar']} dspark={valid['dspark']})")
        continue

    pb = paired_payoff(by_cls["B"])
    pa = paired_payoff(by_cls["A"])
    rows = [("B (pooled)", pb)]
    # per-class: rebuild from cells
    per_cls = {c: [] for c in ("SS", "SL", "LS", "LL")}
    for t in (1, 2, 3):
        a = load_cell(f"ar_c{C}_t{t}")
        d = load_cell(f"dspark_c{C}_t{t}")
        if not a or not d:
            continue
        for p, ra in a.items():
            rd = d.get(p)
            if rd and ra["cls"] in per_cls:
                per_cls[ra["cls"]].append((ra["tpot_ms"], rd["tpot_ms"]))
    for cls in ("SS", "SL", "LS", "LL"):
        rows.append((cls, paired_payoff(per_cls[cls])))
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
    qd = [(q, v) for q, v in b_qdepth if v is not None]
    if len(qd) >= 20:
        bins = [(0, 5), (5, 15), (15, 40), (40, 10**9)]
        ap("B payoff vs in-flight-at-arrival (mixed-batch depth proxy):")
        ap("")
        ap("| in-flight at arrival | n | mean payoff |")
        ap("|---|---:|---:|")
        for lo, hi in bins:
            bp = [v for q, v in qd if lo <= q < hi]
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
    a_anc = anchors[C]["payoff"] if anchors.get(C) else None
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
