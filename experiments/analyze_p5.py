#!/usr/bin/env python3
"""P5 aggregator: in-loop LDM + DSpark-rule vs fixed baselines (reused from P2).

Merges:
  - AR / K4 / K8  : results/p2/p2_k{none,4,8}_c{C}_t{1..3}_mixed_client.out
  - LDM           : results/p5/p5_ldm_c{C}_t{1..3}_mixed_*
  - DSpark-rule   : results/p5/p5_dspark_c{C}_t{1..3}_mixed_*
and reports per-arm K distributions (from decision logs) + throughput.

Usage: .venv/bin/python experiments/analyze_p5.py
"""
import json, os, re, statistics as st

KS_BASE = {"AR": "none", "K4": "4", "K8": "8"}
CS = [32, 96]


def client_thr(path):
    m = None
    try:
        for line in open(path):
            m = re.search(r"steady_out_tokens=(\d+) steady_throughput_tok_s=([\d.]+)", line)
    except Exception:
        return None, None
    if not m:
        return None, None
    return int(m.group(1)), float(m.group(2))


def kdist(path):
    if not os.path.exists(path):
        return None
    ks = []
    for line in open(path):
        r = json.loads(line)
        for d in r.get("decisions", ()):
            v = d.get("Knew", d.get("kstar"))
            if v is not None:
                ks.append(v)
    if not ks:
        return None
    from collections import Counter
    c = Counter(ks)
    tot = sum(c.values())
    return {"mean": round(sum(k * n for k, n in c.items()) / tot, 2),
            "dist": {k: round(n / tot, 3) for k, n in sorted(c.items())},
            "frac_zero": round(c.get(0, 0) / tot, 3)}


def main():
    print("== P5: in-loop adaptive arms vs fixed baselines (mixed workload)")
    print(f"{'arm':>14} | {'C=32 thr±sd':>14} | {'C=96 thr±sd':>14}")
    arms = {}
    for name, kkey in KS_BASE.items():
        for C in CS:
            vals = []
            for t in (1, 2, 3):
                _, thr = client_thr(f"results/p2/p2_k{kkey}_c{C}_t{t}_mixed_client.out")
                if thr:
                    vals.append(thr)
            arms[(name, C)] = vals
    for arm in ("LDM", "DSpark-rule"):
        kkey = "ldm" if arm == "LDM" else "dspark"
        for C in CS:
            vals = []
            for t in (1, 2, 3):
                _, thr = client_thr(f"results/p5/p5_{kkey}_c{C}_t{t}_mixed_client.out")
                if thr:
                    vals.append(thr)
            arms[(arm, C)] = vals

    for name in ("AR", "K4", "K8", "LDM", "DSpark-rule"):
        cells = []
        for C in CS:
            v = arms.get((name, C), [])
            cells.append(f"{st.mean(v):>7.1f}±{st.pstdev(v) if len(v)>1 else 0:<4.1f}" if v else "       --")
        print(f"{name:>14} | " + " | ".join(cells))

    print("\n== K distributions (decision logs)")
    for arm, kkey in (("LDM", "ldm"), ("DSpark-rule", "dspark")):
        for C in CS:
            dists = []
            for t in (1, 2, 3):
                d = kdist(f"results/p5/p5_{kkey}_c{C}_t{t}_mixed_decisions.jsonl")
                if d:
                    dists.append(d)
            if not dists:
                continue
            mean_k = st.mean([d["mean"] for d in dists])
            f0 = st.mean([d["frac_zero"] for d in dists])
            print(f"{arm} C={C}: mean K={mean_k:.2f}, frac(K=0)={f0:.3f}")
            # pooled dist
            from collections import Counter
            pooled = Counter()
            tot = 0
            for d in dists:
                for k, f in d["dist"].items():
                    pooled[int(k)] += f
                    tot += f
            print("   " + " ".join(f"K{k}={v/tot:.2f}" for k, v in sorted(pooled.items())))

    # gap summary
    print("\n== gaps")
    for C in CS:
        ar = st.mean(arms.get(("AR", C), [0])) or None
        best_fixed = max((st.mean(arms[(n, C)]) for n in ("K4", "K8") if arms.get((n, C))), default=None)
        ldm = st.mean(arms[("LDM", C)]) if arms.get(("LDM", C)) else None
        ds = st.mean(arms[("DSpark-rule", C)]) if arms.get(("DSpark-rule", C)) else None
        if ar and best_fixed:
            print(f"C={C}: AR={ar:.1f}, best fixed-K={best_fixed:.1f} (gap {100*(ar-best_fixed)/ar:+.1f}% vs AR)")
        for name, v in (("LDM", ldm), ("DSpark-rule", ds)):
            if v and ar:
                print(f"C={C}: {name}={v:.1f} ({100*(v-ar)/ar:+.1f}% vs AR)")


if __name__ == "__main__":
    main()
