#!/usr/bin/env python3
"""Phase 4.6 analyzer: live-load-aware LDM vs baselines (pre-registered in PROGRESS.md).

Arms at C in {8,32,96} (mixed workload):
  AR / K4 / K8   -> Phase 4.2 cells (results/p2/p2_k{K}_c{C}_t{T}_mixed_client.out)
  LDM            -> Phase 4.5 cells (results/p5/p5_ldm_c{C}_t{T}_mixed_client.out)
  DSpark-rule    -> Phase 4.5 cells (results/p5/p5_dspark_c{C}_t{T}_mixed_client.out)
  ldmload        -> Phase 4.6 cells (results/p4_6/p46_ldmload_c{C}_t{T}_mixed_client.out)

Pre-registered success/falsify bounds:
  C=96: ldmload >= AR - noise AND beats DSpark-rule   (noise = max trial spread of AR)
  C=8 : ldmload >= AR + 5%                            (recovers most of fixed-K8's +10.3%)
Also reports the K distribution from decision logs (does it actually use large K at low
load and commit to SD-off at saturation WITHOUT being told C?).
"""
import json
import os
import re
from collections import Counter


def steady(path):
    if not os.path.exists(path):
        return None
    last = [l for l in open(path) if "steady_throughput_tok_s=" in l]
    if not last:
        return None
    m = re.search(r"steady_throughput_tok_s=([\d.]+)", last[-1])
    return float(m.group(1)) if m else None


def arm_cells(arm, C):
    pats = {
        "AR": f"results/p2/p2_knone_c{C}_t{{T}}_mixed_client.out",
        "K4": f"results/p2/p2_k4_c{C}_t{{T}}_mixed_client.out",
        "K8": f"results/p2/p2_k8_c{C}_t{{T}}_mixed_client.out",
        "LDM": f"results/p5/p5_ldm_c{C}_t{{T}}_mixed_client.out",
        "DSpark-rule": f"results/p5/p5_dspark_c{C}_t{{T}}_mixed_client.out",
        "ldmload": f"results/p4_6/p46_ldmload_c{C}_t{{T}}_mixed_client.out",
    }[arm]
    vals = []
    for T in (0, 1, 2):
        v = steady(pats.format(T=T))
        if v:
            vals.append(v)
    return vals


def kdist(arm, C):
    pats = {
        "LDM": f"results/p5/p5_ldm_c{C}_t{{T}}_mixed_decisions.jsonl",
        "DSpark-rule": f"results/p5/p5_dspark_c{C}_t{{T}}_mixed_decisions.jsonl",
        "ldmload": f"results/p4_6/p46_ldmload_c{C}_t{{T}}_mixed_decisions.jsonl",
    }[arm]
    ks = Counter()
    nsteps = 0
    for T in (0, 1, 2):
        p = pats.format(T=T)
        if not os.path.exists(p):
            continue
        for line in open(p):
            try:
                r = json.loads(line)
            except Exception:
                continue
            nsteps += 1
            for d in r.get("decisions", []):
                k = d.get("kstar", d.get("Knew"))
                if k is not None:
                    ks[int(k)] += 1
    return ks, nsteps


def main():
    arms = ["AR", "K4", "K8", "LDM", "DSpark-rule", "ldmload"]
    Cs = [8, 32, 96]
    print("=" * 86)
    print("Phase 4.6 — live-load-aware adaptive speculation (mixed workload)")
    print("=" * 86)
    hdr = f"{'C':>4} | " + " | ".join(f"{a:>12}" for a in arms) + " | best"
    print(hdr)
    results = {}
    for C in Cs:
        row = f"{C:>4} | "
        vals = {}
        for a in arms:
            v = arm_cells(a, C)
            if v:
                m = sum(v) / len(v)
                vals[a] = (m, min(v), max(v))
                row += f"{m:>12.1f} | "
            else:
                row += f"{'--':>12} | "
        if vals:
            best = max(vals, key=lambda a: vals[a][0])
            row += best
        results[C] = vals
        print(row)

    # pre-registered bounds
    print()
    ok96 = ok8 = False
    if 96 in results and "ldmload" in results[96] and "AR" in results[96]:
        ar = results[96]["AR"]
        noise = ar[2] - ar[1]   # trial spread of AR (max-min over trials)
        ll = results[96]["ldmload"][0]
        ds = results[96].get("DSpark-rule", (None,))[0]
        ok96 = ll >= ar[0] - noise and (ds is None or ll > ds)
        print(f"C=96 bound: ldmload {ll:.1f} vs AR {ar[0]:.1f} (noise {noise:.1f}) "
              f"-> {'PASS' if ok96 else 'FAIL'}; DSpark-rule {ds if ds is None else round(ds,1)}")
    if 8 in results and "ldmload" in results[8] and "AR" in results[8]:
        ar = results[8]["AR"][0]
        ll = results[8]["ldmload"][0]
        ok8 = ll >= ar * 1.05
        print(f"C=8 bound: ldmload {ll:.1f} vs AR {ar:.1f} (+5% bar {ar*1.05:.1f}) "
              f"-> {'PASS' if ok8 else 'FAIL'}")

    # K distributions (the qualitative success question)
    print()
    print("K distribution from decision logs (mean K, frac K=0):")
    for a in ("LDM", "DSpark-rule", "ldmload"):
        for C in Cs:
            ks, nsteps = kdist(a, C)
            if not ks:
                continue
            tot = sum(ks.values())
            mean_k = sum(k * v for k, v in ks.items()) / tot
            f0 = ks.get(0, 0) / tot
            print(f"  {a:>12} C={C:>3}: steps={nsteps:>5} meanK={mean_k:.2f} fracK0={f0:.2%} "
                  f"dist={dict(sorted(ks.items()))}")

    verdict = "SUPPORT (both bounds pass)" if (ok96 and ok8) else \
              ("PARTIAL" if (ok96 or ok8) else "FALSIFY (neither bound met)")
    print()
    print(f"PRE-REGISTERED VERDICT: {verdict}")


if __name__ == "__main__":
    main()
