#!/usr/bin/env python
# P5.2 H-5.2 P3 adjudication: specfree-off vs force-0 vs AR (server-side gen/s, steady t>=20s).
# Verdict per prereg §6:
#   CONFIRMED      : specfree >= force-0 + 50%*(AR-force-0) at C=96 AND within ~2% of AR at C=96
#                    AND no regression below -2% vs force-0 at C=32.
#   PARTIAL        : recovers some but <50% of the gap, or reaches AR at one C only.
#   FALSIFIED      : specfree <= force-0 + 20%*(AR-force-0) at C=96 (mechanism doesn't remove it).
import json, os, statistics as st


def gen_s(fn):
    if not os.path.exists(fn):
        return None
    rows = [json.loads(l) for l in open(fn)]
    mr = [r for r in rows if r.get("t", 0) >= 20]
    if len(mr) < 10:
        return None
    a, b = mr[0], mr[-1]
    dt = max(1e-9, b["t"] - a["t"])
    c = lambda r, k: (r.get(k, 0) or 0)
    return (c(b, "vllm:generation_tokens[counter]") - c(a, "vllm:generation_tokens[counter]")) / dt


def arm_mean(base, arm, C, T):
    vals = []
    for t in range(T):
        g = gen_s(os.path.join(base, f"p52_{arm}_c{C}_t{t}_metrics.jsonl"))
        if g is not None:
            vals.append(g)
    return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)


def main():
    base = "results/p5_2"
    T = 3
    print(f"{'C':>4} {'AR':>9} {'force-0':>9} {'specfree':>9}  gap(AR-f0)  specfree-vs-f0  specfree-vs-AR")
    results = {}
    for C in [32, 96]:
        ar, na = arm_mean(base, "ar", C, T)
        f0, nf = arm_mean(base, "f0", C, T)
        sf, ns = arm_mean(base, "sf", C, T)
        results[C] = dict(AR=ar, force0=f0, specfree=sf, n=(na, nf, ns))
        if None in (ar, f0, sf):
            print(f"{C:>4}  INCOMPLETE (n={na},{nf},{ns})")
            continue
        gap = ar - f0
        recov = (sf - f0) / gap if gap else 0.0
        vs_ar = (sf - ar) / ar * 100.0
        print(f"{C:>4} {ar:9.1f} {f0:9.1f} {sf:9.1f}  {gap:11.1f}  {(sf-f0):+10.1f} ({recov*100:+5.0f}% of gap)  {vs_ar:+8.2f}%")
    json.dump(results, open(os.path.join(base, "p52_p3_verdict.json"), "w"), indent=2)

    # Verdict (C=96 primary, C=32 secondary)
    r96 = results.get(96); r32 = results.get(32)
    if r96 and None not in (r96["AR"], r96["force0"], r96["specfree"]):
        gap = r96["AR"] - r96["force0"]
        recov = (r96["specfree"] - r96["force0"]) / gap if gap else 0.0
        vs_ar = abs(r96["specfree"] - r96["AR"]) / r96["AR"] * 100.0
        reg32 = None
        if r32 and None not in (r32["force0"], r32["specfree"]):
            reg32 = (r32["specfree"] - r32["force0"]) / r32["force0"] * 100.0
        print(f"\n=== VERDICT INPUTS ===")
        print(f"C=96: recovery of gap = {recov*100:.0f}% (CONFIRMED needs >=50%; FALSIFIED if <=20%)")
        print(f"      specfree within ~2% of AR? {'YES' if vs_ar<=2 else 'NO'} ({vs_ar:.2f}%)")
        if reg32 is not None:
            print(f"C=32: specfree vs force-0 = {reg32:+.2f}% (no regression if >= -2%)")
        if recov >= 0.5 and vs_ar <= 2 and (reg32 is None or reg32 >= -2):
            print("=> CONFIRMED")
        elif recov <= 0.2:
            print("=> FALSIFIED (mechanism does not remove the floor)")
        else:
            print("=> PARTIALLY CONFIRMED")
    else:
        print("\n=== VERDICT: incomplete data ===")


if __name__ == "__main__":
    main()
