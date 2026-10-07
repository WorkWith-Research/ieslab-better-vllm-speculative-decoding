#!/usr/bin/env python3
"""Phase-2 oracle-gap analysis from per-request spec-decode hook data.

Inputs (per experiment dir):
  client.jsonl   : one line/request {idx, p, id, out_tokens, ...}
                   (p = designed repetition rate -> acceptance level)
  spec_hook.jsonl: one line/(request, decode-step) {t, req, K, acc}

What we can and cannot do
-------------------------
vLLM verifies all K drafted tokens in a SINGLE target forward pass, so the
wall-clock cost of a verification step grows only sub-linearly with K (and is
amortized across the batch). We therefore CANNOT convert a request's acceptance
profile into an exact throughput oracle gap without a calibrated per-step cost
curve c(K). What we CAN do from the data alone:

  (1) HETEROGENEITY (cost-model-free): each request's "natural K" = mean accept
      length L_r = 1 + sum_{l>=1} J[l]. A wide spread of L_r across requests in
      one batch is direct evidence that a single fixed-K policy simultaneously
      over-speculates some requests and under-speculates others.

  (2) TRUNCATION HEADROOM (cost-model-free lower bound on the oracle gap):
      full potential P = sum_{l=0..Kmax} J[l] accepted+bonus tokens/step; a fixed
      K captures C(K)=sum_{l=0..K} J[l]. Loss(K) = 1 - C(K)/P is the fraction of
      a request's achievable acceptance that truncating at K wastes. Averaged over
      requests this is a lower bound on what per-request dynamic-K could recover
      (it ignores the over-speculation cost side, so it is a floor).

  (3) COST-MODEL SENSITIVITY: bracket the % oracle gap under three plausible
      per-step cost curves c(K) in {linear, sqrt, constant} to show how much of
      the headroom is realizable. Clearly labeled model-dependent.

Usage:
  .venv/bin/python experiments/oracle_gap.py results/P2_K8_r6 --kmax 8
"""
import argparse, json, glob, os
from collections import defaultdict


def load(dirpath):
    clients = {}
    cp = glob.glob(os.path.join(dirpath, "client.jsonl"))
    if cp:
        for line in open(cp[0]):
            r = json.loads(line)
            if r.get("id"):
                clients[r["id"]] = r
    steps = defaultdict(list)  # req -> [acc, acc, ...]
    hp = glob.glob(os.path.join(dirpath, "spec_hook.jsonl"))
    if hp:
        for line in open(hp[0]):
            r = json.loads(line)
            steps[r["req"]].append(r["acc"])
    return clients, steps


def profile(accs, kmax):
    """J[l] = frac(steps with acc>=l), l=0..kmax. None if too few drafting steps."""
    n = len(accs)
    if n < 10:
        return None
    J = [1.0] * (kmax + 1)
    for l in range(1, kmax + 1):
        J[l] = sum(1 for a in accs if a >= l) / n
    return J


def captured(J, k):
    """Accepted+bonus tokens per verification step at speculation length k."""
    return sum(J[:k + 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--kmax", type=int, default=8)
    a = ap.parse_args()

    clients, steps = load(a.dir)

    # hook req_id = "<client completion id>-<engine>-<hash>"; join by prefix.
    def class_of(req):
        for cid, r in clients.items():
            if req == cid or req.startswith(cid + "-"):
                return r.get("p", r.get("class", "?"))
        return "?"

    ks = list(range(1, a.kmax + 1))
    reqs = []
    for rid, accs in steps.items():
        J = profile(accs, a.kmax)
        if not J:
            continue
        L = 1.0 + sum(J[1:])              # natural K (mean accept length at Kmax)
        P = captured(J, a.kmax)           # full potential tokens/step
        reqs.append({"id": rid, "cls": class_of(rid), "J": J, "L": L, "P": P,
                     "nsteps": len(accs)})

    if not reqs:
        print("no per-request data (need >=10 drafting steps/request)"); return

    n = len(reqs)
    print(f"requests with profile: {n}  (Kmax={a.kmax})")

    # ---- (1) heterogeneity: natural-K spread, overall and by acceptance level ----
    Ls = sorted(r["L"] for r in reqs)
    def pct(q): return Ls[min(n - 1, int(q * n))]
    mean_L = sum(Ls) / n
    var = sum((x - mean_L) ** 2 for x in Ls) / n
    cv = (var ** 0.5) / mean_L
    print("\n=== (1) NATURAL-K (mean accept length) HETEROGENEITY ===")
    print(f"  min={Ls[0]:.2f}  p25={pct(.25):.2f}  median={pct(.5):.2f}  "
          f"p75={pct(.75):.2f}  max={Ls[-1]:.2f}")
    print(f"  mean={mean_L:.2f}  std={var**0.5:.2f}  CV={cv:.2f}   "
          f"(CV>~0.3 => requests want meaningfully different K)")

    by_cls = defaultdict(list)
    for r in reqs:
        by_cls[r["cls"]].append(r)
    print("\n  per acceptance-level (p): natural-K mean | J[1] pos0-accept")
    for c, rs in sorted(by_cls.items(), key=lambda kv: str(kv[0])):
        mL = sum(x["L"] for x in rs) / len(rs)
        mJ1 = sum(x["J"][1] for x in rs) / len(rs)
        print(f"    p={c}: n={len(rs):>2}  natural-K={mL:.2f}   J[1]={mJ1:.3f}")

    # ---- (2) truncation headroom: lower bound on oracle gap ----
    print("\n=== (2) TRUNCATION HEADROOM (cost-model-free, floor on oracle gap) ===")
    print("  fixed K | mean captured/step | mean loss vs full potential")
    for k in ks:
        cap = sum(captured(r["J"], k) / r["P"] for r in reqs) / n   # frac of potential
        print(f"    K={k:<2} | {sum(captured(r['J'], k) for r in reqs)/n:.4f} "
              f"| {100*(1-cap):+.2f}%  (acceptance potential left unused)")

    # ---- (3) cost-model sensitivity: % oracle gap under c(K) curves ----
    print("\n=== (3) ORACLE GAP by per-step cost model c(K) [model-dependent] ===")
    print("  score_r(k) = captured_r(k)/c(k); oracle = mean_r max_k score_r(k)")
    models = {"linear": lambda k: 1 + k, "sqrt": lambda k: 1 + (k ** 0.5),
              "constant": lambda k: 1.0}
    for name, c in models.items():
        oracle = sum(max(captured(r["J"], k) / c(k) for k in ks) for r in reqs) / n
        fixed = {k: sum(captured(r["J"], k) / c(k) for r in reqs) / n for k in ks}
        best_k = max(fixed.keys(), key=lambda k: fixed[k])
        gap = oracle / fixed[best_k] - 1
        print(f"    {name:<9}: best fixed K={best_k} -> per-request oracle gain "
              f"{100*gap:+.2f}%")

    # save per-request table
    out = os.path.join(a.dir, "oracle_per_request.jsonl")
    with open(out, "w") as f:
        for r in reqs:
            f.write(json.dumps({"id": r["id"], "p": r["cls"], "natural_K": round(r["L"], 3),
                                "J": [round(x, 3) for x in r["J"]],
                                "nsteps": r["nsteps"]}) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
