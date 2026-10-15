#!/usr/bin/env python3
"""P4.5R — corrected DSpark-rule proxy vs fixed baselines (server-side gen/s).

Prereg: docs/dspark-proxy-validity.md §6. Metric = vLLM generation_tokens counter
rate over the steady window t>=20s (same as Phase 4.6 server-side analysis; the
client steady_throughput_tok_s is open-loop biased and was superseded).

Arms:
  dsparkr (this run): results/p4_5r/p45r_dsparkr_c{C}_t{T}_mixed_metrics.jsonl
  AR / K8 baselines : results/p2/p2_k{none,8}_c{C}_t{T}_mixed_metrics.jsonl (reused)

Hypothesis: with correct B accounting the rule sees B~96-130 at C=96 and commits
toward low-K/SD-off -> lands BETWEEN fixed-K8 and AR.
Falsification: within noise of AR at C=96 AND >= best fixed-K.

Usage: .venv/bin/python experiments/analyze_p45r.py
"""
import json
import os
import statistics as st

P2 = "results/p2"
P45R = "results/p4_5r"


def srv_gen(fn):
    """Server-side gen tokens/s over steady window t>=20s."""
    if not os.path.exists(fn):
        return None
    rows = [json.loads(l) for l in open(fn)]
    mr = [r for r in rows if r.get("t", 0) >= 20]
    if len(mr) < 10:
        return None
    a, b = mr[0], mr[-1]
    dt = max(1e-9, b["t"] - a["t"])

    def c(r, k):
        return r.get(k, 0) or 0

    gen = (c(b, "vllm:generation_tokens[counter]") - c(a, "vllm:generation_tokens[counter]")) / dt
    draft = (c(b, "vllm:spec_decode_num_draft_tokens[counter]") -
             c(a, "vllm:spec_decode_num_draft_tokens[counter]")) / dt
    return {"gen": gen, "draft": draft}


def dec_state(fn):
    """Decision-state stats over the middle 60% of the decision log."""
    if not os.path.exists(fn):
        return None
    dec = [json.loads(l) for l in open(fn)]
    st_ = dec[int(len(dec) * 0.2):int(len(dec) * 0.95)]
    mk, fk0, bl, nd_n = [], [], [], []
    for r in st_:
        if r.get("meanK") is not None:
            mk.append(r["meanK"])
        if r.get("fracK0") is not None:
            fk0.append(r["fracK0"])
        bl.append(r.get("B_live", 0))
        nd_n.append(r.get("n_decided", 0) / max(1, r.get("N", 0)))
    if not mk:
        return None
    return {"meanK": st.mean(mk), "fracK0": st.mean(fk0),
            "B_live": st.mean(bl), "decided_frac": st.mean(nd_n)}


def main():
    print("=" * 84)
    print("P4.5R — corrected DSpark-rule proxy: server-side gen/s (steady t>=20s)")
    print("=" * 84)
    arms = {}
    for C in [32, 96]:
        ds = []
        for T in [1, 2, 3]:
            v = srv_gen(f"{P45R}/p45r_dsparkr_c{C}_t{T}_mixed_metrics.jsonl")
            if v:
                ds.append(v["gen"])
        arms[("dsparkr", C)] = ds
        for name, key in [("AR", "none"), ("K8", "8")]:
            vals = []
            for T in [1, 2, 3]:
                v = srv_gen(f"{P2}/p2_k{key}_c{C}_t{T}_mixed_metrics.jsonl")
                if v:
                    vals.append(v["gen"])
            arms[(name, C)] = vals

    print(f"{'arm':<10} {'C=32 trials':>14} {'C=96 trials':>14}")
    for name in ["AR", "K8", "dsparkr"]:
        c32 = arms.get((name, 32), [])
        c96 = arms.get((name, 96), [])
        f32 = "/".join(f"{x:.0f}" for x in c32) if c32 else "MISSING"
        f96 = "/".join(f"{x:.0f}" for x in c96) if c96 else "MISSING"
        print(f"{name:<10} {f32:>14} {f96:>14}")

    print()
    print("MEAN + verdict vs prereg (docs/dspark-proxy-validity.md §6):")
    for C in [32, 96]:
        ar = st.mean(arms.get(("AR", C), [0])) or None
        k8 = st.mean(arms.get(("K8", C), [0])) or None
        ds = st.mean(arms.get(("dsparkr", C), [0])) or None
        if not (ar and k8 and ds):
            print(f"C={C}: MISSING data (AR={ar} K8={k8} dsparkr={ds})")
            continue
        best_fixed = max(k8, ar)  # AR is the no-drafter fixed baseline; K8 the spec one
        print(f"C={C}: AR={ar:.1f}  K8={k8:.1f} ({100*(k8/ar-1):+.1f}% vs AR)  "
              f"dsparkr={ds:.1f} ({100*(ds/ar-1):+.1f}% vs AR, {100*(ds/k8-1):+.1f}% vs K8)")
        between = (min(ar, k8) <= ds <= max(ar, k8)) if C == 96 else None
        falsified = (abs(ds - ar) / ar < 0.02 and ds >= best_fixed) if C == 96 else None
        if C == 96:
            print(f"    hypothesis (between K8 and AR): {'YES' if between else 'NO'}")
            print(f"    falsification (within noise of AR AND >= best fixed-K): "
                  f"{'TRIGGERED' if falsified else 'not triggered'}")

    print()
    print("Decision state (dsparkr, steady middle 75% of log):")
    for C in [32, 96]:
        vals = []
        for T in [1, 2, 3]:
            s = dec_state(f"{P45R}/p45r_dsparkr_c{C}_t{T}_mixed_decisions.jsonl")
            if s:
                vals.append(s)
        if not vals:
            print(f"C={C}: MISSING decision logs")
            continue
        mk = st.mean(v["meanK"] for v in vals)
        fk0 = st.mean(v["fracK0"] for v in vals)
        bl = st.mean(v["B_live"] for v in vals)
        df = st.mean(v["decided_frac"] for v in vals)
        print(f"C={C}: meanK={mk:.2f}  fracK0={fk0:.3f}  B_live={bl:.1f}  "
              f"decided/N={df:.3f}")


if __name__ == "__main__":
    main()
