#!/usr/bin/env python3
"""Phase 4.6 rev-#4c — SERVER-SIDE analysis (trustworthy cross-arm metric).

Why server-side: the client `steady_throughput_tok_s` = completions-in-window x 256 tokens.
With open-loop workers, requests still in-flight at window end are excluded, so it undercounts
by an amount that depends on each arm's request latency — a per-arm bias. The vLLM
generation_tokens COUNTER rate (steady window t>=20s) is the fair cross-arm throughput.

Arms: AR (spec disabled, Phase 4.2), ldm_load (rev-#4c), force-0 (SD-off floor, spec enabled).
Pre-registered bounds (PROGRESS.md Priority B): C=96 ldm_load>=AR-noise & beats DSpark; C=8 >=+5% vs AR.

Usage: .venv/bin/python experiments/analyze_p46_server.py
"""
import json, os, statistics

P2 = "results/p2"
P46 = "results/p4_6"


def srv(fn):
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
    draft = (c(b, "vllm:spec_decode_num_draft_tokens[counter]") - c(a, "vllm:spec_decode_num_draft_tokens[counter]")) / dt
    return {"gen": gen, "draft": draft}


def dec_state(fn):
    if not os.path.exists(fn):
        return None
    dec = [json.loads(l) for l in open(fn)]
    st = dec[int(len(dec) * 0.3):int(len(dec) * 0.95)]
    mk, fk0 = [], []
    for r in st:
        ks = [d["kstar"] for d in r["decisions"]]
        if not ks:
            continue
        mk.append(sum(ks) / len(ks))
        fk0.append(sum(1 for k in ks if k == 0) / len(ks))
    return {"meanK": statistics.mean(mk), "fracK0": statistics.mean(fk0)} if mk else None


def main():
    print("=" * 84)
    print("Phase 4.6 rev-#4c — server-side gen/s (steady t>=20s) + decision state")
    print("=" * 84)
    ar = {}
    for C in [8, 32, 96]:
        vals = [srv(f"{P2}/p2_knone_c{C}_t{t}_mixed_metrics.jsonl") for t in [1, 2, 3]]
        vals = [v["gen"] for v in vals if v]
        ar[C] = statistics.mean(vals) if vals else None

    print(f"\n{'arm':<18} {'C':>3} | {'gen/s':>8} {'draft/s':>9} {'meanK':>6} {'fracK0':>7} | vs AR")
    print("-" * 84)
    for C in [8, 32, 96]:
        a = ar[C]
        if a:
            print(f"{'AR (spec off)':<18} {C:>3} | {a:8.1f} {'':>9} {'':>6} {'':>7} |   base")
        for t in range(3):
            m = srv(f"{P46}/p46_ldmload_c{C}_t{t}_mixed_metrics.jsonl")
            d = dec_state(f"{P46}/p46_ldmload_c{C}_t{t}_mixed_decisions.jsonl")
            if not m:
                continue
            vs = f"{(m['gen']/a-1)*100:+5.1f}%" if a else "  n/a "
            print(f"{'ldm_load':<18} {C:>3} t{t} | {m['gen']:8.1f} {m['draft']:9.1f} "
                  f"{d['meanK'] if d else 0:6.2f} {d['fracK0'] if d else 0:7.1%} | {vs}")
        f0 = srv(f"{P46}/p46_ldmload_c{C}_tf0_mixed_metrics.jsonl")
        if f0:
            vs = f"{(f0['gen']/a-1)*100:+5.1f}%" if a else "  n/a "
            print(f"{'force0(SD-off)':<18} {C:>3}    | {f0['gen']:8.1f} {f0['draft']:9.1f} {'':>6} {'':>7} | {vs}")
        print("-" * 84)

    # classification (server-side)
    print("CLASSIFICATION vs pre-registered bounds (server-side gen/s):")
    for C, desc in [(96, "ldm_load >= AR - noise (~0.3%) AND beats DSpark-rule"),
                    (8, "ldm_load >= +5% vs AR")]:
        vals = [srv(f"{P46}/p46_ldmload_c{C}_t{t}_mixed_metrics.jsonl") for t in range(3)]
        vals = [v["gen"] for v in vals if v]
        a = ar[C]
        if not vals or not a:
            print(f"  C={C}: INCONCLUSIVE (missing cells/baseline)")
            continue
        mean = statistics.mean(vals)
        noise = max(0.003 * a, 5)
        ok = (mean >= a - noise) if C == 96 else (mean >= a * 1.05)
        print(f"  C={C}: ldm_load mean {mean:.1f} vs AR {a:.1f} ({(mean/a-1)*100:+.1f}%) "
              f"-> {'PASS' if ok else 'FAIL'}   [{desc}]")


if __name__ == "__main__":
    main()
