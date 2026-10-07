#!/usr/bin/env python3
"""Profile SPS(B) — steps/sec vs total verification batch size B — from P2 metrics.

DSpark's 'load model' is a static profiled curve SPS(B). We reconstruct the closest
honest proxy on our hardware: for each P2 cell (K, C), B = C*(1+K) (decode steps in
steady state verify 1+K tokens per active request), and SPS(B) = decode steps/sec
measured from the spec-decode token counters (proposed tokens / (1+K) per second).

Usage: .venv/bin/python experiments/profile_sps.py [--base results/p2] [--out results/p2/sps_table.jsonl]
"""
import json, os, re, sys

BASE = "results/p2"
KS = ["none", "1", "2", "4", "8"]
CS = [8, 32, 96]


def parse_client(path):
    m = None
    with open(path) as f:
        for line in f:
            m = re.search(r"steady_out_tokens=(\d+) steady_throughput_tok_s=([\d.]+)", line)
    if not m:
        return None
    return int(m.group(1)), float(m.group(2))


def main():
    base = sys.argv[sys.argv.index("--base") + 1] if "--base" in sys.argv else BASE
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else os.path.join(base, "sps_table.jsonl")
    rows = []
    for K in KS:
        for C in CS:
            pts = []
            for t in (1, 2, 3):
                cl = f"{base}/p2_k{K}_c{C}_t{t}_mixed_client.out"
                mt_new = f"{base}/p2_k{K}_c{C}_t{t}_mixed_metrics.jsonl"
                mt_old = f"{base}/C{C}_k{K}_t{t}_metrics.jsonl"
                mt = mt_new if os.path.exists(mt_new) else mt_old
                if not (os.path.exists(cl) and os.path.exists(mt)):
                    continue
                tok, thr = parse_client(cl)
                if tok is None:
                    continue
                mrows = [json.loads(l) for l in open(mt)]
                if not mrows or "t" not in mrows[0]:
                    continue
                # steady-state window: drop first 20s of samples (warmup ~20s @ 0.5s interval = 40 rows)
                mr = mrows[40:] if len(mrows) > 80 else mrows
                t0 = mr[0]["t"]; t1 = mr[-1]["t"]
                dt = max(1e-9, t1 - t0)
                k_eff = 0 if K == "none" else int(K)
                prop = sum(r.get("vllm:spec_decode_num_draft_tokens_proposed_d", 0) or 0 for r in mr)
                acc = sum(r.get("vllm:spec_decode_num_accepted_tokens_d", 0) or 0 for r in mr)
                if k_eff == 0:
                    # AR: steps/sec = output tokens / sec (1 token per step per request batch)
                    sps = thr / C
                    B = C
                else:
                    steps = prop / (k_eff + 1) if prop else None
                    if steps is None or steps <= 0:
                        continue
                    sps = steps / dt
                    B = C * (k_eff + 1)
                pts.append({"B": B, "SPS": round(sps, 2), "tok_s": thr, "acc_rate": round(acc / prop, 3) if prop else None})
            if pts:
                rows.append({"K": K, "C": C, "points": pts})
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{'K':>5} {'C':>4} {'B':>6} {'SPS':>8} {'tok/s':>8} {'acc_rate':>9}")
    for r in rows:
        for p in r["points"]:
            print(f"{r['K']:>5} {r['C']:>4} {p['B']:>6} {p['SPS']:>8.2f} {p['tok_s']:>8.1f} {str(p['acc_rate']):>9}")
    print(f"\nwrote {out} ({len(rows)} cells)")


if __name__ == "__main__":
    main()
