#!/usr/bin/env python3
"""P3 aggregator: SPEED-Bench throughput_2k K x load matrix.

Same summary as analyze_p2.py plus a per-category (high_entropy / low_entropy / mixed)
breakdown — the difficulty tiers are a natural proxy for acceptance heterogeneity
(low_entropy = repetitive text = ngram-friendly; high_entropy = diverse = ngram-hostile).

Usage: P2_TAG=speedb_2k .venv/bin/python experiments/analyze_p3.py
"""
import json, os, re, statistics as st

BASE = "results/p2"
KS = ["none", "1", "2", "4", "8"]
CS = [32, 96]
TAG = os.environ.get("P2_TAG", "speedb_2k")


def parse_client(path):
    m = None
    with open(path) as f:
        for line in f:
            m = re.search(r"steady_out_tokens=(\d+) steady_throughput_tok_s=([\d.]+)", line)
    if not m:
        return None
    return int(m.group(1)), float(m.group(2))


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def load_records(K, C):
    recs = []
    for t in (1, 2, 3):
        p = f"/tmp/p2_k{K}_c{C}_t{t}_{TAG}.jsonl"
        if os.path.exists(p):
            try:
                rs = [json.loads(l) for l in open(p)]
                recs.extend(r for r in rs if "ttft_ms" in r and 20 <= r["t"] <= 150)
            except Exception:
                pass
    return recs


def main():
    print(f"== P3 {TAG}: throughput (tok/s, mean over trials)")
    print(f"{'K':>5} | " + " | ".join(f"C={C:<4}" for C in CS))
    thr = {}
    for K in KS:
        cells = []
        for C in CS:
            vals = []
            for t in (1, 2, 3):
                p = f"{BASE}/p2_k{K}_c{C}_t{t}_{TAG}_client.out"
                if os.path.exists(p):
                    r = parse_client(p)
                    if r:
                        vals.append(r[1])
            thr[(K, C)] = st.mean(vals) if vals else None
            cells.append(f"{st.mean(vals):>8.1f}±{st.pstdev(vals) if len(vals)>1 else 0:<4.1f}" if vals else "      --")
        print(f"{K:>5} | " + " | ".join(cells))

    for C in CS:
        rows = {K: float(rows_) for K, rows_ in ((K, thr[(K, C)]) for K in KS) if thr.get((K, C)) is not None}
        if rows:
            best, bv = max(rows.items(), key=lambda kv: kv[1])
            ar = rows.get("none")
            print(f"C={C}: argmax_K={best} ({bv:.1f}); AR={ar:.1f}" if ar else
                  f"C={C}: argmax_K={best} ({bv:.1f})")

    # per-category breakdown (pooled over trials)
    for C in CS:
        print(f"\n== P3 {TAG} C={C}: per-category TTFT p50 / TPOT p50 (ms), tok/req")
        print(f"{'K':>5} | {'high_ent TTFT/TPOT':>22} | {'low_ent TTFT/TPOT':>21} | {'mixed TTFT/TPOT':>20}")
        for K in KS:
            recs = load_records(K, C)
            if not recs:
                continue
            cells = []
            for cat in ("high_entropy", "low_entropy", "mixed"):
                sub = [r for r in recs if r.get("category") == cat]
                if len(sub) < 5:
                    cells.append(f"{'--':>22}")
                    continue
                tt = pct([r["ttft_ms"] for r in sub], 50)
                tp = pct([r["tpot_ms"] for r in sub], 50)
                nt = st.mean([r["ntok"] for r in sub])
                cells.append(f"{tt:>8.0f}/{tp:<6.1f} n={nt:.0f}")
            print(f"{K:>5} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
