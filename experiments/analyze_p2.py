#!/usr/bin/env python3
"""P2 aggregator: fixed-K x concurrency matrix.
Reads results/p2/*_client.out (latency_client summary line) + *_metrics.jsonl,
emits results/p2/summary.csv and the argmax_K table per load.
"""
import json, os, re, sys, statistics as st

BASE = "results/p2"
KS = ["none", "1", "2", "4", "8"]
CS = [8, 32, 96]

def parse_client(path):
    # last line: concurrency=C ok=N err=E steady_out_tokens=T steady_throughput_tok_s=X
    m = None
    with open(path) as f:
        for line in f:
            m = re.search(r"concurrency=(\d+) ok=(\d+) err=(\d+) steady_out_tokens=(\d+) steady_throughput_tok_s=([\d.]+)", line)
    if not m:
        return None
    C, ok, err, tok = (int(x) for x in m.groups()[:4])
    thr = float(m.group(5))
    return {"ok": ok, "err": err, "tok": tok, "thr": thr}

def parse_metrics(path):
    if not os.path.exists(path):
        return {}
    rows = [json.loads(l) for l in open(path)]
    def g(k):
        return [r[k] for r in rows if k in r and r[k] is not None]
    run = g("vllm:num_requests_running[gauge]")
    wait = g("vllm:num_requests_waiting[gauge]")
    kv = g("vllm:kv_cache_usage_perc[gauge]")
    gpu = g("gpu_util")
    acc = [r.get("vllm:spec_decode_num_accepted_tokens_d", 0) or 0 for r in rows]
    prop = [r.get("vllm:spec_decode_num_draft_tokens_proposed_d", 0) or 0 for r in rows]
    out = {}
    if run: out["running_p50"] = st.median(run)
    if wait: out["waiting_max"] = max(wait); out["waiting_p50"] = st.median(wait)
    if kv: out["kv_max"] = max(kv)
    if gpu: out["gpu_p50"] = round(st.median(gpu), 1)
    out["accepted"] = sum(acc); out["proposed"] = sum(prop)
    return out

def main():
    rows = []
    for K in KS:
        for C in CS:
            trials = []
            for t in (1, 2, 3):
                cl = f"{BASE}/p2_k{K}_c{C}_t{t}_mixed_client.out"
                mt = f"{BASE}/C{C}_k{K}_t{t}_metrics.jsonl"
                if not os.path.exists(cl):
                    continue
                c = parse_client(cl)
                m = parse_metrics(mt)
                if c is None:
                    continue
                trials.append({**c, **m})
            if not trials:
                rows.append({"K": K, "C": C, "n": 0})
                continue
            n = len(trials)
            thr = [x["thr"] for x in trials]
            row = {"K": K, "C": C, "n": n,
                   "thr_mean": round(st.mean(thr), 1), "thr_sd": round(st.stdev(thr), 1) if n > 1 else 0.0}
            for k in ("running_p50", "waiting_max", "kv_max", "gpu_p50", "accepted", "proposed"):
                vals = [x[k] for x in trials if k in x]
                row[k] = round(st.mean(vals), 3) if vals else None
            rows.append(row)

    os.makedirs(BASE, exist_ok=True)
    cols = ["K", "C", "n", "thr_mean", "thr_sd", "running_p50", "waiting_max", "kv_max", "gpu_p50", "accepted", "proposed"]
    with open(f"{BASE}/summary.csv", "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")

    # argmax table
    print(f"{'K':>5} | " + " | ".join(f"C={C:<3} thr±sd" for C in CS))
    by = {(r["K"], r["C"]): r for r in rows if r.get("n", 0) > 0}
    for K in KS:
        cells = []
        for C in CS:
            r = by.get((K, C))
            cells.append(f"{r['thr_mean']:>8.1f}±{r['thr_sd']:<4.1f}" if r else "     ----")
        print(f"{K:>5} | " + " | ".join(cells))
    for C in CS:
        best = max(((K, by[(K, C)]) for K in KS if (K, C) in by), key=lambda x: x[1]["thr_mean"])
        others = [(K, by[(K, C)]) for K in KS if K != best[0] and (K, C) in by]
        if others:
            runner = max(others, key=lambda x: x[1]["thr_mean"])
            ru = f", runner-up {runner[0]} ({runner[1]['thr_mean']:.1f})"
        else:
            ru = ""
        print(f"C={C}: argmax_K={best[0]} ({best[1]['thr_mean']:.1f}){ru}")

if __name__ == "__main__":
    main()
