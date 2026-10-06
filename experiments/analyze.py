#!/usr/bin/env python3
"""Aggregate Phase-1 experiment artifacts into a summary table.

Reads results/<exp>/ for each run: <exp>.bench.json (vllm bench serve output) +
metrics.jsonl (0.5s sampler). Emits one row per experiment with the metrics we
care about, sorted by name. Writes results/summary.csv and prints a markdown table.

Key derived metrics:
  - mean_accept_len = 1 + accepted/drafts   (tokens produced per decode step)
  - draft_acc_rate  = accepted / draft_tokens
  - steady-state window = middle 50% of the run by time (skips warmup + drain)
"""
import json, os, sys, glob

RESULTS = "results"

def load_run(exp):
    d = os.path.join(RESULTS, exp)
    bench_path = glob.glob(os.path.join(d, "*.bench.json"))
    metrics_path = os.path.join(d, "metrics.jsonl")
    out = {"exp": exp}
    if bench_path:
        b = json.load(open(bench_path[0]))
        for k in ["completed", "duration", "request_throughput", "output_throughput",
                  "total_token_throughput", "mean_ttft_ms", "p99_ttft_ms",
                  "mean_tpot_ms", "p99_tpot_ms", "mean_e2el_ms"]:
            out[k] = b.get(k)
    if os.path.exists(metrics_path):
        rows = [json.loads(l) for l in open(metrics_path)]
        n = len(rows)
        # steady-state window: middle 50% of samples
        lo, hi = n // 4, 3 * n // 4
        win = rows[lo:hi] or rows
        def col(name): return [r[name] for r in win if name in r]
        running = col("vllm:num_requests_running[gauge]")
        waiting = col("vllm:num_requests_waiting[gauge]")
        kv = col("vllm:kv_cache_usage_perc[gauge]")
        gpu = col("gpu_util")
        out["running_p50"] = sorted(running)[len(running)//2] if running else None
        out["running_max"] = max(running) if running else None
        out["waiting_p50"] = sorted(waiting)[len(waiting)//2] if waiting else None
        out["waiting_max"] = max(waiting) if waiting else None
        out["kv_util_p50"] = sorted(kv)[len(kv)//2] if kv else None
        out["kv_util_max"] = max(kv) if kv else None
        out["gpu_util_p50"] = sorted(gpu)[len(gpu)//2] if gpu else None
        last = rows[-1] if rows else {}
        for k in ["vllm:spec_decode_num_drafts[counter]",
                  "vllm:spec_decode_num_draft_tokens[counter]",
                  "vllm:spec_decode_num_accepted_tokens[counter]",
                  "vllm:num_preemptions_total[gauge]"]:
            out[k.replace("vllm:", "").replace("[counter]", "").replace("[gauge]", "")] = last.get(k)
        # derived acceptance metrics
        drafts = out.get("spec_decode_num_drafts") or 0
        dtokens = out.get("spec_decode_num_draft_tokens") or 0
        accepted = out.get("spec_decode_num_accepted_tokens") or 0
        if drafts:
            out["mean_accept_len"] = round(1 + accepted / drafts, 3)
            out["draft_acc_rate"] = round(accepted / dtokens, 3) if dtokens else None
    return out

def main():
    exps = sorted([e for e in os.listdir(RESULTS)
                   if os.path.isdir(os.path.join(RESULTS, e))])
    runs = [load_run(e) for e in exps]
    # column order
    cols = ["exp", "completed", "duration", "output_throughput", "mean_tpot_ms",
            "p99_tpot_ms", "mean_ttft_ms", "p99_ttft_ms", "running_p50",
            "waiting_p50", "kv_util_max", "gpu_util_p50", "spec_decode_num_drafts",
            "mean_accept_len", "draft_acc_rate", "num_preemptions_total"]
    # csv
    import csv as _csv
    with open(os.path.join(RESULTS, "summary.csv"), "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in runs:
            w.writerow(r)
    # markdown
    def fmt(v, nd=2):
        if v is None: return "-"
        if isinstance(v, float): return f"{v:.{nd}f}"
        return str(v)
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in runs:
        print("| " + " | ".join(fmt(r.get(c), 1 if c in ("duration",) else 2) for c in cols) + " |")
    print(f"\nwrote {os.path.join(RESULTS, 'summary.csv')} ({len(runs)} runs)")

if __name__ == "__main__":
    main()
