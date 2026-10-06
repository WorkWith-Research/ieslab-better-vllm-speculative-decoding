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
        # Active (loaded) window: from the first sample where running >= 50% of
        # peak running, to end. Avoids the model-load idle prefix and the drain tail.
        run_series = [r.get("vllm:num_requests_running[gauge]", 0) for r in rows]
        peak = max(run_series) if run_series else 0
        lo = next((i for i, v in enumerate(run_series) if v >= max(1, peak / 2)), 0)
        win = rows[lo:] or rows
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
    import re as _re
    # Exclude archived/invalid dirs (leading underscore) and non-K sweep runs.
    all_dirs = [e for e in os.listdir(RESULTS)
                if os.path.isdir(os.path.join(RESULTS, e)) and not e.startswith("_")]
    k_runs = sorted([e for e in all_dirs if _re.fullmatch(r"K\d+_r\d+", e)])
    runs = [load_run(e) for e in k_runs]

    def parse(exp):
        m = _re.fullmatch(r"K(\d+)_r(\d+)", exp)
        return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

    # ---- K x load pivot: output throughput + mean accept len + TPOT ----
    ks = sorted({parse(e)[0] for e in k_runs})
    rates = sorted({parse(e)[1] for e in k_runs})
    def cell(k, r, field):
        for run in runs:
            if parse(run["exp"]) == (k, r) and run.get(field) is not None:
                return run[field]
        return None

    def fmt(v, nd=1):
        if v is None: return "-"
        if isinstance(v, float): return f"{v:.{nd}f}"
        return str(v)

    print("\n=== K x load: OUTPUT THROUGHPUT (tok/s) ===")
    hdr = ["rate\\K"] + [f"K={k}" for k in ks] + ["best-K"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    best_by_rate = {}
    for r in rates:
        row = [f"r{r}"]
        vals = {k: cell(k, r, "output_throughput") for k in ks}
        row += [fmt(vals[k]) for k in ks]
        valid = {k: float(v) for k, v in vals.items() if v is not None}
        bk = max(valid, key=lambda k: valid[k]) if valid else None
        best_by_rate[r] = bk
        row.append(f"K={bk}" if bk else "-")
        print("| " + " | ".join(row) + " |")

    print("\n=== K x load: MEAN ACCEPT LEN (tok/step) ===")
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for r in rates:
        row = [f"r{r}"] + [fmt(cell(k, r, "mean_accept_len"), 2) for k in ks] + [""]
        print("| " + " | ".join(row) + " |")

    print("\n=== K x load: MEAN TPOT (ms, lower=better) ===")
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for r in rates:
        row = [f"r{r}"] + [fmt(cell(k, r, "mean_tpot_ms")) for k in ks] + [""]
        print("| " + " | ".join(row) + " |")

    print("\n=== OPTIMUM-K SHIFT (throughput-maximizing K per load) ===")
    for r in rates:
        print(f"  rate={r:>2} req/s -> best K = {best_by_rate[r]}")

    # ---- flat table + csv (all runs) ----
    cols = ["exp", "completed", "duration", "output_throughput", "mean_tpot_ms",
            "p99_tpot_ms", "mean_ttft_ms", "p99_ttft_ms", "running_p50",
            "waiting_p50", "kv_util_max", "gpu_util_p50", "spec_decode_num_drafts",
            "mean_accept_len", "draft_acc_rate", "num_preemptions_total"]
    import csv as _csv
    with open(os.path.join(RESULTS, "summary.csv"), "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in runs:
            w.writerow(r)
    print("\n| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in runs:
        print("| " + " | ".join(fmt(r.get(c), 1 if c in ("duration",) else 2) for c in cols) + " |")
    print(f"\nwrote {os.path.join(RESULTS, 'summary.csv')} ({len(runs)} runs)")

if __name__ == "__main__":
    main()
