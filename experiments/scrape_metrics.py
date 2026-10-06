#!/usr/bin/env python3
"""Generic vLLM /metrics + nvidia-smi scraper -> per-run JSONL.

Parses Prometheus text format directly (robust to metric renames across vLLM
versions): sums _total counters, histogram _sum/_count, gauges as-is.

Usage: scrape_metrics.py --port 8100 --gpu 0 --out results/<exp>/metrics.jsonl --interval 0.5
Stops when /health stops answering (server killed) or Ctrl-C; prints a summary.
"""
import argparse, json, statistics, subprocess, time, urllib.request

def fetch(url, timeout=5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read().decode()
    except Exception:
        return None, None

def parse_prom(text):
    """Return {base_name: value} summing across label sets per series family."""
    out = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        raw, val = parts
        try:
            v = float(val)
        except ValueError:
            continue
        # strip labels
        name = raw.split("{")[0]
        if not name.startswith("vllm:"):
            continue
        if name.endswith("_total"):
            base, kind = name[:-6], "counter"
        elif name.endswith("_sum"):
            base, kind = name[:-4], "h_sum"
        elif name.endswith("_count"):
            base, kind = name[:-6], "h_count"
        elif "_bucket" in name:
            continue
        else:
            base, kind = name, "gauge"
        key = f"{base}[{kind}]"
        out[key] = out.get(key, 0.0) + v
    return out

def gpu_state(gpu):
    try:
        o = subprocess.run(
            ["nvidia-smi", f"--id={gpu}",
             "--query-gpu=utilization.gpu,memory.used,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip()
        u, m, p = (float(x) for x in o.split(","))
        return {"gpu_util": u, "gpu_mem_mib": m, "gpu_power_w": p}
    except Exception:
        return {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=0.5)
    args = ap.parse_args()

    t0 = time.time()
    rows, prev = [], {}
    with open(args.out, "w") as f:
        while True:
            st, body = fetch(f"http://127.0.0.1:{args.port}/metrics")
            if st is None:
                hs, _ = fetch(f"http://127.0.0.1:{args.port}/health", timeout=2)
                if hs != 200:
                    break
            now = parse_prom(body) if body else {}
            row = {"t": round(time.time() - t0, 3)}
            for k, v in now.items():
                row[k] = round(v, 4)
            row.update(gpu_state(args.gpu))
            # per-interval deltas of counters (rates * interval)
            for k, v in now.items():
                if k.endswith("[counter]") and k in prev:
                    row[k.replace("[counter]", "_d")] = round(v - prev[k], 4)
            f.write(json.dumps(row) + "\n")
            rows.append(row)
            prev = {k: v for k, v in now.items() if isinstance(v, float)}
            time.sleep(args.interval)

    def g(k): return [r[k] for r in rows if k in r]
    last = rows[-1] if rows else {}
    print(json.dumps({
        "n_samples": len(rows),
        "duration_s": round(time.time() - t0, 1),
        "waiting_p50": statistics.median(g("vllm:num_requests_waiting[gauge]")) if g("vllm:num_requests_waiting[gauge]") else None,
        "running_p50": statistics.median(g("vllm:num_requests_running[gauge]")) if g("vllm:num_requests_running[gauge]") else None,
        "kv_util_max": max(g("vllm:kv_cache_usage_perc[gauge]")) if g("vllm:kv_cache_usage_perc[gauge]") else None,
        "gpu_util_p50": statistics.median(g("gpu_util")) if g("gpu_util") else None,
        "preemptions_total": last.get("vllm:num_preemptions[counter]"),
        "drafts_total": last.get("vllm:spec_decode_num_drafts[counter]"),
        "draft_tokens_total": last.get("vllm:spec_decode_num_draft_tokens[counter]"),
        "accepted_tokens_total": last.get("vllm:spec_decode_num_accepted_tokens[counter]"),
    }))

if __name__ == "__main__":
    main()
