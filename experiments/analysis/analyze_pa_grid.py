"""Analyze Priority-A grid results (results/pa_grid/trials.jsonl).

Prints a per-config table (throughput, TTFT/TPOT percentiles, GPU util, chosen-K dist),
computes trial variability, and classifies H1 (adaptive K >= best fixed K) using the
pre-registered rule: SUPPORT if LDM mean > best-fixed mean + max(stdevs); FALSIFY if
LDM mean < best-fixed mean - max(stdevs); else INCONCLUSIVE. Also flags AR as best
(speculation net-negative).

Note on variability: fixed seed + greedy decoding makes runs highly deterministic;
near-zero trial spread is expected and reported as such (see PROGRESS.md methodology note).
"""
import json, statistics as st, sys

path = sys.argv[1] if len(sys.argv) > 1 else "results/pa_grid/trials.jsonl"
rows = [json.loads(l) for l in open(path)]
cfgs = ["ar", "k1", "k2", "k4", "k8", "ldm"]

def mean(xs): return st.mean(xs) if xs else None
def sd(xs): return st.pstdev(xs) if len(xs) > 1 else 0.0

by_cfg = {c: [r for r in rows if r.get("config") == c] for c in cfgs}

print(f"{'cfg':6} {'n':2} {'tok/s mean±sd':>14} {'TTFT p50/p95':>14} {'TPOT p50/p95 ms':>16} "
      f"{'E2E p95 s':>10} {'GPU%':>5}  k* dist")
summary = {}
for c in cfgs:
    rs = [r for r in by_cfg[c] if r.get("throughput_tok_s")]
    if not rs:
        print(f"{c:6} MISSING/INVALID trials: {[r.get('error') or 'no-throughput' for r in by_cfg[c]]}")
        continue
    thr = [r["throughput_tok_s"] for r in rs]
    tt50 = mean([r["ttft_ms"]["p50"] for r in rs if r["ttft_ms"]["p50"] is not None])
    tt95 = mean([r["ttft_ms"]["p95"] for r in rs if r["ttft_ms"]["p95"] is not None])
    tp50 = mean([r["tpot_ms"]["p50"] for r in rs if r["tpot_ms"]["p50"] is not None])
    tp95 = mean([r["tpot_ms"]["p95"] for r in rs if r["tpot_ms"]["p95"] is not None])
    e95 = mean([r["e2e_s"]["p95"] for r in rs if r["e2e_s"]["p95"] is not None])
    gpu = mean([r["gpu_util_mean"] for r in rs if r.get("gpu_util_mean") is not None])
    kd = next((r["kstar_dist"] for r in rs if r.get("kstar_dist")), None)
    summary[c] = (mean(thr), sd(thr))
    print(f"{c:6} {len(rs):2d} {mean(thr):8.1f}±{sd(thr):5.1f} "
          f"{tt50 or 0:7.0f}/{tt95 or 0:6.0f} {tp50 or 0:8.1f}/{tp95 or 0:7.1f} "
          f"{e95 or 0:10.2f} {gpu or 0:5.1f}  {kd}")

# classification
fixed = {c: v for c, v in summary.items() if c.startswith("k")}
if not fixed or "ldm" not in summary:
    print("\nCannot classify (missing arms).")
    sys.exit(0)
best_c, (best_m, best_s) = max(fixed.items(), key=lambda kv: kv[1][0])
ar_ok = "ar" in summary
if ar_ok and summary["ar"][0] >= best_m:
    print(f"\nH1 FALSIFIED (stronger): AR ({summary['ar'][0]:.1f}) >= best fixed K={best_c} "
          f"({best_m:.1f}) -> speculation is net-NEGATIVE on this workload.")
else:
    ldm_m, ldm_s = summary["ldm"]
    margin = max(ldm_s, best_s)
    if ldm_m > best_m + margin:
        cls = f"SUPPORT: LDM {ldm_m:.1f} > best fixed K={best_c} {best_m:.1f} (margin {ldm_m-best_m:+.1f} > uncertainty {margin:.1f})"
    elif ldm_m < best_m - margin:
        cls = f"FALSIFY: LDM {ldm_m:.1f} < best fixed K={best_c} {best_m:.1f} (gap {best_m-ldm_m:+.1f} > uncertainty {margin:.1f})"
    else:
        cls = f"INCONCLUSIVE: LDM {ldm_m:.1f} vs best fixed K={best_c} {best_m:.1f} (gap {ldm_m-best_m:+.1f} within uncertainty {margin:.1f})"
    print("\nH1:", cls)
