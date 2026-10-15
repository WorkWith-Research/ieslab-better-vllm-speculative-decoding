#!/usr/bin/env python
# P5.2 H-5.2 P1 analysis: decompose the SD-off floor gap (AR vs force-0) per executed step into
#   cadence_ms = gpu_step_ms (GPU window) + residual_ms (all CPU-side work: scheduler spin that
#                produced this step's plan + model-runner input-batch/attention-metadata bookkeeping
#                sized for spec + enqueue).
# GATE G1: >=50% of the per-step cadence gap attributable to CPU-side spec bookkeeping (residual),
# i.e. NOT the GPU step. The drafter is skipped in both arms, so any residual difference is the
# static spec-path bookkeeping a spec-free off-state would remove.
import json, os, sys


def load(path):
    rows = [json.loads(l) for l in open(path)]
    return [r for r in rows if r.get("gpu_step_ms") is not None and r.get("cadence_ms") is not None]


def p50(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else float("nan")


def steady(rows, lo=0.1, hi=0.85):
    """Exclude ramp (first 10%) and tail; keep the stable middle where running is high."""
    rows = [r for r in sorted(rows, key=lambda r: r["t_cpu_sched"])]
    n = len(rows)
    if n < 30:
        return rows
    a, b = int(n * lo), int(n * hi)
    # within the window, keep steps that actually ran a full batch (running >= 5) to drop prefill-only tail
    return [r for r in rows[a:b] if r.get("running", 0) >= 5]


def summarize(tag, path):
    ss = steady(load(path))
    cad = [r["cadence_ms"] for r in ss]
    gstep = [r["gpu_step_ms"] for r in ss]
    resid = [r["residual_ms"] for r in ss if r.get("residual_ms") is not None]
    gfwd = [r["gpu_fwd_ms"] for r in ss if r.get("gpu_fwd_ms") is not None]
    return {
        "tag": tag, "n_ss": len(ss),
        "cad_p50": p50(cad), "gstep_p50": p50(gstep), "resid_p50": p50(resid), "gfwd_p50": p50(gfwd),
    }


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "results/p5_2"
    out = {}
    for C in [32, 96]:
        ar = os.path.join(base, f"p52_ar_c{C}_obs/steps.jsonl")
        f0 = os.path.join(base, f"p52_f0_c{C}_obs/steps.jsonl")
        if not (os.path.exists(ar) and os.path.exists(f0)):
            print(f"[skip] C={C}: missing {ar} or {f0}")
            continue
        A = summarize("AR", ar)
        F = summarize("FORCE0", f0)
        out[C] = (A, F)
        gap_cad = F["cad_p50"] - A["cad_p50"]
        gap_gstep = F["gstep_p50"] - A["gstep_p50"]
        gap_resid = F["resid_p50"] - A["resid_p50"]
        print(f"\n=== C={C} (per executed step, p50 ms) ===")
        print(f"{'':14}{'AR':>9}{'FORCE0':>9}{'delta(F-A)':>12}")
        for k in ["cad_p50", "gstep_p50", "gfwd_p50", "resid_p50"]:
            print(f"{k:14}{A[k]:9.3f}{F[k]:9.3f}{F[k]-A[k]:+12.3f}")
        print(f"\n  floor per-step cadence gap (FORCE0 - AR) = {gap_cad:+.3f} ms")
        if gap_cad:
            print(f"    -> GPU step window (gpu_step_ms)   : {gap_gstep:+.3f} ms ({100*gap_gstep/gap_cad:.0f}%)")
            print(f"    -> CPU-side residual (spec bookkeeping): {gap_resid:+.3f} ms ({100*gap_resid/gap_cad:.0f}%)")
        print(f"  GATE G1: CPU-side share of floor = {100*gap_resid/gap_cad if gap_cad else 0:.0f}%  (needs >=50%)")
    json.dump({str(k): v for k, v in out.items()}, open(os.path.join(base, "p52_p1_decomp.json"), "w"), indent=2)
    print(f"\nwrote {os.path.join(base,'p52_p1_decomp.json')}")


if __name__ == "__main__":
    main()
