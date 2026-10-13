#!/usr/bin/env python3
"""H-CLASS-1 analysis (prereg docs/hclass1-prereg.md §7).

Reads results/hclass1/<cell>/ artifacts and produces:
  - per-class DSpark/AR TPOT ratios with trial-level + request-paired bootstrap CIs
  - H-C1 / H-C2 / H-C3 verdict inputs
  - aggregate throughput, TTFT/E2E percentiles, acceptance, drafter share, cache hit rate

Usage: hclass1_analyze.py [results_dir]   (default results/hclass1)
"""
import json, math, os, re, sys
from collections import defaultdict

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "results/hclass1"
WARMUP = 20.0          # client-side warmup exclusion (s from run start)
TAIL_FRAC = 0.10       # final 10% of steps excluded (shutdown tail)

# Per-condition class schemes (prereg §4, deviations D1+D2): A is 2 ISL classes with a
# uniform cap; B reproduces H-OBS-3's 4-class ISL x configured-horizon grid.
SCHEME = {
    "A": {"classes": ["S", "L"], "maxtok": {"S": 2048, "L": 2048},
          "isl": {"S": 50, "L": 2048}},
    "B": {"classes": ["SS", "SL", "LS", "LL"],
          "maxtok": {"SS": 256, "SL": 1024, "LS": 256, "LL": 1024},
          "isl": {"SS": 50, "SL": 50, "LS": 2048, "LL": 2048}},
    # Condition C (follow-up, docs/hclass1c-prereg.md): diverse distinct queries + ONE shared cached
    # context doc (system msg), prefix caching ON. Same 2 ISL classes as A but realized ISL = shared
    # doc (~552 tok) + query; uniform output cap 2048.
    "C": {"classes": ["S", "L"], "maxtok": {"S": 2048, "L": 2048},
          "isl": {"S": 600, "L": 2600}},
}
# Pressure levels: filled by calibration (prereg §5b); overridable via HCLASS1_C_LEVELS.
C_LEVELS = ([int(x) for x in os.environ.get("HCLASS1_C_LEVELS", "").split(",") if x.strip()]
            or [8, 32, 96])

def cells_matching(prefix):
    out = []
    if not os.path.isdir(RESULTS):
        return out
    for d in sorted(os.listdir(RESULTS)):
        if d.startswith(prefix) and os.path.exists(os.path.join(RESULTS, d, "client.jsonl")):
            out.append(d)
    return out

def load_client(cell):
    p = os.path.join(RESULTS, cell, "client.jsonl")
    recs = [json.loads(l) for l in open(p) if l.strip()]
    ok = [r for r in recs if "ttft_ms" in r]
    return recs, ok

def steady_ok(cell, ok):
    """Steady window: t >= WARMUP and within the run; also drop requests that STARTED
    after 90% of the max completion time (shutdown tail on request side)."""
    if not ok:
        return []
    tmax = max(r["t"] for r in ok)
    # steady: started after warmup, and e2e completed before run end approximated by tmax
    st = [r for r in ok if WARMUP <= r["t"] <= 0.9 * tmax]
    return st

def load_steps(cell):
    p = os.path.join(RESULTS, cell, "obs/steps.jsonl")
    if not os.path.exists(p):
        return []
    recs = [json.loads(l) for l in open(p) if l.strip()]
    n = len(recs)
    # exclude final 10% of steps (shutdown tail)
    return recs[:max(0, int(n * (1 - TAIL_FRAC)))]

def load_metrics(cell):
    p = os.path.join(RESULTS, cell, "metrics.jsonl")
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p) if l.strip()]

def cache_hit_rate_from_log(cell):
    """Parse 'Prefix cache hit rate: X%' lines from server.log; median of last 20."""
    p = os.path.join(RESULTS, cell, "server.log")
    if not os.path.exists(p):
        return None
    vals = []
    pat = re.compile(r"Prefix cache hit rate:\s*([\d.]+)%")
    for line in open(p, errors="ignore"):
        m = pat.search(line)
        if m:
            vals.append(float(m.group(1)))
    if not vals:
        return None
    tail = sorted(vals[-20:])
    return tail[len(tail) // 2]

def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1)))))
    return xs[k]

def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")

# ---------------------------------------------------------------------------
report = {"cells": {}, "per_class": {}, "verdict_inputs": {}}

for cond in ("A", "B", "C"):
    for arm in ("ar", "dspark"):
        cels = cells_matching(f"{arm}_{cond}_c")
        for cell in cels:
            recs, ok = load_client(cell)
            st = steady_ok(cell, ok)
            steps = load_steps(cell)
            metrics = load_metrics(cell)
            dur = 150.0
            window = max(1e-9, 0.9 * (max((r["t"] for r in ok), default=0)) - WARMUP) if ok else 0
            toks = sum(r["ntok"] for r in st)
            c = {
                "cell": cell, "n_ok": len(ok), "n_err": len(recs) - len(ok), "n_steady": len(st),
                "steady_out_tok_s": round(toks / window, 1) if window else None,
                "per_class": {},
                "steps_n": len(steps),
            }
            for cls in SCHEME[cond]["classes"]:
                rc = [r for r in st if r["cls"] == cls]
                if not rc:
                    continue
                c["per_class"][cls] = {
                    "n": len(rc),
                    "ttft_p50_ms": round(pct([r["ttft_ms"] for r in rc], 50), 2),
                    "ttft_p95_ms": round(pct([r["ttft_ms"] for r in rc], 95), 2),
                    "tpot_p50_ms": round(pct([r["tpot_ms"] for r in rc], 50), 3),
                    "tpot_p95_ms": round(pct([r["tpot_ms"] for r in rc], 95), 3),
                    "e2e_p50_s": round(pct([r["e2e_s"] for r in rc], 50), 3),
                    "e2e_p95_s": round(pct([r["e2e_s"] for r in rc], 95), 3),
                    "ntok_p10": pct([r["ntok"] for r in rc], 10),
                    "ntok_p50": pct([r["ntok"] for r in rc], 50),
                    "ntok_p90": pct([r["ntok"] for r in rc], 90),
                    "frac_at_max": round(sum(1 for r in rc if r["ntok"] >= SCHEME[cond]["maxtok"][cls]) / len(rc), 3),
                }
            # step-side aggregates
            if steps:
                running = [s.get("running", 0) for s in steps]
                waiting = [s.get("waiting", 0) for s in steps]
                c["running_p50"] = pct(running, 50)
                c["running_max"] = max(running) if running else None
                c["waiting_p50"] = pct(waiting, 50)
                c["waiting_frac_pos"] = round(sum(1 for w in waiting if w > 0) / len(waiting), 3)
                prefill_share = (sum(s.get("prefill_tokens", 0) for s in steps) /
                                 max(1, sum(s.get("scheduled_tokens", 0) for s in steps)))
                c["prefill_token_share"] = round(prefill_share, 4)
                if arm == "dspark":
                    shares = [((s["gpu_step_ms"] - s["gpu_fwd_ms"]) / s["gpu_step_ms"])
                              for s in steps
                              if s.get("gpu_step_ms") and s.get("gpu_fwd_ms") and s["gpu_step_ms"] > 0]
                    c["drafter_share_p50"] = round(pct(shares, 50), 4) if shares else None
            # acceptance + prefix-cache from metrics deltas (steady window only: t >= WARMUP)
            sm = [m for m in metrics if m.get("t", 0) >= WARMUP]
            acc = None
            cache_rate = None
            if len(sm) > 2:
                if arm == "dspark":
                    # scraper writes per-interval deltas as "<name>_d" (it strips "[counter]",
                    # not appends "_d") — the earlier "...[counter]_d" key never matched -> None.
                    d_tok = [m.get("vllm:spec_decode_num_draft_tokens_d", 0) or 0 for m in sm]
                    a_tok = [m.get("vllm:spec_decode_num_accepted_tokens_d", 0) or 0 for m in sm]
                    dt, at = sum(d_tok), sum(a_tok)
                    if dt > 0:
                        acc = round(at / dt, 4)
                q = [m.get("vllm:prefix_cache_queries_d", 0) or 0 for m in sm]
                h = [m.get("vllm:prefix_cache_hits_d", 0) or 0 for m in sm]
                if sum(q) > 0:
                    cache_rate = round(sum(h) / sum(q), 4)
            c["acceptance_rate"] = acc
            c["prefix_cache_hit_rate"] = cache_rate
            c["prefix_cache_hit_rate_log"] = cache_hit_rate_from_log(cell)
            report["cells"][cell] = c

# ---- per-class DSpark/AR ratios (request-paired bootstrap by pid within C) ----
def paired_ratio_ci(cond, C, cls, n_boot=2000):
    ar_cells = [f"ar_{cond}_c{C}_t{t}" for t in (1, 2, 3)]
    ds_cells = [f"dspark_{cond}_c{C}_t{t}" for t in (1, 2, 3)]
    ratios = []
    for ac, dc in zip(ar_cells, ds_cells):
        if not (os.path.exists(os.path.join(RESULTS, ac, "client.jsonl")) and
                os.path.exists(os.path.join(RESULTS, dc, "client.jsonl"))):
            continue
        _, ok_ar = load_client(ac)
        _, ok_ds = load_client(dc)
        st_ar = {r["pid"]: r for r in steady_ok(ac, ok_ar)}
        st_ds = {r["pid"]: r for r in steady_ok(dc, ok_ds)}
        common = [p for p in st_ar if p in st_ds and st_ar[p]["cls"] == cls and st_ds[p]["cls"] == cls]
        for p in common:
            a, d = st_ar[p]["tpot_ms"], st_ds[p]["tpot_ms"]
            if a > 0:
                ratios.append(d / a)
    if not ratios:
        return None
    import random as _r
    rng = _r.Random(20261013)
    boots = []
    n = len(ratios)
    for _ in range(n_boot):
        s = [ratios[rng.randrange(n)] for _ in range(n)]
        boots.append(mean(s))
    boots.sort()
    return {"n_paired": n, "mean": round(mean(ratios), 4),
            "ci95": [round(boots[int(0.025 * n_boot)], 4), round(boots[int(0.975 * n_boot)], 4)]}

for cond in ("A", "B", "C"):
    for C in C_LEVELS:
        row = {}
        for cls in SCHEME[cond]["classes"]:
            r = paired_ratio_ci(cond, C, cls)
            if r:
                row[cls] = r
        if row:
            means = {k: v["mean"] for k, v in row.items()}
            lo, hi = min(means.values()), max(means.values())
            row["_cross_class_range_pct"] = round(100 * (hi - lo) / mean(list(means.values())), 2)
            report.setdefault("ratios", {})[f"{cond}_c{C}"] = row

# ---- H-C3 descriptive regression (OLS, ratio ~ logOSL + logISL + acceptance) ----
def ols_hc3():
    rows = []
    for cond in ("A", "B"):
        for C in C_LEVELS:
            key = f"{cond}_c{C}"
            if key not in report.get("ratios", {}):
                continue
            acc = None
            for t in (1, 2, 3):
                c = report["cells"].get(f"dspark_{cond}_c{C}_t{t}", {})
                if c.get("acceptance_rate"):
                    acc = c["acceptance_rate"]
                    break
            for cls, r in report["ratios"][key].items():
                if not isinstance(r, dict):
                    continue
                # realized OSL: use the dspark cell's ntok_p50 (paired population)
                osl = None
                for t in (1, 2, 3):
                    cc = report["cells"].get(f"dspark_{cond}_c{C}_t{t}", {})
                    if cls in cc.get("per_class", {}):
                        osl = cc["per_class"][cls]["ntok_p50"]
                        break
                isl = SCHEME[cond]["isl"][cls]
                if osl and acc:
                    rows.append({"cond": cond, "C": C, "cls": cls, "ratio": r["mean"],
                                 "logOSL": math.log(osl), "logISL": math.log(isl), "acc": acc})
    if len(rows) < 8:
        return {"note": f"insufficient rows ({len(rows)})"}
    # simple OLS via normal equations (4 params incl intercept)
    n = len(rows)
    X = [[1.0, r["logOSL"], r["logISL"], r["acc"]] for r in rows]
    y = [r["ratio"] for r in rows]
    import itertools
    # Gaussian elimination on X'X (4x4)
    XtX = [[sum(X[i][a] * X[i][b] for i in range(n)) for b in range(4)] for a in range(4)]
    Xty = [sum(X[i][a] * y[i] for i in range(n)) for a in range(4)]
    M = [row[:] + [Xty[a]] for a, row in enumerate(XtX)]
    for col in range(4):
        piv = max(range(col, 4), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        for r in range(4):
            if r != col and M[col][col] != 0:
                f = M[r][col] / M[col][col]
                M[r] = [m - f * p for m, p in zip(M[r], M[col])]
    beta = [M[a][4] / M[a][a] for a in range(4)] if all(abs(M[a][a]) > 1e-12 for a in range(4)) else None
    return {"n": n, "beta_intercept_logOSL_logISL_acc": [round(b, 5) for b in beta] if beta else None}

report["verdict_inputs"]["h_c3_ols"] = ols_hc3()

# ---- Condition-C verdict (docs/hclass1c-prereg.md §5): does a diverse-query workload with a shared
# cached context sit closer to A (content-driven gap) or B (reuse-state effect)? ----
def c_verdict():
    # frozen anchors from the closed H-CLASS-1 run (prereg C §1)
    A_ACC, B_ACC = 0.332, 0.611
    rng = os.environ.get("HCLASS1_C_LEVELS", "16,48,96") or "16,48,96"
    clv = [int(x) for x in rng.split(",") if x.strip()]
    acc_by_c, range_by_c, hit_by_c = {}, {}, {}
    for C in clv:
        # trial-mean acceptance over the 3 dspark trials (steady-window metric)
        acs = []
        for t in (1, 2, 3):
            c = report["cells"].get(f"dspark_C_c{C}_t{t}", {})
            if c.get("acceptance_rate") is not None:
                acs.append(c["acceptance_rate"])
        if acs:
            acc_by_c[C] = round(mean(acs), 4)
        row = report.get("ratios", {}).get(f"C_c{C}", {})
        if isinstance(row, dict) and "_cross_class_range_pct" in row:
            range_by_c[C] = row["_cross_class_range_pct"]
        hits = []
        for t in (1, 2, 3):
            c = report["cells"].get(f"dspark_C_c{C}_t{t}", {})
            if c.get("prefix_cache_hit_rate") is not None:
                hits.append(c["prefix_cache_hit_rate"])
        if hits:
            hit_by_c[C] = round(mean(hits), 4)
    out = {"anchors": {"A_acc": A_ACC, "B_acc": B_ACC},
           "C_acc_by_pressure": acc_by_c, "C_cross_class_range_pct_by_pressure": range_by_c,
           "C_cache_hit_rate_by_pressure": hit_by_c}
    if not acc_by_c:
        out["verdict"] = "NO DATA (no valid C cells)"
        return out
    # fixed bands (prereg §5): near_a = acceptance within 0.05 of A AND range <15% everywhere;
    # near_b = acceptance within 0.05 of B OR range >15% at any pressure.
    acc_near_a = all(abs(v - A_ACC) <= 0.05 for v in acc_by_c.values())
    acc_near_b = any(abs(v - B_ACC) <= 0.05 for v in acc_by_c.values())
    range_all_lt15 = (range_by_c is not None and range_by_c != {}) and all(r < 15 for r in range_by_c.values())
    range_any_gt15 = bool(range_by_c) and any(r > 15 for r in range_by_c.values())
    near_a = acc_near_a and (range_all_lt15 if range_by_c else False)
    near_b = acc_near_b or range_any_gt15
    if near_a and not near_b:
        out["verdict"] = ("C~A (content-driven gap): the A<->B acceptance gap is a content-distribution "
                          "effect; caching a shared prefix does not by itself lift acceptance or restore "
                          "the class spread.")
    elif near_b and not near_a:
        out["verdict"] = ("C~B (reuse-state effect): shared-context reuse lifts draft acceptance / restores "
                          "class-dependent payoff even under query diversity; cache/reuse state is a genuine "
                          "causally-available admission feature.")
    else:
        out["verdict"] = ("INTERMEDIATE: partial mediation. Report magnitudes (C_acc, hit rate, range).")
    return out

report["verdict_inputs"]["condition_c"] = c_verdict()

out = os.path.join(RESULTS, "analysis.json")
os.makedirs(RESULTS, exist_ok=True)
json.dump(report, open(out, "w"), indent=2)
print("wrote", out)

# ---- markdown table ----
lines = ["# H-CLASS-1 analysis (hclass1_analyze.py)", ""]
lines.append("## Per-cell summary")
lines.append("| cell | cond | arm | n_steady | tok/s | running p50/max | waiting frac>0 | prefill share | acc rate | cache hit% (metrics) |")
lines.append("|---|---|---|---:|---:|---|---:|---:|---:|---:|")
for cell, c in sorted(report["cells"].items()):
    m = re.match(r"(ar|dspark)_([ABC])_c(\d+)_t(\d+)", cell)
    cond = m.group(2) if m else "?"
    arm = "AR" if (m and m.group(1) == "ar") else "DS"
    ch = c.get("prefix_cache_hit_rate")
    lines.append(f"| {cell} | {cond} | {arm} | {c['n_steady']} | {c.get('steady_out_tok_s')} | "
                 f"{c.get('running_p50')}/{c.get('running_max')} | {c.get('waiting_frac_pos')} | "
                 f"{c.get('prefill_token_share')} | {c.get('acceptance_rate')} | {ch} |")
lines.append("")
lines.append("## Per-class realized OSL + TPOT (steady window)")
for cell, c in sorted(report["cells"].items()):
    lines.append(f"\n### {cell}")
    lines.append("| class | n | TTFT p50/p95 ms | TPOT p50/p95 ms | E2E p50/p95 s | ntok p10/p50/p90 | frac_at_max |")
    lines.append("|---|---:|---|---|---|---|---:|")
    m = re.match(r"(ar|dspark)_([ABC])_c(\d+)_t(\d+)", cell)
    cls_list = SCHEME[m.group(2)]["classes"] if m else []
    for cls in cls_list:
        pc = c.get("per_class", {}).get(cls)
        if not pc:
            continue
        lines.append(f"| {cls} | {pc['n']} | {pc['ttft_p50_ms']}/{pc['ttft_p95_ms']} | "
                     f"{pc['tpot_p50_ms']}/{pc['tpot_p95_ms']} | {pc['e2e_p50_s']}/{pc['e2e_p95_s']} | "
                     f"{pc['ntok_p10']:.0f}/{pc['ntok_p50']:.0f}/{pc['ntok_p90']:.0f} | {pc['frac_at_max']} |")
lines.append("")
lines.append("## Per-class DSpark/AR TPOT ratio (request-paired bootstrap, 2000 resamples)")
for key in sorted(report.get("ratios", {})):
    row = report["ratios"][key]
    lines.append(f"\n### {key} — cross-class range: **{row.get('_cross_class_range_pct')}%**")
    lines.append("| class | n_paired | mean ratio | 95% CI |")
    lines.append("|---|---:|---:|---|")
    cond = key[0]
    for cls in SCHEME[cond]["classes"]:
        r = row.get(cls)
        if isinstance(r, dict):
            lines.append(f"| {cls} | {r['n_paired']} | {r['mean']} | [{r['ci95'][0]}, {r['ci95'][1]}] |")
lines.append("")
lines.append("## H-C3 descriptive OLS (ratio ~ 1 + logOSL + logISL + acceptance)")
json.dump(report["verdict_inputs"]["h_c3_ols"], open(os.path.join(RESULTS, "hc3.json"), "w"))
lines.append(f"```json\n{json.dumps(report['verdict_inputs']['h_c3_ols'], indent=2)}\n```")
open(os.path.join(RESULTS, "analysis_table.md"), "w").write("\n".join(lines))
print("wrote", os.path.join(RESULTS, "analysis_table.md"))
