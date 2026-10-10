#!/usr/bin/env python3
"""H-OBS-1 analysis pipeline (prereg docs/hoobs1-prereg.md §5).

Consumes the 18 main-run cells under results/hoobs1/main/<cell>/ and produces:
  - analysis.json   : all computed quantities (machine-readable, for the doc)
  - analysis_table.md : a compact markdown table for docs/hoobs1-observability.md

§5 windows (fixed in prereg): per C, over the STEADY-STATE window = exclude the
first 10% of steps (ramp-up) and the tail where running < 5. Steps are temporally
correlated, so we report per-cell distributions + cross-cell pressure dependence,
NOT an n-driven CI (prereg A1(5)).

Primary regime = dspark arm (native fixed-K SD — what H-OBS-1 targets). AR and
ctrl_live provide baselines / context. Only cells marked DONE are analyzed.
"""
import json
import os
import statistics as st
import sys

MAIN = "/home/junior1/_dev/better-vllm-speculative-decoding/results/hoobs1/main"
CS = 8, 32, 64
ARMS = ("ar", "dspark", "ctrl_live")


def load_cell(cell):
    d = os.path.join(MAIN, cell)
    if not (os.path.isdir(d) and os.path.exists(os.path.join(d, "DONE"))):
        return None
    steps_p = os.path.join(d, "obs", "steps.jsonl")
    recs = []
    if os.path.exists(steps_p):
        recs = [json.loads(l) for l in open(steps_p)]
    bench = {}
    bp = os.path.join(d, "bench_result.json")
    if os.path.exists(bp):
        bench = json.load(open(bp))
    return {"cell": cell, "recs": recs, "bench": bench}


def steady_state(recs):
    """Return (window_recs, n_total). Drop first 10% (ramp) and tail running<5.
    The tail rule is applied by trimming trailing steps once running decays below
    5 (occupancy decay as prompts finish), not by dropping scattered interior ones."""
    n = len(recs)
    if n < 20:
        return recs, n
    ramp = int(0.10 * n)
    body = recs[ramp:]
    # trim trailing run where running < 5
    cut = len(body)
    for i in range(len(body) - 1, -1, -1):
        if body[i].get("running", 0) < 5:
            cut = i
        else:
            break
    return body[:cut], n


def cadence_ms(recs):
    """Per-step wall time since previous schedule() entry (the controller signal)."""
    out = []
    ts = [r["t_cpu_sched"] for r in recs if "t_cpu_sched" in r]
    for i in range(1, len(ts)):
        out.append((ts[i] - ts[i - 1]) * 1000.0)
    return out


def dist(xs):
    if not xs:
        return None
    s = sorted(xs)
    def q(p):
        k = (len(s) - 1) * p
        f, c = int(k), min(int(k) + 1, len(s) - 1)
        return s[f] + (s[c] - s[f]) * (k - f)
    return {"mean": round(st.mean(xs), 3), "p50": round(q(0.5), 3),
            "p90": round(q(0.9), 3), "n": len(xs)}


def linreg(x, y):
    """OLS y = a + b x; return (r2, slope, intercept)."""
    n = min(len(x), len(y))
    if n < 3:
        return None
    x, y = x[:n], y[:n]
    mx, my = st.mean(x), st.mean(y)
    sxx = sum((a - mx) ** 2 for a in x)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    syy = sum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        return None
    b = sxy / sxx
    a = my - b * mx
    r2 = (sxy * sxy) / (sxx * syy)
    return {"r2": round(r2, 4), "slope": round(b, 5), "intercept_ms": round(a, 3)}


def residual_growth(x, y):
    """Split x into terciles; report residual std of (y - fit) per tercile to see if
    prediction error grows with pressure (H-OBS-1a)."""
    n = min(len(x), len(y))
    if n < 30:
        return None
    x, y = x[:n], y[:n]
    reg = linreg(x, y)
    if not reg:
        return None
    order = sorted(range(n), key=lambda i: x[i])
    terc = [order[:n // 3], order[n // 3:2 * n // 3], order[2 * n // 3:]]
    out = []
    for grp in terc:
        if len(grp) < 5:
            continue
        resid = [y[i] - (reg["intercept_ms"] + reg["slope"] * x[i]) for i in grp]
        out.append({"x_mean": round(st.mean(x[i] for i in grp), 1),
                    "resid_std_ms": round(st.pstdev(resid), 3)})
    return out


def analyze():
    result = {"cells_analyzed": [], "per_arm_C": {}, "overhead": {}, "r4_gap": {}}
    for arm in ARMS:
        for C in CS:
            cell_on = f"{arm}_c{C}_on"
            c = load_cell(cell_on)
            if c is None or not c["recs"]:
                continue
            win, ntot = steady_state(c["recs"])
            gs = [r["gpu_step_ms"] for r in win if r.get("gpu_step_ms") is not None]
            fw = [r["gpu_fwd_ms"] for r in win if r.get("gpu_fwd_ms") is not None]
            run = [r.get("running", 0) for r in win]
            # cadence[i-1] (gap since previous step) pairs with step i's own GPU time.
            cd_all = cadence_ms(win)
            cd_a, gs_a, run_a = [], [], []
            for i in range(1, len(win)):
                g = win[i].get("gpu_step_ms")
                if g is not None:
                    cd_a.append(cd_all[i - 1]); gs_a.append(g); run_a.append(win[i]["running"])
            entry = {
                "cell": cell_on, "n_total": ntot, "n_steady": len(win),
                "cadence_ms": dist(cd_a), "gpu_step_ms": dist(gs_a), "gpu_fwd_ms": dist(fw),
                "running_mean": round(st.mean(run), 2) if run else None,
                "ratio_gpu_over_cadence": (round(st.mean(gs_a) / st.mean(cd_a), 3)
                                           if cd_a and gs_a and st.mean(cd_a) > 0 else None),
                "regress_cadence_on_gpu_step": linreg(gs_a, cd_a),
                "regress_cadence_on_running": linreg(run_a, cd_a),
                "residual_growth_by_running": residual_growth(run_a, cd_a),
            }
            result["per_arm_C"][f"{arm}_c{C}"] = entry
            result["cells_analyzed"].append(cell_on)

    # ---- §5.4 overhead: tracer on vs off per arm (output throughput) ----
    for arm in ARMS:
        for C in CS:
            on = load_cell(f"{arm}_c{C}_on")
            off = load_cell(f"{arm}_c{C}_off")
            if not on or not off or "output_throughput" not in on["bench"] \
               or "output_throughput" not in off["bench"]:
                continue
            o, f_ = on["bench"]["output_throughput"], off["bench"]["output_throughput"]
            result["overhead"][f"{arm}_c{C}"] = {
                "on_tok_s": round(o, 1), "off_tok_s": round(f_, 1),
                "delta_pct": round(100.0 * (o - f_) / f_, 2) if f_ else None,
            }

    # ---- §5.3 H-OBS-1b: occupancy-corrected R4 gap (dspark arm = regime of interest) ----
    # R4 derived true_step ≈ L·C_configured/throughput_out and claimed a 1.5x-5.9x
    # gap vs cadence. Recompute substituting ACTUAL steady-state running for C_cfg,
    # with L (accepted output tokens per running-request per step) measured from the
    # trace. If the corrected estimate collapses toward cadence/gpu_step_ms, the R4
    # gap was an occupancy artifact. NOTE: L·running/thr is only an identity when
    # every running request emits exactly L tokens EVERY step; prefill-heavy steps
    # and within-window occupancy decay break that, so we also report the direct
    # gpu_step_ms as ground truth.
    for C in CS:
        c = load_cell(f"dspark_c{C}_on")
        if not c or not c["recs"] or "output_throughput" not in c["bench"]:
            continue
        win, _ = steady_state(c["recs"])
        cd_all = cadence_ms(win)
        gs = [r["gpu_step_ms"] for r in win if r.get("gpu_step_ms") is not None]
        run = [r.get("running", 0) for r in win]
        thr = c["bench"]["output_throughput"]
        tot_out = c["bench"].get("total_output_tokens", 0)
        # Window-consistent L: total output tokens over the WHOLE run divided by
        # total request-steps over the whole run (one scalar, no window mixing).
        sum_run_all = sum(r.get("running", 0) for r in c["recs"])
        L_all = (tot_out / sum_run_all) if sum_run_all else None
        cad_mean = st.mean(cd_all) if cd_all else None
        gs_mean = st.mean(gs) if gs else None
        entry = {"C": C, "throughput_tok_s": round(thr, 1),
                 "cadence_ms_mean": round(cad_mean, 2) if cad_mean else None,
                 "gpu_step_ms_mean": round(gs_mean, 2) if gs_mean else None,
                 "running_steady_mean": round(st.mean(run), 2) if run else None}
        if L_all and thr and cad_mean:
            # seconds -> ms (x1000). r4 uses configured C; corrected uses actual running.
            r4_step = L_all * C / thr * 1000.0
            corr_step = L_all * st.mean(run) / thr * 1000.0
            entry.update({
                "L_whole_run_tok_per_req_step": round(L_all, 3),
                "r4_derived_step_ms": round(r4_step, 2),
                "r4_gap_vs_cadence_x": round(r4_step / cad_mean, 2),
                "corrected_derived_step_ms": round(corr_step, 2),
                "corrected_gap_vs_cadence_x": round(corr_step / cad_mean, 2),
                "direct_gpu_over_cadence_x": round(gs_mean / cad_mean, 2) if gs_mean else None,
            })
        result["r4_gap"][f"c{C}"] = entry

    # ---- verdict support: decode-only high-occupancy tracking + async overlap ----
    # (a) Does cadence track gpu_step on PURE DECODE steps at high occupancy?
    # Prefill spikes are a composition effect, not a cadence/GPU decoupling.
    decode_only = {}
    for C in CS:
        c = load_cell(f"dspark_c{C}_on")
        if not c or not c["recs"]:
            continue
        win, _ = steady_state(c["recs"])
        cd = cadence_ms(win)   # len(win)-1; cd[i] is the gap BEFORE step i (of win)
        cds, gss, runs = [], [], []
        for i in range(1, len(win)):
            r = win[i]
            if r.get("prefill_tokens", 0) == 0 and r.get("running", 0) >= 16 \
               and r.get("gpu_step_ms") is not None:
                cds.append(cd[i - 1]); gss.append(r["gpu_step_ms"])
                runs.append(r["running"])
        if len(cds) >= 20:
            decode_only[f"dspark_c{C}"] = {
                "n_decode_hi_occ_steps": len(cds),
                "running_mean": round(st.mean(runs), 2),
                "cadence_ms": dist(cds), "gpu_step_ms": dist(gss),
                "ratio_gpu_over_cadence": round(st.mean(gss) / st.mean(cds), 3) if st.mean(cds) else None,
                "regress_cadence_on_gpu": linreg(gss, cds),
            }
    result["decode_only_high_occ"] = decode_only

    # (c) H-OBS-1a DIRECT test: does cadence->gpu_step PREDICTION ERROR grow with
    # pressure? Global OLS fit on pure-decode steps, residual std by running tercile.
    # (The per-arm mixed-workload `residual_growth_by_running` above is confounded by
    # prefill spikes; this isolates the clean decode signal.)
    obs1a = {}
    for C in CS:
        c = load_cell(f"dspark_c{C}_on")
        if not c or not c["recs"]:
            continue
        win, _ = steady_state(c["recs"])
        rows = []
        for i in range(1, len(win)):
            r = win[i]
            if r.get("prefill_tokens", 0) == 0 and r.get("gpu_step_ms") is not None:
                cad = (r["t_cpu_sched"] - win[i - 1]["t_cpu_sched"]) * 1000.0
                rows.append((r["running"], cad, r["gpu_step_ms"]))
        if len(rows) < 60:
            continue
        reg = linreg([x[2] for x in rows], [x[1] for x in rows])
        if not reg:
            continue
        rows.sort(key=lambda x: x[0])
        terc = [rows[:len(rows) // 3], rows[len(rows) // 3:2 * len(rows) // 3],
                rows[2 * len(rows) // 3:]]
        groups, hi_occ_ratio = [], []
        for grp in terc:
            if len(grp) < 5:
                continue
            resid = [g[1] - (reg["intercept_ms"] + reg["slope"] * g[2]) for g in grp]
            ratio = [g[1] / g[2] for g in grp]
            groups.append({"run_mean": round(st.mean(g[0] for g in grp), 1), "n": len(grp),
                           "resid_std_ms": round(st.pstdev(resid), 2),
                           "ratio_p50": round(st.median(ratio), 3)})
            hi_occ_ratio += [x for x, r in zip(ratio, grp) if r[0] >= 16]
        all_ratio = [g[1] / g[2] for g in rows]
        obs1a[f"dspark_c{C}"] = {
            "n_pure_decode_steps": len(rows), "r2_cad_on_gpu": reg["r2"],
            "terciles": groups,
            "n_hi_occ": len(hi_occ_ratio),
            "frac_hi_occ_ratio_gt_1.2": (round(sum(1 for x in hi_occ_ratio if x > 1.2) / len(hi_occ_ratio), 3)
                                         if hi_occ_ratio else None),
            "max_ratio": round(max(all_ratio), 2),
        }
    result["obs1a_direct"] = obs1a

    # (b) Async overlap preserved? in_flight distribution per cell (2-deep queue).
    in_flight = {}
    for arm in ARMS:
        for C in CS:
            c = load_cell(f"{arm}_c{C}_on")
            if not c or not c["recs"]:
                continue
            win, _ = steady_state(c["recs"])
            from collections import Counter
            cf = Counter(r.get("in_flight", 0) for r in win)
            total = sum(cf.values())
            in_flight[f"{arm}_c{C}"] = {
                "dist": {str(k): v for k, v in sorted(cf.items())},
                "frac_ge2": round(sum(v for k, v in cf.items() if k >= 2) / total, 3) if total else None,
            }
    result["in_flight"] = in_flight

    # ---- write outputs ----
    os.makedirs(MAIN, exist_ok=True)
    with open(os.path.join(MAIN, "analysis.json"), "w") as f:
        json.dump(result, f, indent=2)

    md = []
    md.append("# H-OBS-1 §5 analysis (steady-state window; temporally-correlated steps → distributions, not CIs)\n")
    md.append("## Per arm × C — cadence vs direct GPU step time\n")
    md.append("| arm·C | n_ss | cadence p50(ms) | gpu_step p50(ms) | gpu_fwd p50(ms) | runninḡ | gpu/cad | R²(cad~gpu) | R²(cad~run) |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k, e in result["per_arm_C"].items():
        cd_, gs_, fw_ = e["cadence_ms"], e["gpu_step_ms"], e["gpu_fwd_ms"]
        rg = e.get("regress_cadence_on_gpu_step") or {}
        rr = e.get("regress_cadence_on_running") or {}
        md.append(f"| {k} | {e['n_steady']} | {cd_ and cd_['p50']} | {gs_ and gs_['p50']} | "
                  f"{fw_ and fw_['p50']} | {e['running_mean']} | {e['ratio_gpu_over_cadence']} | "
                  f"{rg.get('r2')} | {rr.get('r2')} |")
    md.append("\n## Residual growth of cadence~running fit by running tercile (H-OBS-1a)\n")
    for k, e in result["per_arm_C"].items():
        rgw = e.get("residual_growth_by_running")
        if rgw:
            md.append(f"- **{k}**: " + ", ".join(f"run≈{g['x_mean']}→σ={g['resid_std_ms']}ms" for g in rgw))
    md.append("\n## Instrumentation overhead (tracer on vs off, output tok/s)\n")
    md.append("| arm·C | on | off | Δ% |")
    md.append("|---|---:|---:|---:|")
    for k, e in result["overhead"].items():
        md.append(f"| {k} | {e['on_tok_s']} | {e['off_tok_s']} | {e['delta_pct']}% |")
    md.append("\n## H-OBS-1b — occupancy-corrected R4 gap (dspark arm)\n")
    md.append("| C | thr(tok/s) | cadencē | gpu_step̄ | runninḡ | L(whole-run) | R4 step(ms) | R4 gap× | corr step(ms) | corr gap× | direct gpu/cad× |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k, e in result["r4_gap"].items():
        md.append(f"| {e['C']} | {e.get('throughput_tok_s')} | {e.get('cadence_ms_mean')} | "
                  f"{e.get('gpu_step_ms_mean')} | {e.get('running_steady_mean')} | {e.get('L_whole_run_tok_per_req_step')} | "
                  f"{e.get('r4_derived_step_ms')} | {e.get('r4_gap_vs_cadence_x')} | "
                  f"{e.get('corrected_derived_step_ms')} | {e.get('corrected_gap_vs_cadence_x')} | "
                  f"{e.get('direct_gpu_over_cadence_x')} |")
    md.append("\n## Decode-only high-occupancy steps (dspark, prefill=0, running>=16) — prefill-spike confounder check\n")
    md.append("| cell | n | runninḡ | cadence p50 | gpu_step p50 | gpu/cad | R²(cad~gpu) |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for k, e in result["decode_only_high_occ"].items():
        rg = e.get("regress_cadence_on_gpu") or {}
        md.append(f"| {k} | {e['n_decode_hi_occ_steps']} | {e['running_mean']} | "
                  f"{e['cadence_ms'] and e['cadence_ms']['p50']} | {e['gpu_step_ms'] and e['gpu_step_ms']['p50']} | "
                  f"{e['ratio_gpu_over_cadence']} | {rg.get('r2')} |")
    md.append("\n## H-OBS-1a DIRECT — does cadence→gpu_step prediction error grow with pressure? (pure-decode steps)\n")
    md.append("Global OLS fit per cell; residual std by running tercile. Flat/shrinking σ + ratio_p50≈1 ⇒ no growing bias.\n")
    for k, e in result["obs1a_direct"].items():
        terc = ", ".join(f"run≈{g['run_mean']}(σ={g['resid_std_ms']}ms,r={g['ratio_p50']})" for g in e["terciles"])
        md.append(f"- **{k}**: n={e['n_pure_decode_steps']} R²={e['r2_cad_on_gpu']} | {terc} | "
                  f"hi-occ(≥16) n={e['n_hi_occ']} frac>1.2×={e['frac_hi_occ_ratio_gt_1.2']} max={e['max_ratio']}×")
    md.append("\n## Async overlap preserved? in_flight distribution (steady-state)\n")
    for k, e in result["in_flight"].items():
        md.append(f"- **{k}**: " + ", ".join(f"if={kk}:{vv}" for kk, vv in e["dist"].items())
                  + f"  (frac>=2: {e['frac_ge2']})")
    txt = "\n".join(md) + "\n"
    with open(os.path.join(MAIN, "analysis_table.md"), "w") as f:
        f.write(txt)
    print(f"analyzed {len(result['cells_analyzed'])} cells: {result['cells_analyzed']}")
    print("wrote analysis.json + analysis_table.md under", MAIN)


if __name__ == "__main__":
    analyze()
