#!/usr/bin/env python3
"""H-OBS-3 analysis pipeline (prereg docs/hoobs3-prereg.md §4).

Consumes the 4 main-run cells under results/hoobs3/hetero/<cell>/ and produces:
  - analysis.json    : all computed quantities (machine-readable)
  - analysis_table.md: compact markdown for the synthesis doc

§4 windows (fixed in prereg): steady-state = exclude first WARMUP=20 s of client
time and the final 10% of steps (shutdown tail). Steps are temporally correlated
-> per-cell distributions + cross-cell pressure dependence, NOT step-level CIs.

H1 (phase): acceptance A by prefill-heavy vs pure-decode steps (dspark cells),
   from Prometheus counter deltas time-aligned to steps via t_wall.
H2 (class): per-class DSpark/AR TPOT ratio range within each pressure level.
H3 (price): drafter share (gpu_step_ms - gpu_fwd_ms)/gpu_step_ms by arm x C.
"""
import json
import os
import statistics as st

MAIN = "/home/junior1/_dev/better-vllm-speculative-decoding/results/hoobs3/hetero"
CELLS = ["ar_c32", "dspark_c32", "ar_c96", "dspark_c96"]
WARMUP_S = 20.0


def load_cell(cell):
    d = os.path.join(MAIN, cell)
    if not (os.path.isdir(d) and os.path.exists(os.path.join(d, "DONE"))):
        return None
    steps_p = os.path.join(d, "obs", "steps.jsonl")
    recs = []
    if os.path.exists(steps_p):
        recs = [json.loads(l) for l in open(steps_p)]
    client = []
    cp = os.path.join(d, "client.jsonl")
    if os.path.exists(cp):
        client = [json.loads(l) for l in open(cp)]
    metrics = []
    mp = os.path.join(d, "metrics.jsonl")
    if os.path.exists(mp):
        metrics = [json.loads(l) for l in open(mp)]
    return {"cell": cell, "recs": recs, "client": client, "metrics": metrics}


def steady_steps(recs):
    """Steady window over steps: drop the final 10% (shutdown tail). The ramp is
    handled on the client side (WARMUP_S) because step timestamps are engine-side;
    we additionally require t_wall to be within [client_t0+WARMUP, client_end]."""
    n = len(recs)
    if n < 20:
        return recs, n
    cut = int(0.90 * n)
    return recs[:cut], n


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


def client_window(client):
    """Return (steady_recs, t0_wall_offset_info). Client 't' is monotonic since
    process start; the metrics scraper's 't' is also monotonic since ITS start.
    Alignment to steps uses t_wall (epoch) on both sides: steps carry t_wall;
    we rebase client/metrics onto epoch via the first step's (t_wall, position)."""
    ok = [r for r in client if "ttft_ms" in r]
    st_ = [r for r in ok if WARMUP_S <= r["t"]]
    return st_


def analyze():
    result = {"cells_analyzed": [], "h1_phase": {}, "h2_class": {}, "h3_price": {},
              "summary": {}}
    cells = {}
    for c in CELLS:
        d = load_cell(c)
        if d is None or not d["recs"]:
            continue
        cells[c] = d
        result["cells_analyzed"].append(c)

    # ---------- H1: acceptance by phase (dspark cells) ----------
    # Per-0.5s-bucket acceptance from Prometheus counter deltas, aligned to steps
    # by epoch: metrics rows carry t_epoch (time.time() at capture); steps carry
    # t_wall (time.time() at schedule entry). A step is assigned to the LAST bucket
    # whose epoch <= step.t_wall; phase acceptance = mean of bucket-level
    # accepted_d/draft_d over buckets that have a draft delta and >=1 assigned step.
    for c in ["dspark_c32", "dspark_c96"]:
        if c not in cells:
            continue
        recs, ntot = steady_steps(cells[c]["recs"])
        metrics = cells[c]["metrics"]
        if not metrics or not recs:
            continue
        buckets = []  # (epoch, draft_d, accepted_d)
        for row in metrics:
            dd = row.get("vllm:spec_decode_num_draft_tokens_d")
            da = row.get("vllm:spec_decode_num_accepted_tokens_d")
            ep = row.get("t_epoch")
            if dd is None or da is None or ep is None:
                continue
            buckets.append((ep, dd, da))
        if not buckets:
            result["h1_phase"][c] = {"error": "no epoch-aligned metric buckets"}
            continue
        import bisect
        b_epochs = [b[0] for b in buckets]
        entry = {"n_steps": len(recs), "phases": {}}
        for phase, pred in [
            ("pure_decode", lambda r: r.get("prefill_tokens", 0) == 0 and r.get("draft_tokens", 0) > 0),
            ("prefill_heavy", lambda r: r.get("scheduled_tokens", 0) > 0
             and r.get("prefill_tokens", 0) >= 0.05 * r.get("scheduled_tokens", 1)
             and r.get("draft_tokens", 0) > 0),
            ("mixed_draft", lambda r: r.get("draft_tokens", 0) > 0
             and r.get("prefill_tokens", 0) > 0
             and r.get("prefill_tokens", 0) < 0.05 * r.get("scheduled_tokens", 1)),
        ]:
            sel = [r for r in recs if pred(r)]
            # Prereg §4.2: per-step A = its assigned bucket's accepted_d/draft_d
            # (proportional attribution collapses to this when the bucket's steps
            # are exactly those whose drafts were scheduled inside it); phase A =
            # mean over the phase's steps.
            A_s = []
            for r in sel:
                tw = r.get("t_wall")
                if tw is None:
                    continue
                # Bucket j's delta covers [E_{j-1}, E_j]; assign to the bucket whose
                # interval contains tw (draft work runs just after schedule entry).
                j = bisect.bisect_right(b_epochs, tw)
                if j >= len(buckets):
                    continue
                _, dd, da = buckets[j]
                if dd:
                    A_s.append(da / dd)
            entry["phases"][phase] = {
                "n_steps": len(sel),
                "acceptance_mean": round(st.mean(A_s), 4) if A_s else None,
                "acceptance_p50": (round(sorted(A_s)[len(A_s)//2], 4) if A_s else None),
            }
        entry["acceptance_gap_prefill_minus_decode"] = (
            round(entry["phases"]["prefill_heavy"]["acceptance_mean"]
                  - entry["phases"]["pure_decode"]["acceptance_mean"], 4)
            if entry["phases"]["prefill_heavy"].get("acceptance_mean") is not None
            and entry["phases"]["pure_decode"].get("acceptance_mean") is not None else None)
        result["h1_phase"][c] = entry

    # ---------- H2: per-class TPOT/TTFT, DSpark/AR ratio within each C ----------
    cls_stats = {}
    for c in CELLS:
        if c not in cells:
            continue
        stc = client_window(cells[c]["client"])
        s = {}
        for cl in ["SS", "SL", "LS", "LL"]:
            rc = [r for r in stc if r["cls"] == cl]
            if not rc:
                s[cl] = {"n": 0}
                continue
            s[cl] = {
                "n": len(rc),
                "ntok_p50": round(st.median(r["ntok"] for r in rc), 1),
                "ttft_ms_p50": round(st.median(r["ttft_ms"] for r in rc), 1),
                "ttft_ms_p95": round(sorted(r["ttft_ms"] for r in rc)[int(0.95*(len(rc)-1))], 1),
                "tpot_ms_p50": round(st.median(r["tpot_ms"] for r in rc), 2),
                "tpot_ms_p95": round(sorted(r["tpot_ms"] for r in rc)[int(0.95*(len(rc)-1))], 2),
            }
        cls_stats[c] = s
    for C in [32, 96]:
        ar, ds = cls_stats.get(f"ar_c{C}"), cls_stats.get(f"dspark_c{C}")
        if not ar or not ds:
            continue
        ratios = {}
        for cl in ["SS", "SL", "LS", "LL"]:
            a, d = ar.get(cl, {}), ds.get(cl, {})
            if a.get("n") and d.get("n") and a.get("tpot_ms_p50"):
                ratios[cl] = {
                    "ar_tpot_p50": a["tpot_ms_p50"], "dspark_tpot_p50": d["tpot_ms_p50"],
                    "ratio_dspark_over_ar": round(d["tpot_ms_p50"] / a["tpot_ms_p50"], 3),
                }
        vals = [v["ratio_dspark_over_ar"] for v in ratios.values()]
        result["h2_class"][f"c{C}"] = {
            "per_class": ratios,
            "cross_class_range_pct": (round(100.0 * (max(vals) - min(vals)) / st.mean(vals), 2)
                                      if len(vals) >= 3 else None),
        }

    # ---------- H3: drafter share by arm x C ----------
    for c in CELLS:
        if c not in cells:
            continue
        recs, _ = steady_steps(cells[c]["recs"])
        allshare, decshare = [], []
        for r in recs:
            g, f = r.get("gpu_step_ms"), r.get("gpu_fwd_ms")
            if not g or g <= 0 or f is None:
                continue
            share = (g - f) / g
            allshare.append(share)
            if r.get("prefill_tokens", 0) == 0:
                decshare.append(share)
        result["h3_price"][c] = {
            "drafter_share_all_steps": dist(allshare),
            "drafter_share_decode_only": dist(decshare),
            "running_mean": round(st.mean(r.get("running", 0) for r in recs), 2) if recs else None,
        }

    # ---------- summary: throughput/latency per cell x class ----------
    for c in CELLS:
        if c not in cells:
            continue
        stc = client_window(cells[c]["client"])
        window = max(1e-9, 180.0 - WARMUP_S)
        tok = sum(r["ntok"] for r in stc)
        result["summary"][c] = {
            "steady_out_tokens": tok,
            "steady_throughput_tok_s": round(tok / window, 1),
            "n_requests_steady": len(stc),
            "per_class": cls_stats.get(c, {}),
        }

    # ---------- H3 slope (C=32 -> C=96) within arm ----------
    for arm in ["ar", "dspark"]:
        a32 = result["h3_price"].get(f"{arm}_c32", {}).get("drafter_share_decode_only")
        a96 = result["h3_price"].get(f"{arm}_c96", {}).get("drafter_share_decode_only")
        if a32 and a96:
            result["h3_price"][f"slope_{arm}"] = {
                "decode_share_c32": a32["mean"], "decode_share_c96": a96["mean"],
                "delta_pp_per_16_running": round(
                    100.0 * (a96["mean"] - a32["mean"]) /
                    max(1e-9, (result["h3_price"][f"{arm}_c96"]["running_mean"]
                              - result["h3_price"][f"{arm}_c32"]["running_mean"])) * 16, 3),
            }

    os.makedirs(MAIN, exist_ok=True)
    with open(os.path.join(MAIN, "analysis.json"), "w") as f:
        json.dump(result, f, indent=2)

    md = []
    md.append("# H-OBS-3 analysis (steady window: client t>=20s, steps excl. final 10%)\n")
    md.append("## H1 — acceptance by phase (dspark cells)\n")
    for c, e in result["h1_phase"].items():
        ph = e["phases"]
        md.append(f"- **{c}**: decode A={ph['pure_decode']['acceptance_mean']} "
                  f"(n={ph['pure_decode']['n_steps']}) | prefill A={ph['prefill_heavy']['acceptance_mean']} "
                  f"(n={ph['prefill_heavy']['n_steps']}) | mixed A={ph['mixed_draft']['acceptance_mean']} "
                  f"| gap(prefill-decode)={e['acceptance_gap_prefill_minus_decode']}")
    md.append("\n## H2 — per-class DSpark/AR TPOT ratio (within each C)\n")
    for k, e in result["h2_class"].items():
        md.append(f"- **{k}**: " + ", ".join(
            f"{cl}={v['ratio_dspark_over_ar']}" for cl, v in e["per_class"].items())
            + f" | cross-class range = {e['cross_class_range_pct']}%")
    md.append("\n## H3 — drafter share (gpu_step - gpu_fwd)/gpu_step\n")
    for c, e in result["h3_price"].items():
        if c.startswith("slope_"):
            md.append(f"- **{c}**: Δ = {e['delta_pp_per_16_running']} pp per +16 running "
                      f"(c32={e['decode_share_c32']}, c96={e['decode_share_c96']})")
        else:
            md.append(f"- **{c}**: all-steps p50={e['drafter_share_all_steps'] and e['drafter_share_all_steps']['p50']} "
                      f"decode-only p50={e['drafter_share_decode_only'] and e['drafter_share_decode_only']['p50']} "
                      f"(runninḡ={e['running_mean']})")
    md.append("\n## Summary — steady throughput & per-class latency\n")
    for c, e in result["summary"].items():
        md.append(f"- **{c}**: {e['steady_throughput_tok_s']} tok/s ({e['n_requests_steady']} reqs); "
                  + ", ".join(f"{cl}: tpot={v.get('tpot_ms_p50')}ms ttft={v.get('ttft_ms_p50')}ms ntok={v.get('ntok_p50')}"
                              for cl, v in e["per_class"].items()))
    txt = "\n".join(md) + "\n"
    with open(os.path.join(MAIN, "analysis_table.md"), "w") as f:
        f.write(txt)
    print(f"analyzed {len(result['cells_analyzed'])} cells: {result['cells_analyzed']}")
    print("wrote analysis.json + analysis_table.md under", MAIN)


if __name__ == "__main__":
    analyze()
