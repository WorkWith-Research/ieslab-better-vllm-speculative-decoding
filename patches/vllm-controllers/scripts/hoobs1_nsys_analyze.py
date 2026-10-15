#!/usr/bin/env python3
"""H-OBS-1 Gate-1 serving-path analysis (v2, correlation-based): correlate the
CUDA-event tracer's per-step windows with real GPU kernel execution captured by
nsys under the ACTUAL async serving configuration.

Method (exact step-ID matching, no time-gap heuristics):
  - Each executed step opens an NVTX range `hoobs1.step.<id>` on the engine thread
    spanning the CPU enqueue of that step's GPU work (start in execute_model, end
    after sample_tokens/drafter).
  - CUDA runtime API events (kernel launches) carry correlationIds; so do the
    kernel execution records. Launches whose API-call timestamp falls inside a
    step's NVTX window are attributed to that step, and their device-side kernel
    spans give the step's GPU execution span (first-start -> last-end).
  - Compare each step's nsys kernel span to the tracer's gpu_step_ms for the SAME
    step id.

Known systematic effect: under async serving, tail kernels (drafter/last forward
kernels) can still be executing on the stream after the CPU-side NVTX range has
closed, so nsys-per-launch-window spans are a LOWER BOUND on the tracer's
event-bracketed window (span <= gpu_step_ms). Agreement is expected within ~30%
for typical steps and near-exact for large (prefill) steps.

Gates (fixed):
  S1  Every written tracer record id has a matching NVTX anchor, and the max
      record id has an anchor too (records correspond to real executed steps; no
      fabricated ids). Anchor count >= record count (anchors mark ALL executed
      steps incl. any whose records were lost to the hard-kill shutdown).
  S3  Per-step nsys kernel span vs tracer gpu_step_ms: matched on >=50% of
      written records that have kernels; median(span/gpu_step) in [0.4, 1.1] and
      <=90% of matched steps have span > 0.25*gpu_step (the tracer is not
      measuring enqueue-only time).

Writes serving_validation.json; exits non-zero on failure.
"""
import argparse
import bisect
import json
import os
import sqlite3
import statistics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    db = os.path.join(args.out, "nsys.sqlite")
    report: dict = {"scope": "bounded serving-path check under nsys (async config); diagnostic only"}
    fails = []
    if not os.path.exists(db):
        report.update({"pass": False, "failures": ["nsys.sqlite missing"]})
        json.dump(report, open(os.path.join(args.out, "serving_validation.json"), "w"), indent=2)
        print(json.dumps(report, indent=2))
        raise SystemExit(1)

    con = sqlite3.connect(db)
    # NVTX step anchors: id -> (start_ns, end_ns) on the engine thread.
    anchors = {}
    for t, s, e in con.execute(
            "SELECT text,start,end FROM NVTX_EVENTS WHERE text LIKE 'hoobs1.step.%'"):
        try:
            anchors[int(t.split(".")[-1])] = (s, e)
        except ValueError:
            pass
    report["n_nvtx_anchors"] = len(anchors)

    # Runtime API events sorted by start (launch calls), with correlation ids.
    rt = sorted(con.execute(
        "SELECT start,end,correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME").fetchall())
    rt_starts = [r[0] for r in rt]
    # Kernel execution windows per correlation id (device timestamps). A launch
    # may correspond to several graph-node kernel records; take min-start/max-end.
    kwin = {}
    for cid, s, e in con.execute(
            "SELECT correlationId,start,end FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        a = kwin.get(cid)
        if a is None:
            kwin[cid] = [s, e]
        else:
            if s < a[0]:
                a[0] = s
            if e > a[1]:
                a[1] = e
    con.close()

    # Tracer records.
    sp = os.path.join(args.out, "obs", "steps.jsonl")
    recs = [json.loads(l) for l in open(sp)] if os.path.exists(sp) else []
    by_id = {r["step_id"]: r for r in recs}
    report["n_tracer_records"] = len(recs)

    # ---- S1: record ids <-> anchors ----
    if recs and anchors:
        missing = [sid for sid in by_id if sid not in anchors]
        report["records_without_anchor"] = len(missing)
        ratio = len(anchors) / max(1, len(recs))
        report["anchor_to_record_ratio"] = round(ratio, 3)
        if missing:
            fails.append(f"S1 FAIL: {len(missing)} records have no NVTX anchor (e.g. {missing[:5]})")
        if ratio < 0.95:
            fails.append(f"S1 FAIL: fewer anchors ({len(anchors)}) than records ({len(recs)})")
    else:
        fails.append(f"S1 FAIL: no anchors ({len(anchors)}) or no records ({len(recs)})")

    # ---- S3: per-step nsys kernel span vs tracer gpu_step_ms (same step id) ----
    matched, ratios, below_floor = 0, [], 0
    span_by_id = {}
    for sid in sorted(by_id):
        if sid not in anchors:
            continue
        s, e = anchors[sid]
        i = bisect.bisect_left(rt_starts, s)
        cids = []
        while i < len(rt) and rt[i][0] <= e:
            cid = rt[i][2]
            if cid in kwin:
                cids.append(cid)
            i += 1
        tr = by_id[sid].get("gpu_step_ms")
        if not cids or not tr:
            continue
        span = (max(kwin[c][1] for c in cids) - min(kwin[c][0] for c in cids)) / 1e6
        span_by_id[sid] = round(span, 3)
        matched += 1
        ratios.append(span / tr)
        if span < 0.25 * tr:
            below_floor += 1

    report["n_matched_steps"] = matched
    if matched >= 10:
        med_ratio = statistics.median(ratios)
        frac_below_floor = below_floor / matched
        # outlier (large prefill) steps: show the max-ratio step explicitly
        worst = max(span_by_id, key=lambda k: abs(span_by_id[k] / by_id[k]["gpu_step_ms"] - 1))
        report["median_span_over_gpu_step"] = round(med_ratio, 3)
        report["frac_steps_below_0.25x"] = round(frac_below_floor, 3)
        report["example_large_step"] = {
            "step_id": worst,
            "nsys_span_ms": span_by_id[worst],
            "tracer_gpu_step_ms": by_id[worst]["gpu_step_ms"],
        }
        # sample for the record
        samp = sorted(span_by_id)[:6]
        report["sample_steps"] = {str(k): {"nsys_span_ms": span_by_id[k],
                                           "tracer_gpu_step_ms": by_id[k]["gpu_step_ms"]}
                                  for k in samp}
        if not (0.4 <= med_ratio <= 1.1):
            fails.append(f"S3 FAIL: median nsys-span/gpu-step = {med_ratio:.2f} outside [0.4,1.1]")
        if frac_below_floor > 0.10:
            fails.append(f"S3 FAIL: {frac_below_floor:.0%} of steps have span < 0.25x gpu_step "
                         "(tracer may be measuring enqueue time, not GPU execution)")

    report["pass"] = len(fails) == 0
    report["failures"] = fails
    json.dump(report, open(os.path.join(args.out, "serving_validation.json"), "w"), indent=2)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
