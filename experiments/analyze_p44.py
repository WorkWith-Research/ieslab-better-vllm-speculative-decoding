#!/usr/bin/env python3
"""Phase 4.4 analyzer: heterogeneous ISLxOSL workload — per-class K x C results.

Tests the pre-registered hypotheses (PROGRESS.md):
  H-4.4a: at C=96, argmax_K differs ACROSS classes (SL tolerates larger K than LS/LL).
  H-4.4b: TPOT degradation of K=8 vs AR grows with ISL within a cell (SS <= SL < LS <= LL).

Reads results/p4_4/p44_k{K}_c{C}_t{T}_hetero_client.out (per-class summary lines) and the
raw per-request jsonl is NOT re-read (client already aggregates steady-state per class).
"""
import json
import os
import re
import sys
from collections import defaultdict

BASE = "results/p4_4"
KS = ["none", "1", "2", "4", "8"]
CS = [32, 96]
TRIALS = [0, 1, 2]
CLASSES = ["SS", "SL", "LS", "LL"]

# parse per-class summary lines: "   SS     4     34.1    69.8/71.9       21.5/21.6          256"
CLASS_RE = re.compile(r"^\s*(SS|SL|LS|LL)\s+(\d+)\s+([\d.]+)\s+([\d.]+)/([\d.]+)\s+([\d.]+)/([\d.]+)\s+([\d.]+)")


def cell(K, C, T):
    p = f"{BASE}/p44_k{K}_c{C}_t{T}_hetero_client.out"
    if not os.path.exists(p):
        return None
    rows = open(p).read().splitlines()
    agg = None
    for l in rows:
        m = re.search(r"steady_throughput_tok_s=([\d.]+)", l)
        if m:
            agg = float(m.group(1))
    per_class = {}
    for l in rows:
        m = CLASS_RE.match(l)
        if m:
            cls, n, toks, ttft50, ttft95, tpot50, tpot95, ntok = m.groups()
            per_class[cls] = {"n": int(n), "tok_s": float(toks),
                              "ttft50": float(ttft50), "ttft95": float(ttft95),
                              "tpot50": float(tpot50), "tpot95": float(tpot95),
                              "ntok": float(ntok)}
    return {"agg_tok_s": agg, "per_class": per_class}


def main():
    data = {}
    for K in KS:
        for C in CS:
            cells = [cell(K, C, T) for T in TRIALS]
            cells = [c for c in cells if c]
            if not cells:
                continue
            data[(K, C)] = cells

    print("=" * 78)
    print("Phase 4.4 — heterogeneous ISLxOSL (SS/SL/LS/LL), steady-state aggregate tok/s")
    print("=" * 78)
    hdr = f"{'C':>4} | " + " | ".join(f"{k:>10}" for k in KS) + " | argmax"
    print(hdr)
    for C in CS:
        row = f"{C:>4} | "
        vals = {}
        for K in KS:
            cells = data.get((K, C), [])
            if cells:
                v = sum(c["agg_tok_s"] or 0 for c in cells) / len(cells)
                vals[K] = v
                row += f"{v:>10.1f} | "
            else:
                row += f"{'--':>10} | "
        if vals:
            best = max(vals, key=vals.get)
            row += f"K={best}"
        print(row)

    print()
    print("Per-class steady tok/s (mean over trials):")
    for C in CS:
        print(f"\n--- C={C} ---")
        hdr = f"{'class':>5} | " + " | ".join(f"{k:>9}" for k in KS)
        print(hdr)
        for cls in CLASSES:
            row = f"{cls:>5} | "
            for K in KS:
                cells = data.get((K, C), [])
                v = [c["per_class"][cls]["tok_s"] for c in cells if cls in c["per_class"]]
                row += f"{sum(v)/len(v):>9.1f} |" if v else f"{'--':>9} |"
            print(row)

    # H-4.4a: per-class argmax at C=96
    print()
    print("H-4.4a — per-class argmax_K at C=96:")
    c96 = {}
    for cls in CLASSES:
        vals = {}
        for K in KS:
            cells = data.get((K, 96), [])
            v = [c["per_class"][cls]["tok_s"] for c in cells if cls in c["per_class"]]
            if v:
                vals[K] = sum(v) / len(v)
        if vals:
            best = max(vals, key=vals.get)
            c96[cls] = (best, vals)
            print(f"  {cls}: argmax K={best}   values=" +
                  " ".join(f"K{k}={v:.1f}" for k, v in sorted(vals.items())))
    if len(set(b for b, _ in c96.values())) > 1:
        print("  -> H-4.4a SUPPORT (argmax differs across classes)")
    elif c96:
        print("  -> H-4.4a FALSIFY (all classes share the same argmax at C=96)")

    # H-4.4b: TPOT degradation K8 vs AR by class at C=96
    print()
    print("H-4.4b — TPOT p50 degradation of K=8 vs AR at C=96 (by class):")
    ar = {cls: [c["per_class"][cls]["tpot50"] for c in data.get(("none", 96), []) if cls in c["per_class"]]
          for cls in CLASSES}
    k8 = {cls: [c["per_class"][cls]["tpot50"] for c in data.get(("8", 96), []) if cls in c["per_class"]]
          for cls in CLASSES}
    deg = {}
    for cls in CLASSES:
        if ar[cls] and k8[cls]:
            a, b = sum(ar[cls]) / len(ar[cls]), sum(k8[cls]) / len(k8[cls])
            deg[cls] = (b - a) / a * 100
            print(f"  {cls}: AR {a:.1f}ms -> K8 {b:.1f}ms  ({(b-a)/a*100:+.1f}%)")
    order_ok = all(deg.get(a, -99) <= deg.get(b, 99) for a, b in [("SS", "SL"), ("SL", "LS"), ("LS", "LL")])
    if len(deg) == 4:
        print("  -> monotone SS<=SL<=LS<=LL:", order_ok,
              "-> H-4.4b SUPPORT" if order_ok else "-> H-4.4b FALSIFY (no monotone ISL trend)")

    # dump json for the record
    out = {"cells": {f"{K}_{C}": [c["agg_tok_s"] for c in data.get((K, C), [])] for K in KS for C in CS}}
    with open(f"{BASE}/summary.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {BASE}/summary.json")


if __name__ == "__main__":
    main()
