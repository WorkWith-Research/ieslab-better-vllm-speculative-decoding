#!/usr/bin/env python3
"""Profile SPS(B) — decode steps/sec vs total verification batch size B — from P2 data.

DSpark's 'load model' is a static profiled curve SPS(B). We reconstruct the closest
honest proxy on our hardware: for each P2 cell (K, C), in steady state each decode step
verifies 1+K tokens per active request, so B = C*(1+K) and steps/sec = drafts/s
(cumulative spec_decode_num_drafts counter). AR cells: B=C, SPS = out_tok/s / C.

Uses CUMULATIVE counters (vllm:*[counter]) over the steady window [t>=20s] — NOT sums of
per-sample deltas (those keys are sparse/zero-padded and unreliable).

Usage: .venv/bin/python experiments/profile_sps.py [--base results/p2] [--out ...]
"""
import json, os, sys

BASE = "results/p2"
KS = ["none", "1", "2", "4", "8"]
CS = [8, 32, 96]


def cell_stats(base, K, C, t):
    p = f"{base}/p2_k{K}_c{C}_t{t}_mixed_metrics.jsonl"
    if not os.path.exists(p):
        return None
    rows = [json.loads(l) for l in open(p)]
    mr = [r for r in rows if r.get("t", 0) >= 20]
    if len(mr) < 10:
        return None
    a, b = mr[0], mr[-1]
    dt = max(1e-9, b["t"] - a["t"])

    def c(r, k):
        return r.get(k, 0) or 0

    gen = (c(b, "vllm:generation_tokens[counter]") - c(a, "vllm:generation_tokens[counter]")) / dt
    drafts = (c(b, "vllm:spec_decode_num_drafts[counter]") - c(a, "vllm:spec_decode_num_drafts[counter]")) / dt
    prop = (c(b, "vllm:spec_decode_num_draft_tokens[counter]") - c(a, "vllm:spec_decode_num_draft_tokens[counter]")) / dt
    acc = (c(b, "vllm:spec_decode_num_accepted_tokens[counter]") - c(a, "vllm:spec_decode_num_accepted_tokens[counter]")) / dt
    return {"gen": gen, "drafts": drafts, "prop": prop, "acc": acc, "dt": dt}


def main():
    base = sys.argv[sys.argv.index("--base") + 1] if "--base" in sys.argv else BASE
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else os.path.join(base, "sps_table.jsonl")
    cells = []
    for K in KS:
        for C in CS:
            pts = [cell_stats(base, K, C, t) for t in (1, 2, 3)]
            pts = [p for p in pts if p]
            if not pts:
                continue
            n = len(pts)
            dt = sum(p["dt"] for p in pts)
            gen = sum(p["gen"] * p["dt"] for p in pts) / dt
            drafts = sum(p["drafts"] * p["dt"] for p in pts) / dt
            prop = sum(p["prop"] * p["dt"] for p in pts) / dt
            acc = sum(p["acc"] * p["dt"] for p in pts) / dt
            keff = 0 if K == "none" else int(K)
            if keff == 0:
                B, sps = C, gen / C
            else:
                # Counter identities (exact for vLLM spec decode):
                #   every decode step emits exactly one bonus token per active request,
                #   so steps/s S = (gen/s - accepted/s) / C
                #   verified tokens/step B = C (bonus positions) + proposed/s / S
                sps = (gen - acc) / C
                B = C + prop / sps if sps > 0 else None
            if sps is None or sps <= 0:
                continue
            cells.append({"K": K, "C": C, "B": B, "SPS": round(sps, 2),
                          "tok_s": round(gen, 1), "acc_rate": round(acc / prop, 3) if prop else None,
                          "per_req_tok_s": round(gen / C, 2)})
    with open(out, "w") as f:
        for r in cells:
            f.write(json.dumps(r) + "\n")
    print(f"{'K':>4} {'C':>3} {'B':>5} {'SPS steps/s':>12} {'tok/s':>8} {'acc_rate':>9} {'per-req':>8}")
    for r in sorted(cells, key=lambda x: (x["B"], x["K"])):
        print(f"{r['K']:>4} {r['C']:>3} {r['B']:>5} {r['SPS']:>12.2f} {r['tok_s']:>8.1f} "
              f"{str(r['acc_rate']):>9} {r['per_req_tok_s']:>8.2f}")
    print(f"\nwrote {out} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
