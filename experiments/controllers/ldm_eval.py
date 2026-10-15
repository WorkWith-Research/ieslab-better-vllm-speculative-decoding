#!/usr/bin/env python3
"""Phase-3: measured-cost oracle gap + causal LDM decision-policy evaluation.

Two contributions beyond Phase 2's cost-model bracketing:

1) MEASURED per-step cost model c(k). From Phase-1 we recorded, at each K and load,
   output_throughput (tok/s) and mean accept length L (tokens/step). Decode steps/sec
   = throughput / L, so the relative verification-step cost is c(k) ~ 1/(steps/sec).
   This is a *measured* sub-linear cost curve (not an assumed one), per workload.

2) CAUSAL LDM policy. A realistic lightweight decision model cannot see a request's
   full acceptance profile; it only observes past accept outcomes. We replay each
   request's recorded per-step accept sequence and, at each step, let the policy pick
   k from an online estimate of the request's joint-acceptance profile (EMA over the
   last W steps), optionally modulated by load. Realized value uses the *true* prefix
   acceptance (K-invariant). We then compare:
       best fixed-K  <  causal LDM  <=  oracle (full-knowledge)
   quantifying what fraction of the oracle gap a lightweight causal model actually
   recovers — the honest number for the LDM idea.

Usage:
  .venv/bin/python experiments/ldm_eval.py \
     --phase1 results/K1_r6 results/K2_r6 results/K4_r6 results/K8_r6 \
     --per-request results/P2s_K8_r6/oracle_per_request.jsonl --kmax 8
"""
import argparse, json, glob, math


def load_cost_model(phase1_dirs, kmax):
    """Derive measured relative verify-step cost c(k) from Phase-1 runs.

    For each K dir: steps/sec = output_throughput / mean_accept_len.
    c(k) = (steps/sec at K=1) / (steps/sec at k)  [relative, c(1)=1].
    Interpolate for k not directly measured (piecewise-linear in k on the
    measured points).
    """
    pts = {}
    for d in phase1_dirs:
        bp = glob.glob(d + "/*.bench.json")
        mp = glob.glob(d + "/metrics.jsonl")
        if not bp or not mp:
            continue
        b = json.load(open(bp[0]))
        tp = b.get("output_throughput")
        # mean accept len from the last metrics sample (cumulative counters)
        rows = [json.loads(l) for l in open(mp[0])]
        last = rows[-1]
        drafts = last.get("vllm:spec_decode_num_drafts[counter]", 0) or 0
        acc = last.get("vllm:spec_decode_num_accepted_tokens[counter]", 0) or 0
        if not drafts or not tp:
            continue
        L = 1 + acc / drafts                      # tokens per decode step
        import re as _re
        m = _re.search(r"K(\d+)_r", d)
        K = int(m.group(1)) if m else None
        if K is None:
            continue
        steps_per_sec = tp / L
        pts[K] = (1.0 / steps_per_sec)            # raw cost ∝ 1/steps-per-sec
    if not pts:
        return None
    base = pts[1]
    rel = {k: v / base for k, v in pts.items()}   # c(1)=1
    ks_sorted = sorted(rel)

    def c(k):
        if k in rel:
            return rel[k]
        # piecewise-linear interpolation between measured K points
        if k <= ks_sorted[0]:
            return rel[ks_sorted[0]]
        if k >= ks_sorted[-1]:
            return rel[ks_sorted[-1]]
        for a, b in zip(ks_sorted, ks_sorted[1:]):
            if a <= k <= b:
                t = (k - a) / (b - a)
                return rel[a] + t * (rel[b] - rel[a])
        return rel[ks_sorted[-1]]

    return c


def load_per_request(path):
    out = []
    for line in open(path):
        r = json.loads(line)
        J = r["J"]                       # length kmax+1, J[0]=1
        out.append({"p": r.get("p"), "J": J, "natural_K": r.get("natural_K")})
    return out


def captured(J, k):
    return sum(J[:k + 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1", nargs="+", required=True)
    ap.add_argument("--per-request", required=True)
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--window", type=int, default=8, help="LDM online estimate window")
    a = ap.parse_args()

    c = load_cost_model(a.phase1, a.kmax)
    if c is None:
        print("could not build cost model from phase1 dirs"); return
    ks = list(range(1, a.kmax + 1))
    reqs = load_per_request(a.per_request)
    n = len(reqs)

    print("=== measured per-step cost model c(k) (c(1)=1) ===")
    for k in ks:
        print(f"  c({k}) = {c(k):.3f}")

    # ---- causal LDM policy: estimate J from a sliding window of accept outcomes.
    # We only have the aggregate J per request (not the raw step sequence) in the
    # saved table, so we reconstruct a representative step stream from J: at each
    # position l, the fraction of steps that reach l is J[l]. We simulate by drawing
    # accept length for each step ~ the empirical distribution implied by J:
    #   P(acc = m) = J[m] - J[m+1]  (J[kmax+1]=0). This preserves all prefix probs.
    import random
    rng = random.Random(0)
    def sample_acc(J, kmax):
        # CDF over accept length 0..kmax
        probs = []
        for m in range(kmax + 1):
            pm = J[m] - (J[m + 1] if m + 1 <= kmax else 0.0)
            probs.append(max(0.0, pm))
        tot = sum(probs) or 1.0
        x = rng.random() * tot
        s = 0.0
        for m, p in enumerate(probs):
            s += p
            if x <= s:
                return m
        return kmax

    # pre-generate each request's step stream (equal horizon across requests) once
    N_STEPS = 200
    streams = []
    for r in reqs:
        J = r["J"]
        streams.append([sample_acc(J, a.kmax) for _ in range(N_STEPS)])

    # ---- unified metric: aggregate produced tokens / aggregate verify-cost over the
    # same N_STEPS horizon for every request (so fixed-K, LDM and oracle are directly
    # comparable; produced/step = accepted drafts + bonus). ----
    def fixed_value(k):
        prod = sum(captured(r["J"], k) * N_STEPS for r in reqs)
        cost = c(k) * N_STEPS * n
        return prod / cost

    # oracle: per-request best-k, full knowledge of the profile
    def oracle_value():
        prod = 0.0
        cost = 0.0
        for r in reqs:
            kb = max(ks, key=lambda kk: captured(r["J"], kk) / c(kk))
            prod += captured(r["J"], kb) * N_STEPS
            cost += c(kb) * N_STEPS
        return prod / cost

    # causal LDM replay (produced tokens per step = min(true_acc, k) + bonus; cost c(k)).
    # The policy only sees accept outcomes observed so far (sliding window), never
    # the request's full profile -> a realistic lightweight decision model.
    def ldm_replay(window):
        tot_prod = 0.0
        tot_cost = 0.0
        for r, stream in zip(reqs, streams):
            hist = []
            for acc in stream:
                if len(hist) >= window:
                    hist.pop(0)
                if hist:  # causal: decide from outcomes observed so far (steps < t)
                    eh = [1.0] * (a.kmax + 1)
                    for l in range(1, a.kmax + 1):
                        eh[l] = sum(1 for x in hist if x >= l) / len(hist)
                    k = max(ks, key=lambda kk: sum(eh[:kk + 1]) / c(kk))
                else:     # cold start: no observations yet -> minimal speculation
                    k = ks[0]
                tot_prod += min(acc, k) + 1      # accepted drafts + bonus token
                tot_cost += c(k)                 # relative verify-step cost
                hist.append(acc)
        return tot_prod / tot_cost

    ldm = ldm_replay(a.window)
    oracle = oracle_value()
    best_k = max(ks, key=fixed_value)
    fv_best = fixed_value(best_k)

    print("\n=== value (accepted tokens per unit verify-cost), measured c(k) ===")
    for k in ks:
        print(f"  fixed K={k}: {fixed_value(k):.4f}")
    print(f"  best fixed-K = {best_k} ({fv_best:.4f})")
    print(f"  causal LDM (window={a.window}) = {ldm:.4f}")
    print(f"  oracle (full knowledge)        = {oracle:.4f}")

    def gain(x, base):
        return 100 * (x / base - 1)
    print("\n=== gains vs best fixed-K ===")
    print(f"  causal LDM : {gain(ldm, fv_best):+.2f}%   (recovers "
          f"{100*(ldm-fv_best)/(oracle-fv_best):.0f}% of the oracle gap)")
    print(f"  oracle     : {gain(oracle, fv_best):+.2f}%")


if __name__ == "__main__":
    main()
