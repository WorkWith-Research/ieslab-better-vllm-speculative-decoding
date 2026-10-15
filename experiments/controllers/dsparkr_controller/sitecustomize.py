"""In-loop DSpark-rule baseline — CORRECTED proxy (P4.5R; pre-registered in
docs/dspark-proxy-validity.md §6, 2026-10-09).

Same DECISION RULE as experiments/dspark_controller/sitecustomize.py (Phase 4.5):
  - J_r[l] = P(acc >= l) over request r's last W observed steps (causal)
  - Theta = tau * SPS(B),  tau = sum_r (1 + sum_{j<=K_r} J_r[j]), B = sum_r (1+K_r)
  - global greedy from the all-zero baseline; extend while Theta improves;
    SD-off (all K=0) is an explicit, reachable outcome.

THREE MECHANICAL FIXES ONLY (decision rule byte-for-byte unchanged):
  F1 — decide for EVERY running request each step; decisions persist in last_kstar
       so a K=0 decision survives zero-draft steps; no implicit KMAX default for
       map-absent requests — unknown/cold-start uses the documented cold-start
       value (KMAX).
  F4 — B = true batch: total_num_scheduled_tokens of the step (decode + drafts +
       prefill, verified in Phase 4.6); NO clamp to the profile range — beyond
       the table's max B the curve extrapolates with its last (monotone-
       decreasing) segment slope, per the Phase 5 M1/M2 findings.
  F2 — censoring: a step that observed only k drafts contributes to J[l] for
       l <= k only; positions > k are UNOBSERVED (excluded from the denominator),
       not counted as rejections.

Activated ONLY when VLLM_DSPARKR_OUT is set. Env:
  VLLM_DSPARKR_OUT    jsonl path for per-step decision log
  VLLM_SPS_TABLE      profiled SPS(B) table (jsonl; same file as Phase 4.5: results/p2/sps_table.jsonl)
  VLLM_DSPARKR_KMAX   max draft length (default 8)
  VLLM_DSPARKR_WINDOW acceptance-history window W (default 8)
"""
import json
import os
import time
from collections import defaultdict, deque

OUT = os.environ.get("VLLM_DSPARKR_OUT")


def _load_sps(path):
    pts = []
    if path and os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            if "points" in r:
                for p in r["points"]:
                    if p.get("B") and p.get("SPS"):
                        pts.append((p["B"], p["SPS"]))
            elif r.get("B") and r.get("SPS"):
                pts.append((r["B"], r["SPS"]))
    return sorted(set(pts))


def _sps_interp(pts, B):
    """Interpolate inside the table; EXTRAPOLATE beyond it with the last segment's
    slope (monotone-decreasing per Phase 5 M1/M2). No clamp (F4)."""
    if not pts:
        return None
    if B <= pts[0][0]:
        return pts[0][1]
    if B >= pts[-1][0]:
        (b0, s0), (b1, s1) = pts[-2], pts[-1]
        slope = 0.0 if b1 == b0 else (s1 - s0) / (b1 - b0)
        return s1 + slope * (B - b1)
    for (b0, s0), (b1, s1) in zip(pts, pts[1:]):
        if b0 <= B <= b1:
            if b1 == b0:
                return s1
            return s0 + (s1 - s0) * (B - b0) / (b1 - b0)
    return pts[-1][1]


def _Jvec_censored(hist, prior):
    """Censoring-aware J (F2): hist = list of (k_observed, accepted). For position l,
    only steps with k_observed >= l are informative; J[l] = mean(accepted >= l) over
    those. Positions with no observations fall back to the global prior (itself
    computed censored)."""
    J = list(prior)
    if not hist:
        return J
    for l in range(1, len(J)):
        obs = [a for k, a in hist if k >= l]
        if obs:
            J[l] = sum(1 for a in obs if a >= l) / len(obs)
    return J


def _greedy(reqs, KMAX, sps_pts):
    """Pre-registered rule (unchanged from Phase 4.5): start all-zero; repeatedly
    extend the request whose one-step extension gives the largest Theta gain; stop
    when none improves. Returns {req: K_new}."""
    if not reqs:
        return {}
    n = len(reqs)
    J = {r: v[1] for r, v in reqs.items()}

    def theta(tau, B):
        s = _sps_interp(sps_pts, B)
        return tau * s if s else float(tau)

    K = {r: 0 for r in reqs}
    tau = float(n)                      # all-zero baseline: 1 bonus token per request
    B = float(n)
    theta_base = theta(tau, B)
    while True:
        best_gain, best_r = 0.0, None   # reset each pass (else a stale positive gain
                                        # re-extends a KMAX-saturated request -> OOR)
        for r in reqs:
            if K[r] >= KMAX:
                continue
            dtau = J[r][K[r] + 1]
            if dtau <= 0:
                continue
            gain = theta(tau + dtau, B + 1) - theta(tau, B)
            if gain > best_gain:
                best_gain, best_r = gain, r
        if best_r is None or best_gain <= 0.0:
            break
        K[best_r] += 1
        tau += J[best_r][K[best_r]]
        B += 1.0
    if theta(tau, B) < theta_base:      # SD-off fallback (explicit, reachable)
        return {r: 0 for r in reqs}
    return K


def _install():
    if not OUT:
        return
    try:
        from vllm.v1.core.sched.scheduler import Scheduler
    except Exception as e:  # pragma: no cover
        print(f"[dsparkR] could not import Scheduler: {e}", flush=True)
        return

    KMAX = int(os.environ.get("VLLM_DSPARKR_KMAX", "8"))
    W = int(os.environ.get("VLLM_DSPARKR_WINDOW", "8"))
    SPS_PTS = _load_sps(os.environ.get("VLLM_SPS_TABLE"))
    if not SPS_PTS:
        print("[dsparkR] WARNING: no SPS table loaded — greedy degenerates to tau-only", flush=True)

    state = defaultdict(lambda: deque(maxlen=W))   # req -> deque of (k_observed, accepted)
    step_d = {}                                     # req -> actual scheduled drafts THIS step
    last_kstar = {}                                 # PERSISTENT decisions (F1)
    global_hist = []                                # all observations (censored prior)
    n_obs = [0]
    _fh = open(OUT, "a", buffering=1)

    def _prior():
        J = [1.0] * (KMAX + 1)
        if global_hist:
            for l in range(1, KMAX + 1):
                obs = [a for k, a in global_hist if k >= l]
                if obs:
                    J[l] = sum(1 for a in obs if a >= l) / len(obs)
        return J

    _orig_stats = Scheduler.make_spec_decoding_stats
    _orig_out = Scheduler.update_from_output
    _orig_drafts = Scheduler.update_draft_token_ids

    def _hooked_stats(self, spec_decoding_stats, num_draft_tokens,
                      num_accepted_tokens, num_invalid_spec_tokens, request_id):
        try:
            k_obs = int(num_draft_tokens or 0)
            state[request_id].append((k_obs, int(num_accepted_tokens or 0)))
            global_hist.append((k_obs, int(num_accepted_tokens or 0)))
            n_obs[0] += 1
            step_d[request_id] = k_obs
        except Exception:
            pass
        return _orig_stats(self, spec_decoding_stats, num_draft_tokens,
                           num_accepted_tokens, num_invalid_spec_tokens, request_id)

    def _hooked_out(self, scheduler_output, model_output):
        res = _orig_out(self, scheduler_output, model_output)
        try:
            N = len(self.running) or (len(step_d) if step_d else 0)
            # F4: B = true batch (decode + drafts + prefill). total_num_scheduled_tokens
            # verified in Phase 4.6 to include scheduled spec drafts. Fallback: N + drafts.
            tns = int(getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0)
            B_live = float(tns if tns > 0 else N + sum(step_d.values()))
            prior = _prior()
            # F1: decide for EVERY running request each step.
            reqs = {}
            decs = []
            for req in list(self.running):
                r = req.request_id
                d_r = step_d.get(r, 0)
                hist = state[r] if (r in state and state[r]) else None
                Jv = _Jvec_censored(list(hist), prior) if hist else list(prior)
                reqs[r] = (min(d_r, KMAX) if d_r else KMAX, Jv)
                decs.append({"req": r, "Kobs": d_r, "d": d_r})
            Knew = _greedy(reqs, KMAX, SPS_PTS)
            for d in decs:
                d["Knew"] = Knew.get(d["req"], KMAX)
            # F1: persist (survives zero-draft steps; K=0 stays K=0).
            for r, k in Knew.items():
                last_kstar[r] = k
            ks_vals = list(Knew.values())
            _fh.write(json.dumps({
                "t": round(time.monotonic(), 4), "N": N, "B_live": round(B_live, 1),
                "n_decided": len(ks_vals),
                "meanK": round(sum(ks_vals) / len(ks_vals), 3) if ks_vals else None,
                "fracK0": round(sum(1 for k in ks_vals if k == 0) / len(ks_vals), 4) if ks_vals else None,
                "decisions": decs,
            }) + "\n")
            step_d.clear()
        except Exception:
            pass
        return res

    def _hooked_drafts(self, draft_token_ids):
        res = _orig_drafts(self, draft_token_ids)
        try:
            for req_id in getattr(draft_token_ids, "req_ids", ()):
                # F1: persistent map; unknown/cold-start -> documented KMAX.
                kstar = last_kstar.get(req_id, KMAX)
                request = self.requests.get(req_id)
                if request is not None and len(request.spec_token_ids) > kstar:
                    request.spec_token_ids = request.spec_token_ids[:kstar]
        except Exception:
            pass
        return res

    Scheduler.make_spec_decoding_stats = _hooked_stats
    Scheduler.update_from_output = _hooked_out
    Scheduler.update_draft_token_ids = _hooked_drafts
    print(f"[dsparkR] corrected DSpark-rule proxy active (P4.5R; KMAX={KMAX} W={W} "
          f"sps_points={len(SPS_PTS)} B_max_profile="
          f"{SPS_PTS[-1][0] if SPS_PTS else 'n/a'}) -> {OUT}", flush=True)


_install()
