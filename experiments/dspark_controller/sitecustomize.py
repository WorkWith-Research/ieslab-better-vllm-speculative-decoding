"""In-loop DSpark-rule baseline: batch-level global greedy verification-length selection.

Faithful reimplementation of DSpark's DECISION RULE (arXiv:2607.05147) on our vLLM
prototype, using the closest honest signal proxies (documented in
docs/related-work-dspark.md §5):

  - confidence a_{r,j} (prefix-survival prob at draft position j)
      -> J_r[j] = P(acc >= j) over request r's last W observed steps (causal)
  - profiled throughput curve SPS(B)
      -> experiments/profile_sps.py table from our hardware (VLLM_SPS_TABLE)
  - objective  Theta = tau * SPS(B),  tau = sum_r (1 + sum_{j<=K_r} J_r[j]),
               B   = sum_r (1 + K_r)
  - global greedy: repeatedly extend the request whose marginal extension gives the
    largest Theta gain; stop when no extension improves Theta.

Activated ONLY when VLLM_DSPARK_OUT is set. Same three monkey-patch points as the LDM
controller (make_spec_decoding_stats / update_from_output / update_draft_token_ids),
same eager-mode requirement for variable K.

Env:
  VLLM_DSPARK_OUT   jsonl path for per-step decision log
  VLLM_SPS_TABLE    path to profiled SPS(B) table (jsonl from profile_sps.py)
  VLLM_DSPARK_KMAX  max draft length (default 8)
  VLLM_DSPARK_WINDOW acceptance-history window W (default 8)
"""
import json
import os
import time
from collections import defaultdict, deque

OUT = os.environ.get("VLLM_DSPARK_OUT")


def _load_sps(path):
    # table: list of {B, SPS} -> sorted points for linear interpolation
    pts = []
    if path and os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            for p in r.get("points", []):
                if p.get("B") and p.get("SPS"):
                    pts.append((p["B"], p["SPS"]))
    pts = sorted(set(pts))
    return pts


def _sps_interp(pts, B):
    if not pts:
        return None
    if B <= pts[0][0]:
        return pts[0][1]
    if B >= pts[-1][0]:
        return pts[-1][1]
    for (b0, s0), (b1, s1) in zip(pts, pts[1:]):
        if b0 <= B <= b1:
            if b1 == b0:
                return s1
            return s0 + (s1 - s0) * (B - b0) / (b1 - b0)
    return pts[-1][1]


def _install():
    if not OUT:
        return
    try:
        from vllm.v1.core.sched.scheduler import Scheduler
    except Exception as e:  # pragma: no cover
        print(f"[dspark] could not import Scheduler: {e}", flush=True)
        return

    KMAX = int(os.environ.get("VLLM_DSPARK_KMAX", "8"))
    W = int(os.environ.get("VLLM_DSPARK_WINDOW", "8"))
    SPS_PTS = _load_sps(os.environ.get("VLLM_SPS_TABLE"))
    if not SPS_PTS:
        print("[dspark] WARNING: no SPS table loaded — greedy degenerates to tau-only", flush=True)

    state = defaultdict(lambda: deque(maxlen=W))  # req -> accept history
    global_J = [1.0] * (KMAX + 1)                  # global prior over positions
    n_obs = [0]
    _fh = open(OUT, "a", buffering=1)

    def _Jvec(hist):
        J = [1.0] * (KMAX + 1)
        if hist:
            for l in range(1, KMAX + 1):
                J[l] = sum(1 for x in hist if x >= l) / len(hist)
        else:
            for l in range(1, KMAX + 1):
                J[l] = global_J[l]
        return J

    def _theta(tau, B):
        s = _sps_interp(SPS_PTS, B)
        return tau * s if s else float(tau)

    def _greedy(reqs):
        # reqs: {req_id: (K_current, Jvec)}. Returns {req_id: K_new}.
        K = {r: v[0] for r, v in reqs.items()}
        J = {r: v[1] for r, v in reqs.items()}
        if not reqs:
            return K
        tau = sum(1.0 + sum(J[r][:K[r] + 1]) for r in reqs)
        B = sum(1 + K[r] for r in reqs)
        theta = _theta(tau, B)
        improved = True
        while improved:
            improved = False
            best_r, best_gain, best_tau, best_theta = None, 0.0, tau, theta
            for r in reqs:
                if K[r] >= KMAX:
                    continue
                dtau = J[r][K[r] + 1]
                if dtau <= 0:
                    continue
                new_tau = tau + dtau
                new_theta = _theta(new_tau, B + 1)
                gain = new_theta - theta
                if gain > best_gain:
                    best_r, best_gain, best_tau, best_theta = r, gain, new_tau, new_theta
            if best_r is not None:
                K[best_r] += 1
                tau, B, theta = best_tau, B + 1, best_theta
                improved = True
        return K

    _orig_stats = Scheduler.make_spec_decoding_stats
    _orig_out = Scheduler.update_from_output
    _orig_drafts = Scheduler.update_draft_token_ids

    def _hooked_stats(self, spec_decoding_stats, num_draft_tokens,
                      num_accepted_tokens, num_invalid_spec_tokens, request_id):
        try:
            hist = state[request_id]
            hist.append(num_accepted_tokens)
            n_obs[0] += 1
            for l in range(1, KMAX + 1):
                global_J[l] = ((n_obs[0] - 1) * global_J[l] + (1 if num_accepted_tokens >= l else 0)) / n_obs[0]
            self._ds_pending = getattr(self, "_ds_pending", {})
            self._ds_pending[request_id] = (num_draft_tokens, hist)
        except Exception:
            pass
        return _orig_stats(self, spec_decoding_stats, num_draft_tokens,
                           num_accepted_tokens, num_invalid_spec_tokens, request_id)

    def _hooked_out(self, scheduler_output, model_output):
        res = _orig_out(self, scheduler_output, model_output)
        try:
            pend = getattr(self, "_ds_pending", None)
            if pend:
                reqs = {r: (min(v[0], KMAX), _Jvec(list(v[1]))) for r, v in pend.items()}
                Knew = _greedy(reqs)
                decs = [{"req": r, "Kobs": v[0], "Knew": Knew[r]} for r, v in reqs.items()]
                _fh.write(json.dumps({"t": round(time.monotonic(), 4), "n_active": len(decs),
                                      "decisions": decs}) + "\n")
                self._ds_kstar = Knew
                self._ds_pending = {}
        except Exception:
            pass
        return res

    def _hooked_drafts(self, draft_token_ids):
        res = _orig_drafts(self, draft_token_ids)
        try:
            ks = getattr(self, "_ds_kstar", None)
            for req_id in getattr(draft_token_ids, "req_ids", ()):
                # cold start / unknown: use global-prior greedy value if available else KMAX
                kstar = ks.get(req_id, KMAX) if ks else KMAX
                request = self.requests.get(req_id)
                if request is not None and len(request.spec_token_ids) > kstar:
                    request.spec_token_ids = request.spec_token_ids[:kstar]
        except Exception:
            pass
        return res

    Scheduler.make_spec_decoding_stats = _hooked_stats
    Scheduler.update_from_output = _hooked_out
    Scheduler.update_draft_token_ids = _hooked_drafts
    print(f"[dspark] DSpark-rule baseline active (KMAX={KMAX} W={W} "
          f"sps_points={len(SPS_PTS)}) -> {OUT}", flush=True)


_install()
