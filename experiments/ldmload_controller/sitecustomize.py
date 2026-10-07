"""In-loop LIVE-LOAD-AWARE LDM (Phase 4.6, pre-registered in PROGRESS.md — Priority B).

Decision rule (revision #2, fixed before any run): per request r, each decode step:
    N        = number of running requests this step          (measured)
    B_live   = total scheduled tokens this step              (measured; includes drafts + prefills)
    k*_r     = argmax_{k in 0..KMAX}  N * (1 + sum_{l<=k} J_r[l]) * SPS_path(B_live + (k - k_cur)*N)
where J_r[l] = P(acc >= l) over r's last W observed steps (causal), k_cur = r's K last step,
and SPS_path is the static profile fitted to Phase-4.6 measured points (AR path for k=0, SD
path for k>0). NO concurrency label C appears anywhere — only measured (N, B_live) + profile.

Mechanics: server runs with SPEC_METHOD=ngram K=KMAX; this controller truncates each request's
drafts to its k* in update_draft_token_ids (same enforcement as the Phase 4.5 LDM controller).
k*=0 is a reachable outcome (SD off for that request).

Activated ONLY when VLLM_LDMLOAD_OUT is set. Env:
  VLLM_LDMLOAD_OUT    jsonl path for per-step decision log
  VLLM_LDMLOAD_KMAX   max draft length (default 8)
  VLLM_LDMLOAD_WINDOW acceptance-history window W (default 8)
"""
import json
import math
import os
import time
from collections import defaultdict, deque

OUT = os.environ.get("VLLM_LDMLOAD_OUT")


def _fit_curve(pts):
    """Fit SPS(B) = A/(Bc + Cc*B^D) on log-SPS (grid + local search). Deterministic seed."""
    import random
    rng = random.Random(1234)
    best, bf = None, 1e9
    for A in [30, 40, 50, 60, 70, 80, 90, 100, 120, 150]:
        for Bc in [0.0, 0.5, 1.0, 2.0, 5.0]:
            for Cc in [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]:
                for D in [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5, 3.0]:
                    e = sum((math.log(s) - math.log(A / (Bc + Cc * b ** D))) ** 2 for b, s in pts)
                    if e < bf:
                        bf, best = e, (A, Bc, Cc, D)
    cur, curf = best, bf
    for _ in range(8000):
        nb = tuple(max(1e-7, x * (1 + rng.uniform(-0.2, 0.2))) for x in cur)
        nf = sum((math.log(s) - math.log(nb[0] / (nb[1] + nb[2] * b ** nb[3]))) ** 2 for b, s in pts)
        if nf < curf:
            cur, curf = nb, nf
    return cur


def _sps(p, B):
    return p[0] / (p[1] + p[2] * max(1.0, B) ** p[3])


def _load_profile(path):
    ar, sd = [], []
    if path and os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            if r.get("B") and r.get("SPS"):
                (ar if r.get("path") == "AR" else sd).append((r["B"], r["SPS"]))
    ar_p = _fit_curve(sorted(ar)) if len(ar) >= 3 else None
    sd_p = _fit_curve(sorted(sd)) if len(sd) >= 3 else None
    return ar_p, sd_p


def _install():
    if not OUT:
        return
    try:
        from vllm.v1.core.sched.scheduler import Scheduler
    except Exception as e:  # pragma: no cover
        print(f"[ldmload] could not import Scheduler: {e}", flush=True)
        return

    KMAX = int(os.environ.get("VLLM_LDMLOAD_KMAX", "8"))
    W = int(os.environ.get("VLLM_LDMLOAD_WINDOW", "8"))
    AR_P, SD_P = _load_profile(os.environ.get("VLLM_LDMLOAD_PROFILE"))
    if not (AR_P and SD_P):
        print(f"[ldmload] WARNING: profile incomplete (ar={bool(AR_P)} sd={bool(SD_P)}) — "
              f"controller will run AR-only-ish", flush=True)

    state = defaultdict(lambda: deque(maxlen=W))   # req -> accepted-token history
    k_cur = {}                                      # req -> K last step
    global_J = [1.0] * (KMAX + 1)                   # cold-start prior
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

    def _decide(J, N, B_live, kc):
        # k* = argmax_k N*(1+sum_{l<=k}J[l])*SPS_path(B_live + (k-kc)*N)
        best, bv = 0, -1.0
        for k in range(0, KMAX + 1):
            E = sum(J[1:k + 1])
            B = max(1.0, B_live + (k - kc) * N)
            p = AR_P if k == 0 else SD_P
            s = _sps(p, B) if p else 30.0
            v = N * (1.0 + E) * s
            if v > bv:
                bv, best = v, k
        return best

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
            self._ll_pending = getattr(self, "_ll_pending", {})
            self._ll_pending[request_id] = (num_draft_tokens,)
        except Exception:
            pass
        return _orig_stats(self, spec_decoding_stats, num_draft_tokens,
                           num_accepted_tokens, num_invalid_spec_tokens, request_id)

    def _hooked_out(self, scheduler_output, model_output):
        res = _orig_out(self, scheduler_output, model_output)
        try:
            pend = getattr(self, "_ll_pending", None)
            if pend:
                N = len(self.running) or len(pend)
                B_live = float(getattr(scheduler_output, "total_num_scheduled_tokens", 0)) \
                    or sum(len(v) for v in getattr(scheduler_output, "scheduled_spec_decode_tokens", {}).values()) + N
                knew = {}
                decs = []
                for r in pend:
                    kc = k_cur.get(r, KMAX)
                    hist = state.get(r)
                    if not hist:
                        ks = 4   # cold start (pre-registered)
                    else:
                        ks = _decide(_Jvec(list(hist)), N, B_live, kc)
                    k_cur[r] = ks
                    knew[r] = ks
                    decs.append({"req": r, "Kobs": min(pend[r][0], KMAX), "kstar": ks})
                self._ll_kstar = knew
                _fh.write(json.dumps({"t": round(time.monotonic(), 4), "N": N,
                                      "B_live": round(B_live, 1), "decisions": decs}) + "\n")
                self._ll_pending = {}
        except Exception:
            pass
        return res

    def _hooked_drafts(self, draft_token_ids):
        res = _orig_drafts(self, draft_token_ids)
        try:
            ks = getattr(self, "_ll_kstar", None)
            for req_id in getattr(draft_token_ids, "req_ids", ()):
                kstar = ks.get(req_id, 4) if ks else 4
                request = self.requests.get(req_id)
                if request is not None and len(request.spec_token_ids) > kstar:
                    request.spec_token_ids = request.spec_token_ids[:kstar]
        except Exception:
            pass
        return res

    Scheduler.make_spec_decoding_stats = _hooked_stats
    Scheduler.update_from_output = _hooked_out
    Scheduler.update_draft_token_ids = _hooked_drafts
    print(f"[ldmload] live-load-aware LDM active (KMAX={KMAX} W={W} "
          f"ar_pts={'yes' if AR_P else 'NO'} sd_pts={'yes' if SD_P else 'NO'}) -> {OUT}", flush=True)


_install()
