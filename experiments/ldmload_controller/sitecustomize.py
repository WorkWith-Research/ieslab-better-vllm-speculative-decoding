"""In-loop LIVE-LOAD-AWARE LDM (Phase 4.6, pre-registered in PROGRESS.md — Priority B).

Decision rule (revision #4, fixed before any run; supersedes revision #3's two-regime SPS model):
per request r, each decode step:
    N        = number of running requests this step          (measured)
    B_live   = total scheduled tokens this step              (measured; includes drafts + prefills)
    d_r      = actual draft tokens scheduled for r this step (measured; ngram often < K)
    a_bar    = EMA of batch mean accepted tokens / request-step (measured from per-request stats)
    k*_r     = argmax_{k in 0..KMAX}  [(N-1)*(1+a_bar) + 1 + sum_{l<=k} J_r[l]] * SPS(B_live + (k - d_r))
where J_r[l] = P(acc >= l) over r's last W observed steps (causal), and SPS is ONE static curve
fitted to ALL Phase-4.6 measured profile points (AR and SD cells together).

WHY UNIFIED (revision #3 -> #4, evidence-motivated per charter §13): revision #3 fitted separate
curves for "AR path" (k=0) and "SD path" (k>0). Offline reconstruction of the rev-#3 value function
from its OWN logged live state showed this double-penalized speculation: the SD profile points carry
a small fixed per-step overhead of the spec-decode code path on top of their (larger) B, so at equal
B the "SD curve" sat below the "AR curve" by 10-86% — an artifact of how the profile cells were
constructed (uniform-K), not a separate cost regime. In vLLM's continuous batch, a k=0 request and a
speculating request ride the SAME forward pass; step time is set by TOTAL scheduled tokens B. The
spec-decode-path overhead is constant across all k choices (the server always runs with spec decode
enabled), so it cancels in argmax over k. With one curve, the policy picks k*=8 at the logged C=8
live state and k*=0 at the logged C=32 live state — matching Phase 4.2 ground truth (K8 best at C=8;
AR≈K8 neutral at C=32) using only measured serving state. The delta uses d_r (ACTUAL scheduled
drafts, not the instructed K) because ngram self-drafting produces far fewer drafts than K on most
steps — revision #2's (k-k_cur)*N term was mechanically wrong for that reason.
NO concurrency label C appears anywhere — only measured (N, B_live, d_r, a_bar) + profile.

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
    """Load ALL measured profile points (AR + SD cells) into ONE SPS(B) curve."""
    pts = []
    if path and os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            if r.get("B") and r.get("SPS"):
                pts.append((r["B"], r["SPS"]))
    return _fit_curve(sorted(set(pts))) if len(pts) >= 3 else None


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
    S_P = _load_profile(os.environ.get("VLLM_LDMLOAD_PROFILE"))
    if not S_P:
        print(f"[ldmload] WARNING: profile missing/incomplete — controller will run with flat 30 SPS", flush=True)

    state = defaultdict(lambda: deque(maxlen=W))   # req -> accepted-token history (DRAFTED steps only)
    d_live = {}                                     # req -> actual scheduled draft tokens this step
    a_bar = [0.0]                                   # type: ignore[list-item]  EMA of mean acc/req-step over DRAFTED requests
    A_TAU = 0.95                                    # EMA decay (per step)
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

    def _decide(J, N, B_live, d_r, a_bar_v):
        # k* = argmax_k [(N-1)*(1+a_bar) + 1 + sum_{l<=k}J[l]] * SPS(B_live + (k - d_r))
        best, bv = 0, -1.0
        others = (N - 1) * (1.0 + a_bar_v) if N > 1 else 0.0
        for k in range(0, KMAX + 1):
            E = sum(J[1:k + 1])
            B = max(1.0, B_live + (k - d_r))
            s = _sps(S_P, B) if S_P else 30.0
            v = (others + 1.0 + E) * s
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
            d_live[request_id] = int(num_draft_tokens or 0)   # ACTUAL scheduled drafts this step
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
                # batch mean accepted/request-step (measured) over DRAFTED requests only,
                # INCLUDING the +1 bonus token each drafted request emits every step:
                #   a_bar = (sum acc + n_drafted) / n_drafted
                # (Zero-draft steps never reach make_spec_decoding_stats, so they cannot poison this.)
                drafted = [r for r in pend if d_live.get(r, 0) > 0]
                if drafted:
                    step_acc = (sum(state[r][-1] for r in drafted) + len(drafted)) / len(drafted)
                    a_bar[0] = step_acc if a_bar[0] == 0.0 else A_TAU * a_bar[0] + (1 - A_TAU) * step_acc
                knew = {}
                decs = []
                for r in pend:
                    d_r = d_live.get(r, 0)
                    hist = state.get(r)
                    if not hist:
                        ks = 4   # cold start (pre-registered)
                    else:
                        ks = _decide(_Jvec(list(hist)), N, B_live, d_r, a_bar[0] or 0.0)
                    knew[r] = ks
                    decs.append({"req": r, "Kobs": min(pend[r][0], KMAX), "d": d_r, "kstar": ks})
                self._ll_kstar = knew
                _fh.write(json.dumps({"t": round(time.monotonic(), 4), "N": N,
                                      "B_live": round(B_live, 1), "a_bar": round(a_bar[0] or 0.0, 3),
                                      "decisions": decs}) + "\n")
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
    print(f"[ldmload] live-load-aware LDM active (rev #4, unified SPS curve; KMAX={KMAX} W={W} "
          f"profile={'yes' if S_P else 'NO'}) -> {OUT}", flush=True)


_install()
