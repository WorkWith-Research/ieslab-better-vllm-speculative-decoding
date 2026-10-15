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

REV-#4c PLUMBING FIXES (mechanical only; decision rule byte-for-byte unchanged — see PROGRESS.md):
1. ENFORCEMENT: decide for EVERY running request each step; decisions persist in last_kstar so a
   k*=0 request stays k*=0 across zero-draft steps (rev-#4 fell back to cold-start K=4 one step
   later, so "SD-off" was never actually enforced).
2. COLD START: KMAX instead of 4 (a ceiling of 4 makes k*=8 unreachable — the controller can never
   observe acceptance beyond position 4).
3. THIS-STEP STATE: per-request draft counts now live in a closure dict (step_d) populated by the
   stats hook and cleared at the END of each update_from_output. The old d_live/d_seen/instance-attr
   plumbing leaked stale values across steps (d_r for k*=0 requests carried last step's drafts,
   corrupting the value function's (k - d_r) delta) — instance attributes were also observed to be
   empty at read time in this vLLM build, so closure state is used instead.
4. B_live = total_num_scheduled_tokens (verified against source + live instrumentation to INCLUDE
   scheduled spec drafts: N=96 with 8 drafts -> total_sched=104), falling back to N + step drafts.

Activated ONLY when VLLM_LDMLOAD_OUT is set. Env:
  VLLM_LDMLOAD_OUT    jsonl path for per-step decision log
  VLLM_LDMLOAD_KMAX   max draft length (default 8)
  VLLM_LDMLOAD_WINDOW acceptance-history window W (default 8)
  VLLM_LDMLOAD_FORCE0 diagnostic: force k*=0 for every request (measures the SD-off floor on a
                      spec-enabled server; NOT part of the H-4.6 arm — used to decompose the gap)
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
    FORCE0 = os.environ.get("VLLM_LDMLOAD_FORCE0") == "1"
    S_P = _load_profile(os.environ.get("VLLM_LDMLOAD_PROFILE"))
    if not S_P:
        print(f"[ldmload] WARNING: profile missing/incomplete — controller will run with flat 30 SPS", flush=True)

    state = defaultdict(lambda: deque(maxlen=W))   # req -> accepted-token history (DRAFTED steps only)
    khist = {}                                      # req -> FROZEN acceptance history (persists after drafting stops)
    step_d = {}                                     # req -> ACTUAL scheduled draft tokens THIS step (cleared each step)
    last_kstar = {}                                 # req -> last decided k* (PERSISTENT; survives zero-draft steps)
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
            khist[request_id] = list(hist)          # FROZEN snapshot (persists after drafting stops)
            n_obs[0] += 1
            for l in range(1, KMAX + 1):
                global_J[l] = ((n_obs[0] - 1) * global_J[l] + (1 if num_accepted_tokens >= l else 0)) / n_obs[0]
            step_d[request_id] = int(num_draft_tokens or 0)   # ACTUAL scheduled drafts THIS step
        except Exception:
            pass
        return _orig_stats(self, spec_decoding_stats, num_draft_tokens,
                           num_accepted_tokens, num_invalid_spec_tokens, request_id)

    def _hooked_out(self, scheduler_output, model_output):
        res = _orig_out(self, scheduler_output, model_output)
        try:
            N = len(self.running) or (len(step_d) if step_d else 0)
            # B_live = total scheduled tokens this step. Verified (source + live instrumentation on
            # vLLM 0.19.1) that total_num_scheduled_tokens INCLUDES the scheduled spec drafts
            # (N=96 with 8 drafts -> total_sched=104), so it is exactly the pre-registered signal
            # "running requests + scheduled speculative tokens". Fallback: N + this-step drafts.
            tns = int(getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0)
            B_live = float(tns if tns > 0 else N + sum(step_d.values()))
            # batch mean accepted/request-step (measured) over DRAFTED requests only,
            # INCLUDING the +1 bonus token each drafted request emits every step:
            #   a_bar = (sum acc + n_drafted) / n_drafted
            # (Zero-draft steps never reach make_spec_decoding_stats, so they cannot poison this.)
            drafted = [r for r, d in step_d.items() if d > 0]
            if drafted:
                step_acc = (sum(state[r][-1] for r in drafted) + len(drafted)) / len(drafted)
                a_bar[0] = step_acc if a_bar[0] == 0.0 else A_TAU * a_bar[0] + (1 - A_TAU) * step_acc
            # ENFORCEMENT FIX (rev-#4c, mechanical): decide for EVERY running request each step;
            # decisions persist in last_kstar so k*=0 stays k*=0 across zero-draft steps. Requests
            # without a live drafted history use their FROZEN acceptance history (khist).
            decs = []
            for req in list(self.running):
                r = req.request_id
                d_r = step_d.get(r, 0)   # this-step drafts only (step_d is cleared at end of step)
                hist = state[r] if (r in state and state[r]) else khist.get(r)
                if FORCE0:
                    ks = 0
                elif not hist:
                    # Cold start (rev-#4c, mechanical): KMAX, not 4. A ceiling of 4 makes k*=8
                    # UNREACHABLE — the controller can never observe acceptance at positions >4.
                    # An optimistic KMAX start lets each request observe deep acceptance once before
                    # the regime term takes over (same rationale as the Phase 4.5 LDM cold-start fix).
                    ks = KMAX
                else:
                    ks = _decide(_Jvec(list(hist)), N, B_live, d_r, a_bar[0] or 0.0)
                last_kstar[r] = ks
                decs.append({"req": r, "Kobs": min(step_d.get(r, 0), KMAX), "d": d_r, "kstar": ks})
            self._ll_kstar = dict(last_kstar)
            _fh.write(json.dumps({"t": round(time.monotonic(), 4), "N": N,
                                  "B_live": round(B_live, 1), "a_bar": round(a_bar[0] or 0.0, 3),
                                  "decisions": decs}) + "\n")
            step_d.clear()
        except Exception:
            pass
        return res

    def _hooked_drafts(self, draft_token_ids):
        res = _orig_drafts(self, draft_token_ids)
        try:
            ks = getattr(self, "_ll_kstar", None)
            n_found = n_trunc = n_skip = 0
            for req_id in getattr(draft_token_ids, "req_ids", ()):
                kstar = ks.get(req_id, KMAX) if ks else KMAX
                request = self.requests.get(req_id)
                if request is None:
                    n_skip += 1
                    continue
                cur = len(request.spec_token_ids)
                if cur > kstar:
                    request.spec_token_ids = request.spec_token_ids[:kstar]
                    n_trunc += 1
                else:
                    n_found += 1
            # DEBUG (enforcement check): log every Nth call so we can verify truncation actually happens.
            _hooked_drafts._calls = getattr(_hooked_drafts, "_calls", 0) + 1
            if _hooked_drafts._calls % 200 == 1:
                rids = list(getattr(draft_token_ids, "req_ids", ()))
                sample = {}
                for rid in rids[:3]:
                    rq = self.requests.get(rid)
                    sample[rid[-6:]] = (len(rq.spec_token_ids) if rq is not None else None,
                                        ks.get(rid) if ks else None)
                print(f"[ldmload][drafts] call#{_hooked_drafts._calls} ks={'yes' if ks else 'NO'} "
                      f"ks_n={len(ks) if ks else 0} n_req={len(rids)} "
                      f"trunc={n_trunc} keep={n_found} skip={n_skip} sample(spec_len,kstar)={sample}", flush=True)
        except Exception as e:
            print(f"[ldmload][drafts] EXC {type(e).__name__}: {e}", flush=True)
        return res

    Scheduler.make_spec_decoding_stats = _hooked_stats
    Scheduler.update_from_output = _hooked_out
    Scheduler.update_draft_token_ids = _hooked_drafts
    print(f"[ldmload] live-load-aware LDM active (rev #4c; KMAX={KMAX} W={W} force0={FORCE0} "
          f"profile={'yes' if S_P else 'NO'}) -> {OUT}", flush=True)


_install()
