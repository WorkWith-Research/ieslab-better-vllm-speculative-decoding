"""In-loop BATCH-LEVEL speculation-aware scheduler (Phase 5, H-5.1; pre-registered in PROGRESS.md).

EXTENDS the Phase 4.6 live-load LDM with TWO new scheduler-level mechanisms that Phase 4.6's K-only
action space could not express:

1. BATCH-LEVEL SD on/off DECISION. Per-request greedy fails at saturation because each request's
   argmax prices its OWN accepted tokens against SPS(B) but NOT the slowdown its drafts impose on
   the other N-1 requests (a negative externality that grows with B). The batch-level decision sees
   the total: K_batch = argmax_{k in 0..KMAX}  SPS(B_eff + N_dec*k) * (1 + mean_l[k])
   where N_dec = # decode-phase requests, B_eff = clamp(N_dec + this-step drafts, max=profile-B-max),
   mean_l[k] = mean over DECODE requests of sum_{l<=k} J_r[l]. The single SPS term carries the
   slowdown of ALL decode requests' N_dec*k extra scheduled tokens, so at saturation K_batch commits
   to 0 (SD off for the whole batch) even when a per-request minority would keep speculating.

2. DRAFT-PASS SKIP action. When OFF (K_batch==0), the worker's ngram draft pass is skipped entirely
   (GPUModelRunner.propose_draft_token_ids returns early). vLLM 0.19.1 runs this pass unconditionally
   whenever speculative_config is set (gated on `spec_config is not None`, NOT on whether any drafts
   were scheduled) — that unconditional per-step cost is the ~4.7% fixed spec-path overhead Phase 4.6's
   force-0 floor measured. Skipping it removes that overhead. The main forward path is already AR when
   scheduled_spec_decode_tokens is empty (gpu_model_runner.py:2040).

WHY LOSSLESS: speculative decoding verifies every draft against the target model; a stale/absent draft
is simply rejected, never produces a wrong token. Skipping the draft pass during OFF periods leaves the
ngram drafter's token-history buffer stale, so acceptance may dip for a few steps on resume — but output
correctness is unaffected and the main forward pass does not read the drafter's buffers.

NO concurrency label C anywhere — only measured (N_dec, B_eff, d_r, J_r) + the static SPS profile.
TP=1 -> UniProcExecutor -> scheduler and worker share one process, so the module-level OFF flag set by
the scheduler hook is visible to the worker's propose_draft_token_ids.

MECHANICAL CORRECTIONS (post-smoke, charter §13 — decision rule shape unchanged; see PROGRESS.md):
M1. DECODE-ONLY load signal. The first smoke OOM'd: B_live=total_num_scheduled_tokens INCLUDES chunked
    prefill (observed 1095 at C=96), and feeding that into the decode-fitted SPS curve extrapolates
    out-of-range, spuriously favoring high K during prefill waves -> mass k*=8 on top of a saturated
    prefill batch -> OOM in the rejection sampler's penalty allocation. Speculation only acts on
    DECODE-phase requests (vLLM schedules spec tokens for them), so N_dec and B_eff are computed from
    decode-phase requests only, and B_eff is clamped to the profile's max fitted B (in-distribution).
M2. ALLOCATION MATCHES THE PRICED MODEL. When ON at K_batch=k, k*=k for every DECODE request (uniform),
    k*=0 for prefill requests. The first draft used Phase 4.6's per-request rule on the ON path, which
    scheduled sum_k*=728 against a priced budget of ~91 tokens (frozen acceptance history over-estimated
    J_r after OFF periods starved make_spec_decoding_stats) -> OOM. Uniform K_batch is exactly what
    SPS(B_eff + N_dec*k) priced, and is the truest expression of H-5.1 (the batch decision IS the
    allocation; per-request greedy is what Phase 4.6 falsified).

Activated ONLY when VLLM_LDM_BATCH_OUT is set. Env:
  VLLM_LDM_BATCH_OUT     jsonl path for per-step decision log
  VLLM_LDM_BATCH_KMAX    max draft length (default 8)
  VLLM_LDM_BATCH_WINDOW  acceptance-history window W (default 8)
  VLLM_LDM_BATCH_PROFILE sps_profile.jsonl path
  VLLM_LDM_BATCH_FORCEOFF=1  diagnostic: force OFF always (skip draft pass every step; measures the
                             pure "spec path fully disabled at runtime" behavior — should ~= AR)
"""
import json
import math
import os
import time
from collections import defaultdict, deque

OUT = os.environ.get("VLLM_LDM_BATCH_OUT")

# Module-level OFF flag: set by the scheduler hook (update_from_output), read by the worker's
# propose_draft_token_ids. Same process under TP=1 (UniProcExecutor).
OFF = [False]


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
    pts = []
    if path and os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            if r.get("B") and r.get("SPS"):
                pts.append((r["B"], r["SPS"]))
    if len(pts) < 3:
        return None, None
    return _fit_curve(sorted(set(pts))), max(b for b, _ in pts)


def _install():
    if not OUT:
        return
    try:
        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except Exception as e:  # pragma: no cover
        print(f"[ldm_batch] could not import hooks: {e}", flush=True)
        return

    KMAX = int(os.environ.get("VLLM_LDM_BATCH_KMAX", "8"))
    W = int(os.environ.get("VLLM_LDM_BATCH_WINDOW", "8"))
    FORCEOFF = os.environ.get("VLLM_LDM_BATCH_FORCEOFF") == "1"
    S_P, B_MAX = _load_profile(os.environ.get("VLLM_LDM_BATCH_PROFILE"))
    if not S_P:
        print(f"[ldm_batch] WARNING: profile missing/incomplete — flat 30 SPS", flush=True)

    state = defaultdict(lambda: deque(maxlen=W))   # req -> accepted-token history (DRAFTED steps only)
    khist = {}                                      # req -> FROZEN acceptance history
    step_d = {}                                     # req -> ACTUAL scheduled draft tokens THIS step
    last_kstar = {}                                 # req -> last decided k*
    sched_tok = defaultdict(int)                    # req -> cumulative SCHEDULED tokens (M1: decode-phase detection)
    a_bar = [0.0]                                   # EMA of mean acc/req-step over DRAFTED requests
    A_TAU = 0.95
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

    def _batch_k(N_dec, B_eff, mean_l):
        # Batch-level decision: argmax_k SPS(B_eff + N_dec*k) * (1 + mean_l[k]).
        # The single SPS term carries the slowdown of ALL N_dec decode requests' N_dec*k extra
        # scheduled tokens (the externality per-request greedy misses).
        # NO CLAMP to profile-B-max: clamping makes SPS constant for all k above the knee, so the
        # argmax degenerates to "pick max k" exactly at high load — destroying the externality pricing
        # that is the point of H-5.1 (verified numerically: with clamp, C96 steady -> K_batch=8; without,
        # -> K_batch=0). Extrapolating SPS(B)=A/(Bc+Cc*B^D) beyond the fitted range is monotone-decreasing
        # and physically sensible (bigger batch -> slower per-request), so it prices large B correctly.
        best, bv = 0, -1.0
        for k in range(0, KMAX + 1):
            B = max(1.0, B_eff + N_dec * k)
            s = _sps(S_P, B) if S_P else 30.0
            v = s * (1.0 + mean_l[k])
            if v > bv:
                bv, best = v, k
        return best

    _orig_stats = Scheduler.make_spec_decoding_stats
    _orig_out = Scheduler.update_from_output
    _orig_drafts = Scheduler.update_draft_token_ids
    _orig_propose = GPUModelRunner.propose_draft_token_ids

    def _hooked_stats(self, spec_decoding_stats, num_draft_tokens,
                      num_accepted_tokens, num_invalid_spec_tokens, request_id):
        try:
            hist = state[request_id]
            hist.append(num_accepted_tokens)
            khist[request_id] = list(hist)
            n_obs[0] += 1
            for l in range(1, KMAX + 1):
                global_J[l] = ((n_obs[0] - 1) * global_J[l] + (1 if num_accepted_tokens >= l else 0)) / n_obs[0]
            step_d[request_id] = int(num_draft_tokens or 0)
        except Exception:
            pass
        return _orig_stats(self, spec_decoding_stats, num_draft_tokens,
                           num_accepted_tokens, num_invalid_spec_tokens, request_id)

    def _hooked_out(self, scheduler_output, model_output):
        res = _orig_out(self, scheduler_output, model_output)
        try:
            # M1: track cumulative scheduled tokens per request for decode-phase detection.
            nsched = getattr(scheduler_output, "num_scheduled_tokens", None) or {}
            for r_id, n in nsched.items():
                sched_tok[r_id] += int(n or 0)

            N = len(self.running) or (len(step_d) if step_d else 0)
            tns = int(getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0)
            B_live = float(tns if tns > 0 else N + sum(step_d.values()))   # logged for reference only
            drafted = [r for r, d in step_d.items() if d > 0]
            if drafted:
                step_acc = (sum(state[r][-1] for r in drafted) + len(drafted)) / len(drafted)
                a_bar[0] = step_acc if a_bar[0] == 0.0 else A_TAU * a_bar[0] + (1 - A_TAU) * step_acc

            # M1: decode-phase requests only (prompt fully scheduled). Speculation acts on these.
            reqs = list(self.running)
            dec_reqs = []
            for req in reqs:
                r = req.request_id
                prompt_len = getattr(req, "num_prompt_tokens", None)
                if prompt_len is None:
                    dec_reqs.append(req)   # conservative fallback
                    continue
                if sched_tok.get(r, 0) >= prompt_len:
                    dec_reqs.append(req)
            N_dec = len(dec_reqs)
            B_eff = float(N_dec + sum(step_d.values()))
            if B_MAX:
                B_eff = min(B_eff, B_MAX)

            # mean_l[k] over DECODE requests (frozen history fallback).
            if not dec_reqs:
                OFF[0] = False
                step_d.clear()
                return res
            cum_sums = []
            for req in dec_reqs:
                r = req.request_id
                hist = state[r] if (r in state and state[r]) else khist.get(r)
                J = _Jvec(list(hist)) if hist else None
                cs = [0.0] * (KMAX + 1)
                acc = 0.0
                for l in range(1, KMAX + 1):
                    acc += (J[l] if J else global_J[l])
                    cs[l] = acc
                cum_sums.append(cs)
            mean_l = [sum(c[k] for c in cum_sums) / len(cum_sums) for k in range(KMAX + 1)]

            # BATCH-LEVEL decision (the new mechanism).
            if FORCEOFF:
                K_batch = 0
            else:
                K_batch = _batch_k(N_dec, B_eff, mean_l)
            off = (K_batch == 0)
            OFF[0] = off   # -> worker skips the draft pass this step

            # M2: allocation matches the priced model — uniform K_batch on DECODE requests,
            # k*=0 on prefill requests (drafts for them would be wasted + unpriced).
            dec_ids = {req.request_id for req in dec_reqs}
            decs = []
            for req in reqs:
                r = req.request_id
                d_r = step_d.get(r, 0)
                ks = K_batch if (not off and r in dec_ids) else 0
                last_kstar[r] = ks
                decs.append({"req": r, "Kobs": min(step_d.get(r, 0), KMAX), "d": d_r, "kstar": ks})
            self._lb_kstar = dict(last_kstar)
            # cleanup finished requests
            alive = {req.request_id for req in reqs}
            for r in list(sched_tok):
                if r not in alive:
                    del sched_tok[r]
            _fh.write(json.dumps({"t": round(time.monotonic(), 4), "N": N, "N_dec": N_dec,
                                  "B_live": round(B_live, 1), "B_eff": round(B_eff, 1),
                                  "a_bar": round(a_bar[0] or 0.0, 3),
                                  "K_batch": K_batch, "OFF": off, "decisions": decs}) + "\n")
            step_d.clear()
        except Exception:
            pass
        return res

    def _hooked_drafts(self, draft_token_ids):
        res = _orig_drafts(self, draft_token_ids)
        try:
            ks = getattr(self, "_lb_kstar", None)
            for req_id in getattr(draft_token_ids, "req_ids", ()):
                kstar = ks.get(req_id, KMAX) if ks else KMAX
                request = self.requests.get(req_id)
                if request is None:
                    continue
                cur = len(request.spec_token_ids)
                if cur > kstar:
                    request.spec_token_ids = request.spec_token_ids[:kstar]
        except Exception as e:
            print(f"[ldm_batch][drafts] EXC {type(e).__name__}: {e}", flush=True)
        return res

    def _hooked_propose(self, *args, **kwargs):
        # NEW worker-side action: skip the ngram draft pass when OFF (removes fixed spec-path overhead).
        if OFF[0]:
            # M3 (post-smoke s3 perf fix): also clear the stale req_ids so take_draft_token_ids()
            # returns None IMMEDIATELY in post_step. Without this, _draft_token_req_ids still holds
            # the last ON step's ids, so take_draft_token_ids -> _get_draft_token_ids_cpu does a full
            # draft_token_ids_event.synchronize() every OFF step (a device sync on stale data) —
            # that per-step sync is why s3 measured 2394 gen/s, BELOW the force-0 floor (2437.6)
            # which skips propose but leaves req_ids set and pays the same sync.
            try:
                self._draft_token_req_ids = None
            except Exception:
                pass
            return None
        return _orig_propose(self, *args, **kwargs)

    Scheduler.make_spec_decoding_stats = _hooked_stats
    Scheduler.update_from_output = _hooked_out
    Scheduler.update_draft_token_ids = _hooked_drafts
    GPUModelRunner.propose_draft_token_ids = _hooked_propose
    print(f"[ldm_batch] Phase 5 batch-level SD on/off + draft-pass-skip active "
          f"(KMAX={KMAX} W={W} forceoff={FORCEOFF} profile={'yes' if S_P else 'NO'} Bmax={B_MAX}) -> {OUT}", flush=True)


_install()
