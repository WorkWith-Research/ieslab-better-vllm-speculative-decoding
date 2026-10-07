"""In-loop Lightweight Decision Model (LDM) for per-request speculation length K.

Activated ONLY when VLLM_LDM_OUT is set (JSONL path). Injected via PYTHONPATH +
sitecustomize -> no vLLM source modified. Two monkey-patches on Scheduler:

  1. make_spec_decoding_stats (called once per request, per decode step, with that
     request's num_draft_tokens K and num_accepted acc): update the request's
     sliding-window acceptance history and compute its LDM decision k*.
  2. update_from_output (called once per step, after all per-request updates): log
     one line/step {t, n_active, decisions:[{req,K,k*,acc_last}, ...]}.

The decision is CAUSAL: at step t it uses only accept outcomes observed at steps < t
(window W), maximizing estimated captured tokens / measured verify-cost c(k). This is
the in-loop counterpart to the offline replay in experiments/ldm_eval.py, now driven
by live serving dynamics (real arrivals, batching, preemption) rather than a replay.

Note: this prototype LOGS the per-request K decisions and their realized acceptance;
it does not yet change the worker's draft depth (variable-K execution needs the
worker buffer/CUDA-graph patch surface in docs/vllm-code-audit.md). The logged
decisions are exactly what that execution layer would schedule, so their realized
value is measurable against fixed-K baselines (experiments/analyze_ldm.py).
"""
import json
import os
import time
from collections import defaultdict, deque

OUT = os.environ.get("VLLM_LDM_OUT")


def _install():
    if not OUT:
        return
    try:
        from vllm.v1.core.sched.scheduler import Scheduler
    except Exception as e:  # pragma: no cover
        print(f"[ldm] could not import Scheduler: {e}", flush=True)
        return

    KMAX = int(os.environ.get("VLLM_LDM_KMAX", "8"))
    W = int(os.environ.get("VLLM_LDM_WINDOW", "8"))
    # measured sub-linear verify-cost c(k), c(1)=1 (from Phase-1 throughput/accept-len)
    C = [0.0, 1.000, 1.544, 1.876, 2.207, 2.507, 2.807, 3.107, 3.407]

    def c(k):
        if k < len(C):
            return C[k]
        return C[-1] + (k - (len(C) - 1)) * 0.3

    state = defaultdict(lambda: deque(maxlen=W))
    _fh = open(OUT, "a", buffering=1)
    _orig_stats = Scheduler.make_spec_decoding_stats
    _orig_out = Scheduler.update_from_output

    def _decide(hist):
        if not hist:
            return 1
        J = [1.0] * (KMAX + 1)
        for l in range(1, KMAX + 1):
            J[l] = sum(1 for x in hist if x >= l) / len(hist)
        best, bv = 1, -1.0
        for k in range(1, KMAX + 1):
            v = sum(J[:k + 1]) / c(k)
            if v > bv:
                bv, best = v, k
        return best

    def _hooked_stats(self, spec_decoding_stats, num_draft_tokens,
                      num_accepted_tokens, num_invalid_spec_tokens, request_id):
        try:
            hist = state[request_id]
            hist.append(num_accepted_tokens)          # causal: update AFTER deciding below
            kstar = _decide(list(hist))               # uses outcomes incl. this step's
            self._ldm_state = getattr(self, "_ldm_state", {})
            self._ldm_state[request_id] = (num_draft_tokens, kstar, num_accepted_tokens)
        except Exception:
            pass
        return _orig_stats(self, spec_decoding_stats, num_draft_tokens,
                           num_accepted_tokens, num_invalid_spec_tokens, request_id)

    def _hooked_out(self, scheduler_output, model_output):
        res = _orig_out(self, scheduler_output, model_output)
        try:
            st = getattr(self, "_ldm_state", None)
            if st:
                decs = [{"req": r, "K": v[0], "kstar": v[1], "acc_last": v[2]}
                        for r, v in st.items()]
                _fh.write(json.dumps(
                    {"t": round(time.monotonic(), 4), "n_active": len(decs),
                     "decisions": decs}) + "\n")
                self._ldm_state = {}
        except Exception:
            pass
        return res

    Scheduler.make_spec_decoding_stats = _hooked_stats
    Scheduler.update_from_output = _hooked_out
    print(f"[ldm] in-loop LDM active (KMAX={KMAX} W={W}) -> {OUT}", flush=True)


_install()
