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

Note on execution: with VLLM_LDM_ENFORCE=1 the controller also TRUNCATES each request's
drafts to its current k* in update_draft_token_ids, so the worker verifies a different K per
request (vLLM's SpecDecodeMetadata already supports variable per-request draft counts; FlashAttn
uses variable query lengths). Variable-K decode is non-uniform -> requires enforce_eager (CUDA
graphs need uniform decode), so end-to-end runs use eager mode for both LDM and fixed-K baselines.
"""
import json
import os
import time
from collections import defaultdict, deque

OUT = os.environ.get("VLLM_LDM_OUT")
ENFORCE = os.environ.get("VLLM_LDM_ENFORCE", "0") == "1"


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
    # measured sub-linear verify-cost c(k), c(1)=1 (from Phase-1 throughput/accept-len).
    # c(0) = 1.0: an AR step does one forward pass over the batch (1 token/request) — same
    # baseline work as a K=1 step minus verification; pre-registered for the SD-off fallback.
    C = [1.0, 1.000, 1.544, 1.876, 2.207, 2.507, 2.807, 3.107, 3.407]

    def c(k):
        if k < len(C):
            return C[k]
        return C[-1] + (k - (len(C) - 1)) * 0.3

    state = defaultdict(lambda: deque(maxlen=W))   # request_id -> accept history
    _fh = open(OUT, "a", buffering=1)
    _orig_stats = Scheduler.make_spec_decoding_stats
    _orig_out = Scheduler.update_from_output
    _orig_drafts = Scheduler.update_draft_token_ids

    def _decide(hist):
        # Pre-registered SD-off fallback (PROGRESS.md P5): k*=0 is a candidate — an AR step
        # (c(0)=1) captures exactly 1 token. If no speculative length beats it on the value
        # ratio, the request runs at K=0.
        if not hist:
            return 1
        J = [1.0] * (KMAX + 1)
        for l in range(1, KMAX + 1):
            J[l] = sum(1 for x in hist if x >= l) / len(hist)
        best, bv = 0, 1.0 / c(0)          # k*=0 baseline: captures 1 token at AR cost
        for k in range(1, KMAX + 1):
            v = sum(J[:k + 1]) / c(k)
            if v > bv:
                bv, best = v, k
        return best

    def _hooked_stats(self, spec_decoding_stats, num_draft_tokens,
                      num_accepted_tokens, num_invalid_spec_tokens, request_id):
        try:
            hist = state[request_id]
            # OPTIMISTIC cold start: a request with no history yet is drafted at KMAX so the
            # LDM can OBSERVE deep (position 2+) acceptance. A pessimistic k*=1 start would be
            # self-fulfilling: it only ever drafts 1 token, never sees position 2+, so its
            # estimate stays J[2..]=0 and it is stuck at k*=1 forever (the collapse bug).
            if not hist:
                kstar = KMAX
            else:
                kstar = _decide(list(hist))
            hist.append(num_accepted_tokens)         # record this step's outcome for next time
            self._ldm_state = getattr(self, "_ldm_state", {})
            self._ldm_state[request_id] = (num_draft_tokens, kstar, num_accepted_tokens)
            if ENFORCE:
                # separate dict consumed by update_draft_token_ids (runs later in post_step),
                # so it must NOT be cleared by the logging hook.
                self._ldm_kstar = getattr(self, "_ldm_kstar", {})
                self._ldm_kstar[request_id] = kstar
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

    def _hooked_drafts(self, draft_token_ids):
        # After the original stores full drafts per request, truncate each to its k*.
        res = _orig_drafts(self, draft_token_ids)
        if ENFORCE:
            try:
                ks = getattr(self, "_ldm_kstar", None)
                for req_id in getattr(draft_token_ids, "req_ids", ()):
                    # Unknown request (first decode step, no acceptance observed yet) ->
                    # optimistic KMAX so it can observe deep acceptance (matches the cold-start).
                    kstar = ks.get(req_id, KMAX) if ks else KMAX
                    request = self.requests.get(req_id)
                    if request is not None and len(request.spec_token_ids) > kstar:
                        request.spec_token_ids = request.spec_token_ids[:kstar]
            except Exception:
                pass
        return res

    Scheduler.make_spec_decoding_stats = _hooked_stats
    Scheduler.update_from_output = _hooked_out
    if ENFORCE:
        Scheduler.update_draft_token_ids = _hooked_drafts
    print(f"[ldm] in-loop LDM active (KMAX={KMAX} W={W} enforce={ENFORCE}) -> {OUT}", flush=True)


_install()
