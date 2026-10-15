"""Non-invasive per-request spec-decode observation hook for vLLM 0.19.1.

Activated ONLY when VLLM_SPEC_HOOK_OUT is set (path to a JSONL file). Injected
via PYTHONPATH + sitecustomize so no vLLM source file is modified.

Why: vLLM computes per-request draft/accept counts in
Scheduler.make_spec_decoding_stats (called once per request per step, with the
request_id) but immediately aggregates them into batch-level counters. The
per-request acceptance signal is exactly what a dynamic-K decision model needs,
so we tee it out here for Phase-2 oracle-gap analysis.

Emits one JSON line per (request, decode-step):
  {"t": <monotonic>, "req": <request_id>, "K": <scheduled draft tokens>,
   "acc": <accepted tokens this step>}
"""
import json
import os
import time

OUT = os.environ.get("VLLM_SPEC_HOOK_OUT")


def _install():
    if not OUT:
        return
    try:
        from vllm.v1.core.sched.scheduler import Scheduler
    except Exception as e:  # pragma: no cover - import timing
        print(f"[spec-hook] could not import Scheduler: {e}", flush=True)
        return

    _orig = Scheduler.make_spec_decoding_stats
    _fh = open(OUT, "a", buffering=1)

    def _hooked(self, spec_decoding_stats, num_draft_tokens,
                num_accepted_tokens, num_invalid_spec_tokens, request_id):
        try:
            k = num_draft_tokens
            if num_invalid_spec_tokens:
                k -= num_invalid_spec_tokens.get(request_id, 0)
            _fh.write(json.dumps(
                {"t": round(time.monotonic(), 4), "req": request_id,
                 "K": k, "acc": num_accepted_tokens}) + "\n")
        except Exception:
            pass
        return _orig(self, spec_decoding_stats, num_draft_tokens,
                     num_accepted_tokens, num_invalid_spec_tokens, request_id)

    Scheduler.make_spec_decoding_stats = _hooked
    print(f"[spec-hook] per-request spec-decode hook active -> {OUT}", flush=True)


_install()
