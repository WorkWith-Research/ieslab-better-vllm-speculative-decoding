# P5.2 H-5.2 profiling overlay (P1): per-step GPU-window + CPU-cadence tracer for vLLM 0.19.1.
#
# Ported from H-OBS-1's obs_tracer (dspark-native-validation, vLLM 0.25.0) to the main repo's
# 0.19.1 stack, applied as a sitecustomize monkey-patch (never touches the installed package).
#
# KEY VERSION DIFFERENCE from H-OBS-1: on 0.19.1 the engine core spins Scheduler.schedule() in a
# tight non-blocking loop (~hundreds of calls per executed GPU step), so schedule() is NOT a reliable
# once-per-step anchor (H-OBS-1's 0.25.0 engine blocked between steps, making its cadence clean).
# We therefore anchor EVERYTHING on GPUModelRunner.execute_model, which runs exactly once per
# executed model step in the sync eager loop. running/scheduled/draft are read from the
# scheduler_output passed to execute_model (same data schedule() produced), so no schedule() patch is
# needed at all.
#
# Per executed step we record:
#   cadence_ms  = wall time between consecutive execute_model entries (per-step wall; the signal a
#                 controller sees). Includes CPU scheduling + model-runner prep + enqueue + any idle.
#   gpu_step_ms = CUDA-event window over the full executed step (forward + sampling) on the main stream.
#   gpu_fwd_ms  = CUDA-event window over the target forward only.
#   residual    = cadence_ms - gpu_step_ms  = all CPU-side work for the step (scheduler spin that
#                 produced this step's plan + model-runner input-batch/attention-metadata bookkeeping
#                 sized for spec + enqueue). This is where a spec-path-free off-state would act.
#
# Enabled only when VLLM_P52_OBS is set to an output directory; otherwise no-op.
# Optional VLLM_P52_FORCE0=1: truncate every request's drafts to 0 (replicates ldmload FORCE0 — the
# ngram draft pass still runs, its output discarded) for the P1 force-0 floor arm.

import builtins  # noqa: E402
import json
import os
import sys
import time
from collections import deque

import torch


class _StepEvents:
    __slots__ = ("start", "fwd_end", "end")

    def __init__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.fwd_end = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)


class _Pending:
    __slots__ = ("rec", "events")

    def __init__(self, rec):
        self.rec = rec
        self.events = None


class Tracer:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.path = os.path.join(out_dir, "steps.jsonl")
        self._pending = deque()
        self._by_id = {}
        self._next_id = 0
        self._buf = []
        self._last_em_entry = None  # monotonic time of previous execute_model entry (per-step cadence)
        self._stats = {
            "records_created": 0,
            "records_written_complete": 0,
            "records_dropped": 0,
        }
        os.makedirs(out_dir, exist_ok=True)
        self._write_stats()

    def _write_stats(self):
        try:
            with open(os.path.join(self.out_dir, "collector_stats.json"), "w") as f:
                json.dump(self._stats, f, indent=2)
        except Exception:
            pass

    # ---- one record per EXECUTED step (anchored on execute_model entry) ----
    def begin_step(self, running, scheduled_tokens, draft_tokens):
        """Called at execute_model entry. Returns the step_id for the GPU hooks."""
        now = time.monotonic()
        sid = self._next_id
        cadence_ms = None
        if self._last_em_entry is not None:
            delta = (now - self._last_em_entry) * 1000.0
            # A real step's wall is bounded; a huge gap means the engine idled (run start/tail).
            cadence_ms = round(delta, 3) if 0 < delta < 2000.0 else None
        self._last_em_entry = now
        rec = {
            "step_id": sid,
            "t_cpu_sched": round(now, 6),
            "cadence_ms": cadence_ms,
            "running": running,
            "scheduled_tokens": scheduled_tokens,
            "draft_tokens": draft_tokens,
        }
        self._next_id += 1
        p = _Pending(rec)
        self._pending.append(p)
        self._by_id[sid] = p
        self._current = sid
        ev = _StepEvents()
        p.events = ev
        ev.start.record()
        self._stats["records_created"] += 1
        return sid

    def current_sid(self):
        return getattr(self, "_current", -1)

    def mark_fwd_end(self, step_id):
        if step_id < 0:
            return
        p = self._by_id.get(step_id)
        if p is not None and p.events is not None:
            p.events.fwd_end.record()

    def end_step(self, step_id):
        if step_id < 0:
            return
        p = self._by_id.get(step_id)
        if p is not None and p.events is not None:
            p.events.end.record()
            self._flush_ready()

    def _elapsed_once(self, ev):
        try:
            step_ms = ev.start.elapsed_time(ev.end)
        except Exception:
            return None, None
        try:
            fwd_ms = ev.start.elapsed_time(ev.fwd_end)
        except Exception:
            fwd_ms = None
        return step_ms, fwd_ms

    def _flush_ready(self):
        while self._pending:
            p = self._pending[0]
            ev = p.events
            if ev is None or not ev.end.query() or not ev.start.query():
                break
            step_ms, fwd_ms = self._elapsed_once(ev)
            if step_ms is None:
                break
            p.rec["gpu_step_ms"] = round(step_ms, 3)
            p.rec["gpu_fwd_ms"] = round(fwd_ms, 3) if fwd_ms is not None else None
            # residual = CPU-side work for the step (wall minus GPU window). Negative => GPU ran
            # longer than wall (overlap/idle at boundaries); clamp to 0 for attribution.
            cad = p.rec.get("cadence_ms")
            p.rec["residual_ms"] = round(max(0.0, cad - step_ms), 3) if cad is not None else None
            p.rec["t_flush"] = round(time.monotonic(), 6)
            self._buf.append(json.dumps(p.rec))
            self._by_id.pop(p.rec["step_id"], None)
            self._pending.popleft()
            self._stats["records_written_complete"] += 1
        if len(self._buf) >= 100:
            self.flush()

    def flush(self):
        if not self._buf:
            return
        with open(self.path, "a") as f:
            f.write("\n".join(self._buf) + "\n")
        self._buf.clear()
        self._write_stats()

    def final_flush(self):
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        while self._pending:
            p = self._pending[0]
            ev = p.events
            if ev is None:
                p.rec["gpu_step_ms"] = None
                p.rec["gpu_fwd_ms"] = None
                self._stats["records_dropped"] += 1
            else:
                step_ms, fwd_ms = self._elapsed_once(ev)
                if step_ms is None:
                    for _ in range(5):
                        step_ms, fwd_ms = self._elapsed_once(ev)
                        if step_ms is not None:
                            break
                p.rec["gpu_step_ms"] = round(step_ms, 3) if step_ms is not None else None
                p.rec["gpu_fwd_ms"] = round(fwd_ms, 3) if fwd_ms is not None else None
                cad = p.rec.get("cadence_ms")
                p.rec["residual_ms"] = (
                    round(max(0.0, cad - step_ms), 3)
                    if (cad is not None and step_ms is not None) else None
                )
            p.rec["t_flush"] = round(time.monotonic(), 6)
            self._buf.append(json.dumps(p.rec))
            self._by_id.pop(p.rec["step_id"], None)
            self._pending.popleft()
        self.flush()


_tracer = None


def get_tracer():
    global _tracer
    if _tracer is None:
        out_dir = os.environ.get("VLLM_P52_OBS", "")
        if out_dir:
            import atexit
            _tracer = Tracer(out_dir)
            atexit.register(_tracer.final_flush)
    return _tracer


_installed = False   # patches actually applied (set only on success)
_installing = False  # re-entrancy guard for the import hook


def _install_patches():
    global _installed, _installing
    if _installed:
        return
    _installing = True
    try:
        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner

        # Optional FORCE0 mode (P1 force-0 arm): replicate ldmload_controller's FORCE0 — truncate every
        # request's spec_token_ids to 0 in update_draft_token_ids. The ngram draft pass still RUNS (its
        # output is discarded), matching how P4.6 measured the SD-off floor, so this is a consistent
        # baseline against the P4.6 force-0 numbers.
        if os.environ.get("VLLM_P52_FORCE0") == "1":
            _orig_drafts = Scheduler.update_draft_token_ids

            def _hooked_drafts(self, draft_token_ids):
                res = _orig_drafts(self, draft_token_ids)
                try:
                    for req_id in getattr(draft_token_ids, "req_ids", ()):
                        request = self.requests.get(req_id)
                        if request is not None and len(request.spec_token_ids) > 0:
                            request.spec_token_ids = []
                except Exception:
                    pass
                return res

            Scheduler.update_draft_token_ids = _hooked_drafts

        _orig_em = GPUModelRunner.execute_model

        def _patched_execute_model(self, scheduler_output, intermediate_tensors=None):
            tr = get_tracer()
            if tr is not None:
                nst = getattr(scheduler_output, "num_scheduled_tokens", None) or {}
                running = len(nst)
                scheduled_tokens = int(getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0)
                draft = sum(
                    len(v) for v in (getattr(scheduler_output, "scheduled_spec_decode_tokens", None) or {}).values()
                )
                sid = tr.begin_step(running, scheduled_tokens, draft)
                self._p52_obs_sid = sid
            return _orig_em(self, scheduler_output, intermediate_tensors)

        GPUModelRunner.execute_model = _patched_execute_model

        # GPU window = execute_model entry .. sample_tokens exit. sample_tokens runs the target
        # sampling AND the drafter (propose_draft_token_ids) before returning, so this brackets
        # forward + sampling + drafter on the main stream. Marking fwd_end at its entry splits out
        # the target-forward-only window. This is the clean G1 split: gpu_step_ms = ALL GPU work of
        # the step (incl. the drafter, which force-0 still runs); residual = cadence - gpu_step =
        # pure CPU-side work (scheduler spin + model-runner spec bookkeeping/metadata + enqueue).
        _orig_st = GPUModelRunner.sample_tokens

        def _patched_sample_tokens(self, grammar_output=None):
            tr = get_tracer()
            sid = getattr(self, "_p52_obs_sid", -1)
            if tr is not None and sid >= 0:
                tr.mark_fwd_end(sid)
            out = _orig_st(self, grammar_output)
            if tr is not None and sid >= 0:
                tr.end_step(sid)
            return out

        GPUModelRunner.sample_tokens = _patched_sample_tokens

        _installed = True
    finally:
        _installing = False


_orig_import = builtins.__import__


def _patched_import(name, *args, **kwargs):
    mod = _orig_import(name, *args, **kwargs)
    if not _installed and not _installing and os.environ.get("VLLM_P52_OBS"):
        if name.endswith("gpu_model_runner") or name == "vllm.v1.worker.gpu_model_runner":
            # _install_patches() imports vLLM modules, which re-enter this wrapper; the separate
            # _installing flag (set inside _install_patches) breaks that recursion. We must NOT
            # pre-set _installed here — _install_patches returns early on it and would no-op.
            try:
                _install_patches()
            except Exception as e:  # surfaced, never fatal to serving
                print(f"[p52_obs] deferred install failed: {e}", file=sys.stderr)
    return mod


builtins.__import__ = _patched_import
