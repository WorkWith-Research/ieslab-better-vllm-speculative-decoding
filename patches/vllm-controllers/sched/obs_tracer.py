# H-OBS-1 research patch: direct per-step GPU execution timing (NOT upstream code).
#
# Prereg: docs/hoobs1-prereg.md (better-vllm-speculative-decoding repo).
#
# Purpose: measure, per executed model step, the ACTUAL GPU execution window with
# CUDA events recorded asynchronously and queried lazily (non-blocking query(); NO
# torch.cuda.synchronize / cudaDeviceSynchronize in the hot path), so it can be
# correlated 1:1 with the CPU scheduler cadence that the Track B controller uses.
#
# Enabled only when VLLM_OBS_TRACE is set to an output directory; otherwise
# get_tracer() returns None and every hook is a one-attribute-check no-op.
#
# Output: ONE merged JSONL line per step (steps.jsonl), written lazily once the
# step's GPU completion event has been observed done.
#
# ============================ DATA DICTIONARY ===============================
# field            | units   | source / timing boundary                       | meaning & controller availability
# -----------------+---------+------------------------------------------------+-----------------------------------
# step_id          | index   | tracer, assigned at schedule() exit            | unique per scheduled step; correlates CPU record to GPU events by id
# t_cpu_sched      | s mono  | schedule() ENTRY                               | the reference the Track B controller uses for cadence. Consecutive deltas = CPU cadence signal (CAUSALLY available: known at this step's decision time)
# sched_ms         | ms      | schedule() exit - entry                        | CPU work inside schedule() itself (scheduling + bookkeeping), NOT idle gap and NOT GPU time
# running          | reqs    | scheduler, actual requests scheduled THIS step | actually-scheduled requests (NOT configured concurrency C). C is a client-side cap; running <= min(C, max_num_seqs) and decays at the run tail
# waiting          | reqs    | scheduler                                        | queued-not-running requests at schedule() time
# scheduled_tokens | tokens  | scheduler, total this step                       | TOTAL positions verified on GPU = non-draft work + draft (INCLUDES drafts; verified empirically: pure-decode step ~= running*(1+K))
# prefill_tokens   | tokens  | per-request phase bookkeeping                    | GENUINE prompt-work scheduled this step = sum over reqs of min(scheduled_r, num_prompt_tokens - num_computed_tokens). A decode request that schedules 1+K tokens contributes 0 here (its >1 count is the bonus position + drafts, NOT prefill)
# draft_tokens     | tokens  | scheduler, scheduled_spec_decode_tokens          | draft/spec positions verified this step (the SD verification work); 0 when SD off or no drafts
# sd_enabled       | 0/1     | frozen Track B controller decision for THIS step| requested SD mode for the step. NOTE: distinguishes REQUESTED mode from ACTUAL proposal execution - a step can be sd_enabled=1 yet schedule 0 drafts (draft_tokens=0), and vice-versa across the pipeline
# in_flight        | steps   | tracer, at ENQUEUE time                          | earlier steps whose GPU completion was not yet observed when this step enqueued = outstanding model executions ~= how deep the 2-deep batch queue is used
# gpu_step_ms      | ms      | CUDA events start->end (async record, lazy query) | full executed-step GPU window: target forward + sampling + drafter. NOT "pure drafter time". Causally available only AFTER this step's GPU work completes (t_flush), i.e. it is a LAGGED signal for the NEXT decision
# gpu_fwd_ms       | ms      | CUDA events start->fwd_end                       | target-forward window only (the verification cost the on/off trade-off acts on). Scales with scheduled tokens; ~0-drafter on AR steps
# t_flush          | s mono  | when this record was written                     | when the GPU measurement BECAME AVAILABLE. A causal controller at step t may only use records with t_flush <= decision_time(t)
# ============================================================================
#
# Naming note: an earlier draft called (scheduled_tokens - draft_tokens)
# "prefill_tokens". That is WRONG: in a pure decode step it equals `running`
# (the 1 bonus position per request), and total INCLUDES drafts. The genuine
# prefill/decode split above uses per-request phase bookkeeping instead. Raw
# historical fields are preserved; this schema change is documented here, not
# silently reinterpreted (old smoke traces used the old meaning).
#
# Collector policy: records whose GPU events are not yet complete are DEFERRED
# to a later flush pass (no sleep in the serving hot path). Counts of
# incomplete / delayed / dropped / failed records are written to
# collector_stats.json at final_flush so samples are never silently selected.
#
# Alignment design: cpu_record() assigns a step_id and the scheduler propagates it
# on SchedulerOutput (obs_step_id); the model runner passes that id back to
# start_step/mark_fwd_end/end_step, so records and CUDA events are matched by id,
# not by FIFO position. Dummy/profiling runs carry obs_step_id=-1 and are
# skipped entirely (no record, no events). TP=1 only (UniProcExecutor:
# scheduler and runner share a process; no IPC).

import atexit
import json
import os
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
        # Set by start_step() once the step is executed; None until then.
        self.events: _StepEvents | None = None


class ObsTracer:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        self.path = os.path.join(out_dir, "steps.jsonl")
        self._pending: deque[_Pending] = deque()  # in execution order
        self._by_id: dict[int, _Pending] = {}
        self._next_id = 0
        self._buf: list[str] = []
        # Collector stats (directive §3C): count incomplete/delayed/dropped/
        # failed records instead of silently selecting successful samples.
        self._stats = {
            "records_created": 0,
            "records_written_complete": 0,   # gpu_step_ms present
            "records_written_null_gpu": 0,   # flushed at final_flush w/o GPU time
            "records_failed_elapsed": 0,     # events done but elapsed_time refused (all retries)
            "records_dropped": 0,            # no matching record / no events at flush
        }
        self._write_stats()   # exist from t0; updated on each flush + final_flush

    def _write_stats(self) -> None:
        try:
            with open(os.path.join(self.out_dir, "collector_stats.json"), "w") as f:
                json.dump(self._stats, f, indent=2)
        except Exception:
            pass

    # ---- CPU-side hook (scheduler.schedule()) ----

    def t_now(self) -> float:
        return time.monotonic()

    def cpu_record(self, running: int, waiting: int, scheduled_tokens: int,
                   prefill_tokens: int, draft_tokens: int, sd_enabled: bool,
                   entry_t: float = 0.0, exit_t: float = 0.0) -> int:
        """One record per step that schedules >0 tokens. Returns the step_id to
        propagate on SchedulerOutput so the runner's GPU hooks can match it.
        `entry_t`/`exit_t` bracket schedule(); their delta is sched_ms (CPU work
        inside scheduling, distinct from the cadence gap and from GPU time).
        in_flight = earlier steps whose GPU completion had not yet been
        observed at enqueue time."""
        sid = self._next_id
        rec = {
            "step_id": sid,
            "t_cpu_sched": round(entry_t if entry_t > 0 else time.monotonic(), 6),
            # Wall-clock (time.time()) anchor for cross-process alignment with the
            # metrics scraper / client clocks (H-OBS-3). Same host clock; additive.
            "t_wall": round(time.time(), 6),
            # CPU work inside schedule() itself (ms). 0.0 when not bracketed.
            "sched_ms": round((exit_t - entry_t) * 1000.0, 3) if (entry_t > 0 and exit_t > entry_t) else None,
            "running": running,
            "waiting": waiting,
            "scheduled_tokens": scheduled_tokens,
            "prefill_tokens": prefill_tokens,
            "draft_tokens": draft_tokens,
            "sd_enabled": int(sd_enabled),
            "in_flight": len(self._pending),
        }
        self._next_id += 1
        p = _Pending(rec)
        self._pending.append(p)
        self._by_id[sid] = p
        self._stats["records_created"] += 1
        return sid

    # ---- GPU-side hooks (V2 model runner) ----

    def start_step(self, step_id: int) -> None:
        """Start of an executed step's forward window. Attaches events to the
        matching record by id and records the GPU start event. step_id < 0 means
        no CPU record exists (dummy/profiling run, or tracer off) -> no-op."""
        if step_id < 0:
            return
        p = self._by_id.get(step_id)
        if p is None:
            return  # defensive: no matching scheduler record
        ev = _StepEvents()
        p.events = ev
        ev.start.record()

    def mark_fwd_end(self, step_id: int) -> None:
        """Right after the target forward was launched (before sampling/drafter)."""
        if step_id < 0:
            return
        p = self._by_id.get(step_id)
        if p is not None and p.events is not None:
            p.events.fwd_end.record()

    def end_step(self, step_id: int) -> None:
        """End of the step's GPU work (after sampling + drafter)."""
        if step_id < 0:
            return
        p = self._by_id.get(step_id)
        if p is not None and p.events is not None:
            p.events.end.record()
            self._flush_ready()

    def _flush_ready(self) -> None:
        """Write pending records whose GPU window has fully completed (lazy,
        non-blocking). torch 2.11's elapsed_time requires BOTH the start and end
        events to be *completed* (not merely recorded), so we gate on both
        query() results. Events on one stream complete in order, so stop at the
        first not-yet-complete event; its record STAYS PENDING until a later
        flush observes it done (final_flush is the guaranteed backstop).

        Collector policy (directive §3C): NO sleep in this hot path. If
        elapsed_time still refuses on a single attempt even though both events
        report completed, we defer the record to the next pass rather than
        blocking; repeated refusal is counted and surfaced at final_flush."""
        while self._pending:
            p = self._pending[0]
            ev = p.events
            if ev is None or not ev.end.query() or not ev.start.query():
                break  # this (and all later) steps still running on the GPU
            step_ms, fwd_ms = self._elapsed_once(ev)
            if step_ms is None:
                break  # defer to a later pass; counted if it never resolves
            p.rec["gpu_step_ms"] = round(step_ms, 3)
            p.rec["gpu_fwd_ms"] = round(fwd_ms, 3) if fwd_ms is not None else None
            p.rec["t_flush"] = round(time.monotonic(), 6)
            self._buf.append(json.dumps(p.rec))
            self._by_id.pop(p.rec["step_id"], None)
            self._pending.popleft()
            self._stats["records_written_complete"] += 1
        if len(self._buf) >= 100:
            self.flush()

    @staticmethod
    def _elapsed_once(ev: _StepEvents):
        """(step_ms, fwd_ms) with a SINGLE attempt (no sleep). step_ms=None means
        defer to a later pass — the driver may finalize timestamps lazily even
        after both events report completed."""
        try:
            step_ms = ev.start.elapsed_time(ev.end)
        except Exception:
            return None, None
        try:
            fwd_ms = ev.start.elapsed_time(ev.fwd_end)
        except Exception:
            fwd_ms = None
        return step_ms, fwd_ms

    def flush(self) -> None:
        if not self._buf:
            return
        with open(self.path, "a") as f:
            f.write("\n".join(self._buf) + "\n")
        self._buf.clear()
        # Write collector stats on every batched flush so the file exists and is
        # current even if the process is later killed in abort mode (vLLM V1
        # force-kills EngineCore on SIGTERM before atexit runs). This is a few KB
        # of JSON ~once per 100 steps — negligible. Pending records are visible as
        # created - written_complete, so nothing is silently selected.
        self._write_stats()

    def final_flush(self) -> None:
        """Called once at process exit (atexit). The last few steps' completion
        events may not have been observed yet; synchronize ONCE here (NOT in the
        hot path) and write whatever is pending so no step is lost. Records that
        still cannot be timed are written with null GPU fields and counted —
        never silently dropped."""
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
                    # Give the already-synchronized device a few final attempts
                    # (still outside the hot path); count if it never resolves.
                    for _ in range(5):
                        step_ms, fwd_ms = self._elapsed_once(ev)
                        if step_ms is not None:
                            break
                p.rec["gpu_step_ms"] = round(step_ms, 3) if step_ms is not None else None
                p.rec["gpu_fwd_ms"] = round(fwd_ms, 3) if fwd_ms is not None else None
                if step_ms is None:
                    self._stats["records_failed_elapsed"] += 1
                else:
                    self._stats["records_written_complete"] += 1
            p.rec["t_flush"] = round(time.monotonic(), 6)
            self._buf.append(json.dumps(p.rec))
            self._by_id.pop(p.rec["step_id"], None)
            self._pending.popleft()
        self.flush()   # writes steps + collector_stats (via _write_stats)


_tracer: ObsTracer | None = None


def get_tracer() -> ObsTracer | None:
    """Module-level lazy singleton. None when VLLM_OBS_TRACE is unset."""
    global _tracer
    if _tracer is None:
        out_dir = os.environ.get("VLLM_OBS_TRACE", "")
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            _tracer = ObsTracer(out_dir)
            atexit.register(_tracer.final_flush)
    return _tracer
