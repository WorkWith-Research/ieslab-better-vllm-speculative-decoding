# Track B live-state SD on/off controller — research patch (NOT upstream code).
#
# Directive: "Hermes Research Directive — Native DSpark Baseline and Next
# Research Milestone" §4. Prereg: docs/trackb-prereg.md (R1/R2 revisions).
#
# Design (measured serving state only — NO concurrency thresholds):
#   The controller compares the MEASURED cost of a speculative step against the
#   AR-equivalent cost of producing the SAME number of tokens. A decode step
#   with SD on produces ~L accepted+bonus tokens per request (L = mean
#   acceptance length, load-invariant at ≈2.77 in Phase 6), so:
#       threshold(N) = L * est_ar_ms(N)
#       est_ar_ms(n) = C0 + c1 * n          (Phase 6 AR TPOT fit, per token)
#   Exit (ON -> OFF): the rolling mean of measured SD-on decode-step costs
#     exceeds threshold(running) * (1 + band).
#   Re-entry (OFF -> ON): PROBE-BASED. After a cooldown, the controller turns
#     SD on for `probe_steps` steps and measures the ACTUAL DSpark step cost.
#     Commit to ON if that measured cost is below threshold(running) * (1 - band);
#     otherwise revert to OFF with an extended cooldown. This avoids the
#     circularity of judging "is SD cheap?" from a period when SD was off
#     (measured-while-off cost IS the AR cost, which always looks cheap).
#   Prefill-heavy steps (> prefill_skip_frac non-draft tokens) and idle gaps
#   (> max_step_s between schedule() calls) are excluded from the window.
#
# Phase 6 operating points (results/gate_e_repro/, docs/phase6-reproduction-
# prereg.md §8) classify as: C=8 ON (T_on≈9.04 ms < thr 41.7 ms), C=32 ON
# (26.18 < 79.0 — DSpark still wins +13% throughput there), C=64 OFF
# (47.21 > 110.2 ... measured step cost exceeds L-scaled AR parity) — i.e. the
# controller preserves the low/medium-load benefit and recovers AR parity at
# saturation, matching the measured THROUGHPUT crossover (between C=32 and
# C=64). See docs/trackb-prereg.md [CAL].
#
# Causal-verification trace (prereg §4.4): every step appends
#   {t, sd_enabled, state, measured_ms, threshold_ms, est_ar_ms, occupancy,
#    prefill_share, drafts_scheduled_this_step}
# to VLLM_TB_TRACE (JSONL). Offline assertion: OFF steps => 0 new draft tokens
# scheduled; ON decode steps => K drafts per running request.
#
# Env overrides (for the control arms and debugging):
#   VLLM_TB_MODE = live | on | off | native  (default: live; native = zero-
#   overhead always-on bypass, i.e. the true DSpark-K7 baseline arm)
#   VLLM_TB_TRACE = /path/to/trace.jsonl  (default: disabled)
import json
import os
import time

# Injectable clock (module-level) so unit tests can drive time deterministically.
_now = time.monotonic

# AR per-token step-cost model est_ar_ms(n) = C0 + c1*n, least-squares fit to
# the Phase 6 AR mean-TPOT operating points (results/gate_e_repro/):
# AR TPOT = 13.55 ms @n=8, 21.43 ms @n=32, 34.92 ms @n=64.
# Fit: C0=9.994, c1=0.3838 (max abs error 3.9% across the three points).
_C0_MS = 9.994
_C1_MS_PER_REQ = 0.3838
# Mean acceptance length (accepted+bonus tokens per request per SD step),
# load-invariant in Phase 6 (acc_len ≈ 2.77 at C=8/32/64). Scales the AR
# per-token cost to an AR-equivalent PER-STEP cost for N running requests:
# an AR server producing the same tokens as one SD step needs ~L steps.
_L_ACCEPT = 2.77
_BAND = 0.15            # hysteresis band (±15%)
_COOLDOWN_S = 1.0       # minimum OFF duration before a re-entry probe
_PROBE_STEPS = 3        # SD-on steps used to measure actual DSpark cost
_WINDOW_S = 2.0         # rolling window for the measured step cost
_PREFILL_SKIP_FRAC = 0.3  # steps with >30% non-draft tokens are not decode samples
# Idle-gap filter (robustness, NOT a decision threshold): schedule() is only
# called while requests exist; if the server was idle between two calls the
# wall-clock delta is an idle gap, not a step cost. Real steps on this hardware
# are <~200 ms (max_num_batched_tokens=4096), so deltas above this bound are
# dropped and do not enter the rolling window.
_MAX_STEP_S = 0.5


class TBController:
    def __init__(self, num_spec_tokens: int):
        self.num_spec_tokens = num_spec_tokens
        self.mode = os.environ.get("VLLM_TB_MODE", "live").lower()
        self.trace_path = os.environ.get("VLLM_TB_TRACE", "")
        # `native` mode = zero-overhead bypass: evaluate() returns True with no
        # bookkeeping and record_step() is a no-op. This is the true native
        # DSpark-K7 baseline arm (S3 compares CTRL-ON against it to measure the
        # patch/decision overhead). Any other mode runs the full controller.
        self.active = self.mode != "native"
        # state: "on" (committed), "off", or "probe" (temporary SD-on measurement)
        self.state = "on"
        self._off_since: float | None = None
        self._cooldown = _COOLDOWN_S
        self._samples: list[tuple[float, float]] = []  # (wall_time, step_ms), SD-on decode only
        self._probe_steps_left = 0
        self._buf: list[str] = []
        self._t_last_step: float | None = None

    @property
    def enabled(self) -> bool:
        return self.state in ("on", "probe")

    def est_ar_ms(self, n_reqs: int) -> float:
        """AR per-token step cost (ms) for n running requests (Phase 6 fit)."""
        return _C0_MS + _C1_MS_PER_REQ * max(0, n_reqs)

    def threshold_ms(self, n_reqs: int) -> float:
        """AR-equivalent PER-STEP cost (ms): L AR steps for the tokens one SD step yields."""
        return _L_ACCEPT * self.est_ar_ms(n_reqs)

    def _window_mean(self) -> float | None:
        """Mean of ALL samples in the rolling window (prereg R3: 'rolling mean,
        window W=2 s'). NOT a last-N mean: at C=8 (~50 steps/s) the window holds
        ~100 samples, so a single transient spike cannot push the mean over the
        exit band (last-4 would flap on every spike — see trackb-results §6)."""
        if not self._samples:
            return None
        return sum(ms for _, ms in self._samples) / len(self._samples)

    def evaluate(
        self,
        running: int,
        scheduled_tokens: int,
        prefill_tokens: int,
    ) -> bool:
        """Decide SD on/off for the step about to be scheduled.

        Must be called once per scheduler.schedule() step, BEFORE running
        requests are processed (so an OFF decision can clear pending drafts).
        """
        if not self.active:
            return True  # native mode: always SD on, zero bookkeeping
        now = _now()

        # 1) Measure the previous step's cost — only SD-on decode steps count.
        if self._t_last_step is not None and scheduled_tokens > 0:
            dt = now - self._t_last_step
            prefill_share = (
                prefill_tokens / scheduled_tokens if scheduled_tokens else 0.0
            )
            if (
                0 < dt < _MAX_STEP_S
                and prefill_share <= _PREFILL_SKIP_FRAC
                and self.state in ("on", "probe")
            ):
                self._samples.append((now, dt * 1000.0))
        self._t_last_step = now

        # 2) Prune the rolling window.
        cutoff = now - _WINDOW_S
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.pop(0)

        if self.mode == "on":
            self.state = "on"
        elif self.mode == "off":
            self.state = "off"
        else:  # live
            measured = self._window_mean()
            thr = self.threshold_ms(running)
            if self.state == "on":
                # Exit: enough SD-on decode samples, measured cost above band.
                if (
                    len(self._samples) >= 3
                    and measured is not None
                    and measured > thr * (1.0 + _BAND)
                ):
                    self.state = "off"
                    self._off_since = now
                    self._cooldown = _COOLDOWN_S
            elif self.state == "off":
                # Start a re-entry probe once the cooldown has elapsed.
                if (
                    self._off_since is not None
                    and now - self._off_since >= self._cooldown
                ):
                    self.state = "probe"
                    self._probe_steps_left = _PROBE_STEPS
                    self._samples.clear()  # measure the probe fresh
            else:  # probe
                if self._probe_steps_left > 0:
                    self._probe_steps_left -= 1
                if self._probe_steps_left <= 0 and len(self._samples) >= 2:
                    m = self._window_mean()
                    if m is not None and m < thr * (1.0 - _BAND):
                        # Probe shows SD is cheap here: commit ON, keep the
                        # probe samples as the committed state's baseline.
                        self.state = "on"
                        self._off_since = None
                    else:
                        # Probe shows SD still loses: back to OFF, longer cooldown.
                        self.state = "off"
                        self._off_since = now
                        self._cooldown = max(self._cooldown * 2.0, _COOLDOWN_S)

        return self.enabled

    def record_step(
        self,
        sd_enabled: bool,
        running: int,
        waiting: int,
        kv_usage: float,
        scheduled_tokens: int,
        draft_tokens_scheduled: int,
        prefill_tokens: int,
        max_num_seqs: int,
    ) -> None:
        """Append one causal-verification trace line for the step just built."""
        if not self.active or not self.trace_path:
            return
        now = _now()
        measured = self._window_mean()
        self._buf.append(
            json.dumps(
                {
                    "t": round(now, 6),
                    "sd_enabled": int(sd_enabled),
                    "state": self.state,
                    "measured_ms": (round(measured, 3) if measured is not None else None),
                    "threshold_ms": round(self.threshold_ms(running), 3),
                    "est_ar_ms": round(self.est_ar_ms(running), 3),
                    "occupancy": round(running / max(1, max_num_seqs), 4),
                    "prefill_share": (
                        round(prefill_tokens / scheduled_tokens, 4)
                        if scheduled_tokens
                        else None
                    ),
                    "drafts_scheduled_this_step": draft_tokens_scheduled,
                }
            )
        )
        if len(self._buf) >= 100:
            self.flush_trace()

    def flush_trace(self) -> None:
        if not self.trace_path or not self._buf:
            return
        with open(self.trace_path, "a") as f:
            f.write("\n".join(self._buf) + "\n")
        self._buf.clear()
