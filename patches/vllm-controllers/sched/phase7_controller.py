# Phase 7 (capstone) — request-aware SD on/off controller from live serving state.
# Research patch (NOT upstream code). Prereg: docs/phase7-capstone-prereg.md.
#
# The validated 2-axis regime rule:
#
#     sd_enabled  <-  load_LOW  OR  class_HIGHREUSE
#
#   - load_LOW      : measured KV-cache usage fraction below a threshold, with hysteresis.
#                     [CAL-1] primary signal = kv_cache_manager.usage. Chosen over
#                     running-occupancy / waiting because it separates the c16/c48/c96
#                     operating points cleanly (0.30 / 0.93 / 0.98) and needs no
#                     max_num_seqs constant. Deliberately NOT the schedule()-cadence
#                     signal, which Track B Defect 2 showed decouples from true GPU step
#                     time under async scheduling.
#   - class_HIGHREUSE: rolling MEAN prefix-cache ADMISSION hit-rate above a threshold.
#                     [CAL-2] This is the SD-INDEPENDENT class proxy (see below). It is
#                     measured from KV blocks at request admission — orthogonal to
#                     speculative decoding, so it is available WHILE SD IS OFF. That is
#                     what makes the 2-axis rule self-consistent: both axes are directly
#                     observable in the state we must decide from.
#
# WHY hit-rate and not draft-acceptance (the [CAL-2b] finding): the first class-signal
# design used SD's own draft-acceptance ratio. That has a chicken-and-egg: acceptance can
# only be measured while SD is ON, but at high load the controller is OFF until it knows
# traffic is reusable — so the signal never fires exactly where it must (smoke: global max
# 0.295 < theta_acc 0.45, 0 steps crossed). Prefix-cache hit-rate has no such circularity:
# a request that reuses a cached prefix shows hit-rate ~1.0 at admission regardless of SD.
# The premise check (c96 mixed, SD off) confirmed the reusable prefixes SURVIVE eviction
# pressure from diverse traffic and the signal is cleanly bimodal (~0 vs ~1), so this axis
# works at high load. Because both axes are now directly observable with SD off, the Track B
# cooldown+probe re-entry mechanism is no longer needed — a simple cooldown suffices to
# avoid flapping on the OFF->ON edge.
#
# Modes (VLLM_TB_MODE):
#   live      : full 2-axis rule (the hypothesis arm, CTRL-LIVE).
#   loadonly  : ON iff load_LOW            (CTRL-LOADONLY ablation — class axis removed).
#   classonly : ON iff class_HIGHREUSE     (CTRL-CLASSONLY ablation — load axis removed).
#   on / off  : constant arms (CTRL-ON overhead control / SD-off floor).
#   native    : zero-overhead always-on bypass (true DSpark-K7 baseline) -> handled by the
#               caller not instantiating this controller at all.
#
# Causal-verification trace (prereg S5): every step appends
#   {t, sd_enabled, state, kv_usage, occupancy, hit_rate, load_low, class_highreuse,
#    drafts_scheduled_this_step}
# to VLLM_TB_TRACE (JSONL). Offline assertion: OFF steps => 0 new draft tokens scheduled;
# ON decode steps => K drafts per running request.
import json
import os
import time

_now = time.monotonic

# ---- [CAL] constants (docs/phase7-capstone-prereg.md §7, frozen from existing data) ----
# Load axis: KV-usage hysteresis band. c16 ~0.30 (LOW), c48/c96 ~0.93/0.98 (NOT LOW).
_KV_LOW_ON = 0.55     # load_LOW becomes TRUE when usage < this
_KV_LOW_OFF = 0.75    # load_LOW becomes FALSE when usage > this (hysteresis band 0.55-0.75)
# Class axis: rolling MEAN admission prefix-hit-rate threshold. Premise check (c96 mixed,
# SD off): per-request hit-rate is bimodal (~0 diverse / ~1 reusable); the 2s windowed mean
# tracks the traffic mix. theta_hit=0.45 sits mid-gap (reusable-heavy windows ~0.6-0.9,
# diverse-heavy ~0.0-0.3).
_HIT_HI = 0.45        # class_HIGHREUSE iff rolling mean hit-rate > this
# Rolling window for the hit-rate signal (wall seconds) and the minimum number of admitted
# requests in-window before the class signal is "decided" (else treat as not-highreuse;
# conservative — cold start is ON anyway via load axis at low load).
_WINDOW_S = 2.0
_MIN_ADMISSIONS = 5
# Minimum OFF duration before the controller may re-enter ON (flap guard on the OFF->ON
# edge only; the ON->OFF edge is immediate, matching Track B R3's T_off=1 s).
_COOLDOWN_S = 1.0


class Phase7Controller:
    def __init__(self, num_spec_tokens: int, mode: str | None = None):
        self.num_spec_tokens = num_spec_tokens
        # Mode is normally set via VLLM_TB_MODE; an explicit arg overrides (tests).
        self.mode = (mode or os.environ.get("VLLM_TB_MODE", "live")).lower()
        self.trace_path = os.environ.get("VLLM_TB_TRACE", "")
        # state: "on" or "off". Cold start ON (preserves the low-load benefit by default).
        self.state = "on"
        # rolling hit-rate samples: (wall_time, hit_rate) per admitted request.
        self._hit_samples: list[tuple[float, float]] = []
        # load-axis hysteresis state (None until first kv reading).
        self._load_is_low: bool | None = None
        # OFF->ON flap guard.
        self._off_since: float | None = None
        self._buf: list[str] = []

    @property
    def enabled(self) -> bool:
        return self.state == "on"

    # -- signal helpers ----------------------------------------------------------
    def _update_load(self, kv_usage: float) -> None:
        """Advance the load-axis hysteresis state. Cold start: need clearly low to be
        low (kv < _KV_LOW_ON). Once low, stay low until clearly high (> _KV_LOW_OFF);
        once high, re-enter low only when clearly low (< _KV_LOW_ON)."""
        if self._load_is_low is None:
            self._load_is_low = kv_usage < _KV_LOW_ON
        elif self._load_is_low:
            if kv_usage > _KV_LOW_OFF:
                self._load_is_low = False
        else:
            if kv_usage < _KV_LOW_ON:
                self._load_is_low = True

    @property
    def load_low(self) -> bool:
        return bool(self._load_is_low)

    def _hit_rate_mean(self) -> float | None:
        """Rolling mean admission hit-rate over the window, or None if too few admissions
        to decide."""
        if len(self._hit_samples) < _MIN_ADMISSIONS:
            return None
        return sum(hr for _, hr in self._hit_samples) / len(self._hit_samples)

    def class_highreuse(self) -> bool:
        m = self._hit_rate_mean()
        return m is not None and m > _HIT_HI

    # -- decision ---------------------------------------------------------------
    def evaluate(
        self,
        running: int,
        kv_usage: float,
        prev_hit_rate: float | None,
        prev_n_adm: int,
    ) -> bool:
        """Decide SD on/off for the step about to be scheduled. Called once per
        scheduler.schedule(), BEFORE running requests are processed (so an OFF decision
        can clear pending drafts). `prev_hit_rate`/`prev_n_adm` summarize the previous
        step's admitted requests (the class-signal sample): the mean hit-rate of those
        admissions and their count. Both axes are read from live state — no dependence on
        SD being on, so no probe is needed."""
        if self.mode == "native":
            return True
        now = _now()

        # 1) Record the previous step's admitted-request hit-rate samples. The scheduler
        #    passes one aggregate per step (mean + count); we append `count` copies of the
        #    mean so the window is admission-weighted, not step-weighted.
        if prev_hit_rate is not None and prev_n_adm > 0:
            for _ in range(min(prev_n_adm, 64)):  # cap per-step contribution
                self._hit_samples.append((now, prev_hit_rate))

        # 2) Prune the rolling window.
        cutoff = now - _WINDOW_S
        while self._hit_samples and self._hit_samples[0][0] < cutoff:
            self._hit_samples.pop(0)

        # 3) Advance the load-axis hysteresis state.
        self._update_load(kv_usage)

        if self.mode == "on":
            self.state = "on"
        elif self.mode == "off":
            self.state = "off"
        else:
            ll = self.load_low
            ch = self.class_highreuse()
            if self.mode == "loadonly":
                on = ll
            elif self.mode == "classonly":
                on = ch
            else:  # live — full 2-axis rule
                on = ll or ch

            if self.state == "on":
                if not on:
                    self.state = "off"
                    self._off_since = now
            elif self.state == "off":
                # Re-enter ON once `on` holds. The load axis bypasses the cooldown floor
                # (a genuine drop to low load is unambiguous and cheap — rescue now); the
                # class axis respects the cooldown as a flap guard (its 2 s window already
                # smooths it, so the cooldown only adds a minimum OFF duration).
                if on and (
                    ll or self._off_since is None or now - self._off_since >= _COOLDOWN_S
                ):
                    self.state = "on"
                    self._off_since = None

        return self.enabled

    # -- causal-verification trace ---------------------------------------------
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
        if not self.trace_path:
            return
        now = _now()
        m = self._hit_rate_mean()
        self._buf.append(
            json.dumps(
                {
                    "t": round(now, 6),
                    "sd_enabled": int(sd_enabled),
                    "state": self.state,
                    "mode": self.mode,
                    "kv_usage": round(kv_usage, 4),
                    "occupancy": round(running / max(1, max_num_seqs), 4),
                    "hit_rate": (round(m, 4) if m is not None else None),
                    "load_low": int(self.load_low),
                    "class_highreuse": int(self.class_highreuse()),
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


# Capstone modes that route to Phase7Controller (vs Track B's TBController). `on`/`off`
# are included so the constant arms run the SAME patched dispatch path (KV-usage read +
# hit-rate feed + evaluate) as the policy arms — CTRL-ON then isolates the patch/decision
# mechanism overhead, not a different code path. `native` stays in TBController (the
# zero-overhead always-on bypass = true DSpark-K7 baseline).
PHASE7_MODES = {"live", "loadonly", "classonly", "on", "off"}
