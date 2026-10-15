# Track B — deterministic causal-verification tests (prereg docs/trackb-prereg.md §4.4).
#
# No GPU / no model weights required (scheduler + controller logic only).
#
#   Part 1 (unit): TBController decision logic driven by an injected fake clock.
#                  Cold-start-ON, OFF exit when measured cost > AR*(1+band),
#                  cooldown gating of re-entry, hysteresis re-entry, prefill-step
#                  exclusion, idle-gap exclusion, constant modes, Phase-6 fit.
#   Part 2 (integration): the REAL Scheduler.schedule() path (vLLM's own
#                  create_scheduler/create_requests helpers). CAUSAL assertion:
#                    - decision OFF  => pending request.spec_token_ids cleared, so
#                      scheduled_spec_decode_tokens is empty and each running decode
#                      request schedules exactly 1 token (plain AR step);
#                    - decision ON   => K drafts persist and are scheduled.
#   Part 3: SchedulerOutput.sd_enabled field presence + default.
#
# Run:  .venv/bin/python -m pytest tests/v1/core/test_tb_controller.py -q
import time

import pytest

from vllm.v1.core.sched import tb_controller as tbmod
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.core.sched.tb_controller import TBController
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput

from .utils import create_requests, create_scheduler


# --------------------------------------------------------------------------- #
# Fake clock (injectable into tbmod._now)
# --------------------------------------------------------------------------- #
class FakeClock:
    def __init__(self, t0: float = 1000.0):
        self.t = t0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@pytest.fixture(autouse=True)
def _restore_clock(monkeypatch):
    monkeypatch.setattr(tbmod, "_now", time.monotonic)
    yield


# --------------------------------------------------------------------------- #
# Part 1 — unit tests for TBController decision logic (probe-based design)
# --------------------------------------------------------------------------- #
def _drive_to_off(c, clock):
    """Cold start + 3 expensive SD-on decode steps at n=64 -> state OFF.

    Sequence of evaluate() returns: True (cold), True (1 sample <3),
    True (2 samples <3), False (3rd sample crosses exit band). The helper
    drives the controller and asserts only the final state, not intermediates.
    """
    assert c.evaluate(running=64, scheduled_tokens=512, prefill_tokens=0) is True  # cold, ON
    for _ in range(3):
        clock.advance(0.131)  # 131 ms > exit band thr(64)*1.15 = 110.2 ms
        c.evaluate(running=64, scheduled_tokens=512, prefill_tokens=0)
    assert c.state == "off"


class TestTBControllerDecision:
    def test_cold_start_is_on(self):
        c = TBController(num_spec_tokens=7)
        assert c.evaluate(running=64, scheduled_tokens=512, prefill_tokens=0) is True
        assert c.state == "on"

    def test_off_exit_when_measured_cost_exceeds_threshold(self):
        clock = FakeClock()
        tbmod._now = clock
        c = TBController(num_spec_tokens=7)
        # thr(64) = 2.77*34.56 = 95.7 ms; exit band = 110.2 ms. Feed 131 ms steps.
        assert c.evaluate(running=64, scheduled_tokens=512, prefill_tokens=0) is True  # cold
        clock.advance(0.131); assert c.evaluate(running=64, scheduled_tokens=512, prefill_tokens=0) is True   # sample 1 (<3)
        clock.advance(0.131); assert c.evaluate(running=64, scheduled_tokens=512, prefill_tokens=0) is True   # sample 2 (<3)
        clock.advance(0.131); assert c.evaluate(running=64, scheduled_tokens=512, prefill_tokens=0) is False  # sample 3 -> OFF
        assert c.state == "off"

    def test_off_stays_off_until_cooldown_then_probes(self):
        clock = FakeClock()
        tbmod._now = clock
        c = TBController(num_spec_tokens=7)
        _drive_to_off(c, clock)
        # While OFF (no SD-on samples accumulate), stays OFF through the cooldown.
        for _ in range(3):
            clock.advance(0.035)  # AR-like steps; state off -> not sampled
            assert c.evaluate(running=64, scheduled_tokens=64, prefill_tokens=0) is False
        # Cross the 1 s cooldown -> next step starts a probe (SD on again).
        clock.advance(tbmod._COOLDOWN_S + 0.035)
        assert c.evaluate(running=64, scheduled_tokens=64, prefill_tokens=0) is True
        assert c.state == "probe"

    def test_probe_commit_on_when_dspark_cheap(self):
        clock = FakeClock()
        tbmod._now = clock
        c = TBController(num_spec_tokens=7)
        _drive_to_off(c, clock)
        clock.advance(tbmod._COOLDOWN_S + 0.035)
        assert c.evaluate(running=64, scheduled_tokens=64, prefill_tokens=0) is True  # probe step 1
        # Probe steps with cheap DSpark cost (35 ms < commit band 81.4 ms).
        for _ in range(2):
            clock.advance(0.035)
            c.evaluate(running=64, scheduled_tokens=64, prefill_tokens=0)
        # After probe_steps(3) with >=2 samples -> commit ON.
        clock.advance(0.035)
        assert c.evaluate(running=64, scheduled_tokens=64, prefill_tokens=0) is True
        assert c.state == "on"

    def test_probe_revert_off_when_dspark_expensive(self):
        clock = FakeClock()
        tbmod._now = clock
        c = TBController(num_spec_tokens=7)
        _drive_to_off(c, clock)
        clock.advance(tbmod._COOLDOWN_S + 0.131)
        assert c.evaluate(running=64, scheduled_tokens=64, prefill_tokens=0) is True  # probe step 1 (131 ms sample)
        for _ in range(2):
            clock.advance(0.131)
            c.evaluate(running=64, scheduled_tokens=64, prefill_tokens=0)
        clock.advance(0.131)
        assert c.evaluate(running=64, scheduled_tokens=64, prefill_tokens=0) is False
        assert c.state == "off"
        # Cooldown extended (>= 2 s): a 1.5 s gap must NOT start a new probe.
        clock.advance(1.5)
        assert c.evaluate(running=64, scheduled_tokens=64, prefill_tokens=0) is False
        assert c.state == "off"

    def test_low_load_stays_on(self):
        """Phase 6 C=8 regime: DSpark step cost ~9 ms << threshold -> stays ON."""
        clock = FakeClock()
        tbmod._now = clock
        c = TBController(num_spec_tokens=7)
        assert c.evaluate(running=8, scheduled_tokens=64, prefill_tokens=0) is True
        for _ in range(10):
            clock.advance(0.025)  # ~9 ms TPOT * 2.77 acc_len ≈ 25 ms/step
            assert c.evaluate(running=8, scheduled_tokens=64, prefill_tokens=0) is True
        assert c.state == "on"

    def test_prefill_heavy_steps_excluded_from_window(self):
        clock = FakeClock()
        tbmod._now = clock
        c = TBController(num_spec_tokens=7)
        c.evaluate(running=64, scheduled_tokens=512, prefill_tokens=0)
        # 90% of tokens are prefill (non-draft) -> not a decode sample.
        clock.advance(0.131)
        out = c.evaluate(running=64, scheduled_tokens=512, prefill_tokens=460)
        assert out is True  # only cold step seen (<3 samples) -> still ON

    def test_idle_gap_not_sampled(self):
        clock = FakeClock()
        tbmod._now = clock
        c = TBController(num_spec_tokens=7)
        c.evaluate(running=64, scheduled_tokens=512, prefill_tokens=0)
        clock.advance(10.0)  # idle gap -> previous step not recorded as a cost
        assert c.evaluate(running=64, scheduled_tokens=512, prefill_tokens=0) is True

    def test_mode_on_always_enabled(self):
        c = TBController(num_spec_tokens=7)
        c.mode = "on"
        assert c.evaluate(running=64, scheduled_tokens=512, prefill_tokens=0) is True
        assert c.state == "on"

    def test_mode_off_always_disabled(self):
        c = TBController(num_spec_tokens=7)
        c.mode = "off"
        assert c.evaluate(running=8, scheduled_tokens=64, prefill_tokens=0) is False
        assert c.state == "off"

    def test_single_transient_spike_does_not_flip_off(self):
        """Regression: prereg R3 specifies a full rolling-window mean (W=2 s).
        A single transient decode spike must NOT push the exit decision over the
        band. With the old last-4 mean, one 110 ms spike among ~20 ms steps gave
        mean(last4)=42.5 > exit band 41.6 at C=8 and caused on->off flapping;
        the full window (~100 samples) absorbs it (mean ~21 ms)."""
        clock = FakeClock()
        tbmod._now = clock
        c = TBController(num_spec_tokens=7)
        # threshold(8)=36.19, exit band=41.6. Feed 30 normal ~20 ms SD-on decode
        # steps with ONE 110 ms spike in the middle; all pure decode (pf=0).
        assert c.evaluate(running=8, scheduled_tokens=80, prefill_tokens=0) is True  # cold
        for i in range(30):
            clock.advance(0.110 if i == 15 else 0.020)
            assert c.evaluate(running=8, scheduled_tokens=80, prefill_tokens=0) is True
        assert c.state == "on", f"flapped to {c.state} on a single spike"

    def test_sustained_cost_increase_still_flips_off(self):
        """The fix must not blind the controller: a SUSTAINED rise above the
        exit band (the real saturation signal) still flips OFF."""
        clock = FakeClock()
        tbmod._now = clock
        c = TBController(num_spec_tokens=7)
        # 20 cheap steps, then 40 steps at ~50 ms (above exit band 41.6).
        assert c.evaluate(running=8, scheduled_tokens=80, prefill_tokens=0) is True  # cold
        for _ in range(20):
            clock.advance(0.020)
            c.evaluate(running=8, scheduled_tokens=80, prefill_tokens=0)
        assert c.state == "on"
        for _ in range(40):
            clock.advance(0.050)
            c.evaluate(running=8, scheduled_tokens=80, prefill_tokens=0)
            if c.state == "off":
                break
        assert c.state == "off", "sustained cost rise did not trigger exit"

    def test_mode_native_is_zero_overhead_bypass(self):
        """Native mode (DSpark-K7 baseline arm): always ON, no state mutation."""
        c = TBController(num_spec_tokens=7)
        c.mode = "native"
        c.active = False  # mirrors __init__ for the test
        assert c.evaluate(running=64, scheduled_tokens=512, prefill_tokens=0) is True
        assert c.state == "on" and c._samples == []

    def test_est_ar_model_matches_phase6_fit(self):
        c = TBController(num_spec_tokens=7)
        # Phase 6 AR TPOT: 13.55 @8, 21.43 @32, 34.92 @64 (model within ~4%).
        assert abs(c.est_ar_ms(8) - 13.06) < 0.5
        assert abs(c.est_ar_ms(32) - 22.28) < 0.5
        assert abs(c.est_ar_ms(64) - 34.56) < 0.5

    def test_threshold_is_l_scaled(self):
        c = TBController(num_spec_tokens=7)
        # threshold(N) = L * est_ar(N); check the scaling and Phase-6 ordering:
        # C=8 DSpark step (~25 ms) below thr(8)=41.7; C=64 DSpark step (~131 ms)
        # above exit band thr(64)*1.15=110.2.
        assert abs(c.threshold_ms(8) - 2.77 * c.est_ar_ms(8)) < 1e-9
        assert 25.0 < c.threshold_ms(8)
        assert 131.0 > c.threshold_ms(64) * (1.0 + tbmod._BAND)


# --------------------------------------------------------------------------- #
# Part 2 — integration: REAL Scheduler.schedule() causal assertion
# --------------------------------------------------------------------------- #
def _make_running_decode_reqs(scheduler, n: int, K: int):
    """Add n requests, run a prefill step + update_from_output, then attach K
    pending draft tokens to each (as the prior ON step would have produced)."""
    requests = create_requests(num_requests=n, num_tokens=1)
    req_ids = [r.request_id for r in requests]
    for req in requests:
        scheduler.add_request(req)

    # Prefill step: 1 token each.
    out = scheduler.schedule()
    assert all(out.num_scheduled_tokens[rid] == 1 for rid in req_ids)

    mro = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
        sampled_token_ids=[[0] for _ in range(n)],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(out, mro)

    # Attach K pending draft tokens to every running request.
    drafts = [[1000 + i * 100 + j for j in range(K)] for i in range(n)]
    scheduler.update_draft_token_ids(DraftTokenIds(req_ids, drafts))
    return requests


class TestSchedulerCausalAssertion:
    def test_off_step_clears_pending_drafts(self, monkeypatch):
        """CAUSAL: decision OFF => 0 new draft tokens scheduled this step."""
        K = 7
        scheduler = create_scheduler(num_speculative_tokens=K)
        requests = _make_running_decode_reqs(scheduler, n=4, K=K)

        # Pin the controller to decide OFF for the decode step.
        monkeypatch.setattr(
            scheduler.tb_controller, "evaluate", lambda **k: False
        )

        out = scheduler.schedule()

        assert out.sd_enabled is False
        # Pending drafts cleared -> no new spec tokens scheduled this step.
        assert out.scheduled_spec_decode_tokens == {}
        for req in requests:
            assert req.spec_token_ids == []
            # Plain AR decode step: exactly 1 token per request.
            assert out.num_scheduled_tokens[req.request_id] == 1

    def test_on_step_preserves_drafts(self, monkeypatch):
        """CAUSAL: decision ON => K drafts persist and are scheduled."""
        K = 7
        scheduler = create_scheduler(num_speculative_tokens=K)
        requests = _make_running_decode_reqs(scheduler, n=4, K=K)

        # Pin the controller to decide ON for the decode step.
        monkeypatch.setattr(
            scheduler.tb_controller, "evaluate", lambda **k: True
        )

        out = scheduler.schedule()

        assert out.sd_enabled is True
        total = 0
        for req in requests:
            rid = req.request_id
            # 1 sampled + K draft tokens scheduled.
            assert out.num_scheduled_tokens[rid] == 1 + K
            assert len(out.scheduled_spec_decode_tokens[rid]) == K
            total += len(out.scheduled_spec_decode_tokens[rid])
        assert total == 4 * K

    def test_real_controller_cold_start_keeps_drafts(self):
        """With the REAL controller (no monkeypatch), cold start => ON, so a
        decode step with pending drafts still schedules them (no behavior change
        before any measured cost accumulates)."""
        K = 7
        scheduler = create_scheduler(num_speculative_tokens=K)
        requests = _make_running_decode_reqs(scheduler, n=3, K=K)

        out = scheduler.schedule()

        assert out.sd_enabled is True
        for req in requests:
            assert len(out.scheduled_spec_decode_tokens[req.request_id]) == K


# --------------------------------------------------------------------------- #
# Part 3 — SchedulerOutput.sd_enabled field presence / default
# --------------------------------------------------------------------------- #
def _mk_output(**overrides):
    base = dict(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    base.update(overrides)
    return SchedulerOutput(**base)


class TestSchedulerOutputField:
    def test_sd_enabled_defaults_true(self):
        assert _mk_output().sd_enabled is True

    def test_sd_enabled_settable_false(self):
        assert _mk_output(sd_enabled=False).sd_enabled is False
