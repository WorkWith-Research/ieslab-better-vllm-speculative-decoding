# Phase 7 (capstone) — request-aware SD controller (hit-rate class axis): deterministic
# causal-verification tests. Prereg docs/phase7-capstone-prereg.md §4.4 / S5 / [CAL-2].
# No GPU / no weights.
#
#   Part 1 (unit): Phase7Controller 2-axis rule + load hysteresis, hit-rate class signal
#                  (rolling window + min-admissions gate), OFF->ON cooldown flap guard,
#                  ablation modes, constant arms, causal trace fields.
#   Part 2 (integration): REAL Scheduler.schedule() path with VLLM_TB_MODE=live ->
#                  CAUSAL assertion: decision OFF => pending drafts cleared (0 new spec
#                  tokens scheduled); decision ON => K drafts persist and are scheduled.
#   Part 3: controller-type selection by mode (Phase7Controller iff mode in PHASE7_MODES).
#
# Run: .venv/bin/python -m pytest tests/v1/core/test_phase7_controller.py -q
import json

import pytest

from vllm.v1.core.sched import phase7_controller as p7mod
from vllm.v1.core.sched.phase7_controller import Phase7Controller, PHASE7_MODES
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput

import time
from .utils import create_requests, create_scheduler


class FakeClock:
    def __init__(self, t0: float = 1000.0):
        self.t = t0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@pytest.fixture(autouse=True)
def _restore_clock(monkeypatch):
    monkeypatch.setattr(p7mod, "_now", time.monotonic)
    yield


# --------------------------------------------------------------------------- #
# Part 1 — unit tests for Phase7Controller 2-axis decision logic (hit-rate class axis)
# --------------------------------------------------------------------------- #
class TestTwoAxisRule:
    def test_low_load_is_on_via_load_axis(self):
        """load_LOW (kv 0.30) drives ON even with no hit-rate history yet."""
        c = Phase7Controller(num_spec_tokens=7, mode="live")
        assert c.evaluate(running=16, kv_usage=0.30, prev_hit_rate=None, prev_n_adm=0) is True
        assert c.state == "on"

    def test_high_load_low_hitrate_is_off(self):
        """kv 0.95 (not low) + hit-rate below theta_hit -> OFF (both axes fail)."""
        c = Phase7Controller(num_spec_tokens=7, mode="live")
        for _ in range(3):
            c.evaluate(running=64, kv_usage=0.95, prev_hit_rate=0.1, prev_n_adm=10)
        assert c._hit_rate_mean() < p7mod._HIT_HI
        assert c.evaluate(running=64, kv_usage=0.95, prev_hit_rate=None, prev_n_adm=0) is False
        assert c.state == "off"

    def test_high_load_high_hitrate_is_on_via_class_axis(self):
        """kv 0.95 (not low) BUT hit-rate above theta_hit (reusable traffic) -> ON.
        The capstone's core: the SD-independent class axis keeps SD on at high load for
        reusable traffic, where the load-only arm would be OFF."""
        c = Phase7Controller(num_spec_tokens=7, mode="live")
        for _ in range(3):
            c.evaluate(running=64, kv_usage=0.95, prev_hit_rate=0.9, prev_n_adm=10)
        assert c._hit_rate_mean() > p7mod._HIT_HI
        assert c.evaluate(running=64, kv_usage=0.95, prev_hit_rate=None, prev_n_adm=0) is True
        assert c.state == "on"

    def test_load_axis_orthogonal_to_class(self):
        """The 2x2 payoff surface: ON iff load_LOW OR class_HIGHREUSE, independent of the
        other axis. Verify all four quadrants (>=5 admissions so class is decided)."""
        for kv in (0.30, 0.95):          # load LOW, HIGH
            for hr in (0.1, 0.9):        # class low (diverse), high (reusable)
                c = Phase7Controller(num_spec_tokens=7, mode="live")
                out = c.evaluate(running=64, kv_usage=kv, prev_hit_rate=hr, prev_n_adm=10)
                expect_on = (kv < p7mod._KV_LOW_ON) or (hr > p7mod._HIT_HI)
                assert out is expect_on, f"kv={kv} hr={hr} -> {out}, want {expect_on}"


class TestLoadHysteresis:
    def test_low_regime_stays_low_through_band(self):
        """Once in the low regime, usage rising only to 0.70 (< _KV_LOW_OFF) must NOT
        clear load_low — hysteresis prevents flapping at the boundary."""
        c = Phase7Controller(num_spec_tokens=7, mode="live")
        c.evaluate(running=16, kv_usage=0.30, prev_hit_rate=None, prev_n_adm=0)  # -> low
        assert c.load_low is True
        c.evaluate(running=40, kv_usage=0.70, prev_hit_rate=None, prev_n_adm=0)  # rise to band
        assert c.load_low is True      # still low (below 0.75 off threshold)

    def test_high_regime_requires_clear_drop_to_reenter(self):
        """Once OFF-high, load only re-enters LOW when usage < _KV_LOW_ON (0.55), not at
        the 0.75 line — asymmetric hysteresis band."""
        c = Phase7Controller(num_spec_tokens=7, mode="live")
        for _ in range(3):
            c.evaluate(running=64, kv_usage=0.95, prev_hit_rate=0.1, prev_n_adm=10)  # -> off/high
        assert c.state == "off" and c.load_low is False
        c.evaluate(running=64, kv_usage=0.70, prev_hit_rate=None, prev_n_adm=0)
        assert c.load_low is False     # 0.70 > 0.55 -> still high (no re-entry)
        c.evaluate(running=32, kv_usage=0.40, prev_hit_rate=None, prev_n_adm=0)
        assert c.load_low is True      # 0.40 < 0.55 -> re-enter low


class TestClassSignal:
    def test_rolling_window_prunes_old_samples(self):
        clock = FakeClock()
        p7mod._now = clock
        c = Phase7Controller(num_spec_tokens=7, mode="live")
        for _ in range(3):
            c.evaluate(running=64, kv_usage=0.95, prev_hit_rate=0.9, prev_n_adm=10)
        assert c.class_highreuse() is True
        # Advance past the window with no new admissions: samples age out -> mean None.
        clock.advance(p7mod._WINDOW_S + 1.0)
        c.evaluate(running=64, kv_usage=0.95, prev_hit_rate=None, prev_n_adm=0)
        assert c.class_highreuse() is False

    def test_min_admissions_gate(self):
        """Below _MIN_ADMISSIONS the class signal is 'not decided' (None), never high-reuse."""
        c = Phase7Controller(num_spec_tokens=7, mode="live")
        # A single admission with a high hit-rate but too few to decide.
        c.evaluate(running=64, kv_usage=0.95, prev_hit_rate=0.9, prev_n_adm=2)
        assert c._hit_rate_mean() is None
        assert c.class_highreuse() is False


class TestCooldownFlapGuard:
    def test_off_to_on_requires_cooldown(self):
        """Once OFF, a rising class signal must NOT immediately re-enter ON (flap guard);
        it re-enters only after the cooldown. The load axis bypasses this floor."""
        clock = FakeClock()
        p7mod._now = clock
        c = Phase7Controller(num_spec_tokens=7, mode="live")
        # Drive OFF at high load with low hit-rate (ON->OFF is immediate). 30 low samples.
        for _ in range(3):
            c.evaluate(running=64, kv_usage=0.95, prev_hit_rate=0.1, prev_n_adm=10)
        assert c.state == "off"
        # Reusable traffic returns (heavy: 64 admissions/step) but within the cooldown:
        # the window is now dominated by high hit-rate -> class fires...
        clock.advance(0.3)  # < _COOLDOWN_S
        out = c.evaluate(running=64, kv_usage=0.95, prev_hit_rate=0.9, prev_n_adm=64)
        assert c.class_highreuse() is True   # ...and the signal has crossed theta...
        assert out is False and c.state == "off"  # ...but the cooldown flap guard holds.
        # After the cooldown -> re-enter ON (no probe needed: class is directly observable).
        clock.advance(1.1)
        out = c.evaluate(running=64, kv_usage=0.95, prev_hit_rate=0.9, prev_n_adm=10)
        assert out is True and c.state == "on"

    def test_load_drop_rescues_on_immediately(self):
        """The load axis bypasses the cooldown: if load drops while OFF, it rescues ON
        immediately (unambiguous and cheap)."""
        clock = FakeClock()
        p7mod._now = clock
        c = Phase7Controller(num_spec_tokens=7, mode="live")
        for _ in range(3):
            c.evaluate(running=64, kv_usage=0.95, prev_hit_rate=0.1, prev_n_adm=10)  # off/high
        assert c.state == "off"
        clock.advance(0.01)
        assert c.evaluate(running=16, kv_usage=0.30, prev_hit_rate=None, prev_n_adm=0) is True
        assert c.state == "on"


class TestModes:
    def test_loadonly_ignores_class(self):
        """CTRL-LOADONLY ablation: high hit-rate must NOT turn it ON at high load."""
        c = Phase7Controller(num_spec_tokens=7, mode="loadonly")
        for _ in range(3):
            c.evaluate(running=64, kv_usage=0.95, prev_hit_rate=0.9, prev_n_adm=10)
        assert c.class_highreuse() is True  # class signal IS high...
        assert c.evaluate(running=64, kv_usage=0.95, prev_hit_rate=None, prev_n_adm=0) is False

    def test_classonly_ignores_load(self):
        """CTRL-CLASSONLY ablation: low load must NOT turn it ON with no hit-rate."""
        c = Phase7Controller(num_spec_tokens=7, mode="classonly")
        assert c.evaluate(running=16, kv_usage=0.30, prev_hit_rate=None, prev_n_adm=0) is False

    def test_mode_on_constant(self):
        c = Phase7Controller(num_spec_tokens=7, mode="on")
        assert c.evaluate(running=64, kv_usage=0.95, prev_hit_rate=None, prev_n_adm=0) is True
        assert c.state == "on"

    def test_mode_off_constant(self):
        c = Phase7Controller(num_spec_tokens=7, mode="off")
        assert c.evaluate(running=16, kv_usage=0.30, prev_hit_rate=0.9, prev_n_adm=10) is False
        assert c.state == "off"

    def test_mode_native_bypass(self):
        c = Phase7Controller(num_spec_tokens=7, mode="native")
        assert c.evaluate(running=64, kv_usage=0.95, prev_hit_rate=None, prev_n_adm=0) is True


class TestCausalTrace:
    def test_trace_fields_and_off_step_zero_drafts(self, tmp_path):
        p7mod._now = FakeClock()
        trace = str(tmp_path / "trace.jsonl")
        c = Phase7Controller(num_spec_tokens=7, mode="live")
        c.trace_path = trace
        for _ in range(3):
            c.evaluate(running=64, kv_usage=0.95, prev_hit_rate=0.1, prev_n_adm=10)
        assert c.state == "off"
        c.record_step(sd_enabled=False, running=64, waiting=0, kv_usage=0.95,
                      scheduled_tokens=64, draft_tokens_scheduled=0, prefill_tokens=64,
                      max_num_seqs=64)
        c.flush_trace()
        rows = [json.loads(l) for l in open(trace)]
        assert rows[-1]["sd_enabled"] == 0
        assert rows[-1]["drafts_scheduled_this_step"] == 0
        for k in ("t", "sd_enabled", "state", "kv_usage", "occupancy", "hit_rate",
                  "load_low", "class_highreuse", "drafts_scheduled_this_step"):
            assert k in rows[-1], f"missing trace field {k}"


# --------------------------------------------------------------------------- #
# Part 2 — integration: REAL Scheduler.schedule() causal assertion (VLLM_TB_MODE=live)
# --------------------------------------------------------------------------- #
def _make_running_decode_reqs(scheduler, n: int, K: int):
    """Add n requests, run a prefill step + update_from_output, then attach K pending
    draft tokens to each (as the prior ON step would have produced)."""
    requests = create_requests(num_requests=n, num_tokens=1)
    req_ids = [r.request_id for r in requests]
    for req in requests:
        scheduler.add_request(req)

    out = scheduler.schedule()  # prefill step: 1 token each
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

    drafts = [[1000 + i * 100 + j for j in range(K)] for i in range(n)]
    scheduler.update_draft_token_ids(DraftTokenIds(req_ids, drafts))
    return requests


class TestSchedulerCausalAssertionPhase7:
    def test_live_off_step_clears_pending_drafts(self, monkeypatch):
        """CAUSAL (S5): with VLLM_TB_MODE=live and the controller pinned OFF, a decode
        step schedules 0 new draft tokens (pending drafts cleared)."""
        K = 7
        monkeypatch.setenv("VLLM_TB_MODE", "live")
        scheduler = create_scheduler(num_speculative_tokens=K)
        assert isinstance(scheduler.tb_controller, Phase7Controller)
        requests = _make_running_decode_reqs(scheduler, n=4, K=K)

        monkeypatch.setattr(scheduler.tb_controller, "evaluate", lambda **k: False)
        out = scheduler.schedule()

        assert out.sd_enabled is False
        assert out.scheduled_spec_decode_tokens == {}
        for req in requests:
            assert req.spec_token_ids == []
            assert out.num_scheduled_tokens[req.request_id] == 1  # plain AR step

    def test_live_on_step_preserves_drafts(self, monkeypatch):
        """CAUSAL (S5): with the controller pinned ON, K drafts persist and are
        scheduled for every running request."""
        K = 7
        monkeypatch.setenv("VLLM_TB_MODE", "live")
        scheduler = create_scheduler(num_speculative_tokens=K)
        requests = _make_running_decode_reqs(scheduler, n=4, K=K)

        monkeypatch.setattr(scheduler.tb_controller, "evaluate", lambda **k: True)
        out = scheduler.schedule()

        assert out.sd_enabled is True
        for req in requests:
            rid = req.request_id
            assert out.num_scheduled_tokens[rid] == 1 + K
            assert len(out.scheduled_spec_decode_tokens[rid]) == K

    def test_live_real_controller_low_load_stays_on(self, monkeypatch):
        """With the REAL live controller and low KV usage, cold start stays ON, so a
        decode step with pending drafts still schedules them."""
        K = 7
        monkeypatch.setenv("VLLM_TB_MODE", "live")
        scheduler = create_scheduler(num_speculative_tokens=K)
        requests = _make_running_decode_reqs(scheduler, n=3, K=K)

        # Force the load axis to LOW (kv_cache_manager.usage is a read-only property;
        # patch the controller's hysteresis update instead so it reads as low).
        def _force_low(kv):
            scheduler.tb_controller._load_is_low = True
        monkeypatch.setattr(scheduler.tb_controller, "_update_load", _force_low)

        out = scheduler.schedule()
        assert out.sd_enabled is True
        for req in requests:
            assert len(out.scheduled_spec_decode_tokens[req.request_id]) == K


# --------------------------------------------------------------------------- #
# Part 3 — controller-type selection by mode
# --------------------------------------------------------------------------- #
class TestControllerSelection:
    def test_phase7_modes_select_phase7_controller(self, monkeypatch):
        for m in sorted(PHASE7_MODES):
            monkeypatch.setenv("VLLM_TB_MODE", m)
            s = create_scheduler(num_speculative_tokens=4)
            assert isinstance(s.tb_controller, Phase7Controller), f"mode {m}"

    def test_native_mode_selects_tb_controller(self, monkeypatch):
        from vllm.v1.core.sched.tb_controller import TBController
        monkeypatch.setenv("VLLM_TB_MODE", "native")
        s = create_scheduler(num_speculative_tokens=4)
        assert isinstance(s.tb_controller, TBController)

    def test_unset_mode_defaults_to_native(self, monkeypatch):
        from vllm.v1.core.sched.tb_controller import TBController
        monkeypatch.delenv("VLLM_TB_MODE", raising=False)
        s = create_scheduler(num_speculative_tokens=4)
        assert isinstance(s.tb_controller, TBController)
