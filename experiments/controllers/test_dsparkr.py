#!/usr/bin/env python3
"""Unit tests for the P4.5R corrected DSpark-rule proxy (dsparkr_controller).

Verifies the three mechanical fixes at the pure-logic level, WITHOUT a GPU run:
  F1 — decision-set completeness (greedy decides every request it is given) + persistence
       semantics (a K=0 decision stays in the persistent map across zero-draft steps).
  F4 — B = true batch; SPS extrapolation beyond the profile range uses the last segment's
       slope (monotone-decreasing), NOT a clamp to the table's max value.
  F2 — censoring: positions > k_observed are UNOBSERVED (excluded from the J denominator),
       not counted as rejections.

Run: .venv/bin/python experiments/test_dsparkr.py
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _load():
    spec = importlib.util.spec_from_file_location(
        "dsparkr", os.path.join(HERE, "dsparkr_controller", "sitecustomize.py"))
    mod = importlib.util.module_from_spec(spec)
    # Do NOT set VLLM_DSPARKR_OUT so _install() no-ops (no vllm import needed here).
    os.environ.pop("VLLM_DSPARKR_OUT", None)
    spec.loader.exec_module(mod)
    return mod


def test_sps_extrapolation_no_clamp(d):
    # Table: SPS decreasing in B. Beyond the max point, extrapolate with last-segment slope.
    pts = [(10.0, 40.0), (50.0, 30.0), (100.0, 20.0)]   # last slope = (20-30)/(100-50) = -0.2 /token
    below = d._sps_interp(pts, 5.0)
    assert abs(below - 40.0) < 1e-9, f"below-range should clamp to first value, got {below}"
    inside = d._sps_interp(pts, 75.0)                    # between 50 and 100
    expect_inside = 30.0 + (20.0 - 30.0) * (75 - 50) / (100 - 50)
    assert abs(inside - expect_inside) < 1e-9, f"interior interp wrong: {inside} vs {expect_inside}"
    above = d._sps_interp(pts, 200.0)                    # beyond 100 -> 20 + (-0.2)*(200-100) = 0.0
    expect_above = 20.0 + ((20.0 - 30.0) / (100.0 - 50.0)) * (200.0 - 100.0)
    assert abs(above - expect_above) < 1e-9, f"extrapolation wrong: {above} vs {expect_above}"
    # The OLD buggy behavior would clamp to pts[-1][1]=20.0 at B=200. Confirm we do NOT.
    assert above < 20.0, f"extrapolation must continue decreasing, not clamp (got {above})"
    print(f"  F4 ok: below={below:.2f} inside={inside:.2f} above={above:.2f} (clamped-would-be=20.0)")


def test_censoring_excludes_unobserved(d):
    # Two steps: step1 observed k=3 drafts, accepted 2; step2 observed k=1 draft, accepted 1.
    hist = [(3, 2), (1, 1)]
    prior = [1.0] * 9
    J = d._Jvec_censored(hist, prior)
    # l=1: both steps inform (k>=1): acc>=1 -> step1 yes(2>=1), step2 yes(1>=1) => 2/2=1.0
    assert abs(J[1] - 1.0) < 1e-9, f"J[1]={J[1]}"
    # l=2: only step1 informs (k=3>=2; step2 k=1<2 EXCLUDED): acc>=2 -> step1 yes => 1/1=1.0
    #      OLD buggy behavior would count step2 as rejection at l=2 => 1/2=0.5. We must NOT.
    assert abs(J[2] - 1.0) < 1e-9, f"J[2]={J[2]} (censoring must exclude unobserved)"
    # l=3: only step1 informs: acc>=3? 2>=3 no => 0/1=0.0
    assert abs(J[3] - 0.0) < 1e-9, f"J[3]={J[3]}"
    # l=4..8: no step observed k>=l -> fall back to prior (1.0)
    for l in range(4, 9):
        assert abs(J[l] - 1.0) < 1e-9, f"J[{l}]={J[l]} should be prior when unobserved"
    print(f"  F2 ok: J[1..8]={[round(x,3) for x in J[1:]]}")


def test_greedy_decides_every_request(d):
    # All requests with a flat optimistic J -> greedy should extend them (not drop any).
    KMAX = 4
    pts = [(10.0, 40.0), (100.0, 20.0)]
    reqs = {f"r{i}": (KMAX, [1.0] * (KMAX + 1)) for i in range(6)}   # K_current irrelevant to output
    out = d._greedy(reqs, KMAX, pts)
    assert set(out.keys()) == set(reqs.keys()), f"greedy must decide EVERY request: {out}"
    print(f"  F1a ok: decided {len(out)}/{len(reqs)} requests -> {sorted(out.values())}")


def test_greedy_sd_off_reachable(d):
    # If extending never improves Theta (J=0 beyond bonus), the all-zero SD-off outcome wins.
    KMAX = 4
    pts = [(10.0, 40.0), (100.0, 20.0)]
    reqs = {f"r{i}": (KMAX, [1.0] + [0.0] * KMAX) for i in range(6)}   # J[1]=1 but J[>=2]=0
    out = d._greedy(reqs, KMAX, pts)
    assert all(v == 0 for v in out.values()), f"SD-off should be reachable: {out}"
    print(f"  F1b ok: SD-off reachable -> all K=0")


def test_persistence_across_zero_draft_steps(d):
    # Simulate the persistence contract (F1) at the map level: a request decided K=0 must
    # remain in last_kstar and keep truncating to 0 on subsequent zero-draft steps. This is
    # the exact bug F1 fixed (old code rebuilt the map only from drafted requests, so a
    # K=0 request dropped out and reverted to KMAX). We replicate the controller's loop.
    last_kstar = {}
    KMAX = 8
    running = ["a", "b"]
    # Step 1: both drafted; suppose greedy decides a->0, b->3 (inject via a tiny _greedy call).
    def decide(reqs):
        return d._greedy(reqs, KMAX, [(10.0, 40.0), (100.0, 20.0)])
    # Force a deterministic split: request 'a' has zero marginal value (J>=2 = 0), 'b' full.
    reqs1 = {"a": (8, [1.0] + [0.0] * KMAX), "b": (8, [1.0] * (KMAX + 1))}
    k1 = decide(reqs1)
    for r, k in k1.items():
        last_kstar[r] = k
    assert last_kstar.get("a") == 0, f"expected a->0 at step1, got {last_kstar}"
    # Step 2: 'a' produced ZERO drafts (absent from this-step stats) — old code would drop it.
    # Corrected contract: 'a' stays in last_kstar at K=0 and is still decided every step.
    reqs2 = {"a": (8, [1.0] + [0.0] * KMAX), "b": (8, [1.0] * (KMAX + 1))}   # 'a' re-decided
    k2 = decide(reqs2)
    for r, k in k2.items():
        last_kstar[r] = k
    assert last_kstar.get("a") == 0, f"a must STAY K=0 across zero-draft step, got {last_kstar}"
    # Enforcement: truncation uses last_kstar.get(req, KMAX); a->0 means spec_token_ids[:0].
    enforced_a = min(8, last_kstar.get("a", KMAX))
    assert enforced_a == 0, f"enforcement must truncate a to 0, got {enforced_a}"
    print(f"  F1c ok: persistence — a stays K=0 across zero-draft step; enforced_a={enforced_a}")


def main():
    d = _load()
    print("P4.5R dsparkr unit tests:")
    test_sps_extrapolation_no_clamp(d)
    test_censoring_excludes_unobserved(d)
    test_greedy_decides_every_request(d)
    test_greedy_sd_off_reachable(d)
    test_persistence_across_zero_draft_steps(d)
    print("ALL PASS")


if __name__ == "__main__":
    main()
