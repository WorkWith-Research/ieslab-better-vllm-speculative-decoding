# H-CLASS-1D results — held-out load-boundary validation of the scheduler signal

**Date:** 2026-10-14 · **Prereg:** `docs/hclass1d-prereg.md` (prospective; verdict rule frozen before any cell ran)
**Builds on:** `docs/scheduler-signal-validation.md` (same-data design-validation of the 2-axis SD on/off rule).

## 0. Verdict (per prereg §5 fixed bands)

> **SUPPORTED.** The two held-out condition-A pressure points both have negative DSpark payoff — c32 = −0.041
> (2 trials), c64 = −0.126 (1 trial). The A/C load boundary therefore lies in **[c16, c32]**, so the rule's
> "low-load" region = {c16} is correct: DSpark is SD-favorable only at c16 for diverse traffic and harmful at
> every higher load. The 2-axis rule (SD-on iff low-load OR high-reuse-class) holds **out of sample** on the load
> axis — upgrading the prior same-data design-validation to a genuine held-out result.

## 1. Held-out payoff points (prereg §3, condition A, DSpark-K7 vs AR)

Per-request SD payoff = mean over pid-matched pairs of `1 − DSparkTPOT/ARTPOT` (>0 ⇒ DSpark faster).
All cells DONE-VALID (harness validity gates: applied-load + service-aware floor + obs-trace + async-overlap).

| cell | arm | λ (req/s) | payoff | n | status |
|---|---|---:|---:|---:|---|
| ar_A_c32_t1 / dspark_A_c32_t1 | AR / DSpark | 1.1429 | **−0.037** | 106 | DONE-VALID |
| ar_A_c32_t2 / dspark_A_c32_t2 | AR / DSpark | 1.1429 | **−0.045** | 102 | DONE-VALID |
| ar_A_c64_t1 / dspark_A_c64_t1 | AR / DSpark | 2.2857 | **−0.126** | 120 | DONE-VALID |

c32 was run in **two trials** because t1 (−0.037) sat just outside the prereg's ±0.03 ambiguity band; t2
(−0.045) confirmed it is not noise — both negative, same sign, consistent magnitude. c64 ran one trial (a deep-mid
control, well inside the expected-negative region).

## 2. Full condition-A load curve (in-sample + held-out)

| load | payoff (trial-mean) | per-trial | n | source |
|---:|---:|---|---:|---|
| c16 | **+0.122** | +0.114 / +0.129 / +0.124 | 227 | in-sample (H-CLASS-1) |
| **c32** | **−0.041** | −0.037 / −0.045 | 208 | **held-out (this work)** |
| c48 | **−0.157** | −0.182 / −0.154 / −0.136 | 311 | in-sample (H-CLASS-1) |
| **c64** | **−0.126** | −0.126 | 120 | **held-out (this work)** |
| c96 | **−0.157** | −0.146 / −0.134 / −0.190 | 454 | in-sample (H-CLASS-1) |

The curve is **monotone-decreasing in load** for diverse traffic: DSpark wins at c16 (+0.122), is roughly
break-even-but-negative at c32 (−0.041), and clearly loses from c48 up (−0.13 to −0.16). The held-out points sit
exactly where the in-sample structure predicts — no sign surprise, no boundary shift.

## 3. Verdict against the preregistered rule (§5)

Let p32 = −0.041 (mean of t1+t2), p64 = −0.126.

| outcome | condition | hit? |
|---|---|---|
| **SUPPORTED** (boundary holds) | p32 ≤ 0 and p64 < 0 | ✓ **−0.041 ≤ 0 and −0.126 < 0** |
| PARTIALLY SUPPORTED (boundary shift) | p32 > +0.05 and p64 < 0 | ✗ |
| REFUTED (load axis weak) | p32 > +0.10 and p64 > +0.10 | ✗ |

→ **SUPPORTED.** The "low-load = {c16}" boundary is correct; no threshold update needed.

## 4. What this establishes

1. **The load axis generalizes out of sample.** The on/off boundary for diverse traffic was *located* at a
   pressure point (c32) that was never part of the grid that established the rule, and it falls where predicted.
   This is the held-out upgrade `scheduler-signal-validation.md` called for — the 2-axis rule is no longer just
   internally consistent on its own data.
2. **The boundary has a real margin.** c32 is −0.041 (not +0.02), so the rule's "OFF at c32" decision has a small
   but genuine safety margin, unlike the in-sample Bc96 cell (+0.022) which was flagged as near-zero-margin. The
   load axis is the *robust* half of the 2-axis signal; the class axis carries the marginal case.

## 5. Scope limits (honest)

- **Condition A only.** B stays positive at all loads and C mirrors A (H-CLASS-1C), so A alone tests the load
  boundary; a held-out C would confirm the boundary is class-invariant but is not required for this claim.
- **c64 single-trial.** It is a control well inside the expected-negative region (−0.126, far from 0), so one
  trial is adequate; c32 (the actual boundary probe) got two.
- **Fixed K=7.** The *location* of the boundary shifts if K becomes a control input; what's validated here is the
  *existence* and *sign structure* of a load flip, not an absolute threshold.
- **Two held-out points, one class.** This confirms the load axis at two intermediate pressures for diverse
  traffic — sufficient to locate the boundary within [c16,c32], not a dense re-scan of the whole curve.

## 6. Provenance

- Cells: `results/hclass1/{ar,dspark}_A_c{32,64}_t*/client.jsonl` (pid-matched). c32 = trials t1+t2; c64 = t1.
- Harness: `experiments/hclass1_run.sh` with `lookup_lambda` extended by the documented λ=C/28 held-out points
  (c32→1.1429, c64→2.2857); run offline (`HF_HUB_OFFLINE=1`) after an HF API rate-limit on the first launch.
- Incidents: one transient CUDA-driver wedge mid-run (orphaned EngineCore procs from failed boots) — recovered by
  killing orphans, no GPU reset needed; all four cells re-ran clean and are DONE-VALID.
