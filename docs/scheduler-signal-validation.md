# Scheduler-signal validation — load-regime × traffic-class SD on/off policy

**Date:** 2026-10-13 · **Builds on:** `docs/tracka-results.md` (Track A negative result) and H-CLASS-1 / 1C.
**Question:** does the revised two-axis signal — *load regime* (sign of SD payoff) + *traffic class*
(expected acceptance level) — actually beat single-axis policies on the measured payoff surface?

This is an **analysis of existing H-CLASS-1 A/B/C data** (no new GPU runs). It confirms that the design
implied by Track A's negative result is the right one.

## 0. Verdict

> **SUPPORTED.** The two-axis rule "SD-on iff low-load OR high-reuse-class" recovers the measured payoff
> surface exactly — matching the oracle on all 9 (load × class) cells, capturing 100% of achievable gain with
> zero regret. Every single-axis alternative is strictly worse. This validates setting SD on/off from
> **serving state + traffic class**, not per-request content.

## 1. Measured payoff surface

Per-request SD payoff = mean over pid-matched pairs of `1 − DSparkTPOT/ARTPOT` (>0 ⇒ DSpark is faster),
aggregated over the 3 traffic mixes (t=1,2,3) per cell. n = matched request count.

| cond \ load | c16 (low) | c48 (mid) | c96 (high) |
|---|---:|---:|---:|
| **A** diverse | **+0.122** (n227) | **−0.157** (n311) | **−0.156** (n454) |
| **B** high-reuse | **+0.477** (n648) | **+0.121** (n1200) | **+0.022** (n1530) |
| **C** shared-doc | **+0.109** (n226) | **−0.172** (n323) | **−0.186** (n470) |

Two independent axes, both load-bearing:
- **Load flips A/C** from + (c16) to − (c48/c96): at low load DSpark's draft overhead is amortized and it wins;
  at mid/high load the extra forward passes cost more than they save. (Reproduces H-CLASS-1 C.)
- **Traffic class keeps B positive at every load**: high-reuse prompts are so much more draftable
  (acceptance ~0.61 vs ~0.33) that DSpark stays ahead even at c96, where it loses on diverse traffic.

## 2. Policy decision quality (request-count-weighted, n = 5389 total)

Oracle = "SD-on iff measured cell payoff > 0". A policy's *captured gain* = Σ over cells it turns ON of
payoff·n; regret = fraction of oracle's achievable gain left on the table.

| policy | SD-on cells | captured gain | regret | wrong cells |
|---|---:|---:|---:|---|
| **ORACLE** | 5/9 | 100.0% | 0.0% | — |
| **load OR class (proposed)** | 5/9 | **100.0%** | **0.0%** | none |
| class-only (`B` → on) | 3/9 | 90.3% | 9.7% | Ac16, Cc16 |
| load-only (c16 → on) | 3/9 | 67.0% | 33.0% | Bc48, Bc96 |
| always-on | 9/9 | 51.3% | 48.7% | Ac48, Ac96, Cc48, Cc96 |
| always-off | 0/9 | 0.0% | 100.0% | all 5 positive cells |

Per-cell agreement of the proposed rule with the oracle: **9/9 ✓** (Ac16/Bc16/Cc16 ON by load; Bc48/Bc96 ON
by class; Ac48/Ac96/Cc48/Cc96 OFF). The 2D rule is a *disjunction* of the two axes — each axis alone covers a
different half of the surface, and only their union is complete.

## 3. What this establishes (and what it doesn't)

**Establishes:**
1. The Track-A negative result was *diagnostic*, not a dead end: per-request content prediction is weak
   (R²≈0.05), but the **population-level** signal (load × class) that H-CLASS-1/1C measured is strong enough to
   drive an on/off decision with no regret on this surface. The right granularity is *workload regime*, not
   individual request.
2. **Both axes are necessary.** Load alone forfeits the high-load B wins; class alone forfeits the low-load
   A/C wins. A lightweight scheduler must ingest serving state (queue/running ⇒ load) *and* a coarse traffic-class
   label — exactly the design Track A's interpretation predicted.

**Scope limits (honest):**
- **Same-data validation.** This confirms the 2D rule recovers the measured surface and dominates single-axis
  policies *on the data that established the structure*. It is an internal-consistency / design-validation
  result, not a held-out generalization test. A true claim of out-of-sample performance needs new cells:
  intermediate loads (c32/c64), more traffic classes, or a different draft/target pair.
- **Marginal boundary cell.** Bc96 payoff is only **+0.022** (n1530) — the rule says ON and it matches the
  oracle, but with essentially no margin of safety. In practice this is the cell where the on/off decision is
  least robust to noise; a conservative deployment might treat |payoff| < ~0.03 as "indifferent" (leave at
  default). The other 8 cells have |payoff| ≥ 0.11.
- Payoff measured at fixed K=7 (DSpark default). The *level* of payoff (and thus the exact on/off boundary at
  Bc96) will shift with K; the *structure* (load flips A/C, class rescues B) is expected to hold but should be
  re-checked if K becomes a control input.

## 4. Provenance

- Surface + policy table computed from `results/hclass1/{ar,dspark}_{A,B,C}_c{16,48,96}_t{1,2,3}/client.jsonl`
  (pid-matched). Script logic: payoff = mean(1−DSparkTPOT/ARTPOT) per cell; policies compared request-weighted.
- Inputs are the committed H-CLASS-1 artifacts; no new experiments in this note.
