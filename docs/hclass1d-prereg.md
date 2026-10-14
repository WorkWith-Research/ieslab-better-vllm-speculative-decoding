# H-CLASS-1D prereg — held-out load-boundary validation of the scheduler signal

**Date frozen:** 2026-10-13 · **Status:** prospective (design + verdict rule fixed before any cell runs)
**Motivation:** `docs/scheduler-signal-validation.md` showed the 2-axis rule "SD-on iff low-load OR
high-reuse-class" matches the oracle on all 9 A/B/C × c16/48/96 cells — but that is a **same-data**
design-validation. This prereg makes the load-axis claim **held-out**: run two intermediate pressure points
(c32, c64) for condition A (the class where load flips the sign) and check the rule still holds out of sample,
and pin the exact on/off boundary.

## 1. What is being tested

The measured 9-cell payoff surface has, for condition A (diverse): **c16 = +0.122, c48 = −0.157, c96 = −0.156**.
The load axis therefore flips A from SD-favorable to SD-harmful somewhere between c16 and c48. The 2-axis rule's
"low-load" region is implicitly **{c16}** (the only load where A/C were positive). Two held-out probes:

- **c32** sits at the lambda midpoint of c16/c48 (32/28 = 1.1429, exactly between 0.5714 and 1.7143) → the
  boundary probe. The rule predicts A-c32 → **OFF**.
- **c64** sits between c48 and c96 (64/28 = 2.2857) → a deep-mid-load control. The rule predicts A-c64 → **OFF**.

If the load axis is real and monotone, payoff should be ≤ 0 at both (or, for c32, close to zero). If c32 is
clearly positive, the true boundary is lower than the rule assumes — a genuine held-out failure of the strict
boundary (while still confirming the 2-axis structure).

## 2. Pressure mapping (documented extension, not new calibration)

The frozen §5b confirmatory points are exactly `lambda = C/28` req/s (16→0.5714, 48→1.7143, 96→3.4286). The two
held-out points extend the **same definitional relationship**:

| cell | lambda (req/s) | client workers | server max_num_seqs |
|---|---:|---:|---:|
| A-c32 | 32/28 = **1.1429** | 32 | 64 |
| A-c64 | 64/28 = **2.2857** | 64 | 64 |

These are added to `hclass1_run.sh::lookup_lambda` with a comment citing this prereg (not a re-calibration).

## 3. Cells (minimal, single trial each)

| cell | arm | cond | C | lambda | dur |
|---|---|---|---:|---:|---:|
| ar_A_c32_t1 | AR | A | 32 | 1.1429 | 150 s |
| dspark_A_c32_t1 | DSpark-K7 | A | 32 | 1.1429 | 150 s |
| ar_A_c64_t1 | AR | A | 64 | 2.2857 | 150 s |
| dspark_A_c64_t1 | DSpark-K7 | A | 64 | 2.2857 | 150 s |

Condition A (diverse, `--no-enable-prefix-caching`), same server config / client / seed (20261013) / validity
gates as the rest of H-CLASS-1. One trial per cell is sufficient to locate the boundary within [c16,c48]; if c32
lands within ±0.03 of zero (ambiguous), a second trial (t2) will be added before verdict.

## 4. Metric

Per-request SD payoff at a load = mean over pid-matched pairs of `1 − DSparkTPOT/ARTPOT` (>0 ⇒ DSpark faster),
matching `docs/scheduler-signal-validation.md`. Reported with n (matched request count) and the per-cell DONE-VALID
status.

## 5. Preregistered verdict rule (fixed now)

Let `p32`, `p64` = measured payoff at A-c32, A-c64.

| outcome | condition | interpretation |
|---|---|---|
| **SUPPORTED** (boundary holds) | p32 ≤ 0 and p64 < 0 | A/C load boundary lies in [c16, c32]; "low-load = {c16}" is correct → rule as stated is right at all held-out points. |
| **PARTIALLY SUPPORTED** (boundary shift) | p32 > +0.05 and p64 < 0 | True boundary in [c32, c48]; the 2-axis *structure* holds but "low-load" should be {c16, c32}. Strict boundary refuted; scheduler threshold updated to lambda ≈ 1.14–1.71. |
| **REFUTED** (load axis weak) | p32 > +0.10 and p64 > +0.10 | A stays SD-favorable through mid-load → contradicts H-CLASS-1 C; the load axis is not a clean on/off driver. |

Tie/ambiguous band: if `|p32| ≤ 0.03`, run t2 and use the mean of t1+t2 for the verdict.

## 6. Scope limits (honest)

- Condition A only (the load-flip class). B is positive at all loads and C mirrors A (H-CLASS-1C), so A alone
  tests the load boundary; a held-out C would confirm class-invariance of the boundary but is not required for
  this claim.
- Fixed K=7. The *location* of the boundary will shift if K becomes a control input; the *existence* of a load
  flip is what's tested here.
- Single-GPU (GPU0), one server per cell, fresh each time — same isolation as the rest of the grid.

## 7. Deliverable

`docs/hclass1d-results.md` with the two payoff points, the boundary location, and the verdict against §5; a
PROGRESS entry; commit + push. No new project item unless the verdict is REFUTED (which would reopen the load-axis
question as an issue).
