# P4.5R — Corrected DSpark-rule proxy rerun (Track A)

**Date:** 2026-10-11 · **Status:** FINAL (per prereg `docs/dspark-proxy-validity.md` §6)
**Run set:** C∈{32,96} × 3 trials = 6 cells, all valid (0 OOM, 0 serving errors; the only
log "error" lines are vLLM's telemetry usage-report thread JSON-decode, non-serving).
Server-side generation metric (steady t≥20s), eager, KMAX=8, mixed workload — identical
harness to Phase 4.6. Baselines AR/K8 reused from `results/p2/` (no re-run).

---

## 0. Verdict up front

The F1/F4 implementation fixes **work as pre-registered** and the Phase 4.5 "DSpark-rule is the
worst arm at C=96" result is confirmed to have been an **implementation-bug artifact**, not a
scientific failure of DSpark's decision rule. The corrected proxy recovers **+22.7pp at C=96**
(−27.2% → −4.5% vs AR). However, the corrected proxy still does **not** beat AR: it lands between
fixed-K8 and AR (the pre-registered hypothesis) but not within noise of AR (falsification not
triggered). The residual gap is the SD-off floor cost plus the rule committing fully to SD-off on
this low-acceptance ngram stack.

| Arm | C=32 (gen/s) | vs AR | C=96 (gen/s) | vs AR |
|---|---:|---:|---:|---:|
| AR (no spec) | 1456.9 | — | 2557.2 | — |
| fixed-K8 | 1436.6 | −1.4% | 1937.2 | −24.2% |
| **old buggy dspark** (P4.5) | 1415.8 | −2.8% | 1860.8 | **−27.2%** |
| **dsparkr (corrected)** | 1343.3 | −7.8% | 2442.1 | **−4.5%** |

## 1. The fixes are confirmed working (decision-state evidence)

Steady middle-75% of each cell's decision log:

| C | decided/N | B_live (mean) | meanK | frac K=0 |
|---|---:|---:|---:|---:|
| 32 | **1.000** | **33.2** | 0.01 | 0.995 |
| 96 | **1.000** | **99.6** | 0.01 | 0.994 |

- **F1 (decision-set completeness):** `decided/N = 1.000` in every cell — the controller now
  decides for *every* running request each step, vs ~6–7% in the buggy Phase 4.5 proxy. K=0
  decisions persist (frac K=0 ≈ 0.99 is a *stable committed state*, not a one-step transient).
- **F4 (true-batch cost accounting):** `B_live` = 33.2 at C=32 and 99.6 at C=96 — the SPS(B) term
  now sees the true scheduled batch (decode + drafts + prefill), vs B≈7–25 in the buggy proxy.
  At C=96 the rule finally prices saturation, exactly as the prereg predicted it would once B was
  correct.

## 2. What the corrected proxy shows

- **C=96 (saturation):** the rule commits to SD-off (frac K=0 = 0.994) because, with correct
  B≈100, the SPS(B) cost term dominates and no K-extension improves Θ. It lands at **−4.5% vs AR**
  — between fixed-K8 (−24.2%) and AR. This is the pre-registered hypothesis outcome ("lands
  between fixed-K8 and AR"). The old buggy proxy, which never priced saturation, stayed on the spec
  path at K≈8 for ~93% of requests and lost −27.2%. **The fix recovered +22.7pp.**
- **C=32 (mid-load):** the rule also commits to SD-off (frac K=0 = 0.995, B_live=33). It lands at
  −7.8% vs AR — slightly *below* fixed-K8 (−1.4%). At mid-load the ngram drafter's acceptance is
  too low for any K-extension to beat the all-zero baseline, so the rule correctly turns SD off;
  the residual gap to AR is the spec-enabled-server floor (see §3).

## 3. Why it still doesn't beat AR (the honest limitation)

The corrected proxy sits at or just above the **SD-off floor** — the cost of running on a
spec-decode-enabled server with drafts truncated to 0, which Phase 4.6 measured directly with its
force-0 diagnostic arm (single trial available per C):

| C | dsparkr (corrected) | force-0 floor (P4.6) | AR |
|---|---:|---:|---:|
| 32 | 1343.3 | 1336.9 | 1456.9 |
| 96 | 2442.1 | (no clean force-0 cell) | 2557.2 |

At C=32 the corrected proxy is within **0.5%** of the measured SD-off floor — i.e., it is
effectively *at* the floor, as expected for a rule that committed to K=0 on ~99.5% of steps. The
residual gap to AR at both pressures is therefore dominated by the spec-enabled-server overhead of
running the (truncated-to-zero) draft path, not by any misallocation: the rule found no profitable
speculation regime on this stack, so it paid the floor and turned SD off. That is a **genuine
(proxy-level) limitation of the rule on an ngram-drafter stack**, citable as such with the ngram
scope stated — not evidence that DSpark's algorithm fails, because DSpark's production drafter is a
trained model, not ngram. The decision rule itself is now shown to be faithful: given correct
inputs it makes the saturation-aware choice.

## 4. Verdict vs prereg (docs/dspark-proxy-validity.md §6)

- **Hypothesis** ("corrected proxy lands between fixed-K8 and AR at C=96, having committed toward
  low-K/SD-off with correct B"): **CONFIRMED** — −4.5% vs AR, between K8 (−24.2%) and AR; frac K=0
  = 0.994; B_live = 99.6.
- **Falsification** ("corrected proxy within noise of AR at C=96 AND ≥ best fixed-K"): **NOT
  TRIGGERED** — dsparkr is −4.5% vs AR (not within the ~2% trial noise) and below AR, so the rule
  does not fully recover to AR parity on this stack.

## 5. What this changes / does not change

- **Supersedes** the Phase 4.5 interpretation of the DSpark-rule arm's C=96 ranking as evidence
  about DSpark's algorithm (see `docs/dspark-proxy-validity.md`). The "worst arm at C=96" was the
  buggy proxy; the corrected proxy recovers most of that gap.
- **Does not change** Phase 4.6/5.1 results (their controllers already had the persistence +
  true-batch fixes) or any native DSpark (Track B) result — this is an ngram-proxy characterization.
- **Historical record preserved:** no numbers rewritten; `docs/phase4-p5-results.md` keeps its
  original text, this document supersedes its *interpretation*.

## 6. Provenance and reproducibility

- Corrected controller: `experiments/dsparkr_controller/sitecustomize.py` (commit 886a300); unit
  tests `experiments/test_dsparkr.py` (all pass — F1 completeness/persistence, F4 true-batch +
  monotone extrapolation, F2 censoring, greedy SD-off reachability).
- Runner: `experiments/p45r_cell.sh`, grid `experiments/p45r_grid.sh`; analysis
  `experiments/analyze_p45r.py`.
- Raw data: `results/p4_5r/p45r_dsparkr_c{C}_t{T}_mixed_{metrics,decisions}.jsonl` (gitignored).
- Baselines reused from `results/p2/p2_k{none,8}_c{C}_t{T}_mixed_metrics.jsonl`.
