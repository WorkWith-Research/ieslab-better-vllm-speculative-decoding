# M-BATCH-1 — results: B-class payoff under live MIXED batches

**Date:** 2026-10-15 · **Prereg:** `docs/mbatch1-prereg.md` (design + decision rule frozen before data) ·
**Harness:** `experiments/mbatch1_run.sh` + `experiments/mbatch1_client.py` · **Analysis:** `results/mbatch1/report.md`

## 0. Verdict

> **H1 CONFIRMED at both c48 and c96.** Class-B (high-reuse) DSpark/AR TPOT payoff, measured *inside* a live
> mixed stream, is **negative** at both high loads — the sign flip predicted by the Phase 7 mechanism argument
> is real and large. The class axis has **no headroom at fixed K=7 on this hardware under live serving**. Per the
> preregistered decision rule (§5), the capstone stays closed as falsified and the successor scope is
> batch-aware speculation (per-request K / selective drafts) — changing *what* SD applies to, not *when*.

## 1. Primary result — per-class payoff under mixing (pid-matched, within-rep pairing pooled across 3 reps)

Payoff = mean(1 − DSparkTPOT/ARTPOT); >0 ⇒ DSpark faster. Same convention as
`docs/scheduler-signal-validation.md`. n = paired B completions; bootstrap 2000.

| load | B (pooled) | CI95 | homogeneous anchor (H-CLASS-1) | Δ (mixed − homogeneous) |
|---|---:|---|---:|---:|
| c48 | **−0.0835** (n=226) | [−0.1238, −0.0442] | +0.1209 (n=1200) | **−0.204 → sign flip** |
| c96 | **−0.2896** (n=219) | [−0.3876, −0.2085] | +0.0216 (n=1530) | **−0.311 → sign flip** |

Both CIs are entirely ≤ 0 — not marginal, decisively negative. The homogeneous Bc96 anchor (+0.0216, the
"marginal boundary cell" flagged in the static-surface doc) is confirmed to have been a **homogeneous-batch
artifact**: under live mixing the same traffic class loses ~3× more than its homogeneous margin was worth.

Per B-class (ISL × maxtok): the long-output classes carry the loss; LL (ISL 2048, cap 1024) is the only
class still positive at c48 (+0.321 [0.283, 0.356]) and ~zero at c96 (+0.083 [−0.008, +0.173]). SS/SL (short
output) are strongly negative everywhere (−0.19 to −0.46). Consistent with SD benefit scaling with realized OSL
(amortization of draft overhead per request) while the cost is paid per step regardless.

Control: A-class payoff under mixing is also negative, as expected for diverse traffic (c48 −0.236 [−0.277,
−0.195], c96 −0.319 [−0.390, −0.254]).

## 2. Mechanism probe — payoff vs queue depth at arrival

B payoff by in-flight-at-arrival (client-side co-batch proxy):

| in-flight at arrival | c48 n / mean | c96 n / mean |
|---|---:|---:|
| [5, 15) | 6 / +0.351 | 3 / −0.111 |
| [15, 40) | 54 / +0.003 | 47 / −0.093 |
| [40, ∞) | 166 / **−0.127** | 169 / **−0.348** |

Monotone degradation with queue depth at both loads; at c96 even shallow-queue B requests are negative (the
whole regime is saturated). This is the batch-mix dilution + draft-overhead-at-saturation mechanism measured
directly: a B request's payoff depends on *how many diverse A requests it decodes alongside*, not just its own
prefix reuse.

Server context (tail means): DSpark arm runs at higher KV usage (83–92% vs 68–88% AR) — drafts hold KV — and
lower steady generation throughput (c48: ~987 vs ~1163 tok/s; c96: ~957 vs ~1289 tok/s), consistent with S3's
throughput result. Prefix hit rates are comparable-to-higher on the DSpark arm at c96 (31–34% vs 20–25%), so
the loss is **not** a cache-miss artifact: B prefixes *are* being served from cache, and SD still loses.

## 3. What this establishes

1. **The static payoff surface does not transfer to live serving — now measured, not inferred.** The Phase 7
   results doc argued the flip from batch-mix dilution + draft overhead; M-BATCH-1 measures it: B's payoff goes
   +0.12 → −0.08 (c48) and +0.02 → −0.29 (c96) when B decodes in mixed batches instead of homogeneous ones,
   with a monotone queue-depth gradient.
2. **The class axis has no headroom at fixed K=7 on this hardware under live serving.** No admission-time
   estimator of the *homogeneous* payoff could have saved it: the live payoff itself is negative. (This rules
   out the H2 branch — "S3's failure was timing/estimation" — for this platform/workload.)
3. **The on/off question is the wrong axis at high load.** Even a perfect oracle that knew every request's
   class and served SD *only* for B requests in mixed batches would not recover the homogeneous-B margin,
   because the decision unit (the step/batch) is shared with A traffic. The lever that remains is making
   speculation **batch-aware**: per-request K, selective drafts (draft only the high-reuse subset), or
   batch-level admission — i.e., changing *what* SD applies to.

## 4. Provenance & reproducibility

- Raw data: `results/mbatch1/` (12 cell dirs: client.jsonl, server.log, prompt_manifest.jsonl) — gitignored.
  All 12 cells valid after post-run revalidation with the fixed classifier (prereg §7 documents the bring-up
  classifier bug; every cell had ≥83 B completions).
- Power: 226 (c48) / 219 (c96) pid-matched B pairs pooled across reps — far above the prereg §6 floor of 60.
- Homogeneous anchors recomputed from `results/hclass1/` with within-trial pid pairing (matches the static-
  surface doc's values: +0.121/+0.022 vs here +0.1209/+0.0216).
- Load balance: both arms at KV ≥ 83% and Running ≈ max-num-seqs at c48/c96 (server.log); no arm starved.

## 5. Consequence for the project

Combined with Phase 7 (S1/S3 falsified; S2/S4/S5 passed) and M-BATCH-1 (mechanism confirmed, H1), the project's
research question is fully answered for its fixed-K scope: **a request-aware SD on/off controller driven by
population payoff surfaces cannot beat single-axis policies end-to-end under live mixed serving at K=7 on this
platform, because the population surface itself is wrong for mixed batches.** The two positive byproducts stand:
(a) a causal, overhead-free live SD controller validated end-to-end; (b) the methodological constraint that
homogeneous-cell payoff surfaces do not bound per-step decisions under mixed batches — now with a measured
magnitude and gradient.

Successor scope (per prereg decision rule): **batch-aware speculation** — per-request K or selective drafts, so
the speculation unit matches the request's own economics instead of the batch's average. That is a new phase
with its own prereg; it changes what SD applies to, which no on/off policy can do.
