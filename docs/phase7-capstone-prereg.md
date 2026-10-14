# Phase 7 (capstone) — Preregistration: request-aware SD scheduler from live serving state (load × traffic-class)

- **Date:** 2026-10-14 (KST)
- **Status:** SPEC / PREREGISTRATION. Design + success/falsification criteria fixed. Constants marked **[CAL]** are
  filled from a calibration/smoke step BEFORE any controller trial runs (append-only block). No controller trial may
  start until that block exists and the two open signal-design questions (§5) are resolved by smoke test.
- **This is the capstone** of the project: it turns the *validated* scheduler signal into a *working in-loop
  controller* and shows realized end-to-end serving gain — the original mission ("a lightweight decision model that
  sets SD from live serving state") made concrete.

## 0. What this builds on (the evidence chain)

| result | finding it licenses |
|---|---|
| Phase 4.6 (FALSIFIED as specified) | Per-request greedy K allocation fails at saturation (negative externality); the spec path has a fixed overhead only a scheduler-level enable/disable removes → control must be **workload-regime / batch level**, with a real SD-off state. |
| H-CLASS-1 / 1C (#28) | Speculation payoff is a function of **request class** (draftability), not cache alone; the A/B gap is content/population-driven. |
| Track A (#29, NOT SUPPORTED) | Per-request acceptance is **not** content-predictable from admission features → do NOT build a per-request predictor. The signal is **population/regime level**. |
| scheduler-signal-validation (`0b11bfc`) | The 2-axis rule "SD-on iff low-load OR high-reuse-class" matches the oracle on all 9 load×class cells (100% gain, 0 regret); each axis alone is strictly worse (load-only misses Bc48/64; class-only misses Ac16/Cc16). |
| H-CLASS-1D (`83b2eb3`) | The load axis holds **out of sample**: boundary in [c16, c32]; A load curve monotone-decreasing. |
| Track B (mechanism built) | Per-step SD on/off is **feasible** in the isolated v0.25.0 worktree: `sd_enabled` V2 patch, causal-verification logging (OFF step ⇒ 0 drafts), probe-based re-entry — 20 tests passing. Its *reactive TPOT-trend* trigger is what we replace. |

## 1. Hypothesis (H-CAP)

A controller that turns native DSpark speculation **on/off per scheduling step** as a function of **two measured
serving-state signals** — (i) a load-regime signal and (ii) a traffic-class (draftability) signal, combined as
`ON iff load_LOW OR class_HIGH-REUSE` — recovers most of fixed-K DSpark's loss at/above the crossover **and**
preserves its low-load benefit **and** beats each single-axis variant on a workload that exercises both axes, versus
AR and native fixed-K=7 DSpark on the identical runtime.

This is the end-to-end realization of the signal validated in §0: not just "the rule matches the payoff surface"
(same-data), but "a live controller acting on measured state achieves the predicted decisions and their gains."

## 2. Runtime and arms

- **Runtime:** Track B's isolated worktree (vLLM v0.25.0+cu126, branch `track-b-live-sd`), Qwen3-4B +
  `dspark_qwen3_4b_block7`, GPU exclusive, same server flags as Track B/Phase 6. Reuse the existing `sd_enabled`
  patch + causal-verification logging; **replace only the trigger** (Track B's TPOT-trend rule → the §3 2-axis rule).
- **Arms:**
  1. **AR** — no spec config (baseline).
  2. **DSpark-K7** — native fixed K=7 (the deployment default this controller improves on).
  3. **CTRL-ON** — patched, policy constant ON → quantifies patch/decision overhead (validity control).
  4. **CTRL-LOADONLY** — patched, `ON iff load_LOW` (class axis removed) → isolates the load axis end-to-end.
  5. **CTRL-CLASSONLY** — patched, `ON iff class_HIGH-REUSE` (load axis removed) → isolates the class axis end-to-end.
  6. **CTRL-LIVE** — patched, the full 2-axis rule (the hypothesis arm).

  Arms 4–5 are the capstone-specific addition: they let us test "both axes necessary" *in the live system*, not just
  on the static payoff surface. Expect CTRL-LOADONLY to lose at high-load high-reuse traffic and CTRL-CLASSONLY to
  lose at low-load diverse traffic — mirroring the scheduler-signal-validation regret table.

## 3. The controller (2-axis rule, measured inputs only)

**Decision per scheduling step:** `sd_enabled ← load_LOW OR class_HIGHREUSE`.

- **Load-regime signal (`load_LOW`):** a function of **measured serving state**, NOT a concurrency integer
  (directive §4 prohibition stands: no "if concurrency == N"). Candidate primary: measured **KV-cache usage fraction**
  and/or **running occupancy = len(running)/max_num_seqs** (both continuous, both in-process, both monotone in true
  resident load, and — critically — NOT the `schedule()`-cadence signal that Track B Defect 2 showed decouples from
  GPU step time under async scheduling). `load_LOW` iff below calibrated thresholds **[CAL]**. Thresholds are on
  *measured continuous state*, which the directive permits; only hardcoded concurrency cutoffs are prohibited.
- **Traffic-class signal (`class_HIGHREUSE`):** rolling-window (W) **draft-acceptance ratio** = accepted/draft tokens
  over recent steps — the direct population-level draftability measure H-CLASS-1 established (B ~0.61 vs A/C ~0.33).
  It is **load-invariant** (Phase 6 / Track B R1), so it cleanly carries the *class* axis without conflating with load.
  `class_HIGHREUSE` iff rolling ratio > θ_acc **[CAL]**. Because acceptance is only observable while SD is ON, the
  estimate is held with slow exponential decay across brief OFF periods and refreshed on probe re-entry (reuses Track
  B's probe mechanism). A prefix-cache-hit-rate term may be added as a secondary class proxy (available even when OFF)
  **only if** the smoke test shows acceptance-decay alone flaps; decided in §5, not assumed.
- **Hysteresis / anti-flap:** reuse Track B's cooldown + probe-based re-entry structure (T_off, probe steps, band).
  Cold start = ON (preserves low-load behavior by default).
- **What it is NOT:** no learned model, no per-request K, no variable-K (blocked — `docs/trackb-feasibility.md` B1–B3),
  no hardcoded concurrency threshold anywhere.

**Why these two signals specifically:** they are *orthogonal by construction* — acceptance ratio is load-invariant
(so it carries class, not load) and KV/occupancy is acceptance-independent (so it carries load, not class). This is
exactly the orthogonality that made the 2-axis rule beat single-axis on the payoff surface; implementing it with
orthogonal measured signals is what should reproduce that result live.

## 4. Workload — must exercise BOTH axes (the capstone-specific design)

A single homogeneous workload cannot test a 2-axis rule (Track B's C∈{8,32,64} fixed-mix only moves load). The
controller is validated on **mixed traffic** that varies both axes within one stream:

- **Class blocks:** alternate segments of (a) **diverse** prompts (A/C-like, from the ShareGPT-V3 distinct bank) and
  (b) **high-reuse** prompts (B-like, a small set repeated), so `class_HIGHREUSE` should be high in (b) and low in (a).
- **Load sweep:** run the mixed stream at three applied loads (low / mid / high, mapped to the frozen λ=C/28 points
  c16 / c48 / c96 as in H-CLASS-1, open-loop Poisson for clean load application) so `load_LOW` is true only at low.
- **Expected 2-axis behavior (the prediction being tested):** CTRL-LIVE turns SD **ON** in high-reuse blocks even at
  high load (class axis rescues it — the Bc48/Bc96 cells), and **OFF** in diverse blocks at mid/high load (load axis
  — the Ac48/Ac96 cells), while staying ON at low load regardless of class. CTRL-LOADONLY fails the first;
  CTRL-CLASSONLY fails the second.

## 5. Open signal-design questions — RESOLVED offline 2026-10-14 (see §7 [CAL])

Both design questions turned out to be resolvable from the **existing** H-CLASS-1 A/B/C data (analysis.json + per-cell
metrics.jsonl + obs traces) — more rigorous than a single new smoke run. Resolved and frozen in §7:

1. **Load signal → `kv_cache_usage_perc`** (primary; occupancy secondary). Separates [c16 | c48, c96] cleanly
   (0.30 vs 0.93/0.98); waiting fails (c48=0). Hysteresis band θ_low_on=0.55 / θ_low_off=0.75.
2. **Class signal → rolling draft-acceptance ratio, θ_acc=0.45** (mid-gap: A/C max 0.335 ≪ 0.45 ≪ B min 0.599).
   Load-invariant by construction.

**Remaining open item (needs the live controller, not a separate smoke):** whether acceptance-with-decay stays bimodal in
a *single mixed stream* alternating diverse + high-reuse blocks, and whether it flaps across brief OFF periods. Confirmed
by the first CTRL-LIVE smoke trial (logs rolling ratio + transition count) before the full run set; if it flaps, add
prefix-cache-hit-rate as a secondary class proxy then.

(If either signal turns out to be non-discriminating on this workload — as Track B's acceptance-ratio trigger was for
its *reactive* use — that is a recorded finding bounding the claim, not a silent retune.)

## 6. Success / falsification criteria (fixed before runs)

Let `X̄_C ± CI` be mean output-throughput and 95% CI of arm X in cell C; δ = baseline run-to-run uncertainty.
≥3 valid repetitions per cell, fresh server per trial, counterbalanced order.

**SUCCESS requires ALL:**
- **S1 (crossover recovery):** at high-load diverse blocks, CTRL-LIVE ≥ AR within CI (lower bound > −2%) AND > DSpark-K7
  in ≥2/3 paired rounds with mean Δ>0.
- **S2 (low-load preservation):** at low load, CTRL-LIVE ≥ DSpark-K7 − δ (no meaningful loss of the benefit).
- **S3 (both axes necessary — capstone claim):** on the mixed workload, CTRL-LIVE beats **both** CTRL-LOADONLY and
  CTRL-CLASSONLY in aggregate realized gain (the live analog of the payoff-surface regret table). This is the
  distinguishing result: it shows the *combination* matters end-to-end, not just statically.
- **S4 (overhead):** CTRL-ON vs DSpark-K7 ≤ 1% relative at every load (decision/patch overhead negligible).
- **S5 (causal correctness):** Track B's causal-verification assertion passes in all valid trials (OFF step ⇒ 0 new
  drafts; ON decode step ⇒ K=7 drafts).

**FALSIFIED:** S1 or S3 fails beyond uncertainty after the single preregistered run set (no retune loop; one revision
permitted only on a harness bug, charter §13). **INCONCLUSIVE:** if baseline δ is too large to order arms — report
per-trial values and the uncertainty needed.

**Prohibited claims (even on success):** outperforming native adaptive verification (never run on this hardware);
generalization beyond Qwen3-4B / dspark_qwen3_4b_block7 / SM86; novelty of load-aware K per se (suspended claim list).

## 7. Calibration block [CAL] — filled 2026-10-14 from EXISTING data (stronger than a fresh smoke)

**Provenance note:** both signal choices are resolvable offline from the already-collected H-CLASS-1 A/B/C cells
(`results/hclass1/...`, analysis.json + per-cell metrics.jsonl + obs traces), which is more rigorous than a single new
smoke run. The one sub-question that genuinely requires the live controller (class-signal stability across OFF periods in
a *mixed* stream) is left open and will be confirmed by the first CTRL-LIVE smoke trial before the full run set.

### [CAL-1] Load signal (Q1) — RESOLVED offline
Steady-window medians per condition-A cell (metrics `kv_cache_usage_perc`, obs `running`/`waiting`; max_num_seqs=64):

| cell | kv_cache_usage | occupancy = running/64 | waiting |
|---|---:|---:|---:|
| c16 | 0.303 | 0.250 | 0 |
| c48 | 0.929 | 0.625 | 0 |
| c96 | 0.983 | 0.672 | 35 |

The decision boundary that matters is **[c16 \| c48, c96]** (both mid and high are "not low"). `kv_cache_usage_perc`
(0.30 vs 0.93/0.98) and occupancy (0.25 vs 0.62/0.67) both separate it cleanly; **waiting FAILS** (c48 also = 0). Choose
**primary = `kv_cache_usage_perc`**, secondary = occupancy, with a hysteresis band:

- **load_LOW iff kv_cache_usage < θ_low_on = 0.55** and **load_NOT_LOW (OFF trigger) iff kv_cache_usage > θ_low_off = 0.75**
  (hysteresis band 0.55–0.75; sits mid-gap between c16's 0.30 and c48/c96's 0.93/0.98, ~2× margin each side).

Both are in-process continuous state (not the prohibited `schedule()`-cadence signal, Track B Defect 2), and neither is a
concurrency integer. kv-usage preferred over occupancy because it needs no `max_num_seqs` constant and tracks KV pressure
directly; occupancy retained as a logged cross-check.

### [CAL-2] Class signal (Q2) — RESOLVED by live smoke 2026-10-14 → **NEGATIVE for the reactive class axis**
Per-condition DSpark acceptance from `analysis.json` (steady-window metric deltas, **always-on** DSpark):

| cond | c16 | c48 | c96 | mean |
|---|---:|---:|---:|---:|
| A diverse | 0.335 | 0.333 | 0.329 | 0.332 |
| B high-reuse | 0.601 | 0.599 | 0.617 | 0.606 |
| C shared-doc | 0.302 | 0.303 | 0.318 | 0.308 |

- **θ_acc = 0.45** (mid-gap; A/C max 0.335 ≪ 0.45 ≪ B min 0.599, ~2× margin each side).
- **class_HIGHREUSE iff rolling draft-acceptance ratio > θ_acc.** Load-invariant by construction (acceptance is flat
  across c16→c96 within each condition), so it carries *class*, not load — the orthogonality that makes the 2-axis rule
  beat single-axis.

#### [CAL-2b] Live mixed-stream smoke result (2026-10-14, `experiments/phase7_smoke.sh`, DSpark-K7 live controller)
The open sub-question — *does the class signal stay usable in a single mixed stream under reactive control?* — is now
answered **no**, with a decisive cross-check:

| scenario | kv p50 | SD-on % | acc_ratio global max | steps > θ_acc(0.45) | on/off transitions | causal (OFF⇒0 drafts) |
|---|---:|---:|---:|---:|---:|---|
| HI-MIX (c96, mixed) | 0.987 | **1%** | **0.295** | **0 / 379** | 13 / 3100 | ✓ pass |
| LO-MIX (c10, mixed) | 0.231 | **100%** | 0.555 | 166 / 1196 | 0 / 1200 | ✓ pass |

**Finding:** at high load the reactive class signal **never crosses θ_acc in any stretch** (global max 0.295), even though
H-CLASS-1 measured steady-state B@c96 acceptance at **0.617** — same traffic, same load, always-on DSpark. The controller
therefore stays OFF almost continuously (1% on) and the class axis is **non-functional exactly where it is supposed to
matter**. The gap (0.295 live vs 0.617 steady) is a **measurement artifact of reactive control**, not a fundamental
"draft model fails at c96" effect:

- **Root cause — OFF periods cool the prefix cache.** When SD is off, high-reuse requests still run (plain AR), but the
  controller's class signal can only sample *draft acceptances*, which require SD to be on. The 3-step re-entry probes are
  too short and land on a **cold** prefix cache (acceptance ~0.1–0.3), so they never accumulate enough high-acceptance
  samples in the 2 s window to commit ON — a self-reinforcing OFF lock at high load.
- **Wiring verified correct:** LO-MIX acceptance (p50 0.357, max 0.555) matches the H-CLASS-1 c16 range (~0.33 diverse /
  ~0.60 reuse), so the draft/accepted accumulation and rolling window are sound; the failure is specific to *high-load
  reactive sampling*, not a harness bug.

**Consequence for the capstone (honest, prereg §6):** **S3 ("both axes necessary") is falsified in this reactive form.**
Because the class axis never fires at high load, CTRL-LIVE ≡ CTRL-LOADONLY on the mixed workload — the 2-axis rule does
not beat either single axis end-to-end. The *load* axis works perfectly (LO-MIX 100% ON; HI-MIX correctly OFF except brief
probes; no flapping: 13 transitions / 3100 steps). S5 causal correctness passes in both scenarios.

**Implication (design lesson, not a retune):** a class signal derived from *SD's own acceptances* cannot be measured while
SD is off — the very state it must predict. The class axis needs a proxy that is **available with SD off**: prefix-cache
hit rate (load-invariant, high for reusable traffic) is the natural candidate. This reframes the capstone claim from
"2-axis beats single-axis" to "class must be inferred from an SD-independent state."

#### [CAL-2c] Class axis REBUILT on prefix-cache hit-rate + CONFIRMED (2026-10-14, worktree `2ab37e7`)
The class axis was rebuilt exactly as [CAL-2b]'s implication prescribed: **rolling mean admission prefix-hit-rate**
(measured from KV blocks at request admission — orthogonal to speculative decoding, so available while SD is off),
θ_hit = 0.45. This resolves the chicken-and-egg: both axes are now directly observable in the state we decide from, so
the Track B cooldown+probe re-entry mechanism was dropped (a simple OFF→ON cooldown flap guard suffices; the load axis
bypasses it to rescue immediately on a genuine load drop).

**Premise check** (`experiments/phase7_premise_check.sh`, c96 mixed, SD off — isolates the cache question): do reusable B
prefixes survive eviction pressure from 96 diverse A requests? **Yes.** Per-request admission hit-rate is cleanly bimodal
(623 at ~0.0 diverse / 659 at ~1.0 reusable); the 2 s windowed mean tracks the traffic mix over time (rises to ~0.7–0.9 in
reusable-heavy periods, dips to ~0.0–0.3 in diverse-heavy). Reusable prefixes persist as long as they keep being re-requested
—which is what "high-reuse" means.

**Confirmation smoke** (`experiments/phase7_smoke.sh`, DSpark-K7 live controller, same scenarios as [CAL-2b]):

| scenario | kv p50 | SD-on % (was) | hit_rate max | class_hi fires | on/off transitions | causal (OFF⇒0 drafts) |
|---|---:|---:|---:|---|---:|---|
| HI-MIX (c96, mixed) | 0.987 | **24%** (was 1%) | 0.974 | ✓ up to 100%/window | 29 / 1800 | ✓ pass |
| LO-MIX (c10, mixed) | 0.226 | **100%** (was 100%) | 0.984 | n/a (load axis) | 0 / 1200 | ✓ pass |

**Result: the class axis now works at high load — the exact failure [CAL-2b] had.** At c96, SD-on rose from 1% to 24%,
with the ON steps concentrated in windows where reusable prefixes are actually available (hit-rate peaks 0.6–0.97); the
class axis drives SD on for reusable traffic at high load where the load-only arm stays OFF. The signal is honest: under
c96 contention even B blocks sometimes get evicted (low hit-rate → correctly OFF), so it tracks the *real* reuse
opportunity, not just a traffic-class label. S5 causal correctness passes in both scenarios.

**Consequence:** S3 ("both axes necessary") is now **testable** — CTRL-LIVE turns SD on at high load for reusable traffic
(24%) where CTRL-LOADONLY stays off. The full 6-arm comparison run set proceeds to measure the end-to-end realized gain
and confirm CTRL-LIVE beats both single-axis ablations in aggregate.

**Frozen before any controller trial:** θ_low_on=0.55, θ_low_off=0.75, **θ_hit=0.45**, load signal = kv_cache_usage_perc
(secondary occupancy), class signal = rolling mean admission prefix-hit-rate (SD-independent). OFF→ON cooldown 1 s (load
axis bypasses it); ON→OFF immediate. **[CAL-2c] supersedes [CAL-2b]'s "not launched" status: the class axis is rebuilt and
confirmed functional; the full 6-arm run set now proceeds.**

## 8. Deliverable & scope

- `docs/phase7-capstone-results.md`: the realized-gain table across load×class, per-axis ablation (S3), overhead
  (S4), causal verification (S5), and the verdict against §6.
- Controller code committed to the isolated worktree branch with its own commit; patch diff reproducible.
- Closes the project: converts the validated signal into a working scheduler and quantifies its end-to-end value.

**Honest scope:** validates the *on/off* 2-axis rule at fixed K=7 on one model/draft pair/hardware. It does not make
K itself adaptive (blocked), does not claim generalization, and its class signal is population-level by design (Track A
showed per-request is the wrong granularity).
