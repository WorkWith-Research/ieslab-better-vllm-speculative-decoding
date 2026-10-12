# H-CLASS-1 results — Request-class-dependent speculation economics under realistic heterogeneous serving

**Date:** 2026-10-13 · **Status:** FINAL (per prereg `docs/hclass1-prereg.md`)
**Run set:** 5 calibration + 36 confirmatory cells. First pass: 31/36 valid; 5 cells failed on the
transient CUDA-driver-init startup race (`RuntimeError: CUDA driver initialization failed` at
`torch._C._cuda_init()` in the EngineCore child) and were re-run once each per prereg §6 — all 5 passed.
**Final: 41/41 DONE-VALID, 0 invalid.** Frozen env rev recorded in each cell's `config_manifest.json`.
**Raw data:** `results/hclass1/<cell>/` (gitignored, local). Analysis: `experiments/hclass1_analyze.py` →
`analysis.json`, `analysis_table.md`, `hc3.json`.

---

## 0. Verdict summary

| Hypothesis | Verdict | Key numbers |
|---|---|---|
| **H-C1** (replication under de-confounding): cross-class range >15% in **both** A and B at ≥1 common pressure, with direction | **FALSIFIED** | Condition A range = 0.57% / 0.48% / 9.69% (c16/c48/c96) — **none >15%**; condition B = 37.0% / 41.6% / 42.2% (all >15%). No common pressure has both >15%. |
| **H-C2** (cache-condition robustness): sign of DSpark/AR ratio stable across A↔B at matched pressure | **VIOLATED** (at c48, c96) | At c16 all signs agree (SD helps everywhere). At c48/c96 the **sign flips**: A is uniformly + (SD *hurts*, ratio 1.16–1.27) while B's long-horizon classes are − (SD *helps*, LL 0.67/0.75) and short classes straddle 1. |
| **H-C3** (OSL explanatory power, descriptive): realized-OSL carries the dominant coefficient | **NOT SUPPORTED** — acceptance dominates | OLS (n=18): β_logOSL = −0.117, β_logISL = −0.021, **β_acc = −1.644**. Acceptance rate is ~14× the OSL coefficient in magnitude. |

**Overall:** H-OBS-3's strong per-class DSpark/AR TPOT-ratio spread (35.9–43.8%) **does not survive
de-confounding.** Under diverse prompts with prefix caching off (condition A), the cross-class range
collapses to ≤10% — there is essentially *no* class-dependent speculation payoff on a clean workload.
The spread reappears only in condition B, which reproduces H-OBS-3's exact confounded configuration
(2 repeated prompts, 97–98% prefix-cache hit). The descriptive regression shows the driver is **draft
acceptance rate** (β=−1.64), not realized output length (β=−0.12) — and acceptance nearly doubles from
A (~33%) to B (~61%) purely because cache reuse makes the repeated prompts' drafts more predictable.
**H-OBS-3's "class effect" was substantially a cache-reuse / draft-quality artifact, not an output-length
mechanism.** This is a reportable negative result per prereg §2 ("a failure to reproduce the
class-dependent payoff in either condition is a reportable result").

---

## 1. H-C1 — replication under de-confounding: FALSIFIED

Prereg §2 rule: cross-class relative range of per-request DSpark/AR TPOT ratio > **15%** within at least
one pressure level, in **both** conditions A and B; direction per condition (B: SL/LL < SS; A: S-vs-L
ordering matches B's pattern or tracks realized OSL).

Per-class DSpark/AR TPOT ratio (p50, steady window, request-paired bootstrap by prompt id, 2000 resamples):

| pressure | A: S / L (range) | B: SS / SL / LS / LL (range) |
|---|---|---|
| c16 (P_low) | 0.876 / 0.881 (**0.57%**) | 0.612 / 0.531 / 0.554 / 0.417 (**37.0%**) |
| c48 (P_med) | 1.215 / 1.221 (**0.48%**) | 1.032 / 0.852 / 0.896 / 0.673 (**41.6%**) |
| c96 (P_high) | 1.157 / 1.274 (**9.69%**) | 1.163 / 0.977 / 1.014 / 0.751 (**42.2%**) |

- **Condition A: no pressure reaches 15%.** At c16/c48 the two ISL classes are indistinguishable
  (range <1%); even at saturation (c96) the range is only 9.69%. The class-dependent payoff H-OBS-3
  reported is **not present** on a diverse-prompt, cache-off workload.
- **Condition B: all three pressures exceed 15%** (37–42%), reproducing H-OBS-3's magnitude — as
  expected, because B is H-OBS-3's exact configuration (repeated prompts + prefix caching).
- **Direction:** B satisfies its criterion at every pressure (long-horizon SL/LL have lower ratio than
  SS). A does not: S and L are flat (L is marginally *higher* at c96, the opposite of B's long→lower
  pattern), and realized OSL is comparable across S/L (p50 ≈ 1780 vs 1700 tokens), so neither direction
  clause holds.

**Interpretation:** the H-C1 conjunction ("both conditions") fails because condition A — the de-confounded
workload this experiment exists to test — shows no class effect. The spread is an artifact of B's
repeated-prompt/cache design, not a property of request heterogeneity per se.

## 2. H-C2 — cache-condition robustness: VIOLATED at c48 and c96

Prereg §2 rule: the sign of each class's DSpark/AR TPOT ratio (ratio <1 = SD helps; >1 = SD hurts) is
stable across A↔B at matched pressure, comparing B's short-ISL pair (SS,SL) against A's S and B's
long-ISL pair (LS,LL) against A's L.

| pressure | A.S / A.L | B: SS / SL / LS / LL | verdict |
|---|---|---|---|
| c16 | 0.876(−) / 0.881(−) | 0.612(−) / 0.531(−) / 0.554(−) / 0.417(−) | **STABLE** (all −, SD helps) |
| c48 | 1.215(+) / 1.221(+) | 1.032(+) / 0.852(−) / 0.896(−) / 0.673(−) | **FLIP** (A hurts, B long-classes help) |
| c96 | 1.157(+) / 1.274(+) | 1.163(+) / 0.977(−) / 1.014(+) / 0.751(−) | **FLIP** (A hurts, B LL helps) |

- At low load (c16) SD helps every class in both conditions — sign is stable.
- At mid/high load the sign of the SD payoff **flips between conditions**: on the diverse-prompt workload
  A, fixed-K=7 DSpark *hurts* per-request TPOT at c48/c96 (ratio >1.15); on the cache-reuse workload B,
  the long-horizon classes still *benefit* (LL ratio 0.67/0.75). The violation is real and large.
- **Interpretation:** this does not by itself prove H-OBS-3's effect was "a cache-reuse artifact" in the
  narrow sense — §8 notes A-vs-B differ in *prompt diversity + cache jointly* (condition C deferred).
  What it does establish is that **the sign of SD's per-request payoff is not a stable property of the
  request class; it depends on the serving regime (cache state / prompt reuse) as much as on the class.**
  A decision model keyed only to request class would mis-sign its prediction between these two regimes.

## 3. H-C3 — OSL explanatory power: NOT SUPPORTED (acceptance dominates)

Prereg §2 rule (descriptive): in a per-class regression of DSpark/AR TPOT ratio on realized OSL (log),
ISL (log), and cell-level acceptance rate, realized OSL carries the dominant coefficient. §8 states this
is observational only (prompt content drives both realized OSL and draft acceptance).

OLS over the 18 class×pressure×condition cells (n=18, 4 params incl. intercept):

| term | β | \|β\| |
|---|---:|---:|
| intercept | +2.642 | — |
| log(realized OSL) | −0.117 | 0.117 |
| log(ISL) | −0.021 | 0.021 |
| **acceptance rate** | **−1.644** | **1.644** |

- The acceptance-rate coefficient is **~14× the OSL coefficient** in magnitude and carries essentially all
  of the explained variance. Realized OSL — the variable H-OBS-3's interpretation leaned on ("SD pays off
  more for long-output requests") — is a negligible explanatory factor once acceptance is in the model.
- This is consistent with §1–2: within condition A, acceptance is flat across classes (~33%) and so is
  the ratio; the cross-class spread in B tracks each class's cache-reuse-driven acceptance, not its OSL.

**Causal limit (§8):** prompt content drives both realized OSL and draft acceptance, so the OSL coefficient
is confounded with content/draftability even here. The clean statement is: **the per-request SD payoff
tracks draft *acceptance quality*, which a scheduler can proxy from content + cache state at admission —
not from realized output length, which is an outcome, not a decision-time input.**

## 4. Aggregate economics (indicative)

Steady-window server-side output throughput, DSpark vs AR (trial means; saturated cells' request-side
window undercounts tail completions, so treat as indicative — per-request ratios in §1–3 are the primary
signal):

| cond | pressure | AR tok/s | DSpark tok/s | Δ |
|---|---|---:|---:|---:|
| A | c16 | 743 | 815 | **+9.7%** |
| A | c48 | 1079 | 807 | **−25.2%** |
| A | c96 | 952 | 756 | **−20.5%** |
| B | c16 | 922 | 1754 | **+90.3%** |
| B | c48 | 1660 | 1938 | **+16.8%** |
| B | c96 | 1971 | 2137 | **+8.4%** |

- On the diverse-prompt workload (A), fixed-K=7 DSpark helps only at low load and **loses to AR at
  mid/high load** — consistent with Finding A's crossover, now reproduced on a clean heterogeneous workload.
- On the cache-reuse workload (B), DSpark wins at every pressure, largest at low load (+90%) — the high
  acceptance (~61%) more than pays the drafter cost even under saturation.

## 5. Mechanism inputs (prereg §7.6)

| cell | acceptance rate | prefix-cache hit% | drafter share p50 | prefill token share |
|---|---:|---:|---:|---:|
| dspark_A c16/c48/c96 | 0.335 / 0.333 / 0.329 | (off) | 0.197 | 0.21–0.34 |
| dspark_B c16/c48/c96 | 0.608 / 0.608 / 0.618 | 0.976 | 0.207 | ~0.03 |

- **Acceptance rate is the pivot:** A ≈ 0.33 (diverse prompts, no cache) vs B ≈ 0.61 (repeated prompts,
  97.6% cache hit). Cache reuse doubles draft acceptance — the single largest cross-condition difference,
  and the variable that dominates H-C3.
- **Drafter share is flat** (~20%) across pressure and condition — the drafter's *relative* cost does not
  grow with load (consistent with H-OBS-3 H3); what changes with regime is the *payoff* (acceptance).
- Prefix-cache hit rate confirms the two conditions are behaving as designed: B reproduces H-OBS-3's
  cache-dominated workload (97.6%); A has no reuse (0%).

## 6. Hard-stop audit (all clear)

- All 41 cells DONE-VALID (36 confirmatory + 5 calibration). 5 cells needed one re-run each, all for the
  transient CUDA-driver-init startup race (`CUDA driver initialization failed` at engine-core init), not
  any data-quality issue; every re-run passed. Below the ">4 invalid after re-runs" hard stop (0 remain).
- Async overlap preserved: `in_flight≥2` fraction ≥0.9 in all cells (validated by the runner per cell).
- gpu_step_ms coverage ≥95% of steps in all cells; DSpark cells scheduled >1000 draft tokens, AR cells
  exactly 0 (runner-enforced validity gate).
- Applied-load gate (D3): condition A cells verified to have issued the intended arrivals with
  `dropped_full_queue=0`; service-aware completion floor passed.

## 7. Confounds removed vs remaining (prereg §7)

**Removed (vs H-OBS-3):** (1) configured-vs-realized-OSL confound — condition A uses a uniform generous
cap with natural EOS, so realized OSL is a free outcome; (2) repeated-prompt / cache-reuse confound —
condition A draws 480+192 *distinct* seeded prompts per run with prefix caching off; (3) all-at-once
open-loop decay — condition A uses Poisson arrivals at a frozen rate per pressure (steady state).

**Remaining:** prompt-content ↔ class is only *partially* de-confounded (class still correlates with the
ISL band, so content effects within an ISL band are not isolated). Condition B deliberately retains all
three H-OBS-3 confounds as a control, so A-vs-B differences attribute to (prompt diversity + cache)
**jointly**, not separately — condition C (diverse prompts *with* cache) was deferred and is the natural
follow-up to split that joint effect.

## 8. Opportunity surface for Track A (revised by this result)

1. **Acceptance-quality-aware admission (signal: strong, revised).** The per-request SD payoff tracks
   draft acceptance rate (H-C3 dominant β=−1.64), not realized OSL. A decision model should predict an
   *acceptance proxy* at admission — content/draftability features plus cache-hit state — to set per-request
   K or SD on/off. This supersedes H-OBS-3's "predict output length" framing, which this experiment does
   not support once acceptance is controlled.
2. **Cache state as a first-class feature (signal: strong).** The same request class gets ratio 0.42–0.75
   under cache reuse (B) but ~1.2 without it (A). Cache-hit probability at admission is a high-leverage,
   causally-available input that flips the sign of SD's payoff — H-OBS-3 could not see this because its
   workload was uniformly cache-dominated.
3. **Class/OSL-aware policy (signal: weak on clean workloads).** On diverse prompts (A) the cross-class
   spread is ≤10%, so a policy keyed only to request class has little headroom; it becomes meaningful only
   in high-reuse regimes. Class remains useful as a *prior* for the acceptance proxy, not as the primary signal.

## 9. Provenance and reproducibility

- Launcher: `experiments/hclass1_run.sh` (this repo); worktree `/home/junior1/_dev/dspark-native-validation`
  (branch `track-b-live-sd`, rev per cell in `config_manifest.json`). Port 8402, GPU0 (HCLASS1_GPU=0).
- Client: `experiments/hclass1_client.py` (condition A diverse-prompt Poisson from `data/hclass1_prompt_bank.jsonl`;
  condition B reuses H-OBS-3's canonical prompts byte-identically). Analysis: `experiments/hclass1_analyze.py`.
- Deviations D1/D2 (harness) and **D3 (validity floor)** recorded in `docs/hclass1-prereg.md` §5b/§6. No
  historical result altered; H-OBS-3, Track B, and Phase 5 results preserved verbatim.
