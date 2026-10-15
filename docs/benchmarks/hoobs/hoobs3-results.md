# H-OBS-3 results — Prefill/decode phase economics of native fixed-K DSpark under heterogeneous serving

**Date:** 2026-10-11 · **Status:** FINAL (per prereg `docs/hoobs3-prereg.md`)
**Run set:** 4 cells, all DONE-VALID on first attempt (0 re-runs used). Frozen env rev recorded in each cell's `config_manifest.json`; serving path unchanged since H-OBS-1 (`8f266ba`).
**Raw data:** `results/hoobs3/hetero/<cell>/` (gitignored, local). Analysis: `experiments/hoobs3_analyze.py` → `analysis.json`, `analysis_table.md`.

---

## 0. Verdict summary

| Hypothesis | Verdict | Key numbers |
|---|---|---|
| **H1** (phase): acceptance differs prefill-heavy vs pure-decode by >0.03 abs | **INCONCLUSIVE** (c32 n-guard; c96 label validity) — no signal where labels are clean | c32: decode A=0.6221, prefill A=0.5931, gap=−0.029 (n=2091/68); c96: 0.6185 vs 0.6260, gap=+0.0075 (n=1031/193) |
| **H2** (class): DSpark/AR TPOT ratio range >5% relative within a pressure level | **SUPPORTED** | c32: ratios 0.552–0.798, range **35.9%**; c96: 0.745–1.166, range **43.8%** |
| **H3** (price): drafter share rises with pressure in DSpark arm (>1pp per +16 running), not in AR | **NOT SUPPORTED** | dspark share 0.214→0.209 (Δ −0.25pp/+16); ar share 0.047→0.036 (Δ −0.55pp/+16). Neither increases |

**Overall:** the draft *payoff* is strongly class-dependent (H2), but shows no reliable phase dependence where labels are clean (H1, cache-dominated workload) and no pressure scaling in relative terms (H3). The opportunity surface for a Track A decision model is **per-class**, not per-phase.

## 1. H1 — acceptance by step phase: INCONCLUSIVE (no signal where labels are clean)

Prereg rule: gap >0.03 absolute, ≥100 draft-carrying steps of each phase per cell; else INCONCLUSIVE with the draft-schedule pattern documented.

- **dspark_c32** (clean labels — no waiting queue in this cell, `waiting=0` on every step): pure-decode A = 0.6221 (n=2091), prefill-heavy A = 0.5931 (n=68). Gap = **−0.029** — just under the 0.03 bar, and the prefill-heavy phase has only 68 draft-carrying steps (<100 guard). → **INCONCLUSIVE per the fixed guard** (borderline: direction is a slight *decrease* during prefill, magnitude below bar).
- **dspark_c96:** decode A = 0.6185 (n=1031), prefill-heavy A = 0.6260 (n=193), gap = +0.0075. Both phases ≥100 steps — *but the phase labels are not valid for a clean comparison in this cell*: under async scheduling with a sustained waiting queue, 98.2% of the "pure-decode" steps (prefill_tokens=0) still had `waiting>0`, i.e., new requests were admitted during them; their prompt work is absorbed by prefix caching (the workload reuses ShareGPT turns across requests), so those steps may execute cached-prompt verification that the `prefill_tokens` field does not capture. The c96 gap is therefore **INCONCLUSIVE (label validity)**, reported for transparency, not counted toward the verdict.
- Alignment sanity: per-step A mean (0.6221) vs global counter ratio (0.6263) in c32; 99.9–100% of steady-window steps carry drafts. The runner schedules drafts on essentially every step, including prefill-heavy ones — the "no drafts during prefill" failure mode the guard anticipated did not occur.

**Interpretation:** where labels are clean (C=32), acceptance is essentially flat across phases (0.62 decode vs 0.59 prefill, gap below the fixed bar with small n). A phase-aware on/off or K policy shows **no reliable signal** from acceptance on this axis.

**Scope caveat (workload property, documented per the guard):** the client draws from a shared prompt pool of 4 (one canonical prompt per class, by design for determinism) and `--enable-prefix-caching` is on — so **97.3% of prefix-cache queries hit in both dspark cells** (636,928/654,576 at c32; 684,544/703,630 at c96). Genuine prompt work is therefore rare: prefill-heavy steps are only 3.2% of steps at C=32, and most admitted long-context requests verify from cache with zero `prefill_tokens`. The "no phase signal" result is thus established for a **cache-dominated workload**; it does not by itself characterize diverse-prompt serving (e.g., fresh RAG queries), where prefill-heavy steps would be far more frequent. That regime is out of scope for this prereg and is flagged as a follow-up question, not a re-run.

## 2. H2 — per-class DSpark/AR TPOT ratio: SUPPORTED (strongly)

Per-request TPOT p50, steady window (client t≥20 s), ratio = DSpark/AR within each pressure level:

| class | C=32 AR ms | C=32 DS ms | ratio | C=96 AR ms | C=96 DS ms | ratio |
|---|---:|---:|---:|---:|---:|---:|
| SS (50→256) | 21.50 | 17.15 | **0.798** | 31.32 | 36.52 | **1.166** |
| SL (50→1024) | 22.18 | 15.15 | **0.683** | 31.89 | 29.37 | **0.921** |
| LS (2048→256) | 21.50 | 15.30 | **0.712** | 31.37 | 31.71 | **1.011** |
| LL (2048→1024) | 22.16 | 12.24 | **0.552** | 31.89 | 23.77 | **0.745** |

- Cross-class relative range: **35.9%** (C=32), **43.8%** (C=96) — both far above the 5% bar.
- The class dependence is *structural*, not noise: long-output requests (SL, LL) gain most from speculation at mid-load (ratios 0.55–0.68); short-output SS gains least and **turns negative under saturation** (1.166 at C=96).
- TTFT cost is paid by every class (DSpark TTFT ≈ 2× AR at C=32: ~230 ms vs ~115 ms; both arms pay ~8–10 s queueing at C=96, DSpark +~1.5 s on top).

**Interpretation:** SD's latency benefit is a function of *how many decode steps the request will run* (OSL), which the client — and any admission-time decision model — can predict from the prompt class/length profile before the first token. This is the concrete opportunity surface: **class-aware (or OSL-predictive) admission/K policy**, not phase-aware.

## 3. H3 — drafter share vs pressure: NOT SUPPORTED

Drafter share D = (gpu_step_ms − gpu_fwd_ms)/gpu_step_ms, steady window:

| cell | all-steps p50 | decode-only p50 | runninḡ | drafter abs ms (mean) |
|---|---:|---:|---:|---:|
| ar_c32 | 0.047 | 0.047 | 31.9 | 1.03 (sampler/copy baseline) |
| dspark_c32 | 0.214 | 0.214 | 31.7 | 16.36 |
| ar_c96 | 0.035 | 0.035 | 63.8 | 1.11 |
| dspark_c96 | 0.208 | 0.209 | 63.9 | 31.16 |

- DSpark slope: **−0.25 pp per +16 running** (0.214→0.209); AR slope: −0.55 pp/+16. Neither arm shows an increase; the prereg's >+1pp/+16 bar is missed in both cells, and the direction is slightly *negative*.
- In absolute terms the drafter price does scale (16.4 → 31.2 ms/step) — it scales with batch size as expected — but the target forward scales faster (61→122 ms), so the *relative* share is flat-to-declining. The drafter's cost is ~21% of step time at both pressures.
- Net SD economics at matched C: throughput **DSpark +30.9%** at C=32 (2278 vs 1741 tok/s), **+12.5%** at C=96 (2734 vs 2430 tok/s). The payoff shrinks with pressure even though the relative price doesn't grow — consistent with H2's SS-class inversion at saturation.

## 4. Hard-stop audit (all clear)

- Step-ID correlation: intact; all cells DONE-VALID, 0 invalid, 0 re-runs.
- Async overlap preserved: in_flight≥2 fraction = 0.999 / 0.998 / 0.997 / 0.995 (ar32/ds32/ar96/ds96) — all ≥0.9; max in_flight = 2 (AR) / 3 (DSpark).
- gpu_step_ms coverage: 100% of steps in all cells. Draft tokens: 528,871 (ds32), 589,253 (ds96); AR cells exactly 0.
- C=96 tracer overhead: not separately isolated here (prereg §3 reports-not-assumes); step counts at C=96 are lower (6601/1400) because steps are longer, and the in_flight check above shows async overlap is intact.

## 5. Opportunity surface for Track A (prereg §5 requirement)

1. **Class/OSL-aware admission (signal: strong).** The DSpark/AR TPOT ratio spans 0.55–1.17 within a single pressure level and the ordering is stable across C (LL best, SS worst; SS crosses above 1 only at saturation). A lightweight decision model that predicts output length class at admission has direct, measurable signal for per-request SD on/off or K selection.
2. **Phase-aware policy (signal: none on this workload, with a scope caveat).** Acceptance is flat (≈0.62) across prefill-heavy and pure-decode steps where labels are clean (C=32). Caveat: the workload is cache-dominated (97.3% prefix-cache hit rate; 4 shared prompts), so prefill-heavy steps were rare — the null does not extend to diverse-prompt serving, where a follow-up run could re-test H1 if Track A needs it.
3. **Pressure as a cost feature (weak, relative terms).** The drafter's *relative* price is constant (~21% of step time); what changes with pressure is the *payoff* (net throughput gain 30.9%→12.5%; SS-class TPOT ratio crosses 1). Pressure should enter the model through predicted payoff, not through a growing cost term.

## 6. Provenance and reproducibility

- Launcher: `scripts/run_hoobs3.sh` (worktree `/home/junior1/_dev/dspark-native-validation`, commit `333b37b`).
- Client: `experiments/hetero_client_qwen3.py`; analysis: `experiments/hoobs3_analyze.py` (this repo, commit `73b1cc4`).
- Tracer change for this run: additive `t_wall` field in `obs_tracer.py` (worktree `333b37b`) + `t_epoch` in `scrape_metrics.py` — both purely additive, serving path untouched.
- Environment identical to H-OBS-1 §3 (Qwen3-4B + dspark_qwen3_4b_block7 K=7, 1× RTX 3090 GPU1, vLLM V2 async scheduling, util 0.90, max_num_seqs 64, max_batched_tokens 4096, prefix caching on).
- H-OBS-1 and Track B results preserved verbatim; nothing here alters them.
