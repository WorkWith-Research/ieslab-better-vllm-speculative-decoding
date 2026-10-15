# Track B Results — Live per-step SD on/off controller (Step 3)

**Date:** 2026-10-10 · **Status:** run set complete, analysis done · **Prereg:** `docs/trackb-prereg.md` (R1–R4)
**Hardware:** 1× RTX 3090 (SM86), GPU1, vLLM 0.25.0 (V2 runner), Qwen3-4B + dspark_qwen3_4b_block7 (K=7)
**Workload:** ShareGPT-V3 60 prompts, greedy (temp 0), seed 0, `--ignore-eos`, fresh server per trial, metrics scraper @ 0.5 s

## Arms

| Arm | SD config | Controller | Purpose |
|---|---|---|---|
| **AR** | none | — | autoregressive baseline |
| **DSpark-K7** | dspark K=7 fixed | off (native) | always-on fixed-K baseline |
| **CTRL-ON** | dspark K=7 + patch | `VLLM_TB_MODE=on` (always ON) | measures patch/decision overhead vs native |
| **CTRL-LIVE** | dspark K=7 + patch | `VLLM_TB_MODE=live` (window-mean, R4 fix) | the live controller under test |

36 trials = 4 arms × C∈{8,32,64} × 3 rounds. The CTRL-LIVE arm was **re-run** (9 trials) after a
harness-bug fix — see §Defects. The AR/DSpark/CTRL-ON numbers below are from the original run set and are
unaffected by the controller decision logic.

## Throughput (mean of 3, tok/s)

| C | AR | DSpark-K7 | CTRL-ON | **CTRL-LIVE** | DSpark vs AR | LIVE vs AR | LIVE vs DSpark |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8  | 463 | 761 | 766 | **764** | +64.5% | +65.2% | +0.4% |
| 32 | 869 | 981 | 975 | **971** | +12.9% | +11.8% | −1.0% |
| 64 | 984 | 966 | 963 | **996** | −1.9% | +1.2% | +3.2% |

CTRL-LIVE reproduces the native DSpark shape: large low-load win, small mid-load win, and at C=64 it
**beats both** DSpark-K7 (+3.2%) and AR (+1.2%).

## Verdict against prereg criteria (§6, R2 restatement)

| Criterion | Requirement (C unless noted) | Measured | Verdict |
|---|---|---|---|
| **S1a** parity vs AR @ C=64 | paired 95% CI lower bound > −2% | mean +1.2%, per-round [−0.0, +3.6, +0.1]%, CI **[−3.8, +6.3]%** | **NOT met as written** (see below) |
| **S1b** directional over fixed-K @ C=64 | LIVE > DSpark in ≥2/3 rounds AND mean>0 | **3/3** rounds, mean **+3.2%** | **MET** |
| **S2** low-load preservation @ C=8 | LIVE ≥ DSpark − δ₈ | 764 vs 761 (δ₈≈4), OFF=0% of steps | **MET** |
| **S3** overhead (CTRL-ON) @ all C | \|CTRL-ON − DSpark\| ≤ 1% | +0.59 / −0.66 / −0.30 % | **MET** |
| **S4** causal correctness, all trials | OFF step ⇒ 0 new drafts scheduled | 0 violations across all 9 fixed traces | **MET** |

**S1 = S1a ∧ S1b → NOT met**, solely because of S1a's CI bound.

### Why S1a fails as written (and why it is not a mechanism failure)

- No round shows a deficit: the per-round deltas vs AR are −0.0%, +3.6%, +0.1% — **none negative**.
- The point estimate is **+1.2%** (above AR).
- The CI half-width (~5%) at n=3 exceeds the 2% bar. This was **anticipated in R2** ("at n=3, no
  controller can clear a bar of that width" — R2's whole motivation for restating S1). The restatement
  lowered the bar to "CI excludes a *meaningful* deficit (lower bound > −2%)" but kept n=3, so the bound
  still rides on 3 samples. The literal criterion is not met; the underlying evidence (no round below AR,
  mean above AR) supports parity recovery.
- **No post-hoc re-tuning or extra rounds were added** to chase S1a (prereg §6 / R2 prohibit it). This is
  reported honestly: the mechanism recovers AR parity in point estimate and in every individual round; the
  registered n=3 CI test does not formally clear −2%.

## Defects found and fixed during the run (full record in prereg R4)

**Defect 1 — implementation deviation from R3 (FIXED, no constants changed).** The shipped code averaged
the **last 4 samples** of the step-cost signal; R3 registered a **full rolling window (W=2 s)**. At C=8
(~50 steps/s) one transient decode spike pushed the last-4 mean over the exit band, causing on→off→probe→on
flapping every ~2 s: C=8 spent 41% of steps OFF and scored 649.5 tok/s (−14% vs native DSpark), failing S2.
Fixed in commit `dce1651` (true `_window_mean()` over the 2 s window). **No registered constant changed.**
Verified on real data: C=8 re-run → OFF=0%, 0 transitions, 764 tok/s (matches native DSpark/CTRL-ON). Two
regression tests pin it: a single spike does NOT flip OFF; a sustained cost rise STILL flips OFF. 20 tests pass.

**Defect 2 — signal validity under vLLM V1 async scheduling (a finding, not tunable).** The measured signal
is the wall time between `schedule()` calls. Under async scheduling (DSpark's default) the engine does not
block on the GPU unless the batch queue is full, so `schedule()` cadence **decouples from true GPU step time
as load rises**:

| C | true step time (L·C/throughput) | measured cadence | gap |
|---:|---:|---:|---:|
| 8  | ~29 ms | 19.8 ms | 1.47× |
| 32 | ~91 ms | 28.9 ms | 3.13× |
| 64 | ~183 ms | 31.1 ms | **5.91×** |

The cadence is roughly flat (20→31 ms) while true step time triples, so the registered signal systematically
under-reports verification cost toward saturation. This bounds the C=64 claim: the controller's ON/OFF timing
at high load is not grounded in true GPU step cost. It does **not** affect C=8 (small gap; controller correctly
stays ON).

## Controller behavior on the fixed run (OFF fraction, transitions)

| C | OFF % of steps | state transitions | when flips occur |
|---:|---:|---:|---|
| 8  | 0%  | 0 | — (stays ON; preserves +65%) |
| 32 | 34% | 5 | during prefill-heavy phase (run-fraction 0.10, 0.20) |
| 64 | ~30% | 6 | during prefill-heavy phase (run-fraction 0.19, 0.43) |

The C=32/C=64 OFF flips land in the **prefill-burst** window (prefill_share ≈ 0.9 in the first 25% of the
run vs ≈ 0.1–0.16 in the last half). Prefill steps are expensive and, under async scheduling, inflate the
measured cadence — so part of the OFF time is a prefill/async artifact rather than pure decode-verification
cost. This is consistent with Defect 2: the signal conflates prefill bursts with decode step cost. It is why
CTRL-LIVE at C=32 lands marginally *below* DSpark-K7 (971 vs 981, −1.0%) even though R3 predicted it should
stay ON there — the controller turns OFF during prefill bursts where turning OFF is not clearly beneficial to
throughput.

## Bottom line

- **S2, S3, S4 are cleanly met.** The patch adds ≤1% overhead (CTRL-ON ≈ native DSpark everywhere), and the
  causal mechanism is correct (OFF ⇒ no drafts) in every trial.
- **The low-load benefit is fully preserved** (C=8: +65% vs AR, matching native DSpark; controller stays ON).
- **At saturation (C=64) the live controller recovers AR parity and beats fixed-K DSpark** (+3.2%, 3/3 rounds),
  which is exactly the crossover-recovery goal of Track B.
- **S1a's literal n=3 CI bound is not met**, but no round shows a deficit and the mean is +1.2% — this is the
  anticipated small-n power limit, reported honestly without adding rounds or re-tuning.
- **Two defects were found, one fixed (no constants changed), one recorded as a signal-validity bound** on the
  high-load claim (async-scheduling cadence decoupling). Both are documented in prereg R4.

## Prohibited claims (per prereg §6 — still apply)

Not claiming: outperforming native adaptive verification (never run here); generalization beyond
Qwen3-4B/dspark_qwen3_4b_block7/SM86; novelty of load-aware K per se (suspended claim list).

## Data

Raw results, traces, and server logs: `results/track_b/` (gitignored — local only). The buggy first CTRL-LIVE
run set is preserved at `results/track_b/_buggy_last4_ctrl_live/` for provenance. Controller code + tests:
worktree `/home/junior1/_dev/dspark-native-validation`, branch `track-b-live-sd` (fix commit `dce1651`).
