# Phase 7 capstone — results (S3 live comparison run)

**Date:** 2026-10-15 · **Prereg:** `docs/phase7-capstone-prereg.md` · **Harness:** `experiments/phase7_s3_run.sh`
**Runtime:** dspark-native-validation worktree (`track-b-live-sd`, controller commits `2ab37e7`→`a5b86e9`), Qwen3-4B +
dspark_qwen3_4b_block7 (K=7), 1× RTX 3090, prefix caching ON, max-num-seqs 64. 57 cells: 5 controller arms × {c16,c48,c96}
× 3 reps + AR baseline × {c48,c96} × 3 reps (ar1/ar2), fresh server per trial, shared workload seed 20261013.

## 0. Verdict

> **FALSIFIED (S1 and S3 fail beyond uncertainty).** The live 2-axis controller does **not** beat the single-axis
> ablations end-to-end: at high load it is *worse* than CTRL-LOADONLY, and it recovers only a fraction of the AR baseline.
> The static payoff surface (`docs/scheduler-signal-validation.md`) does **not transfer to live serving** — its Bc96 cell
> (+0.022, DSpark marginally ahead) flips *negative* under live mixed-batch serving. The class axis is a **no-op-to-negative**
> contribution end-to-end; the realized gain of the capstone design reduces to the load axis alone (which itself ≈ native
> DSpark at high load). S2/S4/S5 pass. This is a recorded falsification per prereg §6 — no retune loop.

## 1. Measured result — steady-state generation throughput (tokens/s, server-side)

Metric: vLLM's own `Avg generation throughput` (loggers.py, 10 s intervals), mean over the run excluding the warmup
interval; robust to restricting to saturated-serving intervals (Running≥24) — identical to ±0.5% at c48/c96. This is the
correct metric here: client-completion counts are not, because these long generations have e2e p50 ≈ 40–52 s at c48/c96, so
only 2–39 requests complete in a 45 s window and completion-based throughput is dominated by which wave happens to finish.

| load | arm | r0 | r1 | r2 | mean ± sd | vs AR (c48/96) / DSpark (all) |
|---|---|---:|---:|---:|---:|---:|
| c16 | CTRL-LIVE | 1146 | 1228 | 1227 | **1200 ± 47** | −1.8% vs DSpark (within δ) |
| c16 | DSpark-K7 | 1202 | 1225 | 1239 | 1222 ± 18 | baseline |
| c16 | CTRL-ON | 1381 | 1240 | 1261 | 1294 ± 76 | +5.9% vs DSpark (within δ) |
| c16 | CTRL-CLASSONLY | 1137 | 1090 | 1113 | 1113 ± 23 | −8.9% vs DSpark |
| c16 | CTRL-LOADONLY | 1151 | 1243 | 1391 | 1262 ± 121 | +3.3% vs DSpark (within δ) |
| c48 | CTRL-LIVE | 1233 | 1223 | 1239 | **1232 ± 8** | −27.7% vs AR; +8.0% vs DSpark |
| c48 | DSpark-K7 | 1128 | 1136 | 1160 | 1141 ± 16 | −33.0% vs AR |
| c48 | CTRL-ON | 1139 | 1134 | 1121 | 1131 ± 9 | −33.6% vs AR; −0.9% vs DSpark |
| c48 | CTRL-CLASSONLY | 1149 | 1174 | 1185 | 1170 ± 18 | −31.4% vs AR |
| c48 | CTRL-LOADONLY | 1442 | 1409 | 1420 | **1424 ± 17** | −16.5% vs AR; **+24.8% vs DSpark** |
| c48 | AR (ar1/ar2) | — | — | — | 1705 ± 36 | baseline |
| c96 | CTRL-LIVE | 1109 | 1112 | 1114 | **1111 ± 2** | −29.4% vs AR; +9.5% vs DSpark |
| c96 | DSpark-K7 | 1026 | 1001 | 1017 | 1015 ± 13 | −35.5% vs AR |
| c96 | CTRL-ON | 996 | 1014 | 981 | 997 ± 17 | −36.7% vs AR; −1.8% vs DSpark |
| c96 | CTRL-CLASSONLY | 1051 | 1058 | 1049 | 1052 ± 4 | −33.1% vs AR |
| c96 | CTRL-LOADONLY | 1395 | 1367 | 1386 | **1383 ± 14** | −12.2% vs AR; **+36.2% vs DSpark** |
| c96 | AR (ar1/ar2) | — | — | — | 1574 ± 14 | baseline |

δ (run-to-run uncertainty, max stdev of the two baselines AR/DSpark-K7) ≈ **36 tok/s**. Load balance across arms verified
from server logs: at c96 all arms ran Running≈44–51 / KV≈98%; at c48 Running≈44–51 (AR 63, a self-inflicted difference — AR
serves faster and drains the queue), so no arm was starved relative to another.

## 2. Verdict against prereg §6

- **S1 (crossover recovery) — FAILS.** At c96, CTRL-LIVE = 1111 vs AR = 1574 tok/s: **−29.4% (≈13×δ)**, far outside any CI
  of AR. It does beat DSpark-K7 (+9.5%, >0 in 3/3 paired reps), but the prereg requires ≥ AR within CI — not met. Note the
  failure is *shared by native DSpark-K7 itself* (−35.5% vs AR): at saturated mixed load, K=7 speculation loses to plain AR
  on this hardware/workload. The controller's job was to recover that gap; it recovered ~¼ of it.
- **S2 (low-load preservation) — PASSES.** At c16, CTRL-LIVE = 1200 vs DSpark-K7 = 1222: −1.8%, within δ. The load axis
  keeps SD on at low load and preserves the benefit (DSpark +?% over AR there; live ≈ native).
- **S3 (both axes necessary — capstone claim) — FAILS.** Aggregate realized gain (c48+c96): CTRL-LIVE = 1172 vs
  CTRL-LOADONLY = 1404 (**−16.5%, ~5–7×δ**) and vs CTRL-CLASSONLY = 1111 (+5.4%). The rule beats only the class-only
  ablation; it **loses to the load-only ablation**. "Both axes necessary" is falsified end-to-end: the class axis's high-load
  rescue (SD on for high-reuse traffic at c48/c96) *costs* throughput relative to staying off.
- **S4 (overhead) — PASSES.** CTRL-ON vs DSpark-K7: +5.9% (c16, within the arm's own sd 76), −0.9% (c48), −1.8% (c96). No
  detectable patch/decision overhead beyond run-to-run noise.
- **S5 (causal correctness) — PASSES.** All 36 controller traces: OFF steps with >0 drafts scheduled = **0**.

## 3. Mechanism — why the static surface did not transfer

The static payoff surface (`docs/scheduler-signal-validation.md`) was measured on **homogeneous per-class cells** (A/B/C
each served alone, pid-matched TPOT pairs). Its Bc96 cell said DSpark beats AR by +0.022 — a 2% margin with no safety band
(the doc itself flagged this as the least-robust cell). In **live serving**, batches are *mixed*: at c48/c96 a running batch
contains diverse A requests and high-reuse B requests together, and SD applies to the whole step. Two effects flip the
marginal Bc96 gain negative:

1. **Batch-mix dilution.** The class axis turns SD on when the *admission* hit-rate is high (B block), but the running batch
   at that moment also holds in-flight diverse A requests whose drafts are rejected — per-step acceptance is a mix, below the
   homogeneous-B value the static cell assumed. DSpark's always-on mean acceptance length degrades with load (3.36 c16 →
   3.16 c48 → 3.02 c96 in this run); on *mixed* batches at c96 it is lower still for the diverse fraction.
2. **Draft overhead at saturation.** At KV≈98% every extra draft forward pass is paid against a fully occupied GPU; when
   acceptance is diluted, each speculative step costs more than it saves — exactly the Ac48/Ac96 penalty, now applied to the
   whole mixed batch instead of only the diverse one.

Net: turning SD on for "high-reuse" windows at high load is *worse* than leaving it off (CTRL-LOADONLY), and even native
always-on DSpark-K7 loses 35% to AR — the static surface's "+0.022 Bc96" was an artifact of homogeneous-batch measurement
and did not survive contact with mixed live batches.

**What this establishes:** (1) a per-step, causal, request-aware SD on/off controller is *buildable and correct* in vLLM
(S5 across 36 live traces; S4 shows the mechanism is free); (2) its realized value at fixed K=7 on this hardware/workload is
≈ native DSpark (high load) to slightly-better-than-native (mid/low), i.e. **the capstone's end-to-end gain claim does not
hold** — the 2-axis rule's second axis is a no-op-to-negative contribution live; (3) the static payoff surface must be
re-measured on *mixed* batches before any per-class on/off policy is trusted for deployment decisions.

## 4. Provenance & reproducibility

- Raw data: `results/phase7_s3/` (57 cell dirs: server.log, trace.jsonl, client.jsonl, client.log) — copied from
  `/tmp/phase7_s3` before cleanup; `cells.jsonl` manifest with per-cell rc + timestamps.
- Analysis: steady gen-throughput parsed from server.log (`Avg generation throughput ... Running ... KV cache usage`);
  S5 assertion over trace.jsonl (OFF step ⇒ drafts_scheduled_this_step == 0).
- Controller: worktree `track-b-live-sd` @ `a5b86e9` (+`616fe46` comment-only). Modes: live/loadonly/classonly/on via
  Phase7Controller; dspark = native (zero-overhead bypass); ar = no spec config.
- Known measurement limits: 45 s cells with long generations → c48/c96 completion counts low (why the server-side rate is
  used); AR at c48 self-drains its queue (Running 63 vs ~50) — a workload effect of AR being faster, not a harness bias
  (all SD arms share identical load).

## 5. Consequence for the project

Phase 7 closes with a **negative capstone result** and two positive byproducts: a causal, overhead-free live SD controller
(the mechanism Track B built, now validated end-to-end), and a hard constraint on the static-signal approach — *population
payoff surfaces measured on homogeneous cells do not bound per-step decisions under mixed batches*. The research question
for any successor is therefore not "which on/off rule" but "how to measure/estimate the *mixed-batch* payoff at admission,
or how to make speculation batch-aware (per-request K / selective drafts)", which is outside the fixed-K scope of this phase.
