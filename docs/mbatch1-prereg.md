# M-BATCH-1 — B-class payoff under live MIXED batches (mechanism confirmation)

**Status:** PREREGISTERED 2026-10-15, before any mixed-batch data of this design was collected.
**Parent:** Phase 7 capstone falsification (`docs/phase7-capstone-results.md`). This is NOT a retune or a
re-test of S3 (prereg §6 "single run set" still stands for the capstone verdict). It measures the *mechanism*
S3's analysis argued for, and decides the next direction with evidence.

## 1. Question

Phase 7 found the live 2-axis controller loses to load-only at high load (−16.5% aggregate c48+c96) and argued
the cause: the static payoff surface (`docs/scheduler-signal-validation.md`) was measured on **homogeneous**
per-class cells, where class B's DSpark/AR TPOT payoff is +0.121 (c48, n=1200) and +0.022 (c96, n=1530); under
live **mixed-batch** serving that marginal cell is expected to flip negative (batch-mix acceptance dilution +
draft overhead at KV≈98%).

That argument has never been measured: no run so far has looked at B-class payoff *inside* a mixed stream.
M-BATCH-1 does exactly that, with the identical workload family S3 used.

## 2. Hypotheses (falsifiable)

- **H1 (mechanism confirmed):** DSpark/AR TPOT payoff for class-B requests in the MIX stream is ≤ 0 at c96
  (and < homogeneous +0.121 at c48), i.e. B's marginal gain vanishes under mixed batches.
- **H2 (headroom exists):** B payoff stays > 0 within CI at c48 or c96 despite mixing — then the class axis has
  real live value and S3's loss is a *timing/estimation* failure of the controller, not of the signal.

## 3. Design

- **Arms (2):** `ar` (no speculative config) vs `dspark` (native DSpark-K7 always-on; VLLM_TB_MODE=native).
  Same two arms S3 used for the crossover comparison — no controller involved, so this isolates the *payoff
  surface* from any decision rule.
- **Load (2):** c48, c96 (the high-load regime where S3 diverged; c16 is irrelevant to the claim).
- **Reps (3):** fresh server per trial, identical workload seed 20261013 → both arms draw the *same* class
  sequence and A-prompt assignments from the seed.
- **Workload:** open-loop Poisson arrivals at the H-CLASS-1 frozen pressures (c48: λ=1.7143 req/s, c96:
  λ=3.4286 req/s), each request class A (diverse unique prompt from BANK_A, stop-token chat format) with prob
  0.5 else class B (one of the 4 canonical high-reuse prompts, round-robin over arrival index). Workers = C
  (48/96, matching H-CLASS-1 cond-A concurrency), closed-loop over a FIFO queue, so at c48/c96 arrivals back up
  and **B requests decode in batches alongside in-flight/queued A requests** — the mixed-batch regime under test. Prefix caching ON (B signal requires it).
  This *relaxes* S3's block-alternating closed-loop MIX (whose 45 s cells captured only 3–14
  completions per cell — too few for a paired per-class payoff; B completions ranged 0–10, not zero:
  an earlier "zero B" reading was a classifier bug counting `cls.startswith("B")` when class names
  carry no prefix — see §7) while keeping the same traffic families.
- **Duration:** 150 s per cell (hclass1 convention; ≥3 valid reps per load). WARMUP 10 s excluded from analysis.
- **Server:** Qwen3-4B + dspark_qwen3_4b_block7 (K=7), TP=1, GPU0, gpu-mem-util 0.90, max-num-seqs 64,
  max-num-batched-tokens 4096 — identical to S3 and H-CLASS-1.

Cells: {ar, dspark} × {c48, c96} × {t1,t2,t3} = **12 cells**, ~5 min each (startup + run + teardown) → ~70 min.

## 4. Analysis (pre-specified)

Primary: per-class TPOT payoff for B requests = mean over pid-matched pairs of `1 − DSparkTPOT/ARTPOT`
(>0 ⇒ DSpark is faster for that class — same convention as `docs/scheduler-signal-validation.md`), within each
load, pooling the 3 reps (pid = seeded arrival index, identical across arms). Request-level paired bootstrap
over the pooled pairs (2000 resamples). Payoff ≤ 0 ⇒ DSpark is not faster for that class. Compare against
homogeneous anchors from H-CLASS-1 `results/hclass1/{ar,dspark}_B_c{48,96}_t{1,2,3}` (same prompts, same seed,
back-to-back B only): pid-matched +0.106 (c48, n=432) / +0.006 (c96, n=518).

Secondary (mechanism probes, descriptive):
- A-class payoff in the same cells (control: expected negative — diverse drafts rejected; S3 saw AR beat DSpark
  on A traffic at high load).
- B payoff vs `in_flight_at_arrival` (queue depth proxy at admission) — direct dilution/overhead curve.
- Per-cell server-side context: KV usage, prefix-cache hit rate, avg generation throughput.

## 5. Decision rule (frozen now)

- H1 confirmed (B payoff ≤ 0 within CI at c96, or clearly below homogeneous anchors at both loads): the
  mixed-batch flip is real → the class axis has **no headroom at fixed K=7 on this hardware** under live serving.
  Capstone stays closed as falsified; successor scope = batch-aware speculation (per-request K / selective
  drafts), which changes what SD applies to rather than when.
- H2 (B payoff > 0 within CI at either high load): headroom exists → S3's failure is timing/estimation → pivot
  to estimating the mixed-batch payoff at admission (the actual research question).

Either outcome decides the next phase; no third "it depends" branch. If a cell fails validation it is dropped
from that rep (min 2 valid reps per load required, else the design is reported as underpowered and H1/H2 stand
as untested — not re-run with changed parameters).

## 6. Validity gates (per cell)

- startup-complete in own server.log; client rc=0 or bounded tail (client prints ok/err counts).
- Error fraction <10%.
- ≥20 completed B-class requests per cell (floor = data quality, not power: S3 showed DSpark-K7
  serves ~3× slower than AR at c96 saturation, so dspark cells complete fewer requests; the paired
  analysis pools all 3 reps).
- Power requirement for the primary claim: ≥60 pid-matched B pairs pooled across reps per load — if not
  met, that load is reported as underpowered and H1/H2 stand as untested (no re-run with changed params).
- dspark arm: speculative decoding actually active (server log spec/draft evidence); ar arm: no spec config.
- Same seed both arms → identical class sequence and A-prompt assignments; pairing by pid.

## 7. Honest scope / known confounds

- **Classifier bug (caught during bring-up):** class names carry no "B" prefix (hclass1 convention: A = S/L,
  B = SS/SL/LS/LL), so `cls.startswith("B")` counts zero. All M-BATCH-1 scripts classify by membership in
  {SS, SL, LS, LL}. The same bug affected an earlier reading of S3 raw data ("zero completed B requests");
  the correct S3 counts are 0–10 B completions per 45 s cell — still too few for paired payoff, but not zero.
- This stream is **not** byte-identical to S3's block-alternating MIX (S3's B blocks were closed-loop hammers;
  here B arrivals are Poisson at half rate). It keeps the same traffic families (diverse A bank + canonical B
  prompts), the same server, and the same mixed-batch regime — but it is a *different workload realization*, so
  absolute payoff levels should be compared to the H-CLASS-1 homogeneous anchors as direction/magnitude
  references, not exact like-for-like controls. The claim is about the **sign flip** of B's marginal payoff under
  mixing, which this design isolates.
- TPOT payoff confounds queueing with per-step cost (a B request's TPOT includes waiting for A requests to
  finish); the `in_flight_at_arrival` secondary analysis exists precisely to separate these.
- One model/draft pair, one GPU, K=7 fixed — same scope as Phase 7; this measures a mechanism on that platform,
  not a general law.

## 8. Provenance

Harness: `experiments/mbatch1_run.sh` (server config byte-identical to S3's start_server; open-loop Poisson
client with frozen H-CLASS-1 lambdas). Client: `experiments/mbatch1_client.py` (new; reuses hclass1_prompts'
BANK_A/CANON verbatim — same prompts as S3 and H-CLASS-1). Analysis: `experiments/mbatch1_analyze.py`.
Raw data: `results/mbatch1/` (gitignored). Homogeneous anchors: `results/hclass1/` (H-CLASS-1, prereg
`docs/hclass1-prereg.md`). S3 workload reference: `experiments/phase7_mixed_client.py` + `results/phase7_s3/`.
