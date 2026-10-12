# Preregistration — H-CLASS-1: request-class-dependent speculation economics under realistic heterogeneous serving

**Date:** 2026-10-12 · **Status:** PROSPECTIVE (fixed before any measurement run)
**Parent evidence (treated as prior exploratory, NOT as unobserved confirmatory samples):** H-OBS-3 (`docs/hoobs3-prereg.md`, `docs/hoobs3-results.md`) — H2 class-dependence SUPPORTED but on a confounded workload; Phase 6 crossover reproduction (`docs/phase6-reproduction-prereg.md` §8); Track B Step 3 final verdict (unchanged).
**Directive:** "From Speculative Decoding Characterization to Request-Aware Serving Scheduling" (2026-10-12), task 3.

---

## 0. Scope and what is NOT being done

A **bounded characterization + replication test**: does native fixed-K=7 DSpark's per-request latency
payoff vary systematically by observable request characteristics under a *realistic, diverse-prompt*
heterogeneous workload, and does it survive cache-condition changes? This experiment produces the
evidence gate for (a) causal predictability of class payoff at decision time and (b) any scheduling
mechanism. It implements **no controller**.

- No learned model, no per-request K changes, no new vLLM source changes beyond the already-frozen
  env-gated obs tracer (H-OBS-1 instrumentation; serving path unchanged since `8f266ba`).
- A negative/null result is a valid, reportable outcome (§7). No workload or threshold edits after
  seeing confirmatory data. Calibration (§5) is the only pre-confirmatory measurement and its output
  (pressure points) is frozen in writing before §6 cells run.

## 1. What H-OBS-3 established — and its three confounds (Step-1 audit findings)

H-OBS-3's H2 (per-class DSpark/AR TPOT ratio range 35.9% @C=32, 43.8% @C=96; LL best, SS worst, SS
crossing above 1 at saturation) was measured on a workload with **three confounds** that this audit
resolved by inspecting `experiments/hetero_client_qwen3.py` and the raw client records:

1. **Prompt identity:** exactly 4 prompts total (one canonical prompt per class;
   `PROMPTS = {c: make_prompt(...)}` with a per-ISL `_cache`) reused for all requests in all cells,
   with `--enable-prefix-caching` → **97.3% prefix-cache hit rate** in both dspark cells. Prompt
   content and cache behavior are fully confounded with class.
2. **Realized OSL == configured max_tokens for 100% of requests.** Verified on the raw records:
   `ntok` equals its class's `maxtok` at min=p50=p95=max in every class/cell (frac_at_max = 1.000,
   zero natural-EOS terminations) despite the client's "NATURAL EOS" docstring — the synthetic prompts
   are random concatenations of shuffled ShareGPT turns with no stopping point, so the model never
   emits EOS before `max_tokens`. The class effect is therefore on (ISL, **configured decode
   horizon**), not on natural output length. "Long-output request" in H-OBS-3 means "request allowed
   to run 1024 steps," which a scheduler can know only as the user's `max_tokens`, not as expected
   realized length.
3. **All-at-once open-loop decay:** C client threads each loop back-to-back; at C=96 with
   max_num_seqs=64 this creates an initial sustained queue that decays as requests complete (no
   arrival-rate control), so "C=96" is not a steady operating point.

**H-CLASS-1 de-confounds all three:** diverse distinct prompts per class, natural EOS on real single-turn
prompts (so realized OSL varies and is measured, not fixed by cap), and Poisson arrivals at a frozen
rate per pressure condition (steady-state open loop). The H-OBS-3 repeated-prompt configuration is kept
as an explicit control condition so the original result can be re-measured side-by-side.

## 2. Hypotheses (fixed now)

- **H-C1 (replication under de-confounding):** the cross-class relative range of per-request
  DSpark/AR TPOT ratio exceeds **15%** within at least one pressure level, in **both** cache conditions
  A and B, with the long-realized-OSL classes gaining more than short-realized-OSL classes (direction
  consistent with H-OBS-3's ordering: SL/LL ≤ SS in ratio) in ≥1 condition.
- **H-C2 (cache-condition robustness):** the sign of each class's DSpark/AR TPOT ratio is stable across
  cache conditions A and B at matched pressure (a class that gains in one does not flip to losing in the
  other by more than trial uncertainty). *A violation here would show H-OBS-3's effect was a
  cache-reuse artifact.*
- **H-C3 (OSL explanatory power, descriptive):** in a per-class regression of DSpark/AR TPOT ratio on
  realized OSL (log), ISL (log), and cell-level acceptance rate, realized OSL carries the dominant
  coefficient. *Descriptive only — §8 states the causal limits.*

## 3. Environment (frozen; identical to H-OBS-3 / Track B)

Qwen3-4B + `deepseek-ai/dspark_qwen3_4b_block7` (K=7), 1× RTX 3090 (SM86) on **GPU1**, vLLM V2 runner
(async scheduling on), `--gpu-memory-utilization 0.90`, `--max-num-seqs 64`,
`--max-num-batched-tokens 4096` (chunked prefill active). Worktree `/home/junior1/_dev/dspark-native-validation`
(branch `track-b-live-sd`, rev recorded per cell in its manifest). Port **8402** (H-OBS-3's 8401 kept
free for any residual job). Fresh server per cell.

## 4. Workload (fixed now)

**Prompt construction — diverse, real text, distinct identities.** Source: ShareGPT-V3 single user→assistant
turns (real conversations, not concatenated shreds), tokenized externally with the served model's tokenizer
(`add_special_tokens=False`), sent as integer-id arrays. Sampling is seeded (**20261013**) and **disjoint**:
each request in a run gets a unique prompt id; no prompt is repeated within a condition; the same seeded
assignment is used for AR and DSpark arms (paired by request identity).

| class | ISL (tokens, exact) | max_tokens | intent | prompts/run |
|---|---:|---:|---|---:|
| SS | 50 (truncated real turns) | 256 | short chat | 480 |
| SL | 50 | 1024 | short prompt, long generation | 480 |
| LS | 2048 (real long-context turns) | 256 | RAG-style: long context, short answer | 192 |
| LL | 2048 | 1024 | document QA, long response | 192 |

- ISL is exact by token count (truncate/pad-free: turns are *selected* whose token length brackets the
  target and then truncated to it; padding never used). Selection per class is seeded; the full prompt
  set + sha256 per prompt is written to `results/hclass1/prompts_<condition>.jsonl` before any cell runs.
- **Natural EOS** (no `--ignore-eos`), `temperature=0`, streaming, `stream_options.include_usage` for
  authoritative `completion_tokens`. Realized OSL (`ntok`) is recorded per request and reported as a
  distribution; classes are defined by (ISL, max_tokens) and **realized OSL is an outcome variable**, so
  "long-output class" claims in the analysis always refer to measured realized OSL.
- If a class's realized-OSL distribution collapses (e.g., ≥90% of requests hit `max_tokens` in a smoke),
  that class's max_tokens is lowered **during harness validation only** (before any confirmatory cell) and
  the change recorded here — this is workload construction, not result-driven tuning.

**Cache conditions:**

| condition | prefix caching | prompt reuse | purpose |
|---|---|---|---|
| **A** (primary) | OFF (`--no-enable-prefix-caching`) | diverse distinct prompts | de-confounded: no cache, real content diversity |
| **B** (control) | ON | H-OBS-3's 4 canonical prompts verbatim | re-measure the original confounded configuration side-by-side |

Condition C (diverse prompts + cache ON) is deferred — it is only informative if A and B disagree in an
unexplained way, and is pre-declared as out of scope for the confirmatory grid.

**Arrival process:** Poisson arrivals at a fixed rate λ per condition (open-loop, steady state), 4 client
workers drawing from the class queue round-robin; requests are issued on arrival, not back-to-back.
λ is chosen in calibration (§5) to hit the target occupancy for each pressure level.

## 5. Calibration run (pre-confirmatory; output frozen before §6)

Small independent characterization, **condition A only**, arm = AR (no drafter cost), C-threads ∈ {8, 16,
32, 48, 96} at candidate λ values, 90 s each, obs tracer ON. Record per point: actual running/waiting
distributions, scheduled/prefill/decode token mix, occupancy time-series, throughput, TPOT/TTFT.

**Pre-specified selection rules (fixed now):**
- **P_low:** the lowest candidate point with sustained `waiting_p50 = 0` and running ≥ 16.
- **P_med:** the point where throughput is within 5% of its maximum *and* `waiting_p50` first becomes
  intermittently > 0 (transition).
- **P_high:** the highest candidate point with sustained `waiting_p50 > 0` (saturated regime; expected
  running ≈ max_num_seqs = 64, queue > 0).
If no candidate shows sustained waiting, P_high = highest feasible point and is labeled "highest
feasible" (not saturated) — recorded deviation. The three chosen (λ, C-threads) pairs are written into
this file's §5b before any §6 cell starts; the confirmatory grid uses only those values.

## 5b. Frozen pressure points (filled by calibration; empty until then)

| level | λ (req/s) | client threads | expected regime | chosen (UTC) |
|---|---|---|---|---|
| P_low | — | — | running-limited, no queue | _pending_ |
| P_med | — | — | transition | _pending_ |
| P_high | — | — | sustained queue | _pending_ |

## 6. Confirmatory design (fixed now; runs only after §5b is filled)

**Cells:** arm ∈ {AR, DSpark-K7} × condition ∈ {A, B} × pressure ∈ {P_low, P_med, P_high} × **3 trials
(fresh server each)** = 36 cells. Within a (condition, pressure) block, trial order is counterbalanced
(AR/DSpark alternating; block order A-before-B at each pressure, pressures in ascending order).

**Recorded (per cell):**
- Per request (client): prompt id, class, arrival t, ttft_ms, tpot_ms, e2e_s, realized ntok, ISL.
- Per step (obs tracer): `t_cpu_sched`, `running`, `waiting`, `scheduled_tokens`, `prefill_tokens`,
  `draft_tokens`, `sd_enabled`, `in_flight`, `gpu_step_ms`, `gpu_fwd_ms`.
- Prometheus @0.5 s: `vllm:spec_decode_num_drafts[_tokens]`, `vllm:spec_decode_num_accepted_tokens`,
  running/waiting/KV gauges, prefix-cache hit/queries (conditions A/B), GPU util/mem/power.

**Validity rule (per cell):** server startup-complete marker in its own log; client exit 0 with ≥50% of
issued requests completed and error rate ≤10%; steps.jsonl non-empty with ≥95% of steps carrying
`gpu_step_ms`; DSpark cells scheduled >1000 draft tokens, AR cells exactly 0; realized-OSL distribution
per class not degenerate (≥5% of requests below max_tokens in ≥2 classes — else workload drift vs §4
smoke). Invalid cells preserved, marked, excluded; one re-run attempt each (as in H-OBS-3).

## 7. Analysis plan and verdict rule (fixed now)

Steady-state window: first 30 s excluded (arrival warmup + server ramp), final 10% of steps excluded
(shutdown tail). Steps are temporally correlated → per-cell distributions reported; **inference is at the
trial level** (n=3 independent servers per cell) plus request-level paired bootstrap for per-class ratios
(requests paired across arms by prompt identity within a trial; resampling requests, not steps).

1. **Per-class DSpark/AR TPOT ratio** (p50 per class, steady window), per condition × pressure, with
   95% CI from trial-level means (Welch) and request-paired bootstrap. Cross-class relative range within
   each (condition, pressure).
2. **H-C1:** SUPPORTED if range > 15% in both conditions A and B at ≥1 common pressure level AND the
   long-realized-OSL classes have lower ratio than short-realized-OSL classes in ≥1 condition; FALSIFIED
   if range ≤ 15% in both conditions at all pressures; INCONCLUSIVE otherwise (e.g., one condition only,
   or direction inconsistent).
3. **H-C2:** per-class sign stability across A/B at matched pressure; report the worst flip magnitude vs
   trial uncertainty.
4. **H-C3 (descriptive):** OLS of class ratio on log(realized OSL), log(ISL), cell acceptance rate;
   report coefficients + which variable carries the most variance. Explicitly labeled observational.
5. **Aggregate:** server-side output throughput per cell × condition (steady window); DSpark-vs-AR delta
   with trial-level CI; per-class TTFT and E2E latency percentiles; tail (p95/p99) per class.
6. **Mechanism inputs:** cell-level acceptance rate + accepted tokens/step (Prometheus deltas, time-aligned
   to steps as in H-OBS-3); drafter share `(gpu_step_ms − gpu_fwd_ms)/gpu_step_ms` by arm × pressure;
   prefix-cache hit rate by condition; prefill-token share of scheduled tokens.
7. **Overhead:** AR cells are the no-drafter baseline; tracer overhead reported as in H-OBS-1 (≤0.5% at
   C≤32, measured not assumed at P_high).

**Verdict categories per cell-set:** SUPPORTED / FALSIFIED / INCONCLUSIVE / INVALID with the numbers
above; overall = conjunction summary + explicit statement of which H-OBS-3 confounds were removed and
which residual confounders remain (prompt-content ↔ class is *partially* de-confounded: class still
correlates with prompt length regime, so content effects within an ISL band are not isolated).

**Hard stops (abort, no re-tune):** GPU1 not free at run start (refuse; do not kill external work); OOM or
serving crash in any cell beyond one re-run; obs tracer destroys async overlap (`in_flight ≥ 2` fraction
< 0.9); >4 cells invalid after re-runs; total wall > 20 GPU-hours for calibration + confirmatory combined.

**Boundedness:** calibration ≤ 5 points × 1 arm (≤1.5 GPU-h) + 36 confirmatory cells × ~3.5 min each
(≈ 2.5–4 GPU-h including server starts) — well under the hard stop. No additional arms, no workload edits
after §5b is frozen, no post-hoc window changes.

## 8. Causal limits (fixed now; governs interpretation)

- Realized OSL is an **outcome** of (prompt content, model), not a scheduler input. H-C3's regression is
  observational: prompt content drives both realized OSL and draft acceptance, so the OSL coefficient is
  confounded with content/draftability. No causal claim about "output length causes SD payoff" is made
  from this experiment; that question belongs to the decision-time predictability gate (Deliverable 4),
  which uses only causally-available features.
- Per-class ratios are request-paired across arms within a trial, but trials share a server → trial-level
  n=3 bounds the uncertainty of *ratios*; aggregate throughput claims use trial-level CIs only.
- Condition B reproduces H-OBS-3's confounded design by construction; any A-vs-B difference is attributed
  to (prompt diversity + cache) jointly, not separately (condition C deferred).

## 9. Provenance (filled after run)

- Launcher: `experiments/hclass1_run.sh` (this repo; patterned on H-OBS-3's launcher: GPU lock, health
  verification, per-cell manifest with worktree rev, validity checks).
- Client: `experiments/hclass1_client.py` (diverse-prompt Poisson client; condition B mode reuses the
  canonical 4 prompts). Prompt sets: `results/hclass1/prompts_<cond>.jsonl` + sha256 manifest.
- Analysis: `experiments/hclass1_analyze.py` → `results/hclass1/analysis.json` + `analysis_table.md`.
- Raw artifacts under `results/hclass1/<cell>/` (gitignored; local only). Workload hashes recorded per
  cell in its `config_manifest.json`.
