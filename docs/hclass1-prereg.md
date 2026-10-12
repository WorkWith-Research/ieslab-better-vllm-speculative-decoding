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

1. **Prompt identity:** exactly **2 distinct prompts total** (one per ISL value; the client's
   `make_prompt` `_cache` is keyed by ISL, so SS≡SL share the ISL-50 prompt and LS≡LL share the
   ISL-2048 prompt — verified on raw records: each class shows a single distinct `isl` value), reused
   for all requests in all cells, with `--enable-prefix-caching` → **97.3% prefix-cache hit rate** in
   both dspark cells. Prompt content and cache behavior are fully confounded with the ISL axis; within
   an ISL pair the classes differ only in their configured cap.
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
  A and B. Directional consistency with H-OBS-3 is assessed per condition: in **B** (4 classes), the
  long-horizon classes (SL/LL) show lower DSpark/AR TPOT ratio than SS; in **A** (2 ISL classes S/L),
  the ordering of S vs L is reported descriptively and counts as directionally consistent if it matches
  B's pattern OR if the realized-OSL distribution differs between S and L and the class with the larger
  realized OSL has the lower ratio. *A failure to reproduce the class-dependent payoff in either
  condition is a reportable result (§7).*
- **H-C2 (cache-condition robustness):** the sign of each class's DSpark/AR TPOT ratio is stable across
  cache conditions A and B at matched pressure, for the classes that exist in both (comparing B's SS/SL
  short-ISL pair against A's S, and B's LS/LL long-ISL pair against A's L). *A violation here would show
  H-OBS-3's effect was a cache-reuse artifact.*
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

**Condition A (primary) — two ISL classes, uniform cap.** Classes are defined by **ISL only**; one
uniform `max_tokens = 2048` for both, so the cap is a generous ceiling and realized OSL is a free
outcome variable (natural termination via stop token). The 4-class ISL×OSL-cap grid used in H-OBS-3 is
deliberately NOT replicated here: harness validation showed Qwen3-4B's natural answers to these prompts
are long-tailed (smoke at cap 1024: p50 already at the cap for long-context prompts), so per-class low
caps truncate natural answers and make realized OSL a function of the configured cap — the exact
configured-vs-realized-OSL confound this experiment exists to remove.

| class | ISL (tokens, bracket [t−8, t+16]) | max_tokens | prompts/run |
|---|---:|---:|---:|
| S | 50 (truncated real turns) | 2048 | 480 |
| L | 2048 (real long-context turns, pre-measured pool) | 2048 | 192 |

**Condition B (control) — H-OBS-3's exact configuration** (re-measured side-by-side):

| class | ISL (tokens, exact) | max_tokens | intent |
|---|---:|---:|---|
| SS | 50 (truncated real turns) | 256 | short chat |
| SL | 50 | 1024 | short prompt, long generation |
| LS | 2048 (real long-context turns) | 256 | RAG-style: long context, short answer |
| LL | 2048 | 1024 | document QA, long response |

- ISL is exact by token count (truncate/pad-free: turns are *selected* whose token length brackets the
  target and then truncated to it; padding never used). Selection per class is seeded; the full prompt
  set + sha256 per prompt is written to `results/hclass1/prompts_<condition>.jsonl` before any cell runs.
- **Natural EOS** (no `--ignore-eos`), `temperature=0`, streaming, `stream_options.include_usage` for
  authoritative `completion_tokens`. Realized OSL (`ntok`) is recorded per request and reported as a
  distribution; **realized OSL is an outcome variable**, so any "long-output class" claim in the
  analysis always refers to measured realized OSL, never the configured cap.
- If a class's realized-OSL distribution collapses (e.g., ≥90% of requests hit `max_tokens` in a smoke),
  that class's max_tokens is lowered **during harness validation only** (before any confirmatory cell) and
  the change recorded here — this is workload construction, not result-driven tuning.

**Deviation D1 (recorded during harness validation, 2026-10-12; before any confirmatory cell):**
The first smoke (raw completions, diverse prompts, cache OFF) confirmed distinct prompts and clean
traces but showed `frac_at_max ≈ 1.0` for SS/SL/LS (LL 0.75) — Qwen3-4B on a *bare token
continuation* never emits EOS (no conversational end signal), so realized OSL collapses to the
configured cap and the "long-output class" axis degenerates to the max_tokens setting. This is the
same degenerate behavior as H-OBS-3, now reproduced with diverse prompts, confirming it is a property
of raw-completion prompting, not of prompt repetition. Per the clause above, the workload was adjusted
during harness validation: **condition A serves prompts through the model's chat template**
(`apply_chat_template`, user turn + assistant header) with `stop_token_ids=[151645]`
(`< |im_end|>` = Qwen3-4B eos_token; vLLM drops it from output), so answers can terminate naturally.
Realized ISL is recorded per request (bracket window [target−8, target+16]); realized OSL remains an
outcome variable. **Condition B is unchanged** (raw completions, H-OBS-3 canonical prompts) and now
serves a sharper control: it isolates whether the class effect depends on the decode *horizon*
(realized OSL == cap in B) vs request characteristics under natural termination (A). Condition A's
prompt manifest records templated ids per pid. This deviation is fixed before any confirmatory cell;
no threshold or analysis change accompanies it.

**Deviation D2 (recorded during harness validation, 2026-10-12; before any confirmatory cell):**
The second smoke (chat-templated prompts, stop token, cache OFF) confirmed the termination mechanism
works (natural `im_end` stops observed; ISL brackets exact: S p50=56, L p50≈2040) but showed realized
OSL still pinned to the cap for most classes (SS/LS `frac_at_max = 1.0` at maxtok 256; SL/LL p50 at the
1024 cap): Qwen3-4B's natural answers to ShareGPT questions are long-tailed, so any per-class low cap
is artificial truncation, not a "short-output class". The realized-OSL axis therefore cannot be
controlled by caps on these prompts. Per the clause above, condition A was restructured during harness
validation: **two ISL classes (S/L) with one uniform cap of 2048** (the table above), and the long-ISL
prompt pool is now pre-measured (`data/hclass1_longpool.jsonl`, 1517 turns with templated token length
≥2064, built by `experiments/build_hclass1_longpool.py`) so the 2048-ISL bracket is always reachable.
H-C1/H-C2 were re-stated in §2 to match (range criterion per condition; A's S-vs-L ordering assessed
against realized OSL). Condition B is unchanged and, with its per-class caps, now explicitly measures
the *configured-horizon* version of the class effect — the two conditions jointly separate
decode-horizon dependence from request-characteristic dependence. No threshold change accompanies this
deviation; it is fixed before any confirmatory cell.

**Cache conditions:**

| condition | prefix caching | prompt reuse | purpose |
|---|---|---|---|
| **A** (primary) | OFF (`--no-enable-prefix-caching`) | diverse distinct prompts | de-confounded: no cache, real content diversity |
| **B** (control) | ON | H-OBS-3's 4 canonical prompts verbatim | re-measure the original confounded configuration side-by-side |

Condition C (diverse prompts + cache ON) is deferred — it is only informative if A and B disagree in an
unexplained way, and is pre-declared as out of scope for the confirmatory grid.

**Arrival process:** condition A: Poisson arrivals at a fixed rate λ per condition (open-loop, steady
state), C client workers drawing from the class queue round-robin; requests are issued on arrival, not
back-to-back. λ is chosen in calibration (§5) to hit the target occupancy for each pressure level.
Condition B reproduces H-OBS-3's *back-to-back* worker pattern (no rate control) **by construction** —
it is the original confounded configuration and its offered load is service-time-determined, so A-vs-B
comparisons are about the *sign/direction of class effects*, not matched absolute load (condition C
deferred).

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
issued requests completed and error rate ≤10%; ok-requests ≥ max(15, ⌊0.6·λ·0.9·(DUR−20)⌋) (proportional
to expected arrivals in the steady window — low-load cells legitimately have few requests); steps.jsonl
non-empty with ≥95% of steps carrying `gpu_step_ms`; DSpark cells scheduled >1000 draft tokens, AR cells
exactly 0; async overlap preserved (`in_flight ≥ 2` fraction ≥ 0.9); condition A realized-OSL
distribution not degenerate (overall `frac_at_max < 0.95` — i.e., a majority of requests terminate
naturally below the uniform cap — and per-class ISL p50 within its §4 bracket, else workload drift vs the
§4 smoke). Invalid cells preserved, marked, excluded; one re-run attempt each (as in H-OBS-3).

## 7. Analysis plan and verdict rule (fixed now)

Steady-state window: first 30 s excluded (arrival warmup + server ramp), final 10% of steps excluded
(shutdown tail). Steps are temporally correlated → per-cell distributions reported; **inference is at the
trial level** (n=3 independent servers per cell) plus request-level paired bootstrap for per-class ratios
(requests paired across arms by prompt identity within a trial; resampling requests, not steps).

1. **Per-class DSpark/AR TPOT ratio** (p50 per class, steady window), per condition × pressure, with
   95% CI from trial-level means (Welch) and request-paired bootstrap. Cross-class relative range within
   each (condition, pressure).
2. **H-C1:** SUPPORTED if the cross-class range > 15% in both conditions at ≥1 common pressure level AND
   the direction criterion of §2 holds (B: SL/LL ratio < SS ratio; A: S-vs-L ordering matches B's pattern
   or tracks realized OSL as specified). FALSIFIED if range ≤ 15% in both conditions at all pressures;
   INCONCLUSIVE otherwise (e.g., one condition only, or direction inconsistent).
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
