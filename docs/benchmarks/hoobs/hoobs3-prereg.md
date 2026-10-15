# Preregistration — H-OBS-3: Prefill/decode phase economics of native fixed-K DSpark under heterogeneous serving

**Date:** 2026-10-11 · **Status:** PROSPECTIVE (fixed before any measurement run)
**Parent results:** Track B Step 3 (final, unchanged); H-OBS-1 (`docs/hoobs1-prereg.md`, `docs/hoobs1-observability.md`) — PARTIALLY CONFIRMED.
**Task origin:** queued follow-up after H-OBS-1 ("heterogeneous prefill/decode characterization of native fixed-K DSpark — preregister then run").

---

## 0. Scope and what is NOT being done

A **bounded characterization** of how native fixed-K=7 DSpark's draft economics behave across request phases (prefill-heavy vs pure-decode steps) and request classes under a realistic heterogeneous workload, at two batch pressures. It produces inputs for the Track A decision model (per-phase/per-class acceptance = the state features a lightweight controller would see).

- No learned controller, no per-request K changes, no new vLLM source changes beyond the already-frozen env-gated obs tracer (H-OBS-1 instrumentation; serving path unchanged since `8f266ba`).
- H-OBS-2 remains NOT pursued (per H-OBS-1 verdict). This experiment does not reopen it.
- A negative/null result is a valid, reportable outcome (§5 hard stops). No re-tuning of the workload or arms after seeing data.

## 1. Motivation and what prior results leave open

- **Phase 6 / Track B:** per-position acceptance was load-invariant on *homogeneous* ShareGPT decode (0.72→…→0.03 at all C; acc_len ≈ 2.77). But every cell so far used a single homogeneous prompt distribution and `--ignore-eos`.
- **H-OBS-1:** direct GPU timing shows `gpu_step_ms` = target forward + sampling + drafter, cleanly separable (`gpu_fwd_ms` vs `step − fwd`), at ≤0.5% overhead (C≤32). The drafter cost is ~7.5 ms/step at C=32 — a fixed per-step price paid whether or not the drafts are useful.
- **Open question:** on a *heterogeneous* workload (mixed ISL/OSL, real EOS), does DSpark's acceptance vary systematically by (a) step phase (prefill-heavy vs pure decode — chunked prefill interleaves long-context verification with short decode drafts) and (b) request class? If the draft price is paid uniformly but its payoff varies by phase/class, that is a concrete, measurable opportunity surface for the Track A decision model (and a cost to charge against any SD on/off or K policy).

## 2. Hypotheses (fixed now)

- **H1 (phase):** per-step acceptance rate `A = Σdraft_tokens_accepted / Σdraft_tokens_scheduled` differs between prefill-heavy steps and pure-decode steps by > 0.03 absolute (either direction), on ≥100 steps of each phase in the same cell.
- **H2 (class):** per-request-class DSpark-vs-AR TPOT ratio varies across classes SS/SL/LS/LL with a relative range > 5% within one pressure level (i.e., SD's latency benefit is class-dependent). *Note (fixed):* step-level acceptance cannot be cleanly attributed to a single request class under window overlap at C≥32 (multiple classes' drafts are verified in the same step), so H2 is tested on per-request TPOT, which is directly measured per request.
- **H3 (price scaling):** drafter share `D = gpu_step_ms − gpu_fwd_ms` as a fraction of `gpu_step_ms` increases with batch pressure in the DSpark arm (C=32→96 slope > 1 percentage point per +16 running) while it does not increase in the AR arm (no drafter; D≈sampler/copy baseline).

## 3. Measurement design (fixed now)

**Environment (unchanged from H-OBS-1):** Qwen3-4B + `dspark_qwen3_4b_block7` (K=7), 1× RTX 3090 (SM86) on GPU1, vLLM V2 runner, async scheduling on, `--gpu-memory-utilization 0.90`, `--max-num-seqs 64`, `--max-num-batched-tokens 4096` (chunked prefill active), `--enable-prefix-caching`.

**Workload — heterogeneous, realistic (differs from H-OBS-1 on purpose):**
Four request classes, equal weight, round-robin per worker thread, deterministic seed **20261011** (same seed for all cells):

| class | ISL (tokens) | max_tokens | intent |
|---|---:|---:|---|
| SS | 50 | 256 | chat turn |
| SL | 50 | 1024 | short prompt / long generation |
| LS | 2048 | 256 | RAG: long context, short answer |
| LL | 2048 | 1024 | document QA, long response |

Prompts built from ShareGPT-V3 by concatenating shuffled turns and truncating to the exact ISL, tokenized **externally** with the served model's tokenizer (`add_special_tokens=False`), sent as integer-id arrays (same methodology as Phase 4.4 `hetero_client.py`). Sampling: `temperature=0`, streaming, **no `--ignore-eos`** (natural EOS — realistic OSL). Client duration **180 s**, warmup window **20 s**.

**Arms × pressure = 4 cells:**

| cell | arm | C (client threads) |
|---|---|---:|
| ar_c32 | AR (no speculator) | 32 |
| dspark_c32 | DSpark-K7 native (`VLLM_TB_MODE=native`) | 32 |
| ar_c96 | AR | 96 |
| dspark_c96 | DSpark-K7 native | 96 |

C=96 is intentionally above `max_num_seqs=64` to create a sustained waiting queue (saturation regime); C=32 is the mid-load regime. Fresh server per cell; port **8401**; GPU1; obs tracer ON in all cells (overhead ≤0.5% established at C≤32; C=96 overhead reported, not assumed).

**Recorded:**
- Per step (obs tracer, by `step_id`): `t_cpu_sched`, `running`, `waiting`, `scheduled_tokens`, `prefill_tokens` (genuine prompt work), `draft_tokens`, `sd_enabled`, `in_flight`, `gpu_step_ms`, `gpu_fwd_ms`.
- Per request (client): class, arrival t, ttft_ms, tpot_ms, e2e_s, ntok, isl.
- Prometheus counters @0.5 s: `vllm:spec_decode_num_drafts[_tokens]`, `vllm:spec_decode_num_accepted_tokens`, running/waiting/KV gauges, GPU util/mem/power.

**Validity rule (per cell):** server startup-complete marker in its OWN log; client exit 0 with ≥50% of expected requests completed and no error flood; steps.jsonl non-empty with ≥95% of steps carrying `gpu_step_ms`; DSpark cells scheduled >1000 draft tokens, AR cells exactly 0. Invalid cells are preserved, marked, and excluded from the verdict.

## 4. Analysis plan (fixed now)

Steady-state window: exclude first 20 s (client warmup) and the final 10% of steps (shutdown tail). Steps are temporally correlated → report per-cell distributions and cross-cell pressure dependence, not step-level CIs.

1. **Phase split:** prefill-heavy = `prefill_tokens ≥ 0.05 × scheduled_tokens`; pure decode = `prefill_tokens == 0`; remainder reported separately as mixed.
2. **H1:** A by phase (dspark cells, steady window); report mean ± per-step distribution and the step counts. Per-step acceptance comes from the Prometheus counters' per-interval deltas (`num_accepted_tokens_d / num_draft_tokens_d`) time-aligned to steps.jsonl by `t_cpu_sched`; where a 0.5 s bucket spans multiple steps, attribute proportionally to that bucket's scheduled `draft_tokens`. *Guard (fixed):* if either phase has <100 steps with `draft_tokens > 0` in a cell (e.g. the runner schedules no drafts on prefill-heavy steps), H1 for that cell is reported as INCONCLUSIVE with the draft-schedule pattern documented — not forced into a comparison.
3. **H2:** per-request-class TPOT and TTFT (client records, steady window) for each arm × C; DSpark/AR ratio per class within each pressure level; report the cross-class relative range of the ratio.
4. **H3:** drafter share D by arm × C (steady-window mean of `(gpu_step_ms − gpu_fwd_ms)/gpu_step_ms`, decode-only and all-steps); compare slopes across C within arm.
5. **Throughput/latency summary** per cell × class (steady window): out tok/s, TTFT p50/p95, TPOT p50/p95; DSpark-vs-AR paired comparison within each C.
6. **Overhead:** AR cells are the no-drafter baseline; report DSpark throughput delta vs AR at matched C (this is the SD price+payoff net, not tracer overhead — tracer overhead was established in H-OBS-1 and is separately reported here as dspark-c32's drafter-share stability).

## 5. Verdict rule and hard stops (fixed now)

Report each of H1/H2/H3 as **SUPPORTED / NOT SUPPORTED / INCONCLUSIVE** with the numbers above; overall verdict = the conjunction summary plus an explicit "opportunity surface" statement for Track A (where, if anywhere, a phase- or class-aware decision would have had signal).

Hard stops (stop and report a blocker): direct measurements cannot be correlated by step id; tracer overhead destroys async overlap (in_flight ≥ 2 fraction < 0.9); >1 cell invalid after one re-run attempt each; or the GPU is not free when the run starts (refuse, do not kill external work).

**Boundedness:** exactly 4 cells + up to 4 single re-runs of invalid cells. No additional arms, no workload edits, no post-hoc window changes.

## 6. Provenance

- Launcher: `scripts/run_hoobs3.sh` (worktree `/home/junior1/_dev/dspark-native-validation`, branch `track-b-live-sd`).
- Client: `experiments/hetero_client_qwen3.py` (this repo; adapted from Phase 4.4 `hetero_client.py` for Qwen3-4B + integer-id prompts).
- Analysis: `experiments/hoobs3_analyze.py` (this repo) → `results/hoobs3/hetero/analysis.json` + `analysis_table.md`.
- Raw artifacts under `results/hoobs3/hetero/<cell>/` (gitignored; local only): server log, client JSONL, metrics JSONL, obs trace, collector stats, config manifest (records worktree rev).
- H-OBS-1 and Track B results preserved verbatim; nothing here alters them.
