# Track B Step 1 — Feasibility audit: per-iteration K adaptation in vLLM 0.25.0 (native DSpark)

- **Date:** 2026-10-09 (KST)
- **Directive:** "Hermes Research Directive — Native DSpark Baseline and Next Research Milestone", §4 Step 1.
- **Method:** read-only source audit of the validated v0.25.0+cu126 build (`/home/junior1/_dev/dspark-native-validation`), cross-checked against live server logs from Gate A–E and a smoke run on 2026-10-09 (port 8400, GPU 1). No environment modification.
- **Question:** Can per-iteration K adaptation be done in vLLM 0.25.0 while preserving the DSpark learned drafter, correct target verification, required CUDA graphs, valid draft-prefix semantics, and correct request/batch accounting?

## 1. What v0.25.0 already contains (measured facts)

1. **A native dynamic-K mechanism exists but is V1-runner-only.** `SpeculativeConfig.num_speculative_tokens_per_batch_size` (`vllm/config/speculative.py:173`) defines `(range_start, range_end, K)` ranges; the scheduler builds a dense lookup `dynamic_sd_lookup[batch_size] -> K` (`vllm/v1/core/sched/scheduler.py:235-242`, `vllm/v1/spec_decode/dynamic/utils.py`) and emits `SchedulerOutput.num_spec_tokens_to_schedule` per step (`scheduler.py:1086-1088`). The V1 model runner passes it into every proposer's `propose(num_speculative_tokens=...)` call (`vllm/v1/worker/gpu_model_runner.py:4928-5038`), and the CG manager captures per-K decode graphs for it (`vllm/v1/worker/gpu/cudagraph_utils.py:200-226`, "When using Dynamic SD … capture graphs for all possible values").
2. **DSpark is V2-runner-only.** `VllmConfig.use_v2_model_runner` forces the V2 GPU model runner when `method == "dspark"` (`vllm/config/vllm.py:519-530`: "DSpark is implemented only by the V2 GPU model runner … force V2 for it").
3. **The V2 runner ignores `num_spec_tokens_to_schedule`.** Grep of `vllm/v1/worker/gpu/model_runner.py` and both speculators: zero references to `num_spec_tokens_to_schedule` or `dynamic_sd_lookup`. The field is computed by the shared scheduler but dead in the DSpark execution path.
4. **DSpark draft width is fixed at construction.** `DSparkSpeculator.__init__` sets `num_query_per_req = num_speculative_steps` (`vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:60-63`); FULL CUDA graphs are captured with `uniform_token_count=num_query_per_req` ("Every DFlash step has exactly num_query_per_req tokens, so we can use FULL CGs", `dflash/speculator.py:371-381`); draft buffer is `[max_num_reqs, num_speculative_steps]`.
5. **Rejection sampler and target-side bookkeeping are fixed-K.** `RejectionSampler.num_speculative_steps = spec_config.num_speculative_tokens` (`vllm/v1/worker/gpu/spec_decode/rejection_sampler.py:51`); V2 `decode_query_len = num_speculative_steps + num_new_sampled_tokens_per_step` is a constant used for CG capture, sampler sizing, and expansion bounds (`gpu/model_runner.py:318-346, 896`).
6. **The target-side K=0 path already exists and is the "common case".** When no drafts are scheduled, V2 runs a plain decode step (query_len 1 per request) through the existing `if not draft_tokens:` branch (`gpu/model_runner.py:867-879`) and plain sampler (`gpu/model_runner.py:1066`). This path is exercised every prefill step in normal operation.

## 2. Verdict per candidate action

| Action | Feasible in v0.25.0? | Blocker / note |
|---|---|---|
| Native K=7 (fixed) | YES — baseline, validated (Gates A–E) | — |
| Smaller fixed K (e.g., K=3 server) | YES — just a different `num_speculative_tokens` at server start | Not per-iteration; a separate-server diagnostic only |
| Per-iteration variable K (K∈{0,2,4,7} mid-run) | **NO — major architectural modification required** | Blockers B1–B3 below |
| Per-iteration binary SD on/off (K=7 ↔ no-draft step) | **YES — small surgical patch (~10 lines), both sides** | §3; preserves drafter, verification, accounting; mixed steps fall to PIECEWISE CG (quantifiable overhead) |
| Request-level router between AR and fixed-K servers | YES — no source change | Diagnostic only per directive: admission-granularity, not per-iteration; must be evaluated/described separately |

### Exact blockers for variable K (B1–B3)

- **B1 (worker):** V2 runner's speculator call takes no K argument; `DSparkSpeculator`/`DFlashSpeculator` draft exactly `num_query_per_req` tokens per request from init-time buffers, and FULL decode CGs exist only for that uniform token count. Variable K requires multi-K graph capture or PIECEWISE fallback logic in the drafter path.
- **B2 (verification):** `RejectionSampler` buffers are sized `[num_reqs, num_speculative_steps+1]`; variable per-step K requires variable-width verification (masking/padding) and a corresponding `decode_query_len` generalization.
- **B3 (scheduler→worker channel):** the existing channel (`SchedulerOutput.num_spec_tokens_to_schedule`) is not consumed by V2; wiring it through would still leave B1/B2 in place.

Per the directive: *"If the runtime cannot support valid dynamic K without major architectural modifications, document the exact blocker."* — documented here. No porting of newer adaptive-verification internals is attempted (hardware-blocked per `docs/dspark-native-audit.md` §4).

## 3. Feasible controller surface (for Step 2 preregistration)

**Per-step binary SD on/off with the native DSpark drafter**, via a minimal patch to the isolated v0.25.0 build:

1. **Scheduler side (engine core, CPU):** add `sd_enabled: bool` to `SchedulerOutput`; when disabled, discard pending drafts in `update_draft_token_ids` (`scheduler.py:1948`) so no spec tokens are scheduled next step. Decision inputs available *measured* at the scheduler: `len(self.running)`, `len(self.waiting)`, `kv_cache_manager.usage`, preemption counters, per-request draft/accept history (spec stats), token-budget consumption, prefill share of the current batch.
2. **Worker side (V2 runner):** skip `self.speculator.propose()` (`gpu/model_runner.py:1459`) when `sd_enabled` is false — this is what actually removes the draft-pass cost (the Phase 5.1 lesson: skipping must happen in the execution path, not just bookkeeping).
3. **Correctness consequences (analyzed):**
   - One-step lag: an OFF decision at step t stops *new* drafts; drafts already scheduled from step t−1 are still verified at step t (standard draft-prefix semantics preserved).
   - Mixed steps (some requests with pending drafts, controller off) run through the existing non-uniform/PIECEWISE path — correct, overhead to be measured.
   - K=0 steps are pure decode via the existing "common case" branch — no new verification code.
   - **Causal verification (directive requirement):** log per step `(sd_enabled, num_draft_tokens_scheduled, spec_decode counter deltas)`; assert OFF-steps ⇒ 0 new draft tokens and ON-steps ⇒ K=7 drafts. A decision that doesn't move these counters is a harness bug, not a null result.
4. **What this is NOT:** it is not per-iteration variable K (blocked, §2), not native adaptive verification (not executed on this hardware — do not claim parity), and the router baseline (§2 row 5) must be reported separately if used.

## 4. Risks / open items for Step 2

- Residual overhead of an OFF step vs true AR: to be measured, not assumed (Phase 5.1 showed a ~9% spec-path floor in the *old* ngram runtime; v0.25.0 V2's K=0 path is architecturally plain decode, but this must be verified on this build).
- Decision-rule parameters must be set by calibration against measured state (no hard-coded concurrency thresholds per directive); success/falsification thresholds fixed before runs.
- The patch applies to the isolated worktree only, and only when no experiment is using it (safety constraint).
