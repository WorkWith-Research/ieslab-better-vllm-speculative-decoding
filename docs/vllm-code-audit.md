# vLLM 0.19.1 Speculative-Decoding Scheduling — Code Audit

All paths are relative to `.venv/lib/python3.12/site-packages/vllm/`. Line numbers verified against the installed source on 2026-10-07.

**Runner selection (critical context for items 2 & 5).** `gpu_worker.py:153` sets
`self.use_v2_model_runner = envs.VLLM_USE_V2_MODEL_RUNNER`; `envs.py:239` / `envs.py:1605-1606`
default it to **False**. So the *default* serving path is the **legacy** runner
(`v1/worker/gpu_model_runner.py`, "V1") with proposers in `v1/spec_decode/*.py`. The **new**
runner (`v1/worker/gpu/model_runner.py`, "V2") + `EagleSpeculator` is opt-in via
`VLLM_USE_V2_MODEL_RUNNER=1` (`gpu_worker.py:296-310`). Both are audited below; the scheduler
and KV-cache manager are shared by both.

---

## 1. Lookahead-KV allocation path for spec decode

The scheduler reserves **K extra KV slots per step** (beyond the tokens actually computed) so a
KV-caching proposer (EAGLE / draft model) can write its draft tokens' KV in the same step without
needing fresh blocks. This is the `num_lookahead_tokens` argument threaded from the scheduler into
`KVCacheManager.allocate_slots`.

| # | File : line | Function | Evidence |
|---|-------------|----------|----------|
| 1a | `v1/core/sched/scheduler.py:213-222` | `Scheduler.__init__` | Reads K once at init. `self.num_spec_tokens = self.num_lookahead_tokens = 0`; if `speculative_config`: `self.num_spec_tokens = speculative_config.num_speculative_tokens` (L217); **only** for `use_eagle()` (L218-220) or `uses_draft_model()` (L221-222) does it set `self.num_lookahead_tokens = self.num_spec_tokens`. **ngram / suffix / medusa leave lookahead = 0.** |
| 1b | `v1/core/sched/scheduler.py:463-467` | `schedule()` (running reqs) | `new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens, num_lookahead_tokens=self.num_lookahead_tokens)` — the per-step lookahead for a running/decode request. |
| 1c | `v1/core/sched/scheduler.py:716-718` | `schedule()` (waiting reqs) | `effective_lookahead_tokens = 0 if request.num_computed_tokens == 0 else self.num_lookahead_tokens`. Comment L711-715: P/D-disaggregation edge case. A **fresh** prefill (num_computed==0) reserves **no** lookahead; a resumed/chunked req reserves K. |
| 1d | `v1/core/sched/scheduler.py:746-755` | `schedule()` (waiting reqs) | `self.kv_cache_manager.allocate_slots(request, num_new_tokens, ..., num_lookahead_tokens=effective_lookahead_tokens, ...)` |
| 1e | `v1/core/kv_cache_manager.py:257-267` | `KVCacheManager.allocate_slots` | Signature: `allocate_slots(self, request, num_new_tokens, num_new_computed_tokens=0, new_computed_blocks=None, num_lookahead_tokens: int = 0, ...)`. Docstring L277-279: "The number of speculative tokens to allocate. This is used by spec decode proposers with kv-cache such as eagle." |
| 1f | `v1/core/kv_cache_manager.py:319-321` | (docstring) | Block-layout legend: `new = num_new_tokens, including unverified draft tokens`; `lookahead = num_lookahead_tokens`. |
| 1g | `v1/core/kv_cache_manager.py:361-365` | `allocate_slots` | **The actual reservation math:** `num_tokens_main_model = total_computed_tokens + num_new_tokens`; `num_tokens_need_slot = min(num_tokens_main_model + num_lookahead_tokens, self.max_model_len)`. Lookahead is added on top of the tokens-to-compute. |
| 1h | `v1/core/kv_cache_manager.py:377-409` | `allocate_slots` | `get_num_blocks_to_allocate(num_tokens=num_tokens_need_slot, ...)` (L377-385) and `coordinator.allocate_new_blocks(request_id, num_tokens_need_slot, num_tokens_main_model, ...)` (L404-409). Only verified tokens are cached: `num_tokens_to_cache = min(total_computed + num_new_tokens, request.num_tokens)` (L421-425). |

**Net effect per running decode step under SD:** the request's `num_new_tokens` already includes
the K drafts from the previous step (see item 3 / `num_tokens_with_spec`), and then **K more slots**
are reserved as lookahead so EAGLE can emit its next K draft tokens' KV in-step. For a non-KV
proposer (ngram/suffix) lookahead is 0 and only the scheduled drafts themselves occupy slots.

---

## 2. How draft tokens from step t become scheduled input for step t+1

The loop is: **worker proposes → RPC back to engine core → scheduler stores on request → next
`schedule()` promotes into `scheduled_spec_decode_tokens` → worker reads them as input ids →
rejection sampling → accepted tokens appended to the request token queue.**

### 2a. Worker generates drafts (end of step t)

| # | File : line | Function | Evidence |
|---|-------------|----------|----------|
| 2a1 | `v1/worker/gpu/model_runner.py:1204-1222` | V2 `GPUModelRunner.execute_model` (tail) | `draft_tokens = self.speculator.propose(input_batch, attn_metadata, slot_mappings_by_layer, hidden_states, aux_hidden_states, num_sampled, num_rejected, last_sampled, next_prefill_tokens, temperature, seeds, ...)`; then `self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens` (L1221) and `self.draft_tokens_handler.set_draft_tokens(input_batch, draft_tokens)` (L1222). |
| 2a2 | `v1/worker/gpu/spec_decode/eagle/speculator.py:319-474` | V2 `EagleSpeculator.propose` | Runs the EAGLE model once for the "prefill"/first-token pass (L374-380, eager), samples draft token 0 via `gumbel_sample` (L400-410), then generates drafts 1..K-1 in a loop (`generate_draft`, L203 `for step in range(1, self.num_speculative_steps)`). Returns `self.draft_tokens[:num_reqs]` (L474), shape `[num_reqs, K]`. |
| 2a3 | `v1/worker/gpu/model_runner.py:1228-1229` | V2 `take_draft_token_ids` | `return self.draft_tokens_handler.get_draft_tokens()` → a `DraftTokenIds(req_ids, draft_token_ids)`. |
| 2a4 | `v1/worker/gpu_model_runner.py:4206-4230` | **Legacy** `GPUModelRunner.execute_model` (tail) | `input_fits_in_drafter = ... max_seq_len + self.num_spec_tokens <= effective_drafter_max_model_len`; for EAGLE/draft-model calls `propose_draft_token_ids(sampled_token_ids)` (L4228) or `self.drafter.prepare_next_token_ids_padded(...)`; the drafter object is set at L513-562 (`EagleProposer`, `NgramProposer`, `DraftModelProposer`, `SuffixDecodingProposer`, ...). |
| 2a5 | `v1/spec_decode/eagle.py:481-483, 543` | **Legacy** `EagleProposer.propose` | Early exit `if self.num_speculative_tokens == 1 or self.parallel_drafting`; otherwise `for token_index in range(self.num_speculative_tokens - 1):` generates the remaining drafts. |

### 2b. Drafts travel back to the engine core (scheduler process)

| # | File : line | Function | Evidence |
|---|-------------|----------|----------|
| 2b1 | `v1/executor/uniproc_executor.py:127-128`, `multiproc_executor.py:323-326`, `abstract.py:241-242` | executor RPC | `take_draft_token_ids()` → `collective_rpc("take_draft_token_ids")`. |
| 2b2 | `v1/engine/core.py:415-419` | **Sync** path, `EngineCore.post_step` | `if not self.async_scheduling and self.use_spec_decode and model_executed: draft_token_ids = self.model_executor.take_draft_token_ids(); if ...: self.scheduler.update_draft_token_ids(draft_token_ids)`. |
| 2b3 | `v1/engine/core.py:520-528` | **Async / deferred** path, `step_with_batch_queue` | For structured-output deferral: `draft_token_ids = self.model_executor.take_draft_token_ids(); ...; self.scheduler.update_draft_token_ids_in_output(draft_token_ids, deferred_scheduler_output)`. |

### 2c. Scheduler stores drafts on the request (still not scheduled yet)

| # | File : line | Function | Evidence |
|---|-------------|----------|----------|
| 2c1 | `v1/core/sched/scheduler.py:1669-1689` | `update_draft_token_ids` | For each `(req_id, spec_token_ids)`: **skips prefill chunks** (L1679-1683 "Ignore draft tokens for prefill chunks" → clears `spec_token_ids`); else `request.spec_token_ids = spec_token_ids` (L1689). This is where step-t drafts land on the request. |
| 2c2 | `v1/core/sched/scheduler.py:1691-1727` | `update_draft_token_ids_in_output` | Async variant: trims drafts to the scheduled count (L1710-1713 "Trim drafts to scheduled number of spec tokens (needed for chunked prefill case)"), grammar-validates, pads invalid with `-1` (L1720-1722), writes back into `scheduler_output.scheduled_spec_decode_tokens`. |

### 2d. Next step: scheduler promotes stored drafts into the batch

| # | File : line | Function | Evidence |
|---|-------------|----------|----------|
| 2d1 | `v1/request.py:230-231` | `Request.num_tokens_with_spec` | `return len(self._all_token_ids) + len(self.spec_token_ids)` — the drafts count toward how many tokens the scheduler must "catch up" to. |
| 2d2 | `v1/core/sched/scheduler.py:404-417` | `schedule()` (running reqs) | `num_new_tokens = request.num_tokens_with_spec + request.num_output_placeholders - request.num_computed_tokens`; capped by token budget (L411) and `max_model_len - 1 - num_computed_tokens` (L415-417, "necessary when using spec decoding"). |
| 2d3 | `v1/core/sched/scheduler.py:520-536` | `schedule()` (running reqs) | **The promotion:** `if request.spec_token_ids:` → compute `num_scheduled_spec_tokens = num_new_tokens + num_computed_tokens - num_tokens - placeholders`; slice `spec_token_ids[:num_scheduled_spec_tokens]`; `scheduled_spec_decode_tokens[request.request_id] = spec_token_ids` (L532); then `request.spec_token_ids = []` (L536). |

### 2e. Worker consumes the scheduled drafts as input ids + rejection sampling

| # | File : line | Function | Evidence |
|---|-------------|----------|----------|
| 2e1 | `v1/worker/gpu/model_runner.py:683-709` | V2 `_prepare_inputs` (draft section) | `draft_tokens = scheduler_output.scheduled_spec_decode_tokens`; per-req `num_draft_tokens` from `len(draft_tokens.get(req_id, ()))` (L697-701); `total_num_logits = num_reqs + total_num_draft_tokens`; builds `cu_num_logits`. |
| 2e2 | `v1/worker/gpu/model_runner.py:766-776` | V2 `_prepare_inputs` | `logits_indices = combine_sampled_and_draft_tokens(input_ids, idx_mapping, last_sampled_tokens, query_start_loc, seq_lens, prefill_len, self.req_states.draft_tokens, cu_num_logits, total_num_logits)` — reads the stored draft tokens into the input-id buffer at the draft positions. |
| 2e3 | `v1/worker/gpu/input_batch.py:324` / `:391` / `:465` | V2 kernels | `combine_sampled_and_draft_tokens`, `get_num_sampled_and_rejected`, `post_update`. |
| 2e4 | `v1/worker/gpu/model_runner.py:847-860` | V2 `_sample` | `if input_batch.num_draft_tokens == 0: sampler_output = self.sampler(...)` else `sampler_output = self.rejection_sampler(logits, input_batch, self.speculator.draft_logits)`. |
| 2e5 | `v1/worker/gpu/spec_decode/rejection_sampler.py:508-584` | V2 `RejectionSampler.__call__` | strict / probabilistic / synthetic. Returns `sampled_token_ids` = `[accepted..., bonus]` (variable length per req, up to K+1) and `num_sampled`. |
| 2e6 | `v1/worker/gpu_model_runner.py:4167` | **Legacy** `_sample` | `sampler_output = self._sample(logits, spec_decode_metadata)` — legacy rejection path driven by `spec_decode_metadata` (built in `_calc_spec_decode_metadata`, L2582). |

### 2f. Accepted tokens appended to the request's token queue

| # | File : line | Function | Evidence |
|---|-------------|----------|----------|
| 2f1 | `v1/core/sched/scheduler.py:1362-1390` | `update_from_output` | `generated_token_ids = sampled_token_ids[req_index]`; if scheduled drafts exist: `num_draft_tokens = len(scheduled_spec_token_ids)`, `num_accepted = len(generated_token_ids) - 1`, `num_rejected = num_draft_tokens - num_accepted`; `request.num_computed_tokens -= num_rejected` (L1379); records spec stats. |
| 2f2 | `v1/core/sched/scheduler.py:1627-1643` | `_update_request_with_output` | **The append:** `for num_new, output_token_id in enumerate(new_token_ids, 1): request.append_output_token_ids(output_token_id)` (L1635) then `check_stop`. This is where accepted draft tokens (+ bonus) enter the request's permanent token list. |
| 2f3 | `v1/request.py:204-214` | `Request.append_output_token_ids` | Extends `_output_token_ids` and `_all_token_ids`; `update_block_hashes()`. |
| 2f4 | `v1/worker/gpu/input_batch.py:465` | V2 `post_update` (GPU side) | Advances `req_states.num_computed_tokens.gpu`, `last_sampled_tokens`, and appends into `all_token_ids.gpu` / `total_len.gpu` on-device (mirrors 2f1/2f2 for the GPU request state). |

---

## 3. Where per-request K is read at runtime

**Confirmed:** K is a **single scalar constant** sourced from
`speculative_config.num_speculative_tokens`, captured once at init in every component. There is no
per-request K anywhere today. The scheduler's *scheduled* draft count is already effectively
per-request (`len(request.spec_token_ids)`, item 2d3), but the **lookahead allocation**, the
**worker draft-generation depth**, and all **GPU buffer shapes / CUDA-graph sizing** are hard-wired
to the scalar K.

`SpeculativeConfig.num_speculative_tokens` is defined at `config/speculative.py:71`
(`Field(default=None, gt=0)`). The VllmConfig accessor is `config/vllm.py:447-452`.

### 3a. Scheduler (shared by both runners)

| File : line | Binding | Used as |
|-------------|---------|---------|
| `v1/core/sched/scheduler.py:217` | `self.num_spec_tokens = speculative_config.num_speculative_tokens` | spec stats (`make_spec_decoding_stats`, L1988) |
| `v1/core/sched/scheduler.py:220, 222` | `self.num_lookahead_tokens = self.num_spec_tokens` (eagle / draft_model only) | **lookahead KV allocation** (item 1) — *the scalar that must become a per-request vector* |

### 3b. V2 runner + EagleSpeculator (opt-in)

| File : line | Binding | Used as |
|-------------|---------|---------|
| `v1/worker/gpu/model_runner.py:167` | `self.num_speculative_steps = self.speculative_config.num_speculative_tokens` | decode query len, sampler, expand |
| `v1/worker/gpu/model_runner.py:214` | `num_speculative_tokens=self.num_speculative_steps + 1` | `Sampler` max spec len |
| `v1/worker/gpu/model_runner.py:223` | `max_num_logits=self.max_num_reqs * (self.num_speculative_steps + 1)` | structured-outputs buffer |
| `v1/worker/gpu/model_runner.py:229` | `self.decode_query_len = self.num_speculative_steps + 1` | **CUDA-graph decode sizing** |
| `v1/worker/gpu/model_runner.py:711` | `max_expand_len = self.num_speculative_steps + 1` | `expand_idx_mapping` cap |
| `v1/worker/gpu/spec_decode/eagle/speculator.py:43` | `self.num_speculative_steps = self.speculative_config.num_speculative_tokens` | draft depth |
| `v1/worker/gpu/spec_decode/eagle/speculator.py:76-81` | `self.draft_tokens = torch.zeros(self.max_num_reqs, self.num_speculative_steps, ...)` | **GPU buffer shape [max_reqs, K]** |
| `v1/worker/gpu/spec_decode/eagle/speculator.py:93-99` | `self.draft_logits = torch.zeros(self.max_num_reqs, self.num_speculative_steps, self.vocab_size, ...)` | probabilistic-rejection buffer |
| `v1/worker/gpu/spec_decode/eagle/speculator.py:203, 231` | `for step in range(1, self.num_speculative_steps)` / `if step < self.num_speculative_steps - 1` | **draft-generation loop** |
| `v1/worker/gpu/spec_decode/eagle/speculator.py:306, 412` | `if self.num_speculative_steps == 1:` early exits | capture / propose fast path |
| `v1/worker/gpu/spec_decode/rejection_sampler.py:456` | `self.num_speculative_steps = spec_config.num_speculative_tokens` | passed to strict (L526), probabilistic (L550), synthetic (L572) kernels; synthetic params computed with it (L468-472) |
| `v1/worker/gpu/spec_decode/eagle/cudagraph.py:35` | `super().__init__(..., decode_query_len=1)` | EAGLE *internal* per-step graph uses query_len 1 (each draft step is one token/req) — independent of K |

### 3c. Legacy runner + proposers (default path)

| File : line | Binding | Used as |
|-------------|---------|---------|
| `v1/worker/gpu_model_runner.py:576` | `self.num_spec_tokens = self.speculative_config.num_speculative_tokens` | drafter sizing, draft buffer |
| `v1/worker/gpu_model_runner.py:775` | `self.uniform_decode_query_len = 1 + self.num_spec_tokens` | **CUDA-graph decode sizing** (see item 5c) |
| `v1/worker/gpu_model_runner.py:830-840` | `self.draft_token_ids_cpu = torch.empty((self.max_num_reqs, self.num_spec_tokens), ...)` | **CPU draft buffer shape [max_reqs, K]** |
| `v1/worker/gpu_model_runner.py:4207` | `spec_decode_common_attn_metadata.max_seq_len + self.num_spec_tokens <= effective_drafter_max_model_len` | drafter fit check |
| `v1/spec_decode/eagle.py:78` | `self.num_speculative_tokens = self.speculative_config.num_speculative_tokens` | draft depth |
| `v1/spec_decode/eagle.py:88-94` | `extra_slots_per_request = 1 if not parallel_drafting else num_speculative_tokens` | input-slot accounting |
| `v1/spec_decode/eagle.py:481, 483, 543, 1497` | `== 1` early exit; `range(self.num_speculative_tokens - 1)` loop; capture `range(... if not is_graph_capturing else 1)` | draft-generation loop + capture |
| `v1/spec_decode/ngram_proposer.py:25, 31` | `self.k = ...num_speculative_tokens`; `valid_ngram_draft = np.zeros((max_num_seqs, self.k))` | ngram depth + buffer |
| `v1/spec_decode/ngram_proposer_gpu.py:41, 249` | `self.k = ...num_speculative_tokens` | GPU ngram depth |
| `v1/spec_decode/suffix_decoding.py:19, 83` | `self.num_speculative_tokens = config.num_speculative_tokens`; `min(self.num_speculative_tokens, max_model_len - num_tokens - 1)` | suffix tree depth (already capped per-request) |

### 3d. CUDA-graph / compile-time sizing (K baked into capture shapes)

| File : line | Binding | Used as |
|-------------|---------|---------|
| `v1/cudagraph_dispatcher.py:37-41` | `self.uniform_decode_query_len = 1 if not spec_config else 1 + spec_config.num_speculative_tokens` | **the core uniform-decode assumption**; drives `_create_padded_batch_descriptor` (L139-155) and FULL-graph key set (L206-230) |
| `v1/worker/gpu/cudagraph_utils.py:86-135` | `decode_query_len` ctor arg; `max_decode_tokens = max_num_reqs * decode_query_len`; dispatch asserts `num_tokens % decode_query_len == 0` (L129, L134) | FULL-graph batch descriptor |
| `config/compilation.py:1238-1283` | `adjust_cudagraph_sizes_for_spec_decode(uniform_decode_query_len, tp)` | rounds every capture size up to a multiple of `1 + K` (L1261-1267); raises if none fit (L1273-1280) |
| `config/vllm.py:1393-1400` | `decode_query_len = 1; ... decode_query_len += spec_config.num_speculative_tokens`; used to size `max_num_batched_tokens` (`... * decode_query_len * 2, 512`) | init-time budget sizing |

### 3e. Attention backends / structured output / metrics (secondary K consumers)

| File : line | Binding |
|-------------|---------|
| `v1/attention/backend.py:468, 523-528` | "decodes are 1 + num_speculative_tokens"; spec-decode query-len math |
| `v1/attention/backends/flashinfer.py:561` | `speculative_config.num_speculative_tokens` |
| `v1/attention/backends/gdn_attn.py:87-88`, `mamba_attn.py:97` | `self.num_spec = ...num_speculative_tokens` (hybrid models) |
| `v1/attention/backends/mla/indexer.py:275-281, 441` | `next_n = num_speculative_tokens + 1`; reorder threshold |
| `v1/structured_output/backend_xgrammar.py:71-74` | `max_rollback_tokens = num_speculative_tokens` |
| `v1/structured_output/__init__.py:199`, `backend_outlines.py:86`, `backend_lm_format_enforcer.py:121` | K for grammar rollback sizing |
| `v1/spec_decode/metrics.py:183` | `speculative_config.num_speculative_tokens` (per-position histogram width) |

**Sites that must change for a per-request K vector** (the ones that assume uniformity): the
scheduler lookahead scalar (3a), all GPU/CPU draft buffers shaped `[max_reqs, K]` (3b L76-99, 3c
L830-840), the EAGLE/ngram draft-generation loops `range(K)` (3b L203, 3c L543/1497), the rejection
sampler scalar `num_speculative_steps` (3b L456 → per-req in kernels), and every CUDA-graph sizing
site that uses `uniform_decode_query_len = 1 + K` (3d). The scheduler's *scheduled* draft count is
already per-request and needs no change.

---

## 4. Chunked-prefill + spec-decode interaction

**Token budget.** `token_budget = self.max_num_scheduled_tokens` (`scheduler.py:367`), where
`max_num_scheduled_tokens = scheduler_config.max_num_scheduled_tokens or max_num_batched_tokens`
(`scheduler.py:106-110`). This is the `--max-num-batched-tokens` budget.

**Draft tokens ARE counted against the chunk budget for running requests.** For a running/decode
request, `num_new_tokens = num_tokens_with_spec + placeholders - num_computed_tokens`
(`scheduler.py:404-408`) — and `num_tokens_with_spec` includes `len(spec_token_ids)` (the K drafts,
`request.py:231`). That `num_new_tokens` is then capped by the token budget (`scheduler.py:411`) and
subtracted from it (`token_budget -= num_new_tokens`, L517). So **yes, draft tokens consume the
`--max-num-batched-tokens` budget.** The *lookahead* KV slots (item 1) are **not** part of the token
budget — they are allocated separately via `allocate_slots(..., num_lookahead_tokens=...)` and only
affect block availability.

| # | File : line | Function | Evidence / behavior |
|---|-------------|----------|---------------------|
| 4a | `v1/core/sched/scheduler.py:660-681` | `schedule()` (waiting/prefill reqs) | `num_new_tokens = request.num_tokens - num_computed_tokens` (L665); capped by `long_prefill_token_threshold` (L666-668) and `min(num_new_tokens, token_budget)` (L680). A prefill chunk is bounded purely by the token budget — **no drafts are added to a prefill chunk's compute**. |
| 4b | `v1/core/sched/scheduler.py:716-718` | `schedule()` (waiting reqs) | `effective_lookahead_tokens = 0 if request.num_computed_tokens == 0 else self.num_lookahead_tokens`. A **first** prefill chunk reserves **no** lookahead KV; a **resumed / later** chunk of the same request reserves K. |
| 4c | `v1/core/sched/scheduler.py:992-994` | `_update_after_schedule` | `request.is_prefill_chunk = request.num_computed_tokens < (request.num_tokens + request.num_output_placeholders)` — the flag that gates draft scheduling for chunked prefills. |
| 4d | `v1/core/sched/scheduler.py:1679-1683` | `update_draft_token_ids` | **Special handling:** "Ignore draft tokens for prefill chunks" → if `request.is_prefill_chunk`, clears `spec_token_ids` and skips. So a request still mid-prefill never gets drafts scheduled, even though EAGLE ran on its chunk. |
| 4e | `v1/core/sched/scheduler.py:1710-1713` | `update_draft_token_ids_in_output` | "Trim drafts to scheduled number of spec tokens (needed for chunked prefill case for example)" — `del spec_token_ids[orig_num_spec_tokens:]`. Handles the case where fewer than K drafts were actually scheduled alongside a chunk. |
| 4f | `v1/worker/gpu/spec_decode/eagle/speculator.py:372-380` | V2 `EagleSpeculator.propose` | "Prefill: Run the eagle speculator with eager mode. **TODO(woosuk): Support CUDA graph for prefill.**" — EAGLE always runs the first (prefill) pass eagerly; only the per-step decode draft loop uses FULL graphs. |
| 4g | `v1/worker/gpu/spec_decode/eagle/speculator.py:506-509` | `_prepare_eagle_inputs_kernel` (Triton) | "Chunked prefilling. Get the next prefill token." — when `num_sampled == 0` (a pure prefill chunk), the EAGLE input's next token is taken from `next_prefill_tokens`, not a sampled token. |

**Summary of the interaction:** a prefill chunk scheduled in the same step as decode requests
(a) draws only its own prompt tokens from the token budget — no drafts are attached to it; (b)
reserves K lookahead KV slots **only if** it is a resumed/later chunk (`num_computed_tokens > 0`);
(c) still runs the EAGLE model eagerly over the chunk and may emit draft tokens, but those drafts
are discarded by `update_draft_token_ids` while `is_prefill_chunk` is true. Decode requests in the
same batch carry their K drafts (counted against the budget) plus K lookahead KV slots.

---

## 5. Minimal patch surface for a dynamic-K controller

Goal: (a) inject per-request K each step from outside, (b) adjust lookahead allocation accordingly,
(c) keep CUDA graphs working. The scheduler's *scheduled* draft count is already per-request
(`len(spec_token_ids)`), so the real work is in **lookahead**, **worker draft depth/buffers**, and
**CUDA-graph sizing**.

### 5a. Inject per-request K each step (from outside)

| # | File : line | Change |
|---|-------------|--------|
| A1 | `v1/request.py` (near `spec_token_ids`, ~L230) | Add a per-request field, e.g. `self.spec_len: int = 0` (default to config K). This is the single source of truth the scheduler reads each step. |
| A2 | `v1/core/sched/interface.py:103-113` (Scheduler interface) + `v1/core/sched/scheduler.py` | Add a method, e.g. `def set_spec_lengths(self, spec_lens: dict[str, int]) -> None`, that writes `request.spec_len` for the running/waiting requests. Keep it O(n). |
| A3 | `v1/engine/core.py:449` (and `:380`-ish sync path) | In `step_with_batch_queue` / the sync step, **before** `self.scheduler.schedule()`, call `self.scheduler.set_spec_lengths(controller_decisions)` where `controller_decisions` is produced by your lightweight model over `SchedulerStats` (already built each step — see below). This is the external injection point; it runs in the engine-core process where the controller lives. |
| A4 | `v1/core/sched/scheduler.py:215-222` | Keep the init-time scalar as a **default/upper bound** (`self.num_spec_tokens_max`), but stop using `self.num_lookahead_tokens` as a fixed value in `schedule()` — read `request.spec_len` per request instead. |

> Note: because the scheduler already schedules `len(request.spec_token_ids)` drafts per request
> (item 2d3), you do **not** need to change how many drafts get scheduled — only what K the *worker*
> is told to draft and what lookahead KV is reserved. The cleanest contract: controller sets
> `request.spec_len = K_req` each step; worker drafts up to `K_req`; scheduler reserves `K_req`
> lookahead.

### 5b. Adjust lookahead allocation accordingly

| # | File : line | Change |
|---|-------------|--------|
| B1 | `v1/core/sched/scheduler.py:463-467` | Replace `num_lookahead_tokens=self.num_lookahead_tokens` with `num_lookahead_tokens=request.spec_len` (running reqs). |
| B2 | `v1/core/sched/scheduler.py:716-718, 746-755` | Replace the scalar in `effective_lookahead_tokens` with `request.spec_len` (waiting/resumed reqs); keep the `num_computed_tokens == 0 → 0` rule. |
| B3 | `v1/core/kv_cache_manager.py:257-409` | **No change required.** `allocate_slots` already takes `num_lookahead_tokens` as a per-call argument and does all the math generically (L361-365, L404-409). Passing a per-request value "just works." |

This is the cheapest part of the whole feature: the KV layer is already parameterized.

### 5c. Keep CUDA graphs working

The hard constraint is `uniform_decode_query_len = 1 + K` (`v1/cudagraph_dispatcher.py:37-41`,
`gpu_model_runner.py:775`, `config/compilation.py:1238-1283`). FULL decode graphs are captured at
token-counts that are exact multiples of `1 + K`, and the dispatcher asserts
`num_tokens % uniform_decode_query_len == 0` (`v1/worker/gpu/cudagraph_utils.py:129,134`;
`cudagraph_dispatcher.py:144`). Two viable options:

**Option A — enforce-eager fallback (simplest, safe).** When K varies per step, force the affected
steps to `CUDAGraphMode.NONE`.
- Anchor: `v1/cudagraph_dispatcher.py:273-280` — `dispatch()` already returns `(NONE, BatchDescriptor(num_tokens))` when no captured key matches. If you make the runtime batch non-uniform (per-req K), it naturally falls through to NONE.
- Cheapest trigger: set `compilation_config.cudagraph_mode = NONE` (or `enforce_eager=True`) whenever the controller emits a step whose per-request K is not all-equal. Cost: those steps run eager (no graph replay) — acceptable if dynamic-K steps are a minority.
- No buffer-shape changes needed beyond sizing buffers to `K_max`.

**Option B — pad every decode request to K_max (keeps graphs, recommended).** Capture at
`uniform_decode_query_len = 1 + K_max` and **pad each request's drafts to K_max** so the batch stays
uniform. This reuses existing padding logic.
- Anchor 1: `v1/cudagraph_dispatcher.py:37-41` — set `uniform_decode_query_len = 1 + K_max` (K_max = controller's maximum allowed K, a fixed init constant). All capture shapes (`config/compilation.py:1261-1267`) then stay valid.
- Anchor 2: `v1/core/sched/scheduler.py:520-536` — when promoting drafts, **pad** `spec_token_ids` to length K_max (fill with `-1`) so every decode request contributes exactly K_max draft slots; the rejection sampler already tolerates `-1`/invalid drafts (`scheduler.py:1720-1722` does exactly this padding today).
- Anchor 3: worker buffers sized to K_max — `v1/worker/gpu/spec_decode/eagle/speculator.py:76-99`, `v1/worker/gpu_model_runner.py:830-840`; EAGLE draft loop runs K_max steps (`speculator.py:203` / `eagle.py:543`) but the extra steps for a request with smaller K produce discarded drafts (cheap, and already how padding behaves).
- Result: `num_tokens = num_reqs * (1 + K_max)` is always a multiple of `uniform_decode_query_len`, so FULL/PIECEWISE dispatch (`cudagraph_dispatcher.py:306-323`) keeps hitting captured graphs.

**Shared prerequisites for both options:**
| # | File : line | Change |
|---|-------------|--------|
| C1 | `v1/worker/gpu/spec_decode/eagle/speculator.py:76-99`, `v1/worker/gpu_model_runner.py:830-840` | Size all `[max_num_reqs, K]` draft/draft-logit buffers to **K_max** (a fixed init constant = controller's max K). |
| C2 | `v1/worker/gpu/spec_decode/eagle/speculator.py:203, 412` and `v1/spec_decode/eagle.py:543, 1497` | Draft-generation loops iterate to K_max; per-request effective K is enforced by masking/padding (Option B) or by the rejection sampler's variable `num_sampled` (both). |
| C3 | `v1/worker/gpu/spec_decode/rejection_sampler.py:456, 526, 550, 572` | The kernels take a scalar `num_speculative_steps`; under Option B keep it = K_max (uniform) so no kernel change is needed. Under a fully variable layout you'd need per-req K in the kernels — avoid by choosing B. |

---

## 6. TODO / FIXME / known-limitation comments about dynamic spec length

| # | File : line | Comment | Relevance |
|---|-------------|---------|-----------|
| 6a | `config/speculative.py:627-634` | "Suffix decoding decides the actual number of speculative tokens **dynamically** and treats num_speculative_tokens as a **maximum limit**." → defaults `num_speculative_tokens = suffix_decoding_max_tree_depth`. | **The only place vLLM already varies draft length per step/per request.** Suffix decoding is the in-tree precedent for dynamic K: it caps per-request depth at runtime (`v1/spec_decode/suffix_decoding.py:83` `min(num_speculative_tokens, max_model_len - num_tokens - 1)`). |
| 6b | `v1/spec_decode/suffix_decoding.py:83` | `min(self.num_speculative_tokens, self.max_model_len - num_tokens - 1)` | Per-request dynamic cap already implemented for suffix. |
| 6c | `v1/worker/gpu/spec_decode/eagle/speculator.py:373` | "TODO(woosuk): Support CUDA graph for prefill." | EAGLE's first (prefill) pass is eager; only relevant to item 4, not dynamic K. |
| 6d | `v1/spec_decode/eagle.py:1494-1497` | "FIXME: when using tree-based specdec, adjust number of forward-passes according to the depth of the tree." (loop is `range(self.num_speculative_tokens if not is_graph_capturing else 1)`) | Confirms draft count is a fixed scalar even for tree decoders; a per-request K would need this loop to be per-request. |
| 6e | `v1/spec_decode/ngram_proposer.py:46-48` | "TODO(ekagra-ranjan): bump up the cap from 1 to 8 when TP parallelization for ngram is implemented." | Unrelated to dynamic K (threading), listed for completeness. |
| 6f | `v1/attention/backend.py:423` | "TODO(lucas): remove once we have FULL-CG spec-decode support" | Notes that FULL CUDA-graph + spec-decode support is still maturing per backend — relevant to item 5c (some backends force PIECEWISE/NONE, see `gpu_model_runner.py:6342-6360`). |

**No explicit TODO/FIXME exists for "per-request / dynamic K" in the EAGLE or ngram paths.** The
only dynamic-length mechanism in-tree is suffix decoding (6a/6b), which treats K as a max and varies
the *actual* depth per request at propose time — but it does **not** vary the lookahead KV
allocation, buffer shapes, or CUDA-graph sizing, all of which remain fixed to the scalar
`num_speculative_tokens`.

---

## Minimal patch surface (concrete checklist)

Ordered by dependency; each item is a distinct edit site.

1. **`v1/request.py`** — add `self.spec_len: int` per request (default = config K). *(new field)*
2. **`v1/core/sched/interface.py`** (~L103) + **`v1/core/sched/scheduler.py`** — add
   `set_spec_lengths(dict)`; store as `request.spec_len`. *(new method)*
3. **`v1/engine/core.py:449`** (and sync path ~L380) — call
   `scheduler.set_spec_lengths(controller_output)` immediately before `scheduler.schedule()`, where
   `controller_output` comes from your model over the per-step `SchedulerStats`. *(injection point)*
4. **`v1/core/sched/scheduler.py:215-222`** — demote the init scalar to a max/default; stop using it
   as the fixed lookahead in `schedule()`.
5. **`v1/core/sched/scheduler.py:463-467` and `:716-718, 746-755`** — pass
   `num_lookahead_tokens=request.spec_len` into `allocate_slots` (running + waiting/resumed).
6. **`v1/core/kv_cache_manager.py`** — *no change* (already parameterized by `num_lookahead_tokens`).
7. **Worker buffers → K_max:** `v1/worker/gpu/spec_decode/eagle/speculator.py:76-99`,
   `v1/worker/gpu_model_runner.py:830-840`. *(size to controller's max K)*
8. **Draft depth loops → K_max:** `speculator.py:203, 412`; `eagle.py:543, 1497`.
9. **Rejection sampler:** keep scalar = K_max (Option B) — *no change*; or make per-req (Option A /
   fully-variable) in `rejection_sampler.py:456, 526, 550, 572`.
10. **CUDA-graph sizing:**
    - Option A (eager fallback): rely on `v1/cudagraph_dispatcher.py:273-280` falling through to
      NONE for non-uniform steps; optionally force `cudagraph_mode=NONE`/`enforce_eager` for those
      steps. No capture-shape change.
    - Option B (pad to K_max): set `uniform_decode_query_len = 1 + K_max` at
      `v1/cudagraph_dispatcher.py:37-41` and `gpu_model_runner.py:775`; pad drafts to K_max in
      `scheduler.py:520-536` (reuse the `-1` padding pattern from `:1720-1722`). Capture shapes in
      `config/compilation.py:1261-1267` stay valid.

**Verification anchors already confirmed by the parent** (do not re-derive): `SchedulerStats` is built
every step by `make_stats()` at `scheduler.py:1931-1967` (`num_running_reqs`, `num_waiting_reqs`,
`kv_cache_usage`, `spec_decoding_stats`) and attached to `EngineCoreOutputs.scheduler_stats` at
`scheduler.py:1554`; spec counters live in `v1/spec_decode/metrics.py`. These are the serving-state
features your controller should consume, and they update under load.

### Unverified / explicitly-not-checked
- I did **not** run the server; all line numbers are from static reading of the installed source.
- The exact EAGLE CUDA-graph *capture* code path for the legacy runner (`v1/spec_decode/eagle.py`
  `_determine_batch_execution_and_padding`, L520-522) was read but not traced end-to-end into the
  capture loop; item 5c anchors are given at the dispatcher/compile level where the `1 + K` invariant
  actually lives.
- `medusa.py` and `draft_model.py` proposers were not line-audited (they share the same scalar-K
  pattern via `speculative_config.num_speculative_tokens`); treat them as additional 3b/3c sites if
  you target those methods.
