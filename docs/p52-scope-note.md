# Phase 5.2 — Scope note (claim-scope correction)

**Date:** 2026-10-12 · **Status:** DATED SCOPE NOTE (research-integrity addition per directive §9).
**Supersedes nothing.** The FALSIFIED verdict in `docs/p52-results.md` and the prereg completion note
in `docs/p52-prereg.md` §6 stand unchanged. This note corrects one overbroad inference that crept into
the results document's framing: the tested mechanism was **narrower** than the preregistration proposed,
and the structural conclusion is bounded to one stack.

## 1. What the preregistration proposed (docs/p52-prereg.md §5)

A *per-step spec-path-free off-state* making three runtime metadata / static-field modifications when
k\*=0:

1. `uniform_decode_query_len`: use 1 (not `1+num_spec_tokens`) for that step's attention metadata;
2. `is_spec_decode` / input-batch path: do not take the spec branch (K+1 query layout, draft-token-id
   slots) when k\*=0;
3. Per-step spec buffers: skip preparing draft-token-id / `num_decode_draft_tokens` buffers and the D2H
   copy stream for that step.

## 2. What was actually implemented and tested (P3 adjudication arm "specfree-off")

The tested off-state was **Phase 5.1 M3's mechanism**: force-0 draft truncation (drafts truncated to 0
every step, `VLLM_P52_FORCE0`) **plus** full ngram drafter-pass skip (`VLLM_P52_OFFSTATE`). The §5
metadata/static-field modifications were **never implemented**, because P1's code trace (results doc §2)
showed they are **no-ops in eager mode on this stack**:

- `uniform_decode_query_len` only feeds `_is_uniform_decode` → CUDA-graph dispatch; `pad_attn` is gated
  on `CUDAGraphMode.FULL`; both disabled under `ENFORCE_EAGER=1`.
- FLASH_ATTN v2's forward has **no per-step spec branch** — only init-time CUDA-graph sizing.
- force-0 already takes the AR code path in `_prepare_inputs` (`use_spec_decode=False` when drafts are
  zeroed), so there was no per-step draft branch left to switch off.

So the §5 mechanism and the tested mechanism coincide in *effect* on this stack (both reduce to "the
spec path is already minimal at k\*=0"), but they do not coincide in *content*: the static-field removals
were predicted vacuous, not empirically removed-and-measured. **The prereg's own design constraint**
(§5: "If a field cannot be made runtime-adaptive without touching CUDA-graph capture (eager-only is
fine), document it and fall back to leaving that one field static; the verdict must reflect which fields
were actually removed") is satisfied here by this note: **zero §5 fields were removed, because all three
were verified no-ops in eager before P3.**

## 3. What was measured (the FALSIFIED result, preserved verbatim)

specfree-off (= force-0 truncation + drafter-pass skip) recovers **none** of the ~8–9% SD-off floor:
C=32 identical to force-0 (+0% of gap), C=96 −9% of the AR−force-0 gap (worse than force-0); both arms
~8–9% below AR. 18/18 cells valid, 0 crashes/OOM. G1's literal threshold (≥50% of the per-step cadence
gap in *CPU-side residual*) was missed (49% at C=96, 28% at C=32) **before** proceeding to adjudication;
its falsification trigger was not hit because the floor is entirely spec-path work (results doc §3).

## 4. Which structural causes are supported by code inspection (vLLM 0.19.1, eager)

Supported at code level for **this stack only**:

- The residual floor at k\*=0 = (a) the ngram drafter pass running every step (its output truncated to 0;
  M3's skip of exactly this was A/B-measured and recovered nothing — i.e., the *launch* overhead of the
  drafter path, not its FLOPs, persists) + (b) init-time GPU memory for draft buffers / attention
  workspace that never returns per-step (KV pool −4.4% vs AR, no occupancy/preemption impact measured).
- The floor is **structural to running with `speculative_config` set** on vLLM 0.19.1 eager: no runtime
  off-state reachable from within a spec-enabled server removes it.

## 5. Which runtime alternatives remain UNTESTED (do not claim them settled)

1. **Separate AR server pool / init-time no-spec serving** for the off-state — a genuinely spec-free path;
   expected to reach AR by construction, but never measured here (out of scope: in-loop controllers).
2. **CUDA-graph-mode variants** — §5's fields are live under `CUDAGraphMode.FULL`; the eager no-op
   finding does not transfer to graph-captured serving.
3. **Other stacks** — see §6.

## 6. Scope boundary (the corrected inference)

The FALSIFIED verdict and its mechanism explanation hold for: **Qwen2.5-7B-Instruct / ngram K=8 / vLLM
0.19.1 / ENFORCE_EAGER / TP=1**. They do **not** establish that:

- every possible SD-off implementation is irreducibly slower than AR (only the runtime off-states
  reachable within a spec-enabled server on this stack were tested); or
- the ~8–9% floor magnitude transfers to any other stack — in particular it must **not** be carried over
  to native DSpark / vLLM 0.25.0 (V2 runner, learned drafter, CUDA-graph capture) without direct
  measurement there. Track B's CTRL-ON arm (always-on patched path vs native fixed-K) measured patch
  overhead ≤1%, but a DSpark off-state floor has never been measured and is not claimed to equal the
  ngram-stack value.

**Bottom line:** "a runtime spec-path-free off-state does not remove the SD-off floor" is true *for the
mechanisms tested on vLLM 0.19.1 eager*; "~8–9% irreducible off-state cost" is a stack-specific constant,
not a general serving law.
