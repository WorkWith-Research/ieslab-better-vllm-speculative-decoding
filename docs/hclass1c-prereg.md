# H-CLASS-1C preregistration — diverse queries + shared cached context (prefix caching ON)

**Date opened:** 2026-10-13 · **Status:** PROSPECTIVE (fixed before any C cell runs)
**Parent:** `docs/hclass1-prereg.md` (H-CLASS-1, closed 2026-10-13). This is a bounded follow-up
that splits the joint "prompt diversity + cache" effect that H-CLASS-1's **H-C2 violation** flagged
(prereg §8: condition C deferred; A-vs-B differences attribute to diversity + cache *jointly*).

## 0. Motivation and the single question

H-CLASS-1 found draft acceptance jumps from ~0.33 (condition A: diverse prompts, cache OFF) to ~0.61
(condition B: 2 repeated prompts, cache ON), and that the per-class SD-payoff spread appears only in B.
The results doc attributed this to "cache reuse." But **prefix caching does not change model logits** — it
only skips prefill compute for an identical token prefix. For a *given* prompt, draft acceptance is
cache-invariant. So the A↔B gap must be driven by one (or both) of two confounded causes:

1. **Content:** B's 2 specific repeated prompts happen to be more predictable/draftable than A's diverse mix.
2. **Reuse-state:** something about having a cached/shared prefix lifts draft quality even for the *query*.

Condition C isolates this: **diverse distinct queries (as in A) + one shared context document that all
requests cache (as in B), prefix caching ON.** This is the realistic RAG/agentic serving pattern
(shared retrieved context or system prompt + diverse user turns).

**Question (fixed):** does a diverse-query workload with a *shared cached context* show draft acceptance and
cross-class SD-payoff spread closer to **A** (diverse, no reuse) or **B** (repeated, high reuse)?

## 1. Frozen reference anchors (from the closed H-CLASS-1 run; not re-measured)

| anchor | source | value |
|---|---|---|
| A acceptance (dspark, steady window) | `results/hclass1/analysis.json` | c16/c48/c96 = 0.335 / 0.333 / 0.329 (mean ≈ **0.332**) |
| B acceptance (dspark, steady window) | `results/hclass1/analysis.json` | c16/c48/c96 = 0.608 / 0.608 / 0.618 (mean ≈ **0.611**) |
| A cross-class TPOT-ratio range | H-CLASS-1 §1 | 0.57% / 0.48% / 9.69% (c16/c48/c96) — none >15% |
| B cross-class TPOT-ratio range | H-CLASS-1 §1 | 37.0% / 41.6% / 42.2% (c16/c48/c96) — all >15% |
| A prefix-cache hit rate | H-CLASS-1 §5 | ≈ 0 (cache OFF, diverse) |
| B prefix-cache hit rate | H-CLASS-1 §5 | ≈ 0.976 (repeated prompts) |

## 2. Workload (fixed now)

**Environment:** identical to H-CLASS-1 §3 (Qwen3-4B + `deepseek-ai/dspark_qwen3_4b_block7` K=7,
1× RTX 3090 GPU0, vLLM V2 async scheduling, util 0.90, max_num_seqs 64, max_batched_tokens 4096).
Worktree rev recorded per cell in its `config_manifest.json`. Port 8402. Fresh server per cell.

**Prompt construction (diverse query + shared cached context):**
- **Shared context document D:** one fixed real ShareGPT-V3 turn, content-truncated so its own encoded
  length ≈ 512 tokens (binary search on content length; deterministic seed 20261013). Identical for every
  request in the run. Placed as a **system message** so it forms the token *prefix* of every prompt — this
  is what the prefix cache hits.
- **Distinct query Q:** a distinct real ShareGPT-V3 turn (same content pools + ISL brackets as condition A,
  no in-run reuse), placed as the **user message**.
- Prompt = `apply_chat_template([{system: D}, {user: Q}], add_generation_prompt=True)`, tokenized with
  `add_special_tokens=False`. Served with `stop_token_ids=[151645]` (`<|im_end|>`), so the model answers and
  can terminate naturally (identical OSL semantics to condition A). Realized ISL recorded per request.

**Classes (ISL = D + Q; uniform output cap as in A):**

| class | shared doc D | distinct query Q | realized ISL (approx) | max_tokens |
|---|---:|---|---:|---:|
| S | ~512 tok | ~50 tok (short pool, bracket [42,66]) | ~600 | 2048 |
| L | ~512 tok | ~2048 tok (long pool, bracket [2040,2064]) | ~2600 | 2048 |

**Arrivals:** Poisson at the frozen H-CLASS-1 §5b rates; C-threads {16, 48, 96} = P_low / P_med / P_high.
`temperature=0`, streaming, `stream_options.include_usage`. **Prefix caching ON** (the one difference from A
besides the shared doc).

## 3. Run set (fixed now)

arm ∈ {AR, DSpark-K7} × pressure ∈ {P_low(16), P_med(48), P_high(96)} × **3 trials (fresh server each)** =
**18 cells**. Counterbalanced: arms alternate within each pressure block (ar_t1 dspark_t1 ar_t2 …), pressures
ascending. Cell labels `ar_C_c{16,48,96}_t{1,2,3}` / `dspark_C_c…`.

## 4. Validity rule (per cell)

Identical to H-CLASS-1 §6 as corrected by deviation D3: server startup-complete in its own log; client exit
0 with error rate ≤10%; **condition-C load-applied gate** = condition-A open-loop gate (dispatcher issued ≥
⌊0.85·λ·0.95·DUR⌋ arrivals, `dropped_full_queue=0`) + service-aware completion floor max(15, ⌊0.6·min(λ·0.9·(DUR−20),
SAT_REQ_RATE·DUR)⌋) with SAT_REQ_RATE=0.35; steps.jsonl ≥95% carry `gpu_step_ms`; DSpark >1000 draft tokens,
AR exactly 0; async overlap `in_flight≥2` fraction ≥0.9; realized-OSL not degenerate (overall frac_at_max <0.95).
**C-specific additional guard:** prefix-cache hit rate must be **> 0.05** in the dspark/AR cells — i.e., the
shared document D must actually be cached and reused; if hit rate ≈ 0, the "shared context" is not being
served from cache and the cell does not test the intended condition (flag INVALID-reason, re-run once).

## 5. Analysis and verdict rule (fixed now)

Steady window + inference identical to H-CLASS-1 §7 (WARMUP=20 s excluded, final 10% of steps excluded;
trial-level n=3 for aggregate claims; request-paired bootstrap by prompt id for per-class ratios).

**Readouts (per cell, dspark arm primary):**
1. **Prefix-cache hit rate** (confirms D is cached; expected well above A's ~0, below B's ~0.976).
2. **Draft acceptance rate** (steady window), mean over 3 trials per pressure.
3. **Per-class DSpark/AR TPOT ratio** + cross-class relative range per pressure.

**Verdict (fixed bands; C_acc = C's trial-mean acceptance, compared to the §1 anchors):**
- **C≈A (content-driven gap):** |C_acc − 0.332| ≤ 0.05 AND C cross-class range < 15% at all pressures.
  → The A↔B acceptance gap is a **content-distribution** effect (B's specific prompts are more draftable);
  caching a shared prefix does not by itself lift acceptance or restore the class spread. The Track-A
  "cache state" signal is really a *content-predictability* proxy, not a cache mechanism.
- **C≈B (reuse-state effect):** |C_acc − 0.611| ≤ 0.05 OR C cross-class range > 15% at ≥1 pressure.
  → Shared-context reuse lifts draft acceptance / restores class-dependent payoff even under query diversity;
  cache/reuse state is a genuine, causally-available admission feature.
- **INTERMEDIATE:** otherwise — report the magnitudes (C_acc, hit rate, range) and note partial mediation.

**Causal limit (carried from H-CLASS-1 §8):** realized OSL is an outcome; the regression here is descriptive.
The shared doc D is identical content across requests, so "cache hit" and "shared context" are not separable
within C (both present); C vs A isolates *adding* a shared cached context to diverse queries, but does not
separate KV-reuse from the model simply seeing a longer, richer prompt. That residual is noted, not resolved.

## 6. Hard stops (abort, no re-tune)

GPU0 not free at run start (refuse); OOM/serving crash beyond one re-run per cell; `in_flight≥2` fraction <0.9;
>3 cells invalid after re-runs; total wall > 5 GPU-hours for the C run set. No workload edits after this file
is committed; no post-hoc window changes.

## 7. Provenance (filled after run)

- Launcher `experiments/hclass1_run.sh` (cond-C support added); client `experiments/hclass1_client.py`
  (COND=C branch); bank builder `experiments/build_hclass1_c_bank.py` → `data/hclass1_c_bank.jsonl`.
- Analysis: `experiments/hclass1_analyze.py` (SCHEME["C"] added) → `results/hclass1/analysis_c.json`.
- Raw artifacts under `results/hclass1/<cell>/` (gitignored, local). Workload hashes per cell in manifest.
