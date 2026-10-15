# H-CLASS-1C results — diverse queries + shared cached context (prefix caching ON)

**Date:** 2026-10-13 · **Prereg:** `docs/hclass1c-prereg.md` (prospective; verdict bands fixed before any cell ran)
**Parent:** H-CLASS-1 (`docs/hclass1-results.md`, closed 2026-10-13, commit `22e4cc9`). This follow-up splits the
joint "prompt diversity + cache" effect that H-C2 flagged.

## 0. Verdict (per prereg §5 fixed bands)

> **C≈A — the A↔B draft-acceptance gap is a CONTENT-DISTRIBUTION effect.** Caching a shared prefix does not by
> itself lift draft acceptance or restore the per-class SD-payoff spread. H-OBS-3's class spread (and the ~0.33→0.61
> acceptance jump) comes from B's specific repeated prompts being more predictable/draftable, not from a reusable
> cache state.

Bands applied to trial-mean dspark acceptance vs frozen anchors A=0.332, B=0.611:

| pressure | C acc (t1/t2/t3) | C mean | \|C−A\| | \|C−B\| | C cross-class range |
|---|---|---:|---:|---:|---:|
| c16 (P_low)  | 0.3018 / 0.3032 / 0.3068 | **0.304** | 0.028 | 0.307 | 7.58% |
| c48 (P_med)  | 0.3034 / 0.3035 / 0.3013 | **0.303** | 0.029 | 0.308 | 5.67% |
| c96 (P_high) | 0.3180 / 0.3117 / 0.3166 | **0.315** | 0.017 | 0.296 | 10.73% |

- `near_a`: every C acc within 0.05 of A (max gap 0.029) **and** range <15% at all pressures → **TRUE**.
- `near_b`: no C acc within 0.05 of B; no range >15% → FALSE.
- Verdict rule: `near_a and not near_b` → **C~A (content-driven gap)**.

## 1. Run set

18 cells: arm {AR, DSpark-K7} × pressure {c16, c48, c96} × 3 trials, counterbalanced (arms alternate within each
pressure block), frozen §5b λ = 0.5714 / 1.7143 / 3.4286 req/s, 150 s, GPU0, fresh server per cell, prefix caching
**ON**. **18/18 DONE-VALID**, zero invalids (no re-runs needed this time). Wall ≈ 1.5 h for 17 cells (smoke cell ran
earlier) — well under the 5 GPU-h hard stop.

Workload (prereg §2): one shared ~552-tok ShareGPT context document as a system message (byte-identical prefix
across all requests) + distinct diverse ShareGPT queries; S = doc+short (~600 tok), L = doc+long (~2600 tok);
uniform output cap 2048, stop at `<|im_end|>`.

## 2. The shared context really was cached (condition sanity)

Prefix-cache hit rate (metrics steady window; server.log median agrees):

| pressure | c16 | c48 | c96 |
|---|---:|---:|---:|
| C hit rate | **0.327** | **0.230** | **0.230** |

Between A's ~0 (cache off) and B's 0.976 (fully repeated prompts), as designed; the >5% validity guard passed in
every cell. Hit rate falls from c16 to c48/c96 because higher concurrency evicts shared prefixes more aggressively —
and acceptance is **flat across pressure despite that** (0.304→0.303→0.315), which independently argues against a
cache-state mechanism: even with ~23% of tokens served from cache, acceptance does not move.

## 3. Per-class SD payoff (request-paired DSpark/AR TPOT ratios)

| key | S ratio [95% CI] | L ratio [95% CI] | cross-class range |
|---|---|---|---:|
| C_c16 | 0.931 [0.894, 0.967] | 0.863 [0.824, 0.900] | 7.58% |
| C_c48 | 1.255 [1.215, 1.293] | 1.186 [1.150, 1.225] | 5.67% |
| C_c96 | 1.311 [1.241, 1.391] | 1.178 [1.133, 1.225] | 10.73% |

No class spread (all ranges <15%), mirroring A (0.57/0.48/9.69%) and sharply contrasting B (37.0/41.6/42.2%).
The pressure-dependent sign flip (SD helps at c16, mildly hurts at c48/c96) reproduces H-CLASS-1's load effect —
that part of the payoff is real and independent of prompt content.

## 4. Interpretation

1. **Prefix caching ≠ draft-quality boost.** For a given query, serving (part of) the context from the prefix cache
   changes no logits; C shows acceptance stays at A-level (~0.30–0.32) even with a genuinely cached shared document.
   The 0.33→0.61 jump in B is therefore attributable to **which prompts** are being drafted (B's two repeated turns
   are more predictable than the diverse mix), not to reuse state.
2. **C's acceptance sits slightly BELOW A's** (0.30–0.32 vs 0.33). C's prompts carry ~550 extra context tokens and
   answer longer queries; the modest dip is consistent with a content/context-length effect on draftability, not
   noise against B. Either way it does not approach B.
3. **Refined Track-A signal.** The admission-time predictor should target an **acceptance proxy driven by prompt
   content/draftability** (e.g., perplexity/entropy features of the query+context), with cache state as a *cost*
   feature (TTFT/prefill savings) rather than a *quality* feature. H-OBS-3's "class effect" and H-C2's sign flip are
   both explained: they track content predictability, which in B was confounded with repetition.

## 5. Causal limits (carried from prereg §5)

C vs A isolates *adding* a shared cached context to diverse queries, but within C the "shared document" and "longer,
richer prompt" are not separable (both present). Residual: part of C's slightly lower acceptance may be the added
context length rather than cache mechanics. A would need a cache-ON diverse *no-shared-doc* control to split those;
not run (out of scope for this bounded follow-up). Realized OSL remains an outcome variable; all regression claims
here are descriptive.

## 6. Provenance

- Raw cells: `results/hclass1/{ar,dspark}_C_c{16,48,96}_t{1,2,3}/` (gitignored, local). All DONE-VALID.
- Analysis: `experiments/hclass1_analyze.py` (`SCHEME["C"]`, `condition_c` verdict block) → `results/hclass1/analysis.json`
  (`verdict_inputs.condition_c`), `analysis_table.md`.
- Harness commit: `8d964e2` (prereg + bank builder + client/runner/analyzer C support). Bank: `data/hclass1_c_bank.jsonl`
  (local; shared prefix verified byte-identical across all 2000 entries, LCP=546 tok).
