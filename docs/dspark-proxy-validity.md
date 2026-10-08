# DSpark-rule proxy — validity audit (2026-10-09)

**Scope:** `experiments/dspark_controller/sitecustomize.py` (the "DSpark-rule" arm of Phase 4.5,
results in `results/p5/p5_dspark_*`). This is a **DSpark-inspired decision-rule reimplementation**
using ngram drafting and empirical acceptance — NOT the DSpark system. This audit checks the seven
failure modes listed in the 2026-10-09 research addendum (§5). Raw results are preserved as-is; this
document records which of them are affected.

## Verdict (up front)

Two confirmed implementation flaws affect the policy actually executed in every Phase 4.5 DSpark-rule
cell:

- **F1 — decision-set completeness / KMAX fallback (CONFIRMED BUG).** The decision map is built only
  from requests that had non-zero scheduled drafts last step; all other running requests are enforced
  at `KMAX=8` by default, and a request decided to K=0 drops out of the map and returns to KMAX on its
  next drafted step. The pre-registered "all-K=0 SD-off fallback" was therefore **never effectively
  enforced**, and only ~6–7% of running requests were ever actually decided.
- **F4 — cost accounting (CONFIRMED BUG).** The SPS(B) cost term is evaluated at B = tokens of the
  *decided* subset (~7–28), not the true batch (~96 + prefill). The cost model systematically sees a
  batch ~10× smaller than reality, so it never prices saturation.

Consequences for recorded claims:

| Claim (as written) | Status after audit |
|---|---|
| "DSpark-rule is the worst arm at C=96 (−24.6%)" | **Invalid as a statement about DSpark's rule.** The executed policy was ≈ fixed-K8 for ~93% of requests plus per-request adjustments on ~7%, with a broken off-state. It is valid only as "our buggy proxy implementation lost at C=96." |
| "DSpark's decision rule does not capture the load-driven-K effect" (P4.5 interpretation #1) | **Unsupported.** The SPS(B) term's input B was wrong, so the rule never had a chance to price saturation. |
| "The missing ingredient is a live load-regime signal" (P4.5 interpretation #2) | **Weakened / unproven by this arm.** Still an open hypothesis — but Phase 4.5 cannot be cited as its evidence, because the proxy's cost model never received true batch size. |
| AR / fixed-K4 / fixed-K8 numbers from Phase 4.2 (reused) | Unaffected (no controller in those arms). |
| LDM arm result (−20.2% at C=96, "never commits to SD-off") | **Partially affected.** The LDM controller has the same `ks.get(req_id, KMAX)` default for requests absent from its map, but it *persists* per-request k* (updated in the stats hook, never cleared), so decided K=0 values survive. Its "frac(K=0)=1.2%" is a decision-log statistic over drafted requests; cold-start/new requests still ran at KMAX. The qualitative finding (acceptance-only policy lacks a saturation notion) survives; exact numbers are version-scoped observations. |

**Do not describe these results as a scientific failure of DSpark.** A corrected rerun is
pre-registered below (§6).

## Evidence

### F1 — decision-set completeness / persistent K=0 (CONFIRMED)

Code path (`experiments/dspark_controller/sitecustomize.py`):

1. `_hooked_stats` (per request, per drafted step) appends to `self._ds_pending[request_id]`.
   It is only called when the request had non-zero scheduled drafts last step — vLLM 0.19.1 gates the
   call site on `scheduled_spec_token_ids and generated_token_ids` (scheduler.py:1369) and
   `make_spec_decoding_stats` early-returns on `not num_draft_tokens` (scheduler.py:1985).
2. `_hooked_out` builds decisions **only** from `_ds_pending`, then does `self._ds_kstar = Knew` —
   the map is *replaced* each step; requests not in `_ds_pending` are absent from it.
3. `_hooked_drafts` enforces `kstar = ks.get(req_id, KMAX) if ks else KMAX` — **any request absent
   from the map gets KMAX=8.**

Measured on the C=96 decision logs (`results/p5/p5_dspark_c96_t{1,2,3}_mixed_decisions.jsonl`,
steady window t≥20s, ~138 s each):

| quantity | t1 | t2 | t3 |
|---|---|---|---|
| decided per step (mean) | 6.5 | 6.5 | 6.3 |
| running requests (target C) | ~96 | ~96 | ~96 |
| logged ΣKnew / step (mean) | 18.7 | 18.7 | 18.5 |
| frac K=0 among *decided* | 29.1% | 28.3% | 29.3% |

- **Only ~6–7% of running requests were ever decided.** The other ~93% ran at the KMAX=8 default on
  every drafted step (new arrivals, and — crucially — any request whose previous step produced zero
  drafts).
- **K=0 does not persist:** for every request with Knew=0 at step t, it is *absent* from step t+1's
  decision set in 4,072/4,072 cases; 317 of them reappear at t+2 **with Kobs=8** (full drafts via the
  KMAX fallback). The pre-registered all-zero SD-off outcome was thus a transient, one-step state.
- The "frac(K=0)=29.5%" reported in `docs/phase4-p5-results.md` is the share of *decided* requests
  whose greedy output was 0 — not the share of running requests actually executing at K=0 (which was
  ≈0 by construction).

The Phase 4.6 controller (`experiments/ldmload_controller/sitecustomize.py`) fixed exactly this class
of bug for its own arm (rev-#4c: "decisions persist in last_kstar so a k*=0 request stays k*=0 across
zero-draft steps") — which independently confirms the failure mode was real and known to us, but the
fix was never backported to the dspark controller.

### F4 — cost accounting (CONFIRMED)

The greedy evaluates Θ = τ·SPS(B) with `B = Σ_r (1+K_r)` over **decided requests only** (the `reqs`
dict from `_ds_pending`). Measured logged ΣKnew ≈ 18.7/step plus ~6 decided requests → B evaluated at
≈ 25, while the true step batch is ~96 decode + prefill tokens (server-side steady generation
~1,850 tok/s on a 48 SPS curve ⇒ B_live ≈ 96–130, cf. Phase 4.6 measurements). The profiled SPS table
(`results/p4_6/sps_profile.jsonl`) spans B = 8…130 and is monotonically decreasing there; evaluating it
at B≈25 pins SPS near its maximum for essentially every candidate allocation, so the cost term cannot
express saturation. Prefill tokens are absent from B entirely (no chunked-prefill awareness).

This makes the P4.5 explanation ("the profiled SPS(B) curve underestimates true step-time inflation")
an understatement: the problem is not interpolation error — the input was systematically ~10× too small.

### F2 — censored acceptance observations (PARTIAL, conservative bias)

`J[l] = P(acc ≥ l)` over observed accepted counts. When a request had only k drafts verified, positions
> k were unobserved, yet that step is counted as `acc < l` for all l > k — censored suffixes treated as
hard rejections. This biases J_r[j] **down** at deep positions (conservative: pushes K down). It cannot
explain the over-crediting observed in Phase 4.5; F1/F4 are the operative flaws. Noted for fidelity.

### F3 — probability semantics (OK)

J[l] is already a cumulative prefix-survival estimate; the greedy uses `dtau = J[K+1]` directly as the
marginal expected-accepted-tokens of extending request r from K to K+1, and τ0 = N for the all-zero
baseline. No double-multiplication of cumulative probabilities. Faithful in this respect.

### F5 — allocation algorithm (DOCUMENTED DEVIATION)

The pre-registration described "global greedy sort on prefix-survival with early stop." The actual
implementation is a sequential marginal-gain greedy: repeatedly extend the single request whose one-step
extension gives the largest ΔΘ, stop when none improves. Upstream vLLM's native allocation (PR #47808,
`adaptive_verification.py`) instead picks ONE batch budget B by argmax over expected-tokens/cost and
then admits the globally top-B slots by survival rank (contiguous prefix per request). Different
algorithms; our greedy is a reasonable approximation but not "the official candidate budget selection."
Relevant around non-linear/graph-padded cost curves, which our flat-interpolated SPS table does not
have. Recorded as a deviation, not a bug.

### F6 — enforcement (TIMING OK; DEFAULT IS THE F1 BUG)

Verified in vLLM 0.19.1 source: `EngineCore._process_engine_step` calls `step_fn()` then
`post_step()`, and `post_step()` → `take_draft_token_ids()` → `scheduler.update_draft_token_ids()`
before the next `schedule()`. Truncating `request.spec_token_ids` there IS effective for the next step
(scheduler.py:521–536 reads it at schedule time). So enforcement timing is sound; the KMAX default for
map-absent requests (F1) is the only enforcement defect.

Distinct execution states, as the addendum requires: in these runs "K=0 decided" (one-step transient),
"zero verified drafts" (ngram found no match — server still on spec path), and "AR-only server" (no
speculative_config) are all different; only the last is a true AR baseline.

### F7 — measurement (VERSION-SCOPED LIMITATION)

Phase 4.5 numbers used the client-side `steady_throughput_tok_s` (open-loop, arrival-rate biased), the
metric we later replaced with server-side generation counters in Phase 4.6/5.1. The C=96 gaps (≥20%)
far exceed trial noise, so the *direction* of the ranking is robust; the magnitudes (e.g. −24.6%) are
open-loop-biased and should be cited only as version-scoped observations on the Qwen2.5-7B / ngram /
vLLM 0.19.1 stack — never against a native DSpark baseline.

## Affected runs (preserved, not invalidated as measurements)

- `results/p5/p5_dspark_c{32,96}_t{1,2,3}_mixed_{metrics,decisions}.jsonl` (Phase 4.5 DSpark-rule arm).
- Any downstream citation of the "DSpark-rule" arm's C=96 ranking as evidence about DSpark's algorithm.

Unaffected: AR / K4 / K8 cells (no controller), LDM decision-log statistics over drafted requests
(quantified above with its own caveat).

## 6. Pre-registration — corrected proxy rerun (Track A, deferred)

**Status: pre-registered, NOT yet run.** Per the addendum, native validation (Track B) takes priority;
this rerun runs only if/when it is needed to separate "our old ngram setup" claims from native-baseline
claims. New run ID: **P4.5R**.

- **Fixes (mechanical, decision rule unchanged):**
  1. Decide for **every running request** each step (like Phase 4.6 rev-#4c); persist k* in a map that
     survives zero-draft steps; no KMAX default — unknown/cold-start requests use the pre-specified
     cold-start value (KMAX, documented).
  2. **B = true batch**: `total_num_scheduled_tokens` of the step (decode + drafts + prefill), as
     verified in Phase 4.6; no clamp to profile range (extrapolation monotone-decreasing, per the
     Phase 5 M1/M2 findings).
  3. Censoring: for a request with k observed drafts, count positions > k as *unobserved* (exclude from
     the J denominator) rather than as rejections.
- **Cells:** C ∈ {32, 96} × 3 trials, same mixed workload, server-side generation metric (t≥20s),
  eager, KMAX=8 — identical harness to Phase 4.6 (`experiments/p46_cell.sh` pattern).
- **Hypothesis (what the corrected proxy should show):** if DSpark's rule is faithful, its SPS(B) term
  now sees B≈96–130 at C=96 and commits toward low-K/SD-off; expected to land between fixed-K8 and AR.
  If it still loses badly to AR at C=96 *with correct accounting*, that is a genuine (proxy-level)
  limitation of the rule on this stack — citable as such, with the ngram-drafter scope stated.
- **Falsification:** corrected proxy within noise of AR at C=96 AND ≥ best fixed-K → the rule works and
  Phase 4.5's "worst arm" result was entirely the implementation bug.

## What this does NOT change

- Phase 4.6/5.1 results: their controllers (ldmload, ldm_batch) already had the persistence fix and
  true-batch accounting; they are unaffected by F1/F4 as characterized here.
- The historical record: no numbers are rewritten. `docs/phase4-p5-results.md` keeps its original text;
  this document supersedes its *interpretation* of the DSpark-rule arm, dated 2026-10-09.
