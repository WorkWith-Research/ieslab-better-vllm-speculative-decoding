# Skeptic Pass — 2026-10-07 (charter §11)

Assume the current main claim is wrong. Main claim under test: *"per-request profitable-K
heterogeneity is a real serving phenomenon that a causal controller can exploit."*

## Alternative explanation 1 — ngram artifact (strongest)
ngram acceptance is driven by *context repetition*, not by intrinsic token predictability. Its
acceptance profile is spiky (lag-1 autocorrelation ≈ 0, measured) and its "natural-K" may reflect
where the text happens to repeat, not where speculation is profitable. A model-based drafter with
smoother acceptance could show either (a) much less heterogeneity (phenomenon dies) or (b) more
clean structure (phenomenon strengthens).
**Targeted experiment:** Priority C (pre-registered): Qwen2.5-0.5B draft_model, K-sweep + per-request
heterogeneity via spec_hook. Status: script ready (`pc_draft_model.sh`), queued after the PA grid.

## Alternative explanation 2 — workload construction artifact
The mixed workload's "high-acceptance" class is *templated* prompts (primes, alphabet, repeated
sentences) — deliberately ngram-friendly. If adaptive-K only wins on this artificial mix, the result
is a test of my prompt design, not of real traffic.
**Targeted experiment:** rerun the PA grid's best/worst configs on **real ShareGPT prompts** (we have
the dataset + `sharegpt_client.py`). If LDM vs fixed-K ranking flips or vanishes on real text, the
mixed-workload result is an artifact. Status: planned immediately after PA grid (cheap: 2 configs ×
1 trial each as a screen).

## Alternative explanation 3 — eager-mode / measurement confound
All live SD comparisons run with `enforce_eager` (variable-K requirement). Eager mode changes
absolute throughput and *possibly* the relative K ranking vs CUDA-graph mode. If the K ranking flips
in graph mode, the "best fixed K" baseline is different.
**Targeted experiment:** rerun the fixed-K arms of the PA grid in **graph mode** (no LDM arm — it
can't run there) to check ranking stability. Status: planned after PA grid; ~30 min.

## Alternative explanation 4 — cost-model error driving the replay story
The replay oracle gap (~12%) rests on c(k) measured from Phase-1 random-workload runs. If the real
per-workload cost curve is flatter (E1 suggests it is in busy batches), the oracle gap shrinks and
the "83–89% recovery" becomes a property of a wrong model.
**Targeted experiment:** E2 — re-derive c'(k) from the PA grid itself (throughput ÷ accept-length per
config at C=16) and recompute the oracle on ShareGPT J-profiles with c'. Status: CPU analysis, can run
as soon as PA grid lands.

## Alternative explanation 5 — cold-start / short-request confound in live LDM
The optimistic cold start drafts KMAX for every request's first steps. With max_tokens=256 and
short-ish requests, a non-trivial fraction of each request's life is at K=8 regardless of its true
regime — biasing the LDM arm toward "K=8 with occasional cuts", i.e. toward the fixed-K=8 baseline.
**Targeted experiment:** decision-log analysis of the PA grid's ldm trials (fraction of steps at cold
start vs adapted; throughput contribution per phase). Status: CPU, after grid.

## Verdict so far
No single alternative fully explains A/B v1's INCONCLUSIVE result; E1 (batch coupling) + E3
(non-stationarity) are the leading candidates and both have decisive tests queued (E1 sweep, PA grid
at multiple concurrencies via the C-sweep). The ngram-artifact risk (AE1) is the biggest threat to
the *general* claim and is addressed by Priority C. Nothing here changes the current execution plan;
it confirms the queue: PA grid → E1 sweep → ShareGPT screen → graph-mode check → Priority C.
