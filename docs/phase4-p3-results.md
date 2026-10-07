# Phase 4 — P3: SPEED-Bench throughput_2k (long prefills) K × load matrix (result)

**Question:** with realistic requests carrying ~2k-token prefills (supervisor feedback: diverse prefill/decode
sizes, SPEED-Bench), how does the optimal K change vs the short-prefill P2 result?
**Setup:** Qwen2.5-7B-Instruct, 1× RTX 3090 per GPU, vLLM 0.19.1 eager, **SPEED-Bench throughput_2k**
(1536 prompts, ISL p50=1986 tok / max 3980, pre-tokenized per official methodology; categories:
high_entropy / low_entropy / mixed, 512 each), K ∈ {none,1,2,4,8} × C ∈ {32,96} × 3 trials, OSL cap 512.

## Result — the prefill/decode mix is a second axis of "load" that moves argmax_K

| K | C=32 tok/s | C=96 tok/s |
|---|---|---|
| none (AR) | **695.6** ← argmax | 609.3 |
| 1 | 506.7 | 474.7 |
| 2 | 558.2 | 519.6 |
| 4 | 581.8 | 569.2 |
| 8 | 604.0 (−13%) | **613.8** (+0.7%, within noise) |

- **C=32 (moderate load, heavy prefill stream): AR wins by up to +13%** over K=8. With ~2k-token prefills
  arriving continuously, SD verification tokens (up to C(1+K)=288/step at K=8) crowd out prefill chunks within
  the 2048-token budget → prefills finish slower → fewer requests reach decode → aggregate throughput drops.
  This is the **H-S2/H-S3 prefill-interference effect** predicted before P3. Note the contrast with P2 (ISL~50):
  at the same C=32, K=8 was +1.0% vs AR. Same concurrency, different ISL → opposite ranking.
- **C=96 (queue-saturated: TTFT p50 ≈ 15–16s for AR):** throughput is pinned at ~610 tok/s regardless of K;
  K=8 vs AR is +0.7% — a null result on aggregate. The action is per-request (below).

## Per-category latency split (the LDM-relevant finding)

TTFT p50 / TPOT p50 (ms), pooled over trials:

| C | K | high_entropy TTFT/TPOT | low_entropy TTFT/TPOT | mixed TTFT/TPOT |
|---|---|---|---|---|
| 32 | none | 960 / 44.9 | 584 / 44.8 | 129 / 46.0 |
| 32 | 8 | 641 / 63.4 | 518 / **11.7** | 130 / 97.8 |
| 96 | none | 16380 / 88.0 | 15484 / 88.0 | 14605 / 137.8 |
| 96 | 8 | **8156** / 116.6 | 11048 / **38.2** | **6386** / 233.5 |

- **low_entropy (repetitive, ngram-friendly): K=8 cuts TPOT by ~57%** at C=96 (38.2 vs 88.0ms) — high
  acceptance means ~6–7 tokens/step; step-time inflation is more than paid back.
- **high_entropy (diverse, ngram-hostile): K=8 inflates TPOT +32–60%** at C=96 (116.6 vs 88.0ms) — low
  acceptance means few tokens/step but full step-time inflation.
- At C=96, K=8 also cuts TTFT for every category (e.g. high_entropy 8.2s vs AR 16.4s): faster decode turnover
  frees batch slots for waiting prefills. But the TPOT penalty on high_entropy content is real.

## Interpretation

1. **Two load axes matter, not one:** concurrency C (P2) and prefill intensity ISL (P3) independently move
   argmax_K. P2: C=8→96 flips K=8-best → AR-best. P3: at fixed C=32, ISL 50→2000 flips K=8-best → AR-best.
   A load-driven-K policy (DSpark's premise) needs *both* signals — a single "GPU load" scalar is insufficient.
2. **Aggregate throughput hides per-request structure:** at C=96 the K=8 vs AR aggregate gap is noise (+0.7%),
   yet per-category TPOT swings are ±50–60%. A uniform-K policy is systematically wrong for at least half the
   traffic; a per-request policy that reads acceptance (low_entropy→large K, high_entropy→small/off) has clear
   headroom to beat both uniform arms. This is the strongest measured argument yet for the LDM's per-request K.
3. **SPEED-Bench content exposes acceptance heterogeneity far beyond our synthetic mixed workload** — real
   prompts span repetitive (lists, code-like) and diverse (prose, reasoning) regimes within one serving stream.

## Caveats
- C=96 TTFT ~15s means this cell is a heavy queueing regime (prefill budget contention), not just compute
  saturation — both effects are present and not separated.
- ngram drafter: the low/high_entropy split partly reflects ngram's sensitivity to repetition; a model drafter
  would compress the gap (but the *direction* — heterogeneous K value across content — should persist).
- OSL cap 512 (vs SPEED-Bench default longer generations); steady-window throughput as defined in P2.
- Deterministic server + fixed prompt order per trial → trial variance again measures jitter only (±0–12%).
