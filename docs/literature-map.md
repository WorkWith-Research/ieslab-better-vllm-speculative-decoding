# Literature Map — Speculative Decoding × Serving Scheduling

Scope: what exists for (a) improving SD itself, (b) dynamically adapting speculation,
(c) serving-level / system-level adaptation of SD, and (d) vLLM's current state.
Last updated: 2026-10-07.

## A. Draft quality / acceptance rate (orthogonal to scheduling)

| Work | Idea | Note for us |
|---|---|---|
| EAGLE / EAGLE-2 / EAGLE-3 (Li et al.) | Feature-level autoregressive drafter on target hidden states; EAGLE-3 adds tree attention + confidence-aware sampling | De-facto standard draft in vLLM; our default real-draft method |
| DFlash (z-lab, arXiv 2602.06036) | Block-diffusion drafter: whole block in one forward pass, conditioned on target context features | Parallel drafting → T_draft ≈ const regardless of K; changes the K-cost calculus |
| DSpark (DeepSeek, arXiv 2607.05147) | Semi-autoregressive drafter (parallel backbone + Markov/GRU head) + **confidence head** per position; deployed in DeepSeek-V4 serving | See C — its scheduler is the closest prior work to ours |
| Medusa, MTP, DistillSpec | Multiple heads / multi-token prediction / distilled drafters | MTP is native in vLLM (no separate draft model) |

## B. Adaptive speculation length K — single-request level

All of these decide *per round / per request*, ignoring batch and system state:

| Work | Signal → decision | Result |
|---|---|---|
| Gante 2023 (heuristic) | past acceptance rate → SL | baseline heuristic |
| DISCO (arXiv 2405.04304) | trained classifier on draft prob vector + position → stop drafting | ~10% over best static SL; oracle upper bound ~39% |
| SpecDec++ (arXiv 2405.19715) | MDP formulation; acceptance-prediction head; threshold policy (stop when P(rejection) > h) | +7–11% over fixed-K SD, llama-2 7B/70B |
| AdaEDL (arXiv 2410.18351) | training-free: entropy of draft logits → lower bound on acceptance; stop if below λ | +10–57% vs static; robust across temperatures |
| SVIP (arXiv 2411.18462) | training-free length policy from draft entropy, for long-form / reasoning generation | up to +17% MT-Bench 8K, +22% AIME (QwQ); stacks with EAGLE-2 (+13%) |
| PEARL (arXiv 2408.11850) | parallelize draft+verify (pre-verify first token, post-verify during verify) → adaptive length emerges | up to 3.79× vs AR |
| AdaSD (arXiv 2512.11280) | hyperparameter-free: entropy threshold (stop generating) + JS-distance threshold (relaxed acceptance), both self-updating | up to 49% over standard SD, <2% accuracy loss |

**Takeaway:** the "when to stop drafting" problem is crowded and largely solved at
single-request granularity. None of them look at *other requests, queue, KV cache, or
GPU state*. That's the gap our idea targets.

## C. Serving-level / system-level dynamic SD (our direct competitors)

| Work | Decision scope | Signal → decision | Gaps we can exploit |
|---|---|---|---|
| **Nightjar** (arXiv 2512.22420, vLLM-based) | batch-level K per step + global SD on/off | contextual bandit over (batch size → γ); disables SD + offloads draft to CPU when queue/KV tight; models switch cost | Batch-level only (one K for all requests); bandit state is just load — no per-request acceptance features; no coupling with chunked prefill; SLO/goodput not in objective |
| **DSDE** (arXiv 2509.01083, vLLM 0.8.4 fork) | per-sequence, per-iteration SL + batch-wide adaptive cap | post-hoc KLD-variance "regional stability" signal after verification; cap mitigates stragglers | Reactive (post-hoc), no admission/chunking decisions; runs eager mode only (CUDA-graph re-capture problem); no learned model |
| **DSpark scheduler** (arXiv 2607.05147, DeepSeek-V4 production) | per-request verification length per step | confidence head → prefix survival probs; hardware-aware optimizer max E[accepted]·SPS(B) over lengths, with profiled throughput curve SPS(B) | Tied to their drafter's confidence head & proprietary engine; objective is expected-token-throughput only — no latency/SLO, no admission, no prefill-chunk coupling, no on/off memory decision |
| **TETRIS** (arXiv 2502.15197) | per-request draft *token selection* for verification in a batch | picks which draft tokens to verify to maximize batch throughput | Changes what is verified, not how much compute each request gets; orthogonal but composable |
| **BanditSpec** (arXiv 2505.15141, ICML'25) | per-prompt SD config choice | UCB/EXP3 bandits over configs during generation | Per-prompt, single-stream setting; no system state, no serving loop |

**Gap analysis — where a "Lightweight Decision Model" scheduler is still open:**

1. **Joint decisions.** Every prior work pulls one knob (K, or on/off, or token
   selection). Nobody jointly decides {admission/ordering, per-request K, prefill
   chunk size, SD on/off + draft memory} per scheduling iteration from a shared state.
2. **State richness.** Nightjar/DSDE use load or post-hoc divergence; DSpark uses
   drafter confidence. A small model over (queue depth & mix, per-request acceptance
   history, KV occupancy & fragmentation, GPU util, SLO slack) is unexplored — and is
   where *learning* (vs bandit/heuristic) could beat all of them.
3. **Prefill–decode coupling under SD.** Chunked prefill budgets tokens; with SD the
   decode side costs ~K× more per request. vLLM's scheduler splits on
   `max_num_batched_tokens` blind to K. Dynamic chunk sizing that accounts for
   speculative verify load is unaddressed (vLLM PRs #9291, #47822 show the plumbing
   exists but the policy is static).
4. **KV-cache-aware admission.** Each SD request holds ~K extra lookahead tokens' KV
   per step; under memory pressure that shrinks max batch. No work couples this with
   admission control (Nightjar only reacts globally via on/off).
5. **SLO/goodput objective.** Prior objectives: throughput or latency in isolation.
   Goodput under TTFT/TPOT SLOs with per-request heterogeneity is open.

## D. vLLM implementation state (as of v0.31, v1 engine)

- SD config: `speculative_config` with `method` ∈ {ngram, draft_model, eagle, eagle3,
  mtp, dflash, dspark, ...}, fixed `num_speculative_tokens` = K at engine init.
- Scheduler (`vllm/v1/core/sched/scheduler.py`): per-sequence lookahead KV allocation;
  `num_prefill_lookahead` for MTP-style drafters (chunked-prefill boundary handling,
  delayed caching of re-writable draft KV — PR #50062, #50897).
- Chunked prefill + SD supported (PR #9291); budget = `max_num_batched_tokens`.
- **Synthetic rejection sampler** (`rejection_sample_method="synthetic"`,
  `synthetic_acceptance_rates` / `synthetic_acceptance_length`): simulate any
  per-position acceptance profile *without a draft model* → ideal for controlled
  scheduler experiments (fast, deterministic, no EAGLE weights needed).
- Dynamic K: not supported in upstream v1 runtime (CUDA-graph capture assumes fixed
  K; DSDE ran eager mode). Per-request dynamic K = fork/patch territory.
- Metrics: Prometheus `/metrics` — `vllm:num_requests_waiting`,
  `vllm:num_requests_running`, `vllm:gpu_cache_usage_info`, spec-decode counters
  (drafts, draft tokens, accepted tokens, per-position acceptance).

## Open questions for the observation phase

- Q1. Where exactly does fixed-K SD start to lose to plain AR as load rises? (crossover)
- Q2. How much variance is there in per-request / per-step acceptance *within* a batch?
  Is it predictable from request features (prompt length, task type, context age)?
- Q3. What is the oracle gap: best per-step per-request K vs fixed K, under load?
- Q4. How do TTFT and TPOT interact when prefill chunks and speculative decode share
  the same step budget?
