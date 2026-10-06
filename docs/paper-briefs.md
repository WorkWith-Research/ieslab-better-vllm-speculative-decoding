# Paper Briefs — Serving-Level Dynamic Speculative Decoding

Purpose: precise technical brief on the three closest prior works on *serving-level* dynamic
speculative decoding (SD), plus short notes on TETRIS and BanditSpec. Feeds our related-work
section and baseline design. Focus is on **exact mechanisms** (decision variables, granularity,
signals, objective, engine integration, numbers) — not breadth.

All full texts retrieved from arXiv HTML (multi-pass). Citations: arXiv id + venue if stated.

| Paper | arXiv | Venue | Engine | Decision unit | K/length granularity | On/off? | Chunk size? | Admission? |
|---|---|---|---|---|---|---|---|---|
| **Nightjar** | 2512.22420 (v3) | arXiv preprint (IEEE-format ms, no accepted venue stated) | vLLM 0.8.2 fork | **batch**, per step (bin-locked) | one γ for whole batch | **yes** (γ=0 + draft offload) | no | no (global memory reaction only) |
| **DSDE** | 2509.01083 (v1) | arXiv preprint (IEEE-format ms, no venue stated) | vLLM 0.8.4 fork (**eager**, no CUDA graphs) | **per-sequence**, per iteration | per-request SL + batch cap | no (SL_min=2 floor) | no (KV alloc only) | no |
| **DSpark** | 2607.05147 (v1) | arXiv preprint (DeepSeek-AI tech report) | DeepSeek-V4 proprietary engine (ZOS + CUDA graphs) | **per-request**, per step | per-request verify length l_r (draft γ fixed) | no (always drafts γ; l_r can→0) | no | no |
| TETRIS | 2502.15197 | arXiv preprint | vLLM (sequential pipeline) | per-request token selection within capacity C | which tokens verified, not K itself | n/a | no | no |
| BanditSpec | 2505.15141 | **ICML 2025** | (framework; A100, batch 1 / 1–50 sim) | per-prompt config choice | draft model or γ | n/a | no | no |

---

## 1. Nightjar — "Dynamic Adaptive Speculative Decoding for Large Language Models Serving"

- **Citation:** arXiv:2512.22420 (v3). Rui Li, Zhaoning Zhang, Libo Zhang, Huaimin Wang, Xiang Fu,
  Zhiquan Lai — National University of Defense Technology, China. IEEE-format manuscript; no
  accepted venue printed in the paper.
- **Core claim:** SD helps in low-load/memory-bound regimes but hurts in high-load/compute-bound
  regimes (verification overhead + draft-model memory stealing KV). Adapt γ to load and *turn SD
  off* (offloading the draft model to CPU) when queuing/KV pressure is high.

### Decision variables & granularity
| Variable | Space | Granularity | Timing |
|---|---|---|---|
| Speculative length γ | {0,1,…,Γ_max} | **batch-level** — one value for all requests in the current batch | per decoding step, but **locked per "bin"** (a fixed-duration window) to bound switching cost |
| SD on/off | γ=0 vs γ>0 | global (whole engine) | implied by γ; offload triggered separately |
| Draft-model residency | GPU vs host (offloaded) | global | async, hysteresis-controlled |

- Context for the bandit = **current batch size B** only. No per-request features.

### State / signals used
- Current batch size `B` (bandit context).
- Realized goodput `g_t` (tokens/s) → latency-per-token `l_t = 1/g_t`; empirical mean `g̃_{B,γ}`
  maintained per (batch-size, γ) pair.
- KV-cache free blocks `N_free` vs threshold `τ_low`; waiting-queue cardinality `|Q_wait|`;
  persistence window `T_persist` for confirming low-memory states.

### Objective (formula)
Minimize **latency per token** (not raw goodput, to capture switching cost), as a regret-minimization
problem over T steps:

```
γ*_t = argmin_{γ∈{0..Γ_max}} { 1/g̃_{B,γ} + I(γ_{t-1}=0 ∧ γ>0) · C_switch / γ }      (Eq. 4)
L_t(γ_t) = l_{B_t,γ_t} + I(γ_{t-1}=0 ∧ γ_t>0) · C_switch / γ_t                        (Eq. 2/5)
Regret R(T) = Σ_t [ L_t(γ_t) − L_t(γ*) ]  →  minimize; bound R(T)=Õ(√T) (Thm 1)
```

- `C_switch` = KV-cache reconstruction cost when re-enabling SD, **amortized by γ** (divided by γ to
  discourage short-sighted switching). Looked up from an offline table `C_switch(L_max, B)` where
  `L_max = max_i L_i` is the effective skip length (max new-token lag in the batch). Measured on RTX
  4090: **17.87 ms → 102.03 ms** (input 128–512, batch 32/64).
- Algorithm: **ADA-BINGREEDY-style contextual MAB** (batch size as context). Hierarchical blocks/bins;
  exploration prob `p = 1/b_B`; arm locked per bin; exponential block growth `H_j = 2^{j−1}`.
  Arm-selection overhead ≈ 1e-5 s vs ~0.034 s/token generation (~3400× cheaper).

### Engine integration (vLLM)
- Implemented by **extending vLLM 0.8.2**. Three layers: (1) Scheduler (continuous batching, feeds B),
  (2) Planner/bandit (step-wise γ + on/off), (3) **elastic Memory Manager** ("Squeeze/Expand").
- **Elastic memory:** when SD is disabled AND `N_free < τ_low` persists for `T_persist`, offload draft
  weights to host → KV pool expands by `N_draft = ⌈S_draft / B_block⌉` blocks (in-place expansion).
  Contraction = physical compaction + logical block-table remapping via a **custom Triton vectorized
  migration kernel**; reload is async (CUDA streams + non-blocking DMA) overlapping compute.
- Eager vs CUDA graphs: **not explicitly discussed.** The modeled `C_switch` *is* the KV-reconstruction
  reconfiguration cost they account for; no CUDA-graph re-capture limitation stated (unlike DSDE).

### Experimental setup
- **Models / GPUs:** DeepSeek-R1-Distill-Qwen-7B + DeepSeek-R1-DRAFT-Qwen2.5-0.5B (1× RTX 4090 24GB);
  Vicuna-13b-v1.5 + vicuna-68m (1× A100 40GB); Qwen2.5-32B-Instruct + Qwen2.5-0.5B-Instruct
  (2× L20 48GB, TP) for multi-GPU scalability.
- **Datasets / load:** ShareGPT, Alpaca, SpecBench (480 instances each); Azure LLM Inference trace
  segment; Poisson arrivals at varying QPS (dynamic request rate).
- **Baselines:** Standard SD (chain γ=3), Vanilla AR (w/o SD), DSD (goodput linear-regression),
  BanditSpec (UCB MAB, no batch context), TETRIS (entropy-based token selection). 5 runs averaged.

### Headline numbers (conditions)
- **+27.29%** avg throughput vs w/o SD; **+7.39%** over strongest baseline (TETRIS); +8.32% over
  standard SD; +22.89% over DSD, +19.76% over BanditSpec.
- **Up to 12.90%** lower mean end-to-end latency vs standard SD; up to **38.35%** on 13B/SpecBench
  (latency gain scales with target-model size — near-zero on fast 7B, large on heavy 13B).
- Offload ablation: **6315.9 tok/s at 35 req/s** vs 5982.6 without offload (compute-bound regime).
- Threshold sensitivity: ~10% free-KV buffer optimal → 6110 tok/s plateau vs 5768.12 fixed.
- Elastic-op overhead (30B, 2×L20): expansion 143.9 ms; contraction 11.9 ms (Triton); reload dispatch
  21.9 µs CPU.

### Stated limitations / what's missing
- **Batch-level K only** — one γ for the entire batch; no per-request heterogeneity in speculation.
- Bandit state is **just load (batch size)**; no per-request acceptance history, no KV occupancy as a
  decision feature (KV only drives the on/off offload), no GPU-util signal.
- **SLO/goodput not in the objective** — minimizes latency-per-token with throughput as the reward;
  no TTFT/TPOT SLO constraint or goodput-under-SLO formulation.
- **No chunked-prefill coupling** — prefill enters only through `C_switch` (KV reconstruction cost);
  no dynamic chunk sizing.
- Regret bound assumes a **stationary environment** per batch size.

### Chunked-prefill / KV-cache / SLO-goodput notes
- **KV-cache:** central to the design — draft weights vs KV are a zero-sum memory competition; offload
  reclaims `N_draft` blocks. DeepSeek-V3 example: 14B MTP draft = 1.4 GB/GPU ≈ **~49k KV tokens** under
  MLA, which vLLM wastes by keeping the draft resident even when SD is off.
- **Chunked-prefill:** only via `C_switch` modeling; no chunk-size decision.
- **SLO/goodput:** goodput `g_{B,γ}` is the reward; reformulated to latency-per-token to include
  switching cost. No explicit SLO constraint.

---

## 2. DSDE — "Dynamic Speculative Decoding with KLD Stability for Real-World Serving"

- **Citation:** arXiv:2509.01083 (v1). Mingyu Yang, Jae-Young Choi, Kihyo Moon, Minsung Jang, Eunjoo
  Joen — Samsung SDS Cloud Research Team, Korea. IEEE-format manuscript; no accepted venue stated.
- **Core claim:** a *training-free* dynamic per-sequence SL driven by a **post-hoc** signal (variance of
  KLD divergence = "regional stability"), plus an adaptive batch-wide cap to fix the straggler problem.

### Decision variables & granularity
| Variable | Space | Granularity | Timing |
|---|---|---|---|
| Speculation length SL_i(t) | [SL_min, SL_max], SL_min=2 fixed | **per-sequence** (per-request) | **per decoding iteration** ("Ragged Q" = non-uniform lengths within one batch) |
| Adaptive cap SL_cap | real → applied as uniform upper bound | **batch-wide** | per step |

- No SD on/off (SL floored at 2). No admission, no chunk-size decision.

### State / signals used (all post-hoc, after rejection sampling)
- Per-token **KLD** between draft and target distributions from recent verification steps.
- `SF = exp(2·μ_KLD_last) − 1` — Scale Factor from mean KLD of the most recently verified sequence.
- `WVIR = Var_w(KLD_short) / Var_w(KLD_long)` — Weighted Variance Intensity Ratio; weighted variance
  with exponential decay δ=0.85, short window N=10, long window N=30. WVIR>1 → growing instability.
- Calibration-phase stats: `SL_A,max` (max tokens accepted in any step during a brief pre-processing
  phase), mean & max KLD over that phase.

### Objective / decision rule (formula)
No formal optimization for SL itself — a **heuristic closed-form**:

```
SL_max = SL_A,max · (1 + μ_KLD,pre / (KLD_pre,max + ε))     (Eq. 1; data-informed upper bound,
                                                             anchored to observed max accepted length)
SF   = exp(2·μ_KLD,last) − 1                                 (Eq. 3)
WVIR = Var_w(KLD_short) / Var_w(KLD_long)                    (Eq. 4)
SL̂_i = (1 − SF·WVIR)·(SL_max − SL_min) + SL_min   if SF·WVIR ≤ 1, else SL_min   (Eq. 8)
SL_cap = argmin_c (1/N)Σ_i (c − SL̂_i)²  =  (1/N) Σ_i SL̂_i    (Eqs. 9–11; arithmetic mean)
```

- The cap is the MSE-minimizing consensus length → mitigates stragglers (outlier long predictions
  stalling the batch).

### Engine integration (vLLM)
- **Extends vLLM 0.8.4** (v0 engine, not v1). New **"SL Adapter"** module after the rejection sampler;
  modifies the **Look-ahead Scheduler** to pre-map/reallocate KV blocks per sequence from SL_i(t);
  integrates the **FlashAttention-2 variable-length kernel** in the Target Worker so ragged proposal
  lengths verify in one pass with **no padding** (verification runs along `SL_max(t)=max_i SL_i(t)` with
  per-sequence validity masks).
- Lookahead slots computed from SL_i(t) and "applied uniformly to prefill, decode, and chunked prefill"
  (KV-allocation consistency across all three phases — but **no dynamic chunk sizing**).
- **Runs in EAGER mode, NO CUDA Graphs** — explicitly stated limitation: changing SL each step would
  require re-capturing graphs; they note vLLM v1 piecewise CUDA graphs may mitigate.

### Experimental setup
- **Models / GPUs:** LLaMA-3.1-70B-Instruct + LLaMA-3.2-1B-Instruct (main); Gemma-27B + Gemma-2B
  (low-acceptance/divergent regime). Single server, **8× A100 80GB**.
- **Datasets:** CNN/DM, XSum, GSM8K, HotpotQA, NQ, HumanEval, ShareGPT, WMT14 (8 datasets).
- **Metric:** end-to-end request latency = avg completion time of **128 prompts** under varying batch
  sizes; throughput vs batch size 1–64.
- **Baselines:** Autoregressive (no SD), **Static-opt** (per-dataset profiled best SL over {2,4,6,8,10}),
  AdaEDL (base=7). Signal-correlation study: draft entropy vs mean KLD vs WVIR.

### Headline numbers (conditions)
- LLaMA-70B, temp 0.0: **DSDE 13.97 s** vs Static-opt 13.44 s vs AdaEDL 13.83 s vs AR 38.41 s → **2.75×**
  speedup (competitive with static-opt *without* profiling). Temp 1.0: 19.19 s vs static-opt 18.02,
  AdaEDL 17.64 (sampling noise widens the gap for a lagging signal).
- **Static-opt profiling cost: ~9825 s (~2.7 h) per dataset, ~22 h total for 8 datasets** — DSDE avoids it.
- Low-acceptance (Gemma): DSDE stays near static-opt (e.g., **180%** of static-opt on CNNDM) while
  AdaEDL degrades to 234% (forward-looking entropy fails under high divergence).
- Scalability (CNN/DM, batch 1→64): No-Cap per-seq scales only **11.21×** (t=0)/11.92× (t=1); **with cap
  12.16× / 13.01×** — the cap recovers most of the straggler loss.
- Signal correlations are weak (they acknowledge): temp 0.0 → entropy r=−0.339, mean KLD −0.164, WVIR
  +0.128; all ≈0 at temp 1.0.

### Stated limitations / what's missing
- **Eager mode only** (no CUDA graphs) — a real deployment cost they flag for future work.
- **Reactive / post-hoc signal** with one-step lag; weak token-level predictive power (their own
  correlation study).
- **No admission control, no chunked-prefill sizing decision** (only KV allocation applied uniformly),
  **no SD on/off** (SL_min=2 floor keeps speculation always on).
- **No learned model**; heuristic formula. **SLO not in the objective** (latency/goodput mentioned as
  outcomes, no TTFT/TPOT constraint).

### Chunked-prefill / KV-cache / SLO-goodput notes
- **KV-cache:** per-sequence lookahead KV pre-mapping/reallocation; FA2 varlen kernel avoids padding.
- **Chunked-prefill:** handled only for KV-allocation consistency (lookahead slots applied to
  prefill/decode/chunked-prefill); no chunk-size policy.
- **SLO/goodput:** goodput cited as an improved outcome; no SLO-constrained objective.

---

## 3. DSpark — "Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation"

- **Citation:** arXiv:2607.05147 (v1). Xin Cheng, Xingkai Yu, Chenze Shao, Jiashi Li, Yunfan Xiong, et al.
  — Peking University + DeepSeek-AI. Technical-report style; no venue. Open-sources DSpark checkpoints +
  the **DeepSpec** training repo (Eagle3/DFlash/DSpark).
- **Core claim:** two parts — (1) a **semi-autoregressive drafter** (parallel backbone + lightweight
  sequential head) to fix parallel-drafter suffix decay; (2) **confidence-scheduled verification**: a
  confidence head gives per-position prefix-survival probabilities, and a **hardware-aware prefix
  scheduler** prunes each request's verification length by expected return under current engine load.

### Decision variables & granularity
| Variable | Space | Granularity | Timing |
|---|---|---|---|
| Scheduled verification length l_r | {0,…,γ} | **per-request** | **per decoding step** (batch-level global greedy) |
| Draft block size γ | fixed (7 offline; 5 in production "DSpark-5") | per deployment | **not** dynamically decided by the scheduler |

- Key nuance: the scheduler decides how much of the *already-drafted* γ-block to **verify** per request.
  The draft itself always runs for the full γ (fixed draft-side cost). l_r=0 → verify only the anchor/bonus
  token (minimal speculation) but drafting still happened. No SD on/off, no admission, no chunk size.

### State / signals used
- **Confidence head** `c_k ∈ (0,1)` per draft position = conditional prob token k survives verification
  given all preceding accepted: `c_k = σ(wᵀ[h_k ; W_1[x_{k−1}]])` (linear + sigmoid on backbone hidden
  state + Markov embedding of previous token). Trained with BCE against the analytical per-step
  acceptance rate `c*_k = 1 − ½‖p_d − p_t‖₁` (total-variation distance).
- **Post-hoc calibration — Sequential Temperature Scaling (STS):** calibrates the cumulative product
  `∏_{i≤k} c_i` left-to-right to minimize ECE on a held-out set. Raw head is overconfident (ECE 3–8%,
  ROC-AUC 0.81–0.90); STS → **ECE ~1%**.
- Prefix survival `a_{r,j} = ∏_{i≤j} c_{r,i}`.
- **Profiled engine throughput curve SPS(B)** (steps/sec vs forward-pass batch size B) — profiled once at
  init, stored as a cost table. This is the "load" input, but it is a **static profiled curve**, not a
  real-time measured queue/KV/GPU state.

### Objective (formula)
Maximize **expected system-wide token throughput** over per-request verification lengths:

```
B = Σ_r (1 + l_r)                       total verification batch size (tokens) to target
τ = Σ_r (1 + Σ_{j≤l_r} a_{r,j})          expected accepted tokens
Θ = τ · SPS(B)                           maximize over {l_1,…,l_R}        (Alg. 1)
```

- **Greedy solution:** sort all candidate extensions (r,j) globally by `a_{r,j}` descending; admit
  incrementally, update Θ via SPS lookup; **early-stop when Θ ≤ Θ_best** to preserve the non-anticipating
  (lossless) property — a retrospective global search would leak future token x_{r,k} into admission and
  bias the output distribution (counterexample in their Appendix A). Early-stop yields the *global* max
  iff Θ is unimodal (assumes smooth SPS).

### Engine integration (DeepSeek-V4, not vLLM)
- Deployed in **DeepSeek's proprietary serving engine** with **Zero-Overhead Scheduling (ZOS)** +
  continuous CUDA-graph replay. Per-step dynamic draft tokens clash with both → scheduler runs
  **asynchronously**: the truncation length K (capacity limit) is set from confidence-head outputs **two
  steps prior**; the current step's candidates are still sorted by their actual up-to-date cumulative
  confidences → effectively a **dynamic top-K selection**. Dropping the early-stop break enables an
  unconstrained global search; causality is preserved because the decision depends only on info from two
  steps earlier (the async gap is the "causal barrier").
- Variable-length execution: all tokens across requests are **flattened and processed as independent
  elements**; intra-sequence dependencies carried by a marker tensor in sparse attention. Only the
  index-attention and compress kernels needed modification for DeepSeek-V4.

### Experimental setup
- **Offline (draft quality; scheduler disabled, fixed block γ=7):** targets Qwen3-{4B,8B,14B}, Gemma4-12B;
  drafters DSpark vs Eagle3 (AR) vs DFlash (parallel), all retrained in the same framework on
  Open-PerfectBlend (1.3M samples). Benchmarks: GSM8K/MATH500/AIME25, MBPP/HumanEval/Live-CodeBench,
  MT-Bench/Alpaca/Arena-Hard; temp 1.0, chain drafting; metric = accepted length τ per round.
- **Online (production):** DeepSeek-V4-Flash (preview) & V4-Pro (preview); DSpark-5 (γ=5, Markov head,
  3 MoE-layer backbone + mHC + sliding-window-128 attention). Baseline = **MTP-1** (the prior single-token
  production setup). Live user traffic; SLA anchors Flash 80 & 120 tok/s/user, Pro 35 & 50 tok/s/user.

### Headline numbers (conditions)
- Offline accepted length: DSpark vs Eagle3 **+30.9% / +26.7% / +30.0%** (4B/8B/14B macro-avg); vs DFlash
  **+16.3% / +18.4% / +18.3%**. Domain effect: math/code accept ~5.1–5.6 vs chat ~3.5 (Qwen3-4B).
- Online, **at matched throughput**: per-user generation speed **+60–85%** (V4-Flash), **+57–78%**
  (V4-Pro). At moderate SLA: **+51%** aggregate throughput over MTP-1 (Flash @80 tok/s), +52% (Pro @35).
  At strict SLA the baseline is near its operational boundary → nominal **+661%** (Flash @120) / +406%
  (Pro @50); they interpret these as frontier-extension, not representative speedups.
- Verification budget dynamics: expands from MTP-1's static **2 tokens** to **~4–6 tokens/request** at
  moderate concurrency (<200 Flash, <150 Pro), then shrinks smoothly as load saturates.
- Confidence-threshold sweep (offline, Qwen3-4B): chat acceptance 45.7%→95.7%, math 76.9%→92.5%, code
  67.6%→92.0% with pruning.

### Stated limitations / what's missing
- **Fixed draft-side cost:** the full γ-block is always generated by the parallel backbone; for
  low-acceptance queries this drafting compute is unrecoverable (no difficulty-aware early exit in the
  drafter — named as future work).
- Objective is **expected token throughput only** — no latency/SLO term inside the scheduling objective
  (SLA anchors are used in *evaluation*, not the optimization).
- Tied to their **proprietary engine + confidence head**; SPS(B) assumed smooth/unimodal for the lossless
  early-stop version (real curve is jagged → they drop early-stop and go async).
- **No admission control, no prefill-chunk coupling, no KV-cache-aware memory decision.**

### Chunked-prefill / KV-cache / SLO-goodput notes
- **Chunked-prefill:** not addressed — decode-only verification scheduling.
- **KV-cache:** mentions "fixed KV-cache capacity per request" as a resource limit that keeps effective
  batch size below the compute-saturating threshold (so total throughput and per-user tok/s become
  correlated, simplifying their objective); no KV-aware admission.
- **SLO/goodput:** goodput framed in related work; scheduler maximizes Θ=τ·SPS(B); SLA anchors used to
  characterize the Pareto frontier, not as a constraint in the objective.

---

## 4. TETRIS — (5-line skim) "Optimal Draft Token Selection for Batch Speculative Decoding"

- **Citation:** arXiv:2502.15197. Wu, Zhou (NUS); Verma, Prakash, Rus, Low (SMART/CSAIL MIT). arXiv preprint.
- **Decision scope:** a "manager" between draft and target selects *which* draft tokens (i,j) to verify
  under a fixed total verification capacity C = Σ_i k_i — longer windows for "easy" requests, shorter for
  "hard" ones. It changes what is verified, not how much compute each request gets via K per se.
- **Objective:** maximize per-step throughput `G_step = (E[Σ 1_{i,t}] + N)/τ_step` (accepted + bonus
  tokens per step); under mild assumptions per-step optimality ⟺ global optimality (Thms 1–2).
- **Signal/mechanism:** greedy max-heap over cumulative acceptance `∏_{t≤j} p_{i,t}` (p = draft output
  probability as surrogate for true acceptance); O(C log N), GPU scatter_max <0.3 ms vs >2.5 ms draft/token.
- **Setup/numbers:** vLLM sequential pipeline; Vicuna-33B / Llama-70B / Llama-405B on 4×L40, 8×L40, 8×H100;
  fixed batch 64; ShareGPT/Arena/Tough/Sonnet; vs standard SD + DSD → up to **~5.25%** throughput over best
  baseline (up to 9.27% vs standard SD), latency up to 6.13%/9.32%; projected up to 12.04% under a
  (not-implemented) parallelized pipeline. Limitation: sequential vLLM pipeline caps realized gains.

## 5. BanditSpec — (5-line skim) "Adaptive Speculative Decoding via Bandit Algorithms"

- **Citation:** arXiv:2505.15141, **ICML 2025**. Hou, Zhang (NUS); Du, Pang, Du (Sea AI Lab); Zhang (SMU);
  Pan (NUS); Tan (NUS); Yang (Yale). Code: github.com/sail-sg/BanditSpec.
- **Decision scope:** per-prompt SD *configuration* choice — each hyperparameter spec S_i (a draft model,
  or a speculation length γ) is an arm in a Multi-Armed Bandit; re-decides as the prefix grows over one
  request. Two algorithms: **UCBSpec** and **EXP3Spec**.
- **Objective:** minimize **stopping time** ST = number of SpecDecSub calls until EOS; performance measured
  by *stopping-time regret* vs the best fixed config; upper bounds under stochastic & adversarial rewards,
  with an info-theoretic lower bound showing UCBSpec is optimal up to constants.
- **Setup/numbers:** LLaMA3-8B / Qwen2-7B on a single A100; batch size 1 (draft-model arms: PLD/Rest/Suffix
  Tree/Eagle-2) and batch 1–50 serving sim (γ∈{0..4}, Eagle-1); SpecBench/Alpaca/Code Editor/Debug Bench.
  UCBSpec best on all (e.g., LLaMA3 SpecBench 105.7 tok/s vs Eagle-2 98.15; Debug Bench +13%/+19%);
  spec-length exp approaches the oracle best across batch sizes 1–50 (16 runs).
- **Key gap for us:** single-request / per-prompt, **no serving-loop state** (no queue, KV cache, GPU load,
  admission, or chunking) — Nightjar explicitly calls its static design "incompatible with continuous
  batching."

---

## Positioning vs our idea

Our Lightweight Decision Model (LDM): a small model over **rich serving state** (queue depth & mix,
per-request acceptance history, KV occupancy/fragmentation, GPU util, SLO slack) that jointly decides, per
scheduling iteration: **{admission/ordering, per-request K, prefill chunk size, SD on/off + draft memory}**,
with a **prefill–decode / KV-cache coupling** and an **SLO/goodput objective**. What each paper does NOT do:

### Nightjar (2512.22420) — closest on "on/off + memory"
- **Does not** make K per-request: one γ for the whole batch; LDM sets **per-request K** from per-request
  acceptance history, exploiting intra-batch heterogeneity they explicitly ignore.
- **Does not** use rich state: bandit context is only batch size B (plus realized goodput); no per-request
  features, no KV occupancy as a *decision* input (KV only triggers the global offload), no GPU-util or
  queue-mix signal. LDM's model consumes all of these jointly.
- **Does not** couple with chunked prefill: prefill appears only as `C_switch` (reconstruction cost); LDM
  sizes the prefill chunk *aware of* the K× decode verify load sharing the same step budget.
- **Does not** do admission control: it reacts globally (on/off + offload) but never decides *which*
  requests to admit/reorder per iteration; LDM does joint admission + K.
- **Objective gap:** minimizes latency-per-token with throughput as reward; no TTFT/TPOT SLO constraint or
  goodput-under-SLO. LDM optimizes goodput subject to SLOs (their stated "SLO" is only a related-work
  reference, not an objective).

### DSDE (2509.01083) — closest on "per-request K"
- **Does not** do on/off or admission: SL floored at 2 (speculation always on), no request admission/
  ordering; LDM can turn SD off per request and admit/reorder.
- **Signal is post-hoc & single-source:** KLD variance (one-step lag, weak token-level correlation by their
  own study) vs LDM's forward-looking per-request acceptance history + system state.
- **Does not** size prefill chunks: lookahead KV slots are applied uniformly to prefill/decode/chunked-
  prefill for allocation consistency only; no dynamic chunk policy coupled to K.
- **No KV-aware admission / memory decision:** they reallocate per-sequence lookahead KV but never couple it
  to admission or draft-memory residency (contrast Nightjar's offload); LDM treats KV occupancy + draft
  memory as first-class state.
- **Runs eager, no CUDA graphs** — a deployment handicap; LDM must solve the dynamic-K-under-CUDA-graphs
  problem (piecewise capture / per-request K buckets) that DSDE punts on.
- **Objective gap:** heuristic SL rule + MSE cap; no SLO/goodput objective, no latency term in an explicit
  optimization.

### DSpark (2607.05147) — closest on "per-request verify length under load"
- **Does not** decide draft K: γ is fixed at deploy time; the scheduler only prunes *verification* length
  l_r of an already-drafted block, so it pays the full draft cost even for hopeless requests (their stated
  limitation). LDM can set per-request **K** (and turn SD off) so low-value requests don't pay draft cost.
- **"Load" is a static profiled curve SPS(B), not live state:** no real-time queue, KV occupancy, GPU util,
  or per-request acceptance history in the decision; LDM reads live serving state each iteration.
- **Tied to a proprietary engine + a trained confidence head** (STS-calibrated); not portable to vLLM's
  stock drafter/metrics without retraining a confidence head. LDM is a lightweight model over observable
  serving metrics (works with EAGLE/ngram/MTP and synthetic acceptance profiles).
- **Objective gap:** maximizes expected token throughput Θ=τ·SPS(B); SLA anchors are evaluation-only, not in
  the objective; no admission, no prefill-chunk coupling, no KV-aware memory decision. LDM optimizes
  goodput under SLOs with those couplings explicit.

### Cross-cutting gaps LDM owns (no paper covers all of these jointly)
1. **Joint decision vector** {admission/ordering, per-request K, prefill chunk size, SD on/off + draft
   memory} in one iteration from shared state — every paper pulls exactly one knob (K, or on/off+memory,
   or verify-length, or token selection).
2. **Rich live serving state** (queue mix, per-request acceptance history, KV occupancy/fragmentation, GPU
   util, SLO slack) as model inputs — prior signals are load-only (Nightjar), post-hoc KLD (DSDE), or a
   profiled curve + drafter confidence (DSpark).
3. **Prefill–decode & KV-cache coupling** under SD — chunk budget shared with ~K× decode verify load, and
   per-request lookahead-KV vs admission; unaddressed by all three (vLLM splits on `max_num_batched_tokens`
   blind to K).
4. **SLO/goodput objective** with per-request heterogeneity — prior objectives are throughput or
   latency-per-token in isolation; goodput under TTFT/TPOT SLOs is open.

### Baseline design implications (for our eval)
- **Nightjar** = batch-level bandit K + global on/off/draft-offload → implement as a vLLM fork baseline;
  expect it to win the *memory-reclamation* regime (high load, KV-tight) but lose intra-batch
  heterogeneity and any SLO-constrained goodput.
- **DSDE** = per-request KLD-variance SL + mean cap → strongest per-request-K baseline in eager mode;
  expect it to expose the CUDA-graph/eager overhead gap and the post-hoc-signal lag under non-stationary load.
- **DSpark scheduler** = per-request verify-length from a profiled SPS(B) → use its *greedy top-K over
  survival probs* as an oracle-ish baseline for the "verify budget allocation" sub-problem; note it needs a
  confidence head (we can substitute vLLM's synthetic acceptance profiles or EAGLE draft probabilities).
- **TETRIS / BanditSpec** = per-request token selection / per-prompt config → orthogonal, composable
  baselines for the "what to verify" and "which config" axes respectively.
