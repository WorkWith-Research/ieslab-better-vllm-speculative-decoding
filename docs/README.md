# Documentation Index & Research Map

This directory contains pre-registrations, experimental analyses, empirical results, and code audits for the speculative decoding research program.

---

## 1. Literature & Architecture Audits
- [`literature-map.md`](./literature-map.md): Landscape of speculative decoding literature.
- [`paper-briefs.md`](./paper-briefs.md): Key summaries of seminal papers (DSpark, Lookahead, Medusa, etc.).
- [`related-work-adaptive-sd.md`](./related-work-adaptive-sd.md): Survey of adaptive speculative decoding algorithms.
- [`related-work-dspark.md`](./related-work-dspark.md): Deep dive into DeepSeek DSpark architecture.
- [`vllm-code-audit.md`](./vllm-code-audit.md): Complete audit of vLLM's internal speculative decoding pipeline.
- [`replay-vs-live-gap.md`](./replay-vs-live-gap.md): Analysis of offline replay vs live continuous-batching execution.
- [`skeptic-pass-2026-10-07.md`](./skeptic-pass-2026-10-07.md) & [`synthesis-2026-10-07.md`](./synthesis-2026-10-07.md): Initial methodological critiques and synthesis.

---

## 2. Phase 1–3: Static Sweeps & N-gram Baselines
- [`observation-plan.md`](./observation-plan.md): Initial instrumentation and telemetry plan.
- [`serving-load-characterization.md`](./serving-load-characterization.md): Characterization of concurrency and queue dynamics.
- [`phase1-results.md`](./phase1-results.md): Offline acceptance and latency profiling across workloads.
- [`phase2-results.md`](./phase2-results.md): Variable-concurrency sweeps on templated vs diverse streams.
- [`phase3-results.md`](./phase3-results.md): Fixed-K vs dynamic K feasibility on synthetic workloads.

---

## 3. Phase 4: Load & Saturation Regime Discovery
- [`p2-load-matrix-results.md`](./p2-load-matrix-results.md) & [`phase4-p2-results.md`](./phase4-p2-results.md): K × Concurrency load matrix showing optimal K collapses under saturation.
- [`phase4-p3-results.md`](./phase4-p3-results.md): SPEED-Bench 2k long-prefill verification vs chunked prefill competition.
- [`phase4-p44-results.md`](./phase4-p44-results.md): Profiling step costs across token batch sizes.
- [`phase4-p5-results.md`](./phase4-p5-results.md): Verification cost non-linearity under high batch occupancy.
- [`phase4-p6-results.md`](./phase4-p6-results.md): Unified Step-Per-Second (SPS) curve fitting across decode batch size $B$.

---

## 4. Phase 5: Batch-Level Cost & Externality Pricing (LDM)
- [`phase5-p1-results.md`](./phase5-p1-results.md): Batch-level dynamic controller pricing the externality of drafts.
- [`p52-prereg.md`](./p52-prereg.md) & [`p52-results.md`](./p52-results.md): Multi-query latency and step-time scaling validation.
- [`p52-scope-note.md`](./p52-scope-note.md): Scope and boundaries of the empirical cost model.

---

## 5. DSpark Audits & Upstream Native Validation
- [`dspark-native-audit.md`](./dspark-native-audit.md): Upstream vLLM DSpark integration audit and hardware runnability gates.
- [`dspark-proxy-validity.md`](./dspark-proxy-validity.md): Audit identifying F1/F4 bugs in early proxy models.
- [`p45r-results.md`](./p45r-results.md): Rerun of proxy models under corrected cost accounting.
- [`dspark-native-results.md`](./dspark-native-results.md): Native DSpark ($K=7$) benchmarks on Qwen3-4B under isolated runtime.
- [`dspark-audit-synthesis.md`](./dspark-audit-synthesis.md): Strategic synthesis and pivot to live serving-state controllers.
- [`phase6-reproduction-prereg.md`](./phase6-reproduction-prereg.md): Reproduction protocol on isolated vLLM v0.25.0 build.

---

## 6. Hardware Observability & Micro-Profiling (H-OBS)
- [`hoobs1-prereg.md`](./hoobs1-prereg.md) & [`hoobs1-observability.md`](./hoobs1-observability.md): Nanosecond-resolution Nsight Systems (nsys) step-level profiling.
- [`hoobs3-prereg.md`](./hoobs3-prereg.md) & [`hoobs3-results.md`](./hoobs3-results.md): Heterogeneous input/output length profiling on live DSpark serving.

---

## 7. Traffic Classes & Mixed-Batch Validation (H-CLASS & M-BATCH)
- [`hclass1-prereg.md`](./hclass1-prereg.md) & [`hclass1-results.md`](./hclass1-results.md): Prefix-caching traffic classes (reusable vs diverse prompts).
- [`hclass1c-prereg.md`](./hclass1c-prereg.md) & [`hclass1c-results.md`](./hclass1c-results.md): Shared context doc cache survival under diverse eviction pressure.
- [`hclass1d-prereg.md`](./hclass1d-prereg.md) & [`hclass1d-results.md`](./hclass1d-results.md): Held-out load boundary probes.
- [`mbatch1-prereg.md`](./mbatch1-prereg.md) & [`mbatch1-results.md`](./mbatch1-results.md): Live mixed-stream payoff surface across concurrent traffic classes.
- [`scheduler-signal-validation.md`](./scheduler-signal-validation.md): Static payoff surface validation.

---

## 8. Track A & Track B Development
- [`tracka-prereg.md`](./tracka-prereg.md) & [`tracka-results.md`](./tracka-results.md): Track A offline MLP/regressor predictor.
- [`trackb-feasibility.md`](./trackb-feasibility.md), [`trackb-prereg.md`](./trackb-prereg.md), & [`trackb-results.md`](./trackb-results.md): Track B in-engine live-state SD on/off controller with probe re-entry.

---

## 9. Phase 7: Capstone Live Controller & Final Verdict
- [`phase7-capstone-prereg.md`](./phase7-capstone-prereg.md): Pre-registration of the 2-axis controller (`sd_enabled <- load_LOW OR class_HIGHREUSE`).
- [`phase7-capstone-results.md`](./phase7-capstone-results.md): Final experimental verdict on live mixed-batch serving.
