# Documentation Index & Research Map

This directory contains pre-registrations, experimental analyses, empirical results, and code audits for the speculative decoding research program. Documents are organized into dedicated thematic subdirectories, with root symlinks preserved for backwards compatibility.

---

## 📁 Directory Structure

```text
docs/
├── audits/            # Literature surveys, architectural audits, and upstream DSpark code audits
├── methodology/       # Instrumentation plans, load characterization, and synthesis notes
├── phases/            # Phase-by-phase experimental investigations (Phases 1 through 7)
│   ├── phase1_3/      # Static sweeps and n-gram baselines
│   ├── phase4/        # K × Concurrency load matrices and saturation discovery
│   ├── phase5/        # Batch-level LDM externality pricing
│   ├── phase6/        # Native DSpark validation & proxy correction
│   └── phase7/        # Phase 7 capstone 2-axis controller & final results
└── benchmarks/        # Specialized micro-benchmarks
    ├── hoobs/         # H-OBS nanosecond Nsight Systems (nsys) & step GPU timing
    ├── hclass/        # H-CLASS prompt reuse and prefix caching classes
    ├── mbatch/        # M-BATCH live mixed-batch payoff evaluations
    └── tracks/        # Track A (ML predictors) & Track B (In-engine controllers)
```

---

## 1. Audits & Literature Surveys (`audits/`)
- [`audits/literature-map.md`](./audits/literature-map.md): Landscape of speculative decoding literature.
- [`audits/paper-briefs.md`](./audits/paper-briefs.md): Key summaries of seminal papers (DSpark, Lookahead, Medusa, etc.).
- [`audits/related-work-adaptive-sd.md`](./audits/related-work-adaptive-sd.md): Survey of adaptive speculative decoding algorithms.
- [`audits/related-work-dspark.md`](./audits/related-work-dspark.md): Deep dive into DeepSeek DSpark architecture.
- [`audits/vllm-code-audit.md`](./audits/vllm-code-audit.md): Complete audit of vLLM's internal speculative decoding pipeline.
- [`audits/dspark-native-audit.md`](./audits/dspark-native-audit.md): Upstream vLLM DSpark integration audit and hardware runnability gates.
- [`audits/dspark-proxy-validity.md`](./audits/dspark-proxy-validity.md): Audit identifying F1/F4 bugs in early proxy models.
- [`audits/dspark-audit-synthesis.md`](./audits/dspark-audit-synthesis.md): Strategic synthesis and pivot to live serving-state controllers.

---

## 2. Methodology & Observability Plans (`methodology/`)
- [`methodology/observation-plan.md`](./methodology/observation-plan.md): Initial instrumentation and telemetry plan.
- [`methodology/serving-load-characterization.md`](./methodology/serving-load-characterization.md): Characterization of concurrency and queue dynamics.
- [`methodology/replay-vs-live-gap.md`](./methodology/replay-vs-live-gap.md): Analysis of offline replay vs live continuous-batching execution.
- [`methodology/skeptic-pass-2026-10-07.md`](./methodology/skeptic-pass-2026-10-07.md) & [`methodology/synthesis-2026-10-07.md`](./methodology/synthesis-2026-10-07.md): Initial methodological critiques and synthesis.
- [`methodology/scheduler-signal-validation.md`](./methodology/scheduler-signal-validation.md): Static payoff surface validation.

---

## 3. Phases (`phases/`)

### Phase 1–3: Static Sweeps & Baselines (`phases/phase1_3/`)
- [`phases/phase1_3/phase1-results.md`](./phases/phase1_3/phase1-results.md): Offline acceptance and latency profiling across workloads.
- [`phases/phase1_3/phase2-results.md`](./phases/phase1_3/phase2-results.md): Variable-concurrency sweeps on templated vs diverse streams.
- [`phases/phase1_3/phase3-results.md`](./phases/phase1_3/phase3-results.md): Fixed-K vs dynamic K feasibility on synthetic workloads.

### Phase 4: Load & Saturation Regime Discovery (`phases/phase4/`)
- [`phases/phase4/p2-load-matrix-results.md`](./phases/phase4/p2-load-matrix-results.md) & [`phases/phase4/phase4-p2-results.md`](./phases/phase4/phase4-p2-results.md): K × Concurrency load matrix showing optimal K collapses under saturation.
- [`phases/phase4/phase4-p3-results.md`](./phases/phase4/phase4-p3-results.md): SPEED-Bench 2k long-prefill verification vs chunked prefill competition.
- [`phases/phase4/phase4-p44-results.md`](./phases/phase4/phase4-p44-results.md): Profiling step costs across token batch sizes.
- [`phases/phase4/phase4-p5-results.md`](./phases/phase4/phase4-p5-results.md): Verification cost non-linearity under high batch occupancy.
- [`phases/phase4/phase4-p6-results.md`](./phases/phase4/phase4-p6-results.md): Unified Step-Per-Second (SPS) curve fitting across decode batch size $B$.

### Phase 5: Batch-Level Cost & Externality Pricing (`phases/phase5/`)
- [`phases/phase5/phase5-p1-results.md`](./phases/phase5/phase5-p1-results.md): Batch-level dynamic controller pricing the externality of drafts.
- [`phases/phase5/p52-prereg.md`](./phases/phase5/p52-prereg.md) & [`phases/phase5/p52-results.md`](./phases/phase5/p52-results.md): Multi-query latency and step-time scaling validation.
- [`phases/phase5/p52-scope-note.md`](./phases/phase5/p52-scope-note.md): Scope and boundaries of the empirical cost model.

### Phase 6: Native DSpark Validation (`phases/phase6/`)
- [`phases/phase6/p45r-results.md`](./phases/phase6/p45r-results.md): Rerun of proxy models under corrected cost accounting.
- [`phases/phase6/dspark-native-results.md`](./phases/phase6/dspark-native-results.md): Native DSpark ($K=7$) benchmarks on Qwen3-4B under isolated runtime.
- [`phases/phase6/phase6-reproduction-prereg.md`](./phases/phase6/phase6-reproduction-prereg.md): Reproduction protocol on isolated vLLM v0.25.0 build.

### Phase 7: Capstone Live Controller (`phases/phase7/`)
- [`phases/phase7/phase7-capstone-prereg.md`](./phases/phase7/phase7-capstone-prereg.md): Pre-registration of the 2-axis controller (`sd_enabled <- load_LOW OR class_HIGHREUSE`).
- [`phases/phase7/phase7-capstone-results.md`](./phases/phase7/phase7-capstone-results.md): Final experimental verdict on live mixed-batch serving.

---

## 4. Benchmarks (`benchmarks/`)

### Hardware Observability (`benchmarks/hoobs/`)
- [`benchmarks/hoobs/hoobs1-prereg.md`](./benchmarks/hoobs/hoobs1-prereg.md) & [`benchmarks/hoobs/hoobs1-observability.md`](./benchmarks/hoobs/hoobs1-observability.md): Nanosecond-resolution Nsight Systems (nsys) step-level profiling.
- [`benchmarks/hoobs/hoobs3-prereg.md`](./benchmarks/hoobs/hoobs3-prereg.md) & [`benchmarks/hoobs/hoobs3-results.md`](./benchmarks/hoobs/hoobs3-results.md): Heterogeneous input/output length profiling on live DSpark serving.

### Traffic Classes & Prefix Caching (`benchmarks/hclass/`)
- [`benchmarks/hclass/hclass1-prereg.md`](./benchmarks/hclass/hclass1-prereg.md) & [`benchmarks/hclass/hclass1-results.md`](./benchmarks/hclass/hclass1-results.md): Prefix-caching traffic classes (reusable vs diverse prompts).
- [`benchmarks/hclass/hclass1c-prereg.md`](./benchmarks/hclass/hclass1c-prereg.md) & [`benchmarks/hclass/hclass1c-results.md`](./benchmarks/hclass/hclass1c-results.md): Shared context doc cache survival under diverse eviction pressure.
- [`benchmarks/hclass/hclass1d-prereg.md`](./benchmarks/hclass/hclass1d-prereg.md) & [`benchmarks/hclass/hclass1d-results.md`](./benchmarks/hclass/hclass1d-results.md): Held-out load boundary probes.

### Mixed Batches (`benchmarks/mbatch/`)
- [`benchmarks/mbatch/mbatch1-prereg.md`](./benchmarks/mbatch/mbatch1-prereg.md) & [`benchmarks/mbatch/mbatch1-results.md`](./benchmarks/mbatch/mbatch1-results.md): Live mixed-stream payoff surface across concurrent traffic classes.

### Controller Tracks (`benchmarks/tracks/`)
- [`benchmarks/tracks/tracka-prereg.md`](./benchmarks/tracks/tracka-prereg.md) & [`benchmarks/tracks/tracka-results.md`](./benchmarks/tracks/tracka-results.md): Track A offline MLP/regressor predictor.
- [`benchmarks/tracks/trackb-feasibility.md`](./benchmarks/tracks/trackb-feasibility.md), [`benchmarks/tracks/trackb-prereg.md`](./benchmarks/tracks/trackb-prereg.md), & [`benchmarks/tracks/trackb-results.md`](./benchmarks/tracks/trackb-results.md): Track B in-engine live-state SD on/off controller with probe re-entry.
