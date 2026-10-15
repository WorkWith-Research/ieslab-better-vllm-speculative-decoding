# Adaptive Speculative Decoding Under Serving Load & Dynamic State

A rigorous empirical research project evaluating whether **dynamic, serving-state-aware speculative decoding (SD)** can outperform static speculation schedules and pure autoregressive (AR) baselines under continuous-batching serving workloads.

---

## 📁 Repository Structure

```text
├── docs/                # 49 research documents (audits, pre-registrations, results, literature map)
│   ├── audits/          # Upstream code audits and literature surveys
│   ├── methodology/     # Telemetry plans, serving characterization, and syntheses
│   ├── phases/          # Phase 1 through 7 experimental records (phase1_3/ through phase7/)
│   ├── benchmarks/      # Micro-benchmarks (hoobs/, hclass/, mbatch/, tracks/)
│   └── README.md        # Comprehensive catalog and thematic guide to all documentation
├── experiments/         # Benchmarks, client workloads, analysis pipelines, and runtime hooks
│   ├── clients/         # Workload generators & benchmark clients (Poisson, ShareGPT, SPEED-Bench)
│   ├── runners/         # Benchmark runners, parameter sweeps, and execution shell scripts
│   ├── analysis/        # Offline metrics parsers, nsys correlators, and visualization scripts
│   ├── controllers/     # Legacy vLLM v0.19.1 sitecustomize controller hooks
│   ├── data_prep/       # Prompt bank builders and dataset synthesis
│   ├── track_a/         # Track A offline machine learning data collection & model training
│   ├── utils/           # Server process management, metric scrapers, and profiling tools
│   └── README.md        # Functional index of all experiment scripts
├── patches/             # Patches & standalone controller implementations for vLLM
│   ├── vllm-v0.25.0-track-b-live-sd.patch   # Complete patch against vLLM v0.25.0
│   ├── vllm-controllers/                   # Standalone controllers, tests, & profiling scripts
│   └── README.md                            # Instructions for building and testing the runtime
├── data/                # Prompt banks and generated benchmark datasets
├── results/             # Raw experimental outputs, telemetry logs, and JSONL traces
├── IDEA.md              # Original hypothesis and problem statement
└── PROGRESS.md          # Comprehensive chronological research log and experiment journal
```

> [!NOTE]
> For seamless backwards compatibility, all original paths in `docs/` and `experiments/` are preserved via relative symlinks pointing to their new categorized locations.

---

## 🔬 Research Overview & Key Findings

### 1. The Core Problem
Under low server load, speculative decoding provides massive latency speedups because GPU compute is memory-bandwidth bound. However, under high concurrency and saturated continuous batching, extra verification tokens create severe externalities: they inflate step times, waste memory bandwidth, and displace chunked prefills.

### 2. What We Tested
Across 7 research phases:
1. **Phases 1–3**: Characterized static $K$ speculation degradation on synthetic and ShareGPT workloads.
2. **Phase 4**: Discovered the optimal speculation budget $K^*$ collapses to 0 as batch occupancy and KV cache approach saturation.
3. **Phase 5 (LDM)**: Formulated batch-level externality pricing to dynamically throttle speculation.
4. **DSpark Native Audit & Observability (H-OBS)**: Audited upstream vLLM's native DSpark implementation (`method="dspark"`) and built microsecond-resolution CUDA event and Nsight Systems (nsys) GPU instrumentation.
5. **Phase 7 Capstone**: Implemented and benchmarked a live 2-axis controller in vLLM's scheduler:
   $$\text{sd\_enabled} \leftarrow \text{load\_LOW} \lor \text{class\_HIGHREUSE}$$
   measuring live KV-cache utilization and prefix-cache admission hit-rates.

### 3. Empirical Verdicts
- **Static Payoff Surfaces Fail to Transfer**: While prefix-reusable queries benefit from speculation in isolation, under live mixed-batch serving the class axis fails to overcome the batch slowdown.
- **Load-Only Control Wins**: Throttling or disabling speculative decoding strictly based on KV cache load (`CTRL-LOADONLY`) recovered substantial throughput at saturation (**+24.8% at c48** and **+36.2% at c96** over native fixed-K DSpark).

Full details are documented in [`docs/phases/phase7/phase7-capstone-results.md`](./docs/phases/phase7/phase7-capstone-results.md) and [`PROGRESS.md`](./PROGRESS.md).

---

## 🛠️ Validation Runtime & Patches

The live runtime experiments were validated against an isolated build of vLLM (`v0.25.0`) with custom scheduling controllers. All custom modifications from that runtime are committed in [`patches/`](./patches/):
- **Full patch**: [`patches/vllm-v0.25.0-track-b-live-sd.patch`](./patches/vllm-v0.25.0-track-b-live-sd.patch)
- **Standalone code**: [`patches/vllm-controllers/`](./patches/vllm-controllers/)
  - [`phase7_controller.py`](./patches/vllm-controllers/sched/phase7_controller.py)
  - [`tb_controller.py`](./patches/vllm-controllers/sched/tb_controller.py)
  - [`obs_tracer.py`](./patches/vllm-controllers/sched/obs_tracer.py)
  - Unit tests in [`patches/vllm-controllers/tests/`](./patches/vllm-controllers/tests/)

To apply the patch to a fresh vLLM clone, refer to [`patches/README.md`](./patches/README.md).
