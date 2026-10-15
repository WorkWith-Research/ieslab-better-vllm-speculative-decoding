# Experiments Directory Map

This directory contains the test harnesses, clients, data builders, analyzers, and early runtime controller hooks used across Phases 1 through 7 of the speculative decoding project. Files are organized into functional subdirectories, with root symlinks preserved for backwards compatibility.

---

## 📁 Directory Structure

```text
experiments/
├── clients/       # Workload generators, traffic replay, and benchmark clients
├── runners/       # Benchmark runners, parameter sweeps, and execution shell scripts
├── analysis/      # Offline metrics parsers, nsys correlators, and visualization scripts
├── controllers/   # Legacy vLLM v0.19.1 sitecustomize controller hooks & test scripts
├── data_prep/     # Prompt bank builders and dataset synthesis
├── track_a/       # Track A offline machine learning data collection & model training
└── utils/         # Server process management, metric scrapers, and profiling tools
```

---

## 1. Workload Generators & Clients (`clients/`)
- [`clients/phase7_mixed_client.py`](./clients/phase7_mixed_client.py): Phase 7 mixed-stream benchmark client with controlled prompt reuse.
- [`clients/mbatch1_client.py`](./clients/mbatch1_client.py): Open-loop mixed-stream client for M-BATCH-1.
- [`clients/mixed_client.py`](./clients/mixed_client.py), [`clients/mixed_workload_client.py`](./clients/mixed_workload_client.py): Mixed Poisson workloads (templated high-acceptance + diverse low-acceptance).
- [`clients/hetero_client.py`](./clients/hetero_client.py), [`clients/hetero_client_qwen3.py`](./clients/hetero_client_qwen3.py): Heterogeneous ISL/OSL request distributions.
- [`clients/sharegpt_client.py`](./clients/sharegpt_client.py): Real-world ShareGPT request replay.
- [`clients/speed_bench_client.py`](./clients/speed_bench_client.py): Long-prompt (2k ISL) client measuring prefill competition.
- [`clients/latency_client.py`](./clients/latency_client.py): Per-request TTFT, TPOT, and E2E latency recorder.
- [`clients/hclass1_client.py`](./clients/hclass1_client.py), [`clients/hclass1_prompts.py`](./clients/hclass1_prompts.py): H-CLASS-1 client runner.
- [`clients/spectrum_client.py`](./clients/spectrum_client.py): Multi-length prompt spectrum client.

---

## 2. Benchmark Runners & Sweeps (`runners/`)
- **Phase 1 & 2**: `phase1_sweep*.sh`, `phase2_mixed.sh`, `p1_saturation.sh`, `p1b_saturation_high.sh`, `p2_grid.sh`, `p2_cell.sh`, `ae2_sharegpt_screen.sh`, `ae3_graph_mode.sh`, `e1_coupling.sh`, `pa_grid.sh`, `pc_draft_model.sh`.
- **Phase 4**: `p44_grid.sh`, `p44_cell.sh`, `p45r_grid.sh`, `p45r_cell.sh`, `p46_cell.sh`, `p46_profile.sh`.
- **Phase 5**: `p5_grid.sh`, `p5_cell.sh`, `p51_grid.sh`, `p51_cell.sh`, `p51_controls.sh`, `p52_p1_grid.sh`, `p52_p3_grid.sh`, `p52_sf_smoke.sh`, `p52_smoke.sh`, `ldm_ab.sh`.
- **H-CLASS & M-BATCH**: `hclass1_smoke.sh`, `hclass1_run.sh`, `mbatch1_run.sh`.
- **Phase 7 Capstone**: [`runners/phase7_smoke.sh`](./runners/phase7_smoke.sh), [`runners/phase7_premise_check.sh`](./runners/phase7_premise_check.sh), [`runners/phase7_s3_run.sh`](./runners/phase7_s3_run.sh).
- **General**: [`runners/run_experiment.sh`](./runners/run_experiment.sh).

---

## 3. Offline Analysis Pipelines (`analysis/`)
- [`analysis/hclass1_analyze.py`](./analysis/hclass1_analyze.py): Analyzes prefix-cache hit-rate distributions across classes A, B, and C.
- [`analysis/hoobs1_analyze.py`](./analysis/hoobs1_analyze.py), [`analysis/hoobs3_analyze.py`](./analysis/hoobs3_analyze.py): Nsight Systems step-correlation and GPU execution breakdowns.
- [`analysis/mbatch1_analyze.py`](./analysis/mbatch1_analyze.py): Payoff surface analysis across mixed batch compositions.
- Phase-specific analyzers: `analyze_p2.py`, `analyze_p3.py`, `analyze_p44.py`, `analyze_p45r.py`, `analyze_p46.py`, `analyze_p46_server.py`, `analyze_p5.py`, `analyze_pa_grid.py`, `oracle_gap.py`, `p52_analyze_p1.py`, `p52_analyze_p3.py`.

---

## 4. Server Utilities & Micro-Profiling (`utils/`)
- [`utils/run_server.sh`](./utils/run_server.sh): Script to configure and spawn the vLLM server (handles GPU allocation, ports, eager mode, and spec config).
- [`utils/kill_server.sh`](./utils/kill_server.sh): Process-tree cleaner to prevent orphaned vLLM engine background processes.
- [`utils/scrape_metrics.py`](./utils/scrape_metrics.py): Asynchronous scraper logging Prometheus metrics and GPU utilization.
- [`utils/profile_sps.py`](./utils/profile_sps.py): Measures steps-per-second scaling.
- [`utils/probe_gauges.py`](./utils/probe_gauges.py): Telemetry probe.

---

## 5. Early Controller Injection Hooks (`controllers/`)
These packages monkey-patch early vLLM versions (v0.19.1) without modifying source directly:
- `dspark_controller/`, `dsparkr_controller/`: Early DSpark budget proxy hooks.
- `ldm_controller/`, `ldm_batch_controller/`, `ldmload_controller/`: Phase 4 & 5 LDM controllers.
- `spec_hook/`: Hooks into vLLM speculative stats generator.
- `p52_obs/`: Step tracer port for v0.19.1.
- `test_dsparkr.py`, `ldm_eval.py`: Evaluation scripts.

---

## 6. Prompt Bank Builders (`data_prep/`)
- [`data_prep/build_hclass1_prompt_bank.py`](./data_prep/build_hclass1_prompt_bank.py), [`data_prep/build_hclass1_c_bank.py`](./data_prep/build_hclass1_c_bank.py), [`data_prep/build_hclass1_longpool.py`](./data_prep/build_hclass1_longpool.py): Generates class A (independent), B (reused prompt), and C (diverse query with shared context document) datasets.

---

## 7. Track A Machine Learning Models (`track_a/`)
- [`track_a/tracka_collect.py`](./track_a/tracka_collect.py): Offline training label collection.
- [`track_a/tracka_train.py`](./track_a/tracka_train.py): MLP model training & cross-validation.
