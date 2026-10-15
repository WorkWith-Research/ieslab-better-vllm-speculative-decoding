# Experiments Directory Map

This directory contains the test harnesses, clients, data builders, analyzers, and early runtime controller hooks used across Phases 1 through 7 of the speculative decoding project.

---

## 1. Core Testbed & Server Management
- [`run_server.sh`](./run_server.sh): Script to configure and spawn the vLLM server (handles GPU allocation, ports, eager mode, and spec config).
- [`kill_server.sh`](./kill_server.sh): Process-tree cleaner to prevent orphaned vLLM engine background processes.
- [`scrape_metrics.py`](./scrape_metrics.py): Asynchronous scraper logging Prometheus metrics and GPU utilization.

---

## 2. Workload Generators & Client Harnesses
- **Mixed & Poisson Streams**:
  - [`mixed_client.py`](./mixed_client.py), [`mixed_workload_client.py`](./mixed_workload_client.py): Mixed synthetic workloads (templated high-acceptance + diverse low-acceptance).
  - [`phase7_mixed_client.py`](./phase7_mixed_client.py): Phase 7 mixed-stream benchmark client with controlled prompt reuse.
  - [`mbatch1_client.py`](./mbatch1_client.py): Open-loop mixed-stream client for M-BATCH-1.
- **Heterogeneous & Trace-Driven**:
  - [`hetero_client.py`](./hetero_client.py), [`hetero_client_qwen3.py`](./hetero_client_qwen3.py): Heterogeneous ISL/OSL request distributions.
  - [`sharegpt_client.py`](./sharegpt_client.py): Real-world ShareGPT request replay.
  - [`speed_bench_client.py`](./speed_bench_client.py): Long-prompt (2k ISL) client measuring prefill competition.
  - [`latency_client.py`](./latency_client.py): Per-request TTFT, TPOT, and E2E latency recorder.
- **Traffic Class Banks (H-CLASS-1)**:
  - [`build_hclass1_prompt_bank.py`](./build_hclass1_prompt_bank.py), [`build_hclass1_c_bank.py`](./build_hclass1_c_bank.py), [`build_hclass1_longpool.py`](./build_hclass1_longpool.py): Generates class A (independent), B (reused prompt), and C (diverse query with shared context document) datasets.
  - [`hclass1_client.py`](./hclass1_client.py), [`hclass1_prompts.py`](./hclass1_prompts.py): H-CLASS-1 client runner.

---

## 3. Experiment Matrix Runners
- **Phase 1 & 2 (Initial Profiling & Sweeps)**:
  - `phase1_sweep*.sh`, `phase2_mixed.sh`, `p1_saturation.sh`, `p1b_saturation_high.sh`, `p2_grid.sh`, `p2_cell.sh`.
- **Phase 4 (Load Saturation & Step Profiling)**:
  - `p44_grid.sh`, `p44_cell.sh`, `p45r_grid.sh`, `p45r_cell.sh`, `p46_profile.sh`, `p46_cell.sh`.
- **Phase 5 (Batch-level LDM Externality)**:
  - `p5_grid.sh`, `p5_cell.sh`, `p51_grid.sh`, `p51_cell.sh`, `p51_controls.sh`, `p52_*.sh`.
- **H-CLASS & M-BATCH (Traffic Classes & Prefix Cache)**:
  - `hclass1_smoke.sh`, `hclass1_run.sh`, `mbatch1_run.sh`.
- **Phase 7 Capstone**:
  - `phase7_smoke.sh`, `phase7_premise_check.sh`, `phase7_s3_run.sh`.

---

## 4. Offline Analysis Pipelines
- `analyze_p2.py`, `analyze_p3.py`, `analyze_p44.py`, `analyze_p45r.py`, `analyze_p46.py`, `analyze_p46_server.py`, `analyze_p5.py`, `analyze_pa_grid.py`.
- [`hclass1_analyze.py`](./hclass1_analyze.py): Analyzes prefix-cache hit-rate distributions across classes A, B, and C.
- [`hoobs1_analyze.py`](./hoobs1_analyze.py), [`hoobs3_analyze.py`](./hoobs3_analyze.py): Nsight Systems step-correlation and GPU execution breakdowns.
- [`mbatch1_analyze.py`](./mbatch1_analyze.py): Payoff surface analysis across mixed batch compositions.

---

## 5. Early Injection Hooks (`sitecustomize.py`)
These packages monkey-patch early vLLM versions (v0.19.1) without modifying source directly:
- `dspark_controller/`, `dsparkr_controller/`: Early DSpark budget proxy hooks.
- `ldm_controller/`, `ldm_batch_controller/`, `ldmload_controller/`: Phase 4 & 5 LDM controllers.
- `spec_hook/`: Hooks into vLLM speculative stats generator.
- `p52_obs/`: Step tracer port for v0.19.1.
