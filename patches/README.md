# vLLM DSpark & Speculative Decoding Controllers Patch

This directory contains the custom scheduler controllers, instrumentation, and test suites developed for dynamic Speculative Decoding (SD) under live serving state, as validated in the companion runtime environment (`dspark-native-validation`).

## Base Version
- **Target upstream**: [vLLM](https://github.com/vllm-project/vllm) tag `v0.25.0` (commit `702f481`).
- **Hardware & Toolchain compatibility**: CUDA 12.6, PyTorch 2.11+, GCC 13 (built on RTX 3090 / SM86).

---

## Contents

### 1. Patch File
- [`vllm-v0.25.0-track-b-live-sd.patch`](./vllm-v0.25.0-track-b-live-sd.patch): The unified git patch containing all 10 commits on branch `track-b-live-sd` (Phase 7 capstone, Track B live controller, per-step CUDA timing, and H-OBS instrumentation).

### 2. Standalone Controller Source (`vllm-controllers/`)
For direct review and reuse without needing a full vLLM tree:
- **`sched/`**:
  - `phase7_controller.py`: Phase 7 Capstone request-aware 2-axis SD controller (`sd_enabled <- load_LOW OR class_HIGHREUSE`) driven by measured KV-cache usage and prefix-cache admission hit-rate.
  - `tb_controller.py`: Track B live-state SD controller (per-step cost comparison and probe re-entry).
  - `obs_tracer.py`: Per-step observation tracer dumping JSONL telemetry of scheduler state, drafts, and GPU step time.
- **`tests/`**:
  - `test_phase7_controller.py`: 24 unit tests covering 2x2 payoff surface, hysteresis, windowed signals, and causal draft verification.
  - `test_tb_controller.py`: 20 unit tests covering Track B state machine and probe dynamics.
- **`scripts/`**:
  - Profiling, Nsight Systems analyzers, and runner harnesses (`run_hoobs1.sh`, `run_hoobs3.sh`, `hoobs1_nsys_analyze.py`, `hoobs1_arms.py`).

---

## How to Apply the Patch

To reproduce the validation runtime:

```bash
# 1. Clone upstream vLLM at tag v0.25.0
git clone https://github.com/vllm-project/vllm.git vllm-runtime
cd vllm-runtime
git checkout v0.25.0

# 2. Apply this patch
git apply ../patches/vllm-v0.25.0-track-b-live-sd.patch

# 3. Run controller unit tests
pytest tests/v1/core/test_phase7_controller.py tests/v1/core/test_tb_controller.py
```
