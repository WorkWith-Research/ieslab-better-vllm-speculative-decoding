#!/usr/bin/env python3
"""H-OBS-1 experiment-arm resolver — table-driven, unit-testable.

Directive §3A: separate the shared DSpark *model* arguments from the mutually
exclusive *controller-mode* selection; reject unknown arms; make instrumentation
(tracer) ON/OFF a separate axis; and give every arm an unambiguous label.

This is the single source of truth for what each arm means. The launcher
(run_hoobs1.sh) shells out to `resolve` so the exact same logic that is unit-
tested here is what actually launches servers. Run `--selftest` to verify every
arm resolves to its required effective behavior (a regression test).

Arms (policy axis):
  ar        : NO speculative model configured; NO live-controller policy active.
  dspark    : learned DSpark drafter, fixed K=7, native mode (always-on SD).
  ctrl_live : same learned drafter + the frozen Track B live-state controller.

Instrumentation axis (separate): tracer on/off. The historical label
`dspark_off` meant "tracer OFF" (SD still ON) — it is NOT an arm here; it maps
to arm=dspark, tracer=off. We keep that mapping documented and use unambiguous
labels going forward.
"""
import json
import sys

# The learned DSpark drafter + fixed K=7 (identical for both SD arms).
DSPEC_CFG = {
    "method": "dspark",
    "model": "deepseek-ai/dspark_qwen3_4b_block7",
    "num_speculative_tokens": 7,
}

# Table: arm -> effective behavior. tb_mode None means VLLM_TB_MODE is UNSET
# (no controller policy). drafter False means no --speculative-config at all.
ARM_TABLE = {
    "ar": {
        "description": "Autoregressive baseline: no speculative model, no controller.",
        "drafter": False,
        "tb_mode": None,          # unset -> no live policy (num_spec_tokens==0 anyway)
        "sd_expected": "off",     # no drafts ever scheduled
    },
    "dspark": {
        "description": "Native fixed-K DSpark: learned drafter K=7, always-on (native mode).",
        "drafter": True,
        "tb_mode": "native",      # zero-overhead always-on bypass = true baseline arm
        "sd_expected": "on",
    },
    "ctrl_live": {
        "description": "Native DSpark drafter K=7 + frozen Track B live-state SD on/off controller.",
        "drafter": True,
        "tb_mode": "live",        # the frozen controller decides per step
        "sd_expected": "on-off",  # controller may turn SD off at saturation
    },
}


def resolve(arm: str, tracer: str = "on") -> dict:
    """Return the effective configuration for (arm, tracer). Raises on unknown."""
    if arm not in ARM_TABLE:
        raise ValueError(
            f"unknown arm {arm!r}; valid arms: {sorted(ARM_TABLE)}"
        )
    if tracer not in ("on", "off"):
        raise ValueError(f"tracer must be 'on' or 'off', got {tracer!r}")
    spec = ARM_TABLE[arm]
    env = {}
    # Controller mode (mutually exclusive axis). UNSET for AR -> no policy.
    if spec["tb_mode"] is not None:
        env["VLLM_TB_MODE"] = spec["tb_mode"]
    # Instrumentation axis (separate from policy). Tracer off => unset.
    # VLLM_OBS_TRACE path is filled in by the launcher with the per-cell dir.
    if tracer == "on":
        env["VLLM_OBS_TRACE"] = "<per-cell-dir>"
    return {
        "arm": arm,
        "tracer": tracer,
        "description": spec["description"],
        # CLI args to append to the api_server command.
        "cli_args": (
            ["--speculative-config", json.dumps(DSPEC_CFG)] if spec["drafter"] else []
        ),
        # Env vars to set (and any that MUST be cleared) for a clean subprocess.
        "env_set": env,
        # Env vars that must be UNSET regardless of inheritance, so a stale
        # value from the parent shell cannot silently change the arm.
        "env_clear": ["VLLM_TB_MODE", "VLLM_OBS_TRACE", "VLLM_TB_TRACE"],
        "sd_expected": spec["sd_expected"],
    }


def _selftest() -> int:
    """Regression test: assert every arm resolves to its required behavior."""
    failures = []

    # AR: no drafter, no controller mode set.
    ar = resolve("ar")
    if any("speculative-config" in a for a in ar["cli_args"]):
        failures.append("AR must not configure a speculative model")
    if "VLLM_TB_MODE" in ar["env_set"]:
        failures.append("AR must not enable a live-controller policy")

    # DSpark: drafter present, native mode.
    ds = resolve("dspark")
    if not any("speculative-config" in a for a in ds["cli_args"]):
        failures.append("DSpark must configure the learned drafter")
    if ds["env_set"].get("VLLM_TB_MODE") != "native":
        failures.append("DSpark must run in native (always-on) mode")

    # CTRL-LIVE: same drafter, live controller.
    cl = resolve("ctrl_live")
    if not any("speculative-config" in a for a in cl["cli_args"]):
        failures.append("CTRL-LIVE must configure the learned drafter")
    if cl["env_set"].get("VLLM_TB_MODE") != "live":
        failures.append("CTRL-LIVE must enable the frozen live controller")

    # DSpark and CTRL-LIVE share the identical drafter args (only mode differs).
    if ds["cli_args"] != cl["cli_args"]:
        failures.append("DSpark and CTRL-LIVE must use the identical drafter config")

    # Tracer is a separate axis: toggling it must not change policy/drafter.
    for arm in ARM_TABLE:
        on, off = resolve(arm, "on"), resolve(arm, "off")
        if on["cli_args"] != off["cli_args"]:
            failures.append(f"{arm}: tracer must not change drafter args")
        if on["env_set"].get("VLLM_TB_MODE") != off["env_set"].get("VLLM_TB_MODE"):
            failures.append(f"{arm}: tracer must not change controller mode")
        if "VLLM_OBS_TRACE" in off["env_set"]:
            failures.append(f"{arm}: tracer=off must leave VLLM_OBS_TRACE unset")

    # Unknown arms are rejected.
    for bad in ("dspark_off", "DARK", "", "live"):
        try:
            resolve(bad)
            failures.append(f"unknown arm {bad!r} was NOT rejected")
        except ValueError:
            pass

    if failures:
        print("ARM SELFTEST FAIL:")
        for f in failures:
            print("  -", f)
        return 1
    print("ARM SELFTEST OK: all arms resolve to their required effective behavior")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        raise SystemExit(_selftest())
    # Otherwise print resolved config for a given arm/tracer (used by the launcher).
    arm = sys.argv[1] if len(sys.argv) > 1 else "dspark"
    tracer = sys.argv[2] if len(sys.argv) > 2 else "on"
    print(json.dumps(resolve(arm, tracer), indent=2))
