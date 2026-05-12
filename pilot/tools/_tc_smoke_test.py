"""Quick smoke test for `_tuning_config.TuningConfig`. Not part of CI; safe to delete."""
from __future__ import annotations
import subprocess
import sys

from pilot.tools._tuning_config import TuningConfigError, load_tuning_config


def main() -> int:
    subprocess.run(
        [
            sys.executable, "-m", "pilot", "session", "init",
            "--plan", "examples/megatron/configs/MI355X/deepseek_v2_lite-FP8-pretrain.yaml",
            "--session-id", "_tc_test",
            "--force",
            "--base-override", "micro_batch_size=1",
            "--base-override", "global_batch_size=8",
            "--rounds", "4", "--smoke-iters", "10", "--train-iters", "20", "--timeout-s", "600",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )

    tc = load_tuning_config(
        "/shared/amdgpu/home/xiaoming_peng_qle/workspace/Primus/pilot/state/_tc_test/tuning.yaml",
        required=True,
    )
    print("session_id        :", tc.session_id)
    print("session_dir       :", tc.session_dir)
    print("plan_path_abs     :", tc.plan_path_abs())
    print("cluster_config_abs:", tc.cluster_config_abs())
    print("base_overrides    :", tc.base_overrides)
    print("smoke_trace_dir   :", tc.stage_trace_dir("smoke"))
    print("baseline_trace_dir:", tc.stage_trace_dir("baseline"))
    print("trial_t1          :", tc.stage_trace_dir("optimize_loop", trial_id=1))
    print("trial_t7          :", tc.stage_trace_dir("optimize_loop", trial_id=7))
    print("smoke_default     :", tc.stage_default("smoke"))

    for bad_call in (
        lambda: tc.stage_trace_dir("optimize_loop"),
        lambda: tc.stage_trace_dir("unknown_stage"),
    ):
        try:
            bad_call()
        except TuningConfigError as e:
            print("expected-error    :", e.kind, "|", str(e)[:80])

    print("ALL CHECKS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
