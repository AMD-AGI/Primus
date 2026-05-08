from __future__ import annotations

from pathlib import Path

import yaml

from pilot.tools import constraint, report, state, tune_single


def test_state_checkpoint_resume_trim_handoff(tmp_path: Path) -> None:
    tuning_state = {
        "session_id": "s1",
        "current_stage": "SMOKE",
        "round_id": 2,
        "stage_history": [],
        "run_history": [{"id": "large"}],
        "champion_id": "c1",
        "budget_used": {"rounds": 2},
    }

    checkpoint_path = Path(state.checkpoint(tuning_state, root=str(tmp_path)))
    assert checkpoint_path.exists()
    assert (tmp_path / "tuning_state.yaml").exists()
    assert state.resume(checkpoint_path)["session_id"] == "s1"

    trimmed = state.trim(tuning_state, keep=["session_id", "champion_id"])
    assert trimmed == {"session_id": "s1", "champion_id": "c1"}

    handoff_path = Path(state.handoff("s1", reason="test", next_action_hint="resume"))
    assert handoff_path.exists()


def test_constraint_check_and_failure_diagnosis() -> None:
    plan = {
        "modules": {
            "pre_trainer": {
                "overrides": {
                    "tensor_model_parallel_size": 2,
                    "pipeline_model_parallel_size": 1,
                    "expert_model_parallel_size": 1,
                    "micro_batch_size": 2,
                    "global_batch_size": 8,
                    "train_iters": 5,
                    "seq_length": 128,
                }
            }
        }
    }
    cluster = {"mode": "single", "single": {"max_local_gpus": 4}}
    result = constraint.check(plan, cluster)
    assert result["valid"]
    assert result["derived"]["data_parallel"] == 2

    bad = constraint.check(
        {"overrides": {"tensor_model_parallel_size": 3, "micro_batch_size": 1, "global_batch_size": 4}},
        cluster,
    )
    assert not bad["valid"]

    failure = constraint.diagnose_failure({
        "status": "hung",
        "symptoms": {"hang_suspected": True, "evidence": [{"kind": "hang_hint"}]},
    })
    assert failure["kind"] == "HANG"
    assert failure["suggested_transition"] == "PREFLIGHT"


def test_replan_and_settle_pick_faster_candidate() -> None:
    base_plan = {
        "modules": {
            "pre_trainer": {
                "overrides": {
                    "tensor_model_parallel_size": 1,
                    "pipeline_model_parallel_size": 1,
                    "expert_model_parallel_size": 1,
                    "micro_batch_size": 1,
                    "global_batch_size": 8,
                    "train_iters": 3,
                    "seq_length": 128,
                }
            }
        }
    }
    cluster = {"mode": "single", "single": {"max_local_gpus": 4}}
    snapshot = {
        "status": "completed",
        "metrics": {
            "latest": {"loss": 1.0, "iter_time_ms": 100.0, "tflops": 10.0},
            "history": {"iter_time_ms": [100.0, 98.0], "tflops": [10.0], "loss": [1.2, 1.0]},
            "loss_finite": True,
        },
        "symptoms": {
            "oom_detected": False,
            "hang_suspected": False,
            "nccl_error": False,
            "cuda_error": False,
            "python_error": False,
            "loss_nan_or_inf": False,
        },
    }
    diagnosis = tune_single.diagnose(snapshot)
    pool = tune_single.replan(
        base_plan=base_plan,
        cluster=cluster,
        diagnosis=diagnosis,
        round_id=1,
        max_candidates=2,
        train_iters=5,
    )
    assert pool["candidates"]
    assert pool["candidates"][0]["overrides"]["train_iters"] == 5

    history = [
        {"id": "baseline", "measurement": {"status": "completed", "median_iter_time_ms": 100.0}},
        {"id": "candidate", "measurement": {"status": "completed", "median_iter_time_ms": 80.0}},
    ]
    settled = tune_single.settle(history, champion_id="baseline")
    assert settled["champion_id"] == "candidate"
    assert settled["promoted"]


def test_report_build_with_tuning_state(tmp_path: Path) -> None:
    plan = tmp_path / "plan.yaml"
    plan.write_text("modules:\n  pre_trainer:\n    overrides:\n      train_iters: 3\n")
    tuning_state = {
        "session_id": "s1",
        "current_stage": "REPORT",
        "stage_history": [],
        "budget_used": {"rounds": 1},
        "champion_id": "candidate",
        "run_history": [
            {
                "id": "baseline",
                "stage": "BASELINE",
                "measurement": {"status": "completed", "median_iter_time_ms": 100.0},
            },
            {
                "id": "candidate",
                "stage": "OPTIMIZE_LOOP.EXECUTE",
                "overrides": {"micro_batch_size": 2},
                "measurement": {"status": "completed", "median_iter_time_ms": 80.0},
            },
        ],
    }
    state_path = tmp_path / "tuning_state.yaml"
    state_path.write_text(yaml.safe_dump(tuning_state))

    result = report.build(
        plan_path=plan,
        tuning_state=state_path,
        out_dir=tmp_path / "reports",
        report_id="single_node_test",
    )
    report_path = Path(result["artifacts"][0]["ref"])
    built = yaml.safe_load(report_path.read_text())
    assert built["tuning"]["champion_id"] == "candidate"
    assert built["tuning"]["improvement_pct"] == 25.0
