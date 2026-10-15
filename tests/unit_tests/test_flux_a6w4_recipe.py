"""Static/CPU contracts for the MLPerf FLUX A6W4 integration."""

import os
from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "examples/mlperf/flux1"
IMAGE = (
    "zirui3/primus-v26.7-flux:v0.1@"
    "sha256:082afc328d867f79ab5c4a46bc29c909bd891e068a32506a10768e6a6cc7e7c8"
)


def _profile_env() -> dict[str, str]:
    command = (
        "source examples/mlperf/flux1/config_4n_gbs1024.sh; "
        "env -0"
    )
    result = subprocess.run(
        ["bash", "-c", command],
        cwd=ROOT,
        env={"PATH": os.environ["PATH"]},
        check=True,
        capture_output=True,
    )
    return dict(item.split("=", 1) for item in result.stdout.decode().split("\0") if item)


def test_four_node_handoff_defaults():
    env = _profile_env()
    expected = {
        "NNODES": "4",
        "DP_REPLICATE": "1",
        "LOCAL_BATCH_SIZE": "32",
        "GRADIENT_ACCUMULATION_STEPS": "1",
        "GLOBAL_BATCH_SIZE": "1024",
        "LR": "0.00025",
        "WARMUP_STEPS": "1600",
        "FSDP_SDMA_MODULES": "single_blocks",
        "TORCHINDUCTOR_CPP_WRAPPER": "1",
        "FLUX_A6W4_DENSE_ACT": "0",
        "FLUX_A6W4_FC1_GELU_DUAL": "1",
        "FLUX_A6W4_GELU_PACK_DOUBLE": "1",
        "FLUX_FWD_GELU_PACK": "1",
        "FLUX_FPROP_SWITCH_STEP": "7325",
        "FLUX_FPROP_SWITCH_PRECOMPILE": "1",
        "FLUX_FP4_PASSES": "dgrad,wgrad",
        "FLUX_FP4_FPROP_CAST": "a6w4",
    }
    assert {name: env[name] for name in expected} == expected


def test_image_defaults_agree_by_digest():
    for name in ("Dockerfile", "README.md", "README-dev.md", "run_with_docker.sh"):
        assert IMAGE in (RECIPE / name).read_text()


def _fake_launcher(script: str, tmp_path: Path) -> list[str]:
    data = tmp_path / "data"
    output = tmp_path / "output"
    bin_dir = tmp_path / "bin"
    data.mkdir()
    output.mkdir()
    bin_dir.mkdir()
    command = "docker" if script == "run_with_docker.sh" else "srun"
    fake = bin_dir / command
    fake.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n')
    fake.chmod(0o755)
    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "DATA_ROOT": str(data),
        "OUTPUT_ROOT": str(output),
        "FLUX_FPROP_SWITCH_PRECOMPILE": "0",
        "NCCL_IB_DISABLE": "1",
        "MASTER_ADDR": "127.0.0.1",
    }
    if command == "srun":
        env.update(
            SLURM_JOB_ID="123",
            SLURM_NNODES="4",
            SLURM_JOB_NODELIST="test-nodes",
        )
    result = subprocess.run(
        ["bash", str(RECIPE / script)],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines()


@pytest.mark.parametrize("script", ["run_with_docker.sh", "run_with_docker_slurm.sh"])
def test_launchers_forward_conditional_flux_vars_but_not_path(script, tmp_path):
    args = _fake_launcher(script, tmp_path)
    assert not any(arg == "PATH" or arg.startswith("PATH=") or arg == "--env=PATH" for arg in args)
    assert any(
        arg == "FLUX_FP4_ALL_GATHER" or arg == "FLUX_FP4_ALL_GATHER=0"
        for arg in args
    )


def test_fprop_switch_hooks_cover_resume_and_require_mlperf():
    source = (ROOT / "primus/backends/diffusion/trainers/base.py").read_text()
    precompile = source.index("flux_fprop_switch.precompile(self)")
    resumed_switch = source.index("flux_fprop_switch.switch_if_due", precompile)
    first_batch = source.index("for batch_idx, batch in enumerate", resumed_switch)
    assert precompile < resumed_switch < first_batch
    assert "FLUX_FPROP_SWITCH_STEP requires MLPerf mode" in source


def test_fprop_switch_is_off_without_step():
    from primus.backends.diffusion.models.quantization import flux_fprop_switch

    with patch.dict(os.environ, {}, clear=True):
        assert flux_fprop_switch.switch_step() is None


def test_fprop_switch_rejects_nonpositive_step():
    from primus.backends.diffusion.models.quantization import flux_fprop_switch

    with patch.dict(
        os.environ,
        {"FLUX_FPROP_SWITCH_STEP": "0", "FLUX_FP4_ALL_GATHER": "0"},
        clear=True,
    ), pytest.raises(ValueError, match="must be > 0"):
        flux_fprop_switch.switch_step()
