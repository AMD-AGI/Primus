"""Static/CPU contracts for the MLPerf FLUX A6W4 integration."""

import os
import re
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "examples/mlperf/flux1"
IMAGE = "rocm/mlperf@sha256:f4f4df148369ee52ccc79eb699913d321a6a532ad03dc03a6bc951593d503e16"
LEGACY_PROFILES = ("config_1n_gbs512.sh", "config_1n_gbs1024.sh", "config_2n_gbs1024.sh")


def _profile_env(cluster: str = "", profile: str = "config_4n_gbs1024.sh", extra_env=None) -> dict[str, str]:
    command = f"source examples/mlperf/flux1/{profile}; env -0"
    result = subprocess.run(
        ["bash", "-c", command],
        cwd=ROOT,
        env={"PATH": os.environ["PATH"], "FLUX_CLUSTER": cluster, **(extra_env or {})},
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


def test_single_node_matches_amd_schedule_without_four_node_extensions():
    env = _profile_env(profile="config_1n_gbs256_a6w4.sh")
    expected = {
        "NNODES": "1",
        "DP_REPLICATE": "1",
        "LOCAL_BATCH_SIZE": "32",
        "GRADIENT_ACCUMULATION_STEPS": "1",
        "GLOBAL_BATCH_SIZE": "256",
        "LR": "0.00025",
        "WARMUP_STEPS": "800",
        "EVAL_BATCH_SIZE": "32",
        "FSDP_SDMA_MODULES": "single_blocks",
        "FLUX_FP4_FPROP_CAST": "a6w4",
        "FLUX_FPROP_SWITCH_STEP": "",
        "TORCHINDUCTOR_CPP_WRAPPER": "0",
        "FLUX_A6W4_FC1_GELU_DUAL": "0",
        "FLUX_A6W4_PACK_HARVEST": "0",
        "FLUX_A6W4_GELU_PACK_DOUBLE": "0",
        "FLUX_A6W4_LNMOD_PACK": "0",
        "FLUX_A6W4_LEGS_ONE_LAUNCH": "0",
        "TARGET_ACCURACY": "0.586",
        "VAL_CHECK_INTERVAL": "262144",
        "MLLOG_CONFIG_FILENAME": "config_1n_gbs256_a6w4.sh",
    }
    assert {name: env[name] for name in expected} == expected


def test_v269_single_node_profile_preserves_schedule_and_tunes_communication():
    env = _profile_env(profile="config_1n_gbs256_a6w4_v26_9.sh")
    baseline = _profile_env(profile="config_1n_gbs256_a6w4.sh")
    expected = {
        "GPU_MAX_HW_QUEUES": "4",
        "CUDA_DEVICE_MAX_CONNECTIONS": "2",
        "FLUX_FSDP_RS_STREAM": "normal",
        "TORCH_DIST_USE_NCCL2": "1",
        "MLLOG_CONFIG_FILENAME": "config_1n_gbs256_a6w4_v26_9.sh",
    }
    assert {key: env[key] for key in expected} == expected
    for key in (
        "NNODES",
        "LOCAL_BATCH_SIZE",
        "GRADIENT_ACCUMULATION_STEPS",
        "GLOBAL_BATCH_SIZE",
        "LR",
        "WARMUP_STEPS",
        "SEED",
        "EVAL_BATCH_SIZE",
        "VAL_CHECK_INTERVAL",
        "FSDP_ALL_GATHER_BACKEND",
        "FSDP_SDMA_MODULES",
        "FLUX_AG_COLPACK_KMAJOR",
    ):
        assert env[key] == baseline[key]
    overrides = {
        "GPU_MAX_HW_QUEUES": "2",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "FLUX_FSDP_RS_STREAM": "default",
    }
    overridden = _profile_env(profile="config_1n_gbs256_a6w4_v26_9.sh", extra_env=overrides)
    assert {key: overridden[key] for key in overrides} == overrides


@pytest.mark.parametrize("suffix", ["", "_v26_9"])
def test_two_node_gbs512_preserves_single_node_kernels(suffix):
    profile = f"config_2n_gbs512_a6w4{suffix}.sh"
    env = _profile_env("crusoe", profile=profile)
    single = _profile_env("crusoe", profile=f"config_1n_gbs256_a6w4{suffix}.sh")
    assert (env["NNODES"], env["GLOBAL_BATCH_SIZE"]) == ("2", "512")
    assert env["MLLOG_CONFIG_FILENAME"] == profile
    assert env["NCCL_CROSS_NIC"] == "0"
    assert env["NCCL_DMABUF_ENABLE"] == "0"
    assert env["NCCL_NET_GDR_LEVEL"] == "3"
    assert env["NCCL_NET_GDR_READ"] == "1"
    assert _profile_env("dccs", profile=profile)["NCCL_DMABUF_ENABLE"] == "1"
    for key in (
        "LOCAL_BATCH_SIZE",
        "GRADIENT_ACCUMULATION_STEPS",
        "LR",
        "WARMUP_STEPS",
        "FLUX_FPROP_SWITCH_STEP",
        "FLUX_A6W4_FC1_GELU_DUAL",
        "TORCHINDUCTOR_CPP_WRAPPER",
        "FSDP_SDMA_MODULES",
        "FLUX_AG_COLPACK_KMAJOR",
    ):
        assert env[key] == single[key]
    assert _profile_env("crusoe", profile=profile, extra_env={"NCCL_CROSS_NIC": "2"})["NCCL_CROSS_NIC"] == "2"


@pytest.mark.parametrize("nodes,ga", [(1, 4), (2, 2)])
def test_v269_gbs1024_profiles_preserve_kernel_recipe(nodes, ga):
    profile = f"config_{nodes}n_gbs1024_a6w4_v26_9.sh"
    env = _profile_env("crusoe", profile=profile)
    assert env["NNODES"] == str(nodes)
    assert env["LOCAL_BATCH_SIZE"] == "32"
    assert env["GRADIENT_ACCUMULATION_STEPS"] == str(ga)
    assert env["GLOBAL_BATCH_SIZE"] == "1024"
    assert env["WARMUP_STEPS"] == "1600"
    assert _profile_env(profile=profile, extra_env={"WARMUP_STEPS": "120"})["WARMUP_STEPS"] == "120"
    assert nodes * int(env["GPUS_PER_NODE"]) * 32 * ga == 1024
    assert env["MLLOG_CONFIG_FILENAME"] == profile
    assert env["FLUX_FPROP_SWITCH_STEP"] == ""
    assert env["FLUX_A6W4_FC1_GELU_DUAL"] == "0"
    assert env["FSDP_SDMA_MODULES"] == "single_blocks"


def test_two_node_adaptation_preserves_four_node_schedule():
    env = _profile_env(profile="config_2n_gbs1024_a6w4.sh")
    four_node = _profile_env()
    assert (env["NNODES"], env["GRADIENT_ACCUMULATION_STEPS"]) == ("2", "2")
    assert env["MLLOG_CONFIG_FILENAME"] == "config_2n_gbs1024_a6w4.sh"
    assert "FLUX_FSDP_KEEP_UNSHARDED" not in env
    for name in (
        "LOCAL_BATCH_SIZE",
        "GLOBAL_BATCH_SIZE",
        "DP_REPLICATE",
        "LR",
        "WARMUP_STEPS",
        "FLUX_FPROP_SWITCH_STEP",
        "FLUX_FP4_FPROP_CAST",
        "TARGET_ACCURACY",
        "VAL_CHECK_INTERVAL",
    ):
        assert env[name] == four_node[name]


@pytest.mark.parametrize(
    "profile", ["config_1n_gbs256_a6w4.sh", "config_2n_gbs1024_a6w4.sh", "config_4n_gbs1024.sh"]
)
@pytest.mark.parametrize(
    "backend,override,expected",
    [("rccl_sdma", None, "1"), ("rccl_sdma", "0", "0"), ("default", None, None)],
)
def test_sdma_selects_nccl2_without_overriding_controls(profile, backend, override, expected):
    overrides = {"FSDP_ALL_GATHER_BACKEND": backend}
    if override is not None:
        overrides["TORCH_DIST_USE_NCCL2"] = override
    env = _profile_env(profile=profile, extra_env=overrides)
    assert env.get("TORCH_DIST_USE_NCCL2") == expected


def test_cluster_network_profiles():
    dccs = _profile_env("dccs")
    assert {
        "NCCL_SOCKET_IFNAME": dccs["NCCL_SOCKET_IFNAME"],
        "GLOO_SOCKET_IFNAME": dccs["GLOO_SOCKET_IFNAME"],
        "NCCL_NET_PLUGIN": dccs["NCCL_NET_PLUGIN"],
        "NCCL_RMA_DISABLE": dccs["NCCL_RMA_DISABLE"],
        "DATALOADER_NUM_WORKERS": dccs["DATALOADER_NUM_WORKERS"],
    } == {
        "NCCL_SOCKET_IFNAME": "fenic",
        "GLOO_SOCKET_IFNAME": "fenic",
        "NCCL_NET_PLUGIN": "none",
        "NCCL_RMA_DISABLE": "0",
        "DATALOADER_NUM_WORKERS": "4",
    }
    crusoe = _profile_env("crusoe")
    assert {
        "NCCL_SOCKET_IFNAME": crusoe["NCCL_SOCKET_IFNAME"],
        "GLOO_SOCKET_IFNAME": crusoe["GLOO_SOCKET_IFNAME"],
        "NCCL_NET_PLUGIN": crusoe["NCCL_NET_PLUGIN"],
        "NCCL_RMA_DISABLE": crusoe["NCCL_RMA_DISABLE"],
        "NCCL_IB_HCA": crusoe["NCCL_IB_HCA"],
    } == {
        "NCCL_SOCKET_IFNAME": "ens3",
        "GLOO_SOCKET_IFNAME": "ens3",
        "NCCL_NET_PLUGIN": "librccl-anp.so",
        "NCCL_RMA_DISABLE": "1",
        "NCCL_IB_HCA": "ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7",
    }


def test_runtime_image_pin_agrees_with_documentation():
    for name in ("README.md", "run_with_docker.sh"):
        assert IMAGE in (RECIPE / name).read_text()


def _fake_launcher(
    script: str,
    tmp_path: Path,
    image: str = IMAGE,
    profile: str = "config_4n_gbs1024.sh",
    extra_env=None,
    scontrol_script=None,
) -> list[str]:
    data = tmp_path / "data"
    output = tmp_path / "output"
    bin_dir = tmp_path / "bin"
    data.mkdir()
    output.mkdir()
    bin_dir.mkdir()
    # Keep scheduler discovery deterministic and never invoke host schedulers.
    for name in ("bash", "dirname", "readlink", "basename", "sed", "env", "grep", "mkdir"):
        (bin_dir / name).symlink_to(shutil.which(name))
    if scontrol_script is not None:
        scontrol = bin_dir / "scontrol"
        scontrol.write_text("#!/usr/bin/env bash\n" + scontrol_script + "\n")
        scontrol.chmod(0o755)
    command = "docker" if script == "run_with_docker.sh" else "srun"
    fake = bin_dir / command
    fake.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n'
        "while (( $# )); do\n"
        '  if [[ "$1" == "--env" ]]; then\n'
        '    name=$2; printf "ENV:%s=%s\\n" "$name" "${!name}"; shift\n'
        '  elif [[ "$1" =~ ^[a-zA-Z_][a-zA-Z_0-9]*= ]]; then\n'
        '    printf "ENV:%s\\n" "$1"\n'
        "  fi\n"
        "  shift\n"
        "done\n"
    )
    fake.chmod(0o755)
    env = {
        "PATH": str(bin_dir),
        "DATA_ROOT": str(data),
        "OUTPUT_ROOT": str(output),
        "FLUX_CLUSTER": "crusoe",
        "FLUX_CONFIG": profile,
        "DOCKER_IMAGE": image,
        "FLUX_FPROP_SWITCH_PRECOMPILE": "0",
        "NCCL_IB_DISABLE": "1",
        "MASTER_ADDR": "127.0.0.1",
    }
    if command == "srun":
        env.update(
            SLURM_JOB_ID="123",
            SLURM_NNODES=_profile_env(profile=profile)["NNODES"],
            SLURM_JOB_NODELIST="test-nodes",
        )
    env.update(extra_env or {})
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
    assert any(arg == "FLUX_FP4_ALL_GATHER" or arg == "FLUX_FP4_ALL_GATHER=0" for arg in args)
    for name in ("NCCL_NET_PLUGIN", "NCCL_RMA_DISABLE", "NCCL_IB_HCA", "TORCH_DIST_USE_NCCL2"):
        assert any(arg == name or arg.startswith(f"{name}=") for arg in args)
    assert IMAGE in args or f"DOCKER_IMAGE={IMAGE}" in args


def test_libionic_mount_uses_version_independent_soname(tmp_path):
    args = _fake_launcher(
        "run_with_docker.sh",
        tmp_path,
        profile="config_2n_gbs512_a6w4_v26_9.sh",
        extra_env={"LIBIONIC_ABI4_PATH": "/host/libionic.so"},
    )
    assert "/host/libionic.so:/usr/lib/x86_64-linux-gnu/libionic.so.1:ro" in args


@pytest.mark.parametrize("privileged", ["0", "1"])
def test_v269_launcher_forwards_image_dependency_contract(tmp_path, privileged):
    dependency_env = {"PRIMUS_SKIP_PIP": "1", "PRIMUS_AUTO_INSTALL": "0", "PRIMUS_AUTO_DEPS_SYNC": "0"}
    args = _fake_launcher(
        "run_with_docker.sh",
        tmp_path,
        profile="config_1n_gbs256_a6w4_v26_9.sh",
        extra_env={"DOCKER_PRIVILEGED": privileged, **dependency_env},
    )
    assert ("--privileged" in args) == (privileged == "1")
    forwarded = dict(arg[4:].split("=", 1) for arg in args if arg.startswith("ENV:"))
    assert {key: forwarded[key] for key in dependency_env} == dependency_env
    assert forwarded["FLUX_FSDP_RS_STREAM"] == "normal"
    assert forwarded["GPU_MAX_HW_QUEUES"] == "4"
    assert forwarded["CUDA_DEVICE_MAX_CONNECTIONS"] == "2"
    assert forwarded["TORCH_DIST_USE_NCCL2"] == "1"


@pytest.mark.parametrize("script", ["run_with_docker.sh", "run_with_docker_slurm.sh"])
def test_launchers_preserve_verified_local_image_override(script, tmp_path):
    image = "sha256:f0a1f5c0641ce55d37ab6ce35ef02f859ed6e3b6573b222dbc701d9715a6e482"
    args = _fake_launcher(script, tmp_path, image=image)
    assert image in args or f"DOCKER_IMAGE={image}" in args
    assert IMAGE not in args and f"DOCKER_IMAGE={IMAGE}" not in args


@pytest.mark.parametrize("name", ["flux.1_schnell_t2i-pretrain.yaml", "flux.1_schnell_t2i-native.yaml"])
def test_both_launch_modes_expose_existing_profiler_controls(name):
    config = yaml.safe_load((RECIPE / name).read_text())
    if "modules" in config:
        config = config["modules"]["pre_trainer"]["overrides"]
    for key in (
        "profile",
        "profile_rank",
        "profile_wait_steps",
        "profile_warmup_steps",
        "profile_active_steps",
        "profile_output_dir",
        "profile_with_stack",
    ):
        assert config["runtime"][key].startswith("${" + key.upper())


@pytest.mark.parametrize("profile", LEGACY_PROFILES)
@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"FLUX_FP8_ALL_GATHER": "1"},
        {"FSDP2_HSDP_FP8_ALL_REDUCE": "e4m3"},
        {"FSDP2_HSDP_FP8_ALL_REDUCE": "e5m2"},
        {
            "MLLOG_LOWEST_NUMERICAL_PRECISION_IN_LINEAR": "fp8",
            "MLLOG_LOWEST_NUMERICAL_PRECISION_IN_ATTN": "bfloat16",
            "MLLOG_LOWEST_NUMERICAL_PRECISION_IN_COMM": "fp8",
            "MLLOG_CONFIG_FILENAME": "custom.sh",
        },
    ],
)
def test_legacy_profiles_initialize_mlperf_from_clean_environment(profile, overrides):
    from primus.backends.diffusion.trainers.base import BaseWanTrainer

    env = _profile_env(profile=profile, extra_env=overrides)
    assert "TORCH_DIST_USE_NCCL2" not in env
    trainer = BaseWanTrainer.__new__(BaseWanTrainer)
    trainer.rank = 1
    trainer.mlperf_enabled = True
    with patch.dict(os.environ, env, clear=True):
        trainer._setup_mlperf()
    assert trainer.mlperf_precision == {
        "linear": "fp8",
        "attn": "bfloat16",
        "comm": "fp8" if overrides else "bfloat16",
    }
    assert trainer.mlperf_config_filename == overrides.get("MLLOG_CONFIG_FILENAME", profile)


@pytest.mark.parametrize("profile", LEGACY_PROFILES)
@pytest.mark.parametrize("script", ["run_with_docker.sh", "run_with_docker_slurm.sh"])
@pytest.mark.parametrize("mode", ["primus", "native"])
@pytest.mark.parametrize("enabled", [False, True])
def test_legacy_launchers_forward_metadata_and_resolve_profiler_yaml(
    profile, script, mode, enabled, tmp_path
):
    from primus.core.config.yaml_loader import parse_yaml

    overrides = {"LAUNCH_MODE": mode}
    if enabled:
        overrides.update(
            PROFILE="true",
            PROFILE_RANK="1",
            PROFILE_WAIT_STEPS="7",
            PROFILE_WARMUP_STEPS="3",
            PROFILE_ACTIVE_STEPS="4",
            PROFILE_WITH_STACK="true",
            PROFILE_OUTPUT_DIR="/output/custom-profile",
            MLLOG_CONFIG_FILENAME="custom.sh",
            FLUX_FP8_ALL_GATHER="1",
            MLLOG_LOWEST_NUMERICAL_PRECISION_IN_LINEAR="fp8",
            MLLOG_LOWEST_NUMERICAL_PRECISION_IN_ATTN="bfloat16",
            MLLOG_LOWEST_NUMERICAL_PRECISION_IN_COMM="fp8",
        )
    args = _fake_launcher(script, tmp_path, profile=profile, extra_env=overrides)
    forwarded = dict(arg[4:].split("=", 1) for arg in args if arg.startswith("ENV:"))
    expected = _profile_env("crusoe", profile=profile, extra_env=overrides)
    for name in (
        "MLLOG_CONFIG_FILENAME",
        "MLLOG_LOWEST_NUMERICAL_PRECISION_IN_LINEAR",
        "MLLOG_LOWEST_NUMERICAL_PRECISION_IN_ATTN",
        "MLLOG_LOWEST_NUMERICAL_PRECISION_IN_COMM",
        "PROFILE",
        "PROFILE_RANK",
        "PROFILE_WAIT_STEPS",
        "PROFILE_WARMUP_STEPS",
        "PROFILE_ACTIVE_STEPS",
        "PROFILE_WITH_STACK",
    ):
        assert forwarded[name] == expected[name]
    if script == "run_with_docker.sh":
        with patch.dict(os.environ, forwarded, clear=True):
            if mode == "primus":
                config = parse_yaml(str(RECIPE / "flux.1_schnell_t2i-pretrain.yaml"))
                runtime = config["modules"]["pre_trainer"]["overrides"]["runtime"]
            else:
                text = os.path.expandvars((RECIPE / "flux.1_schnell_t2i-native.yaml").read_text())
                assert not re.search(r"\$(?:\w+|\{[^}]+\})", text)
                runtime = yaml.safe_load(text)["runtime"]
        assert runtime["profile"] is enabled
        assert runtime["profile_rank"] == (1 if enabled else 0)
        assert runtime["profile_wait_steps"] == (7 if enabled else 30)
        assert runtime["profile_active_steps"] == (4 if enabled else 5)
        assert runtime["profile_with_stack"] is enabled
        if enabled:
            assert runtime["profile_output_dir"] == "/output/custom-profile"


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


@pytest.mark.parametrize(
    "scontrol_script,master,peers,expected",
    [
        (None, "", "peer-one:2222,peer-two:3333", "peer-one"),
        ("exit 1", "", "peer-one:2222,peer-two:3333", "peer-one"),
        ("exit 0", "", "peer-one:2222,peer-two:3333", "peer-one"),
        ('printf "scheduler-one\\nscheduler-two\\n"', "", "peer-one:2222", "scheduler-one"),
        ("exit 1", "explicit-master", "", "explicit-master"),
    ],
)
def test_slurm_master_discovery_uses_successful_nonempty_expansion(
    scontrol_script, master, peers, expected, tmp_path
):
    args = _fake_launcher(
        "run_with_docker_slurm.sh",
        tmp_path,
        profile="config_2n_gbs1024_a6w4.sh",
        extra_env={"MASTER_ADDR": master, "SPUR_PEER_NODES": peers},
        scontrol_script=scontrol_script,
    )
    assert f"MASTER_ADDR={expected}" in args
    assert "MASTER_PORT=29500" in args


@pytest.mark.parametrize("scontrol_script", [None, "exit 1", "exit 0"])
def test_slurm_master_discovery_fails_without_scheduler_hosts_or_peers(scontrol_script, tmp_path):
    with pytest.raises(subprocess.CalledProcessError) as error:
        _fake_launcher(
            "run_with_docker_slurm.sh",
            tmp_path,
            profile="config_2n_gbs1024_a6w4.sh",
            extra_env={"MASTER_ADDR": "", "SPUR_PEER_NODES": ""},
            scontrol_script=scontrol_script,
        )
    assert "Set MASTER_ADDR" in error.value.stderr
    assert "ENV:" not in error.value.stdout


@pytest.mark.parametrize(
    "backend,kernel,compiler_exit,expected_exit",
    [
        ("rccl_sdma", "6.8.0", 0, 0),
        ("none", "old", 1, 0),
        ("protocol", "unused", 0, 0),
        ("rccl_sdma", "6.7.9", 0, 2),
        ("rccl_sdma", "unparseable", 0, 2),
        ("rccl_sdma", "6.8.0", 7, 7),
    ],
)
def test_native_launch_runs_shared_sdma_hook_once_before_workers(
    backend, kernel, compiler_exit, expected_exit, tmp_path
):
    launch_dir = tmp_path / "launch"
    launch_dir.mkdir()
    args = _fake_launcher(
        "run_with_docker.sh",
        launch_dir,
        profile="config_1n_gbs256_a6w4.sh",
        extra_env={
            "LAUNCH_MODE": "native",
            "FSDP_ALL_GATHER_BACKEND": backend,
            "MLPERF_CLEAR_CACHES": "false",
        },
    )
    # Reuse the exact Docker payload and forwarded environment, but stub all work.
    env_start = next(index for index, arg in enumerate(args) if arg.startswith("ENV:"))
    payload = "\n".join(args[args.index("-lc") + 1 : env_start])
    env = dict(arg.removeprefix("ENV:").split("=", 1) for arg in args[env_start:] if arg.startswith("ENV:"))
    container = tmp_path / "container"
    hooks = container / "runner/helpers/hooks"
    (hooks / "sdma").mkdir(parents=True)
    shutil.copyfile(
        ROOT / "runner/helpers/hooks/06_enable_sdma_all_gather.sh", hooks / "06_enable_sdma_all_gather.sh"
    )
    (hooks / "sdma/hip_attr_drain_preload.c").touch()
    if backend == "protocol":
        (hooks / "06_enable_sdma_all_gather.sh").write_text(
            "#!/bin/bash\nprintf '%s\\n' 'env.HOOK_LITERAL=spaces ; $(touch injected)' 'env.9INVALID=bad' 'touch injected'\n"
        )
    bin_dir = container / "bin"
    bin_dir.mkdir()
    log = container / "calls"
    for name, body in {
        "uname": f'echo uname >> "{log}"; echo "{kernel}"',
        "gcc": f'echo gcc >> "{log}"; exit {compiler_exit}',
        "torchrun": f'echo worker >> "{log}"; env | grep -E "^(NCCL_CUMEM_ENABLE|NCCL_LOCAL_REGISTER|NCCL_CTA_POLICY|TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK|LD_PRELOAD|FSDP_ALL_GATHER_BACKEND|HOOK_LITERAL)="; printf "ARG:%s\\n" "$@"',
    }.items():
        stub = bin_dir / name
        stub.write_text("#!/usr/bin/env bash\n" + body + "\n")
        stub.chmod(0o755)
    result = subprocess.run(
        ["bash", "-c", payload],
        cwd=container,
        env={**env, "PATH": f"{bin_dir}:{os.environ['PATH']}", "OUTPUT_DIR": str(container / "output")},
        text=True,
        capture_output=True,
    )
    assert result.returncode == expected_exit, result.stderr
    calls = log.read_text().splitlines() if log.exists() else []
    if expected_exit:
        assert "worker" not in calls
        assert "RCCL-SDMA" in result.stderr or compiler_exit != 0
    elif backend == "rccl_sdma":
        assert calls == ["uname", "gcc", "worker"]
        for setting in (
            "NCCL_CUMEM_ENABLE=1",
            "NCCL_LOCAL_REGISTER=0",
            "NCCL_CTA_POLICY=2",
            "TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=true",
            "LD_PRELOAD=/tmp/libhip_attr_drain.so",
            "FSDP_ALL_GATHER_BACKEND=rccl_sdma",
            "ARG:--nproc_per_node=8",
        ):
            assert setting in result.stdout
    else:
        assert calls == ["worker"]
        assert "NCCL_CUMEM_ENABLE=" not in result.stdout
        assert "LD_PRELOAD=" not in result.stdout
        if backend == "protocol":
            assert "HOOK_LITERAL=spaces ; $(touch injected)" in result.stdout
            assert "9INVALID=" not in result.stdout
            assert not (container / "injected").exists()
