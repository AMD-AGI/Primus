###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the SpecForge backend (CPU only, no SpecForge install required).

Coverage:
    1. Adapter registration and lazy load via BackendRegistry
    2. Experiment YAML -> `specforge train` argv
    3. SpecForge working-directory resolution
    4. The pretrain hook emits env.RUN_MODE=single so Primus skips torchrun
"""

import os
import subprocess
import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

from primus.backends.specforge.argument_builder import (
    build_capture_argv,
    build_specforge_argv,
    flatten_overrides,
    is_online_train,
    resolve_specforge_root,
    specforge_mode,
    specforge_role,
    specforge_train_mode,
)
from primus.backends.specforge.online_launch import (
    build_online_overrides,
    build_role_argv,
    expand_device_list,
    nnodes,
    node_rank,
    online_settings,
    resolve_head_ip,
    resolve_local_ip,
    resolve_routable_ip,
    sglang_server_argv,
    validate_online_identity,
)
from primus.backends.specforge.specforge_adapter import SpecForgeAdapter
from primus.backends.specforge.specforge_pretrain_trainer import (
    SpecForgePretrainTrainer,
    align_visible_devices,
    clear_partial_distributed_env,
    resolve_filter_min_kept,
)
from primus.backends.specforge.stack_preflight import (
    apply_rocm_stack_env,
    collect_issues,
    enforce_rocm_stack,
)
from primus.core.backend.backend_registry import BackendRegistry
from primus.core.launcher.parser import PrimusParser

# SpecForge stack preflight imports patched sglang / aiter that only exist in
# the SpecForge overlay image. The Primus CI image does not ship that stack,
# so these tests fail collection/runtime there. Re-enable once CI has the
# overlay (or the tests are split into overlay-free vs overlay-only).
pytestmark = pytest.mark.skip(
    reason="requires SpecForge overlay image (patched sglang/aiter); not in Primus CI image"
)

PRIMUS_ROOT = Path(__file__).resolve().parents[4]
EXAMPLE_CONFIG = PRIMUS_ROOT / "examples" / "specforge" / "configs" / "qwen3.5-4b-dflash-offline.yaml"
CAPTURE_CONFIG = PRIMUS_ROOT / "examples" / "specforge" / "configs" / "qwen3.5-4b-dflash-offline-capture.yaml"
ONLINE_CONFIG = PRIMUS_ROOT / "examples" / "specforge" / "configs" / "qwen3.5-4b-dflash-online-2node.yaml"
PREPARE_HOOK = PRIMUS_ROOT / "runner/helpers/hooks/train/pretrain/specforge/prepare.py"


@pytest.fixture
def specforge_checkout(tmp_path):
    """A directory that resolve_specforge_root() recognizes as a SpecForge tree."""
    root = tmp_path / "SpecForge"
    (root / "configs").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'specforge'\n")
    (root / "configs" / "qwen3.5-4b-dflash.yaml").write_text("model: dummy\n")
    (root / "configs" / "qwen3.5-4b-dflash.json").write_text(
        '{"dflash_config": {"target_layer_ids": [1, 8, 15, 22, 29]}}\n'
    )
    (root / "scripts").mkdir()
    (root / "scripts" / "prepare_hidden_states.py").write_text("# stub\n")
    return root


@pytest.fixture
def hidden_states(tmp_path):
    """A non-empty directory standing in for captured hidden states."""
    path = tmp_path / "hidden_states"
    path.mkdir()
    (path / "shard-0000.pt").write_bytes(b"")
    return path


@pytest.fixture
def experiment_env(monkeypatch, specforge_checkout, hidden_states, tmp_path):
    """Environment the example experiment YAML interpolates."""
    env = {
        "SPECFORGE_ROOT": str(specforge_checkout),
        "SPECFORGE_CONFIG": str(specforge_checkout / "configs" / "qwen3.5-4b-dflash.yaml"),
        "HIDDEN_STATES_PATH": str(hidden_states),
        "OUTPUT_DIR": str(tmp_path / "out"),
        "MAX_STEPS": "20",
        "CAPTURE_DATA_PATH": str(tmp_path / "sharegpt.jsonl"),
        "CAPTURE_BATCH_SIZE": "8",
        "BLOCK_SIZE": "16",
        "TARGET_MODEL": "Qwen/Qwen3.5-4B",
        "RUN_ID": "unit-online-1",
        "RUN_ROOT": str(tmp_path / "online-run"),
        "CONSUMER_STATE_DIR": str(tmp_path / "consumer-state"),
        "TRAIN_DATA_PATH": str(tmp_path / "sharegpt.jsonl"),
        "NNODES": "2",
        "NODE_RANK": "0",
    }
    (tmp_path / "sharegpt.jsonl").write_text("{}\n")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return env


def load_pre_trainer_params(config_path):
    """Parse an experiment YAML into the params the adapter receives."""
    from primus.core.config.primus_config import get_module_config, load_primus_config

    cfg = load_primus_config(Path(config_path))
    module = get_module_config(cfg, "pre_trainer")
    assert module is not None, "pre_trainer module missing from config"
    return module


class TestRegistration:
    def test_adapter_registered_on_import(self):
        import primus.backends.specforge  # noqa: F401

        assert BackendRegistry.has_adapter("specforge")
        assert BackendRegistry._adapters["specforge"] is SpecForgeAdapter

    def test_get_adapter_lazy_loads_backend(self):
        """A cleared registry re-populates, because registration is an import side effect.

        The reload stands in for the lazy import in ``_load_backend``, which is a
        no-op here: this module is already in ``sys.modules``.
        """
        import importlib

        import primus.backends.specforge as backend_module

        original = BackendRegistry._adapters.copy()
        try:
            BackendRegistry._adapters.pop("specforge", None)
            importlib.reload(backend_module)
            adapter = BackendRegistry.get_adapter("specforge")
        finally:
            BackendRegistry._adapters = original

        assert isinstance(adapter, SpecForgeAdapter)
        assert adapter.framework == "specforge"

    def test_load_trainer_class(self):
        adapter = SpecForgeAdapter()
        assert adapter.load_trainer_class("pretrain") is SpecForgePretrainTrainer
        with pytest.raises(ValueError, match="Unsupported stage"):
            adapter.load_trainer_class("sft")

    def test_setup_backend_path_tolerates_missing_checkout(self, monkeypatch):
        """SpecForge ships as a wheel, so a missing third_party/ dir is not fatal."""
        monkeypatch.delenv("SPECFORGE_ROOT", raising=False)
        monkeypatch.delenv("BACKEND_PATH", raising=False)
        adapter = SpecForgeAdapter()
        assert adapter.setup_backend_path() == ""

    def test_setup_backend_path_uses_specforge_root(self, monkeypatch, specforge_checkout):
        monkeypatch.setenv("SPECFORGE_ROOT", str(specforge_checkout))
        adapter = SpecForgeAdapter()
        resolved = adapter.setup_backend_path()
        assert Path(resolved) == specforge_checkout.resolve()


class TestOverrideFlattening:
    def test_nested_and_dotted_keys_both_flatten(self):
        flat = flatten_overrides({"training": {"max_steps": 20}, "data.path": "/x"})
        assert flat == {"training.max_steps": "20", "data.path": "/x"}

    def test_booleans_use_lowercase_hydra_form(self):
        assert flatten_overrides({"model": {"use_liger_kernel": False}}) == {
            "model.use_liger_kernel": "false"
        }

    def test_namespace_input(self):
        ns = SimpleNamespace(training=SimpleNamespace(max_steps=5))
        assert flatten_overrides(ns) == {"training.max_steps": "5"}


class TestArgvBuilder:
    def test_config_is_required(self):
        with pytest.raises(ValueError, match="specforge_config"):
            build_specforge_argv(SimpleNamespace())

    def test_argv_shape(self):
        params = SimpleNamespace(
            specforge_config="/sf/configs/a.yaml",
            specforge_overrides={"training.max_steps": 20},
            output_dir="/out",
        )
        argv = build_specforge_argv(params)
        assert argv[:4] == ["specforge", "train", "--config", "/sf/configs/a.yaml"]
        assert "training.max_steps=20" in argv
        assert "output_dir=/out" in argv

    def test_resume_from_becomes_hydra_override(self):
        params = SimpleNamespace(
            specforge_config="/sf/a.yaml",
            specforge_overrides={"training.resume_from": "/ckpt/step10"},
        )
        argv = build_specforge_argv(params)
        assert "training.resume_from=/ckpt/step10" in argv

    def test_producer_drops_resume_from(self):
        params = SimpleNamespace(
            specforge_config="/sf/a.yaml",
            specforge_overrides={"training.resume_from": "/ckpt/step10", "training.max_steps": 2510},
        )
        producer = build_specforge_argv(params, role="producer")
        consumer = build_specforge_argv(
            params, extra_overrides=["training.resume_from=/ckpt/step10"], role="consumer"
        )
        assert "training.resume_from=/ckpt/step10" not in producer
        assert "training.max_steps=2510" in producer
        assert "training.resume_from=/ckpt/step10" in consumer

    def test_explicit_output_dir_override_wins(self):
        params = SimpleNamespace(
            specforge_config="/sf/a.yaml",
            specforge_overrides={"output_dir": "/explicit"},
            output_dir="/alias",
        )
        assert "output_dir=/explicit" in build_specforge_argv(params)
        assert "output_dir=/alias" not in build_specforge_argv(params)

    def test_custom_entrypoint(self):
        params = SimpleNamespace(specforge_config="/sf/a.yaml", specforge_entrypoint="specforge-wrapper")
        assert build_specforge_argv(params)[0] == "specforge-wrapper"

    def test_role_is_inserted_after_config(self):
        params = SimpleNamespace(
            specforge_config="/sf/a.yaml", specforge_overrides={"training.max_steps": 20}
        )
        argv = build_specforge_argv(params, role="producer")
        assert argv[:6] == ["specforge", "train", "--config", "/sf/a.yaml", "--role", "producer"]
        assert "training.max_steps=20" in argv
        consumer = build_specforge_argv(params, extra_overrides=["run_id=x"], role="consumer")
        assert consumer[4:6] == ["--role", "consumer"]
        assert "run_id=x" in consumer

    def test_offline_argv_has_no_role(self):
        params = SimpleNamespace(specforge_config="/sf/a.yaml")
        assert "--role" not in build_specforge_argv(params)

    def test_role_both_is_rejected(self):
        with pytest.raises(ValueError, match="both"):
            specforge_role(SimpleNamespace(specforge_role="both"))
        with pytest.raises(ValueError, match="unknown specforge_role"):
            specforge_role(SimpleNamespace(specforge_role="sidecar"))


class TestSpecForgeModes:
    def test_mode_is_train_or_capture(self):
        assert specforge_mode(SimpleNamespace(specforge_mode="train")) == "train"
        assert specforge_mode(SimpleNamespace(specforge_mode="capture")) == "capture"
        assert specforge_mode(SimpleNamespace()) == "train"

    def test_online_offline_are_not_modes(self):
        with pytest.raises(ValueError, match="specforge_train_mode: online"):
            specforge_mode(SimpleNamespace(specforge_mode="online"))
        with pytest.raises(ValueError, match="specforge_train_mode: offline"):
            specforge_mode(SimpleNamespace(specforge_mode="offline"))

    def test_train_mode_is_required_for_train(self):
        with pytest.raises(ValueError, match="specforge_train_mode is required"):
            specforge_train_mode(SimpleNamespace(specforge_mode="train"))
        with pytest.raises(ValueError, match="specforge_train_mode is required"):
            specforge_train_mode(SimpleNamespace())

    def test_train_mode_online_or_offline(self):
        assert (
            specforge_train_mode(SimpleNamespace(specforge_mode="train", specforge_train_mode="offline"))
            == "offline"
        )
        assert (
            specforge_train_mode(SimpleNamespace(specforge_mode="train", specforge_train_mode="online"))
            == "online"
        )

    def test_capture_ignores_train_mode(self):
        assert specforge_train_mode(SimpleNamespace(specforge_mode="capture")) is None
        assert (
            specforge_train_mode(SimpleNamespace(specforge_mode="capture", specforge_train_mode="offline"))
            is None
        )

    def test_is_online_train(self):
        assert is_online_train(SimpleNamespace(specforge_mode="train", specforge_train_mode="online"))
        assert not is_online_train(SimpleNamespace(specforge_mode="train", specforge_train_mode="offline"))
        assert not is_online_train(SimpleNamespace(specforge_mode="capture"))
        assert not is_online_train(SimpleNamespace(specforge_mode="online"))


class TestCaptureArgv:
    def test_capture_argv_runs_prepare_hidden_states(self):
        params = SimpleNamespace(
            specforge_mode="capture",
            specforge_capture={
                "target_model_path": "Qwen/Qwen3.5-4B",
                "strategy": "dflash",
                "trust_remote_code": True,
                "sglang_disable_radix_cache": True,
                "sglang_attention_backend": "aiter",
                "nproc_per_node": 1,
                "filter_output_path": "/filtered",
                "filter_min_kept": 1,
            },
        )
        argv = build_capture_argv(params)
        assert argv[:5] == [
            "torchrun",
            "--standalone",
            "--nproc_per_node",
            "1",
            "scripts/prepare_hidden_states.py",
        ]
        assert "--trust-remote-code" in argv
        assert "--sglang-disable-radix-cache" in argv
        assert argv[argv.index("--sglang-attention-backend") + 1] == "aiter"
        assert "filter_output_path" not in " ".join(argv)
        assert "--filter-min-kept" not in " ".join(argv)
        assert "--target-model-path" in argv

    def test_capture_omits_false_store_true_flags(self):
        params = SimpleNamespace(
            specforge_mode="capture",
            specforge_capture={"sglang_disable_radix_cache": False, "trust_remote_code": "false"},
        )
        argv = build_capture_argv(params)
        joined = " ".join(argv)
        assert "--sglang-disable-radix-cache" not in argv
        assert "--trust-remote-code" not in argv
        assert "false" not in joined

    def test_capture_defaults_aiter_and_radix_when_omitted(self):
        params = SimpleNamespace(
            specforge_mode="capture",
            specforge_capture={"target_model_path": "Qwen/Qwen3.5-4B"},
        )
        argv = build_capture_argv(params)
        assert "--sglang-disable-radix-cache" in argv
        assert argv[argv.index("--sglang-attention-backend") + 1] == "aiter"


class TestWorkdirResolution:
    def test_explicit_param_wins(self, specforge_checkout, tmp_path, monkeypatch):
        other = tmp_path / "other"
        other.mkdir()
        monkeypatch.setenv("SPECFORGE_ROOT", str(other))
        params = SimpleNamespace(specforge_root=str(specforge_checkout))
        assert resolve_specforge_root(params) == specforge_checkout

    def test_env_fallback(self, specforge_checkout):
        params = SimpleNamespace()
        env = {"SPECFORGE_ROOT": str(specforge_checkout)}
        assert resolve_specforge_root(params, env=env) == specforge_checkout

    def test_inferred_from_config_ancestor(self, specforge_checkout):
        config = specforge_checkout / "configs" / "dflash.yaml"
        config.write_text("{}")
        params = SimpleNamespace(specforge_config=str(config))
        assert resolve_specforge_root(params, env={}) == specforge_checkout

    def test_unresolvable_returns_none(self):
        params = SimpleNamespace(specforge_config="relative/path.yaml")
        assert resolve_specforge_root(params, env={}) is None


class TestExampleExperiment:
    def test_yaml_converts_to_specforge_argv(self, experiment_env):
        module = load_pre_trainer_params(EXAMPLE_CONFIG)
        assert module.framework == "specforge"

        adapter = SpecForgeAdapter()
        backend_args = adapter.convert_config(module.params)

        argv = build_specforge_argv(backend_args)
        assert argv[:3] == ["specforge", "train", "--config"]
        assert argv[3] == experiment_env["SPECFORGE_CONFIG"]
        assert "training.max_steps=20" in argv
        assert "model.use_liger_kernel=false" in argv
        assert f"data.hidden_states_path={experiment_env['HIDDEN_STATES_PATH']}" in argv
        assert f"output_dir={experiment_env['OUTPUT_DIR']}" in argv
        assert backend_args.specforge_root == experiment_env["SPECFORGE_ROOT"]
        assert backend_args.specforge_mode == "train"
        assert backend_args.specforge_train_mode == "offline"

    def test_capture_yaml_converts_to_prepare_hidden_states_argv(self, experiment_env):
        module = load_pre_trainer_params(CAPTURE_CONFIG)
        adapter = SpecForgeAdapter()
        backend_args = adapter.convert_config(module.params)
        assert backend_args.specforge_mode == "capture"
        assert backend_args.specforge_train_mode is None
        argv = build_capture_argv(backend_args)
        assert "scripts/prepare_hidden_states.py" in argv
        assert "--sglang-disable-radix-cache" in argv
        assert "--sglang-disable-radix-cache false" not in " ".join(argv)
        assert argv[argv.index("--sglang-attention-backend") + 1] == "aiter"
        assert "--strategy" in argv
        assert argv[argv.index("--strategy") + 1] == "dflash"
        assert argv[argv.index("--data-path") + 1] == experiment_env["CAPTURE_DATA_PATH"]
        assert (
            Path(argv[argv.index("--output-path") + 1])
            == Path(experiment_env["OUTPUT_DIR"]) / "hidden_states_raw"
        )
        assert "--filter-output-path" not in argv

    def test_trainer_rejects_missing_entrypoint(self, experiment_env):
        backend_args = SimpleNamespace(
            specforge_mode="train",
            specforge_train_mode="offline",
            specforge_config=experiment_env["SPECFORGE_CONFIG"],
            specforge_entrypoint="specforge-does-not-exist",
            specforge_root=experiment_env["SPECFORGE_ROOT"],
            specforge_overrides={},
        )
        trainer = SpecForgePretrainTrainer(backend_args=backend_args)
        with pytest.raises(RuntimeError, match="not found on PATH"):
            trainer.init()


class TestDistributedEnvHandoff:
    """primus-cli exports MASTER_ADDR/MASTER_PORT even in single mode."""

    def test_partial_env_is_cleared(self):
        env = {"MASTER_ADDR": "localhost", "MASTER_PORT": "1234"}
        assert clear_partial_distributed_env(env) == ["MASTER_ADDR", "MASTER_PORT"]
        assert env == {}

    def test_complete_torchrun_env_is_preserved(self):
        env = {
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "1234",
            "RANK": "0",
            "WORLD_SIZE": "8",
            "LOCAL_RANK": "0",
        }
        assert clear_partial_distributed_env(env) == []
        assert env["MASTER_ADDR"] == "localhost"

    def test_empty_env_is_a_noop(self):
        env = {}
        assert clear_partial_distributed_env(env) == []


class TestVisibleDeviceAlignment:
    """base_env.sh widens HIP to the whole node; vLLM asserts HIP == CUDA (job 99815)."""

    def test_whole_node_hip_is_narrowed_to_the_allocation(self):
        env = {"HIP_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7", "CUDA_VISIBLE_DEVICES": "0"}
        assert align_visible_devices(env) == ("0,1,2,3,4,5,6,7", "0")
        assert env["HIP_VISIBLE_DEVICES"] == "0"

    def test_matching_values_are_left_alone(self):
        env = {
            "HIP_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        }
        assert align_visible_devices(env) is None
        assert env["HIP_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"

    def test_unset_cuda_keeps_hip(self):
        env = {"HIP_VISIBLE_DEVICES": "0,1"}
        assert align_visible_devices(env) is None
        assert env["HIP_VISIBLE_DEVICES"] == "0,1"

    def test_empty_env_is_a_noop(self):
        env = {}
        assert align_visible_devices(env) is None
        assert env == {}


class TestCaptureExitCode:
    """Job 101977 captured 176 shards then primus-cli reported torchrun exit 1."""

    def test_system_exit_zero_is_not_wrapped_as_training_failure(self):
        from primus.core.runtime.train_runtime import PrimusRuntime

        runtime = PrimusRuntime(args=SimpleNamespace())

        def abort(*_args, **_kwargs):
            raise SystemExit(0)

        runtime._initialize_configuration = abort
        with pytest.raises(SystemExit) as caught:
            runtime.run_train_module("pre_trainer")
        assert caught.value.code == 0

    def test_capture_train_returns_after_successful_subprocess(self, monkeypatch):
        trainer = SpecForgePretrainTrainer(
            backend_args=SimpleNamespace(specforge_mode="capture", specforge_capture={})
        )
        trainer.argv = ["torchrun"]
        monkeypatch.setattr(
            "primus.backends.specforge.specforge_pretrain_trainer.subprocess.run",
            lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
        )
        assert trainer.train() is None


class TestFilterMinKept:
    """kept < 8 used to fail 1-GPU smokes that only kept a few shards."""

    def test_default_is_one(self):
        assert resolve_filter_min_kept(None) == 1
        assert resolve_filter_min_kept({}) == 1
        assert resolve_filter_min_kept({"filter_min_kept": ""}) == 1

    def test_explicit_value(self):
        assert resolve_filter_min_kept({"filter_min_kept": "8"}) == 8
        assert resolve_filter_min_kept({"filter_min_kept": 4}) == 4
        assert resolve_filter_min_kept({"filter_min_kept": 0}) == 1

    def _trainer(self, capture):
        trainer = SpecForgePretrainTrainer(
            backend_args=SimpleNamespace(specforge_mode="capture", specforge_capture=capture)
        )
        trainer.argv = ["torchrun"]
        return trainer

    def _stub_filter(self, monkeypatch, kept, dropped):
        fake = SimpleNamespace(filter_dflash_dir=lambda *_args, **_kwargs: (kept, dropped))
        monkeypatch.setitem(sys.modules, "primus.backends.specforge.filter_hidden_states", fake)

    def test_three_kept_shards_pass_default_min(self, monkeypatch):
        monkeypatch.setattr(
            "primus.backends.specforge.specforge_pretrain_trainer.subprocess.run",
            lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
        )
        self._stub_filter(monkeypatch, 3, 5)
        trainer = self._trainer({"output_path": "/raw", "filter_output_path": "/filtered"})
        assert trainer.train() is None

    def test_three_kept_shards_fail_when_min_is_eight(self, monkeypatch):
        monkeypatch.setattr(
            "primus.backends.specforge.specforge_pretrain_trainer.subprocess.run",
            lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
        )
        self._stub_filter(monkeypatch, 3, 5)
        trainer = self._trainer(
            {"output_path": "/raw", "filter_output_path": "/filtered", "filter_min_kept": 8}
        )
        with pytest.raises(SystemExit, match="too few kept shards \\(3\\); need at least 8"):
            trainer.train()


class TestPrepareHook:
    """The hook must tell primus-cli to skip torchrun; SpecForge self-launches."""

    def run_hook(self, env_overrides, extra_args=(), config=EXAMPLE_CONFIG):
        env = dict(os.environ)
        env.update(env_overrides)
        env["PYTHONPATH"] = os.pathsep.join([str(PRIMUS_ROOT), env.get("PYTHONPATH", "")])
        return subprocess.run(
            [
                sys.executable,
                str(PREPARE_HOOK),
                "--config",
                str(config),
                "--data_path",
                str(PRIMUS_ROOT / "data"),
                "--primus_path",
                str(PRIMUS_ROOT),
                *extra_args,
            ],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(PRIMUS_ROOT),
        )

    def test_emits_single_run_mode(self, experiment_env):
        result = self.run_hook(experiment_env)
        assert result.returncode == 0, result.stderr
        assert "env.RUN_MODE=single" in result.stdout
        assert "env.GPUS_PER_NODE=1" in result.stdout
        assert f"env.SPECFORGE_ROOT={experiment_env['SPECFORGE_ROOT']}" in result.stdout
        combined = (result.stdout + result.stderr).lower()
        assert "stack preflight ok" in combined
        if enforce_rocm_stack({**experiment_env, **dict(os.environ)}):
            assert "env.SGLANG_USE_AITER=1" in result.stdout
            assert "env.SGLANG_DISABLE_RADIX_CACHE=1" in result.stdout

    def test_capture_yaml_emits_single_run_mode(self, experiment_env):
        result = self.run_hook(experiment_env, config=CAPTURE_CONFIG)
        assert result.returncode == 0, result.stderr
        assert "env.RUN_MODE=single" in result.stdout
        assert "env.GPUS_PER_NODE=1" in result.stdout
        combined = (result.stdout + result.stderr).lower()
        assert "stack preflight ok" in combined

    def test_fails_fast_on_empty_hidden_states(self, experiment_env, tmp_path):
        empty = tmp_path / "empty_hidden_states"
        empty.mkdir()
        result = self.run_hook({**experiment_env, "HIDDEN_STATES_PATH": str(empty)})
        assert result.returncode != 0
        assert "empty" in result.stderr.lower()

    def test_fails_fast_on_missing_hidden_states(self, experiment_env, tmp_path):
        result = self.run_hook({**experiment_env, "HIDDEN_STATES_PATH": str(tmp_path / "nope")})
        assert result.returncode != 0
        assert "not a directory" in result.stderr.lower()


class TestStackPreflight:
    def _params(self, specforge_checkout, hidden_states, **overrides):
        return SimpleNamespace(
            specforge_mode="train",
            specforge_train_mode="offline",
            specforge_config=str(specforge_checkout / "configs" / "qwen3.5-4b-dflash.yaml"),
            specforge_root=str(specforge_checkout),
            specforge_overrides={"data.hidden_states_path": str(hidden_states), **overrides},
        )

    def test_apply_fills_aiter_and_radix_defaults(self):
        env = {}
        applied = apply_rocm_stack_env(env)
        assert dict(applied)["SGLANG_USE_AITER"] == "1"
        assert env["SGLANG_DISABLE_RADIX_CACHE"] == "1"

    def test_apply_does_not_override_explicit_zero(self):
        env = {"SGLANG_USE_AITER": "0"}
        apply_rocm_stack_env(env)
        assert env["SGLANG_USE_AITER"] == "0"
        assert env["SGLANG_DISABLE_RADIX_CACHE"] == "1"

    def test_empty_hidden_states_is_an_issue(self, specforge_checkout, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        params = self._params(specforge_checkout, empty)
        issues = collect_issues(params, env={"PRIMUS_SPECFORGE_ENFORCE_ROCM": "0"})
        assert any("empty" in item.lower() for item in issues)

    def test_aiter_opt_out_fails_when_enforced(self, specforge_checkout, hidden_states):
        params = self._params(specforge_checkout, hidden_states)
        issues = collect_issues(
            params,
            env={
                "PRIMUS_SPECFORGE_ENFORCE_ROCM": "1",
                "SGLANG_USE_AITER": "0",
                "SGLANG_DISABLE_RADIX_CACHE": "1",
            },
        )
        assert any("SGLANG_USE_AITER=0" in item for item in issues)

    def test_radix_opt_out_fails_when_enforced(self, specforge_checkout, hidden_states):
        params = self._params(specforge_checkout, hidden_states)
        issues = collect_issues(
            params,
            env={
                "PRIMUS_SPECFORGE_ENFORCE_ROCM": "1",
                "SGLANG_USE_AITER": "1",
                "SGLANG_DISABLE_RADIX_CACHE": "0",
            },
        )
        assert any("SGLANG_DISABLE_RADIX_CACHE=0" in item for item in issues)

    def test_missing_specforge_config_file(self, specforge_checkout, hidden_states):
        params = self._params(specforge_checkout, hidden_states)
        params.specforge_config = str(specforge_checkout / "configs" / "nope.yaml")
        issues = collect_issues(params, env={"PRIMUS_SPECFORGE_ENFORCE_ROCM": "0"})
        assert any("specforge_config is not a file" in item for item in issues)

    def test_hip_visible_devices_opts_into_enforce(self):
        assert enforce_rocm_stack({"HIP_VISIBLE_DEVICES": "0"}, kind="missing") is True
        assert enforce_rocm_stack({}, kind="missing") is False

    def test_capture_skips_hidden_states_existence(self, specforge_checkout, tmp_path):
        data = tmp_path / "sharegpt.jsonl"
        data.write_text("{}\n")
        out = tmp_path / "raw"
        params = SimpleNamespace(
            specforge_mode="capture",
            specforge_root=str(specforge_checkout),
            specforge_capture={
                "data_path": str(data),
                "output_path": str(out),
                "draft_model_config": "configs/qwen3.5-4b-dflash.json",
                "sglang_disable_radix_cache": True,
            },
        )
        issues = collect_issues(params, env={"PRIMUS_SPECFORGE_ENFORCE_ROCM": "0"})
        assert issues == []

    def test_capture_missing_data_path(self, specforge_checkout):
        params = SimpleNamespace(
            specforge_mode="capture",
            specforge_root=str(specforge_checkout),
            specforge_capture={"output_path": "/tmp/raw"},
        )
        issues = collect_issues(params, env={"PRIMUS_SPECFORGE_ENFORCE_ROCM": "0"})
        assert any("data_path" in item for item in issues)

    def test_train_requires_train_mode(self, specforge_checkout, hidden_states):
        params = self._params(specforge_checkout, hidden_states)
        params.specforge_train_mode = None
        issues = collect_issues(params, env={"PRIMUS_SPECFORGE_ENFORCE_ROCM": "0"})
        assert any("specforge_train_mode is required" in item for item in issues)

    def test_online_as_mode_is_rejected(self, specforge_checkout, hidden_states):
        params = self._params(specforge_checkout, hidden_states)
        params.specforge_mode = "online"
        issues = collect_issues(params, env={"PRIMUS_SPECFORGE_ENFORCE_ROCM": "0"})
        assert any("specforge_train_mode: online" in item for item in issues)


class TestOnlineLaunch:
    def _params(self, specforge_checkout, tmp_path, **online):
        run_root = tmp_path / "run"
        consumer = tmp_path / "consumer-state"
        return SimpleNamespace(
            specforge_mode="train",
            specforge_train_mode="online",
            specforge_config=str(specforge_checkout / "configs" / "qwen3.5-4b-dflash.yaml"),
            specforge_root=str(specforge_checkout),
            specforge_overrides={"model.use_liger_kernel": False},
            specforge_online={
                "run_id": "unit-1",
                "run_root": str(run_root),
                "consumer_state_dir": str(consumer),
                "server_gpus": "0",
                "trainer_gpus": "0",
                "trainer_nproc": 1,
                **online,
            },
            output_dir=str(tmp_path / "out"),
        )

    def test_head_ip_overrides_rewrite_loopback(self, specforge_checkout, tmp_path):
        params = self._params(specforge_checkout, tmp_path)
        settings = online_settings(params)
        overrides = build_online_overrides("10.3.14.7", settings, output_dir=str(tmp_path / "out"))
        joined = " ".join(overrides)
        assert "127.0.0.1" not in joined
        assert 'deployment.disaggregated.server_urls=["http://10.3.14.7:30000"]' in overrides
        assert "deployment.disaggregated.mooncake_master_server_addr=10.3.14.7:35551" in overrides
        assert (
            "deployment.disaggregated.mooncake_metadata_server=http://10.3.14.7:35880/metadata" in overrides
        )
        assert "deployment.trainer.nnodes=1" in overrides
        argv = build_role_argv(params, "producer", overrides)
        assert argv[4:6] == ["--role", "producer"]
        consumer = build_role_argv(params, "consumer", overrides)
        assert consumer[4:6] == ["--role", "consumer"]
        assert "--node-rank" not in consumer

    def test_trainer_nnodes_emits_master_addr_and_node_rank(self, specforge_checkout, tmp_path):
        params = self._params(specforge_checkout, tmp_path, trainer_nnodes=2)
        settings = online_settings(params)
        overrides = build_online_overrides(
            "10.3.14.7", settings, trainer_master_addr="10.9.9.1", output_dir=str(tmp_path / "out")
        )
        assert "deployment.trainer.nnodes=2" in overrides
        assert "deployment.trainer.master_addr=10.9.9.1" in overrides
        argv = build_role_argv(params, "consumer", overrides, node_rank=1)
        assert argv[argv.index("--node-rank") + 1] == "1"
        producer = build_role_argv(params, "producer", overrides)
        assert "deployment.trainer.master_addr=10.9.9.1" in producer

    def test_resolves_head_ip_override_not_hostname(self):
        ip = resolve_routable_ip(env={"HEAD_IP": "10.9.8.7", "MASTER_ADDR": "gpu-node-001.example"})
        assert ip == "10.9.8.7"

    def test_head_ip_beats_interface_ip(self):
        ip = resolve_routable_ip(
            env={"HEAD_IP": "10.9.8.7", "PRIMUS_SPECFORGE_BIND_IFACE": "eth0"},
            interface_ip="10.4.5.6",
            hostname_ips=["127.0.0.1"],
        )
        assert ip == "10.9.8.7"

    def test_hostname_ipv4_fallback_skips_loopback(self):
        ip = resolve_routable_ip(env={}, hostname_ips=["127.0.0.1", "10.1.2.3"])
        assert ip == "10.1.2.3"

    def test_local_ip_ignores_job_wide_head_ip(self):
        ip = resolve_local_ip(
            env={"HEAD_IP": "10.9.8.7", "PRIMUS_SPECFORGE_BIND_IFACE": "eth0"},
            interface_ip="10.4.5.6",
            hostname_ips=["127.0.0.1"],
        )
        assert ip == "10.4.5.6"

    def test_local_ip_override_beats_interface(self):
        ip = resolve_local_ip(
            env={"PRIMUS_SPECFORGE_LOCAL_IP": "10.2.2.2", "HEAD_IP": "10.9.8.7"},
            interface_ip="10.4.5.6",
        )
        assert ip == "10.2.2.2"

    def test_head_ip_still_honors_head_override(self):
        assert resolve_head_ip(env={"HEAD_IP": "10.9.8.7"}, interface_ip="10.4.5.6") == "10.9.8.7"

    def test_expand_single_device_id_to_count(self):
        assert expand_device_list("0", 8) == "0,1,2,3,4,5,6,7"
        assert expand_device_list("1", 4) == "1,2,3,4"
        assert expand_device_list("0,2", 8) == "0,2"
        assert expand_device_list("0", 1) == "0"

    def test_eight_plus_eight_defaults_expand_gpu_lists(self, specforge_checkout, tmp_path):
        params = self._params(specforge_checkout, tmp_path, server_count=8, trainer_nproc=8)
        settings = online_settings(params)
        assert settings["server_gpus"] == "0,1,2,3,4,5,6,7"
        assert settings["trainer_gpus"] == "0,1,2,3,4,5,6,7"
        assert not validate_online_identity(settings, rank=0, nodes=2)

    def test_bind_interface_defaults_empty(self, specforge_checkout, tmp_path):
        settings = online_settings(self._params(specforge_checkout, tmp_path), env={})
        assert settings["bind_interface"] == ""

    def test_bind_interface_from_env(self, specforge_checkout, tmp_path):
        settings = online_settings(
            self._params(specforge_checkout, tmp_path),
            env={"PRIMUS_SPECFORGE_BIND_IFACE": "eth0"},
        )
        assert settings["bind_interface"] == "eth0"

    def test_rank_helpers(self):
        assert node_rank({"NODE_RANK": "1", "SLURM_NODEID": "0"}) == 1
        assert nnodes({"NNODES": "2"}) == 2

    def test_identity_rejects_single_node(self, specforge_checkout, tmp_path):
        params = self._params(specforge_checkout, tmp_path)
        settings = online_settings(params)
        issues = validate_online_identity(settings, rank=0, nodes=1)
        assert any("capture_nnodes+trainer_nnodes=2" in item for item in issues)

    def test_identity_accepts_capture_plus_trainer_counts(self, specforge_checkout, tmp_path):
        params = self._params(specforge_checkout, tmp_path, capture_nnodes=1, trainer_nnodes=2)
        settings = online_settings(params)
        assert not validate_online_identity(settings, rank=2, nodes=3)
        issues = validate_online_identity(settings, rank=0, nodes=2)
        assert any("capture_nnodes+trainer_nnodes=3" in item for item in issues)

    def test_sglang_aux_layers_match_qwen35_draft(self, specforge_checkout, tmp_path):
        params = self._params(specforge_checkout, tmp_path)
        argv = sglang_server_argv(0, online_settings(params))
        ids = argv[argv.index("--spec-capture-aux-layer-ids") + 1 :]
        flags = ids[: ids.index("--port")] if "--port" in ids else ids
        # port comes after layer ids in our argv... actually port is after layer ids.
        # argv has ... --spec-capture-aux-layer-ids 1 8 15 22 29 --port 30000 ...
        port_at = argv.index("--port")
        layer_at = argv.index("--spec-capture-aux-layer-ids")
        assert argv[layer_at + 1 : port_at] == ["1", "8", "15", "22", "29"]
        assert "--host" in argv and argv[argv.index("--host") + 1] == "0.0.0.0"
        assert "--disable-radix-cache" in argv
        assert "--attention-backend" in argv

    def test_capture_layer_ids_env_overrides_draft_json(self, specforge_checkout, tmp_path):
        params = self._params(specforge_checkout, tmp_path)
        settings = online_settings(params, env={"CAPTURE_LAYER_IDS": "2 4"})
        argv = sglang_server_argv(0, settings)
        port_at = argv.index("--port")
        layer_at = argv.index("--spec-capture-aux-layer-ids")
        assert argv[layer_at + 1 : port_at] == ["2", "4"]

    def test_capture_layer_ids_ignore_other_draft_json(self, specforge_checkout, tmp_path):
        other = specforge_checkout / "configs" / "other-draft.json"
        other.write_text('{"dflash_config": {"target_layer_ids": [3, 9]}}\n')
        params = self._params(specforge_checkout, tmp_path)
        settings = online_settings(params)
        assert settings["capture_layer_ids"] == (1, 8, 15, 22, 29)
        (specforge_checkout / "configs" / "qwen3.5-4b-dflash.json").unlink()
        settings = online_settings(params)
        assert settings["capture_layer_ids"] == (1, 8, 15, 22, 29)

    def test_online_yaml_has_no_cluster_ips(self):
        text = ONLINE_CONFIG.read_text(encoding="utf-8")
        assert "specforge_mode: train" in text
        assert "specforge_train_mode: online" in text
        for line in text.splitlines():
            assert "127.0.0.1" not in line.split("#", 1)[0]

    def test_specforge_online_tree_has_no_site_cluster_names(self):
        banned = (
            "shared_nfs",
            "amd-spur",
            "amd-burst",
            "crusoe",
            "crsuse2",
            "m2m_nobackup",
            "ens3",
        )
        roots = [
            PRIMUS_ROOT / "examples" / "specforge",
            PRIMUS_ROOT / "primus" / "configs" / "modules" / "specforge",
            PRIMUS_ROOT / "primus" / "backends" / "specforge",
        ]
        hits = []
        for root in roots:
            for path in root.rglob("*"):
                if not path.is_file() or path.suffix.lower() in {".png", ".jpg", ".pyc"}:
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore").lower()
                for token in banned:
                    if token in text:
                        hits.append(f"{path.relative_to(PRIMUS_ROOT)}:{token}")
        assert hits == []

    def test_online_yaml_converts_to_role_argv(self, experiment_env, specforge_checkout, tmp_path):
        module = load_pre_trainer_params(ONLINE_CONFIG)
        adapter = SpecForgeAdapter()
        backend_args = adapter.convert_config(module.params)
        assert backend_args.specforge_mode == "train"
        assert backend_args.specforge_train_mode == "online"
        settings = online_settings(backend_args, env=experiment_env)
        overrides = build_online_overrides("10.1.2.3", settings)
        argv = build_role_argv(backend_args, "producer", overrides)
        assert "--role" in argv
        assert argv[argv.index("--role") + 1] == "producer"
        assert "model.use_liger_kernel=false" in argv

    def test_online_preflight_skips_hidden_states(self, specforge_checkout, tmp_path):
        params = self._params(specforge_checkout, tmp_path)
        issues = collect_issues(
            params,
            env={
                "PRIMUS_SPECFORGE_ENFORCE_ROCM": "0",
                "NNODES": "2",
                "NODE_RANK": "0",
            },
        )
        assert not any("hidden" in item.lower() for item in issues)

    def test_online_preflight_rejects_stale_run_root(self, specforge_checkout, tmp_path):
        params = self._params(specforge_checkout, tmp_path)
        run_root = tmp_path / "run"
        run_root.mkdir()
        (run_root / "stale.sqlite").write_text("x")
        issues = collect_issues(
            params,
            env={"PRIMUS_SPECFORGE_ENFORCE_ROCM": "0", "NNODES": "2", "NODE_RANK": "0"},
        )
        assert any("run_root is not empty" in item for item in issues)

    def test_online_preflight_allows_trainer_ip_only(self, specforge_checkout, tmp_path):
        params = self._params(specforge_checkout, tmp_path)
        run_root = tmp_path / "run"
        run_root.mkdir()
        (run_root / "trainer.ip").write_text("10.1.2.3\n")
        issues = collect_issues(
            params,
            env={"PRIMUS_SPECFORGE_ENFORCE_ROCM": "0", "NNODES": "2", "NODE_RANK": "0"},
        )
        assert not any("run_root is not empty" in item for item in issues)

    def test_online_preflight_non_rank0_allows_in_progress_run_root(self, specforge_checkout, tmp_path):
        params = self._params(specforge_checkout, tmp_path)
        run_root = tmp_path / "run"
        run_root.mkdir()
        (run_root / "head.ip").write_text("10.1.2.3\n")
        (run_root / "mooncake.log").write_text("x")
        issues = collect_issues(
            params,
            env={"PRIMUS_SPECFORGE_ENFORCE_ROCM": "0", "NNODES": "3", "NODE_RANK": "1"},
        )
        assert not any("run_root is not empty" in item for item in issues)

    def test_online_preflight_requires_two_nodes(self, specforge_checkout, tmp_path):
        params = self._params(specforge_checkout, tmp_path)
        issues = collect_issues(
            params,
            env={"PRIMUS_SPECFORGE_ENFORCE_ROCM": "0", "NNODES": "1", "NODE_RANK": "0"},
        )
        assert any("capture_nnodes+trainer_nnodes=2" in item for item in issues)

    def test_rank0_and_rank1_dispatch(self, specforge_checkout, tmp_path, monkeypatch):
        from primus.backends.specforge import online_supervisor as supervisor

        calls = []
        monkeypatch.setattr(supervisor, "_run_capture_node", lambda *a, **k: calls.append("capture"))
        monkeypatch.setattr(supervisor, "_run_trainer_node", lambda *a, **k: calls.append("trainer"))
        params = self._params(specforge_checkout, tmp_path)
        monkeypatch.setenv("NNODES", "2")
        monkeypatch.setenv("NODE_RANK", "0")
        supervisor.run_online(params)
        monkeypatch.setenv("NODE_RANK", "1")
        supervisor.run_online(params)
        assert calls == ["capture", "trainer"]

    def test_dispatch_multi_trainer_and_capture_replica(self, specforge_checkout, tmp_path, monkeypatch):
        from primus.backends.specforge import online_supervisor as supervisor

        calls = []
        monkeypatch.setattr(supervisor, "_run_capture_node", lambda *a, **k: calls.append("head"))
        monkeypatch.setattr(
            supervisor, "_run_capture_replica", lambda params, settings, rank: calls.append(("replica", rank))
        )
        monkeypatch.setattr(
            supervisor,
            "_run_trainer_node",
            lambda params, settings, specforge_rank: calls.append(("trainer", specforge_rank)),
        )
        params = self._params(specforge_checkout, tmp_path, capture_nnodes=2, trainer_nnodes=2)
        monkeypatch.setenv("NNODES", "4")
        monkeypatch.setenv("NODE_RANK", "0")
        supervisor.run_online(params)
        monkeypatch.setenv("NODE_RANK", "1")
        supervisor.run_online(params)
        monkeypatch.setenv("NODE_RANK", "2")
        supervisor.run_online(params)
        monkeypatch.setenv("NODE_RANK", "3")
        supervisor.run_online(params)
        assert calls == ["head", ("replica", 1), ("trainer", 0), ("trainer", 1)]

    def test_mooncake_http_404_counts_as_up(self, monkeypatch):
        from primus.backends.specforge import online_supervisor as supervisor

        def boom(*_a, **_k):
            raise urllib.error.HTTPError("http://x/metadata", 404, "not found", hdrs=None, fp=None)

        monkeypatch.setattr(supervisor.urllib.request, "urlopen", boom)
        assert supervisor._http_up("http://x/metadata") is True
        assert supervisor._http_ok("http://x/metadata") is False

    def test_capture_stack_failed_detects_dead_sglang(self):
        from primus.backends.specforge import online_supervisor as supervisor

        class Dead:
            def poll(self):
                return 1

        settings = {
            "mooncake_http_port": 35880,
            "mooncake_rpc_port": 35551,
        }
        reason = supervisor._capture_stack_failed("10.0.0.1", settings, None, [Dead()])
        assert reason == "SGLang server 0 exited"

    def test_specforge_child_env_forces_single_node_rank(self, monkeypatch):
        from primus.backends.specforge import online_supervisor as supervisor

        monkeypatch.setenv("NODE_RANK", "1")
        monkeypatch.setenv("NNODES", "2")
        monkeypatch.setenv("SLURM_NODEID", "1")
        monkeypatch.setenv("SLURM_NNODES", "2")
        env = supervisor._specforge_child_env({}, "0")
        assert env["NODE_RANK"] == "0"
        assert env["NNODES"] == "1"
        assert "SLURM_NODEID" not in env
        assert "SLURM_NNODES" not in env
        assert env["CUDA_VISIBLE_DEVICES"] == "0"

        env = supervisor._specforge_child_env({}, "0", specforge_node_rank=1, specforge_nnodes=2)
        assert env["NODE_RANK"] == "1"
        assert env["NNODES"] == "2"

    def test_online_train_does_not_execvp(self, specforge_checkout, tmp_path, monkeypatch):
        called = []
        monkeypatch.setattr(
            "primus.backends.specforge.online_supervisor.run_online",
            lambda params: called.append("online"),
        )
        monkeypatch.setattr("os.execvp", lambda *a, **k: called.append("exec"))
        params = self._params(specforge_checkout, tmp_path)
        trainer = SpecForgePretrainTrainer(backend_args=params)
        trainer.argv = None
        trainer.workdir = None
        trainer.train()
        assert called == ["online"]

    def test_prepare_hook_online_emits_single_run_mode(self, experiment_env):
        result = TestPrepareHook().run_hook(experiment_env, config=ONLINE_CONFIG)
        assert result.returncode == 0, result.stderr
        assert "env.RUN_MODE=single" in result.stdout
        assert "env.GPUS_PER_NODE=1" in result.stdout
        combined = (result.stdout + result.stderr).lower()
        assert "stack preflight ok" in combined
        assert "hidden-states" not in combined


class TestPrimusParserAcceptsSpecForge:
    def test_framework_dispatch_name(self, experiment_env):
        """prepare_experiment.sh routes on this exact string."""
        parser = PrimusParser()
        cfg = parser.parse(SimpleNamespace(config=str(EXAMPLE_CONFIG)))
        assert cfg.get_module_config("pre_trainer").framework == "specforge"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
