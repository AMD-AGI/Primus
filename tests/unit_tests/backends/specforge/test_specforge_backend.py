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
from pathlib import Path
from types import SimpleNamespace

import pytest

from primus.backends.specforge.argument_builder import (
    build_capture_argv,
    build_specforge_argv,
    flatten_overrides,
    resolve_specforge_root,
)
from primus.backends.specforge.specforge_adapter import SpecForgeAdapter
from primus.backends.specforge.specforge_pretrain_trainer import (
    SpecForgePretrainTrainer,
    align_visible_devices,
    clear_partial_distributed_env,
)
from primus.backends.specforge.stack_preflight import (
    apply_rocm_stack_env,
    collect_issues,
    enforce_rocm_stack,
)
from primus.core.backend.backend_registry import BackendRegistry
from primus.core.launcher.parser import PrimusParser

PRIMUS_ROOT = Path(__file__).resolve().parents[4]
EXAMPLE_CONFIG = PRIMUS_ROOT / "examples" / "specforge" / "configs" / "qwen3.5-4b-dflash-offline.yaml"
CAPTURE_CONFIG = PRIMUS_ROOT / "examples" / "specforge" / "configs" / "qwen3.5-4b-dflash-offline-capture.yaml"
PREPARE_HOOK = PRIMUS_ROOT / "runner/helpers/hooks/train/pretrain/specforge/prepare.py"


@pytest.fixture
def specforge_checkout(tmp_path):
    """A directory that resolve_specforge_root() recognizes as a SpecForge tree."""
    root = tmp_path / "SpecForge"
    (root / "configs").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'specforge'\n")
    (root / "configs" / "qwen3.5-4b-dflash.yaml").write_text("model: dummy\n")
    (root / "configs" / "qwen3.5-4b-dflash.json").write_text("{}\n")
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

    def test_capture_yaml_converts_to_prepare_hidden_states_argv(self, experiment_env):
        module = load_pre_trainer_params(CAPTURE_CONFIG)
        adapter = SpecForgeAdapter()
        backend_args = adapter.convert_config(module.params)
        assert backend_args.specforge_mode == "capture"
        argv = build_capture_argv(backend_args)
        assert "scripts/prepare_hidden_states.py" in argv
        assert "--sglang-disable-radix-cache" in argv
        assert "--sglang-disable-radix-cache false" not in " ".join(argv)
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


class TestPrimusParserAcceptsSpecForge:
    def test_framework_dispatch_name(self, experiment_env):
        """prepare_experiment.sh routes on this exact string."""
        parser = PrimusParser()
        cfg = parser.parse(SimpleNamespace(config=str(EXAMPLE_CONFIG)))
        assert cfg.get_module_config("pre_trainer").framework == "specforge"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
