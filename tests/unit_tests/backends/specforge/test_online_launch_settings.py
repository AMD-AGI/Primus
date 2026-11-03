###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Pure-Python online_settings coverage. No SpecForge overlay required."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from primus.backends.specforge.online_launch import online_settings, sglang_server_argv


@pytest.fixture
def specforge_checkout(tmp_path):
    root = tmp_path / "SpecForge"
    (root / "configs").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'specforge'\n")
    (root / "configs" / "qwen3.5-4b-dflash.yaml").write_text(
        "model:\n  draft_model_config: configs/qwen3.5-4b-dflash.json\n"
    )
    (root / "configs" / "qwen3.5-4b-dflash.json").write_text(
        '{"dflash_config": {"target_layer_ids": [1, 8, 15, 22, 29]}}\n'
    )
    return root


def _params(specforge_checkout, tmp_path, **online):
    return SimpleNamespace(
        specforge_mode="train",
        specforge_train_mode="online",
        specforge_config=str(specforge_checkout / "configs" / "qwen3.5-4b-dflash.yaml"),
        specforge_root=str(specforge_checkout),
        specforge_overrides={"model.use_liger_kernel": False},
        specforge_online={
            "run_id": "unit-1",
            "run_root": str(tmp_path / "run"),
            "consumer_state_dir": str(tmp_path / "consumer-state"),
            "server_gpus": "0",
            "trainer_gpus": "0",
            "trainer_nproc": 1,
            **online,
        },
        output_dir=str(tmp_path / "out"),
    )


def _write_hydra(root: Path, body: str, name: str = "recipe.yaml") -> Path:
    path = root / "configs" / name
    path.write_text(body, encoding="utf-8")
    return path


class TestOnlineLaunchSettings:
    def test_sglang_aux_layers_match_draft_json(self, specforge_checkout, tmp_path):
        params = _params(specforge_checkout, tmp_path)
        argv = sglang_server_argv(0, online_settings(params, env={}))
        port_at = argv.index("--port")
        layer_at = argv.index("--spec-capture-aux-layer-ids")
        assert argv[layer_at + 1 : port_at] == ["1", "8", "15", "22", "29"]
        assert argv.count("--skip-tokenizer-init") == 1
        assert "--disable-radix-cache" in argv

    def test_capture_layer_ids_ignore_online_and_env_overrides(self, specforge_checkout, tmp_path):
        params = _params(specforge_checkout, tmp_path, capture_layer_ids="2,4")
        settings = online_settings(params, env={"CAPTURE_LAYER_IDS": "9 9"})
        assert settings["capture_layer_ids"] == (1, 8, 15, 22, 29)

    def test_override_draft_beats_hydra_file(self, specforge_checkout, tmp_path):
        other = specforge_checkout / "configs" / "other.json"
        other.write_text('{"dflash_config": {"target_layer_ids": [3, 9]}}\n', encoding="utf-8")
        params = _params(specforge_checkout, tmp_path)
        params.specforge_overrides = {"model.draft_model_config": "configs/other.json"}
        assert online_settings(params, env={})["capture_layer_ids"] == (3, 9)

    def test_missing_named_draft_does_not_fall_through(self, specforge_checkout, tmp_path):
        params = _params(specforge_checkout, tmp_path)
        params.specforge_overrides = {"model.draft_model_config": "configs/missing.json"}
        with pytest.raises(ValueError, match="is not a file"):
            online_settings(params, env={})

    def test_relative_draft_ignores_cwd_shadow(self, specforge_checkout, tmp_path, monkeypatch):
        shadow = tmp_path / "cwd"
        (shadow / "configs").mkdir(parents=True)
        (shadow / "configs" / "qwen3.5-4b-dflash.json").write_text(
            '{"dflash_config": {"target_layer_ids": [2, 4]}}\n',
            encoding="utf-8",
        )
        monkeypatch.chdir(shadow)
        params = _params(specforge_checkout, tmp_path)
        assert online_settings(params, env={})["capture_layer_ids"] == (1, 8, 15, 22, 29)

    def test_relative_specforge_config_resolves_under_root(self, specforge_checkout, tmp_path, monkeypatch):
        shadow = tmp_path / "cwd"
        (shadow / "configs").mkdir(parents=True)
        (shadow / "configs" / "qwen3.5-4b-dflash.yaml").write_text(
            "model:\n  draft_model_config: configs/not-the-checkout.json\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(shadow)
        params = _params(specforge_checkout, tmp_path)
        params.specforge_config = "configs/qwen3.5-4b-dflash.yaml"
        assert online_settings(params, env={})["capture_layer_ids"] == (1, 8, 15, 22, 29)

    def test_relative_specforge_config_missing_under_root_raises(
        self, specforge_checkout, tmp_path, monkeypatch
    ):
        shadow = tmp_path / "cwd"
        (shadow / "configs").mkdir(parents=True)
        (shadow / "configs" / "only-cwd.yaml").write_text(
            "model:\n  draft_model_config: configs/qwen3.5-4b-dflash.json\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(shadow)
        params = _params(specforge_checkout, tmp_path)
        params.specforge_config = "configs/only-cwd.yaml"
        with pytest.raises(ValueError, match="specforge_config is not a file"):
            online_settings(params, env={})

    def test_malformed_yaml_raises(self, specforge_checkout, tmp_path):
        hydra = _write_hydra(specforge_checkout, "model: [\n")
        params = _params(specforge_checkout, tmp_path)
        params.specforge_config = str(hydra)
        with pytest.raises(ValueError, match="could not read"):
            online_settings(params, env={})

    def test_unresolved_interpolation_raises(self, specforge_checkout, tmp_path):
        hydra = _write_hydra(
            specforge_checkout,
            "model:\n"
            "  draft_model_config: configs/qwen3.5-4b-dflash.json\n"
            "  target_model_path: ${TARGET_MODEL}\n",
        )
        params = _params(specforge_checkout, tmp_path)
        params.specforge_config = str(hydra)
        params.specforge_overrides = {}
        with pytest.raises(ValueError, match="unresolved interpolation"):
            online_settings(params, env={})

    def test_non_scalar_model_field_raises(self, specforge_checkout, tmp_path):
        hydra = _write_hydra(
            specforge_checkout,
            "model:\n  draft_model_config:\n    path: configs/qwen3.5-4b-dflash.json\n",
        )
        params = _params(specforge_checkout, tmp_path)
        params.specforge_config = str(hydra)
        params.specforge_overrides = {}
        with pytest.raises(ValueError, match="must be a scalar"):
            online_settings(params, env={})

    def test_target_model_beats_hydra_file(self, specforge_checkout, tmp_path):
        hydra = _write_hydra(
            specforge_checkout,
            "model:\n"
            "  draft_model_config: configs/qwen3.5-4b-dflash.json\n"
            "  target_model_path: /models/from-file\n",
        )
        params = _params(specforge_checkout, tmp_path)
        params.specforge_config = str(hydra)
        params.specforge_overrides = {}
        settings = online_settings(params, env={"TARGET_MODEL": "/models/from-env"})
        assert settings["target_model_path"] == "/models/from-env"

    def test_online_target_beats_target_model_env(self, specforge_checkout, tmp_path):
        params = _params(specforge_checkout, tmp_path, target_model_path="/models/online")
        settings = online_settings(params, env={"TARGET_MODEL": "/models/from-env"})
        assert settings["target_model_path"] == "/models/online"

    def test_override_target_beats_target_model_env(self, specforge_checkout, tmp_path):
        params = _params(specforge_checkout, tmp_path)
        params.specforge_overrides = {"model.target_model_path": "/models/override"}
        settings = online_settings(params, env={"TARGET_MODEL": "/models/from-env"})
        assert settings["target_model_path"] == "/models/override"

    def test_skip_tokenizer_init_env_strings(self, specforge_checkout, tmp_path):
        params = _params(specforge_checkout, tmp_path)
        assert online_settings(params, env={"SKIP_TOKENIZER_INIT": "false"})["skip_tokenizer_init"] is False
        assert online_settings(params, env={"SKIP_TOKENIZER_INIT": "no"})["skip_tokenizer_init"] is False
        with pytest.raises(ValueError, match="boolean"):
            online_settings(params, env={"SKIP_TOKENIZER_INIT": "maybe"})

    def test_string_false_in_online_settings_disables_skip(self, specforge_checkout, tmp_path):
        params = _params(specforge_checkout, tmp_path, skip_tokenizer_init="false")
        argv = sglang_server_argv(0, online_settings(params, env={}))
        assert "--skip-tokenizer-init" not in argv

    def test_extra_args_skip_tokenizer_flag_is_deduped(self, specforge_checkout, tmp_path):
        params = _params(
            specforge_checkout,
            tmp_path,
            sglang_extra_args="--skip-tokenizer-init --attention-backend aiter",
        )
        argv = sglang_server_argv(0, online_settings(params, env={}))
        assert argv.count("--skip-tokenizer-init") == 1
        params.specforge_online["skip_tokenizer_init"] = False
        argv = sglang_server_argv(0, online_settings(params, env={}))
        assert argv.count("--skip-tokenizer-init") == 0

    def test_capture_layer_ids_follow_absolute_draft_json(self, specforge_checkout, tmp_path):
        draft = tmp_path / "muse-spark-dflash2.json"
        draft.write_text(
            '{"dflash_config": {"target_layer_ids": [1, 16, 30, 44, 59]}}\n',
            encoding="utf-8",
        )
        hydra = _write_hydra(
            specforge_checkout,
            "model:\n"
            f"  draft_model_config: {draft.as_posix()}\n"
            "  target_model_path: /models/muse-target\n",
        )
        params = _params(specforge_checkout, tmp_path)
        params.specforge_config = str(hydra)
        params.specforge_overrides = {}
        settings = online_settings(params, env={})
        assert settings["capture_layer_ids"] == (1, 16, 30, 44, 59)
        assert settings["target_model_path"] == "/models/muse-target"

    def test_capture_layer_ids_from_eagle_config(self, specforge_checkout, tmp_path):
        draft = tmp_path / "deepseek-eagle3.json"
        draft.write_text(
            '{"architectures": ["LlamaForCausalLMEagle3"], '
            '"eagle_config": {"eagle_aux_hidden_state_layer_ids": [1, 29, 57]}}\n',
            encoding="utf-8",
        )
        hydra = _write_hydra(specforge_checkout, "model:\n" f"  draft_model_config: {draft.as_posix()}\n")
        params = _params(specforge_checkout, tmp_path)
        params.specforge_config = str(hydra)
        params.specforge_overrides = {}
        assert online_settings(params, env={})["capture_layer_ids"] == (1, 29, 57)

    def test_eagle_without_layer_list_uses_target_depth(self, specforge_checkout, tmp_path):
        target = tmp_path / "target"
        target.mkdir()
        (target / "config.json").write_text(
            '{"text_config": {"num_hidden_layers": 48}, "num_hidden_layers": 1}\n',
            encoding="utf-8",
        )
        draft = tmp_path / "qwen-coder-eagle3.json"
        draft.write_text(
            '{"architectures": ["LlamaForCausalLMEagle3"], "num_hidden_layers": 1}\n',
            encoding="utf-8",
        )
        hydra = _write_hydra(
            specforge_checkout,
            "model:\n"
            f"  draft_model_config: {draft.as_posix()}\n"
            f"  target_model_path: {target.as_posix()}\n",
        )
        params = _params(specforge_checkout, tmp_path)
        params.specforge_config = str(hydra)
        params.specforge_overrides = {}
        assert online_settings(params, env={})["capture_layer_ids"] == (1, 23, 44)
