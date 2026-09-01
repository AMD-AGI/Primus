# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Tests for data preprocessing config parsing, validation, and authentication.

Tests the Megatron preprocessing YAML-to-CLI mapping, public CLI compatibility,
and the authentication priority chain.
"""

import argparse
import os
import tempfile
from unittest.mock import patch

import pytest

from primus.backends.megatron.data.diffusion.preprocessing.auth import (
    setup_hf_authentication,
)
from primus.backends.megatron.data.diffusion.preprocessing.config import (
    CONFIG_PATH_BY_DEST as _CONFIG_PATH_BY_DEST,
)
from primus.backends.megatron.data.diffusion.preprocessing.config import (
    ENCODED_CONFIG_DEFAULTS as _ENCODED_CONFIG_DEFAULTS,
)
from primus.backends.megatron.data.diffusion.preprocessing.config import (
    flatten_preprocessing_config as _flatten_preprocessing_config,
)
from primus.backends.megatron.data.diffusion.preprocessing.config import (
    get_encoded_config_defaults as _get_encoded_parser_defaults,
)
from primus.cli.subcommands.data import (
    _load_config_with_cli_overrides,
    _validate_preprocessing_config,
    register_subcommand,
)
from tests.utils import PrimusUT


class TestFlattenPreprocessingConfig(PrimusUT):
    """Tests for _flatten_preprocessing_config YAML-to-flat-dict mapping."""

    def test_all_sections_mapped(self):
        """All 6 YAML sections produce correct flat keys."""
        config = {
            "source": {
                "type": "huggingface",
                "hf_dataset": "diffusers/pokemon",
                "hf_split": "train",
            },
            "data_format": {
                "image_key": "jpg",
                "caption_key": "json.caption",
            },
            "output": {
                "output_dir": "/data/out",
                "shard_size": 500,
            },
            "model": {
                "model_path": "my-model",
                "batch_size": 4,
                "precision": "fp16",
            },
            "image": {
                "image_size": 512,
                "variable_size": True,
                "center_crop": False,
            },
            "auth": {
                "hf_token_file": "/path/to/token",
            },
        }
        flat = _flatten_preprocessing_config(config)

        assert flat["source_type"] == "huggingface"
        assert flat["hf_dataset"] == "diffusers/pokemon"
        assert flat["image_key"] == "jpg"
        assert flat["caption_key"] == "json.caption"
        assert flat["output_dir"] == "/data/out"
        assert flat["shard_size"] == 500
        assert flat["model_path"] == "my-model"
        assert flat["batch_size"] == 4
        assert flat["precision"] == "fp16"
        assert flat["image_size"] == 512
        assert flat["variable_size"] is True
        assert flat["center_crop"] is False
        assert flat["hf_token_file"] == "/path/to/token"

    def test_partial_config(self):
        """Partial config with only source + output produces only those keys."""
        config = {
            "source": {"type": "directory", "input_dir": "/data/images"},
            "output": {"output_dir": "/data/out"},
        }
        flat = _flatten_preprocessing_config(config)

        assert flat["source_type"] == "directory"
        assert flat["input_dir"] == "/data/images"
        assert flat["output_dir"] == "/data/out"
        assert "model_path" not in flat
        assert "image_size" not in flat

    def test_flatten_does_not_inject_defaults(self):
        """Flattening is structural; defaults are applied by the shared merge path."""
        assert _flatten_preprocessing_config({"model": {}, "image": {}}) == {}


class TestValidatePreprocessingConfig(PrimusUT):
    """Tests for _validate_preprocessing_config."""

    def _make_args(self, **kwargs):
        defaults = {
            "source_type": "huggingface",
            "output_dir": "/data/out",
            "hf_dataset": "test/dataset",
            "input_dir": None,
            "input_path": None,
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_missing_source_type_raises(self):
        """Missing source_type raises ValueError."""
        args = self._make_args(source_type=None)
        with self.assertRaises(ValueError, msg="source_type"):
            _validate_preprocessing_config(args)

    def test_missing_output_dir_raises(self):
        """Missing output_dir raises ValueError."""
        args = self._make_args(output_dir=None)
        with self.assertRaises(ValueError, msg="output_dir"):
            _validate_preprocessing_config(args)

    def test_huggingface_without_hf_dataset_raises(self):
        """HuggingFace source without hf_dataset raises ValueError."""
        args = self._make_args(source_type="huggingface", hf_dataset=None)
        with self.assertRaises(ValueError, msg="hf-dataset"):
            _validate_preprocessing_config(args)

    def test_directory_without_input_dir_raises(self):
        """Directory source without input_dir raises ValueError."""
        args = self._make_args(source_type="directory", input_dir=None)
        with self.assertRaises(ValueError, msg="input-dir"):
            _validate_preprocessing_config(args)

    def test_valid_config_passes(self):
        """Valid config raises no errors."""
        args = self._make_args()
        _validate_preprocessing_config(args)


class TestLoadConfigWithCliOverrides(PrimusUT):
    """Tests for _load_config_with_cli_overrides merge logic.

    These drive the real parser rather than hand-building a namespace: only the
    parser can express the difference between "flag omitted" and "flag typed
    with a value that happens to equal the default", which is precisely the
    distinction the merge depends on.
    """

    def _merge(self, *argv):
        parser = argparse.ArgumentParser(prog="primus")
        register_subcommand(parser.add_subparsers(dest="command"))
        args = parser.parse_args(["data", "diffusion-encoded", *argv])
        return _load_config_with_cli_overrides(args)

    @patch("primus.core.utils.yaml_utils.parse_yaml")
    def test_cli_overrides_yaml(self, mock_parse_yaml):
        """Explicitly set CLI args override YAML config values."""
        mock_parse_yaml.return_value = {
            "source": {"type": "huggingface", "hf_dataset": "yaml-dataset"},
            "model": {"batch_size": 4},
        }

        result = self._merge("--config", "test.yaml", "--batch-size", "16")

        assert result.batch_size == 16
        assert result.hf_dataset == "yaml-dataset"

    @patch("primus.core.utils.yaml_utils.parse_yaml")
    def test_yaml_used_when_flag_not_passed(self, mock_parse_yaml):
        """YAML values are used for options the user did not pass."""
        mock_parse_yaml.return_value = {"model": {"batch_size": 4}}

        result = self._merge("--config", "test.yaml")

        assert result.batch_size == 4

    @patch("primus.core.utils.yaml_utils.parse_yaml")
    def test_partial_yaml_section_is_deep_merged_with_defaults(self, mock_parse_yaml):
        """A partial nested section inherits the remaining canonical defaults."""
        mock_parse_yaml.return_value = {"model": {"batch_size": 4}, "image": {"image_size": 512}}

        result = self._merge("--config", "test.yaml")

        assert result.batch_size == 4
        assert result.precision == "bf16"
        assert result.t5_max_length == 512
        assert result.image_size == 512
        assert result.variable_size is False
        assert result.center_crop is True

    @patch("primus.core.utils.yaml_utils.parse_yaml")
    def test_explicit_cli_wins_when_value_equals_default(self, mock_parse_yaml):
        """A flag typed with its default value still overrides the YAML.

        Regression test: the merge used to infer "was this typed?" by comparing
        against the default table, so passing the default value looked identical
        to passing nothing and the YAML silently won.
        """
        mock_parse_yaml.return_value = {
            "model": {"batch_size": 64, "precision": "fp32", "t5_max_length": 256},
            "image": {"image_size": 512},
        }

        result = self._merge(
            "--config",
            "test.yaml",
            "--batch-size",
            "8",  # 8 is also the parser default
            "--precision",
            "bf16",  # bf16 is also the parser default
            "--t5-max-length",
            "512",  # 512 is also the parser default
            "--image-size",
            "1024",  # 1024 is also the parser default
        )

        assert result.batch_size == 8
        assert result.precision == "bf16"
        assert result.t5_max_length == 512
        assert result.image_size == 1024

    @patch("primus.core.utils.yaml_utils.parse_yaml")
    def test_omitted_yaml_section_falls_back_to_defaults(self, mock_parse_yaml):
        """Options in no YAML section and not typed still resolve to defaults.

        Regression test: the merged namespace was built from the YAML plus the
        flags believed to be explicit, so a key absent from both simply vanished
        and _prepare_encoded raised AttributeError on it.
        """
        mock_parse_yaml.return_value = {
            "source": {"type": "huggingface", "hf_dataset": "yaml-dataset"},
            "output": {"output_dir": "/data/out"},
        }

        result = self._merge("--config", "test.yaml")

        # The optional image: section is absent, so these come from the defaults.
        assert result.image_size == 1024
        assert result.variable_size is False
        assert result.center_crop is True
        assert result.max_size == 1024

    @patch("primus.core.utils.yaml_utils.parse_yaml")
    def test_every_configurable_key_present(self, mock_parse_yaml):
        """The merged namespace exposes every key the pipeline may read."""
        mock_parse_yaml.return_value = {"source": {"type": "huggingface"}}

        result = self._merge("--config", "test.yaml")

        missing = [key for key in _get_encoded_parser_defaults() if not hasattr(result, key)]
        assert missing == []

    def test_no_config_applies_defaults(self):
        """Without a config file, defaults are applied and CLI args preserved."""
        result = self._merge("--batch-size", "16")

        assert result.batch_size == 16
        assert result.image_size == 1024
        assert result.precision == "bf16"


class TestParserDefaults(PrimusUT):
    """The raw parser keeps argparse defaults; the encoded parser reads them from
    _get_encoded_parser_defaults(). Guard the two against drifting apart."""

    def test_raw_parser_defaults_match_table(self):
        parser = argparse.ArgumentParser(prog="primus")
        register_subcommand(parser.add_subparsers(dest="command"))
        raw = vars(parser.parse_args(["data", "diffusion-raw"]))
        table = _get_encoded_parser_defaults()

        mismatched = {
            key: (value, table[key]) for key, value in raw.items() if key in table and value != table[key]
        }
        assert mismatched == {}

    def test_default_schema_and_flat_mapping_cover_the_same_fields(self):
        """Every canonical default has exactly one YAML-to-CLI mapping."""
        default_paths = {
            (section, key) for section, values in _ENCODED_CONFIG_DEFAULTS.items() for key in values
        }

        assert set(_CONFIG_PATH_BY_DEST.values()) == default_paths
        assert set(_CONFIG_PATH_BY_DEST) == set(_get_encoded_parser_defaults())

    def test_encoded_parser_suppresses_configurable_defaults(self):
        """Only explicitly supplied configurable options enter the namespace."""
        parser = argparse.ArgumentParser(prog="primus")
        register_subcommand(parser.add_subparsers(dest="command"))

        parsed = vars(parser.parse_args(["data", "diffusion-encoded"]))

        assert set(parsed).isdisjoint(_get_encoded_parser_defaults())

    def test_compatibility_entrypoint_preserves_public_commands(self):
        """The thin compatibility entry point preserves all public command names."""
        cases = [
            ("diffusion-raw", []),
            ("diffusion-encoded", []),
            ("diffusion-ingest", ["--config", "ingest.yaml"]),
        ]
        for command, extra_args in cases:
            with self.subTest(command=command):
                parser = argparse.ArgumentParser(prog="primus")
                register_subcommand(parser.add_subparsers(dest="command"))

                parsed = parser.parse_args(["data", command, *extra_args])

                assert parsed.data_command == command
                assert callable(parsed.func)


class TestSetupHfAuthenticationPriority(PrimusUT):
    """Tests for setup_hf_authentication priority chain."""

    def test_file_takes_priority_over_env(self):
        """Token file takes priority over HF_TOKEN env var."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".token", delete=False) as f:
            f.write("hf_file_token_123")
            token_path = f.name
        try:
            os.chmod(token_path, 0o600)
            old_env = os.environ.get("HF_TOKEN")
            os.environ["HF_TOKEN"] = "hf_env_token_456"
            try:
                token = setup_hf_authentication(token_file=token_path, use_env=True)
                assert token == "hf_file_token_123"
            finally:
                if old_env is None:
                    os.environ.pop("HF_TOKEN", None)
                else:
                    os.environ["HF_TOKEN"] = old_env
        finally:
            os.unlink(token_path)

    def test_env_takes_priority_over_cache(self):
        """HF_TOKEN env var is used when no file is provided."""
        old_env = os.environ.get("HF_TOKEN")
        os.environ["HF_TOKEN"] = "hf_env_token_789"
        try:
            token = setup_hf_authentication(token_file=None, use_env=True)
            assert token == "hf_env_token_789"
        finally:
            if old_env is None:
                os.environ.pop("HF_TOKEN", None)
            else:
                os.environ["HF_TOKEN"] = old_env

    def test_no_auth_returns_none(self):
        """No auth sources returns None."""
        old_env = os.environ.pop("HF_TOKEN", None)
        try:
            with patch.object(
                type(__import__("pathlib").Path()),
                "exists",
                return_value=False,
            ):
                token = setup_hf_authentication(token_file=None, use_env=True)
                assert token is None
        finally:
            if old_env is not None:
                os.environ["HF_TOKEN"] = old_env


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
