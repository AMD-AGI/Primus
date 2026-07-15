# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Tests for data preprocessing config parsing, validation, and authentication.

Tests the YAML-to-CLI mapping in data.py and the auth priority chain in auth.py.
"""

import argparse
import os
import tempfile
from unittest.mock import patch

import pytest

from primus.backends.megatron.data.diffusion.preprocessing.auth import (
    setup_hf_authentication,
)
from primus.cli.subcommands.data import (
    _flatten_preprocessing_config,
    _get_encoded_parser_defaults,
    _load_config_with_cli_overrides,
    _validate_preprocessing_config,
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

    def test_model_defaults(self):
        """Model section injects correct defaults when keys are absent."""
        config = {"model": {}}
        flat = _flatten_preprocessing_config(config)

        assert flat["model_path"] == "black-forest-labs/FLUX.1-dev"
        assert flat["batch_size"] == 8
        assert flat["precision"] == "bf16"
        assert flat["device"] == "cuda"
        assert flat["t5_max_length"] == 512

    def test_image_defaults(self):
        """Image section injects correct defaults when keys are absent."""
        config = {"image": {}}
        flat = _flatten_preprocessing_config(config)

        assert flat["image_size"] == 1024
        assert flat["variable_size"] is False
        assert flat["center_crop"] is True
        assert flat["max_size"] == 1024


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
    """Tests for _load_config_with_cli_overrides merge logic."""

    @patch("primus.core.utils.yaml_utils.parse_yaml")
    def test_cli_overrides_yaml(self, mock_parse_yaml):
        """Explicitly set CLI args override YAML config values."""
        mock_parse_yaml.return_value = {
            "source": {"type": "huggingface", "hf_dataset": "yaml-dataset"},
            "model": {"batch_size": 4},
        }
        defaults = _get_encoded_parser_defaults()
        args = argparse.Namespace(
            config="test.yaml",
            batch_size=16,
            **{k: v for k, v in defaults.items() if k not in ("config", "batch_size")},
        )

        result = _load_config_with_cli_overrides(args)

        assert result.batch_size == 16
        assert result.hf_dataset == "yaml-dataset"

    @patch("primus.core.utils.yaml_utils.parse_yaml")
    def test_yaml_used_when_cli_is_default(self, mock_parse_yaml):
        """YAML values used when CLI arg equals its default."""
        mock_parse_yaml.return_value = {
            "model": {"batch_size": 4},
        }
        defaults = _get_encoded_parser_defaults()
        args = argparse.Namespace(config="test.yaml", **{k: v for k, v in defaults.items() if k != "config"})

        result = _load_config_with_cli_overrides(args)

        assert result.batch_size == 4

    def test_no_config_passthrough(self):
        """No config file returns args unchanged."""
        original = argparse.Namespace(config=None, batch_size=8)
        result = _load_config_with_cli_overrides(original)
        assert result is original


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
