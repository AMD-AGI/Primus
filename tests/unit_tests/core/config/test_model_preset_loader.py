###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for ModelPresetLoader.
"""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from primus.core.config.model_preset_loader import ModelPresetLoader


class TestModelPresetLoader:
    """Test ModelPresetLoader functionality."""

    @pytest.fixture
    def temp_models_dir(self, monkeypatch):
        """Create a temporary models directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create framework subdirectories
            megatron_dir = Path(tmpdir) / "megatron"
            torchtitan_dir = Path(tmpdir) / "torchtitan"
            megatron_dir.mkdir()
            torchtitan_dir.mkdir()

            # Create sample model preset files
            llama2_7b_config = {
                "num_layers": 32,
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "seq_length": 4096,
                "batch_size": 16,
            }
            with open(megatron_dir / "llama2_7B.yaml", "w") as f:
                yaml.dump(llama2_7b_config, f)

            llama3_8b_config = {
                "num_layers": 32,
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "seq_length": 8192,
                "batch_size": 32,
            }
            with open(megatron_dir / "llama3_8B.yaml", "w") as f:
                yaml.dump(llama3_8b_config, f)

            titan_config = {
                "model_type": "llama",
                "dim": 4096,
                "n_layers": 32,
            }
            with open(torchtitan_dir / "llama_base.yaml", "w") as f:
                yaml.dump(titan_config, f)

            # Mock the MODELS_ROOT module
            class MockModelsRoot:
                __file__ = os.path.join(tmpdir, "__init__.py")

            monkeypatch.setattr("primus.core.config.model_preset_loader.MODELS_ROOT", MockModelsRoot())

            yield tmpdir

    def test_load_model_preset_success(self, temp_models_dir):
        """Test successfully loading a model preset."""
        preset = ModelPresetLoader.load("llama2_7B", framework="megatron")

        assert preset["num_layers"] == 32
        assert preset["hidden_size"] == 4096
        assert preset["num_attention_heads"] == 32
        assert preset["seq_length"] == 4096
        assert preset["batch_size"] == 16

    def test_load_different_framework(self, temp_models_dir):
        """Test loading model from different framework."""
        preset = ModelPresetLoader.load("llama_base", framework="torchtitan")

        assert preset["model_type"] == "llama"
        assert preset["dim"] == 4096
        assert preset["n_layers"] == 32

    def test_load_model_not_found(self, temp_models_dir):
        """Test loading non-existent model raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Model preset 'non_existent' not found"):
            ModelPresetLoader.load("non_existent", framework="megatron")

    def test_load_framework_not_found(self, temp_models_dir):
        """Test loading from non-existent framework raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found for framework 'unknown_framework'"):
            ModelPresetLoader.load("llama2_7B", framework="unknown_framework")

    def test_load_multiple_models_same_framework(self, temp_models_dir):
        """Test loading multiple models from the same framework."""
        preset1 = ModelPresetLoader.load("llama2_7B", framework="megatron")
        preset2 = ModelPresetLoader.load("llama3_8B", framework="megatron")

        assert preset1["seq_length"] == 4096
        assert preset2["seq_length"] == 8192
        assert preset1["batch_size"] == 16
        assert preset2["batch_size"] == 32


class TestModelPresetLoaderWithExtends:
    """Test ModelPresetLoader with extends functionality."""

    @pytest.fixture
    def temp_models_dir_with_extends(self, monkeypatch):
        """Create a temporary models directory with extends support."""
        with tempfile.TemporaryDirectory() as tmpdir:
            megatron_dir = Path(tmpdir) / "megatron"
            megatron_dir.mkdir()

            # Create base config
            base_config = {
                "num_layers": 32,
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "batch_size": 16,
                "optimizer": {
                    "type": "adam",
                    "lr": 1e-4,
                    "weight_decay": 0.1,
                },
            }
            with open(megatron_dir / "base.yaml", "w") as f:
                yaml.dump(base_config, f)

            # Create derived config with extends
            llama_config = {
                "extends": ["base.yaml"],
                "seq_length": 4096,
                "batch_size": 32,  # Override base
                "optimizer": {
                    "lr": 3e-4,  # Override only lr
                },
            }
            with open(megatron_dir / "llama_custom.yaml", "w") as f:
                yaml.dump(llama_config, f)

            # Mock the MODELS_ROOT module
            class MockModelsRoot:
                __file__ = os.path.join(tmpdir, "__init__.py")

            monkeypatch.setattr("primus.core.config.model_preset_loader.MODELS_ROOT", MockModelsRoot())

            yield tmpdir

    def test_load_with_extends(self, temp_models_dir_with_extends):
        """Test loading model preset with extends."""
        preset = ModelPresetLoader.load("llama_custom", framework="megatron")

        # From base
        assert preset["num_layers"] == 32
        assert preset["hidden_size"] == 4096
        assert preset["num_attention_heads"] == 32

        # Override from derived
        assert preset["batch_size"] == 32

        # New field from derived
        assert preset["seq_length"] == 4096

        # Deep merge optimizer
        assert preset["optimizer"]["type"] == "adam"  # From base
        assert preset["optimizer"]["lr"] == 3e-4  # Override from derived
        assert preset["optimizer"]["weight_decay"] == 0.1  # From base

    def test_load_with_nested_extends(self, temp_models_dir_with_extends):
        """Test loading with multiple levels of extends."""
        megatron_dir = Path(temp_models_dir_with_extends) / "megatron"

        # Create a third level
        advanced_config = {
            "extends": ["llama_custom.yaml"],
            "num_layers": 40,  # Override
            "use_flash_attn": True,  # New field
        }
        with open(megatron_dir / "llama_advanced.yaml", "w") as f:
            yaml.dump(advanced_config, f)

        preset = ModelPresetLoader.load("llama_advanced", framework="megatron")

        # From base (through llama_custom)
        assert preset["hidden_size"] == 4096

        # From llama_custom
        assert preset["seq_length"] == 4096
        assert preset["batch_size"] == 32

        # Override in advanced
        assert preset["num_layers"] == 40

        # New in advanced
        assert preset["use_flash_attn"] is True


class TestModelPresetLoaderMergeWithUserParams:
    """Test merge_with_user_params functionality."""

    def test_merge_empty_user_params(self):
        """Test merging with empty user params."""
        preset = {"num_layers": 32, "hidden_size": 4096}
        user_params = {}

        result = ModelPresetLoader.merge_with_user_params(preset, user_params)

        assert result == preset
        assert result["num_layers"] == 32
        assert result["hidden_size"] == 4096

    def test_merge_override_params(self):
        """Test merging with override params."""
        preset = {"num_layers": 32, "hidden_size": 4096, "batch_size": 16}
        user_params = {"batch_size": 64, "lr": 1e-3}

        result = ModelPresetLoader.merge_with_user_params(preset, user_params)

        # From preset
        assert result["num_layers"] == 32
        assert result["hidden_size"] == 4096

        # Override from user
        assert result["batch_size"] == 64

        # New from user
        assert result["lr"] == 1e-3

    def test_merge_nested_dict_override(self):
        """Test merging with nested dictionary override."""
        preset = {
            "model": {"num_layers": 32, "hidden_size": 4096},
            "optimizer": {"type": "adam", "lr": 1e-4, "weight_decay": 0.1},
        }
        user_params = {
            "optimizer": {"lr": 3e-4, "beta1": 0.9},  # Override lr, add beta1
            "scheduler": {"type": "cosine"},  # New top-level key
        }

        result = ModelPresetLoader.merge_with_user_params(preset, user_params)

        # Model unchanged
        assert result["model"]["num_layers"] == 32
        assert result["model"]["hidden_size"] == 4096

        # Optimizer deep merged
        assert result["optimizer"]["type"] == "adam"  # From preset
        assert result["optimizer"]["lr"] == 3e-4  # Override from user
        assert result["optimizer"]["weight_decay"] == 0.1  # From preset
        assert result["optimizer"]["beta1"] == 0.9  # New from user

        # New scheduler
        assert result["scheduler"]["type"] == "cosine"

    def test_merge_list_override(self):
        """Test that lists are replaced, not merged."""
        preset = {
            "layers": [1, 2, 3],
            "config": {"values": [10, 20, 30]},
        }
        user_params = {
            "layers": [4, 5],
            "config": {"values": [40, 50]},
        }

        result = ModelPresetLoader.merge_with_user_params(preset, user_params)

        # Lists should be replaced entirely
        assert result["layers"] == [4, 5]
        assert result["config"]["values"] == [40, 50]

    def test_merge_preserves_original(self):
        """Test that merge doesn't modify original dicts."""
        preset = {"num_layers": 32, "batch_size": 16}
        user_params = {"batch_size": 64, "lr": 1e-3}

        original_preset = preset.copy()
        original_user = user_params.copy()

        result = ModelPresetLoader.merge_with_user_params(preset, user_params)

        # Original dicts should be unchanged
        assert preset == original_preset
        assert user_params == original_user

        # Result should have merged values
        assert result["batch_size"] == 64
        assert result["lr"] == 1e-3


class TestModelPresetLoaderIntegration:
    """Integration tests for ModelPresetLoader."""

    @pytest.fixture
    def complex_models_dir(self, monkeypatch):
        """Create a complex models directory for integration testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            megatron_dir = Path(tmpdir) / "megatron"
            megatron_dir.mkdir()

            # Base transformer config
            base_transformer = {
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "dropout": 0.1,
                "optimizer": {"type": "adam", "lr": 1e-4},
            }
            with open(megatron_dir / "transformer_base.yaml", "w") as f:
                yaml.dump(base_transformer, f)

            # LLaMA specific extensions
            llama_base = {
                "extends": ["transformer_base.yaml"],
                "num_layers": 32,
                "vocab_size": 32000,
                "optimizer": {"lr": 3e-4},  # Override lr
            }
            with open(megatron_dir / "llama_base.yaml", "w") as f:
                yaml.dump(llama_base, f)

            # LLaMA 7B variant
            llama_7b = {
                "extends": ["llama_base.yaml"],
                "hidden_size": 4096,
                "seq_length": 4096,
            }
            with open(megatron_dir / "llama2_7B.yaml", "w") as f:
                yaml.dump(llama_7b, f)

            # Mock the MODELS_ROOT module
            class MockModelsRoot:
                __file__ = os.path.join(tmpdir, "__init__.py")

            monkeypatch.setattr("primus.core.config.model_preset_loader.MODELS_ROOT", MockModelsRoot())

            yield tmpdir

    def test_full_workflow_load_and_merge(self, complex_models_dir):
        """Test full workflow: load preset with extends + merge user params."""
        # Load model preset (with multi-level extends)
        preset = ModelPresetLoader.load("llama2_7B", framework="megatron")

        # Verify preset loaded correctly from all levels
        assert preset["hidden_size"] == 4096  # From transformer_base, overridden in llama_7b
        assert preset["num_attention_heads"] == 32  # From transformer_base
        assert preset["num_layers"] == 32  # From llama_base
        assert preset["vocab_size"] == 32000  # From llama_base
        assert preset["seq_length"] == 4096  # From llama_7b
        assert preset["optimizer"]["lr"] == 3e-4  # Override in llama_base

        # Merge with user params
        user_params = {
            "batch_size": 64,
            "seq_length": 8192,  # Override
            "optimizer": {"lr": 1e-5, "beta1": 0.9},  # Override lr, add beta1
        }

        final_config = ModelPresetLoader.merge_with_user_params(preset, user_params)

        # Check final merged config
        assert final_config["hidden_size"] == 4096  # From preset
        assert final_config["num_layers"] == 32  # From preset
        assert final_config["batch_size"] == 64  # From user
        assert final_config["seq_length"] == 8192  # User override
        assert final_config["optimizer"]["type"] == "adam"  # From preset
        assert final_config["optimizer"]["lr"] == 1e-5  # User override
        assert final_config["optimizer"]["beta1"] == 0.9  # From user


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
