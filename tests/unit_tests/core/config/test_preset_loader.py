from unittest.mock import patch

import pytest

from primus.core.config.preset_loader import PresetLoader


class TestPresetLoader:

    @patch("primus.core.config.preset_loader.parse_yaml")
    @patch("primus.core.config.preset_loader.MODELS_ROOT")
    def test_load_success(self, mock_models_root, mock_parse_yaml):
        # Mock file structure path
        mock_models_root.__file__ = "/mock/primus/configs/models/__init__.py"

        # Mock parse_yaml return
        expected_config = {"model": "test"}
        mock_parse_yaml.return_value = expected_config

        # We also need to mock os.path.exists to return True
        with patch("os.path.exists", return_value=True) as mock_exists:
            result = PresetLoader.load("my_model", "my_framework")

            assert result == expected_config

            # Verify path construction
            expected_path = "/mock/primus/configs/models/my_framework/my_model.yaml"
            mock_exists.assert_called_with(expected_path)
            mock_parse_yaml.assert_called_with(expected_path)

    @patch("primus.core.config.preset_loader.MODELS_ROOT")
    def test_load_not_found(self, mock_models_root):
        mock_models_root.__file__ = "/mock/primus/configs/models/__init__.py"

        with patch("os.path.exists", return_value=False):
            with pytest.raises(FileNotFoundError, match="Preset 'missing' not found"):
                PresetLoader.load("missing", "framework")

    def test_merge_with_user_params(self):
        preset = {"a": 1, "nested": {"x": 10}}
        params = {"b": 2, "nested": {"x": 20}}

        result = PresetLoader.merge_with_user_params(preset, params)

        assert result["a"] == 1
        assert result["b"] == 2
        assert result["nested"]["x"] == 20
