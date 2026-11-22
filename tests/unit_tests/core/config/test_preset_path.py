###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from primus.core.config.preset_loader import PresetLoader


@patch("primus.core.config.preset_loader.MODELS_ROOT")
def test_preset_path_is_the_file_load_reads(mock_models_root):
    mock_models_root.__file__ = os.path.join(os.sep, "mock", "primus", "configs", "models", "__init__.py")
    want = os.path.join(os.sep, "mock", "primus", "configs", "models", "fw", "my_model.yaml")
    assert PresetLoader.preset_path("my_model", "fw") == want
    assert PresetLoader.preset_path("my_model.yaml", "fw") == want
    assert PresetLoader.preset_path("my_model.yml", "fw") == want[: -len("yaml")] + "yml"
    assert PresetLoader.preset_path(Path("my_model"), "fw") == want
    with pytest.raises(ValueError, match="path traversal"):
        PresetLoader.preset_path("../../../../etc/passwd", "fw")
    with patch("os.path.exists", return_value=True) as mock_exists, patch(
        "primus.core.config.preset_loader.parse_yaml", return_value={}
    ) as mock_parse:
        PresetLoader.load("my_model", "fw")
        mock_exists.assert_any_call(want)
        mock_parse.assert_called_once_with(want)
