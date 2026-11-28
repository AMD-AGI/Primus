###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from primus.core.config.preset_loader import PresetLoader


@pytest.fixture
def configs(tmp_path):
    """A primus/configs tree in tmp_path with one preset, models/fw/my_model.yaml (a: 1)."""
    models = tmp_path / "configs" / "models"
    (models / "fw").mkdir(parents=True)
    (models / "fw" / "my_model.yaml").write_text("a: 1\n")
    with patch("primus.core.config.preset_loader.MODELS_ROOT") as root:
        root.__file__ = str(models / "__init__.py")
        yield models / "fw"


def test_preset_path_is_the_file_load_reads(configs):
    want = str(configs / "my_model.yaml")
    assert PresetLoader.preset_path("my_model", "fw") == want
    assert PresetLoader.preset_path("my_model.yaml", "fw") == want
    assert PresetLoader.preset_path("my_model.yml", "fw") == str(configs / "my_model.yml")
    assert PresetLoader.preset_path(Path("my_model"), "fw") == want
    with pytest.raises(ValueError, match="path traversal"):
        PresetLoader.preset_path("../../../../etc/passwd", "fw")
    with patch("primus.core.config.preset_loader.parse_yaml", return_value={"a": 1}) as mock_parse:
        assert PresetLoader.load("my_model", "fw") == {"a": 1}
        mock_parse.assert_called_once_with(want)


class _BytesPath:
    def __fspath__(self):
        return b"my_model"


@pytest.mark.parametrize("name", [None, 3, b"my_model", _BytesPath()])
def test_preset_path_refuses_a_name_that_is_not_text(name):
    with pytest.raises(TypeError, match="preset name must be a str"):
        PresetLoader.preset_path(name, "fw")
    with pytest.raises(TypeError, match="preset name must be a str"):
        PresetLoader.load(name, "fw")


@pytest.mark.parametrize("name", ["", "foo/", ".yaml", "dir/.yaml"])
def test_load_reports_the_path_it_looked_at_for_a_name_without_a_file(configs, name):
    """A malformed name is a missing preset: load's 'not found ... Expected: <path>' error, as on main."""
    path = PresetLoader.preset_path(name, "fw")
    with pytest.raises(FileNotFoundError, match="Expected: " + re.escape(path)):
        PresetLoader.load(name, "fw")


@pytest.mark.parametrize("name", [" my_model", "my_model\n", "my_model.yaml "])
def test_preset_path_keeps_whitespace_as_load_always_did(configs, name):
    """No silent strip: the name is used exactly as given."""
    filename = name if name.endswith(".yaml") else name + ".yaml"
    want = os.path.join(str(configs), filename)
    assert PresetLoader.preset_path(name, "fw") == want
    with pytest.raises(FileNotFoundError, match="Expected: " + re.escape(want)):
        PresetLoader.load(name, "fw")
