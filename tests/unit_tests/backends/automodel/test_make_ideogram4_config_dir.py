###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the weightless Ideogram-4 config-directory tool.

Needs the diffusers that has Ideogram-4. Everything is built on the meta device,
so nothing is allocated.
"""

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("diffusers.models.transformers.transformer_ideogram4")

TOOL = Path(__file__).resolve().parents[4] / "tools" / "automodel" / "make_ideogram4_config_dir.py"


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("make_ideogram4_config_dir", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_it_writes_a_config_that_reloads(tool, tmp_path):
    out = tmp_path / "cfg"
    assert tool.main(["--out", str(out), "--overrides", '{"num_layers": 2}']) == 0
    config = json.loads((out / "transformer" / "config.json").read_text())
    assert config["num_layers"] == 2


def test_a_bad_override_fails_without_writing_or_allocating(tool, tmp_path):
    """No fallback to a real allocation, and no empty directory left behind."""
    out = tmp_path / "cfg"
    with pytest.raises(TypeError):
        tool.main(["--out", str(out), "--overrides", '{"not_a_config_key": 1}'])
    assert not out.exists()


def test_the_overrides_must_be_a_json_object(tool, tmp_path):
    assert tool.main(["--out", str(tmp_path / "cfg"), "--overrides", "[1]"]) == 2
