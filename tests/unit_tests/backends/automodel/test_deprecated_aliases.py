###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The backend was first published as ``nemo_automodel``. Configs that still say
``framework: nemo_automodel`` and code importing the original module paths must
keep working, with a warning, until the aliases are removed.
"""

import argparse
import importlib
import sys
import warnings
from pathlib import Path

import pytest

from primus.backends.automodel import AutoModelAdapter, AutoModelPretrainTrainer
from primus.backends.automodel.argument_builder import AutoModelConfigBuilder
from primus.cli.subcommands import train
from primus.core.config.primus_config import load_primus_config
from primus.core.launcher.parser import resolve_framework_alias
from primus.core.utils import file_utils

ROOT = Path(__file__).resolve().parents[4]
EXAMPLE = ROOT / "examples/automodel/configs/MI355X/diffusion/wan2_2_t2v_a14b-pretrain.yaml"


def test_old_framework_name_resolves_with_a_warning():
    with pytest.warns(FutureWarning, match="framework: automodel"):
        assert resolve_framework_alias("nemo_automodel") == "automodel"


def test_current_framework_names_pass_through_silently():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for name in ("automodel", "megatron", "torchtitan"):
            assert resolve_framework_alias(name) == name


@pytest.mark.parametrize("name", ["nemo_automodel", "automodel"])
def test_train_pretrain_takes_backend_as_its_own_option(name):
    """Not as an abbreviation of --backend_path, and not as a config override."""
    parser = argparse.ArgumentParser()
    train.register_subcommand(parser.add_subparsers(dest="command"))

    args, unknown = parser.parse_known_args(["train", "pretrain", "--config", "exp.yaml", "--backend", name])

    assert args.backend == name
    assert args.backend_path is None
    assert unknown == []


def test_old_style_config_loads_the_automodel_presets(tmp_path, monkeypatch):
    monkeypatch.setattr(file_utils, "create_path_if_not_exists", lambda *args, **kwargs: None)
    text = EXAMPLE.read_text(encoding="utf-8")
    assert "framework: automodel" in text
    old_style = tmp_path / "old_style.yaml"
    old_style.write_text(text.replace("framework: automodel", "framework: nemo_automodel"), encoding="utf-8")

    with pytest.warns(FutureWarning):
        cfg = load_primus_config(old_style, None)

    (module,) = cfg.modules
    assert module.framework == "automodel"
    assert module.params


def test_old_import_paths_reexport_the_renamed_classes(monkeypatch):
    for name in [m for m in sys.modules if m.startswith("primus.backends.nemo_automodel")]:
        monkeypatch.delitem(sys.modules, name)

    with pytest.warns(DeprecationWarning, match="primus.backends.automodel"):
        shim = importlib.import_module("primus.backends.nemo_automodel")
    adapter = importlib.import_module("primus.backends.nemo_automodel.nemo_automodel_adapter")
    trainer = importlib.import_module("primus.backends.nemo_automodel.nemo_automodel_pretrain_trainer")
    builder = importlib.import_module("primus.backends.nemo_automodel.argument_builder")

    assert shim.NemoAutomodelAdapter is adapter.NemoAutomodelAdapter is AutoModelAdapter
    assert (
        shim.NemoAutomodelPretrainTrainer is trainer.NemoAutomodelPretrainTrainer is AutoModelPretrainTrainer
    )
    assert builder.NemoAutomodelConfigBuilder is AutoModelConfigBuilder
