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
from primus.backends.backend_aliases import resolve_backend_alias
from primus.cli.subcommands import train
from primus.core.backend.backend_registry import BackendRegistry
from primus.core.config.primus_config import load_primus_config
from primus.core.utils import file_utils

ROOT = Path(__file__).resolve().parents[4]
EXAMPLE = ROOT / "examples/automodel/configs/MI355X/diffusion/wan2_2_t2v_a14b-pretrain.yaml"


def test_old_framework_name_resolves_with_a_warning():
    with pytest.warns(FutureWarning, match="framework: automodel"):
        assert resolve_backend_alias("nemo_automodel") == "automodel"


def test_current_framework_names_pass_through_silently():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for name in ("automodel", "megatron", "torchtitan"):
            assert resolve_backend_alias(name) == name


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


def forget_the_old_package(monkeypatch):
    for name in [m for m in sys.modules if m.startswith("primus.backends.nemo_automodel")]:
        monkeypatch.delitem(sys.modules, name)


def test_old_import_paths_reexport_the_renamed_classes(monkeypatch):
    forget_the_old_package(monkeypatch)

    with pytest.warns(FutureWarning, match="primus.backends.automodel"):
        shim = importlib.import_module("primus.backends.nemo_automodel")
    adapter = importlib.import_module("primus.backends.nemo_automodel.nemo_automodel_adapter")
    trainer = importlib.import_module("primus.backends.nemo_automodel.nemo_automodel_pretrain_trainer")
    builder = importlib.import_module("primus.backends.nemo_automodel.argument_builder")

    assert shim.NemoAutomodelAdapter is adapter.NemoAutomodelAdapter is AutoModelAdapter
    assert (
        shim.NemoAutomodelPretrainTrainer is trainer.NemoAutomodelPretrainTrainer is AutoModelPretrainTrainer
    )
    assert builder.NemoAutomodelConfigBuilder is AutoModelConfigBuilder
    assert "AutoModelConfigBuilder" in builder.__all__


def test_the_old_name_is_still_registered(monkeypatch):
    """For code that looks the backend up in the registry directly, as before the rename."""
    forget_the_old_package(monkeypatch)
    monkeypatch.setattr(
        BackendRegistry,
        "_adapters",
        {k: v for k, v in BackendRegistry._adapters.items() if k != "nemo_automodel"},
    )
    monkeypatch.setattr(
        BackendRegistry,
        "_trainer_classes",
        {k: v for k, v in BackendRegistry._trainer_classes.items() if k[0] != "nemo_automodel"},
    )

    with pytest.warns(FutureWarning):
        adapter = BackendRegistry.get_adapter("nemo_automodel")

    assert isinstance(adapter, AutoModelAdapter)
    assert BackendRegistry.get_trainer_class("nemo_automodel") is AutoModelPretrainTrainer


def test_projection_reads_the_old_framework_name_as_the_new_one(tmp_path):
    from types import SimpleNamespace

    from primus.cli.subcommands import projection

    exp = tmp_path / "exp.yaml"
    exp.write_text("modules:\n  pre_trainer:\n    framework: nemo_automodel\n", encoding="utf-8")

    with pytest.warns(FutureWarning):
        assert projection._experiment_framework(SimpleNamespace(config=str(exp))) == "automodel"
