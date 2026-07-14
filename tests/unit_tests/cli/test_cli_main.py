###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse

import pytest

import primus.cli.main as cli_main


def test_iter_subcommand_modules_includes_builtin():
    modules = set(cli_main._iter_subcommand_modules())

    assert "primus.cli.subcommands.train" in modules
    assert "primus.cli.subcommands.benchmark" in modules
    assert "primus.cli.subcommands.projection" in modules


def test_load_subcommands_invokes_register(monkeypatch):
    captured = []
    module_paths = ["x.alpha", "x.beta"]

    def fake_import(name):
        parser = argparse.ArgumentParser()

        def register(subparsers):
            captured.append((name, subparsers))
            parser.set_defaults(func=lambda *_args, **_kwargs: None)
            return parser

        module = type("FakeModule", (), {})()
        module.register_subcommand = register
        return module

    monkeypatch.setattr(cli_main.importlib, "import_module", fake_import)

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd")

    cli_main._load_subcommands(subparsers, module_paths)

    assert captured == [("x.alpha", subparsers), ("x.beta", subparsers)]


def test_load_subcommands_requires_func(monkeypatch):
    module_paths = ["x.alpha"]

    def fake_import(name):
        parser = argparse.ArgumentParser()
        parser.add_subparsers(dest="suite")

        def register(subparsers):
            return parser

        module = type("FakeModule", (), {})()
        module.register_subcommand = register
        return module

    monkeypatch.setattr(cli_main.importlib, "import_module", fake_import)

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd")

    with pytest.raises(RuntimeError, match="set_defaults"):
        cli_main._load_subcommands(subparsers, module_paths)


# ─────────────────────────────────────────────────────────────────────────────
# _discover_subcommands
# ─────────────────────────────────────────────────────────────────────────────


def test_discover_subcommands_maps_name_to_module_path():
    cmds = cli_main._discover_subcommands()
    assert cmds["train"] == "primus.cli.subcommands.train"
    assert "projection" in cmds and "benchmark" in cmds


# ─────────────────────────────────────────────────────────────────────────────
# _extract_command (argv -> subcommand name)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "argv, expected",
    [
        (["train", "--config", "x"], "train"),  # first positional
        (["--debug", "train"], "train"),  # skip leading flags
        (["--", "projection"], "projection"),  # token after --
        (["--"], None),  # bare --
        (["bogus"], "bogus"),  # unknown token returned verbatim
        ([], None),  # empty
        (["--debug"], None),  # all flags
    ],
)
def test_extract_command(argv, expected):
    assert cli_main._extract_command(argv, {"train", "projection"}) == expected


# ─────────────────────────────────────────────────────────────────────────────
# _register_subcommand error paths
# ─────────────────────────────────────────────────────────────────────────────


def _subparsers():
    parser = argparse.ArgumentParser()
    return parser.add_subparsers(dest="cmd")


def test_register_subcommand_import_failure_raises(monkeypatch):
    def boom(name):
        raise ImportError("nope")

    monkeypatch.setattr(cli_main.importlib, "import_module", boom)
    with pytest.raises(RuntimeError, match="Failed to import"):
        cli_main._register_subcommand(_subparsers(), "x.bad")


def test_register_subcommand_missing_hook_raises(monkeypatch):
    monkeypatch.setattr(cli_main.importlib, "import_module", lambda name: type("M", (), {})())
    with pytest.raises(AttributeError, match="register_subcommand"):
        cli_main._register_subcommand(_subparsers(), "x.nohook")


def test_register_subcommand_returning_none_raises(monkeypatch):
    def fake(name):
        m = type("M", (), {})()
        m.register_subcommand = lambda subparsers: None
        return m

    monkeypatch.setattr(cli_main.importlib, "import_module", fake)
    with pytest.raises(RuntimeError, match="must return the parser"):
        cli_main._register_subcommand(_subparsers(), "x.noret")
