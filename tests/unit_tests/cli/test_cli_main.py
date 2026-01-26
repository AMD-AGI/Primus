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
    assert "primus.cli.subcommands.preflight" in modules
    assert "primus.cli.subcommands.projection" in modules


def test_load_subcommands_invokes_register(monkeypatch):
    captured = []

    def fake_iter():
        return ["x.alpha", "x.beta"]

    def fake_import(name):
        parser = argparse.ArgumentParser()

        def register(subparsers):
            captured.append((name, subparsers))
            parser.set_defaults(func=lambda *_args, **_kwargs: None)
            return parser

        module = type("FakeModule", (), {})()
        module.register_subcommand = register
        return module

    monkeypatch.setattr(cli_main, "_iter_subcommand_modules", fake_iter)
    monkeypatch.setattr(cli_main.importlib, "import_module", fake_import)

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd")

    cli_main._load_subcommands(subparsers)

    assert captured == [("x.alpha", subparsers), ("x.beta", subparsers)]


def test_load_subcommands_requires_func(monkeypatch):
    def fake_iter():
        return ["x.alpha"]

    def fake_import(name):
        parser = argparse.ArgumentParser()
        parser.add_subparsers(dest="suite")

        def register(subparsers):
            return parser

        module = type("FakeModule", (), {})()
        module.register_subcommand = register
        return module

    monkeypatch.setattr(cli_main, "_iter_subcommand_modules", fake_iter)
    monkeypatch.setattr(cli_main.importlib, "import_module", fake_import)

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd")

    with pytest.raises(RuntimeError, match="set_defaults"):
        cli_main._load_subcommands(subparsers)
