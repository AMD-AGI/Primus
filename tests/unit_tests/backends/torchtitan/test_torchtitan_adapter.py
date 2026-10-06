###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import importlib.metadata

import primus.backends.torchtitan.torchtitan_adapter as adapter_module
from primus.backends.torchtitan.torchtitan_adapter import TorchTitanAdapter


def _metadata_missing(_name):
    raise importlib.metadata.PackageNotFoundError


def test_detect_backend_version_prefers_package_metadata(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "9.9.9")

    assert TorchTitanAdapter().detect_backend_version() == "9.9.9"


def test_detect_backend_version_falls_back_to_source_tree(monkeypatch, tmp_path):
    adapter_file = tmp_path / "primus" / "backends" / "torchtitan" / "torchtitan_adapter.py"
    version_file = tmp_path / "third_party" / "torchtitan" / "assets" / "version.txt"
    version_file.parent.mkdir(parents=True)
    version_file.write_text("0.2.2\n", encoding="utf-8")

    monkeypatch.setattr(importlib.metadata, "version", _metadata_missing)
    monkeypatch.setattr(adapter_module, "__file__", str(adapter_file))

    assert TorchTitanAdapter().detect_backend_version() == "0.2.2"


def test_detect_backend_version_is_unknown_without_metadata_or_source(monkeypatch, tmp_path):
    adapter_file = tmp_path / "primus" / "backends" / "torchtitan" / "torchtitan_adapter.py"

    monkeypatch.setattr(importlib.metadata, "version", _metadata_missing)
    monkeypatch.setattr(adapter_module, "__file__", str(adapter_file))

    assert TorchTitanAdapter().detect_backend_version() == "unknown"
