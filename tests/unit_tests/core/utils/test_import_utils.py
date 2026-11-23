###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for primus.core.utils.import_utils.lazy_import.

lazy_import tries a list of module paths and returns the first one that exposes
the requested symbol, else raises ImportError. Tested with real stdlib modules
so no mocking of importlib is required.
"""

import sys
import types

import pytest

from primus.core.utils.import_utils import MLPERF_COMMON_MODULES, lazy_import


@pytest.fixture(autouse=True)
def _silence_logger(monkeypatch):
    """lazy_import logs via log_rank_0, which needs a configured global logger.
    Stub it so these tests exercise import logic without logger setup."""
    monkeypatch.setattr("primus.core.utils.import_utils.log_rank_0", lambda *a, **k: None)


def test_lazy_import_returns_symbol_from_first_valid_path():
    # `os.getcwd` exists -> returned directly
    fn = lazy_import(["os"], "getcwd")
    assert fn() == __import__("os").getcwd()


def test_lazy_import_falls_back_to_later_path():
    # first module exists but lacks the symbol path resolution still finds it in
    # a later candidate module that does provide it.
    sep = lazy_import(["nonexistent_module_xyz", "os"], "sep")
    assert sep == __import__("os").sep


def test_lazy_import_raises_when_symbol_missing_everywhere():
    with pytest.raises(ImportError):
        lazy_import(["nonexistent_a", "nonexistent_b"], "whatever")


def test_lazy_import_skips_module_without_symbol():
    assert lazy_import(["json", "os"], "sep") == __import__("os").sep
    with pytest.raises(ImportError):
        lazy_import(["json"], "sep")


def _fake_module(monkeypatch, name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)


def test_mlperf_common_preferred_over_primus_mllog(monkeypatch):
    _fake_module(monkeypatch, "mlperf_common", MLPerfSFTLogger="new")
    _fake_module(monkeypatch, "primus_mllog", MLPerfSFTLogger="old")
    assert lazy_import(MLPERF_COMMON_MODULES, "MLPerfSFTLogger") == "new"


def test_nvidia_mlperf_common_falls_back_to_primus_mllog(monkeypatch):
    # rocm/primus images up to v26.9: NVIDIA's mlperf_common next to primus_mllog.
    _fake_module(monkeypatch, "mlperf_common")
    _fake_module(monkeypatch, "primus_mllog", MLPerfSFTLogger="old")
    assert lazy_import(MLPERF_COMMON_MODULES, "MLPerfSFTLogger") == "old"
