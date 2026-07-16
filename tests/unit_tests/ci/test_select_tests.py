###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for tools/ci/select_tests.py (classify-based PR test selection)."""

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location("select_tests", _ROOT / "tools/ci/select_tests.py")
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)
select = _MOD.select_targets

FULL = ["tests/unit_tests/"]


# --- unit-test selection ---------------------------------------------------
def test_empty_runs_full():
    assert select([]) == FULL


def test_global_change_runs_full():
    assert select([".github/workflows/ci.yaml"]) == FULL
    assert select(["tools/ci/select_tests.py"]) == FULL
    assert select(["runner/helpers/x.sh"]) == FULL
    assert select(["requirements.txt"]) == FULL
    assert select(["primus/core/launcher/initialize.py"]) == FULL


def test_non_py_under_primus_runs_full():
    # configs / fixtures can't be localized to a unit dir -> fail-safe.
    assert select(["primus/configs/x.yaml"]) == FULL


def test_backend_maps_to_its_unit_dir():
    out = select(["primus/backends/megatron/training/global_vars.py"])
    assert "tests/unit_tests/backends/megatron/" in out
    assert "tests/unit_tests/megatron/" in out  # megatron's extra GPU-operator tests


def test_backend_without_unit_dir_runs_full():
    # transformer_engine has no tests/unit_tests/backends/transformer_engine/.
    assert select(["primus/backends/transformer_engine/x.py"]) == FULL


def test_component_maps_to_isomorphic_dir():
    assert select(["primus/core/projection/engine.py"]) == ["tests/unit_tests/core/projection/"]
    assert select(["primus/agents/a.py"]) == ["tests/unit_tests/agents/"]


def test_changed_unit_test_runs_its_dir():
    assert select(["tests/unit_tests/agents/test_tools.py"]) == ["tests/unit_tests/agents/"]


def test_docs_only_runs_full():
    assert select(["README.md", "docs/guide.md"]) == FULL


# --- E2E selection (pass a fixed suite set for determinism) ----------------
SUITES = {"megatron", "torchtitan", "maxtext"}


def e2e(files):
    return _MOD.select_e2e(files, SUITES)


def test_e2e_empty_runs_all():
    assert set(e2e([])) == SUITES


def test_e2e_global_runs_all():
    assert set(e2e([".github/workflows/ci.yaml"])) == SUITES
    assert set(e2e(["runner/helpers/x.sh"])) == SUITES


def test_e2e_backend_with_trainer_runs_its_suite():
    assert e2e(["primus/backends/megatron/x.py"]) == ["megatron"]
    assert e2e(["examples/maxtext/configs/x.yaml"]) == ["maxtext"]
    assert e2e(["tests/trainer/test_torchtitan_trainer.py"]) == ["torchtitan"]


def test_e2e_backend_without_trainer_runs_all():
    # No trainer suite (bridge / hummingbirdxt / transformer_engine) -> fail-safe all.
    assert set(e2e(["primus/backends/megatron_bridge/x.py"])) == SUITES
    assert set(e2e(["primus/backends/hummingbirdxt/x.py"])) == SUITES
    assert set(e2e(["primus/backends/transformer_engine/x.py"])) == SUITES


def test_e2e_component_change_runs_all():
    assert set(e2e(["primus/core/trainer/base.py"])) == SUITES


def test_e2e_docs_only_runs_none():
    assert e2e(["README.md"]) == []


def test_e2e_added_trainer_is_auto_picked_up():
    # Adding a bridge trainer makes a bridge change run it -- no code change here.
    suites = SUITES | {"megatron_bridge"}
    assert _MOD.select_e2e(["primus/backends/megatron_bridge/x.py"], suites) == ["megatron_bridge"]
