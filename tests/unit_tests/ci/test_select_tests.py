###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for tools/ci/select_tests.py (component-aware PR test selection)."""

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location("select_tests", _ROOT / "tools/ci/select_tests.py")
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)
select = _MOD.select_targets

FULL = ["tests/unit_tests/"]


def test_empty_runs_full():
    assert select([]) == FULL


def test_ci_change_runs_full():
    assert select([".github/workflows/ci.yaml"]) == FULL


def test_tools_change_runs_full():
    assert select(["tools/ci/select_tests.py"]) == FULL


def test_root_requirements_runs_full():
    assert select(["requirements.txt"]) == FULL
    assert select(["requirements-jax.txt"]) == FULL


def test_launcher_runs_full():
    assert select(["primus/core/launcher/initialize.py"]) == FULL


def test_unknown_primus_path_runs_full():
    assert select(["primus/brand_new_area/x.py"]) == FULL


def test_megatron_backend_maps_to_both_dirs():
    out = select(["primus/backends/megatron/training/global_vars.py"])
    assert "tests/unit_tests/backends/megatron/" in out
    assert "tests/unit_tests/megatron/" in out


def test_projection_takes_precedence_over_core():
    assert select(["primus/core/projection/engine.py"]) == ["tests/unit_tests/core/projection/"]


def test_generic_core_maps_to_core_tests():
    assert select(["primus/core/trainer/base.py"]) == ["tests/unit_tests/core/"]


def test_cli_and_runner_map_to_cli_tests():
    assert select(["primus/cli/main.py"]) == ["tests/unit_tests/cli/"]
    assert select(["runner/helpers/foo.sh"]) == ["tests/unit_tests/cli/"]
    assert select(["primus_cli.py"]) == ["tests/unit_tests/cli/"]


def test_changed_test_file_runs_its_own_dir():
    assert select(["tests/unit_tests/agents/test_tools.py"]) == ["tests/unit_tests/agents/"]


def test_multiple_components_are_unioned():
    out = select(["primus/agents/a.py", "primus/cli/b.py"])
    assert set(out) == {"tests/unit_tests/agents/", "tests/unit_tests/cli/"}


def test_docs_only_change_is_ignored_then_full():
    # No source/test matched -> fail-safe to full suite.
    assert select(["README.md", "docs/guide.md"]) == FULL
