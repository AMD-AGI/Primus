###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for tools/ci/select_tests.py (classify-based E2E suite selection).

Only E2E selection is covered: the unit-test suite is always run in full (see
select_tests.py's module docstring for why), so there's nothing to select
there anymore.
"""

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location("select_tests", _ROOT / "tools/ci/select_tests.py")
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)

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
    # No trainer suite (bridge / hummingbirdxt / transformer_engine / diffusion) -> fail-safe all.
    assert set(e2e(["primus/backends/megatron_bridge/x.py"])) == SUITES
    assert set(e2e(["primus/backends/hummingbirdxt/x.py"])) == SUITES
    assert set(e2e(["primus/backends/transformer_engine/x.py"])) == SUITES
    assert set(e2e(["primus/backends/diffusion/x.py"])) == SUITES


def test_e2e_component_change_runs_all():
    # Any other primus/ or tests/unit_tests/ change -- not just the
    # explicitly-listed GLOBAL_TRIGGERS -- also runs everything: classify()
    # maps it to "component", which select_e2e() treats the same as "global".
    assert set(e2e(["primus/core/trainer/base.py"])) == SUITES
    assert set(e2e(["primus/core/launcher/parser.py"])) == SUITES
    assert set(e2e(["tests/unit_tests/core/patches/test_patch.py"])) == SUITES


def test_e2e_bare_examples_file_runs_all():
    # A bare examples/<file> (no backend subdir) is shared launcher plumbing
    # -- e.g. test_maxtext_trainer.py shells out to examples/run_pretrain.sh
    # directly -- so it must not be silently ignored like a docs-only change.
    assert set(e2e(["examples/run_pretrain.sh"])) == SUITES
    # examples/<backend>/... is unaffected: still maps to that one backend.
    assert e2e(["examples/maxtext/configs/x.yaml"]) == ["maxtext"]


def test_e2e_docs_only_runs_none():
    assert e2e(["README.md"]) == []


def test_e2e_added_trainer_is_auto_picked_up():
    # Adding a bridge trainer makes a bridge change run it -- no code change here.
    suites = SUITES | {"megatron_bridge"}
    assert _MOD.select_e2e(["primus/backends/megatron_bridge/x.py"], suites) == ["megatron_bridge"]
