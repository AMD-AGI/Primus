###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for tools/ci/coverage_summary.py (merging coverage.py JSON reports)."""

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location("coverage_summary", _ROOT / "tools/ci/coverage_summary.py")
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)


def report(files):
    """A coverage.py JSON report: {path under primus/: (statements, executed lines)}."""
    return {
        "files": {
            f"primus/{path}": {"summary": {"num_statements": stmts}, "executed_lines": list(lines)}
            for path, (stmts, lines) in files.items()
        }
    }


def test_reports_merge_by_executed_lines():
    mi300 = report({"core/a/x.py": (10, range(1, 5))})
    mi355 = report({"core/a/x.py": (10, range(3, 7))})
    e2e = report({"core/a/x.py": (10, range(1, 9))})
    out = _MOD.render([mi300, mi355], "t", [e2e])
    assert "**Unit 60.0% -> Unit+E2E 80.0%**" in out


def test_columns_share_files_and_statements_when_reports_disagree():
    unit = report({"core/a/x.py": (10, range(1, 11))})
    e2e = report({"core/a/x.py": (12, range(1, 13)), "core/a/y.py": (5, range(1, 6))})
    out = _MOD.render([unit], "t", [e2e])
    assert "**Unit 58.8% -> Unit+E2E 100.0%**" in out
    assert "Unit 58.8% -> Unit+E2E 100.0%._" in out
