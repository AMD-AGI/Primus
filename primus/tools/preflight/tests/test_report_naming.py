###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for ``preflight_perf_test._ensure_report_file_name`` /
``_announce_report_paths`` (consolidate-preflight-direct-wrappers, R5).

These tests pin the contract of the Python-side replacements for the
old bash ``run_preflight_direct.sh`` behavior:

* Every preflight run that doesn't get an explicit ``--report-file-name``
  must write to a unique, timestamped path. Stale reports from a previous
  run must NEVER be mistaken for the current run's output.
* Once reports exist, rank 0 must print their absolute paths so an
  operator running under ``primus-cli direct`` can find them without
  guessing the dump-path layout.
* Best-effort: the announcement must never raise, even when nothing was
  written (the per-run output may have failed earlier).
"""

from __future__ import annotations

import argparse
import os
import re
from datetime import datetime, timedelta
from typing import Optional

from primus.tools.preflight.preflight_args import add_preflight_parser
from primus.tools.preflight.preflight_perf_test import (
    _announce_report_paths,
    _ensure_report_file_name,
)


def _build_preflight_parser() -> argparse.ArgumentParser:
    """Construct the same parser that `primus-cli preflight` registers,
    isolated from the broader primus.cli wiring so tests can exercise
    just the argparse contract.
    """
    p = argparse.ArgumentParser(prog="preflight-test")
    add_preflight_parser(p)
    return p


# ---------------------------------------------------------------------------
# `--report-file-name` default contract
# ---------------------------------------------------------------------------


def test_preflight_args_report_file_name_default_is_none():
    """The argparse default for ``--report-file-name`` must be ``None`` so
    ``_ensure_report_file_name`` can detect "user did not pass it" and
    auto-generate a unique name. A non-None default would silently
    suppress the auto-naming logic and re-introduce the "every run
    overwrites preflight_report.md" footgun the plan removed."""
    parser = _build_preflight_parser()
    ns = parser.parse_args([])
    assert ns.report_file_name is None, (
        "argparse default must stay None -- _ensure_report_file_name "
        "uses falsy check to decide whether to auto-generate"
    )


# ---------------------------------------------------------------------------
# `_ensure_report_file_name`
# ---------------------------------------------------------------------------


_AUTO_NAME_RE = re.compile(r"^preflight-(\d+)N-(\d{8})-(\d{6})$")


def test_ensure_report_file_name_user_value_wins(monkeypatch):
    """When the user passed ``--report-file-name=foo`` it must survive
    untouched -- the auto-name path must never overwrite an explicit
    value."""
    monkeypatch.setenv("NNODES", "32")  # would otherwise affect auto-name
    ns = argparse.Namespace(report_file_name="my-custom-name")
    out = _ensure_report_file_name(ns)
    assert out == "my-custom-name"
    assert ns.report_file_name == "my-custom-name"


def test_ensure_report_file_name_auto_uses_nnodes_env(monkeypatch):
    """When ``NNODES`` is set, the auto-name must use it directly. This
    is the critical case for info-only mode where no process group is
    initialized and ``get_rank_world()`` would return world=1."""
    monkeypatch.setenv("NNODES", "42")
    ns = argparse.Namespace(report_file_name=None)
    out = _ensure_report_file_name(ns)
    m = _AUTO_NAME_RE.match(out)
    assert m is not None, f"auto-name {out!r} doesn't match expected pattern"
    assert m.group(1) == "42"
    assert ns.report_file_name == out


def test_ensure_report_file_name_idempotent(monkeypatch):
    """Calling ``_ensure_report_file_name`` twice in the same run must
    return the same name -- otherwise different code paths (run /
    info / announce) would each compute their own timestamped name and
    write to different files."""
    monkeypatch.setenv("NNODES", "8")
    ns = argparse.Namespace(report_file_name=None)
    first = _ensure_report_file_name(ns)
    # Sleep is unnecessary -- if the function recomputed it would use
    # `datetime.now()` again, which is enough to diverge at second
    # granularity over a slow run. We test the stronger invariant: the
    # second call returns the stored value verbatim, without recomputing.
    monkeypatch.setattr(
        "primus.tools.preflight.preflight_perf_test.datetime",
        _ExplodingDatetime,  # any access to datetime would raise
    )
    second = _ensure_report_file_name(ns)
    assert second == first


class _ExplodingDatetime:
    """datetime stub that raises on any use -- proves a method doesn't
    reach the auto-name regeneration path."""

    @classmethod
    def now(cls, *_a, **_kw):
        raise AssertionError(
            "_ensure_report_file_name regenerated the timestamp on " "the second call (it must be idempotent)"
        )


def test_ensure_report_file_name_unique_per_run(monkeypatch):
    """Two independent invocations (different `args` objects with
    `report_file_name=None`) at different timestamps must produce
    distinct names -- this is the guarantee that protects against
    stale-report aliasing under back-to-back runs."""
    monkeypatch.setenv("NNODES", "4")

    class _FrozenDatetime:
        _now = datetime(2026, 5, 7, 12, 0, 0)

        @classmethod
        def now(cls, *_a, **_kw):
            return cls._now

    monkeypatch.setattr("primus.tools.preflight.preflight_perf_test.datetime", _FrozenDatetime)
    ns_a = argparse.Namespace(report_file_name=None)
    name_a = _ensure_report_file_name(ns_a)
    # Bump the clock by one second to simulate the next run.
    _FrozenDatetime._now = _FrozenDatetime._now + timedelta(seconds=1)
    ns_b = argparse.Namespace(report_file_name=None)
    name_b = _ensure_report_file_name(ns_b)

    assert name_a != name_b
    assert name_a.endswith("12-00-00".replace("-", ""))
    assert name_b.endswith("12-00-01".replace("-", ""))


def test_ensure_report_file_name_handles_invalid_nnodes(monkeypatch):
    """A malformed ``NNODES`` (non-digit) must NOT crash the auto-name
    helper; it must fall back to deriving from torch's world size, with a
    safe minimum of 1."""
    monkeypatch.setenv("NNODES", "")  # explicitly empty
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    monkeypatch.delenv("GPUS_PER_NODE", raising=False)
    ns = argparse.Namespace(report_file_name=None)
    out = _ensure_report_file_name(ns)
    m = _AUTO_NAME_RE.match(out)
    assert m is not None, f"auto-name {out!r} doesn't match expected pattern"
    # When world isn't initialized we end up with nnodes=max(1, 1//8)=1.
    assert int(m.group(1)) >= 1


# ---------------------------------------------------------------------------
# `_announce_report_paths`
# ---------------------------------------------------------------------------


def _make_args(dump_path: str, name: Optional[str]) -> argparse.Namespace:
    return argparse.Namespace(dump_path=dump_path, report_file_name=name)


def test_announce_report_paths_finds_existing_md(tmp_path, capsys, monkeypatch):
    """Smoke: ``_announce_report_paths`` must print absolute paths for
    every report variant that exists on disk."""
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    (tmp_path / "preflight-2N-20260507-120000.md").write_text("# report")
    (tmp_path / "preflight-2N-20260507-120000_perf.md").write_text("# perf")

    args = _make_args(str(tmp_path), "preflight-2N-20260507-120000")
    _announce_report_paths(args)

    out = capsys.readouterr().out
    assert "preflight-2N-20260507-120000.md" in out
    assert "preflight-2N-20260507-120000_perf.md" in out
    # Paths must be absolute -- operators copy them into srun nodelists
    # / scp commands; a relative path here is a footgun under SLURM
    # where the cwd doesn't match across nodes.
    for line in out.splitlines():
        if line.startswith("[Primus:Preflight] Report:"):
            path = line.split("Report:", 1)[1].strip()
            assert os.path.isabs(path), f"non-absolute report path: {path!r}"


def test_announce_report_paths_warns_when_no_files(tmp_path, capsys, monkeypatch):
    """When the helper is called but no report files exist (e.g. preflight
    crashed before writing), it must emit a WARN line so the operator
    isn't left wondering where the report went -- but it must NOT raise."""
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    args = _make_args(str(tmp_path), "ghost-name")
    _announce_report_paths(args)  # must not raise
    out = capsys.readouterr().out
    assert "WARN" in out
    assert "ghost-name" in out


def test_announce_report_paths_skips_non_rank0(tmp_path, capsys, monkeypatch):
    """Non-rank-0 ranks must stay silent -- otherwise every node would
    print the same paths and clutter the launcher output."""
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "8")
    (tmp_path / "preflight-1N-20260507-120000.md").write_text("# report")
    args = _make_args(str(tmp_path), "preflight-1N-20260507-120000")
    _announce_report_paths(args)
    out = capsys.readouterr().out
    assert out == "", f"non-rank-0 announce produced output: {out!r}"


def test_announce_report_paths_no_name_is_noop(tmp_path, capsys, monkeypatch):
    """Defensive: if ``args.report_file_name`` is somehow still None when
    we reach the announcement (e.g. an early bail-out before
    ``_ensure_report_file_name`` was called), the helper must be a
    silent no-op rather than crash on the format-string path build."""
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    args = _make_args(str(tmp_path), None)
    _announce_report_paths(args)  # must not raise
    assert capsys.readouterr().out == ""
