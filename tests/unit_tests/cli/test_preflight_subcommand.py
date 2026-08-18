###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for ``preflight.register_subcommand`` (pure argparse surface).

Covers the parser wiring and the ``tools/preflight/preflight_args`` helper it
imports (selection flags, --check-* aliases, defaults). ``run()`` probes GPU /
network / host hardware and raises SystemExit, so it is out of scope.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from primus.cli.subcommands import preflight
from primus.tools.preflight import mori_preflight


def _build_parser():
    parser = argparse.ArgumentParser(prog="primus")
    subparsers = parser.add_subparsers(dest="cmd")
    returned = preflight.register_subcommand(subparsers)
    return parser, returned


def test_register_returns_parser_wired_to_run():
    _, returned = _build_parser()
    assert returned.get_default("func") is preflight.run


def test_defaults():
    parser, _ = _build_parser()
    args = parser.parse_args(["preflight"])
    assert args.check_host is False
    assert args.check_gpu is False
    assert args.check_network is False
    assert args.perf_test is False
    assert args.dist_timeout_sec == 120
    assert args.dump_path == "output/preflight"
    # Default is None so the tool auto-generates a unique timestamped name
    # (preflight-{NNODES}N-{YYYYMMDD-HHMMSS}) at run time.
    assert args.report_file_name is None
    assert args.save_pdf is True
    assert args.mori is False
    assert args.mori_nodes is None
    assert args.mori_smoke_numel == 67108864


def test_selection_flags():
    parser, _ = _build_parser()
    args = parser.parse_args(["preflight", "--gpu", "--network"])
    assert args.check_gpu is True
    assert args.check_network is True
    assert args.check_host is False


def test_check_alias_maps_to_same_dest():
    # --host and --check-host share dest=check_host.
    parser, _ = _build_parser()
    args = parser.parse_args(["preflight", "--check-host"])
    assert args.check_host is True


def test_disable_pdf_sets_store_false():
    parser, _ = _build_parser()
    args = parser.parse_args(["preflight", "--disable-pdf"])
    assert args.save_pdf is False


def test_perf_test_flag():
    parser, _ = _build_parser()
    args = parser.parse_args(["preflight", "--perf-test", "--plot"])
    assert args.perf_test is True
    assert args.plot is True


def test_mori_mode_flags():
    parser, _ = _build_parser()
    args = parser.parse_args(
        [
            "preflight",
            "--mori",
            "--mori-nodes",
            "node-1,node-2,node-3",
            "--mori-socket-ifname",
            "fenic",
            "--mori-gid-index",
            "1",
        ]
    )
    assert args.mori is True
    assert args.mori_nodes == "node-1,node-2,node-3"
    assert args.mori_socket_ifname == "fenic"
    assert args.mori_gid_index == 1


def test_mori_mode_maps_cli_to_helper_args(monkeypatch, tmp_path):
    parser, _ = _build_parser()
    args = parser.parse_args(
        [
            "preflight",
            "--mori",
            "--dump-path",
            str(tmp_path),
            "--mori-nodes",
            "node-1,node-2",
            "--mori-max-jobs",
            "12",
        ]
    )
    repo_root = Path(mori_preflight.__file__).resolve().parents[3]
    monkeypatch.setenv("PRIMUS_PATH", str(repo_root))
    captured = []
    monkeypatch.setattr(
        mori_preflight,
        "run_orchestrator",
        lambda *values: captured.extend(values) or 0,
    )
    assert mori_preflight.run_mori_preflight(args, []) == 0
    native_args, native_repo_root, native_log_dir = captured
    assert native_args is args
    assert native_args.mori_nodes == "node-1,node-2"
    assert native_args.mori_max_jobs == 12
    assert native_log_dir.parent == tmp_path
    assert native_repo_root == repo_root
