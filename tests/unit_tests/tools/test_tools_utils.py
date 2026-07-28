###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for primus.tools.utils single-process (no-torchrun) fallbacks.

These guard the standalone path: the benchmark tools must run on a single GPU
without an initialized process group. Pure CPU; no GPU / dist required.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import torch.distributed as dist  # noqa: E402

from primus.tools.utils import (  # noqa: E402
    derive_path,
    get_rank_world,
    is_rank_0,
    parse_bytes,
    parse_sizes_list,
    round_up_div,
)


def test_is_rank_0_true_without_dist():
    # Regression: is_rank_0 used to raise when dist was not initialized, which
    # crashed single-process `primus-cli benchmark gemm`. It must now treat the
    # single-process case as rank 0 (mirroring gather_records' world==1 path).
    assert not dist.is_initialized()
    assert is_rank_0() is True


def test_get_rank_world_defaults_single_process(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert get_rank_world() == (0, 1)


def test_parse_bytes_suffixes():
    assert parse_bytes("1K") == 1024
    assert parse_bytes("2M") == 2 * 1024**2
    assert parse_bytes("1G") == 1024**3
    assert parse_bytes("4096") == 4096


def test_parse_sizes_list():
    assert parse_sizes_list("1K,2K") == {1024, 2048}
    assert parse_sizes_list("") == set()


def test_round_up_div():
    assert round_up_div(10, 4) == 3
    assert round_up_div(8, 4) == 2


def test_derive_path():
    assert derive_path("./r.md", "_rank") == "./r_rank.md"
    assert derive_path("./r.csv.gz", "_t") == "./r_t.csv.gz"
    assert derive_path("", "_x") == ""
    assert derive_path("-", "_x") == ""
