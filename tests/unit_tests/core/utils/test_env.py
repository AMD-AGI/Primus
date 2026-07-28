###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for primus.core.utils.env (distributed env parsing + safe flush)."""

from __future__ import annotations

from primus.core.utils import constant_vars as const
from primus.core.utils.env import flush_before_hard_exit, get_torchrun_env

_ENV = (
    "RANK",
    "WORLD_SIZE",
    "NODE_RANK",
    "NNODES",
    "MASTER_ADDR",
    "MASTER_PORT",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
)


def _clear(monkeypatch):
    for k in _ENV:
        monkeypatch.delenv(k, raising=False)


def test_get_torchrun_env_defaults(monkeypatch):
    _clear(monkeypatch)
    e = get_torchrun_env()
    assert e["rank"] == int(const.LOCAL_NODE_RANK)
    assert e["world_size"] == int(const.LOCAL_WORLD_SIZE)
    assert e["master_addr"] == const.LOCAL_MASTER_ADDR
    assert e["master_port"] == int(const.LOCAL_MASTER_PORT)


def test_get_torchrun_env_reads_torchrun_vars(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("MASTER_ADDR", "host-a")
    monkeypatch.setenv("MASTER_PORT", "12345")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    e = get_torchrun_env()
    assert (e["rank"], e["world_size"]) == (3, 8)
    assert e["master_addr"] == "host-a" and e["master_port"] == 12345
    assert (e["local_rank"], e["local_world_size"]) == (1, 4)


def test_get_torchrun_env_jax_fallback(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("NODE_RANK", "2")
    monkeypatch.setenv("NNODES", "4")
    e = get_torchrun_env()
    assert (e["rank"], e["world_size"]) == (2, 4)


def test_get_torchrun_env_torchrun_takes_precedence_over_jax(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("RANK", "5")
    monkeypatch.setenv("NODE_RANK", "2")
    monkeypatch.setenv("WORLD_SIZE", "16")
    monkeypatch.setenv("NNODES", "4")
    e = get_torchrun_env()
    assert (e["rank"], e["world_size"]) == (5, 16)


def test_flush_before_hard_exit_is_noop_without_coverage(monkeypatch):
    monkeypatch.delenv("COVERAGE_PROCESS_START", raising=False)
    flush_before_hard_exit()  # best-effort, must not raise
