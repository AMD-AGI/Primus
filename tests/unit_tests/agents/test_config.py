###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for tuning-agent config loading (``config.py``)."""

from __future__ import annotations

import pytest

pytest.importorskip("primus.agents.tuning_agent.config")

from primus.agents.tuning_agent.config import (  # noqa: E402
    AgentConfig,
    _from_env,
    _resolve_api_key,
    load_config,
)

_CRED_ENV = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LLM_API_KEY", "OPENAI_API_BASE", "LLM_MODEL")


def test_resolve_api_key_priority(monkeypatch):
    for k in _CRED_ENV:
        monkeypatch.delenv(k, raising=False)
    assert _resolve_api_key() == ""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k-anthropic")
    assert _resolve_api_key() == "k-anthropic"
    monkeypatch.setenv("OPENAI_API_KEY", "k-openai")
    assert _resolve_api_key() == "k-openai"  # OPENAI wins (first in priority order)


def test_from_env_treats_unset_and_empty_as_default(monkeypatch):
    monkeypatch.delenv("PRIMUS_TEST_X", raising=False)
    assert _from_env("PRIMUS_TEST_X", "fallback") == "fallback"
    monkeypatch.setenv("PRIMUS_TEST_X", "")
    assert _from_env("PRIMUS_TEST_X", "fallback") == "fallback"
    monkeypatch.setenv("PRIMUS_TEST_X", "value")
    assert _from_env("PRIMUS_TEST_X", "fallback") == "value"


def test_load_config_parses_sections(tmp_path, monkeypatch):
    for k in _CRED_ENV:
        monkeypatch.delenv(k, raising=False)
    tc = tmp_path / "target_cluster.yaml"
    tc.write_text(
        "target_cluster:\n"
        "  name: testcluster\n"
        "  num_nodes: 4\n"
        "  gpus_per_node: 8\n"
        "  gpu_arch: mi300x\n"
        "available_for_benchmark:\n"
        "  has_gpu: true\n"
        "  benchmark_gpus: 8\n"
        "optimization:\n"
        "  hbm_capacity_gb: 256\n"
        "  budget:\n"
        "    max_proposals: 10\n"
        "  axes:\n"
        "    gbs: true\n"
        "agent:\n"
        "  llm:\n"
        "    model: openai/gpt-4o\n"
        "    timeout: 100\n"
        "  prompt_extras:\n"
        "    - hint one\n"
    )
    wl = tmp_path / "workload.yaml"
    wl.write_text("modules: {}\n")

    cfg = load_config(tc, wl)

    assert isinstance(cfg, AgentConfig)
    assert (cfg.target_cluster.name, cfg.target_cluster.num_nodes) == ("testcluster", 4)
    assert cfg.target_cluster.gpu_arch == "mi300x"
    assert cfg.benchmark_host.has_gpu is True and cfg.benchmark_host.benchmark_gpus == 8
    assert cfg.optimization.hbm_capacity_gb == 256.0
    assert cfg.optimization.budget.max_proposals == 10
    assert cfg.optimization.axes["gbs"] is True  # YAML override merged in
    assert cfg.optimization.axes["tp"] is True  # default axis preserved
    assert cfg.llm.model == "openai/gpt-4o" and cfg.llm.timeout == 100
    assert "hint one" in cfg.extra_prompt
    assert "testcluster" in str(cfg.out_dir)  # default out_dir nests cluster name


def test_load_config_rejects_non_mapping(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError, match="not a mapping"):
        load_config(bad, tmp_path / "workload.yaml")


def test_load_config_out_dir_override(tmp_path):
    tc = tmp_path / "tc.yaml"
    tc.write_text("target_cluster:\n  name: c\n")
    cfg = load_config(tc, tmp_path / "wl.yaml", out_dir=tmp_path / "custom_out")
    assert str(cfg.out_dir).endswith("custom_out")
