"""Unit tests for the native MORI multi-node preflight orchestrator."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from primus.tools.preflight import mori_preflight as helper


def test_resolve_nodes_csv_deduplicates():
    assert helper.resolve_nodes("node1,node2,node1,node3") == [
        "node1",
        "node2",
        "node3",
    ]


def test_resolve_nodes_file(tmp_path):
    node_file = tmp_path / "nodes.txt"
    node_file.write_text("node1\nnode2\nnode1\n")
    assert helper.resolve_nodes(f"@{node_file}") == ["node1", "node2"]


def test_network_probe_uses_one_remote_command(monkeypatch):
    calls = []
    monkeypatch.setattr(
        helper,
        "remote_output",
        lambda *args: calls.append(args) or "fenic\t1\t10.0.0.1",
    )
    assert helper.probe_network("node1", None, None, None) == ("fenic", 1, "10.0.0.1")
    assert len(calls) == 1


def test_multinode_smoke_launches_every_rank(monkeypatch, tmp_path):
    args = SimpleNamespace(
        mori_socket_ifname="fenic",
        mori_gid_index=1,
        mori_master_addr="10.0.0.1",
        mori_master_port=29610,
        mori_smoke_numel=1024,
    )
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(helper.subprocess, "run", fake_run)
    monkeypatch.setattr(helper, "is_local", lambda _node: True)
    monkeypatch.setattr(helper.time, "sleep", lambda _seconds: None)

    assert helper.run_multinode(args, ["node1", "node2"], tmp_path) == 0
    launches = [command[-1] for command in commands]
    assert len(launches) == 2
    assert any("--node_rank=0" in command for command in launches)
    assert any("--node_rank=1" in command for command in launches)


def test_matching_fingerprints_pass(tmp_path):
    results = [
        helper.NodeResult("node1", 0, "nic=ionic hash=abc ccqe=True", tmp_path / "1.log"),
        helper.NodeResult("node2", 0, "nic=ionic hash=abc ccqe=True", tmp_path / "2.log"),
    ]
    helper.validate_fingerprints(results)


def test_mismatched_fingerprints_fail(tmp_path):
    results = [
        helper.NodeResult("node1", 0, "nic=ionic hash=abc ccqe=True", tmp_path / "1.log"),
        helper.NodeResult("node2", 0, "nic=ionic hash=def ccqe=False", tmp_path / "2.log"),
    ]
    with pytest.raises(RuntimeError, match="node-stack mismatch"):
        helper.validate_fingerprints(results)
