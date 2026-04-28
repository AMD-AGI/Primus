###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from primus.tools.preflight.network.info import Finding


def test_resolve_default_is_primus_package():
    import primus.tools.preflight.cluster_sphere.paths as paths_mod

    expected = Path(paths_mod.__file__).resolve().parent
    assert paths_mod.resolve_cluster_sphere_root() == expected


def test_resolve_override_primus_cluster_sphere_env(tmp_path, monkeypatch):
    alt = tmp_path / "override"
    alt.mkdir()
    monkeypatch.setenv("PRIMUS_CLUSTER_SPHERE_ROOT", str(alt.resolve()))

    from primus.tools.preflight.cluster_sphere.paths import resolve_cluster_sphere_root

    assert resolve_cluster_sphere_root() == alt.resolve()


def test_resolve_override_must_exist(tmp_path, monkeypatch):
    monkeypatch.setenv("PRIMUS_CLUSTER_SPHERE_ROOT", str(tmp_path / "nonexistent"))

    import primus.tools.preflight.cluster_sphere.paths as paths_mod

    expected = Path(paths_mod.__file__).resolve().parent
    assert paths_mod.resolve_cluster_sphere_root() == expected


def test_env_recommender_nccl_exports_no_devices():
    from primus.tools.preflight.cluster_sphere import env_recommender as er

    eng = er.EnvRecommenderEngine()
    with patch.object(er.glob, "glob", return_value=[]):
        result = eng.build_result()
        assert result.devices == []
        assert "No RDMA devices" in "".join(result.warnings)


def test_collect_findings_warns_when_empty_sysfs():
    from primus.tools.preflight.cluster_sphere import env_recommender as er
    from primus.tools.preflight.cluster_sphere.env_recommender import collect_cluster_sphere_env_findings

    with patch.object(er.glob, "glob", return_value=[]):
        findings = collect_cluster_sphere_env_findings()
        assert len(findings) == 1
        assert findings[0].level == "warn"


def test_emit_markdown_firmware_report_section():
    from primus.tools.preflight.cluster_sphere.report import emit_cluster_sphere_env_markdown

    findings = [
        Finding(
            level="info",
            message="Cluster Sphere RDMA environment recommendations",
            details={
                "warnings": [],
                "devices": [
                    {
                        "rdma": "mlx5_0",
                        "pci": "—",
                        "netdev": "eth0",
                        "firmware": "fwA",
                        "gid_index": "3",
                        "gid_value": "::",
                        "vendor": "MLNX",
                    },
                ],
                "firmware_by_version": {"fwA": ["mlx5_0"], "fwB": ["mlx5_1"]},
                "nccl_exports": [],
            },
        ),
    ]
    md = emit_cluster_sphere_env_markdown("test-host", findings)
    assert "**Firmware report:**" in md
    assert "| fwA | mlx5_0 |" in md
    assert "| fwB | mlx5_1 |" in md


def test_verbs_pair_routes_by_slurm_procid(monkeypatch):
    from primus.tools.preflight.cluster_sphere import __main__ as cs_main

    args = MagicMock()
    args.device = None
    args.port = None
    args.timeout = 120
    args.client_delay = 0.0

    monkeypatch.delenv("SLURM_PROCID", raising=False)
    assert cs_main._cmd_verbs_pair(args) == 2

    monkeypatch.setenv("SLURM_PROCID", "2")
    assert cs_main._cmd_verbs_pair(args) == 2

    monkeypatch.setenv("SLURM_PROCID", "1")
    monkeypatch.delenv("SERVER_RDMA_IP", raising=False)
    assert cs_main._cmd_verbs_pair(args) == 2

    monkeypatch.setenv("SERVER_RDMA_IP", "10.0.0.1")
    mock_cli = MagicMock(return_value=0)
    monkeypatch.setattr(cs_main, "_cmd_verbs_client", mock_cli)
    assert cs_main._cmd_verbs_pair(args) == 0
    mock_cli.assert_called_once()
    assert args.server_ip == "10.0.0.1"


def test_nccl_exports_with_stub_devices(monkeypatch):
    from primus.tools.preflight.cluster_sphere.env_recommender import EnvRecommenderEngine
    from primus.tools.preflight.cluster_sphere.env_recommender import _DeviceInfo

    eng = EnvRecommenderEngine()
    eng._devices = [
        _DeviceInfo("mlx5_0", "0000:01:00.0", "eth0", "fw1", "3", "::ffff:1.2.3.4", "MLNX")
    ]
    monkeypatch.setattr(EnvRecommenderEngine, "get_socket_ifname_value", lambda self: "eth0")
    warns: list = []
    exports = eng.build_nccl_exports(warns)
    joined = "\n".join(exports)
    assert "NCCL_IB_HCA=mlx5_0" in joined
    assert "NCCL_SOCKET_IFNAME=eth0" in joined
