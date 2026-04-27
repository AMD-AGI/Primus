###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch


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
