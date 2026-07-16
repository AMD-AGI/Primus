###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the projection communication model (``collective_model``).

These are pure analytical functions (inputs: a hardware/parallelism ``args``
namespace + message size + GPU count; output: a modelled time in ms). They do
not touch the GPU, NCCL, or origami, so the tests are CPU-only, fast and
deterministic.

The minimal set below targets the collective functions left uncovered by the
simulate integration runs (direct_alltoall, sendrecv, Ring*, cp_allgather,
pod_latency_and_volume_protocol, get_bandwidth_and_latency, allgather,
reduce_scatter). The top-level selectors additionally exercise the Ring /
single-shot / hierarchical algorithm branches they choose among.
"""

from types import SimpleNamespace

import pytest

from primus.core.projection.module_profilers import collective_model as cm

MSG = 64 * 1024 * 1024  # 64 MB message


def _args(hp=1, cp=1, ep=1, num_nodes=1, switch_topology=False, node_topology="mesh"):
    """Build a hardware/parallelism args namespace (values mirror MI300X)."""
    return SimpleNamespace(
        # bandwidth / latency (GB/s, us) — from examples/hardware_configs/mi300x.yaml
        node_bw=448.0,
        node_lat=0.45,
        pod_bw=50.0,
        pod_lat=2.0,
        cluster_bw=25.0,
        cluster_lat=10.0,
        # topology
        node_size=8,
        pod_size=64,
        nics_per_node=8,
        switch_topology=switch_topology,
        node_topology=node_topology,
        vector_flops=163.0e12,
        # parallelism dims
        hp=hp,
        cp=cp,
        ep=ep,
        pp=1,
        num_nodes=num_nodes,
        # protocol / launch latencies
        write_latency=0.3,
        write_resp=0.3,
        hbm_latency=0.1,
        kernel_launch_latency=2.0,
        bw_eff=0.8,
    )


# ----------------------- helper / protocol functions -----------------------


@pytest.mark.parametrize("domain,expect_bw", [(4, 448.0), (32, 50.0), (256, 25.0)])
def test_get_bandwidth_and_latency_tiers(domain, expect_bw):
    bw, lat = cm.get_bandwidth_and_latency(_args(), domain)
    assert bw > 0 and lat > 0
    # node tier returns (mesh-derated) node bw; pod/cluster return their tier bw.
    if domain > 64:
        assert bw == expect_bw


@pytest.mark.parametrize("proto", ["simple", "ll", "ll64", "ll128"])
def test_pod_latency_and_volume_protocol(proto):
    lat, size = cm.pod_latency_and_volume_protocol(_args(), MSG, proto)
    assert lat > 0 and size > 0


def test_pod_latency_unknown_protocol_raises():
    with pytest.raises(ValueError):
        cm.pod_latency_and_volume_protocol(_args(), MSG, "bogus")


def test_sendrecv_intra_and_inter_node():
    assert cm.sendrecv(_args(), MSG) > 0
    # pp/inter-node placement path
    assert cm.sendrecv(_args(hp=8, num_nodes=2), MSG) > 0


# ----------------------- collective algorithms -----------------------


def test_cp_allgather_node_pod_and_early_return():
    assert cm.cp_allgather(_args(cp=4), MSG, gpus=4, protocol="simple") > 0  # within node
    assert cm.cp_allgather(_args(cp=16), MSG, gpus=16, protocol="simple") > 0  # within pod
    assert cm.cp_allgather(_args(), MSG, gpus=1) == 0  # trivial early return


def test_direct_alltoall_inter_node():
    # Requires hp == 1 and node_size < hp*gpus <= pod_size.
    assert cm.direct_alltoall(_args(hp=1), MSG, gpus=16, protocol="simple") > 0


def test_ring_allgather_tiers():
    assert cm.RingAllgather(_args(), MSG, gpus=8, protocol="simple") > 0  # node
    assert cm.RingAllgather(_args(), MSG, gpus=16, protocol="simple") > 0  # pod
    assert cm.RingAllgather(_args(), MSG, gpus=128, protocol="simple") > 0  # cluster


def test_ring_reduce_scatter_tiers():
    assert cm.RingRS(_args(), MSG, gpus=8, protocol="simple") > 0
    assert cm.RingRS(_args(), MSG, gpus=32, protocol="simple") > 0
    assert cm.RingRS(_args(), MSG, gpus=128, protocol="simple") > 0


# ----------------------- top-level selectors (exercise multiple algos) -----------------------


@pytest.mark.parametrize("gpus", [8, 16, 128])
def test_allgather_selector(gpus):
    # selects among run_allgather (bruck), RingAllgather, single_shot_allgather
    assert cm.allgather(_args(), MSG, gpus=gpus) > 0


@pytest.mark.parametrize("gpus", [8, 16, 128])
def test_reduce_scatter_selector(gpus):
    # selects among run_reduce_scatter, RingRS, single_shot_reduce_scatter
    assert cm.reduce_scatter(_args(), MSG, gpus=gpus) > 0


@pytest.mark.parametrize("gpus", [8, 16, 128])
def test_allreduce_selector(gpus):
    # selects among Ring, bruck, hypercube, single_shot, hierarchical allreduce
    assert cm.allreduce(_args(), MSG, gpus=gpus) > 0


def test_alltoall_selector_single_and_multi_node():
    assert cm.alltoall(_args(hp=1, ep=8), MSG, gpus=8) > 0  # within node
    assert cm.alltoall(_args(hp=1, ep=16), MSG, gpus=16) > 0  # inter-node (direct/hierarchical)


# ----------------------- model-property assertions (not just ">0") -----------------------


def test_trivial_cases_return_zero():
    """gpus==1 or msg_size==0 must short-circuit to 0 (covers early-return branches)."""
    a = _args()
    assert cm.allgather(a, MSG, gpus=1) == 0
    assert cm.reduce_scatter(a, MSG, gpus=1) == 0
    assert cm.allreduce(a, MSG, gpus=1) == 0
    assert cm.RingAllgather(a, 0, gpus=8, protocol="simple") == 0
    assert cm.RingRS(a, 0, gpus=8, protocol="simple") == 0
    assert cm.cp_allgather(a, 0, gpus=8, protocol="simple") == 0


@pytest.mark.parametrize("coll", ["allreduce", "allgather", "reduce_scatter"])
def test_time_increases_with_message_size(coll):
    """Larger messages must not be cheaper (bandwidth-bound monotonicity)."""
    a = _args()
    fn = getattr(cm, coll)
    small = fn(a, 1 * 1024 * 1024, gpus=16)
    large = fn(a, 256 * 1024 * 1024, gpus=16)
    assert 0 < small <= large


def test_inter_node_not_faster_than_intra_node():
    """Same message, more GPUs spanning nodes shouldn't beat the single-node case."""
    a = _args()
    intra = cm.allreduce(a, MSG, gpus=8)  # fits one node
    inter = cm.allreduce(a, MSG, gpus=64)  # spans the pod
    assert inter >= intra > 0
