###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest

pytest.importorskip("primus.agents.tuning_agent.legality")

from primus.agents.tuning_agent.config import TargetCluster
from primus.agents.tuning_agent.legality import (
    ArchitectureRecord,
    TrialConfig,
    derive_legality,
    validate,
)


def _arch_moe():
    return ArchitectureRecord(
        workload_path="/fake.yaml",
        num_layers=8,
        hidden_size=4096,
        is_moe=True,
        pipeline_model_parallel_size=4,
        virtual_pipeline_model_parallel_size=2,
    )


def _cluster():
    return TargetCluster(num_nodes=4, gpus_per_node=8, gpu_arch="mi355x")


def test_validate_rejects_recompute_with_zbv():
    arch = _arch_moe()
    cluster = _cluster()
    legality = derive_legality(arch, cluster)
    cfg = TrialConfig(
        tp=1, pp=4, ep=1, cp=1, mbs=2, gbs=128, vpp=2,
        pp_schedule="zbv-formatted",
        recompute_granularity="full",
        recompute_num_layers=4,
    )
    ok, reason = validate(cfg, arch, cluster, legality)
    assert not ok
    assert "split-wgrad" in reason
