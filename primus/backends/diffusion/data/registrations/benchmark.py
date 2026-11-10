###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Register the AMDiffusionBenchmark dataset builder."""

from __future__ import annotations


def build_benchmark_dataset(dataset_config: dict):
    from primus.backends.diffusion.data.benchmark import build_amdb_dataset

    return build_amdb_dataset(dataset_config)
