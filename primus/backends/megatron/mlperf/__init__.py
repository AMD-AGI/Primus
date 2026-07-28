###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Primus MLPerf logging integration for the Megatron pretrain backend.

This package integrates the (formerly external) ``primus_mllog`` thin wrapper
directly into Primus, mapping it onto the new ``BaseTrainer`` lifecycle.

``mlperf_logging`` and other MLPerf-only dependencies are imported lazily
inside methods so that importing this package (which happens whenever the
Megatron backend is loaded) never breaks non-MLPerf runs.
"""

from primus.backends.megatron.mlperf.mlperf_logger import MLPerfLogger, ThroughputTimer
from primus.backends.megatron.mlperf.mlperf_pretrain_trainer import (
    MLPerfMegatronPretrainTrainer,
)

__all__ = [
    "MLPerfLogger",
    "ThroughputTimer",
    "MLPerfMegatronPretrainTrainer",
]
