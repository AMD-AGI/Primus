###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Primus MLPerf logging integration for the Megatron pretrain backend.

Pretrain uses :class:`MLPerfMegatronPretrainTrainer`.  Megatron-Bridge SFT /
post-train workloads use ``primus.backends.megatron_bridge.mlperf_sft``.

``mlperf_logging`` is imported lazily inside methods so non-MLPerf runs are
unaffected.
"""

from primus.backends.megatron.mlperf.mlperf_logger import MLPerfLogger, ThroughputTimer
from primus.backends.megatron.mlperf.mlperf_pretrain_trainer import (
    MLPerfMegatronPretrainTrainer,
)

__all__ = [
    "MLPerfLogger",
    "ThroughputTimer",
    "MLPerfMegatronPretrainTrainer",
    "run_synthetic_warmup",
    "reset_fp8_state",
    "seed_fp8_amax",
]


def __getattr__(name: str):
    if name in ("run_synthetic_warmup", "reset_fp8_state", "seed_fp8_amax"):
        from primus.backends.megatron.mlperf import warmup

        return getattr(warmup, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
