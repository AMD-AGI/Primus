###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# Must run before anything imports Megatron-LM (megatron/core/jit.py uses
# torch.jit.script at import time).
from primus.core.utils.torch_compat import ensure_torch_jit_script

ensure_torch_jit_script()

from primus.backends.megatron.core.dsv4_features import (
    apply_defaults as _apply_dsv4_defaults,
)

# Fill the DeepSeek-V4 fast-path defaults before any module that reads them at import time is loaded.
_apply_dsv4_defaults()

from primus.backends.megatron.megatron_adapter import MegatronAdapter
from primus.backends.megatron.megatron_pretrain_trainer import MegatronPretrainTrainer
from primus.backends.megatron.megatron_sft_trainer import MegatronSFTTrainer
from primus.core.backend.backend_registry import BackendRegistry

BackendRegistry.register_adapter("megatron", MegatronAdapter)
BackendRegistry.register_trainer_class(MegatronPretrainTrainer, "megatron")
BackendRegistry.register_trainer_class(MegatronSFTTrainer, "megatron", "sft")

# Export trainers for convenience
# Use lazy import for FluxPretrainTrainer to avoid Megatron dependency
# when importing data pipeline components
__all__ = [
    "MegatronAdapter",
    "FluxPretrainTrainer",
    "MegatronPretrainTrainer",
    "MegatronSFTTrainer",
]


def __getattr__(name):
    """Lazy import for trainer classes to avoid Megatron dependency on module import."""
    if name == "FluxPretrainTrainer":
        from primus.backends.megatron.flux_pretrain_trainer import FluxPretrainTrainer

        return FluxPretrainTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
