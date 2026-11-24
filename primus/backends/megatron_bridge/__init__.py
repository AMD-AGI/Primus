###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# Must run before anything imports Megatron-LM (megatron/core/jit.py uses
# torch.jit.script at import time).
from primus.core.utils.torch_compat import ensure_torch_jit_script

ensure_torch_jit_script()

from primus.backends.megatron_bridge.megatron_bridge_adapter import (
    MegatronBridgeAdapter,
)
from primus.backends.megatron_bridge.megatron_bridge_posttrain_trainer import (
    MegatronBridgePosttrainTrainer,
)
from primus.backends.megatron_bridge.megatron_bridge_pretrain_trainer import (
    MegatronBridgePretrainTrainer,
)
from primus.core.backend.backend_registry import BackendRegistry

# Register adapter
BackendRegistry.register_adapter("megatron_bridge", MegatronBridgeAdapter)

# Register both pretrain and posttrain trainers under the stage-based registry.
# This way `BackendRegistry.get_trainer_class("megatron_bridge", "<stage>")` works
# for both stages, while ``MegatronBridgeAdapter.load_trainer_class`` still keeps
# a hard-coded fallback for pre-registry callers (see adapter for details).
BackendRegistry.register_trainer_class(MegatronBridgePretrainTrainer, "megatron_bridge", "pretrain")
BackendRegistry.register_trainer_class(MegatronBridgePosttrainTrainer, "megatron_bridge", "sft")
