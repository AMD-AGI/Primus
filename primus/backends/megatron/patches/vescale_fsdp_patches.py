###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
veScale RaggedShard FSDP Patch
==============================

Registers a Megatron training patch that replaces
``PrimusTorchFullyShardedDataParallel`` with
``PrimusVeScaleRaggedShardFSDP`` when both conditions are met:

  1. ``use_torch_fsdp2 = True``  (standard FSDP2 is enabled in config)
  2. ``PRIMUS_VESCALE_RAGGED_SHARD_FSDP=1``  (env var opt-in)

This keeps the default FSDP2 path unchanged and makes the veScale path
an explicit opt-in.

Activation
----------
To enable, set the environment variable before launching:

.. code-block:: bash

    export PRIMUS_VESCALE_RAGGED_SHARD_FSDP=1

and ensure ``use_torch_fsdp2: true`` is set in the trainer YAML config.

How it works
------------
The patch replaces the ``TorchFullyShardedDataParallel`` class reference in
Megatron's distributed module and the ``torch_FSDP`` reference in the
training loop, identical to how ``torch_fsdp2_patches.py`` works.

``PrimusVeScaleRaggedShardFSDP`` implements the same interface as
``PrimusTorchFullyShardedDataParallel``:
  - Constructor signature: ``(config, ddp_config, module, sub_modules_to_wrap, ...)``
  - Methods: ``forward``, ``state_dict``, ``load_state_dict``,
    ``start_grad_sync``, ``finish_grad_sync``, ``zero_grad_buffer``,
    ``broadcast_params``, ``scale_gradients``, ``no_sync``

Key difference from standard FSDP2
------------------------------------
Standard FSDP2:
  Each parameter is individually ``Shard(0)`` sharded. Batched all-gather
  requires interleaved padding/copy because parameter shapes differ.

veScale RaggedShard FSDP:
  All parameters per FSDP unit are flattened into ONE contiguous buffer.
  The flat buffer is sharded with ``RaggedShard(dims=(0,), local_units=...)``.
  All-gather = single ``redistribute(Replicate)`` on the flat buffer.
  → Zero interleaved copies; one collective per FSDP unit.

See ``primus/backends/megatron/core/distributed/vescale_ragged_shard_fsdp.py``
for the full implementation.
"""

import os

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0


def _vescale_ragged_shard_enabled(ctx: PatchContext) -> bool:
    """
    Activation condition:
      1. use_torch_fsdp2 must be True (FSDP2 path is selected)
      2. PRIMUS_VESCALE_RAGGED_SHARD_FSDP=1 env var must be set
    """
    fsdp2_on = getattr(get_args(ctx), "use_torch_fsdp2", False)
    env_on = os.environ.get("PRIMUS_VESCALE_RAGGED_SHARD_FSDP", "0").strip() == "1"
    return fsdp2_on and env_on


@register_patch(
    "megatron.fsdp.vescale_ragged_shard",
    backend="megatron",
    phase="before_train",
    description=(
        "Replace Megatron's TorchFullyShardedDataParallel with "
        "PrimusVeScaleRaggedShardFSDP when use_torch_fsdp2 is enabled "
        "AND PRIMUS_VESCALE_RAGGED_SHARD_FSDP=1 is set."
    ),
    condition=_vescale_ragged_shard_enabled,
    # priority=60 ensures this patch runs AFTER torch_fsdp2_patches (priority=50),
    # so PrimusVeScaleRaggedShardFSDP correctly overwrites PrimusTorchFullyShardedDataParallel.
    priority=60,
)
def patch_vescale_ragged_shard_fsdp(ctx: PatchContext) -> None:
    """
    Inject ``PrimusVeScaleRaggedShardFSDP`` as the FSDP implementation.

    Two injection points are patched (identical to ``torch_fsdp2_patches.py``):
      1. ``megatron.core.distributed.torch_fully_sharded_data_parallel``
         → ``TorchTorchFullyShardedDataParallel``
      2. ``megatron.training.training`` → ``torch_FSDP``
    """
    # ---- Patch 1: Megatron FSDP module reference -------------------------
    import megatron.core.distributed.torch_fully_sharded_data_parallel as fsdp_module

    from primus.backends.megatron.core.distributed.vescale_ragged_shard_fsdp import (
        PrimusVeScaleRaggedShardFSDP,
    )

    fsdp_module.TorchTorchFullyShardedDataParallel = PrimusVeScaleRaggedShardFSDP
    log_rank_0(
        "[Patch:megatron.fsdp.vescale_ragged_shard]  "
        "megatron.core.distributed...TorchTorchFullyShardedDataParallel "
        f"→ {PrimusVeScaleRaggedShardFSDP.__name__}"
    )

    # ---- Patch 2: Training loop reference --------------------------------
    from megatron.training import training as megatron_training

    megatron_training.torch_FSDP = PrimusVeScaleRaggedShardFSDP
    log_rank_0(
        f"[Patch:megatron.fsdp.vescale_ragged_shard]  "
        f"megatron.training.training.torch_FSDP "
        f"→ {PrimusVeScaleRaggedShardFSDP.__name__}"
    )

    log_rank_0(
        "[Patch:megatron.fsdp.vescale_ragged_shard]  "
        "veScale RaggedShard FSDP is ACTIVE.  "
        "Parameters will use zero-copy contiguous flat-buffer sharding."
    )
