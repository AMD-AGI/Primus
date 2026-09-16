###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Opt-in GPT-OSS LM-head wiring for the Primus-Turbo BF16 FlyDSL GEMM.

Megatron constructs the GPT output layer directly as its native
``ColumnParallelLinear``, bypassing Primus' spec provider. This patch replaces
only that output projection when explicitly enabled by
``PRIMUS_TURBO_ROUTE_GPTOSS_LM_HEAD=1``. Router GEMMs and every other dense
BF16/FP32 GEMM keep their normal Transformer Engine / hipBLASLt path.
"""

import os

from primus.backends.megatron.patches.turbo.utils import is_primus_turbo_can_patch
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "0").strip().lower() in {"1", "true", "yes", "on"}


def _can_route_gptoss_dense_gemm(ctx: PatchContext) -> bool:
    args = get_args(ctx)
    return (
        _env_enabled("PRIMUS_TURBO_ROUTE_GPTOSS_LM_HEAD")
        and bool(getattr(args, "use_turbo_gemm", False))
        and int(getattr(args, "tensor_model_parallel_size", 1)) == 1
        and is_primus_turbo_can_patch(ctx)
    )


@register_patch(
    "megatron.turbo.gptoss_dense_gemm",
    backend="megatron",
    phase="before_train",
    description="Route only GPT-OSS LM-head GEMMs through Primus-Turbo FlyDSL",
    condition=_can_route_gptoss_dense_gemm,
)
def patch_gptoss_dense_gemm(ctx: PatchContext) -> None:
    del ctx
    from megatron.core.models.gpt import gpt_model

    from primus.backends.megatron.core.extensions.primus_turbo import (
        PrimusTurboBF16OutputColumnParallelLinear,
    )

    # GPTModel bypasses the backend spec provider for its output projection.
    # Swap the constructor only while GPTModel.__init__ runs; assigning it
    # permanently on the shared tensor_parallel module would affect unrelated
    # native ColumnParallelLinear users in the same process.
    original_gpt_init = gpt_model.GPTModel.__init__

    def _gpt_init_with_turbo_output(self, *args, **kwargs):
        original_column_parallel = gpt_model.tensor_parallel.ColumnParallelLinear
        gpt_model.tensor_parallel.ColumnParallelLinear = (
            PrimusTurboBF16OutputColumnParallelLinear
        )
        try:
            return original_gpt_init(self, *args, **kwargs)
        finally:
            gpt_model.tensor_parallel.ColumnParallelLinear = original_column_parallel

    gpt_model.GPTModel.__init__ = _gpt_init_with_turbo_output

    log_rank_0(
        "[Patch:megatron.turbo.gptoss_dense_gemm] Routed only GPT output GEMMs "
        "through the Primus-Turbo BF16 FlyDSL backend"
    )
