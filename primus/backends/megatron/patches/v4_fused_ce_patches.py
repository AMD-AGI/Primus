###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Route Megatron's TE cross entropy to the fused Primus CE kernel (opt3-mem, opt-in PRIMUS_V4_FUSED_CE=1).

``LanguageModule.compute_language_model_loss`` calls the module-level ``te_parallel_cross_entropy`` when
``cross_entropy_loss_fusion`` is on and ``cross_entropy_fusion_impl == 'te'`` (the DSv4 proxy config). TE's Triton
path costs ~1.37 ms/step on MI455X (online_softmax + cross_entropy + element_mul + fp32->bf16 grad cast);
``_triton_common/fused_ce.py`` does the same math in ~0.28 ms (split-V forward partials + combine, one backward
pass that writes the bf16 logits grad directly). Fallback to TE inside ``primus_parallel_cross_entropy`` for
TP > 1 or label smoothing. No host sync: ``is_cg_capturable`` is irrelevant (the fused kernel never calls
``torch.equal``), so it is safe under ``--cuda_graph_impl full_iteration``.
"""

import os

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0


def _enabled(ctx: PatchContext) -> bool:
    del ctx
    return os.environ.get("PRIMUS_V4_FUSED_CE", "0") == "1"


@register_patch(
    "megatron.v4.fused_cross_entropy",
    backend="megatron",
    phase="before_train",
    description="TE parallel cross entropy -> fused Primus CE (PRIMUS_V4_FUSED_CE=1)",
    condition=_enabled,
)
def patch_v4_fused_ce(ctx: PatchContext) -> None:
    del ctx
    from megatron.core.models.common.language_module import language_module as lm

    from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common.fused_ce import (
        primus_parallel_cross_entropy,
    )

    if getattr(lm.te_parallel_cross_entropy, "_primus_fused_ce", False):
        return

    def te_parallel_cross_entropy(logits, labels, tp_group, is_cg_capturable=False):
        del is_cg_capturable
        return primus_parallel_cross_entropy(logits, labels, 0.0, False, tp_group, -100)

    te_parallel_cross_entropy._primus_fused_ce = True
    lm.te_parallel_cross_entropy = te_parallel_cross_entropy
    try:
        from megatron.core.extensions import transformer_engine as te_ext

        te_ext.te_parallel_cross_entropy = te_parallel_cross_entropy
    except Exception:  # pragma: no cover - TE extension unavailable
        pass
    log_rank_0("[Patch:megatron.v4.fused_cross_entropy] te_parallel_cross_entropy -> primus fused CE")
