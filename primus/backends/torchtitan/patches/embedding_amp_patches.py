###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
TorchTitan Embedding AMP Autocast Patch

This patch mirrors ``TorchTitanPretrainTrainer.patch_torchtitan_embedding_amp``
using the generic Primus patch system, so that AMP/mixed-precision alignment
for ``nn.Embedding`` can be managed in a backend-agnostic way.

Behavior:
    - When enabled via ``primus_turbo.enable_embedding_autocast`` in the
      module config, globally patches ``nn.Embedding.__init__`` to register
      a forward hook that:
        * When AMP/autocast is active, casts outputs to the AMP dtype
          (bf16/fp16) when necessary.
        * Otherwise, leaves outputs unchanged.
"""

from primus.core.patches import PatchContext, get_param, register_patch
from primus.modules.module_utils import log_rank_0


@register_patch(
    "torchtitan.embedding.amp_alignment",
    backend="torchtitan",
    phase="setup",
    description="Align nn.Embedding outputs with AMP/autocast dtype",
    condition=lambda ctx: get_param(ctx, "primus_turbo.enable_embedding_autocast", False),
)
def patch_torchtitan_embedding_amp(ctx: PatchContext) -> None:
    """
    Monkey patch for AMP precision mismatch in nn.Embedding.
    """
    import torch
    import torch.nn as nn

    log_rank_0(
        "[Patch:torchtitan.embedding.amp_alignment] "
        "Installing nn.Embedding AMP/mixed precision alignment patch...",
    )

    def _hook(module, inp, out):
        if not isinstance(out, torch.Tensor) or not out.is_floating_point():
            return out

        if torch.is_autocast_enabled():
            runtime_dtype = torch.get_autocast_gpu_dtype()
            if out.dtype != runtime_dtype:
                return out.to(runtime_dtype)
        return out

    orig_init = nn.Embedding.__init__

    def new_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self.register_forward_hook(_hook)

    nn.Embedding.__init__ = new_init
    log_rank_0(
        "[Patch:torchtitan.embedding.amp_alignment] "
        "nn.Embedding.__init__ patched for AMP/mixed precision alignment.",
    )
