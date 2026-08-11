###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Torch Fused AdamW Opt-In Patch
================================

Set ``PRIMUS_TORCH_OPTIM=1`` to force Megatron's optimizer construction to
use ``torch.optim.AdamW(fused=True)`` instead of TE/Apex ``FusedAdam``. This
exists purely for bit-level reproducibility experiments against
flash-linear-attention's (FLA) reference training runs, which use the plain
PyTorch optimizer.

Mechanism:
    * ``megatron.core.optimizer`` picks its ``Adam``/``SGD``/
      ``USING_PYTORCH_OPTIMIZER`` symbols once, at import time, based on
      whichever of TE / Apex / stock Torch is importable. We override those
      module attributes directly (binding replacement) -- the private
      ``_get_megatron_optimizer_based_on_param_groups`` reads
      ``USING_PYTORCH_OPTIMIZER`` as a bare module global at call time, so
      reassigning it is enough to route construction through
      ``torch.optim.AdamW`` for the main (non CPU-offload) path.
    * The ``fused=True`` kwarg is not otherwise reachable via attribute
      reassignment (it's set inside a local ``kwargs`` dict built partway
      through the function body), so that one line is added via a targeted
      source-string rewrite.
"""

import os

import torch

from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched
from primus.backends.megatron.patches._source_patch_utils import patch_function_source
from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0

_PATCH_KEY = "megatron.optimizer.torch_fused_adam"

_ORI_CODE = "adam_cls = torch.optim.AdamW if config.decoupled_weight_decay else torch.optim.Adam"
_NEW_CODE = _ORI_CODE + '\n                kwargs["fused"] = True'


def _enabled() -> bool:
    return os.environ.get("PRIMUS_TORCH_OPTIM", "0") == "1"


def _install_torch_fused_adam_patch() -> None:
    import megatron.core.optimizer as optimizer_module

    if is_patched(optimizer_module, _PATCH_KEY):
        log_rank_0(f"[Patch:{_PATCH_KEY}] megatron.core.optimizer already patched; skipping.")
        return

    optimizer_module.Adam = torch.optim.AdamW
    optimizer_module.SGD = torch.optim.SGD
    optimizer_module.USING_PYTORCH_OPTIMIZER = True

    patch_function_source(
        optimizer_module,
        "_get_megatron_optimizer_based_on_param_groups",
        _ORI_CODE,
        _NEW_CODE,
    )

    mark_patched(optimizer_module, _PATCH_KEY)
    log_rank_0(
        f"[Patch:{_PATCH_KEY}] PRIMUS_TORCH_OPTIM=1 -> using torch.optim.AdamW(fused=True) "
        "instead of TE/Apex FusedAdam."
    )


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description=(
        "Opt-in: use torch.optim.AdamW(fused=True) instead of TE/Apex FusedAdam, "
        "to bit-match FLA's reference optimizer for reproducibility experiments."
    ),
    priority=40,
    condition=lambda ctx: _enabled(),
)
def patch_torch_fused_adam(ctx: PatchContext) -> None:
    _install_torch_fused_adam_patch()
