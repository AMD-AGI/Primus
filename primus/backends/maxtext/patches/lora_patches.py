###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MaxText LoRA mesh patch (ROCm/maxtext 4270c96a1).

Flax 0.12.2 made ``get_global_mesh()`` read the mesh set by ``jax.set_mesh``
rather than the legacy ``with Mesh(...)`` context, so unboxing the restored LoRA
params in ``lora_utils.setup_initial_lora_state`` has to run under
``jax.set_mesh(mesh)``.

The same commit also touches ``src/maxtext/examples/shardings.py``, a standalone
example script outside the training path; that part is not ported.
"""

from primus.backends.maxtext.patches._source_patch_utils import (
    any_in_source,
    rewrite_source,
)
from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0

_ANCHOR = "        lora_state = max_utils.unbox_logicallypartioned(lora_state)\n"
_NEW = (
    "        with jax.set_mesh(mesh):\n"
    "          lora_state = max_utils.unbox_logicallypartioned(lora_state)\n"
)


@register_patch(
    patch_id="maxtext.lora.set_mesh_unbox",
    backend="maxtext",
    phase="setup",
    description="Unbox restored LoRA params under jax.set_mesh(mesh) (flax 0.12.2 get_global_mesh regression)",
    condition=lambda ctx: True,
)
def patch_lora_set_mesh(ctx: PatchContext) -> None:
    del ctx
    prefix = "[Patch:maxtext.lora.set_mesh_unbox]"
    try:
        from maxtext.utils import lora_utils
    except ImportError as e:
        warning_rank_0(f"{prefix} maxtext.utils.lora_utils not importable; skipped: {e}")
        return
    if any_in_source(lora_utils, "setup_initial_lora_state", [_NEW]):
        log_rank_0(f"{prefix} setup_initial_lora_state already unboxes under jax.set_mesh; not patched.")
        return
    rewrite_source(lora_utils, "setup_initial_lora_state", [(_ANCHOR, _NEW)])
    log_rank_0(f"{prefix} setup_initial_lora_state unboxes LoRA params under jax.set_mesh(mesh).")
