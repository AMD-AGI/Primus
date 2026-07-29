###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MLP FLA SwiGLU Patch
======================

Routes ``MLP``'s gated-linear-unit activation through
flash-linear-attention's (FLA) Triton-fused ``swiglu`` kernel (one fwd + one
bwd kernel) instead of Megatron's naive two-kernel ``silu(x_glu) * x_linear``,
saving ~20 ms/iter on GDN/KDA/Mamba hybrid training. Only takes effect when
``config.gated_linear_unit`` and the activation is ``F.silu`` -- i.e. it is a
no-op for GeLU-based MLPs, MoE, and any config where
``use_te_activation_func`` / ``bias_activation_fusion`` are already handling
the activation through a different (already-fused) code path.

Toggle: ``args.use_fla_fused_swiglu`` (resolved by ``fla_runtime_patches.py``
from ``PRIMUS_FLA_SWIGLU`` / YAML ``use_fla_fused_swiglu``, default True).

Source-string rewrite style: the injection points sit inside ``MLP.__init__``
and the non-fused branch of ``MLP.forward``, which a plain wrapper cannot
reach without duplicating the whole (large) method body.
"""

from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched
from primus.backends.megatron.patches._source_patch_utils import patch_method_source
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

_PATCH_KEY = "megatron.mlp.fla_swiglu"

# Inserted right before linear_fc2 construction in MLP.__init__, mirroring
# where megatron_patches/03-mlp-fla-swiglu.patch spliced in its detection.
_INIT_ORI = "self.linear_fc2 = submodules.linear_fc2("
_INIT_NEW = (
    "from megatron.training import get_args as _get_args\n"
    "        self._use_fla_swiglu = False\n"
    "        if getattr(_get_args(), 'use_fla_fused_swiglu', True):\n"
    "            if self.config.gated_linear_unit and self.config.activation_func == F.silu:\n"
    "                try:\n"
    "                    from fla.modules.activations import swiglu as _fla_swiglu\n"
    "                    self._use_fla_swiglu = True\n"
    "                    self._fla_swiglu_fn = _fla_swiglu\n"
    "                except ImportError:\n"
    "                    pass\n"
    "\n        " + _INIT_ORI
)

# Inserted in MLP.forward's non-fused-bias-activation branch, in place of the
# unconditional `glu(intermediate_parallel)` call.
_FORWARD_ORI = "intermediate_parallel = glu(intermediate_parallel)"
_FORWARD_NEW = (
    "if self._use_fla_swiglu:\n"
    "                    x_glu, x_linear = torch.chunk(intermediate_parallel, 2, dim=-1)\n"
    "                    intermediate_parallel = self._fla_swiglu_fn(x_glu, x_linear)\n"
    "                else:\n"
    "                    " + _FORWARD_ORI
)


def _install_mlp_fla_swiglu_patch() -> None:
    from megatron.core.transformer.mlp import MLP

    if is_patched(MLP, _PATCH_KEY):
        log_rank_0(f"[Patch:{_PATCH_KEY}] MLP already patched; skipping.")
        return

    patch_method_source(MLP, "__init__", _INIT_ORI, _INIT_NEW)
    patch_method_source(MLP, "forward", _FORWARD_ORI, _FORWARD_NEW)

    mark_patched(MLP, _PATCH_KEY)
    log_rank_0(
        f"[Patch:{_PATCH_KEY}] Patched MLP.__init__/forward to use FLA's Triton "
        "swiglu kernel when use_fla_fused_swiglu is set."
    )


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description=(
        "Route MLP's gated-linear-unit activation through FLA's Triton-fused "
        "swiglu kernel instead of Megatron's naive silu*x implementation."
    ),
    # Runs after fla_runtime_knobs (priority=-100) has resolved args.use_fla_fused_swiglu.
    priority=50,
    condition=lambda ctx: getattr(get_args(ctx), "use_fla_fused_swiglu", False),
)
def patch_mlp_fla_swiglu(ctx: PatchContext) -> None:
    _install_mlp_fla_swiglu_patch()
