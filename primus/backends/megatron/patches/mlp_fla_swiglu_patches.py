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
saving ~20 ms/iter on GDN/KDA/Mamba hybrid training.

Only takes effect for plain, unclamped SwiGLU: ``config.gated_linear_unit`` with
``F.silu``, no ``use_te_activation_func`` module slot, no
``activation_func_clamp_value`` and no ``glu_linear_offset``. ``F.silu`` on its
own is not a sufficient test -- Kimi K3 keeps that config value only to satisfy
the whitelist in ``TransformerConfig.__post_init__`` while supplying the real,
soft-clamped ``SituActivation`` through the module slot, and DeepSeek-V4 style
configs reach the eager ``glu()`` with a clamp bound. FLA's kernels implement
neither, so those configs fall through to Megatron's own code, as do GeLU-based
MLPs and MoE.

Toggle: ``args.use_fla_fused_swiglu`` (resolved by ``fla_runtime_patches.py``
from ``PRIMUS_FLA_SWIGLU`` / YAML ``use_fla_fused_swiglu``, default True).

Memory variant: ``args.use_fla_fused_swiglu_linear`` (default False) goes one
step further and fuses the activation *into* ``linear_fc2`` via FLA's
``swiglu_linear``. Megatron computes ``swiglu`` as its own op and then feeds the
result to ``linear_fc2``, which saves that ffn-wide tensor for its weight
gradient; ``swiglu_linear`` instead saves only the two fc1 halves and recomputes
the activation in the backward. That removes one ffn-wide tensor per MLP layer
(measured 0.188 GiB per unit micro-batch on GDN-300M, ffn 4096, seq 2048 -- the
entire measured activation gap vs FLA).

Correctness note: this bypasses ``linear_fc2``'s
``linear_with_grad_accumulation_and_async_allreduce``, so the fc2 weight
gradient lands in ``param.grad`` instead of being fused straight into
``param.main_grad``. That is safe -- Megatron's DDP backward post-hook folds
``param.grad`` into ``main_grad`` whenever ``grad_added_to_main_grad`` is False
-- but it does forgo the wgrad-accumulation fusion, so the path is gated to the
simple case it was measured on: TP=1, no sequence parallel, no bias, no
per-token scaling, non-expert MLPs.

Source-string rewrite style: the injection points sit inside ``MLP.__init__``
and the non-fused branch of ``MLP.forward``, which a plain wrapper cannot
reach without duplicating the whole (large) method body.
"""

from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched
from primus.backends.megatron.patches._source_patch_utils import (
    patch_method_source,
    patch_method_source_multi,
)
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

_PATCH_KEY = "megatron.mlp.fla_swiglu"

# `activation_func == F.silu` alone does not mean "plain SwiGLU". Kimi K3 keeps
# that value only to satisfy the whitelist in TransformerConfig.__post_init__
# while routing the real, soft-clamped activation through the `activation_func`
# module slot (`use_te_activation_func`, mlp.py:226); DeepSeek-V4 style configs
# instead reach the eager `glu()` with a clamp bound and/or a non-zero
# `glu_linear_offset`. FLA's kernels implement none of that, so both fused paths
# below are restricted to the plain, unclamped, zero-offset case and everything
# else falls through to Megatron's own code.
_PLAIN_SWIGLU_COND = (
    "self.config.gated_linear_unit\n"
    "                    and self.config.activation_func == F.silu\n"
    "                    and not getattr(self.config, 'use_te_activation_func', False)\n"
    "                    and getattr(self.config, 'activation_func_clamp_value', None) is None\n"
    "                    and not getattr(self.config, 'glu_linear_offset', 0)"
)

# Inserted right before linear_fc2 construction in MLP.__init__, mirroring
# where megatron_patches/03-mlp-fla-swiglu.patch spliced in its detection.
_INIT_ORI = "self.linear_fc2 = submodules.linear_fc2("
_INIT_NEW = (
    "from megatron.training import get_args as _get_args\n"
    "        self._use_fla_swiglu = False\n"
    "        if getattr(_get_args(), 'use_fla_fused_swiglu', True):\n"
    "            if (" + _PLAIN_SWIGLU_COND + "):\n"
    "                try:\n"
    "                    from fla.modules.activations import swiglu as _fla_swiglu\n"
    "                    self._use_fla_swiglu = True\n"
    "                    self._fla_swiglu_fn = _fla_swiglu\n"
    "                except ImportError:\n"
    "                    pass\n"
    "        self._use_fla_swiglu_linear = False\n"
    "        if getattr(_get_args(), 'use_fla_fused_swiglu_linear', False):\n"
    "            if (" + _PLAIN_SWIGLU_COND + "\n"
    "                    and self.config.tensor_model_parallel_size == 1\n"
    "                    and not self.config.sequence_parallel\n"
    "                    and not self.config.add_bias_linear\n"
    "                    and not is_expert):\n"
    "                try:\n"
    "                    from fla.modules.activations import swiglu_linear as _fla_swiglu_linear\n"
    "                    self._use_fla_swiglu_linear = True\n"
    "                    self._fla_swiglu_linear_fn = _fla_swiglu_linear\n"
    "                except ImportError:\n"
    "                    pass\n"
    "\n        " + _INIT_ORI
)

# Inserted at the top of MLP.forward's activation section: when the fused
# swiglu+fc2 path is active we short-circuit both the activation and linear_fc2,
# so the ffn-wide activation output is never saved for backward.
_FWD_FUSED_ORI = 'nvtx_range_push(suffix="activation")'
_FWD_FUSED_NEW = (
    "if (getattr(self, '_use_fla_swiglu_linear', False)\n"
    "                and per_token_scale is None and bias_parallel is None):\n"
    "            x_glu, x_linear = torch.chunk(intermediate_parallel, 2, dim=-1)\n"
    "            output = self._fla_swiglu_linear_fn(\n"
    "                x_glu, x_linear, self.linear_fc2.weight, None\n"
    "            )\n"
    "            return output, None\n"
    "        " + _FWD_FUSED_ORI
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
    # Both forward rewrites must go in one pass: a method can only be
    # source-patched once (inspect.getsource cannot re-read an exec'd function).
    patch_method_source_multi(
        MLP,
        "forward",
        [(_FWD_FUSED_ORI, _FWD_FUSED_NEW), (_FORWARD_ORI, _FORWARD_NEW)],
    )

    mark_patched(MLP, _PATCH_KEY)
    log_rank_0(
        f"[Patch:{_PATCH_KEY}] Patched MLP.__init__/forward to use FLA's Triton "
        "swiglu kernel when use_fla_fused_swiglu is set, and to fuse swiglu into "
        "linear_fc2 when use_fla_fused_swiglu_linear is set."
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
    condition=lambda ctx: getattr(get_args(ctx), "use_fla_fused_swiglu", False)
    or getattr(get_args(ctx), "use_fla_fused_swiglu_linear", False),
)
def patch_mlp_fla_swiglu(ctx: PatchContext) -> None:
    _install_mlp_fla_swiglu_patch()
