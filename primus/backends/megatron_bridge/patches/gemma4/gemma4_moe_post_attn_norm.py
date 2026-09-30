###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Restore Gemma 4 MoE's post-attention norm when the layers are built without TE.

On the Gemma 4 **MoE** model the post-attention layernorm is attached only when
the block spec is built with ``use_transformer_engine=True``. Built with
``False``, the model has no post-attention normalisation anywhere, and nothing
warns. ``_gemma4_block_spec`` (``modeling_gemma4.py:1517``) patches
``core_attention`` unconditionally but puts the norm-carrying projection behind a
guard::

    attn_spec.submodules.core_attention = Gemma4TEDotProductAttention   # always
    if use_transformer_engine:
        attn_spec.submodules.linear_proj = TERowParallelLinearLayerNorm  # TE only

``TERowParallelLinearLayerNorm`` is ``TERowParallelLinear`` plus a
``post_layernorm`` on the output -- it *is* Gemma's post-attention norm, and its
own docstring names Gemma2/Gemma3 as the models that need it. With the guard
false ``linear_proj`` stays a plain ``RowParallelLinear``, which applies none.

Why there is no fallback
------------------------
The Gemma-4 *Dense* path defines a dedicated submodules dataclass carrying
``post_self_attn_layernorm`` / ``post_mlp_layernorm`` /
``post_per_layer_input_norm`` as real fields, which
``Gemma4DenseTransformerLayer.forward`` applies. The **MoE** layer
(``Gemma4TransformerLayer``) takes the *generic* ``TransformerLayerSubmodules``,
which has no such field, and its ``__init__`` adds only
``pre_shared_expert_layernorm`` and ``post_ffn_layernorm``. So on the MoE model
that TE-only ``linear_proj`` is the sole carrier of the post-attention norm, and
simply adding the submodule field upstream would not be enough -- the MoE
layer's forward would also have to call it.

This patch therefore mirrors what TE does rather than inventing a third
mechanism: it supplies a local ``RowParallelLinear`` that applies the same norm
to its output. Same mathematics, different kernel, which is the relationship the
two paths are supposed to have.

Opt-in, deliberately
--------------------
This changes the model's mathematics, so it is gated on
``PRIMUS_GEMMA4_MOE_POST_ATTN_NORM=1`` and does nothing by default. Two reasons:
a silent change here would invalidate comparisons against numbers already
measured without it, and keeping it explicit makes the A/B measurable -- the same
config with and without the flag isolates exactly what the missing norm costs.

Measured context: on a 6-layer 26B proxy the two paths diverge from the first
optimizer step, 15.12 (TE) against 18.25 (local). Initialisation on this stack is
deterministic -- a repeated run reproduced its loss bitwise -- so that offset is a
real implementation difference rather than noise. How much of it this norm
accounts for is exactly what the flag is for; it is not yet established.

Checkpoint note: this adds a ``post_layernorm.weight`` per layer, so a
checkpoint written with the flag set will not load without it, and vice versa.
The same is true of the TE path relative to plain local, since that is where the
parameter comes from in the first place.
"""

from __future__ import annotations

import os
from typing import Any

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0

_PATCHED_ATTR = "_primus_gemma4_moe_post_attn_norm_patched"
_ENV_VAR = "PRIMUS_GEMMA4_MOE_POST_ATTN_NORM"


def enabled() -> bool:
    """True when the caller asked for the post-attention norm to be restored."""
    return os.environ.get(_ENV_VAR) == "1"


def _build_norm_carrying_row_parallel_linear() -> type | None:
    """Return a local RowParallelLinear that post-norms its output, or None.

    Built lazily rather than at import time: megatron and the Gemma 4 modelling
    module are both heavy, and this patch is a no-op unless the flag is set.
    """
    try:
        from megatron.bridge.models.gemma.modeling_gemma4 import Gemma4RMSNorm
        from megatron.core.tensor_parallel.layers import RowParallelLinear
    except Exception as exc:  # pragma: no cover - import shape depends on install
        log_rank_0(f"[Patch:gemma4.moe_post_attn_norm] cannot import the pieces needed: {exc}")
        return None

    class RowParallelLinearLayerNorm(RowParallelLinear):
        """Row-parallel linear with a post-norm on the output.

        The local counterpart of Bridge's ``TERowParallelLinearLayerNorm``.
        ``Gemma4RMSNorm`` is used rather than megatron's generic ``RMSNorm``
        because it is the HF-Gemma-4 expression (``pow(-0.5)`` instead of
        ``rsqrt``, weight applied multiplicatively over a ones-initialised
        parameter), which is what the rest of the MoE layer's own norms use.
        """

        def __init__(self, input_size: int, output_size: int, *, config, **kwargs):
            super().__init__(input_size, output_size, config=config, **kwargs)
            self.post_layernorm = Gemma4RMSNorm(
                config,
                output_size,
                eps=config.layernorm_epsilon,
            )

        def forward(self, input_):
            output, bias = super().forward(input_)
            if bias is not None:
                # Matches TERowParallelLinearLayerNorm's refusal: normalising
                # before a deferred bias add would apply the norm to the wrong
                # quantity. Gemma sets add_bias_linear=False, so this should not
                # trigger; it is here so that it fails loudly if that changes.
                raise ValueError(
                    "RowParallelLinearLayerNorm assumes add_bias_linear=False. "
                    "Post-norm before a deferred bias addition is incorrect."
                )
            return self.post_layernorm(output), bias

    return RowParallelLinearLayerNorm


def _wrap_spec(spec: Any, norm_cls: type) -> Any:
    """Wrap a block-spec callable so local layers get the post-attention norm.

    The spec is a callable invoked later to build the block, so the norm cannot
    be attached now -- the layer specs do not exist yet. Wrapping and
    post-processing the result is what keeps this independent of how the spec was
    constructed.
    """

    def build(config, *args, **kwargs):
        block_spec = spec(config, *args, **kwargs)

        replaced = 0
        carrying = 0
        total = 0
        for layer_spec in getattr(block_spec, "layer_specs", ()):
            total += 1
            submodules = getattr(layer_spec, "submodules", None)
            attn_spec = getattr(submodules, "self_attention", None)
            attn_submodules = getattr(attn_spec, "submodules", None)
            if attn_submodules is None:
                continue

            # The Gemma-4 *Dense* layer carries this norm as a real submodule
            # field that its own forward applies, so adding a norm-carrying
            # projection there would normalise twice. Only the MoE layer, which
            # uses the generic TransformerLayerSubmodules and has no such field,
            # is missing it. Checked explicitly rather than relying on the dense
            # spec happening not to be a block spec with layer_specs.
            existing = getattr(submodules, "post_self_attn_layernorm", None)
            if existing is not None and getattr(existing, "__name__", "") != "IdentityOp":
                carrying += 1
                continue

            current = getattr(attn_submodules, "linear_proj", None)
            # Only replace the plain local projection. If it already carries a
            # norm -- TE's variant, or this one on a second pass -- leave it be
            # rather than stacking a second normalisation on the same output.
            if current is None or not isinstance(current, type):
                continue
            if issubclass(current, norm_cls) or "LayerNorm" in current.__name__:
                carrying += 1
                continue

            attn_submodules.linear_proj = norm_cls
            replaced += 1
            carrying += 1

        # Counted separately on purpose. Layer specs commonly *share* one
        # self-attention submodules object, so a single assignment can cover every
        # layer -- reporting only the assignment count would read as though most
        # layers had been missed.
        if carrying:
            log_rank_0(
                f"[Patch:gemma4.moe_post_attn_norm] post-attention norm present on "
                f"{carrying}/{total} layer spec(s) via {norm_cls.__name__} "
                f"({replaced} assignment(s); specs may share one submodules object)"
            )
        else:
            log_rank_0(
                f"[Patch:gemma4.moe_post_attn_norm] no local linear_proj found across "
                f"{total} layer spec(s); nothing to do"
            )
        return block_spec

    setattr(build, _PATCHED_ATTR, True)
    return build


def _apply(container: Any) -> None:
    if not enabled():
        return

    model = getattr(container, "model", None)
    if model is None:
        return

    # Decided on transformer_impl rather than on the spec's current binding, so
    # the answer does not depend on whether gemma4.local_spec has rebound it yet.
    impl = getattr(model, "transformer_impl", None)
    if impl != "local":
        log_rank_0(
            f"[Patch:gemma4.moe_post_attn_norm] {_ENV_VAR}=1 but transformer_impl={impl!r}, "
            "not 'local' -- TransformerEngine supplies this norm itself, leaving it alone"
        )
        return

    spec = getattr(model, "transformer_layer_spec", None)
    if spec is None or getattr(spec, _PATCHED_ATTR, False):
        return

    if not callable(spec):
        log_rank_0(
            f"[Patch:gemma4.moe_post_attn_norm] {_ENV_VAR}=1 but transformer_layer_spec is "
            f"{type(spec).__name__}, which is not callable -- leaving it alone"
        )
        return

    # Secondary safety net for the ordering dependency described on the patch
    # function below: a spec still bound to TE here means gemma4.local_spec has
    # not run, and wrapping now would be pointless because its rebinding would
    # replace this wrapper outright.
    if getattr(spec, "keywords", {}).get("use_transformer_engine") is True:
        log_rank_0(
            f"[Patch:gemma4.moe_post_attn_norm] {_ENV_VAR}=1 and transformer_impl='local', but "
            "the spec is still TE-bound, so gemma4.local_spec has not rebound it yet. Declining "
            "rather than installing a wrapper that would be discarded -- check patch order."
        )
        return

    norm_cls = _build_norm_carrying_row_parallel_linear()
    if norm_cls is None:
        return

    model.transformer_layer_spec = _wrap_spec(spec, norm_cls)
    log_rank_0(
        f"[Patch:gemma4.moe_post_attn_norm] {_ENV_VAR}=1: wrapped the layer spec so local "
        "attention projections post-norm their output (Gemma's post-attention norm)"
    )


@register_patch(
    "gemma4.moe_post_attn_norm",
    backend="megatron_bridge",
    phase="setup",
    description="Restore Gemma 4 MoE's post-attention norm on the non-TE layer spec",
)
def patch_gemma4_moe_post_attn_norm(ctx: PatchContext) -> None:
    try:
        from primus.backends.megatron_bridge import (
            megatron_bridge_posttrain_trainer,
            megatron_bridge_pretrain_trainer,
        )
    except Exception:
        return

    # ORDERING REQUIREMENT: this must register *after* gemma4.local_spec.
    #
    # Both patches wrap load_recipe_config, so the one that registers later wraps
    # outside and runs its own work last. That matters because local_spec
    # *replaces* model.transformer_layer_spec outright when it rebinds to
    # use_transformer_engine=False; if this patch installed its wrapper first,
    # that assignment would throw it away and the norm would silently not appear.
    #
    # The package __init__ imports these modules in alphabetical order, which
    # puts "gemma4_local_spec" before "gemma4_moe_post_attn_norm" and therefore
    # satisfies this by construction. _apply() also declines loudly if it finds a
    # still-TE-bound spec, so a future reordering fails visibly instead of
    # quietly dropping the normalisation.
    for module in (megatron_bridge_pretrain_trainer, megatron_bridge_posttrain_trainer):
        original = getattr(module, "load_recipe_config", None)
        if original is None or getattr(original, _PATCHED_ATTR, False):
            continue

        def load_recipe_config(backend_args, _original=original):
            container = _original(backend_args)
            _apply(container)
            return container

        setattr(load_recipe_config, _PATCHED_ATTR, True)
        module.load_recipe_config = load_recipe_config
