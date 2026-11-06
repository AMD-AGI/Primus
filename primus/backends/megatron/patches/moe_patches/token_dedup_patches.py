###############################################################################
# Copyright (c) 2026, Zyphra Technologies Inc. All rights reserved.
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MoE token dedup patches.

Port of ROCm/Megatron-LM PR #165 (``--moe-token-dedup``) as Primus patches over
Megatron-core v0.19's all-to-all token dispatcher. ``third_party/Megatron-LM``
is not modified; the dispatcher subclass lives in
``primus/backends/megatron/core/transformer/moe/token_dedup_dispatcher.py``.

Config keys (``primus/configs/modules/megatron/primus_megatron_module.yaml``):

* ``moe_token_dedup`` -- one All-to-All row per unique (token, destination EP
  rank) instead of one per (token, expert), in both directions.
* ``moe_token_dedup_weighted_combine`` -- apply the routing weights in the
  dedup combine's reduce kernel instead of inside the experts.

Patches:

* ``megatron.moe.token_dedup_args`` (build_args): validates the flags against
  the dispatcher / routing settings.
* ``megatron.moe.token_dedup`` (before_train): exposes the options on
  ``TransformerConfig`` and binds the Primus dispatcher in place of
  ``MoEAlltoAllTokenDispatcher``.
* ``megatron.moe.token_dedup_experts_probs`` (before_train): lets the expert
  MLPs accept ``permuted_probs=None`` for the weighted combine.
"""

from __future__ import annotations

from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0

_PREFIX = "[Patch:megatron.moe.token_dedup]"


def _dedup_enabled(args) -> bool:
    return bool(getattr(args, "moe_token_dedup", False))


def _feature_enabled(ctx: PatchContext) -> bool:
    return _dedup_enabled(get_args(ctx))


def _weighted_combine_enabled(ctx: PatchContext) -> bool:
    args = get_args(ctx)
    return _dedup_enabled(args) and bool(getattr(args, "moe_token_dedup_weighted_combine", False))


class _ArgsView:
    """Read-through view over the Primus params and the Megatron backend args.

    In the ``build_args`` phase ``get_args(ctx)`` is the Primus module params (only
    the keys the YAML set), while ``ctx.extra['backend_args']`` carries every
    Megatron argument with its default. Reads try the params first, then the
    backend args; writes go to both so new keys survive the post-phase merge.
    """

    def __init__(self, ctx: PatchContext):
        self._params = get_args(ctx)
        self._backend = ctx.extra.get("backend_args") or None

    def get(self, name, default=None):
        for ns in (self._params, self._backend):
            if ns is not None and hasattr(ns, name):
                return getattr(ns, name)
        return default

    def set(self, name, value) -> None:
        for ns in (self._params, self._backend):
            if ns is not None:
                setattr(ns, name, value)


# --------------------------------------------------------------------------- #
# build_args: validation
# --------------------------------------------------------------------------- #


def _validate_common(args: _ArgsView, feature: str) -> None:
    dispatcher = args.get("moe_token_dispatcher_type", "allgather")
    if dispatcher != "alltoall":
        raise ValueError(f"{feature} only supports moe_token_dispatcher_type: alltoall, got '{dispatcher}'.")
    if args.get("use_turbo_deepep", False):
        raise ValueError(
            f"{feature} uses the all-to-all dispatcher; use_turbo_deepep switches the model to "
            "the DeepEP ('flex') dispatcher. Set use_turbo_deepep: false."
        )
    if args.get("use_turbo_mega_moe", False):
        raise ValueError(f"{feature} is not supported with the fused MegaMoE layer (use_turbo_mega_moe).")
    if args.get("use_deprecated_20241209_moe_layer", False):
        raise ValueError(f"{feature} is not supported with the deprecated 20241209 MoE layer.")


@register_patch(
    "megatron.moe.token_dedup_args",
    backend="megatron",
    phase="build_args",
    description="Validate moe_token_dedup against the MoE dispatcher and routing settings",
    condition=_feature_enabled,
)
def patch_token_dedup_args(ctx: PatchContext):
    args = _ArgsView(ctx)

    if args.get("moe_token_dedup", False):
        _validate_common(args, "moe_token_dedup")
        if args.get("moe_expert_capacity_factor", None) is not None or args.get(
            "moe_pad_expert_input_to_capacity", False
        ):
            raise ValueError(
                "moe_token_dedup supports dropless routing only; remove moe_expert_capacity_factor "
                "and moe_pad_expert_input_to_capacity."
            )
        if args.get("moe_router_padding_for_quantization", False):
            raise ValueError(
                "moe_token_dedup requires exactly top-k routing rows; disable moe_router_padding_for_quantization."
            )
        etp = args.get("expert_tensor_parallel_size", None)
        if etp not in (None, 1):
            raise ValueError(
                f"moe_token_dedup does not support expert tensor parallelism (expert_tensor_parallel_size={etp})."
            )
        if args.get("cuda_graph_impl", "none") not in (None, "none"):
            raise ValueError("moe_token_dedup does not support CUDA graphs (cuda_graph_impl must be 'none').")
        log_rank_0(
            f"{_PREFIX} moe_token_dedup enabled (weighted_combine={bool(args.get('moe_token_dedup_weighted_combine', False))})"
        )


# --------------------------------------------------------------------------- #
# before_train: config exposure and dispatcher binding
# --------------------------------------------------------------------------- #


@register_patch(
    "megatron.moe.token_dedup",
    backend="megatron",
    phase="before_train",
    description="Bind the token-dedup all-to-all dispatcher in place of MoEAlltoAllTokenDispatcher",
    condition=_feature_enabled,
)
def patch_token_dedup(ctx: PatchContext):
    import megatron.core.transformer.transformer_config as config_mod
    from megatron.core.transformer.moe import moe_layer as moe_layer_mod

    from primus.backends.megatron.core.transformer.moe.token_dedup_dispatcher import (
        PrimusDedupAlltoAllTokenDispatcher,
        build_alltoall_token_dispatcher,
    )

    args = get_args(ctx)

    # 1. Expose the options on TransformerConfig (class attributes: the dataclass is built
    #    from args by field name, so Primus-only keys never reach the instance otherwise).
    config_mod.TransformerConfig.moe_token_dedup = _dedup_enabled(args)
    config_mod.TransformerConfig.moe_token_dedup_weighted_combine = bool(
        getattr(args, "moe_token_dedup_weighted_combine", False)
    )
    log_rank_0(
        f"{_PREFIX} TransformerConfig.moe_token_dedup={config_mod.TransformerConfig.moe_token_dedup}, "
        f"moe_token_dedup_weighted_combine={config_mod.TransformerConfig.moe_token_dedup_weighted_combine}"
    )

    # 2. Dispatcher binding. MoELayer.__init__ looks the class up as a module global.
    moe_layer_mod.MoEAlltoAllTokenDispatcher = build_alltoall_token_dispatcher
    log_rank_0(
        f"{_PREFIX} Patched megatron.core.transformer.moe.moe_layer.MoEAlltoAllTokenDispatcher -> "
        f"{PrimusDedupAlltoAllTokenDispatcher.__name__}"
    )


# --------------------------------------------------------------------------- #
# before_train: experts accept permuted_probs=None (weighted combine)
# --------------------------------------------------------------------------- #


def _wrap_forward_probs_optional(cls, label: str) -> None:
    """Make ``cls.forward`` accept ``permuted_probs=None`` by substituting unit weights.

    With the dedup weighted combine the routing weights are applied in the combine's
    reduce kernel, so the experts must not weight their rows. Megatron-core's expert
    MLPs unconditionally multiply by ``permuted_probs``; a ones tensor keeps every
    (fused or unfused) activation path intact at the cost of one elementwise multiply.
    """
    if is_patched(cls, "megatron.moe.token_dedup_experts_probs"):
        return
    orig_forward = cls.forward

    def forward(self, permuted_local_hidden_states, tokens_per_expert, permuted_probs=None, *a, **kw):
        if permuted_probs is None:
            permuted_probs = permuted_local_hidden_states.new_ones((permuted_local_hidden_states.shape[0],))
        return orig_forward(self, permuted_local_hidden_states, tokens_per_expert, permuted_probs, *a, **kw)

    cls.forward = forward
    mark_patched(cls, "megatron.moe.token_dedup_experts_probs")
    log_rank_0(f"{_PREFIX} Patched {label}.forward to accept permuted_probs=None (dedup weighted combine)")


@register_patch(
    "megatron.moe.token_dedup_experts_probs",
    backend="megatron",
    phase="before_train",
    description="Let TEGroupedMLP / PrimusGroupedMLP / SequentialMLP run with permuted_probs=None for the dedup weighted combine",
    condition=_weighted_combine_enabled,
)
def patch_token_dedup_experts_probs(ctx: PatchContext):
    del ctx
    from megatron.core.transformer.moe import experts as experts_mod

    _wrap_forward_probs_optional(experts_mod.TEGroupedMLP, "TEGroupedMLP")
    _wrap_forward_probs_optional(experts_mod.SequentialMLP, "SequentialMLP")
    try:
        from primus.backends.megatron.core.transformer.experts import PrimusGroupedMLP
    except Exception as exc:  # noqa: BLE001
        warning_rank_0(f"{_PREFIX} PrimusGroupedMLP not importable ({exc}); not patched")
        return
    _wrap_forward_probs_optional(PrimusGroupedMLP, "PrimusGroupedMLP")
