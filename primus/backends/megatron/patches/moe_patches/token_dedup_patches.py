###############################################################################
# Copyright (c) 2026, Zyphra Technologies Inc. All rights reserved.
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MoE token dedup and co-activation expert placement patches.

Port of ROCm/Megatron-LM PR #165 (``--moe-token-dedup``) and PR #166
(``--moe-expert-placement``) as Primus patches over Megatron-core v0.19's
all-to-all token dispatcher. ``third_party/Megatron-LM`` is not modified; the
dispatcher subclasses live in
``primus/backends/megatron/core/transformer/moe/token_dedup_dispatcher.py``.

Config keys (``primus/configs/modules/megatron/primus_megatron_module.yaml``):

* ``moe_token_dedup`` -- one All-to-All row per unique (token, destination EP
  rank) instead of one per (token, expert), in both directions.
* ``moe_token_dedup_weighted_combine`` -- apply the routing weights in the
  dedup combine's reduce kernel instead of inside the experts.
* ``moe_expert_placement`` -- path to a placement JSON (built offline by
  ``tools/moe_expert_placement/build_expert_placement.py``) that reorders which
  global expert lives on which EP rank, globally or per layer.

Patches:

* ``megatron.moe.token_dedup_args`` (build_args): validates the flags against
  the dispatcher / routing settings and parses the placement JSON (validation
  only; the table is re-loaded from the path at ``before_train``, since a dict
  with integer layer keys cannot live on the Primus params namespace).
* ``megatron.moe.token_dedup`` (before_train): exposes the options on
  ``TransformerConfig``, binds the Primus dispatchers in place of
  ``MoEAlltoAllTokenDispatcher`` and wires the (per-layer) placement through
  ``BaseMoELayer`` / ``MoELayer.set_layer_number``.
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


def _placement_path(args):
    return getattr(args, "moe_expert_placement", None) or None


def _feature_enabled(ctx: PatchContext) -> bool:
    args = get_args(ctx)
    return _dedup_enabled(args) or _placement_path(args) is not None


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
# build_args: validation and placement parsing
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
    description=(
        "Validate moe_token_dedup / moe_expert_placement against the MoE dispatcher and "
        "routing settings; parse the expert placement JSON"
    ),
    condition=_feature_enabled,
)
def patch_token_dedup_args(ctx: PatchContext):
    from primus.backends.megatron.core.transformer.moe.expert_placement import load_moe_expert_placement

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

    path = args.get("moe_expert_placement", None) or None
    if path is not None:
        _validate_common(args, "moe_expert_placement")
        # Validate now (fail early); the table itself is re-loaded at before_train.
        table = load_moe_expert_placement(
            path, args.get("num_experts", None), int(args.get("expert_model_parallel_size", 1) or 1)
        )
        log_rank_0(
            f"{_PREFIX} moe_expert_placement loaded from {path}: "
            f"default={'yes' if table['default'] is not None else 'no'}, "
            f"per-layer entries={len(table['layers'])}"
        )


# --------------------------------------------------------------------------- #
# before_train: config exposure, dispatcher binding, placement wiring
# --------------------------------------------------------------------------- #


def _set_experts_placement(experts, local_expert_indices) -> None:
    """Record the true global expert ids on the experts module (checkpoint expert-axis
    offsets) and on its grouped linear children (TE grouped GEMM keys)."""
    if experts is None:
        return
    experts.local_expert_indices = list(local_expert_indices)
    for name in ("linear_fc1", "linear_fc2"):
        child = getattr(experts, name, None)
        if child is not None:
            child.local_expert_indices = list(local_expert_indices)


def _resolve_layer_placement(layer) -> None:
    from primus.backends.megatron.core.transformer.moe.expert_placement import compute_layer_expert_placement

    layer.expert_order, layer.local_expert_indices = compute_layer_expert_placement(
        layer.config, layer.layer_number, layer.ep_size, layer.ep_rank
    )
    assert all(x < layer.config.num_moe_experts for x in layer.local_expert_indices)


def _propagate_layer_placement(layer) -> None:
    dispatcher = getattr(layer, "token_dispatcher", None)
    if dispatcher is not None and hasattr(dispatcher, "set_expert_order"):
        dispatcher.set_expert_order(layer.expert_order, layer.local_expert_indices)
    _set_experts_placement(getattr(layer, "experts", None), layer.local_expert_indices)


@register_patch(
    "megatron.moe.token_dedup",
    backend="megatron",
    phase="before_train",
    description=(
        "Bind the token-dedup / expert-placement all-to-all dispatchers in place of "
        "MoEAlltoAllTokenDispatcher and wire the per-layer expert placement through MoELayer"
    ),
    condition=_feature_enabled,
)
def patch_token_dedup(ctx: PatchContext):
    import megatron.core.transformer.transformer_config as config_mod
    from megatron.core import utils
    from megatron.core.transformer.moe import moe_layer as moe_layer_mod

    from primus.backends.megatron.core.transformer.moe.expert_placement import load_moe_expert_placement
    from primus.backends.megatron.core.transformer.moe.token_dedup_dispatcher import (
        PrimusAlltoAllTokenDispatcher,
        PrimusDedupAlltoAllTokenDispatcher,
        build_alltoall_token_dispatcher,
    )

    args = get_args(ctx)
    placement_path = getattr(args, "moe_expert_placement", None) or None
    placement_table = None
    if placement_path is not None:
        placement_table = load_moe_expert_placement(
            placement_path,
            getattr(args, "num_experts", None),
            int(getattr(args, "expert_model_parallel_size", 1) or 1),
        )

    # 1. Expose the options on TransformerConfig (class attributes: the dataclass is built
    #    from args by field name, so Primus-only keys never reach the instance otherwise).
    config_mod.TransformerConfig.moe_token_dedup = _dedup_enabled(args)
    config_mod.TransformerConfig.moe_token_dedup_weighted_combine = bool(
        getattr(args, "moe_token_dedup_weighted_combine", False)
    )
    config_mod.TransformerConfig.moe_expert_placement = placement_table
    log_rank_0(
        f"{_PREFIX} TransformerConfig.moe_token_dedup={config_mod.TransformerConfig.moe_token_dedup}, "
        f"moe_token_dedup_weighted_combine={config_mod.TransformerConfig.moe_token_dedup_weighted_combine}, "
        f"moe_expert_placement={'set' if placement_table is not None else None}"
    )

    # 2. Dispatcher binding. MoELayer.__init__ looks the class up as a module global.
    moe_layer_mod.MoEAlltoAllTokenDispatcher = build_alltoall_token_dispatcher
    log_rank_0(
        f"{_PREFIX} Patched megatron.core.transformer.moe.moe_layer.MoEAlltoAllTokenDispatcher -> "
        f"{PrimusDedupAlltoAllTokenDispatcher.__name__ if _dedup_enabled(args) else PrimusAlltoAllTokenDispatcher.__name__}"
    )

    # 3. Placement wiring on the MoE layer (no-op for the contiguous default).
    BaseMoELayer = moe_layer_mod.BaseMoELayer
    MoELayer = moe_layer_mod.MoELayer
    if is_patched(BaseMoELayer, "megatron.moe.token_dedup"):
        return

    orig_base_init = BaseMoELayer.__init__
    orig_moe_init = MoELayer.__init__
    orig_set_layer_number = BaseMoELayer.set_layer_number

    def base_init(self, *a, **kw):
        orig_base_init(self, *a, **kw)
        self.ep_size = utils.get_pg_size(self.ep_group)
        self.ep_rank = utils.get_pg_rank(self.ep_group)
        # The layer number is usually unknown here (TransformerLayer calls set_layer_number()
        # afterwards); resolve with what we have, refine in set_layer_number().
        _resolve_layer_placement(self)

    def moe_init(self, *a, **kw):
        orig_moe_init(self, *a, **kw)
        _propagate_layer_placement(self)

    def set_layer_number(self, layer_number):
        orig_set_layer_number(self, layer_number)
        _resolve_layer_placement(self)
        _propagate_layer_placement(self)

    BaseMoELayer.__init__ = base_init
    MoELayer.__init__ = moe_init
    BaseMoELayer.set_layer_number = set_layer_number
    mark_patched(BaseMoELayer, "megatron.moe.token_dedup")
    log_rank_0(
        f"{_PREFIX} Patched BaseMoELayer.__init__/set_layer_number and MoELayer.__init__ for expert placement"
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
