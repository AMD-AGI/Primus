###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark: distributed-checkpoint optimizer state with fp32 params in a bf16 model.

Muse Spark keeps the router weight, ``e_score_correction_bias`` and the residual gates
fp32 in a bf16 model. ``Float16OptimizerWithFloat16Params.sharded_state_dict`` numbers
the optimizer params from ``float16_groups`` only, but torch numbers the optimizer
state over *all* params of every group (fp16 main copies and fp32 params, interleaved).
With an fp32 param in a group the ids shift and the save fails ("Optimizer shape ...
does not match model shape ..." or ``KeyError``). This patch maps ids in the
optimizer's real order; a model with no fp32 params takes the upstream path unchanged.

The non-distributed optimizer is one optimizer without expert parallelism and a chain
of two (dense, experts) with it. Its tensors are stored by parameter key, so they load
in either layout, but the checkpoint's common part (param groups, the shared step) has
the shape of the saving layout and ``ChainedOptimizer.load_state_dict`` then fails
("Expected 2 entries in state dict, but got 3"). The second patch takes the param
groups from the loading optimizer and the step from the checkpoint in that case.
"""

from primus.backends.megatron.core.models.muse_spark.model_types import (
    is_muse_spark_model_type,
)
from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0


def optimizer_order_model_params(optimizer):
    """Model params in the inner optimizer's order: the bf16 model param for each fp32
    main copy, the param itself for an fp32 param; plus, per group, the positions of the
    main copies (the order of ``fp32_from_float16_groups``)."""
    main_to_model = {}
    for model_group, main_group in zip(optimizer.float16_groups, optimizer.fp32_from_float16_groups):
        for model_param, main_param in zip(model_group, main_group):
            main_to_model[id(main_param)] = model_param
    params, fp16_positions = [], []
    for group in optimizer.optimizer.param_groups:
        positions = []
        for j, p in enumerate(group["params"]):
            if id(p) in main_to_model:
                positions.append(j)
                params.append(main_to_model[id(p)])
            else:
                params.append(p)
        fp16_positions.append(positions)
    return params, fp16_positions


def fp16_optimizer_sharded_state_dict(self, model_sharded_state_dict, is_loading=False, metadata=None):
    """``Float16OptimizerWithFloat16Params.sharded_state_dict`` with ids in optimizer order."""
    from megatron.core.dist_checkpointing.optimizer import (
        get_param_id_to_sharded_param_map,
        make_sharded_optimizer_tensor,
        optim_state_to_sharding_state,
    )

    if is_loading:
        self.init_state_fn(self.optimizer, self.config)
    state_dict = self.state_dict()
    params, fp16_positions = optimizer_order_model_params(self)
    id_to_sharded_param_map = get_param_id_to_sharded_param_map(model_sharded_state_dict, params)

    groups = state_dict["optimizer"]["param_groups"]
    assert len(state_dict["fp32_from_fp16_params"]) == len(groups) == len(fp16_positions)
    state_dict["fp32_from_fp16_params"] = [
        [
            make_sharded_optimizer_tensor(
                id_to_sharded_param_map[state_group["params"][j]],
                fp32_param,
                prefix="optimizer.state.fp32_param",
            )
            for j, fp32_param in zip(positions, fp32_group, strict=True)
        ]
        for fp32_group, state_group, positions in zip(
            state_dict["fp32_from_fp16_params"], groups, fp16_positions
        )
    ]
    step = self._extract_common_per_param_step(state_dict["optimizer"])
    optim_state_to_sharding_state(state_dict["optimizer"], id_to_sharded_param_map, exclude_keys="step")
    if step:
        state_dict["optimizer"]["state"]["common_step"] = step
    return state_dict


@register_patch(
    "megatron.muse_spark.fp16_optimizer_fp32_param_ckpt",
    backend="megatron",
    phase="before_train",
    description=(
        "Muse Spark: Float16OptimizerWithFloat16Params.sharded_state_dict maps optimizer state ids "
        "in the optimizer's order (fp32 params kept in a bf16 model shift upstream's ids)."
    ),
    condition=is_muse_spark_model_type,
)
def patch_muse_spark_fp16_optimizer_ckpt(ctx: PatchContext):
    from megatron.core.optimizer import optimizer as opt_module

    cls = opt_module.Float16OptimizerWithFloat16Params
    orig = cls.sharded_state_dict
    if getattr(orig, "_primus_muse_spark", False):
        return

    def sharded_state_dict(self, model_sharded_state_dict, is_loading=False, metadata=None):
        if getattr(self, "is_stub_optimizer", False) or not any(
            getattr(self, "fp32_from_fp32_groups", None) or []
        ):
            return orig(self, model_sharded_state_dict, is_loading, metadata)
        return fp16_optimizer_sharded_state_dict(self, model_sharded_state_dict, is_loading, metadata)

    sharded_state_dict._primus_muse_spark = True
    cls.sharded_state_dict = sharded_state_dict
    log_rank_0(
        "[Patch:megatron.muse_spark.fp16_optimizer_fp32_param_ckpt] optimizer checkpoint ids follow the optimizer's param order"
    )


def _common_step(entry):
    if isinstance(entry, dict):
        state = (entry.get("optimizer") or {}).get("state") or {}
        return state.get("common_step")
    return None


def own_param_groups(optimizer):
    """``optimizer``'s param groups in ``torch.optim.Optimizer.state_dict`` form (params as
    running indices), with every group key (``wd_mult``, ``lr_mult``, ...) that
    ``_filter_and_reorder_param_groups`` matches on. The checkpoint's groups belong to the
    saving layout's optimizers and do not apply."""
    groups, start = [], 0
    for group in optimizer.optimizer.param_groups:
        packed = {k: v for k, v in group.items() if k != "params"}
        packed["params"] = list(range(start, start + len(group["params"])))
        start += len(group["params"])
        groups.append(packed)
    return groups


def _complete_optimizer_state(part, optimizer, step):
    """``part`` (one optimizer's loaded tensors) with ``optimizer``'s own param groups and
    the step the checkpoint (saved in another layout) shares."""
    part = dict(part)
    inner = dict(part.get("optimizer") or {})
    inner["param_groups"] = own_param_groups(optimizer)
    state = dict(inner.get("state") or {})
    if step is not None:  # the saved step wins over the one built for loading
        state["common_step"] = step
    inner["state"] = state
    part["optimizer"] = inner
    return part


def relayout_chained_optimizer_state(chained, state_dict):
    """Per-optimizer state dicts for ``chained`` from a checkpoint saved with another
    number of chained optimizers, or None when the layouts match (upstream path)."""
    if not isinstance(state_dict, dict):
        return None
    n = len(chained.chained_optimizers)
    ints = sorted(k for k in state_dict if isinstance(k, int))
    named = [k for k in state_dict if not isinstance(k, int)]
    if n > 1 and named and ints == list(range(n)):
        # saved by one optimizer: its common part sits at the top level
        step = _common_step(state_dict)
        return [
            _complete_optimizer_state(state_dict[i], opt, step)
            for i, opt in enumerate(chained.chained_optimizers)
        ]
    if n == 1 and ints and "param_groups" not in (state_dict.get("optimizer") or {}):
        # saved by a chain: the common parts sit under 0, 1, ...
        step = next((s for s in (_common_step(state_dict[i]) for i in ints) if s is not None), None)
        part = {k: v for k, v in state_dict.items() if not isinstance(k, int)}
        return [_complete_optimizer_state(part, chained.chained_optimizers[0], step)]
    return None


@register_patch(
    "megatron.muse_spark.chained_optimizer_relayout_load",
    backend="megatron",
    phase="before_train",
    description=(
        "Muse Spark: load a non-distributed optimizer checkpoint saved with another number of "
        "chained optimizers (expert parallel on / off)."
    ),
    condition=is_muse_spark_model_type,
)
def patch_muse_spark_chained_optimizer_load(ctx: PatchContext):
    from megatron.core.optimizer.optimizer import ChainedOptimizer

    orig = ChainedOptimizer.load_state_dict
    if getattr(orig, "_primus_muse_spark", False):
        return

    def load_state_dict(self, state_dict):
        parts = relayout_chained_optimizer_state(self, state_dict)
        if parts is None:
            return orig(self, state_dict)
        log_rank_0(
            f"[Patch:megatron.muse_spark.chained_optimizer_relayout_load] optimizer checkpoint of another "
            f"layout: {len(parts)} optimizer(s) take their own param groups and the saved step"
        )
        for optimizer, part in zip(self.chained_optimizers, parts, strict=True):
            optimizer.load_state_dict(part)
        if len(self.chained_optimizers) > 1:
            self._synchronize_steps()

    load_state_dict._primus_muse_spark = True
    ChainedOptimizer.load_state_dict = load_state_dict
