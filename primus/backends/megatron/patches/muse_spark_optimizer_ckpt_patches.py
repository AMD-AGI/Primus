###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark: distributed-checkpoint optimizer state with fp32 params in a bf16 model.

Muse Spark keeps the router weight, ``e_score_correction_bias`` and the residual gates
fp32 in a bf16 model. ``Float16OptimizerWithFloat16Params.sharded_state_dict`` numbers
the optimizer params from ``float16_groups`` only, but torch numbers the optimizer
state over *all* params of every group (the fp32 main copies of the bf16 params and the
fp32 params, interleaved).
With an fp32 param in a group the ids shift and the save fails ("Optimizer shape ...
does not match model shape ..." or ``KeyError``). This patch maps ids in the
optimizer's real order; a model with no fp32 params takes the upstream path unchanged.

The non-distributed optimizer is one optimizer without expert parallelism and a chain
of two (dense, experts) with it. Its tensors are stored by parameter key, so they load
in either layout, but the checkpoint's common part (param groups, the shared step) has
the shape of the saving layout and ``ChainedOptimizer.load_state_dict`` then fails
("Expected 2 entries in state dict, but got 3"). The second patch takes the param
groups from the loading optimizer and the step from the checkpoint in that case.

Both patches replace a method of a shared Megatron class for the rest of the process. The
replacement checks, each time it runs, that the optimizer holds a Muse Spark parameter
(``model_types.mark_muse_spark_param``: the residual gates and the router weight); for any
other model it calls upstream's method unchanged.
"""

from primus.backends.megatron.core.models.muse_spark.model_types import (
    fail_fast,
    has_muse_spark_param,
    is_muse_spark_model_type,
)
from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0

# the patch ids (registration, fail_fast label and log prefix)
CHAINED_OPTIMIZER_RELAYOUT_LOAD_PATCH = "megatron.muse_spark.chained_optimizer_relayout_load"
FP16_OPTIMIZER_FP32_PARAM_CKPT_PATCH = "megatron.muse_spark.fp16_optimizer_fp32_param_ckpt"

_MODEL_PARAM_GROUPS = ("float16_groups", "fp32_from_fp32_groups", "model_float16_groups", "model_fp32_groups")


def holds_muse_spark_params(optimizer) -> bool:
    """True when ``optimizer`` (a Megatron optimizer, or a chain of them) optimizes a
    Muse Spark model: one of its model params is marked by ``mark_muse_spark_param``."""
    members = getattr(optimizer, "chained_optimizers", None) or [optimizer]
    for opt in members:
        groups = [g for name in _MODEL_PARAM_GROUPS for g in (getattr(opt, name, None) or [])]
        inner = getattr(opt, "optimizer", None)
        groups += [g["params"] for g in (getattr(inner, "param_groups", None) or [])]
        if any(has_muse_spark_param(g) for g in groups):
            return True
    return False


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
    """``Float16OptimizerWithFloat16Params.sharded_state_dict`` with ids in optimizer order.

    The same steps as upstream's method (``test_the_copy_calls_every_helper_upstream_calls``
    fails when upstream gains one): init on load, the id map, the GTP_remat backfill, the
    fp32 main copies, the common step. Only the id order differs. ``metadata`` is accepted
    for the signature; upstream's method does not read it either."""
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
    # upstream's GTP_remat backfill (distributed Muon): it indexes params by their position
    # in chain(groups), so it gets this function's optimizer-order list as one group
    from megatron.core.optimizer import optimizer as opt_module

    backfill = getattr(opt_module, "_backfill_gtp_sharded_param_map", None)
    if backfill is not None:  # absent in older Megatron
        backfill(id_to_sharded_param_map, [params], model_sharded_state_dict)

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
    FP16_OPTIMIZER_FP32_PARAM_CKPT_PATCH,
    backend="megatron",
    phase="before_train",
    description=(
        "Muse Spark: Float16OptimizerWithFloat16Params.sharded_state_dict maps optimizer state ids "
        "in the optimizer's order (fp32 params kept in a bf16 model shift upstream's ids)."
    ),
    condition=is_muse_spark_model_type,
)
@fail_fast(FP16_OPTIMIZER_FP32_PARAM_CKPT_PATCH)
def patch_muse_spark_fp16_optimizer_ckpt(ctx: PatchContext):
    from megatron.core.optimizer import optimizer as opt_module

    cls = opt_module.Float16OptimizerWithFloat16Params
    orig = cls.sharded_state_dict
    if getattr(orig, "_primus_muse_spark", False):
        return

    def sharded_state_dict(self, model_sharded_state_dict, is_loading=False, metadata=None):
        if (
            getattr(self, "is_stub_optimizer", False)
            or not any(getattr(self, "fp32_from_fp32_groups", None) or [])
            or not holds_muse_spark_params(self)
        ):
            return orig(self, model_sharded_state_dict, is_loading, metadata)
        return fp16_optimizer_sharded_state_dict(self, model_sharded_state_dict, is_loading, metadata)

    sharded_state_dict._primus_muse_spark = True
    cls.sharded_state_dict = sharded_state_dict
    log_rank_0(
        f"[Patch:{FP16_OPTIMIZER_FP32_PARAM_CKPT_PATCH}] optimizer checkpoint ids follow the optimizer's param order"
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
        n = len(group["params"])
        packed["params"] = list(range(start, start + n))
        start += n
        groups.append(packed)
    return groups


def _complete_optimizer_state(part, optimizer, step, grad_scaler=None):
    """``part`` (one optimizer's loaded tensors) with ``optimizer``'s own param groups, the
    step the checkpoint (saved in another layout) shares and its ``grad_scaler`` state (the
    fp16 dynamic loss scale; None: none saved)."""
    part = dict(part)
    if grad_scaler is not None and part.get("grad_scaler") is None:
        part["grad_scaler"] = grad_scaler
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
        scaler = state_dict.get("grad_scaler")
        return [
            _complete_optimizer_state(state_dict[i], opt, step, scaler)
            for i, opt in enumerate(chained.chained_optimizers)
        ]
    if n == 1 and ints and "param_groups" not in (state_dict.get("optimizer") or {}):
        # saved by a chain: the common parts sit under 0, 1, ... and the per-param tensors
        # (the Adam moments) in the top-level "optimizer" part. Without that part a load
        # would restart the optimizer state silently: refuse it.
        if not isinstance(state_dict.get("optimizer"), dict):
            raise ValueError(
                f"optimizer checkpoint of a chain of {len(ints)} optimizer(s) has no top-level "
                "'optimizer' part with the per-parameter state (keys: "
                f"{sorted(map(str, state_dict))}); loading it would drop the optimizer moments"
            )
        step = next((s for s in (_common_step(state_dict[i]) for i in ints) if s is not None), None)
        scalers = (
            state_dict[i].get("grad_scaler") if isinstance(state_dict[i], dict) else None for i in ints
        )
        scaler = next((s for s in scalers if s is not None), None)
        part = {k: state_dict[k] for k in named}
        return [_complete_optimizer_state(part, chained.chained_optimizers[0], step, scaler)]
    return None


@register_patch(
    CHAINED_OPTIMIZER_RELAYOUT_LOAD_PATCH,
    backend="megatron",
    phase="before_train",
    description=(
        "Muse Spark: load a non-distributed optimizer checkpoint saved with another number of "
        "chained optimizers (expert parallel on / off)."
    ),
    condition=is_muse_spark_model_type,
)
@fail_fast(CHAINED_OPTIMIZER_RELAYOUT_LOAD_PATCH)
def patch_muse_spark_chained_optimizer_load(ctx: PatchContext):
    from megatron.core.optimizer.optimizer import ChainedOptimizer

    orig = ChainedOptimizer.load_state_dict
    if getattr(orig, "_primus_muse_spark", False):
        return

    def load_state_dict(self, state_dict):
        parts = relayout_chained_optimizer_state(self, state_dict) if holds_muse_spark_params(self) else None
        if parts is None:
            return orig(self, state_dict)
        log_rank_0(
            f"[Patch:{CHAINED_OPTIMIZER_RELAYOUT_LOAD_PATCH}] optimizer checkpoint of another "
            f"layout: {len(parts)} optimizer(s) take their own param groups and the saved step"
        )
        for optimizer, part in zip(self.chained_optimizers, parts, strict=True):
            optimizer.load_state_dict(part)
        if len(self.chained_optimizers) > 1:
            self._synchronize_steps()
        return None

    load_state_dict._primus_muse_spark = True
    ChainedOptimizer.load_state_dict = load_state_dict
