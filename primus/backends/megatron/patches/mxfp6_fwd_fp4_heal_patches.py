###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Forward healing for the MXFP6 local spec: the joint blocks' MXFP4 forwards go back to MXFP6 at a preset iteration.

WHAT IT DOES
------------
``mxfp6_fwd_fp4_joint_mlp`` and ``mxfp6_fwd_fp4_joint_txt_proj`` run some of the joint blocks' forward GEMMs in MXFP4.
From ``mxfp6_fwd_fp4_heal_iter`` on, those GEMMs run MXFP6 instead. Each joint MLP (``MXFP6FusedMLP``) and each joint
block's text out-projection (``MXFP6RowParallelLinear``) carries ``_mxfp6_fwd_fp4_healed``; the forward reads it where it
chooses the format, so the backward (MXFP4, captured at forward time) is unchanged and nothing else is touched: no
weights, no optimizer state, no gradient buffers.

HOW IT IS APPLIED
-----------------
At a ``train_step`` boundary, either every planned layer at once or ``mxfp6_fwd_fp4_heal_layers_per_iter`` layers per
iteration in ``mxfp6_fwd_fp4_heal_order``. The healed set is a pure function of the iteration (never a counter, never a
loss or memory value), so every rank flips the same layers at the same step without communicating, the MLPerf warmup's
re-entry into ``train_step`` at a fixed iteration is harmless, and a resumed run lands in the right state.

The flag is read inside the per-block compiled forwards, so Dynamo guards it: a healed layer runs its own cache entry,
shared by every healed instance of the block class. ``prewarm_healed_graphs`` compiles those entries (training and
evaluation) during the MLPerf warmup, so the switch itself compiles nothing.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

_PATCH_KEY = "megatron.mxfp6.fwd_fp4_heal"

# The owning transformer layer index in a module's qualified name; unanchored, so it survives the DDP / Float16Module
# name prefixes.
_LAYER_RE = re.compile(r"transformer\.layers\.(\d+)\.")

HealPlan = List[Tuple[int, List[torch.nn.Module]]]


def _log(msg: str) -> None:
    log_rank_0(f"[Patch:mxfp6_fwd_fp4_heal] {msg}")


def _heal_iter(args) -> int:
    return int(getattr(args, "mxfp6_fwd_fp4_heal_iter", 0) or 0)


def _needs_fwd_fp4_heal(ctx: PatchContext) -> bool:
    args = get_args(ctx)
    return args is not None and _heal_iter(args) > 0


def _is_heal_target(name: str, module: torch.nn.Module) -> bool:
    """A joint MLP, or a joint block's text out-projection: the two places that choose the MXFP4 forward."""
    if not hasattr(module, "_mxfp6_fwd_fp4_healed"):
        return False
    if getattr(module, "_mxfp6_joint", False):
        return True
    return name.endswith("self_attention.added_linear_proj")


def build_heal_plan(models: Sequence[torch.nn.Module], order: str = "deep_to_shallow") -> HealPlan:
    """Every heal target, grouped by its transformer layer index, in switch order.

    Derived from the live module tree, so it covers exactly the modules that can run the MXFP4 forward.
    """
    by_layer: Dict[int, List[torch.nn.Module]] = {}
    for chunk in models:
        for name, module in chunk.named_modules():
            if not _is_heal_target(name, module):
                continue
            match = _LAYER_RE.search(name)
            if match is None:
                raise RuntimeError(f"mxfp6_fwd_fp4_heal: heal target {name} is outside transformer.layers")
            by_layer.setdefault(int(match.group(1)), []).append(module)
    indices = sorted(by_layer, reverse=order == "deep_to_shallow")
    return [(idx, by_layer[idx]) for idx in indices]


def target_layer_count(iteration: Optional[int], heal_iter: int, layers_per_iter: int, plan_len: int) -> int:
    """How many planned layers are healed at ``iteration``. Pure.

    ``layers_per_iter <= 0`` heals everything at the boundary; a rate at or above the plan length is the same thing.
    """
    if iteration is None or heal_iter <= 0 or iteration < heal_iter:
        return 0
    if layers_per_iter <= 0:
        return plan_len
    return min(plan_len, (iteration - heal_iter + 1) * layers_per_iter)


def set_heal(plan: HealPlan, count: int) -> None:
    """Heal the first ``count`` planned layers and un-heal the rest (absolute, so repeating it is a no-op)."""
    for position, (_, modules) in enumerate(plan):
        for module in modules:
            module._mxfp6_fwd_fp4_healed = position < count


def _unique_graph_count() -> int:
    """Dynamo's compiled-graph counter, or -1 when it cannot be read."""
    try:
        from torch._dynamo.utils import counters

        return int(counters["stats"].get("unique_graphs", 0))
    except Exception:  # pragma: no cover - torch internal moved
        return -1


def heal_prewarm_wanted(args) -> bool:
    """Whether the MLPerf warmup should pre-warm the healed graphs."""
    return _heal_iter(args) > 0 and bool(getattr(args, "mxfp6_fwd_fp4_heal_prewarm", True))


def prewarm_healed_graphs(models: Sequence[torch.nn.Module], run: Callable[[], None], what: str) -> None:
    """Run ``run`` (a training step or the validation warmup, at production shapes) with every planned layer healed,
    then restore. Under torch.compile (the warmup steps before it have compiled graphs) this compiles the healed cache
    entries, and the Dynamo graph counter must rise: otherwise the flag is not guarded and the switch would be a silent
    no-op."""
    plan = build_heal_plan(models)
    if not plan:
        raise RuntimeError("mxfp6_fwd_fp4_heal: no joint MLP or text out-projection found to heal")
    before = _unique_graph_count()
    set_heal(plan, len(plan))
    try:
        run()
    finally:
        set_heal(plan, 0)
    after = _unique_graph_count()
    if before == 0:
        _log(f"Pre-warm ran the healed {what} eagerly (torch.compile off)")
        return
    if before < 0 or after < 0:
        _log(f"WARNING: could not read Dynamo's graph counter; healed {what} pre-warm unverified")
        return
    if after <= before:
        raise RuntimeError(
            f"mxfp6_fwd_fp4_heal: the healed {what} pre-warm compiled no new graph (unique_graphs stayed at {after}). "
            "Dynamo did not guard _mxfp6_fwd_fp4_healed, so the switch would leave the MXFP4 forward running."
        )
    _log(f"Pre-warmed the healed {what} graphs (unique_graphs {before} -> {after})")


def _report_heal(args, iteration: Optional[int], healed: int, total: int) -> None:
    """Log the switch. Peak memory is reported, never fed back into the decision."""
    peak_gib = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    _log(f"iteration={iteration}: {healed}/{total} joint layers on the MXFP6 forward; peak allocated {peak_gib:.2f} GiB")
    if not getattr(args, "mlperf_mode", False):
        return
    try:
        from mlperf_logging import mllog

        mllog.get_mllogger().event(
            key="mxfp6_fwd_fp4_heal",
            value={"iteration": iteration, "layers_healed": healed, "layers_total": total},
        )
    except Exception as exc:  # pragma: no cover - logger not initialised
        _log(f"mllog emission skipped ({exc}).")


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description="Move the joint blocks' MXFP4 forwards back to MXFP6 at a preset iteration.",
    priority=46,
    condition=_needs_fwd_fp4_heal,
)
def patch_mxfp6_fwd_fp4_heal(ctx: PatchContext) -> None:
    """Wrap ``train_step`` so the switch lands on an iteration boundary (inside ``mlperf_warmup``, priority 95, which
    re-enters it at a fixed iteration; the pure-function form makes that harmless)."""
    import megatron.training.training as megatron_training

    from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched

    if is_patched(megatron_training, _PATCH_KEY):
        _log("Already applied; skipping re-wrap.")
        return

    args = get_args(ctx)
    heal_iter = _heal_iter(args)
    layers_per_iter = int(getattr(args, "mxfp6_fwd_fp4_heal_layers_per_iter", 0) or 0)
    order = getattr(args, "mxfp6_fwd_fp4_heal_order", "deep_to_shallow")
    if getattr(args, "mlperf_mode", False) and not (
        getattr(args, "mxfp6_fwd_fp4_heal_prewarm", True) and int(getattr(args, "warmup_train_steps", 0) or 0) > 0
    ):
        _log("WARNING: no healed-graph pre-warm (needs warmup_train_steps > 0 and mxfp6_fwd_fp4_heal_prewarm); the "
             "switch will compile inside the timed run.")
    train_iters = int(getattr(args, "train_iters", 0) or 0)
    if train_iters and heal_iter >= train_iters:
        _log(f"WARNING: mxfp6_fwd_fp4_heal_iter={heal_iter} >= train_iters={train_iters}; the switch never happens.")

    original_train_step = megatron_training.train_step
    state: Dict[str, object] = {"plan": None, "applied": 0, "graphs_at_switch": None}

    def _patched_train_step(*fn_args, **fn_kwargs):
        if state["plan"] is None:
            model = fn_args[2] if len(fn_args) > 2 else fn_kwargs.get("model")
            if model is not None:
                models = model if isinstance(model, (list, tuple)) else [model]
                plan = build_heal_plan(models, order)
                if not plan:
                    raise RuntimeError(
                        "mxfp6_fwd_fp4_heal_iter > 0 but no joint MLP or text out-projection with a heal flag was "
                        "found: the switch would be a no-op. Check that the model uses the MXFP6 local spec."
                    )
                state["plan"] = plan
                mode = "all at once" if layers_per_iter <= 0 else f"{layers_per_iter}/iter"
                _log(f"Planned {len(plan)} joint layers; switch at iteration {heal_iter}, {mode}, order={order}")
        plan = state["plan"] or []

        # The iteration kwarg is authoritative; curr_iteration is the live value Megatron sets before train_step
        # (args.iteration only moves on checkpoint load / save).
        iteration = fn_kwargs.get("iteration")
        if iteration is None:
            from megatron.training import get_args as megatron_get_args

            iteration = getattr(megatron_get_args(), "curr_iteration", None)

        if state["graphs_at_switch"] is not None:  # the step after a change: did the switch compile anything?
            grown = _unique_graph_count() - state["graphs_at_switch"]
            if grown > 0:
                _log(f"WARNING: {grown} graph(s) compiled across the switch; the healed graphs were not pre-warmed.")
            state["graphs_at_switch"] = None

        target = target_layer_count(iteration, heal_iter, layers_per_iter, len(plan))
        if target != state["applied"]:
            set_heal(plan, target)
            state["applied"] = target
            # No torch.cuda.empty_cache() here: the saved backward packs keep their size, and the healed forward's
            # buffers were allocated by the warmup's pre-warm step, so the cached blocks serve both. Emptying the cache
            # would stall this step on freeing and re-allocating the whole pool.
            _report_heal(args, iteration, target, len(plan))
            state["graphs_at_switch"] = _unique_graph_count()

        return original_train_step(*fn_args, **fn_kwargs)

    megatron_training.train_step = _patched_train_step
    mark_patched(megatron_training, _PATCH_KEY)
    _log(f"Wrapped train_step (heal_iter={heal_iter}, priority=46)")
