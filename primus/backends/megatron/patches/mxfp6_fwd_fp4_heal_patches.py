###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Forward healing for the MXFP6 local spec: at a preset iteration the forward moves to a more accurate format.

WHAT IT DOES
------------
Target ``mxfp6`` (``mxfp6_fwd_fp4_heal_target``, the default):
``mxfp6_fwd_fp4_joint_mlp`` and ``mxfp6_fwd_fp4_joint_txt_proj`` run some of the joint blocks' forward GEMMs in MXFP4.
From ``mxfp6_fwd_fp4_heal_iter`` on, those GEMMs run MXFP6 instead. Each joint MLP (``MXFP6FusedMLP``) and each joint
block's text out-projection (``MXFP6RowParallelLinear``) carries ``_mxfp6_fwd_fp4_healed``; the forward reads it where it
chooses the format, so the backward (MXFP4, captured at forward time) is unchanged and nothing else is touched: no
weights, no optimizer state, no gradient buffers.

Target ``fp8``: every forward linear of the healed transformer layers (joint and single, both streams) runs tensorwise
FP8 -- E4M3 with delayed per-tensor scales on aiter's FP8 tilescale kernels -- with every forward fusion kept; the MXFP4
backward is unchanged (the FP8 packs still write the MX columns it reads). Forward hooks on each healed layer switch
``Mxfp6Gates.fwd_fp8`` on around its forward (the gate is read where each GEMM's format is chosen, so Dynamo guards
it), and every linear weight carries its scales (``F8State``): the packs max each operand's amax into it, and at each
step boundary the multipliers are rolled from the amax history (448 / max(history)). A layer's first FP8 step is a
seed step (``fwd_fp8_seed``): its scales come from the operands themselves.

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


_LAYER_NAME_RE = re.compile(r"transformer\.layers\.(\d+)$")


def build_heal_plan(models: Sequence[torch.nn.Module], order: str = "deep_to_shallow", target: str = "mxfp6") -> HealPlan:
    """Every heal target, grouped by its transformer layer index, in switch order.

    Derived from the live module tree. ``mxfp6``: the modules that can run the MXFP4 forward. ``fp8``: the transformer
    layers themselves (every forward linear in them switches).
    """
    by_layer: Dict[int, List[torch.nn.Module]] = {}
    for chunk in models:
        for name, module in chunk.named_modules():
            if target == "fp8":
                match = _LAYER_NAME_RE.search(name)
                if match is not None:
                    by_layer.setdefault(int(match.group(1)), []).append(module)
                continue
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


def set_heal(plan: HealPlan, count: int, target: str = "mxfp6", seed_from: Optional[int] = None) -> None:
    """Heal the first ``count`` planned layers and un-heal the rest (absolute, so repeating it is a no-op). ``fp8``:
    layers ``seed_from`` .. ``count - 1`` take this step as their seed step (no amax history yet); default none."""
    lo = count if seed_from is None else seed_from
    for position, (_, modules) in enumerate(plan):
        for module in modules:
            if target == "fp8":
                module._mxfp6_fwd_fp8_healed = position < count
                module._mxfp6_fwd_fp8_seed = lo <= position < count
            else:
                module._mxfp6_fwd_fp4_healed = position < count


_F8_SAVED: List[Tuple[bool, bool]] = []


def _f8_pre(module, args, kwargs=None):
    from primus.backends.megatron.core.models.diffusion.common.mxfp6_gates import gates

    g = gates()
    _F8_SAVED.append((g.fwd_fp8, g.fwd_fp8_seed))
    g.fwd_fp8 = bool(getattr(module, "_mxfp6_fwd_fp8_healed", False))
    g.fwd_fp8_seed = g.fwd_fp8 and bool(getattr(module, "_mxfp6_fwd_fp8_seed", False))


def _f8_post(module, args, output):
    from primus.backends.megatron.core.models.diffusion.common.mxfp6_gates import gates

    g = gates()
    g.fwd_fp8, g.fwd_fp8_seed = _F8_SAVED.pop()


def f8_setup(plan: HealPlan) -> Dict[int, list]:
    """Target ``fp8``: attach delayed scales to every MXFP6 linear weight of the planned layers and hook the layers'
    forwards (once each). Returns {layer index: [F8State, ...]}."""
    from primus.backends.megatron.core.extensions.primus_turbo_mxfp6_local import (
        F8State,
        MXFP6ColumnParallelLinear,
        MXFP6RowParallelLinear,
        f8_available,
    )

    if not f8_available():
        raise RuntimeError("mxfp6_fwd_fp4_heal_target fp8 needs a Primus-Turbo / aiter with the FP8 tilescale kernels")
    out = {}
    for idx, layers in plan:
        states = []
        for layer in layers:
            for mod in layer.modules():
                if isinstance(mod, (MXFP6ColumnParallelLinear, MXFP6RowParallelLinear)):
                    w = mod.weight
                    if getattr(w, "_mxfp6_f8", None) is None:
                        w._mxfp6_f8 = F8State(w.device)
                    states.append(w._mxfp6_f8)
            if not getattr(layer, "_mxfp6_f8_hooked", False):
                layer.register_forward_pre_hook(_f8_pre)
                layer.register_forward_hook(_f8_post)
                layer._mxfp6_f8_hooked = True
        out[idx] = states
    return out


_F8_FILE = "mxfp6_fwd_fp8_scales.pt"
_IN_WARMUP = [False]


class warmup_scope:
    """The MLPerf warmup's extent: the train_step wrapper passes its calls straight through."""

    def __enter__(self):
        _IN_WARMUP[0] = True

    def __exit__(self, *exc):
        _IN_WARMUP[0] = False


def f8_checkpoint(plan: HealPlan, states: Dict[int, list], ran: int) -> dict:
    """The delayed FP8 scales of every planned layer (identical on every rank: f8_step reduces the amaxes), keyed by
    layer index, plus how many planned layers ran FP8 in the last step."""
    return {"version": 1, "ran": int(ran), "layers": {idx: [st.state_dict() for st in states[idx]] for idx, _ in plan}}


def f8_restore(plan: HealPlan, states: Dict[int, list], ckpt: dict) -> int:
    """Load ``f8_checkpoint``'s scales into ``states``; returns its ``ran``. Raises if the plan does not match."""
    layers = ckpt["layers"]
    if set(layers) != {idx for idx, _ in plan} or any(len(layers[idx]) != len(states[idx]) for idx, _ in plan):
        raise RuntimeError("mxfp6_fwd_fp4_heal: the checkpoint's FP8 scales do not match this model's healed layers")
    for idx, _ in plan:
        for st, sd in zip(states[idx], layers[idx]):
            st.load_state_dict(sd)
    return int(ckpt["ran"])


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


def heal_target(args) -> str:
    return str(getattr(args, "mxfp6_fwd_fp4_heal_target", "mxfp6") or "mxfp6")


def prewarm_healed_graphs(models: Sequence[torch.nn.Module], run: Callable[[], None], what: str,
                          target: str = "mxfp6", seed: bool = False) -> None:
    """Run ``run`` (a training step or the validation warmup, at production shapes) with every planned layer healed,
    then restore. Under torch.compile (the warmup steps before it have compiled graphs) this compiles the healed cache
    entries, and the Dynamo graph counter must rise: otherwise the flag is not guarded and the switch would be a silent
    no-op. ``fp8``: ``seed`` compiles the seed-step variant; the scales are reset afterwards (warmup data)."""
    plan = build_heal_plan(models, target=target)
    if not plan:
        raise RuntimeError("mxfp6_fwd_fp4_heal: nothing found to heal")
    states = f8_setup(plan) if target == "fp8" else {}
    before = _unique_graph_count()
    set_heal(plan, len(plan), target, 0 if seed else None)
    try:
        run()
    finally:
        set_heal(plan, 0, target)
        for ss in states.values():
            for st in ss:
                st.reset()
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


def _report_heal(args, iteration: Optional[int], healed: int, total: int, target: str = "mxfp6") -> None:
    """Log the switch. Peak memory is reported, never fed back into the decision."""
    peak_gib = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    what = "joint layers on the MXFP6 forward" if target == "mxfp6" else "layers on the tensorwise FP8 forward"
    _log(f"iteration={iteration}: {healed}/{total} {what}; peak allocated {peak_gib:.2f} GiB")
    if not getattr(args, "mlperf_mode", False):
        return
    try:
        from mlperf_logging import mllog

        mllog.get_mllogger().event(
            key="mxfp6_fwd_fp4_heal",
            value={"iteration": iteration, "layers_healed": healed, "layers_total": total, "target": target},
        )
    except Exception as exc:  # pragma: no cover - logger not initialised
        _log(f"mllog emission skipped ({exc}).")


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description="Move the forward to a more accurate format (MXFP6 or tensorwise FP8) at a preset iteration.",
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
    target = heal_target(args)
    if getattr(args, "mlperf_mode", False) and not (
        getattr(args, "mxfp6_fwd_fp4_heal_prewarm", True) and int(getattr(args, "warmup_train_steps", 0) or 0) > 0
    ):
        _log("WARNING: no healed-graph pre-warm (needs warmup_train_steps > 0 and mxfp6_fwd_fp4_heal_prewarm); the "
             "switch will compile inside the timed run.")
    train_iters = int(getattr(args, "train_iters", 0) or 0)
    if train_iters and heal_iter >= train_iters:
        _log(f"WARNING: mxfp6_fwd_fp4_heal_iter={heal_iter} >= train_iters={train_iters}; the switch never happens.")

    original_train_step = megatron_training.train_step
    state: Dict[str, object] = {"plan": None, "applied": 0, "graphs_at_switch": None, "f8": {}, "f8_ran": 0}

    def _patched_train_step(*fn_args, **fn_kwargs):
        if _IN_WARMUP[0]:  # synthetic warmup steps (the pre-warms force the flags themselves)
            return original_train_step(*fn_args, **fn_kwargs)
        if state["plan"] is None:
            model = fn_args[2] if len(fn_args) > 2 else fn_kwargs.get("model")
            if model is not None:
                models = model if isinstance(model, (list, tuple)) else [model]
                plan = build_heal_plan(models, order, target)
                if not plan:
                    raise RuntimeError(
                        "mxfp6_fwd_fp4_heal_iter > 0 but nothing to heal was found: the switch would be a no-op. "
                        "Check that the model uses the MXFP6 local spec."
                    )
                state["plan"] = plan
                if target == "fp8":
                    state["f8"] = f8_setup(plan)
                mode = "all at once" if layers_per_iter <= 0 else f"{layers_per_iter}/iter"
                _log(f"Planned {len(plan)} layers (target {target}); switch at iteration {heal_iter}, {mode}, "
                     f"order={order}")
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

        count = target_layer_count(iteration, heal_iter, layers_per_iter, len(plan))
        if target == "fp8" and state.get("f8_ckpt") is not None:
            # resumed from a checkpoint with FP8 scales: continue their history instead of seeding again
            ckpt, state["f8_ckpt"] = state["f8_ckpt"], None
            state["f8_ran"] = f8_restore(plan, state["f8"], ckpt)
            set_heal(plan, count, target, seed_from=min(state["f8_ran"], count))
            state["f8_seeded"] = count > state["f8_ran"]
            _log(f"Restored the FP8 scales of {len(plan)} layers ({state['f8_ran']} ran FP8 at the save)")
        if target == "fp8":
            # Step boundary: the layers that ran FP8 in the step before roll their scales from its amaxes. Then this
            # step's layers -- the newly healed ones seed their scales from their own operands.
            from primus.backends.megatron.core.extensions.primus_turbo_mxfp6_local import f8_step

            ran = state["f8_ran"]
            if ran:
                # amaxes max-reduced over every rank (no pipeline parallelism: every rank holds every layer), so all
                # ranks quantize with the same scales
                f8_step([st for pos, (idx, _) in enumerate(plan) if pos < ran for st in state["f8"][idx]],
                        int(iteration or 0), torch.distributed.group.WORLD if torch.distributed.is_initialized() else None)
            # Flags change only with the healed count or to end a seed step: the warmup's re-entry (count 0) must not
            # undo the pre-warm's forced flags.
            if count != ran or state.get("f8_seeded"):
                set_heal(plan, count, target, seed_from=ran)
                state["f8_seeded"] = count > ran
            state["f8_ran"] = count
        if count != state["applied"]:
            if target != "fp8":
                set_heal(plan, count, target)
            state["applied"] = count
            # No torch.cuda.empty_cache() here: the saved backward packs keep their size, and the healed forward's
            # buffers were allocated by the warmup's pre-warm step, so the cached blocks serve both. Emptying the cache
            # would stall this step on freeing and re-allocating the whole pool.
            _report_heal(args, iteration, count, len(plan), target)
            state["graphs_at_switch"] = _unique_graph_count()

        return original_train_step(*fn_args, **fn_kwargs)

    _patched_train_step.heal_state = state  # inspection (tests)
    megatron_training.train_step = _patched_train_step

    if target == "fp8":
        # The delayed scales ride along with checkpoints (a side file in the iteration directory, rank 0 writes it);
        # a resume without one seeds again.
        from megatron.training import checkpointing

        original_save = megatron_training.save_checkpoint
        original_load = checkpointing.load_checkpoint

        def _save_with_f8(iteration, *a, **kw):
            out = original_save(iteration, *a, **kw)
            plan_ = state["plan"]
            if plan_ and state["f8_ran"] and getattr(args, "save", None):
                import os

                if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                    d = checkpointing.get_checkpoint_name(args.save, iteration, return_base_dir=True)
                    os.makedirs(d, exist_ok=True)
                    torch.save(f8_checkpoint(plan_, state["f8"], state["f8_ran"]), os.path.join(d, _F8_FILE))
            return out

        def _load_with_f8(*a, **kw):
            out = original_load(*a, **kw)
            iteration = out[0] if isinstance(out, tuple) else out
            load_dir = getattr(args, "load", None)
            if load_dir and iteration:
                import os

                f = os.path.join(checkpointing.get_checkpoint_name(load_dir, iteration, return_base_dir=True), _F8_FILE)
                if os.path.exists(f):
                    state["f8_ckpt"] = torch.load(f, map_location="cpu", weights_only=True)
                    _log(f"Found FP8 scales in {f}")
            return out

        megatron_training.save_checkpoint = _save_with_f8
        checkpointing.load_checkpoint = _load_with_f8
        megatron_training.load_checkpoint = _load_with_f8
    mark_patched(megatron_training, _PATCH_KEY)
    _log(f"Wrapped train_step (heal_iter={heal_iter}, priority=46)")
