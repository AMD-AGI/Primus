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


def _f8_pre(module, args, kwargs=None):
    from primus.backends.megatron.core.models.diffusion.common.mxfp6_gates import gates

    g = gates()
    # Saved on the layer itself, so a post-hook can only restore what its own pre-hook saved (an always-call post-hook
    # also runs when an earlier pre-hook raised and this one never did).
    module._mxfp6_f8_saved = (g.fwd_fp8, g.fwd_fp8_seed)
    g.fwd_fp8 = bool(getattr(module, "_mxfp6_fwd_fp8_healed", False))
    g.fwd_fp8_seed = g.fwd_fp8 and bool(getattr(module, "_mxfp6_fwd_fp8_seed", False))


def _f8_post(module, args, output):
    from primus.backends.megatron.core.models.diffusion.common.mxfp6_gates import gates

    saved = getattr(module, "_mxfp6_f8_saved", None)
    if saved is None:
        return
    module._mxfp6_f8_saved = None
    g = gates()
    g.fwd_fp8, g.fwd_fp8_seed = saved


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
                # always_call: the gates are restored even when the layer's forward raises
                layer.register_forward_hook(_f8_post, always_call=True)
                layer._mxfp6_f8_hooked = True
        out[idx] = states
    return out


def _all_ranks_ok(ok: bool) -> bool:
    """Whether ``ok`` holds on every rank (one tiny all-reduce; warmup only). A check that fails on one rank alone
    must not raise there alone: the other ranks would wait in their next collective until the distributed timeout."""
    if not torch.distributed.is_initialized():
        return ok
    t = torch.tensor([1 if ok else 0], dtype=torch.int32, device=torch.cuda.current_device())
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN)
    return bool(t.item())


def f8_coverage(states: Dict[int, list]) -> Tuple[int, List[int]]:
    """After a healed training step: how many of the planned layers' linear weights ran the FP8 forward (a pack
    wrote their amax slot), and the layer indices with a weight that did not. A weight falls back to its MX forward
    when aiter has no FP8 kernel for the GEMM's shape -- silently, inside the compiled forward -- so the pre-warm,
    which runs at production shapes, is where that shows. One host sync (warmup only)."""
    used, missed = 0, []
    for idx, ss in states.items():
        for st in ss:
            if any(float(st.amax[r].item()) > 0.0 for r in st.ROLES):
                used += 1
            elif idx not in missed:
                missed.append(idx)
    return used, missed


_IN_WARMUP = [False]


class warmup_scope:
    """The MLPerf warmup's extent: the train_step wrapper passes its calls straight through."""

    def __enter__(self):
        _IN_WARMUP[0] = True

    def __exit__(self, *exc):
        _IN_WARMUP[0] = False


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
                          target: str = "mxfp6", seed: bool = False, strict: bool = False) -> None:
    """Run ``run`` (a training step or the validation warmup, at production shapes) with every planned layer healed,
    then restore. Under torch.compile (the warmup steps before it have compiled graphs) this compiles the healed cache
    entries, and the Dynamo graph counter must rise: otherwise the flag is not guarded and the switch would be a silent
    no-op. ``fp8``: ``seed`` compiles the seed-step variant; the scales are reset afterwards (warmup data). The seed
    training step also checks that every linear weight of the healed layers ran the FP8 forward (``f8_coverage``):
    a warning, or an error with ``strict``."""
    plan = build_heal_plan(models, target=target)
    if not plan:
        raise RuntimeError("mxfp6_fwd_fp4_heal: nothing found to heal")
    states = f8_setup(plan) if target == "fp8" else {}
    before = _unique_graph_count()
    set_heal(plan, len(plan), target, 0 if seed else None)
    # A block class past Dynamo's per-code-object cache limit would run its healed variant eagerly from then on --
    # silently, and the global graph counter below cannot tell which code object compiled. Make that an error here,
    # before the clock starts.
    import torch._dynamo

    fail_knobs = [k for k in ("fail_on_recompile_limit_hit", "fail_on_cache_limit_hit")
                  if hasattr(torch._dynamo.config, k)]
    saved_knobs = {k: getattr(torch._dynamo.config, k) for k in fail_knobs}
    for k in fail_knobs:
        setattr(torch._dynamo.config, k, True)
    try:
        run()
        if states and seed and what.startswith("training"):
            used, missed = f8_coverage(states)
            total = sum(len(ss) for ss in states.values())
            if not _all_ranks_ok(not missed):
                msg = (f"mxfp6_fwd_fp4_heal: linear weights in the healed layers did not run the FP8 forward on some "
                       f"rank (no aiter FP8 kernel for their GEMM shapes); here {total - used} of {total}, layers "
                       f"{sorted(missed)}")
                if strict:
                    raise RuntimeError(msg)
                _log("WARNING: " + msg)
            else:
                _log(f"All {total} linear weights of the healed layers ran the FP8 forward")
            # The step-boundary scale update, with its all-reduce over every rank: run once here, so the first time
            # the timed run issues it is not the first time ever.
            from primus.backends.megatron.core.extensions.primus_turbo_mxfp6_local import f8_step

            f8_step([st for ss in states.values() for st in ss], 0,
                    torch.distributed.group.WORLD if torch.distributed.is_initialized() else None)
    finally:
        for k, v in saved_knobs.items():
            setattr(torch._dynamo.config, k, v)
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
    if not _all_ranks_ok(after > before):
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
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return  # the result log is rank 0's, as for every other MLPerf record
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
    if target == "fp8":
        # A healed block class carries training and evaluation entries for the MX, FP8-seed and FP8 forwards (plus
        # the grad-mode variants): close to Dynamo's default of 8 per code object, past which it falls back to eager
        # without an error. Leave room.
        import torch._dynamo

        for knob in ("cache_size_limit", "recompile_limit"):
            if hasattr(torch._dynamo.config, knob) and getattr(torch._dynamo.config, knob) < 16:
                setattr(torch._dynamo.config, knob, 16)
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

    mark_patched(megatron_training, _PATCH_KEY)
    _log(f"Wrapped train_step (heal_iter={heal_iter}, priority=46)")
