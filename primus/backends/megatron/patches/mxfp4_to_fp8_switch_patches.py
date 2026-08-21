###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Runtime MXFP4 -> FP8 precision switch for Flux (Megatron local spec).

WHAT IT DOES
------------
At a preset iteration every MXFP4 linear flips its ``_fp8_mode`` flag and starts
dispatching to the dynamic tensorwise FP8 autograd Function instead of the MXFP4
one. Neither precision stores a quantized weight -- both keep a plain BF16
``nn.Parameter`` and quantize inside forward -- so the flip needs no weight
conversion, no optimizer state remap, and no checkpoint. That last point is the
reason this exists: MLPerf forbids a checkpoint-mediated transition.

WHY ONE BOUNDARY AND NOT A RAMP
-------------------------------
The two Flux block classes are instantiated 19 and 38 times, but Dynamo keys its
code cache on the *code object*, so with ``inline_inbuilt_nn_modules`` the
instances share compiled graphs. Flipping the whole model therefore costs on the
order of two recompiles, not 57, and pre-warming those two during warmup removes
even that. ``mxfp4_to_fp8_layers_per_iter`` keeps the layer-at-a-time ramp
available for the case where full FP8 does not fit in memory.

THE CONVERTED SET IS A PURE FUNCTION OF THE ITERATION
-----------------------------------------------------
Never a call counter. Three things depend on this:

1. ``mlperf_warmup`` (priority 95) wraps this patch (priority 46) and re-enters
   the inner chain ``warmup_steps + 1`` times with the *same* iteration. A
   counter-driven switch would fire during warmup.
2. Rank agreement. Every rank must flip the identical set at the identical
   iteration. If the decision took input from loss, grad norms or memory, ranks
   would desynchronize and the next collective would *hang* rather than fail.
   Peak memory is logged here but must never feed the decision.
3. Resumability, with no switch progress to persist anywhere.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

_PATCH_KEY = "megatron.mxfp4.to_fp8_switch"

# Matches the owning transformer layer index in a module's qualified name. Left
# unanchored on purpose so it survives the DDP / Float16Module name prefixes
# without any unwrapping.
_LAYER_RE = re.compile(r"transformer\.layers\.(\d+)\.")

LayerPlan = List[Tuple[int, List[torch.nn.Module]]]


def _log(msg: str) -> None:
    log_rank_0(f"[Patch:mxfp4_to_fp8_switch] {msg}")


def _needs_mxfp4_to_fp8_switch(ctx: PatchContext) -> bool:
    args = get_args(ctx)
    if args is None:
        return False
    return int(getattr(args, "mxfp4_to_fp8_switch_iter", 0) or 0) > 0


def _assert_fp8_backend_available() -> None:
    """Reject a run that intends to switch on a Primus-Turbo without FlyDSL.

    ``_init_fp8_switch_state`` resolves the backend leniently so a stack without
    FLYDSL can still build pure-MXFP4 models. This is the callsite that knows the
    run actually means to switch, so this is where a missing backend is fatal
    rather than a silent downgrade to whatever the dispatcher picks.
    """
    from primus.backends.megatron.core.extensions.primus_turbo_mxfp4_local import (
        _SWITCH_FP8_BACKEND,
    )

    if _SWITCH_FP8_BACKEND is None:
        raise RuntimeError(
            "mxfp4_to_fp8_switch_iter > 0 requires BackendType.FLYDSL in Primus-Turbo, "
            "which this build does not expose. Upgrade Primus-Turbo, or unset "
            "mxfp4_to_fp8_switch_iter to keep the run in MXFP4."
        )

    # A GEMM backend pinned *in code* is stored as a plain dict that
    # get_gemm_backend indexes directly, so a dict covering only FP4 -- exactly what
    # set_gemm_backend(AITER, PrecisionType.FP4) builds, and what the MXFP4
    # preshuffle error recommends -- raises KeyError on the first FP8 GEMM, deep
    # inside a custom op long after the switch fired. Probing it here turns that
    # into a startup failure naming the fix. The env-var path is read with .get and
    # degrades to the module's default backend, so it needs no such probe.
    from primus_turbo.pytorch.core.backend import GlobalBackendManager, PrecisionType

    try:
        GlobalBackendManager.get_gemm_backend(PrecisionType.FP8)
    except KeyError as exc:
        raise RuntimeError(
            "mxfp4_to_fp8_switch_iter > 0 but the in-code Primus-Turbo GEMM backend "
            "map has no FP8 entry, so the post-switch FP8 GEMMs would fail with "
            f"KeyError({exc}). Pin both precisions -- "
            "GlobalBackendManager.set_gemm_backend(BackendType.FLYDSL, PrecisionType.FP8) "
            "alongside the FP4 pin, or drop the in-code pin and export "
            "PRIMUS_TURBO_GEMM_BACKEND=FP4:AITER,FP8:FLYDSL instead."
        ) from exc


def build_layer_plan(
    models: Sequence[torch.nn.Module],
    order: str = "deep_to_shallow",
) -> Tuple[LayerPlan, List[torch.nn.Module]]:
    """Group every MXFP4 linear by its owning transformer layer index.

    Derived from the live module tree rather than from ``num_layers``: the 19 + 38
    figure is only a ``FluxConfig`` default, and it is wrong whenever
    ``sensitive_layers_enabled`` is set, since those layers are built as
    ``Float8*ParallelLinear`` or plain linears and have no ``_fp8_mode`` to flip.

    Returns the ordered ``(layer_index, linears)`` plan plus any MXFP4 linears that
    sit outside the layer stack. For Flux the second list is empty -- every MXFP4
    linear comes from the block specs -- but it is collected and converted anyway
    so an unexpected one cannot leave the model in a silently mixed end state.
    """
    from primus.backends.megatron.core.extensions.primus_turbo_mxfp4_local import (
        MXFP4ColumnParallelLinear,
        MXFP4RowParallelLinear,
    )

    mxfp4_types = (MXFP4ColumnParallelLinear, MXFP4RowParallelLinear)
    by_layer: Dict[int, List[torch.nn.Module]] = {}
    extras: List[torch.nn.Module] = []

    for chunk in models:
        for name, module in chunk.named_modules():
            if not isinstance(module, mxfp4_types):
                continue
            match = _LAYER_RE.search(name)
            if match is None:
                extras.append(module)
            else:
                by_layer.setdefault(int(match.group(1)), []).append(module)

    indices = sorted(by_layer)
    if order == "deep_to_shallow":
        indices.reverse()

    return [(idx, by_layer[idx]) for idx in indices], extras


def _linear_class_names(models: Sequence[torch.nn.Module]) -> set:
    """Class names of every linear-like module, for the empty-plan error message."""
    names = set()
    for chunk in models:
        for module in chunk.modules():
            cls = type(module).__name__
            if "Linear" in cls:
                names.add(cls)
    return names


def target_layer_count(
    iteration: Optional[int],
    switch_iter: int,
    layers_per_iter: int,
    plan_len: int,
) -> int:
    """How many layers must be in FP8 at ``iteration``. Pure; no side effects.

    ``layers_per_iter <= 0`` converts everything at the boundary. A rate at or
    above the plan length degenerates to the same thing, so one code path covers
    both the single-boundary switch and the ramp fallback.
    """
    if iteration is None or iteration < switch_iter:
        return 0
    if layers_per_iter <= 0:
        return plan_len
    steps_taken = iteration - switch_iter + 1
    return min(plan_len, steps_taken * layers_per_iter)


def set_fp8_mode(plan: LayerPlan, extras: Sequence[torch.nn.Module], count: int) -> None:
    """Put the first ``count`` planned layers into FP8 and the rest in MXFP4.

    Written as an absolute assignment rather than an incremental flip so that
    calling it repeatedly with the same ``count`` is a genuine no-op -- which is
    what makes warmup's re-entry harmless.
    """
    for position, (_, linears) in enumerate(plan):
        fp8 = position < count
        for linear in linears:
            linear._fp8_mode = fp8

    # Non-layer linears ride along with the first flip: once any conversion has
    # started they are FP8, so the end state is never partially converted.
    for linear in extras:
        linear._fp8_mode = count > 0


def _unique_graph_count() -> int:
    """Dynamo's compiled-graph counter, or -1 when it cannot be read."""
    try:
        from torch._dynamo.utils import counters

        return int(counters["stats"].get("unique_graphs", 0))
    except Exception:  # pragma: no cover - torch internal moved
        return -1


def prewarm_fp8_graphs(
    models: Sequence[torch.nn.Module],
    run_step: Callable[[], None],
    order: str = "deep_to_shallow",
) -> None:
    """Trace the FP8 arm once at production shapes, then restore MXFP4.

    This is what makes the switch seamless, and it is also the only runtime check
    that the switch is not a silent no-op. ``run_step`` must be a real
    grad-enabled training step at the production ``micro_batch_size``:

    - Grad enabled, so AOTAutograd compiles the backward partition too. A
      ``no_grad`` pre-warm would leave the backward to compile at the boundary.
    - Production shapes, because ``automatic_dynamic_shapes`` is on -- pre-warming
      at a different batch size would mark dims dynamic and change the graph for
      everyone.

    The check is structural: ``unique_graphs`` must increase, proving Dynamo
    guarded ``_fp8_mode`` and traced a distinct graph rather than reusing the
    MXFP4 entry. It deliberately does *not* check numerics; there is no eager FP8
    reference mid-warmup, and a second forward at production shapes would be
    expensive. Numerical equivalence to the production Float8 path is covered by
    the unit tests instead, which is why those are the load-bearing ones.
    """
    plan, extras = build_layer_plan(models, order)
    if not plan and not extras:
        _log("Pre-warm skipped: no MXFP4 linears found.")
        return

    before = _unique_graph_count()
    set_fp8_mode(plan, extras, len(plan))
    try:
        run_step()
    finally:
        set_fp8_mode(plan, extras, 0)

    after = _unique_graph_count()
    if before < 0 or after < 0:
        _log("WARNING: could not read Dynamo's unique_graphs counter; pre-warm unverified.")
        return
    if after <= before:
        raise RuntimeError(
            "MXFP4 -> FP8 pre-warm traced no new graph "
            f"(unique_graphs stayed at {after}). Dynamo did not guard _fp8_mode, so the "
            "switch would be a silent no-op: the run would log a successful switch while "
            "still training in MXFP4. Make the dispatch Dynamo-observable before using "
            "this config."
        )
    _log(f"Pre-warm traced the FP8 graph (unique_graphs {before} -> {after}); restored MXFP4.")


def _report_switch(args, iteration: Optional[int], converted: int, total: int) -> None:
    """Log the conversion. Peak memory is reported, never fed back into the decision."""
    peak_gib = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    _log(
        f"iteration={iteration}: {converted}/{total} layers now FP8; "
        f"peak allocated {peak_gib:.2f} GiB"
    )

    # The MLPerf logger only exists when mlperf_mode is set, and the MXFP4 configs
    # this derives from do not set it. An unguarded emission would crash the run at
    # the one iteration that must not fail.
    if not getattr(args, "mlperf_mode", False):
        return
    try:
        from mlperf_logging import mllog

        mllog.get_mllogger().event(
            key="mxfp4_to_fp8_switch",
            value={"iteration": iteration, "layers_converted": converted, "layers_total": total},
        )
    except Exception as exc:  # pragma: no cover - logger not initialised
        _log(f"mllog emission skipped ({exc}).")


@register_patch(
    "megatron.mxfp4.to_fp8_switch",
    backend="megatron",
    phase="before_train",
    description=(
        "Flip MXFP4 linears to dynamic tensorwise FP8 at a preset iteration, "
        "on pre-warmed torch.compile graphs."
    ),
    priority=46,
    condition=_needs_mxfp4_to_fp8_switch,
)
def patch_mxfp4_to_fp8_switch(ctx: PatchContext) -> None:
    """Wrap ``train_step`` so the switch lands on an iteration boundary.

    Priority 46 sits after the delayed-scaling preamble (40) and the FSDP2 FP8
    cache refresh (45), and before ``empty_cache_interval`` (50). It is *inside*
    ``mlperf_warmup`` (95), so warmup does re-enter it; the pure-function form
    above is what makes that harmless, which makes the placement a convenience
    rather than a correctness argument.
    """
    import megatron.training.training as megatron_training

    from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched

    if is_patched(megatron_training, _PATCH_KEY):
        _log("Already applied; skipping re-wrap.")
        return

    args = get_args(ctx)
    switch_iter = int(getattr(args, "mxfp4_to_fp8_switch_iter", 0) or 0)
    layers_per_iter = int(getattr(args, "mxfp4_to_fp8_layers_per_iter", 0) or 0)
    order = getattr(args, "mxfp4_to_fp8_order", "deep_to_shallow")

    _assert_fp8_backend_available()

    original_train_step = megatron_training.train_step
    state: Dict[str, object] = {"plan": None, "extras": None, "applied": 0}

    # *args/**kwargs passthrough rather than the explicit eight-arg signature the
    # delayed-FP8 patch uses. Restating the signature means *synthesizing* an
    # `iteration=` on the way down, which breaks against train_step_seq_split
    # (parallelism/train_step_patches.py, priority 40) whose wrapper takes seven
    # positionals and no iteration at all. Forwarding exactly what arrived adds no
    # such coupling.
    def _patched_train_step(*fn_args, **fn_kwargs):
        model = fn_args[2] if len(fn_args) > 2 else fn_kwargs.get("model")

        if state["plan"] is None and model is not None:
            models = model if isinstance(model, (list, tuple)) else [model]
            plan, extras = build_layer_plan(models, order)
            if not plan and not extras:
                # An empty plan is the one failure this patch cannot survive: every
                # later step is a no-op, so the run would train to completion in
                # MXFP4 while reporting a successful switch. Name what was actually
                # found, since the usual cause is a model built from some other
                # spec provider than PrimusTurboMXFP4LocalSpecProvider.
                raise RuntimeError(
                    "mxfp4_to_fp8_switch_iter > 0 but no MXFP4 linear was found in "
                    f"the model, so the switch would be a no-op. Linear-like modules "
                    f"present: {sorted(_linear_class_names(models))}. Check that fp4 "
                    "is set and that transformer_impl selects the MXFP4 local spec."
                )
            state["plan"] = plan
            state["extras"] = extras
            mode = "all at once" if layers_per_iter <= 0 else f"{layers_per_iter}/iter"
            _log(
                f"Planned {len(plan)} MXFP4 layers ({len(extras)} unindexed linears); "
                f"switch at iteration {switch_iter}, {mode}, order={order}"
            )

        plan = state["plan"] or []
        extras = state["extras"] or []

        # The iteration kwarg is authoritative. args.iteration only moves on
        # checkpoint load/save, so it must not be used as the gate; curr_iteration
        # is the live value Megatron sets just before calling train_step.
        iteration = fn_kwargs.get("iteration")
        if iteration is None:
            from megatron.training import get_args as megatron_get_args

            iteration = getattr(megatron_get_args(), "curr_iteration", None)

        target = target_layer_count(iteration, switch_iter, layers_per_iter, len(plan))
        if target != state["applied"]:
            set_fp8_mode(plan, extras, target)
            state["applied"] = target

            # The caching allocator is holding blocks sized for FP4 tensors while
            # the next forward asks for differently sized FP8 ones, so it may
            # cudaMalloc fresh segments with unusable cached ones sitting idle.
            # This is the only real transient at the switch -- no saved activations
            # are live here, the pool is empty between the optimizer step and the
            # next forward. Deliberately bypasses the empty_cache_interval throttle,
            # which exists because empty_cache is expensive; at one call per switch
            # the cost is irrelevant.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _report_switch(args, iteration, target, len(plan))

        return original_train_step(*fn_args, **fn_kwargs)

    megatron_training.train_step = _patched_train_step
    mark_patched(megatron_training, _PATCH_KEY)
    _log(f"Wrapped train_step (switch_iter={switch_iter}, priority=46)")
