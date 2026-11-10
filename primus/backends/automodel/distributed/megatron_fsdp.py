###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Megatron-FSDP (ZeRO-1/2/3) for the AutoModel diffusion recipe.

WHY:
  ``fsdp.backend: megatron_fsdp`` with ``fsdp.zero_dp_strategy: 1`` gives ZeRO-1:
  optimizer state sharded, parameters and gradients replicated. Gradients are
  reduce-scattered and the updated shards all-gathered after the step, with ring
  collectives only and no DTensors in the traced region.

  AutoModel supports Megatron-FSDP -- ``ModelParallelizer`` wraps a
  ``MegatronFSDPConfig`` itself -- but the diffusion recipe never reaches it:
  ``_build_diffusion_mesh_context`` always builds an FSDP2 strategy and drops
  ``fsdp.backend``, ``compile_parallelized_model`` compiles FSDP2 models only, and
  the recipe never shards the optimizer it builds after parallelization, which
  would leave ZeRO-1 as DDP with extra steps.

WHAT (NO diffusers / AutoModel fork), installed only when ``fsdp.backend`` asks:
  A. ``_build_diffusion_mesh_context``: build a ``MegatronFSDPConfig`` mesh
     context. ``fsdp.*`` keys are translated (``reduce_dtype`` becomes
     ``grad_reduce_in_fp32``, fp32 when unset as on FSDP2) and filtered: AutoModel
     rejects a key the target dataclass does not have, and the FSDP2 presets
     carry several.
  B. ``compile_parallelized_model``: per-layer ``torch.compile`` of the wrapped
     transformer when ``fsdp.enable_compile`` is set. After wrapping, a run that
     asked for activation checkpointing and got none stops.
  C. ``OptimizerConfig.build``, in every config class that defines one, including
     classes defined later: shard the optimizer of a Megatron-FSDP model through
     AutoModel's ``fully_shard_optimizer``. A config that sets
     ``supports_megatron_fsdp_sharding = False`` (the Dion family) is refused, as
     AutoModel's own recipes refuse it.
  D. With ``primus_mfsdp.no_param_reshard``, drop the parameter release
     and backward re-gather hooks below ZeRO-3 (``drop_param_reshard_hooks``).
  E. ``get_sync_ctx``, the recipe's per-micro-batch context: under ZeRO-1,
     reduce the gradients once per optimizer step (``sync_context``).

PRECISION:
  Megatron-FSDP has no compute-dtype knob, and below ZeRO-3 parameters are never
  gathered, so there is no point at which split storage and compute dtypes
  (``model.torch_dtype: float32`` / ``model.compute_dtype: bfloat16``, the FLUX
  preset) could be reconciled: the whole forward and backward would run in fp32
  at about half the speed, without a warning. Patch A refuses that, and any dtype
  but bfloat16, the only one tested here. The optimizer still keeps fp32 main
  weights, as FSDP2 does for the presets' fp32 storage: ``fsdp.preserve_fp32_weights``
  defaults to true here, and false trains bf16 weights with bf16 moments.

Not supported here: context parallelism (diffusers' CP needs FSDP2), LoRA,
``fsdp.cpu_offload``, ``fsdp.overlap_param_gather: false`` below ZeRO-3 and a
``ddp:`` section. Activation checkpointing is off unless
``fsdp.activation_checkpointing`` turns it on (FSDP2 defaults it on), and then
only Primus's whole-block checkpointing applies (FLUX.1, Ideogram-4): AutoModel
skips its own on Megatron-FSDP. Checkpoint saving is untested on this path.
"""
from __future__ import annotations

import contextlib
import functools
import logging
from dataclasses import fields
from typing import Any, Dict

from primus.backends.automodel import options

logger = logging.getLogger(__name__)

LOG_PREFIX = "[PrimusMegatronFSDP]"
MISSING_PACKAGE = (
    "fsdp.backend: megatron_fsdp needs the megatron-fsdp package, which AutoModel pins "
    "(pip install megatron-fsdp==0.5.0)."
)
_BACKEND_ALIASES = frozenset({"megatron_fsdp", "megatron-fsdp", "mfsdp"})
_SIZE_KEYS = frozenset({"dp_size", "dp_replicate_size", "tp_size", "cp_size", "pp_size", "ep_size"})
_PATCHED = "_primus_mfsdp_patched"
_SHARDED = "_primus_mfsdp_sharded"
_SUBCLASS_HOOK = "_primus_mfsdp_subclass_hook"
# How Megatron-FSDP keys the hooks patch D drops in its own bookkeeping.
_RELEASE_KEY = "release module "
_REGATHER_KEY = "all-gather "
# fsdp.zero_dp_strategy also takes Megatron-FSDP's strategy names.
_ZERO_STAGES = {"no_shard": 0, "optim": 1, "optim_grads": 2, "optim_grads_params": 3}
# The strategies whose replicated gradient buffer is reduced in place (patch E).
_IN_PLACE_REDUCE = frozenset({"no_shard", "optim"})

# The recipe builds one transformer per process, after the mesh context.
_STATE: Dict[str, Any] = {"enable_compile": False}


def is_megatron_fsdp_backend(value: Any) -> bool:
    return str(value or "").strip().lower() in _BACKEND_ALIASES


def no_param_reshard_enabled() -> bool:
    return options.flag("primus_mfsdp.no_param_reshard")


# --------------------------------------------------------------------------- #
# Patch A: the mesh context                                                   #
# --------------------------------------------------------------------------- #
def _reduce_dtype_is_fp32(value: Any) -> bool:
    """Unset means fp32, as on AutoModel's FSDP2 path."""
    if value is None:
        return True
    return str(value).lower().replace("torch.", "") in {"float32", "fp32"}


def _zero_stage(value: Any) -> int:
    name = str(value).strip().lower()
    return _ZERO_STAGES[name] if name in _ZERO_STAGES else int(value)


def check_dtypes(dtype, compute_dtype) -> None:
    """Refuse dtypes that would make this path train in fp32, or in an untested dtype (see PRECISION)."""
    import torch

    if compute_dtype is not None and compute_dtype != dtype:
        raise ValueError(
            "Megatron-FSDP cannot serve split storage/compute dtypes "
            f"(model.torch_dtype={dtype}, model.compute_dtype={compute_dtype}). FSDP2 casts parameters "
            "as it all-gathers them; Megatron-FSDP has no compute-dtype setting, and below "
            "zero_dp_strategy 3 parameters are never gathered, so the model would train in "
            f"{dtype}. Set model.torch_dtype and model.compute_dtype to bfloat16; the optimizer "
            "keeps fp32 main weights (fsdp.preserve_fp32_weights)."
        )
    if dtype == torch.float32:
        raise ValueError(
            "Megatron-FSDP would train in fp32 here: model.torch_dtype is float32 and there is no "
            "gather point at which to cast down. Set model.torch_dtype to bfloat16; the optimizer "
            "keeps fp32 main weights (fsdp.preserve_fp32_weights)."
        )
    if dtype != torch.bfloat16:
        raise ValueError(
            f"Megatron-FSDP is tested with bfloat16 only on the diffusion path, and model.torch_dtype "
            f"is {dtype}. Set model.torch_dtype and model.compute_dtype to bfloat16."
        )


def strategy_section(fsdp_options: Dict[str, Any]) -> tuple[Dict[str, Any], bool, list]:
    """Translate an ``fsdp:`` section into AutoModel's ``megatron_fsdp`` distributed section.

    Returns the section, whether per-layer compile was asked for, and the keys
    dropped because ``MegatronFSDPConfig`` has no such field. A whitelist rather
    than a blacklist: the FSDP2 keys the presets carry change with AutoModel, and
    AutoModel raises on every one that reaches the dataclass. Unset, the
    reduction and the main weights are fp32, as on FSDP2.
    """
    opts = dict(fsdp_options)
    opts.pop("backend", None)
    opts.setdefault("grad_reduce_in_fp32", _reduce_dtype_is_fp32(opts.pop("reduce_dtype", None)))
    opts.setdefault("preserve_fp32_weights", True)
    enable_compile = bool(opts.pop("enable_compile", False))
    if bool(opts.pop("cpu_offload", False)):
        raise ValueError(
            "fsdp.cpu_offload is not supported with fsdp.backend: megatron_fsdp, which would train "
            "without it; use fsdp.backend: fsdp2 for CPU offload."
        )
    if int(opts.get("cp_size", 1) or 1) > 1:
        raise ValueError(
            "fsdp.cp_size > 1 is not supported with fsdp.backend: megatron_fsdp; diffusers' context "
            "parallelism needs FSDP2."
        )
    if not bool(opts.get("overlap_param_gather", True)) and _zero_stage(opts.get("zero_dp_strategy", 3)) < 3:
        raise ValueError(
            "fsdp.overlap_param_gather: false is not supported below fsdp.zero_dp_strategy 3 with "
            "fsdp.backend: megatron_fsdp: every rank keeps a full copy of the weights, and without "
            "it nothing gathers the shards the other ranks updated, so training would go on with "
            "stale weights."
        )

    from nemo_automodel.components.distributed.config import MegatronFSDPConfig

    keep = {f.name for f in fields(MegatronFSDPConfig)} | _SIZE_KEYS | {"activation_checkpointing"}
    dropped = sorted(set(opts) - keep)
    section = {"strategy": "megatron_fsdp", **{k: v for k, v in opts.items() if k in keep}}
    return section, enable_compile, dropped


def build_mesh_context(*, fsdp_options: Dict[str, Any], world_size: int, dtype, compute_dtype):
    check_dtypes(dtype, compute_dtype)
    section, enable_compile, dropped = strategy_section(fsdp_options)

    from nemo_automodel.components.distributed import MeshContext
    from nemo_automodel.recipes._dist_utils import parse_distributed_section

    if dropped:
        logger.info("%s dropped fsdp.* keys Megatron-FSDP has no field for: %s", LOG_PREFIX, dropped)
    parsed = parse_distributed_section(section)
    _STATE["enable_compile"] = enable_compile
    strategy = parsed["strategy_config"]
    logger.info(
        "%s routing the transformer to Megatron-FSDP: zero_dp_strategy=%s, grad_reduce_in_fp32=%s, "
        "preserve_fp32_weights=%s, average_in_collective=%s, enable_compile=%s",
        LOG_PREFIX,
        strategy.zero_dp_strategy,
        strategy.grad_reduce_in_fp32,
        strategy.preserve_fp32_weights,
        strategy.average_in_collective,
        enable_compile,
    )
    return MeshContext.build(
        strategy_config=strategy,
        parallelism_sizes=parsed["parallelism_sizes"],
        activation_checkpointing=parsed["activation_checkpointing"],
        world_size=world_size,
    )


def _install_mesh_context_patch() -> None:
    import nemo_automodel.recipes.diffusion.train as train_mod

    orig = train_mod._build_diffusion_mesh_context
    if getattr(orig, _PATCHED, False):
        return

    @functools.wraps(orig)
    def _build(*, fsdp_cfg, ddp_cfg, world_size, dtype, compute_dtype=None, lora_enabled, **kwargs):
        section = fsdp_cfg.to_dict() if hasattr(fsdp_cfg, "to_dict") else fsdp_cfg
        if section is None or not is_megatron_fsdp_backend(section.get("backend")):
            return orig(
                fsdp_cfg=fsdp_cfg,
                ddp_cfg=ddp_cfg,
                world_size=world_size,
                dtype=dtype,
                compute_dtype=compute_dtype,
                lora_enabled=lora_enabled,
                **kwargs,
            )
        if ddp_cfg is not None:
            raise ValueError("Cannot specify both 'fsdp' and 'ddp' configurations.")
        if lora_enabled:
            raise ValueError("LoRA is not supported with fsdp.backend: megatron_fsdp.")
        return build_mesh_context(
            fsdp_options=dict(section), world_size=world_size, dtype=dtype, compute_dtype=compute_dtype
        )

    setattr(_build, _PATCHED, True)
    train_mod._build_diffusion_mesh_context = _build


# --------------------------------------------------------------------------- #
# Patch B: after wrapping, and per-layer compile                              #
# --------------------------------------------------------------------------- #
def _is_megatron_fsdp_context(mesh_context) -> bool:
    from nemo_automodel.components.distributed.config import MegatronFSDPConfig

    return isinstance(getattr(mesh_context, "strategy_config", None), MegatronFSDPConfig)


def is_megatron_fsdp_model(model) -> bool:
    from nemo_automodel.components.distributed.megatron_fsdp import (
        HAS_MEGATRON_FSDP,
        MegatronFSDP,
    )

    return HAS_MEGATRON_FSDP and isinstance(model, MegatronFSDP)


def check_activation_checkpointing(wrapped, mesh_context) -> None:
    """Stop the run when activation checkpointing was asked for and nothing is checkpointed.

    AutoModel's Megatron-FSDP path never checkpoints: ``parallelize_megatron_fsdp``
    reads a ``MegatronFSDPConfig`` field the recipe's parser leaves unset, and the
    upstream parallelizers checkpoint in ``_apply``, which only FSDP2 calls. Primus's
    whole-block checkpointing runs before the strategy is dispatched, so it does
    apply, for the models it exists for (FLUX.1, Ideogram-4).
    """
    if not getattr(mesh_context, "activation_checkpointing", False):
        return
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointWrapper,
    )

    if any(isinstance(module, CheckpointWrapper) for module in wrapped.modules()):
        return
    name = type(getattr(wrapped, "module", wrapped)).__name__
    raise ValueError(
        f"fsdp.activation_checkpointing is on, but nothing in {name} was checkpointed: with "
        "fsdp.backend: megatron_fsdp only Primus's whole-block checkpointing applies (FLUX.1, "
        "Ideogram-4), and AutoModel skips its own. Turn it off, or use fsdp.backend: fsdp2."
    )


def _buffer_dtypes(wrapped, buffer: str) -> list:
    groups = getattr(getattr(wrapped, "param_and_grad_buffer", None), "parameter_groups", None) or ()
    return sorted({str(buf.dtype) for buf in (getattr(g, buffer, None) for g in groups) if buf is not None})


def _after_wrap(wrapped) -> None:
    if not is_megatron_fsdp_model(wrapped):
        return
    if no_param_reshard_enabled():
        drop_param_reshard_hooks(wrapped)
    # The module's parameters are the optimizer's, fp32 whenever the main weights are;
    # the forward computes in the model weight buffer's dtype. fp32 there is the silent
    # half-speed forward that check_dtypes exists to prevent.
    compute = _buffer_dtypes(wrapped, "model_weight_buffer") or sorted(
        {str(p.dtype) for p in wrapped.module.parameters()}
    )
    main = _buffer_dtypes(wrapped, "main_weight_buffer") or compute
    logger.info(
        "%s wrapped %s (strategy=%s, compute dtype=%s, main weight dtype=%s)",
        LOG_PREFIX,
        type(wrapped.module).__name__,
        wrapped.data_parallel_sharding_strategy,
        compute,
        main,
    )
    if compute == ["torch.float32"]:
        logger.error(
            "%s the model weights are fp32: this run computes in fp32. Do not compare it with a bf16 run.",
            LOG_PREFIX,
        )


def _install_parallelize_patches() -> None:
    import nemo_automodel._diffusers.auto_diffusion_pipeline as pipeline

    orig_parallelize = pipeline.parallelize_model
    if not getattr(orig_parallelize, _PATCHED, False):

        @functools.wraps(orig_parallelize)
        def _parallelize(model, mesh_context):
            wrapped = orig_parallelize(model, mesh_context)
            if _is_megatron_fsdp_context(mesh_context):
                check_activation_checkpointing(wrapped, mesh_context)
                _after_wrap(wrapped)
            return wrapped

        setattr(_parallelize, _PATCHED, True)
        pipeline.parallelize_model = _parallelize

    orig_compile = pipeline.compile_parallelized_model
    if not getattr(orig_compile, _PATCHED, False):

        @functools.wraps(orig_compile)
        def _compile(model, mesh_context):
            if not _is_megatron_fsdp_context(mesh_context):
                return orig_compile(model, mesh_context)
            if not _STATE["enable_compile"]:
                return None
            from nemo_automodel.components.distributed.parallelizer import (
                _apply_per_layer_compile,
            )

            # MegatronFSDP holds the model at .module and has no __getattr__, so handed
            # the wrapper the layer discovery would not find the block lists.
            inner = getattr(model, "module", model)
            _apply_per_layer_compile(inner)
            logger.info("%s per-layer torch.compile applied to %s", LOG_PREFIX, type(inner).__name__)
            return None

        setattr(_compile, _PATCHED, True)
        pipeline.compile_parallelized_model = _compile


# --------------------------------------------------------------------------- #
# Patch C: shard the optimizer                                                #
# --------------------------------------------------------------------------- #
def _optimizer_config_classes():
    """Every class in the optimizer-config hierarchy that defines its own ``build``.

    Patching the base alone is not enough: a config naming a plain torch optimizer
    is wrapped in a factory config whose ``build`` never chains to ``super()``
    (``models/ideogram4/zero1.py`` hit the same).
    """
    from nemo_automodel.components.optim.optimizer import OptimizerConfig

    seen, stack, found = set(), [OptimizerConfig], []
    while stack:
        cls = stack.pop()
        if id(cls) in seen:
            continue
        seen.add(id(cls))
        if "build" in vars(cls):
            found.append(cls)
        stack.extend(cls.__subclasses__())
    return found


def shard_optimizers(model, optimizers, *, allow: bool = True):
    """Register each optimizer of a Megatron-FSDP ``model`` with it; others pass through.

    ``allow`` is the optimizer config's ``supports_megatron_fsdp_sharding``.
    """
    if not is_megatron_fsdp_model(model):
        return optimizers
    names = ", ".join(type(o).__name__ for o in optimizers)
    if not allow:
        raise ValueError(
            f"{names} cannot be sharded by Megatron-FSDP: its optimizer config sets "
            "supports_megatron_fsdp_sharding = False. Use another optimizer, or fsdp.backend: fsdp2."
        )
    from nemo_automodel.components.distributed.megatron_fsdp import (
        fully_shard_optimizer,
    )

    sharded, fresh = [], False
    for optimizer in optimizers:
        # A build that calls a wrapped super().build() gets back optimizers that are sharded already.
        if not getattr(optimizer, _SHARDED, False):
            optimizer = fully_shard_optimizer(model, optimizer)
            setattr(optimizer, _SHARDED, True)
            fresh = True
        sharded.append(optimizer)
    if fresh:
        logger.info("%s sharded %s across the data-parallel group", LOG_PREFIX, names)
    return sharded


def _wrap_build(cls) -> None:
    build = vars(cls)["build"]
    if getattr(build, _PATCHED, False):
        return

    @functools.wraps(build)
    def _build(self, *args, **kwargs):
        optimizers = build(self, *args, **kwargs)
        model = args[0] if args else kwargs.get("model")
        return shard_optimizers(
            model, optimizers, allow=getattr(self, "supports_megatron_fsdp_sharding", True)
        )

    setattr(_build, _PATCHED, True)
    cls.build = _build


def _wrap_later_subclasses(base) -> None:
    """Wrap the ``build`` of every subclass of ``base`` defined from now on.

    The walk sees only the classes that exist at install time. One defined later,
    say in the module an ``optimizer._target_`` names, would otherwise build an
    optimizer that is never sharded.
    """
    if vars(base).get(_SUBCLASS_HOOK, False):
        return
    previous = vars(base).get("__init_subclass__")

    def __init_subclass__(cls, **kwargs):
        if previous is not None:
            previous.__func__(cls, **kwargs)
        else:
            super(base, cls).__init_subclass__(**kwargs)
        if "build" in vars(cls):
            _wrap_build(cls)

    base.__init_subclass__ = classmethod(__init_subclass__)
    setattr(base, _SUBCLASS_HOOK, True)


def _install_optimizer_patch() -> None:
    from nemo_automodel.components.optim.optimizer import OptimizerConfig

    for cls in _optimizer_config_classes():
        _wrap_build(cls)
    _wrap_later_subclasses(OptimizerConfig)


# --------------------------------------------------------------------------- #
# Patch D: no parameter release and re-gather below ZeRO-3                    #
# --------------------------------------------------------------------------- #
def _wrapped_backward_handler(fn):
    """The handler a ``create_custom_backward_hook`` wrapper closes over, or None."""
    # Registered through @torch.compiler.disable, which keeps the qualname but not
    # the closure; the real function is under __wrapped__.
    fn = getattr(fn, "__wrapped__", fn)
    code = getattr(fn, "__code__", None)
    closure = getattr(fn, "__closure__", None)
    if code is None or closure is None or "custom_backward_handler" not in code.co_freevars:
        return None
    return closure[code.co_freevars.index("custom_backward_handler")].cell_contents


def drop_param_reshard_hooks(wrapped) -> None:
    """Stop releasing and re-gathering parameters around each unit below ZeRO-3.

    Megatron-FSDP registers ``_post_forward`` on every unit module, which calls
    ``release_module_parameters`` with no sharding-strategy guard, plus a hook that
    all-gathers the parameters again for the backward. Under ``optim`` and
    ``optim_grads`` the parameters were never sharded, so the pair moves the whole
    model over the wire a second time per step for nothing.

    Removing the two hook families is preferable to passing no unit modules: the
    unit list also puts the pre-forward unshard hook on the transformer blocks
    rather than on every submodule. With no backward unshard hook,
    ``TrainingState.PRE_BACKWARD`` is never set either, so forward prefetch cannot
    latch off under ZeRO-1/2.
    """
    strategy = getattr(wrapped, "data_parallel_sharding_strategy", None)
    if strategy == "optim_grads_params":
        logger.info(
            "%s primus_mfsdp.no_param_reshard ignored under zero_dp_strategy 3: the parameters "
            "really are sharded, so the re-gather is required.",
            LOG_PREFIX,
        )
        return

    # Megatron-FSDP keeps the handle of each unit's release hook, keyed by the unit's name.
    hooks = getattr(wrapped, "forward_hooks", {})
    releases = [handle for key, handle in hooks.items() if key.startswith(_RELEASE_KEY)]
    for handle in releases:
        handle.remove()
    wrapped.forward_hooks = {k: v for k, v in hooks.items() if not k.startswith(_RELEASE_KEY)}

    # The backward hook factory serves two handlers, and only one re-gathers. The
    # other, _root_pre_backward, queues the callback that finishes gradient
    # reduction; remove it and every gradient comes out zero. pre_backward is a
    # partial over the real function, so the two are told apart by identity.
    root_handler = getattr(getattr(wrapped, "pre_backward", None), "func", None)
    regathers = 0
    if root_handler is None:
        logger.warning(
            "%s cannot identify the root pre-backward handler; leaving the backward re-gather in place.",
            LOG_PREFIX,
        )
    else:
        # The re-gather hooks are keyed by class name, so 57 FLUX blocks collapse
        # onto two entries; walking the modules is what finds them all.
        for module in wrapped.modules():
            module_hooks = getattr(module, "_forward_hooks", {})
            for hook_id, fn in list(module_hooks.items()):
                handler = _wrapped_backward_handler(fn)
                if handler is None or handler is root_handler:
                    continue
                module_hooks.pop(hook_id)
                getattr(module, "_forward_hooks_with_kwargs", {}).pop(hook_id, None)
                getattr(module, "_forward_hooks_always_called", {}).pop(hook_id, None)
                regathers += 1
        wrapped.backward_pre_hooks = {
            k: v
            for k, v in getattr(wrapped, "backward_pre_hooks", {}).items()
            if not k.startswith(_REGATHER_KEY)
        }

    logger.info(
        "%s dropped %d post-forward parameter releases and %d backward re-gather hooks (strategy=%s); %s",
        LOG_PREFIX,
        len(releases),
        regathers,
        strategy,
        (
            "parameters are gathered once per step"
            if root_handler is not None
            else "the backward re-gather stays"
        ),
    )


# --------------------------------------------------------------------------- #
# Patch E: one gradient reduction per optimizer step under ZeRO-1             #
# --------------------------------------------------------------------------- #
def sync_context(model, is_optim_step):
    """The context a micro-batch of ``model`` runs in, or None to leave it to AutoModel.

    AutoModel wraps with ``model_auto_sync`` on, so every backward reduces the
    gradients and attaches them to the optimizer, and the recipe's context is a
    no-op for Megatron-FSDP. Under ZeRO-1 the gradient buffer is replicated and
    reduced in place, so the next micro-batch of the same step would accumulate
    onto reduced gradients (megatron-fsdp's ``set_model_auto_sync`` warns of
    this). Every micro-batch but the last runs with auto-sync off, and the last
    reduces once and attaches the gradients before the recipe clips them.
    Strategies that shard the gradients reduce into their shard on every backward
    and are left alone.
    """
    if not is_megatron_fsdp_model(model) or model.data_parallel_sharding_strategy not in _IN_PLACE_REDUCE:
        return None
    # The attribute, not set_model_auto_sync(), which repeats that warning on every call.
    model.model_auto_sync = bool(is_optim_step)
    return contextlib.nullcontext() if is_optim_step else model.no_sync()


def _install_sync_context_patch() -> None:
    import nemo_automodel.recipes.diffusion.train as train_mod

    orig = train_mod.get_sync_ctx
    if getattr(orig, _PATCHED, False):
        return

    @functools.wraps(orig)
    def _get_sync_ctx(model, is_optim_step, *args, **kwargs):
        context = sync_context(model, is_optim_step)
        return orig(model, is_optim_step, *args, **kwargs) if context is None else context

    setattr(_get_sync_ctx, _PATCHED, True)
    train_mod.get_sync_ctx = _get_sync_ctx


def install() -> None:
    """Install patches A-E. Idempotent; edits no AutoModel source."""
    from nemo_automodel.components.distributed.megatron_fsdp import HAS_MEGATRON_FSDP

    if not HAS_MEGATRON_FSDP:
        raise ImportError(MISSING_PACKAGE)
    _install_mesh_context_patch()
    _install_parallelize_patches()
    _install_optimizer_patch()
    _install_sync_context_patch()
