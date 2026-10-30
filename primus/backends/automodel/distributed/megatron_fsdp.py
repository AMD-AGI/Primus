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
  collectives only and no DTensors in the traced region. On FLUX.1-schnell it is
  the base of the fastest FP8 recipe measured on this backend.

  AutoModel supports Megatron-FSDP -- ``ModelParallelizer`` wraps a
  ``MegatronFSDPConfig`` itself -- but the diffusion recipe never reaches it:
  ``_build_diffusion_mesh_context`` always builds an FSDP2 strategy and drops
  ``fsdp.backend``, ``compile_parallelized_model`` compiles FSDP2 models only, and
  the recipe never shards the optimizer it builds after parallelization, which
  would leave ZeRO-1 as DDP with extra steps.

WHAT (NO diffusers / AutoModel fork), installed only when ``fsdp.backend`` asks:
  A. ``_build_diffusion_mesh_context``: build a ``MegatronFSDPConfig`` mesh
     context. ``fsdp.*`` keys are translated (``reduce_dtype`` becomes
     ``grad_reduce_in_fp32``) and filtered: AutoModel rejects a key the target
     dataclass does not have, and the FSDP2 presets carry several.
  B. ``compile_parallelized_model``: per-layer ``torch.compile`` of the wrapped
     transformer when ``fsdp.enable_compile`` is set.
  C. ``OptimizerConfig.build``: shard the optimizer of a Megatron-FSDP model
     through AutoModel's ``fully_shard_optimizer``.
  D. With ``primus_mfsdp.no_param_reshard``, drop the parameter release
     and backward re-gather hooks below ZeRO-3 (``drop_param_reshard_hooks``).

PRECISION:
  Megatron-FSDP has no compute-dtype knob, and below ZeRO-3 parameters are never
  gathered, so there is no point at which split storage and compute dtypes
  (``model.torch_dtype: float32`` / ``model.compute_dtype: bfloat16``, the FLUX
  preset) could be reconciled: the whole forward and backward would run in fp32
  at about half the speed, without a warning. Patch A refuses that; use
  ``model.torch_dtype: bfloat16``, with ``fsdp.preserve_fp32_weights`` for fp32
  main weights.

Not supported here: context parallelism (diffusers' CP needs FSDP2), LoRA, and a
``ddp:`` section. Activation checkpointing and checkpoint saving are untested on
this path.
"""
from __future__ import annotations

import functools
import logging
from dataclasses import fields
from typing import Any, Dict

from primus.backends.automodel import options

logger = logging.getLogger(__name__)

_LOG_PREFIX = "[PrimusMegatronFSDP]"
_BACKEND_ALIASES = frozenset({"megatron_fsdp", "megatron-fsdp", "mfsdp"})
_SIZE_KEYS = frozenset({"dp_size", "dp_replicate_size", "tp_size", "cp_size", "pp_size", "ep_size"})
_PATCHED = "_primus_mfsdp_patched"

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
    if value is None:
        return False
    return str(value).lower().replace("torch.", "") in {"float32", "fp32"}


def check_dtypes(dtype, compute_dtype) -> None:
    """Refuse dtypes that would make this path train in fp32 (see PRECISION)."""
    import torch

    if compute_dtype is not None and compute_dtype != dtype:
        raise ValueError(
            "Megatron-FSDP cannot serve split storage/compute dtypes "
            f"(model.torch_dtype={dtype}, model.compute_dtype={compute_dtype}). FSDP2 casts parameters "
            "as it all-gathers them; Megatron-FSDP has no compute-dtype setting, and below "
            "zero_dp_strategy 3 parameters are never gathered, so the model would train in "
            f"{dtype}. Set model.torch_dtype and model.compute_dtype to bfloat16 (and "
            "fsdp.preserve_fp32_weights for fp32 main weights)."
        )
    if dtype == torch.float32:
        raise ValueError(
            "Megatron-FSDP would train in fp32 here: model.torch_dtype is float32 and there is no "
            "gather point at which to cast down. Set model.torch_dtype to bfloat16."
        )


def strategy_section(fsdp_options: Dict[str, Any]) -> tuple[Dict[str, Any], bool, list]:
    """Translate an ``fsdp:`` section into AutoModel's ``megatron_fsdp`` distributed section.

    Returns the section, whether per-layer compile was asked for, and the keys
    dropped because ``MegatronFSDPConfig`` has no such field. A whitelist rather
    than a blacklist: the FSDP2 keys the presets carry change with AutoModel, and
    AutoModel raises on every one that reaches the dataclass.
    """
    from nemo_automodel.components.distributed.config import MegatronFSDPConfig

    opts = dict(fsdp_options)
    opts.pop("backend", None)
    if "reduce_dtype" in opts:
        reduce_fp32 = _reduce_dtype_is_fp32(opts.pop("reduce_dtype"))
        opts.setdefault("grad_reduce_in_fp32", reduce_fp32)
    enable_compile = bool(opts.pop("enable_compile", False))
    if int(opts.get("cp_size", 1) or 1) > 1:
        raise ValueError(
            "fsdp.cp_size > 1 is not supported with fsdp.backend: megatron_fsdp; diffusers' context "
            "parallelism needs FSDP2."
        )

    keep = {f.name for f in fields(MegatronFSDPConfig)} | _SIZE_KEYS | {"activation_checkpointing"}
    dropped = sorted(set(opts) - keep)
    section = {"strategy": "megatron_fsdp", **{k: v for k, v in opts.items() if k in keep}}
    return section, enable_compile, dropped


def build_mesh_context(*, fsdp_options: Dict[str, Any], world_size: int, dtype, compute_dtype):
    from nemo_automodel.components.distributed import MeshContext
    from nemo_automodel.recipes._dist_utils import parse_distributed_section

    check_dtypes(dtype, compute_dtype)
    section, enable_compile, dropped = strategy_section(fsdp_options)
    if dropped:
        logger.info("%s dropped fsdp.* keys Megatron-FSDP has no field for: %s", _LOG_PREFIX, dropped)
    parsed = parse_distributed_section(section)
    if parsed["activation_checkpointing"]:
        logger.warning(
            "%s activation checkpointing is untested with Megatron-FSDP on the diffusion path.", _LOG_PREFIX
        )
    _STATE["enable_compile"] = enable_compile
    strategy = parsed["strategy_config"]
    logger.info(
        "%s routing the transformer to Megatron-FSDP: zero_dp_strategy=%s, grad_reduce_in_fp32=%s, "
        "preserve_fp32_weights=%s, average_in_collective=%s, enable_compile=%s",
        _LOG_PREFIX,
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


def _after_wrap(wrapped) -> None:
    if not is_megatron_fsdp_model(wrapped):
        return
    if no_param_reshard_enabled():
        drop_param_reshard_hooks(wrapped)
    # Every distinct dtype, not the first: under preserve_fp32_weights the fp32 main
    # weights sit beside the bf16 compute copies, which is healthy. fp32 alone is the
    # silent half-speed forward that check_dtypes exists to prevent.
    dtypes = sorted({str(p.dtype) for p in wrapped.module.parameters()})
    logger.info(
        "%s wrapped %s (strategy=%s, parameter dtypes=%s)",
        _LOG_PREFIX,
        type(wrapped.module).__name__,
        wrapped.data_parallel_sharding_strategy,
        dtypes,
    )
    if dtypes == ["torch.float32"]:
        logger.error(
            "%s every parameter is fp32 and there is no bf16 copy: this run computes in fp32. "
            "Do not compare it against a bf16 arm.",
            _LOG_PREFIX,
        )


def _install_parallelize_patches() -> None:
    import nemo_automodel._diffusers.auto_diffusion_pipeline as pipeline

    orig_parallelize = pipeline.parallelize_model
    if not getattr(orig_parallelize, _PATCHED, False):

        @functools.wraps(orig_parallelize)
        def _parallelize(model, mesh_context):
            wrapped = orig_parallelize(model, mesh_context)
            if _is_megatron_fsdp_context(mesh_context):
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
            logger.info("%s per-layer torch.compile applied to %s", _LOG_PREFIX, type(inner).__name__)
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


def shard_optimizers(model, optimizers):
    """Register each optimizer of a Megatron-FSDP ``model`` with it; others pass through."""
    if not is_megatron_fsdp_model(model):
        return optimizers
    from nemo_automodel.components.distributed.megatron_fsdp import (
        fully_shard_optimizer,
    )

    sharded = [fully_shard_optimizer(model, optimizer) for optimizer in optimizers]
    logger.info(
        "%s sharded %s across the data-parallel group",
        _LOG_PREFIX,
        ", ".join(type(o).__name__ for o in optimizers),
    )
    return sharded


def _install_optimizer_patch() -> None:
    for cls in _optimizer_config_classes():
        build = vars(cls)["build"]
        if getattr(build, _PATCHED, False):
            continue

        def _make(orig_build):
            @functools.wraps(orig_build)
            def _build(self, *args, **kwargs):
                optimizers = orig_build(self, *args, **kwargs)
                model = args[0] if args else kwargs.get("model")
                return shard_optimizers(model, optimizers)

            setattr(_build, _PATCHED, True)
            return _build

        cls.build = _make(build)


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
    model over the wire a second time per step for nothing: on FLUX.1-schnell
    ZeRO-1 that was 47.5 GB of all-gather per step instead of 23.8 GB.

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
            _LOG_PREFIX,
        )
        return

    # The backward hook factory serves two handlers, and only one re-gathers. The
    # other, _root_pre_backward, queues the callback that finishes gradient
    # reduction; remove it and every gradient comes out zero. pre_backward is a
    # partial over the real function, so the two are told apart by identity.
    root_handler = getattr(getattr(wrapped, "pre_backward", None), "func", None)
    if root_handler is None:
        logger.warning(
            "%s cannot identify the root pre-backward handler; leaving the backward re-gather in place.",
            _LOG_PREFIX,
        )

    removed = {"release": 0, "regather": 0}
    for module in wrapped.modules():
        for hook_id, fn in list(getattr(module, "_forward_hooks", {}).items()):
            handler = _wrapped_backward_handler(fn)
            if getattr(fn, "__qualname__", "").endswith("._post_forward"):
                kind = "release"
            elif handler is not None and root_handler is not None and handler is not root_handler:
                kind = "regather"
            else:
                continue
            module._forward_hooks.pop(hook_id, None)
            getattr(module, "_forward_hooks_with_kwargs", {}).pop(hook_id, None)
            getattr(module, "_forward_hooks_always_called", {}).pop(hook_id, None)
            removed[kind] += 1
    # Megatron-FSDP's own bookkeeping keys the backward entries by class name, so 57
    # FLUX blocks collapse onto two entries; the walk above is what finds them all.
    wrapped.forward_hooks = {
        k: v for k, v in getattr(wrapped, "forward_hooks", {}).items() if not k.startswith("release module ")
    }
    if root_handler is not None:
        wrapped.backward_pre_hooks = {}

    logger.info(
        "%s dropped %d post-forward parameter releases and %d backward re-gather hooks (strategy=%s); "
        "parameters are gathered once per step",
        _LOG_PREFIX,
        removed["release"],
        removed["regather"],
        strategy,
    )


def install() -> None:
    """Install patches A-D. Idempotent; edits no AutoModel source."""
    from nemo_automodel.components.distributed.megatron_fsdp import HAS_MEGATRON_FSDP

    if not HAS_MEGATRON_FSDP:
        raise ImportError(
            "fsdp.backend: megatron_fsdp needs the megatron-fsdp package, which AutoModel pins "
            "(pip install megatron-fsdp==0.5.0)."
        )
    _install_mesh_context_patch()
    _install_parallelize_patches()
    _install_optimizer_patch()
