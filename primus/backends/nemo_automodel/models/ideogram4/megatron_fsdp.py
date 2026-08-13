###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Megatron-FSDP (ZeRO-1/2/3) for Ideogram-4 in the AutoModel diffusion recipe.

WHY:
  ``zero_dp_strategy=1`` gives ZeRO-1: optimizer state sharded, parameters and
  gradients replicated. Gradients are reduce-scattered and the updated shards are
  all-gathered after the step -- two shard-exchange units per step against three
  for FSDP2 ZeRO-3 and three for ZeroRedundancyOptimizer on top of DDP. It uses
  ring collectives only, with no broadcasts, which matters on xGMI where tree
  broadcasts load links asymmetrically. And because parameters stay replicated
  there are no DTensors in the traced region.

  The diffusion recipe cannot reach any of this today: it hardcodes ``fsdp2`` as
  the strategy and its optimizer is built after parallelization, so nothing ever
  shards it. Automodel itself has full Megatron-FSDP support -- version adaptation,
  precision translation, deferred optimizer sharding, mesh compatibility -- the
  diffusion path just never routes to it.

WHAT (NO diffusers / Automodel fork) -- three env-gated monkeypatches:
  A. ``_build_diffusion_parallel_manager_args`` hardcodes ``strategy: fsdp2`` and
     drops ``fsdp.backend`` on the floor. Branch on the backend and emit
     MegatronFSDPConfig fields instead. This is more than renaming a string:
     ``_resolve_strategy_config`` raises on any key that is not a field of the
     target dataclass, and the builder injects FSDP2-only keys unconditionally,
     so the branch has to filter and translate rather than forward.
  B. ``_create_parallel_manager`` knows only ``ddp`` and ``fsdp2``. Add a
     ``megatron_fsdp`` branch returning a shim around MegatronFSDPManager, whose
     ``parallelize`` returns a ``(model, optimizer)`` tuple the tuple-unaware call
     site would otherwise assign whole to the pipeline.
  C. The recipe builds its optimizer after parallelization and never calls
     ``shard_optimizers_for_megatron_fsdp``, so the optimizer stays replicated and
     ZeRO-1 silently degrades to DDP-with-extra-steps. Patch ``OptimizerConfig.build``
     -- the same seam zero1.py uses -- and map Automodel's ``fully_shard_optimizer``
     over the list it returns.

PRECISION, THE ONE THAT SILENTLY INVALIDATES A BENCHMARK:
  The Ideogram preset ships split dtypes (``torch_dtype: float32`` /
  ``compute_dtype: bfloat16``), which is legal on FSDP2 only because FSDP2 casts
  parameters to the compute dtype as it all-gathers them. Megatron-FSDP has no
  compute-dtype knob, and under ZeRO-1 parameters are never gathered, so there is
  no point at which such a cast could happen. The model would materialise in fp32
  and the whole forward and backward would run in fp32 -- roughly half the
  throughput, for reasons that have nothing to do with ZeRO level. Nothing warns
  you, so Patch A refuses to run rather than let it happen: pass
  ``--model.torch_dtype=bfloat16`` and get bf16 compute with an fp32 main-weight
  buffer from ``preserve_fp32_weights`` instead.

Activation (env, no config schema change):
    PRIMUS_IDEOGRAM_MFSDP=1   install all three patches
"""
from __future__ import annotations

import functools
import logging
import os
from dataclasses import fields
from typing import Any, Dict

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "True", "yes", "on"}

_BACKEND_ALIASES = {"megatron_fsdp", "megatron-fsdp", "mfsdp"}
_MANAGER_TYPE = "megatron_fsdp"

# The wrapped MegatronFSDP model, published by the Patch B shim and consumed by
# the Patch C optimizer patch. The recipe builds exactly one transformer per
# process, and the optimizer is built strictly after parallelization, so a single
# module-level reference is the whole handshake.
_WRAPPED_MODEL = None


def is_mfsdp_enabled() -> bool:
    return os.getenv("PRIMUS_IDEOGRAM_MFSDP", "0") in _TRUTHY


def _is_mfsdp_backend(backend: Any) -> bool:
    return str(backend).lower() in _BACKEND_ALIASES


def _reduce_dtype_is_fp32(value: Any) -> bool:
    """Map an FSDP2 ``reduce_dtype`` onto Megatron-FSDP's ``grad_reduce_in_fp32``."""
    import torch

    if value is None:
        return False
    if isinstance(value, torch.dtype):
        return value == torch.float32
    return str(value).lower() in {"float32", "fp32", "torch.float32"}


# ---------------------------------------------------------------------------
# Patch A: config passthrough and key translation
# ---------------------------------------------------------------------------
def _build_mfsdp_manager_args(
    *,
    fsdp_options: Dict[str, Any],
    world_size: int,
    dtype,
    compute_dtype,
) -> Dict[str, Any]:
    import torch
    from nemo_automodel.components.distributed.config import MegatronFSDPConfig
    from nemo_automodel.recipes._dist_utils import parse_distributed_section

    if compute_dtype is not None and compute_dtype != dtype:
        raise ValueError(
            "Megatron-FSDP cannot serve split storage/compute dtypes "
            f"(model.torch_dtype={dtype}, model.compute_dtype={compute_dtype}). "
            "FSDP2 gets away with this by casting parameters as it all-gathers them; "
            "Megatron-FSDP has no compute-dtype knob, and under zero_dp_strategy<3 "
            "parameters are never gathered, so the model would silently train in "
            f"{dtype}. Set model.torch_dtype=bfloat16 (and fsdp.preserve_fp32_weights "
            "for fp32 master weights) instead."
        )
    if dtype == torch.float32:
        raise ValueError(
            "Megatron-FSDP on this path would train in fp32: model.torch_dtype is "
            "float32 and there is no gather point at which to cast down. Set "
            "model.torch_dtype=bfloat16."
        )

    options = dict(fsdp_options)
    options.pop("backend", None)

    if "reduce_dtype" in options:
        options["grad_reduce_in_fp32"] = _reduce_dtype_is_fp32(options.pop("reduce_dtype"))

    enable_compile = bool(options.pop("enable_compile", False))

    # Whitelist rather than blacklist. The builder injects FSDP2-only keys
    # unconditionally and _resolve_strategy_config raises on every one of them, so
    # a blacklist would have to be updated in lockstep with Automodel forever.
    keep = {f.name for f in fields(MegatronFSDPConfig)} | {
        "dp_size",
        "dp_replicate_size",
        "tp_size",
        "cp_size",
        "pp_size",
        "ep_size",
        "activation_checkpointing",
    }
    dropped = sorted(set(options) - keep)
    strategy_options = {k: v for k, v in options.items() if k in keep}
    if dropped:
        logger.info("[PrimusIdeogramMFSDP] dropped FSDP2-only fsdp.* keys: %s", dropped)

    parsed = parse_distributed_section({"strategy": _MANAGER_TYPE, **strategy_options})

    return {
        "_manager_type": _MANAGER_TYPE,
        "_enable_compile": enable_compile,
        "world_size": world_size,
        "dp_size": parsed["dp_size"],
        "dp_replicate_size": parsed["dp_replicate_size"],
        "tp_size": parsed["tp_size"],
        "cp_size": parsed["cp_size"],
        "pp_size": parsed["pp_size"],
        "ep_size": parsed["ep_size"],
        **parsed["strategy_config"].to_dict(),
        "activation_checkpointing": parsed["activation_checkpointing"],
    }


def _install_manager_args_patch() -> bool:
    import nemo_automodel.recipes.diffusion.train as train_mod

    orig = train_mod._build_diffusion_parallel_manager_args
    if getattr(orig, "_primus_mfsdp_patched", False):
        return True

    @functools.wraps(orig)
    def _build_args(*, fsdp_cfg, ddp_cfg, world_size, dtype, compute_dtype=None, lora_enabled, **kwargs):
        cfg = fsdp_cfg
        if hasattr(cfg, "to_dict"):
            cfg = cfg.to_dict()
        backend = (cfg or {}).get("backend")
        if not (is_mfsdp_enabled() and cfg is not None and _is_mfsdp_backend(backend)):
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
            raise ValueError("LoRA is not supported on the Megatron-FSDP path.")

        args = _build_mfsdp_manager_args(
            fsdp_options=dict(cfg),
            world_size=world_size,
            dtype=dtype,
            compute_dtype=compute_dtype,
        )
        logger.info(
            "[PrimusIdeogramMFSDP] routing to Megatron-FSDP: zero_dp_strategy=%s, "
            "grad_reduce_in_fp32=%s, preserve_fp32_weights=%s, enable_compile=%s",
            args.get("zero_dp_strategy"),
            args.get("grad_reduce_in_fp32"),
            args.get("preserve_fp32_weights"),
            args.get("_enable_compile"),
        )
        return args

    _build_args._primus_mfsdp_patched = True
    train_mod._build_diffusion_parallel_manager_args = _build_args
    return True


# ---------------------------------------------------------------------------
# Patch B: manager factory
# ---------------------------------------------------------------------------
class _MegatronFSDPManagerShim:
    """Adapt MegatronFSDPManager to the diffusion call site.

    ``_apply_parallelization`` does ``parallel_module = manager.parallelize(comp_module)``
    and assigns the result straight onto the pipeline, but MegatronFSDPManager
    returns ``(model, optimizer)``. Unwrap it, and publish the wrapped model so the
    optimizer patch can find it.

    ``maybe_compile`` exists only on FSDP2Manager and the call site gates on
    ``hasattr``, so the shim has to define it or ``fsdp.enable_compile`` is
    silently dropped and this arm runs eager against compiled baselines.
    """

    def __init__(self, inner, enable_compile: bool = False):
        self._inner = inner
        self.device_mesh = inner.device_mesh
        self.enable_compile = enable_compile
        self.megatron_fsdp_model = None

    def parallelize(self, model, optimizer=None):
        global _WRAPPED_MODEL

        # Captured before wrapping and stamped on the inner module, mirroring the
        # DDP branch: MegatronFSDP holds the real model as ``self.module``, so the
        # wrapper's state-dict keys gain a ``module.`` prefix that
        # ``from_pretrained`` would later reject.
        pre_shard_keys = list(model.state_dict().keys())
        wrapped, _ = self._inner.parallelize(model, optimizer=optimizer)
        inner_module = getattr(wrapped, "module", wrapped)
        setattr(inner_module, "_pre_shard_hf_state_dict_keys", pre_shard_keys)

        self.megatron_fsdp_model = wrapped
        _WRAPPED_MODEL = wrapped

        # The compute dtype is logged because an fp32 forward is the failure this
        # path is most able to produce and least able to announce: it trains, it
        # converges, and it is simply half the speed of every arm it is compared
        # against. Patch A refuses the configuration that causes it; this reports
        # what actually landed.
        # Report EVERY distinct parameter dtype, not the first one. Under
        # preserve_fp32_weights the fp32 main weights are exposed through
        # .parameters() alongside the bf16 compute copies, so reading one parameter
        # reports float32 and looks exactly like the silent fp32 forward this line
        # is here to catch. Seeing {bfloat16, float32} together is the healthy
        # mixed-precision state; float32 ALONE is the failure.
        dtypes = sorted({str(p.dtype) for p in inner_module.parameters()})
        logger.info(
            "[PrimusIdeogramMFSDP] wrapped %s in %s (zero_dp_strategy=%s, "
            "param dtypes=%s, preserve_fp32_weights=%s)",
            type(inner_module).__name__,
            type(wrapped).__name__,
            self._inner.zero_dp_strategy,
            dtypes,
            self._inner.preserve_fp32_weights,
        )
        if dtypes == ["torch.float32"]:
            logger.error(
                "[PrimusIdeogramMFSDP] every parameter is fp32 and there is no bf16 "
                "copy: this forward and backward will run in fp32 at roughly half "
                "the throughput, for reasons unrelated to the ZeRO level. Do not "
                "compare this run against a bf16 arm."
            )
        return wrapped

    def maybe_compile(self, model):
        """Per-layer torch.compile over the MegatronFSDP-wrapped transformer.

        ``_apply_per_layer_compile`` is not FSDP2-specific -- it walks the block
        lists and calls ``nn.Module.compile()`` in place -- but it must be handed
        the inner module: MegatronFSDP holds the real model at ``.module`` and
        defines no ``__getattr__``, so the wrapper answers ``hasattr(...,
        "transformer_blocks")`` with False and the function would silently fall
        back to heuristic layer extraction.

        Stages 1 and 2 are both measured clean on Ideogram-4: one dynamo frame, zero
        graph breaks, zero recompiles, traces structurally identical to the FSDP2
        ZeRO-2 arm (TORCH_TRACE + tlparse at mbs=4, 1024px).

        That was not the prediction. The worry was that stages 2 and 3 swap parameter
        storage in and out of the flat buffer and dynamo would break on it or capture
        a stale reference. It does not happen here because the unit boundary IS the
        transformer block: per-layer compile traces the block body while the storage
        manipulation runs in hooks outside the traced region, so dynamo never sees
        it. Expect that to stop holding if the unit modules are ever set to something
        finer than the compiled layer.

        Stage 3 is untested.
        """
        if not self.enable_compile:
            return

        from nemo_automodel.components.distributed.parallelizer import (
            _apply_per_layer_compile,
        )

        strategy = self._inner.zero_dp_strategy
        if strategy not in (1, 2):
            logger.warning(
                "[PrimusIdeogramMFSDP] applying per-layer torch.compile at "
                "zero_dp_strategy=%s, which has not been checked for graph breaks. "
                "Stages 1 and 2 are measured clean; verify this one with CDEBUG=1 "
                "before trusting its step time.",
                strategy,
            )

        inner_module = getattr(model, "module", model)
        _apply_per_layer_compile(inner_module)
        logger.info(
            "[PrimusIdeogramMFSDP] per-layer torch.compile applied to %s " "(zero_dp_strategy=%s)",
            type(inner_module).__name__,
            strategy,
        )

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _create_mfsdp_manager(args: Dict[str, Any], enable_compile: bool = False):
    import torch
    from nemo_automodel.components.distributed import DistributedSetup, ParallelismSizes
    from nemo_automodel.components.distributed.config import MegatronFSDPConfig
    from nemo_automodel.components.distributed.megatron_fsdp import MegatronFSDPManager

    world_size = args.get("world_size")
    if world_size is None:
        world_size = torch.distributed.get_world_size()

    parallelism = ParallelismSizes(
        dp_size=args.get("dp_size"),
        dp_replicate_size=args.get("dp_replicate_size"),
        tp_size=args.get("tp_size", 1),
        pp_size=args.get("pp_size", 1),
        cp_size=args.get("cp_size", 1),
        ep_size=args.get("ep_size", 1),
    )
    field_names = {f.name for f in fields(MegatronFSDPConfig)}
    strategy = MegatronFSDPConfig(**{k: v for k, v in args.items() if k in field_names})

    distributed_setup = DistributedSetup.build(
        strategy=strategy,
        parallelism_sizes=parallelism,
        activation_checkpointing=args.get("activation_checkpointing", False),
        world_size=world_size,
    )
    manager = MegatronFSDPManager(
        distributed_setup.strategy_config,
        device_mesh=distributed_setup.mesh_context.device_mesh,
    )
    return _MegatronFSDPManagerShim(manager, enable_compile=enable_compile)


def _install_manager_factory_patch() -> bool:
    import nemo_automodel._diffusers.auto_diffusion_pipeline as pipe_mod

    orig = pipe_mod._create_parallel_manager
    if getattr(orig, "_primus_mfsdp_patched", False):
        return True

    @functools.wraps(orig)
    def _create(manager_args: Dict[str, Any]):
        if str(manager_args.get("_manager_type", "")).lower() != _MANAGER_TYPE:
            return orig(manager_args)
        args = dict(manager_args)
        args.pop("_manager_type", None)
        enable_compile = bool(args.pop("_enable_compile", False))
        return _create_mfsdp_manager(args, enable_compile=enable_compile)

    _create._primus_mfsdp_patched = True
    pipe_mod._create_parallel_manager = _create
    return True


# ---------------------------------------------------------------------------
# Patch C: deferred optimizer sharding
# ---------------------------------------------------------------------------
def _optimizer_config_classes():
    """Every class in the OptimizerConfig hierarchy that defines its OWN ``build``.

    Patching only the base class is not enough and fails silently: YAML
    ``_target_: torch.optim.AdamW`` resolves to a plain torch class, so
    build_optimizer_config() wraps it in ``OptimizerFromFactoryConfig``, which
    overrides ``build`` and never chains to ``super()``. See zero1.py, which hit
    exactly this.
    """
    from nemo_automodel.components.optim.optimizer import OptimizerConfig

    seen, stack, out = set(), [OptimizerConfig], []
    while stack:
        cls = stack.pop()
        if id(cls) in seen:
            continue
        seen.add(id(cls))
        if "build" in vars(cls):
            out.append(cls)
        stack.extend(cls.__subclasses__())
    return out


def _shard_optimizer(optimizer):
    """Register one already-built optimizer with the wrapped MegatronFSDP model."""
    from nemo_automodel.components.distributed.megatron_fsdp import (
        fully_shard_optimizer,
    )

    if _WRAPPED_MODEL is None:
        logger.error(
            "[PrimusIdeogramMFSDP] optimizer built but no MegatronFSDP model was "
            "recorded; optimizer state stays REPLICATED and this is not ZeRO-1. "
            "Check that parallelization ran before the optimizer was built."
        )
        return optimizer
    sharded = fully_shard_optimizer(_WRAPPED_MODEL, optimizer)
    logger.info(
        "[PrimusIdeogramMFSDP] sharded %s across the DP group via fully_shard_optimizer",
        type(optimizer).__name__,
    )
    return sharded


def _install_optimizer_patch() -> bool:
    patched = []
    for cls in _optimizer_config_classes():
        if getattr(vars(cls)["build"], "_primus_mfsdp_patched", False):
            patched.append(cls.__name__)
            continue

        def _make(orig_build):
            @functools.wraps(orig_build)
            def _build_mfsdp(self, *args, **kwargs):
                optimizers = orig_build(self, *args, **kwargs)
                if not is_mfsdp_enabled() or _WRAPPED_MODEL is None:
                    return optimizers
                return [_shard_optimizer(opt) for opt in optimizers]

            _build_mfsdp._primus_mfsdp_patched = True
            return _build_mfsdp

        cls.build = _make(vars(cls)["build"])
        patched.append(cls.__name__)
    logger.info("[PrimusIdeogramMFSDP] optimizer sharding patch installed on %s", patched)
    return bool(patched)


def install() -> bool:
    """Install the Megatron-FSDP patches. No-op unless PRIMUS_IDEOGRAM_MFSDP is set.

    Idempotent; modifies NO Automodel source (module-level monkeypatches only).
    """
    if not is_mfsdp_enabled():
        return False

    from nemo_automodel.components.distributed.megatron_fsdp import HAS_MEGATRON_FSDP

    if not HAS_MEGATRON_FSDP:
        raise ImportError(
            "PRIMUS_IDEOGRAM_MFSDP=1 but megatron_fsdp is not importable. "
            "Install it with: pip install --no-deps megatron-fsdp==0.5.0"
        )

    ok_args = _install_manager_args_patch()
    ok_factory = _install_manager_factory_patch()
    ok_optim = _install_optimizer_patch()
    return ok_args and ok_factory and ok_optim
