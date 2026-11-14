###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark latent MoE and router.

HF ``MuseSparkMoE``::

    w, idx = gate(x)                      # sigmoid(fp32 logits * mult), bias for choice only
    h = pre_expert_norm(pre_expert_proj(x))          # centered RMSNorm on the latent
    y = sum_k w_k * post_expert_norm(expert_k(h))    # input-scaled RMSNorm per expert output
    out = post_expert_proj(y)

Megatron's ``MoELayer`` with ``moe_latent_size`` already has
``fc1_latent_proj`` (= pre_expert_proj) and ``fc2_latent_proj``
(= post_expert_proj). This module adds:

* :class:`MuseSparkTopKRouter`: fp32 gating scaled by ``moe_router_output_multiplier``.
* :class:`MuseSparkMoELayer`: ``pre_expert_norm`` after ``fc1_latent_proj``; it owns
  ``post_expert_norm`` (one weight shared by all experts).
* :class:`MuseSparkGroupedMLP`, :class:`MuseSparkSequentialMLP` and the Primus-Turbo
  ``PrimusGroupedMLP`` variant (see :func:`muse_spark_expert_cls`): experts that apply ``post_expert_norm``
  to each expert output and then the routing prob (the backend applies the prob
  inside the activation, before ``linear_fc2``; the norm makes that order wrong
  here). With ``muse_spark_fused_post_expert_norm`` (default on) the norm and the prob run as
  one Triton kernel per pass (:mod:`.post_expert_norm`), with the same roundings. Expert TP is not supported: ``post_expert_norm`` needs full expert output
  rows, so ``MuseSparkTransformerConfig`` requires ``expert_tensor_parallel_size == 1``.
"""

from dataclasses import replace
from typing import Optional

import torch
from megatron.core.dist_checkpointing.mapping import ShardedTensorFactory
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.experts import SequentialMLP, TEGroupedMLP
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.moe_utils import (
    maybe_skip_or_early_return_by_cudagraph,
)

from primus.backends.megatron.core.models.muse_spark.model_types import (
    config_flag,
    mark_muse_spark_param,
)
from primus.backends.megatron.core.transformer.muse_spark.forward_pre_hooks import (
    run_forward_pre_hooks,
)
from primus.backends.megatron.core.transformer.muse_spark.norms import (
    MuseSparkCenteredRMSNorm,
    MuseSparkInputScaledRMSNorm,
    apply_keeping_fp32,
)
from primus.backends.megatron.core.transformer.muse_spark.post_expert_norm import (
    post_expert_norm_probs,
)
from primus.core.utils.module_utils import log_rank_0

try:
    # Subclass Primus's router explicitly: ``topk_router_patches`` rebinds
    # ``megatron.core.transformer.moe.router.TopKRouter`` only in training (build_args), so
    # an import-time ``TopKRouter`` binding would depend on the import order.
    from primus.backends.megatron.core.transformer.moe.router import PrimusTopKRouter
except ImportError:  # pragma: no cover - no Triton (CPU-only install)
    PrimusTopKRouter = None
    from megatron.core.transformer.moe.router import TopKRouter as _RouterBase
else:
    _RouterBase = PrimusTopKRouter

# The args PrimusTopKRouter.routing reads. Megatron-only arg sets (the converter's
# build_margs, the logprob-parity harness, unit tests) lack them; there the upstream
# TopKRouter.routing runs.
_PRIMUS_ROUTER_ARGS = (
    "router_logit_softcapping",
    "enable_primus_turbo",
    "moe_use_fused_router_with_aux_score",
    "moe_router_force_load_balancing",
)


def _primus_router_args_present():
    """True / False once the global args exist (they do not change after that), None while
    they are unset (the caller must not cache that answer)."""
    try:
        from megatron.training import global_vars
    except ImportError:  # pragma: no cover
        return False
    args = getattr(global_vars, "_GLOBAL_ARGS", None)
    if args is None:
        return None
    return all(hasattr(args, k) for k in _PRIMUS_ROUTER_ARGS)


__all__ = [
    "MuseSparkTopKRouter",
    "MuseSparkMoELayer",
    "MuseSparkGroupedMLP",
    "MuseSparkSequentialMLP",
    "muse_spark_expert_cls",
    "muse_spark_turbo_grouped_mlp_cls",
]


class _FSDP2LayoutBase(MegatronModule):
    """Empty leading base: lets FSDP2's ``fully_shard`` swap the router's class.

    Megatron's ``Router(ABC, MegatronModule)`` (a base of ``PrimusTopKRouter``) lists
    ``ABC`` first, so CPython rejects
    ``module.__class__ = FSDP<Router>`` ("object layout differs"). With this base first,
    the class's layout base is ``nn.Module`` like FSDPModule's. The MRO is unchanged otherwise.
    """


class MuseSparkTopKRouter(_FSDP2LayoutBase, _RouterBase):
    """Sigmoid top-k router; logits = linear * ``moe_router_output_multiplier``, computed in
    fp32 (fp64 when ``moe_router_dtype == 'fp64'``; bf16 / unset run fp32, like HF).

    The router weight is stored fp32 through Float16Module (HF keeps ``mlp.gate.weight`` fp32).
    Under FSDP2 the fp32 weight is its own parameter group (muse_spark/fsdp2_islands.py).
    """

    fsdp2_dtype_island = True

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.output_multiplier = float(config.moe_router_output_multiplier)
        mark_muse_spark_param(self.weight)

    def _apply(self, fn, *args, **kwargs):
        # Keep the router weight fp32 through Float16Module's blanket ``.bfloat16()``
        # (HF keeps ``mlp.gate.weight`` fp32). Only a cast to bf16 / fp16 snapshots the
        # fp32 value and restores it (lossless; a bf16-built weight is promoted). Float16Module
        # casts before DDP / the distributed optimizer wrap the model, so no grad buffer
        # exists yet. ``.to(device)`` / ``.cuda()`` / fp32 casts do not clone.
        return apply_keeping_fp32(self, "weight", fn, super()._apply, *args, **kwargs)

    def routing(self, logits: torch.Tensor, **kwargs):
        # PrimusTopKRouter.routing under Primus's args (training); upstream routing when the
        # global args are Megatron-only or unset (converter / parity harness / unit tests).
        # Decided once per router after the global args exist (cached on the instance);
        # while they are unset the answer is not cached.
        use_primus = self.__dict__.get("_use_primus_routing")
        if use_primus is None:
            use_primus = _primus_router_args_present()
            if use_primus is not None:
                self.__dict__["_use_primus_routing"] = use_primus
        if PrimusTopKRouter is not None and not use_primus:
            # skip PrimusTopKRouter in the MRO: the upstream TopKRouter.routing
            return super(PrimusTopKRouter, self).routing(logits, **kwargs)
        return super().routing(logits, **kwargs)

    def _router_dtype(self):
        # HF computes the router logits in fp32; fp64 is honoured, bf16 / None fall back to fp32.
        return torch.float64 if getattr(self.config, "moe_router_dtype", None) == "fp64" else torch.float32

    def gating(self, input: torch.Tensor):
        # HF: F.linear(x.float(), W.float()) * output_multiplier. Plain torch GEMM
        # (TE's general_gemm finds no algorithm on ROCm for bf16 input x fp32 weight).
        dtype = self._router_dtype()
        # one .to (a no-op when device and dtype match); a temporary copy under
        # use_cpu_initialization before the caller moves the model (the parameter is kept)
        weight = self.weight.to(device=input.device, dtype=dtype)
        bias = None if self.bias is None else self.bias.to(device=input.device, dtype=dtype)
        logits = torch.nn.functional.linear(input.to(dtype), weight, bias)
        return logits * self.output_multiplier


def _run_forward_pre_hooks(module, x):
    """Run ``module``'s forward pre-hooks (``forward_pre_hooks.run_forward_pre_hooks``) and return the
    input after them; the fused post-expert norm takes one positional input only."""
    args, kwargs = run_forward_pre_hooks(module, (x,))
    if kwargs or len(args) != 1:
        raise RuntimeError(
            f"a forward pre-hook of {type(module).__name__} changed its call to {len(args)} args / "
            f"kwargs {sorted(kwargs)}; the fused post-expert norm takes one input"
        )
    return args[0]


class _PostExpertNormMixin:
    """Run the experts with unit probs, then ``post_expert_norm`` and the routing prob."""

    def _init_post_expert_norm(self):
        # Set by MuseSparkMoELayer (not registered here: the weight belongs to the MoE layer).
        object.__setattr__(self, "_muse_post_expert_norm", None)

    def forward(self, permuted_local_hidden_states, tokens_per_expert, permuted_probs):
        # Expert TP is rejected by MuseSparkTransformerConfig (ETP must be 1), so the
        # expert output rows here are complete and the norm sees the full row.
        output, output_bias = super().forward(
            permuted_local_hidden_states, tokens_per_expert, torch.ones_like(permuted_probs)
        )
        if output_bias is not None:
            raise RuntimeError("Muse Spark experts have no bias (add_bias_linear must be False)")
        norm = self._muse_post_expert_norm
        if norm is not None:
            if config_flag(self.config, "muse_spark_fused_post_expert_norm", True):
                # The kernel reads norm.weight without norm.forward, so run the norm's
                # forward-pre-hooks first: with overlap_param_gather the DDP hook waits there
                # for the weight's all-gather (as PrimusGroupedMLP does for fc1 / fc2).
                output = _run_forward_pre_hooks(norm, output)
                # one Triton kernel per pass, same roundings as below (eager on CPU)
                output = post_expert_norm_probs(output, norm.weight, permuted_probs, norm.eps)
                return output, output_bias
            output = norm(output)
        # HF rounding order (MuseSparkTextExperts): the norm output is rounded to the model
        # dtype, then multiplied by the fp32 routing weight and rounded again. Fusing the
        # prob into the norm would drop the first rounding and move off HF's numerics.
        output = (output.float() * permuted_probs.float().unsqueeze(-1)).to(output.dtype)
        return output, output_bias


class MuseSparkGroupedMLP(_PostExpertNormMixin, TEGroupedMLP):
    """TEGroupedMLP (grouped GEMM) with the per-expert ``post_expert_norm``."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_post_expert_norm()


class MuseSparkSequentialMLP(_PostExpertNormMixin, SequentialMLP):
    """SequentialMLP (one MLP per expert) with the per-expert ``post_expert_norm``."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_post_expert_norm()


_TURBO_GROUPED_MLP = None


def muse_spark_turbo_grouped_mlp_cls():
    """``PrimusGroupedMLP`` (Primus-Turbo grouped GEMM) with the per-expert ``post_expert_norm``.

    Built on first use, so the subclass is defined only when the spec provider hands out
    Turbo grouped linears (``muse_spark_expert_cls`` imports ``PrimusGroupedMLP`` for its
    ``issubclass`` check).
    """
    global _TURBO_GROUPED_MLP
    if _TURBO_GROUPED_MLP is None:
        from primus.backends.megatron.core.transformer.experts import PrimusGroupedMLP

        class MuseSparkTurboGroupedMLP(_PostExpertNormMixin, PrimusGroupedMLP):
            """PrimusGroupedMLP with the per-expert ``post_expert_norm`` (probs applied after it).

            Checkpoints: the Turbo grouped linears train one stacked ``weights`` Parameter
            (the per-expert ``weight{i}`` are frozen views of it), so the sharded state dict
            is bound to ``weights`` (see :func:`stack_expert_sharded_state_dict`); the keys
            and offsets on disk stay those of TEGroupedMLP / SequentialMLP.
            """

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._init_post_expert_norm()
                self._register_load_state_dict_pre_hook(_sync_stacked_expert_weights, with_module=True)

            def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
                sd = super().sharded_state_dict(prefix, sharded_offsets, metadata)
                for name in ("linear_fc1", "linear_fc2"):
                    linear = getattr(self, name)
                    if hasattr(linear, "weights"):
                        stack_expert_sharded_state_dict(sd, f"{prefix}{name}.", linear)
                return sd

        _TURBO_GROUPED_MLP = MuseSparkTurboGroupedMLP
    return _TURBO_GROUPED_MLP


def stack_expert_sharded_state_dict(sharded_sd, key_prefix, linear):
    """Bind the per-expert sharded entries ``<key_prefix>weight{i}`` to ``linear.weights``.

    Primus-Turbo grouped linears keep the trained expert weights in one stacked
    ``weights`` Parameter ([E, out, in]); ``weight{i}`` are frozen views registered after
    DDP set-up. TEGroupedMLP.sharded_state_dict emits one ShardedTensor (or SwiGLU
    ShardedTensorFactory) per view, which matches no optimizer param: the optimizer's
    sharded_state_dict then fails (``KeyError`` in ``id_to_sharded_param_map``). This
    replaces them with one ShardedTensorFactory over ``weights`` that builds exactly
    the per-expert entries (same keys, offsets and SwiGLU split), so the files on disk
    do not change; on load it stacks them back. In place; returns ``sharded_sd``.
    """

    n = linear.num_gemms
    names = [f"{key_prefix}weight{i}" for i in range(n)]
    if not all(k in sharded_sd for k in names):
        return sharded_sd
    subs = [sharded_sd.pop(k) for k in names]
    key0 = subs[0].key
    # TE grouped experts share one key (they differ by the expert offset); the parents
    # may rename it later (layer / optimizer prefixes), so build_fn uses the key it gets.
    if any(sub.key != key0 for sub in subs):
        raise ValueError(f"Muse Spark Turbo experts: per-expert keys differ under {key_prefix!r}")

    def build_fn(key, data, replica_id, flattened_range):
        if flattened_range is not None:
            raise NotImplementedError(
                "Muse Spark Turbo experts: flattened optimizer ranges of the stacked expert "
                "weights are not supported. The distributed optimizer's 'fully_sharded_model_space' "
                "format (only used to load pre-MCore-0.13 checkpoints) makes them; save with "
                "Megatron's default 'dp_reshardable' format or with "
                "--dist_ckpt_optim_fully_reshardable ('fully_reshardable'), or use the "
                "non-distributed optimizer"
            )
        out = {}
        for i, sub in enumerate(subs):
            # every expert keeps the stacked tensor's key; the experts differ by their offsets
            if isinstance(sub, ShardedTensorFactory):
                out[f"expert{i}"] = sub.build_fn(key, data[i], replica_id, None)
            else:
                out[f"expert{i}"] = replace(
                    sub, key=key, data=data[i], dtype=data.dtype, replica_id=replica_id
                )
        return out

    def merge_fn(sub_sd):
        parts = []
        for i, sub in enumerate(subs):
            part = sub_sd[f"expert{i}"]
            parts.append(sub.merge_fn(part) if isinstance(sub, ShardedTensorFactory) else part)
        return torch.stack(parts)

    sharded_sd[f"{key_prefix}weights"] = ShardedTensorFactory(
        key0, linear.weights, build_fn, merge_fn, subs[0].replica_id
    )
    return sharded_sd


def _sync_stacked_expert_weights(module, state_dict, prefix, *args, **kwargs):
    """Load pre-hook: fill ``weights`` from ``weight{i}`` (TE / legacy checkpoints) or
    ``weight{i}`` from ``weights`` (torch_dist checkpoints), so both forms load strictly."""
    for name in ("linear_fc1", "linear_fc2"):
        linear = getattr(module, name, None)
        if linear is None or not hasattr(linear, "weights"):
            continue
        p = f"{prefix}{name}."
        stacked_key = f"{p}weights"
        per = [f"{p}weight{i}" for i in range(linear.num_gemms)]
        # the weight{i} views exist only after the first forward / state_dict()
        views = all(f"weight{i}" in linear._parameters for i in range(linear.num_gemms))
        if stacked_key in state_dict:
            stacked = state_dict[stacked_key]
            if views:
                for i, k in enumerate(per):
                    state_dict.setdefault(k, stacked[i])
            else:
                for k in per:
                    state_dict.pop(k, None)
        elif all(k in state_dict for k in per):
            state_dict[stacked_key] = torch.stack([state_dict[k] for k in per])
            if not views:
                for k in per:
                    state_dict.pop(k)


def muse_spark_expert_cls(base_cls):
    """The Muse Spark expert class for the spec provider's ``base_cls``.

    Raises for an unknown base instead of handing its submodules to another class.
    """
    try:
        from primus.backends.megatron.core.transformer.experts import PrimusGroupedMLP
    except ImportError:  # no Primus-Turbo extension importable: only TE / sequential
        PrimusGroupedMLP = None
    if PrimusGroupedMLP is not None and issubclass(base_cls, PrimusGroupedMLP):
        return muse_spark_turbo_grouped_mlp_cls()
    if issubclass(base_cls, TEGroupedMLP):
        return MuseSparkGroupedMLP
    if issubclass(base_cls, SequentialMLP):
        return MuseSparkSequentialMLP
    raise TypeError(f"Muse Spark has no expert variant for {base_cls.__name__}")


_LOGGED_EXPERTS = False


class _LatentWidthConfig:
    """A live view of a transformer config with ``hidden_size`` replaced by
    ``moe_latent_size``. Every other attribute is read from the shared config at access
    time, so later changes to that config are seen (a copy would go stale)."""

    __slots__ = ("_base",)

    def __init__(self, base):
        object.__setattr__(self, "_base", base)

    @property
    def hidden_size(self):
        return self._base.moe_latent_size

    def __getattr__(self, name):
        # Only reached when normal lookup fails. ``_base`` itself must not recurse: it is
        # unset only on an instance made with ``object.__new__`` (no __init__); copy and
        # pickle go through __reduce__ below, which calls __init__.
        if name == "_base":
            raise AttributeError(name)
        return getattr(self._base, name)

    def __reduce__(self):
        # copy / deepcopy / pickle rebuild through __init__ (the default slot state would be
        # restored with our __setattr__, which writes to the shared config instead)
        return (type(self), (self._base,))

    def __setattr__(self, name, value):
        if name == "hidden_size":
            raise AttributeError("hidden_size of the latent dispatch config is moe_latent_size")
        setattr(self._base, name, value)


def latent_dispatch_config(config):
    """The config the Primus-Turbo DeepEP dispatcher sees with latent MoE.

    That dispatcher reshapes tokens with ``config.hidden_size`` in
    ``dispatch_preprocess`` (the upstream dispatchers use the input's last dim), but
    with latent MoE the dispatched tokens are ``moe_latent_size`` wide. Its constructor
    sizes nothing from ``hidden_size`` (DeepEP buffers are sized at dispatch time from
    the tensor), so giving it this view after construction is enough."""
    return _LatentWidthConfig(config)


def _turbo_deepep_dispatcher_cls():
    """``PrimusTurboDeepEPTokenDispatcher``, or None when Primus-Turbo is not importable
    (not installed, or its import queries a GPU and finds none: RuntimeError)."""
    try:
        from primus.backends.megatron.core.extensions.primus_turbo import (
            PrimusTurboDeepEPTokenDispatcher,
        )
    except (ImportError, RuntimeError):
        return None
    return PrimusTurboDeepEPTokenDispatcher


class MuseSparkMoELayer(MoELayer):
    """MoELayer with Muse Spark latent norms."""

    def __init__(self, config, submodules=None, layer_number: Optional[int] = None, *args, **kwargs):
        super().__init__(config, submodules, layer_number, *args, **kwargs)
        latent = config.moe_latent_size
        assert latent, "Muse Spark MoE needs moe_latent_size"
        self.pre_expert_norm = MuseSparkCenteredRMSNorm(config, latent, eps=config.layernorm_epsilon)
        self.post_expert_norm = MuseSparkInputScaledRMSNorm(config, latent, eps=config.post_norm_eps)
        deepep_cls = _turbo_deepep_dispatcher_cls()
        if deepep_cls is not None and isinstance(self.token_dispatcher, deepep_cls):
            self.token_dispatcher.config = latent_dispatch_config(config)
        if isinstance(self.experts, _PostExpertNormMixin):
            global _LOGGED_EXPERTS
            if not _LOGGED_EXPERTS:
                _LOGGED_EXPERTS = True
                log_rank_0(
                    f"Muse Spark experts: {type(self.experts).__name__} "
                    f"(fc1={type(self.experts.linear_fc1).__name__ if hasattr(self.experts, 'linear_fc1') else '-'}), "
                    f"dispatcher: {type(self.token_dispatcher).__name__}"
                )
            object.__setattr__(self.experts, "_muse_post_expert_norm", self.post_expert_norm)
        else:
            raise TypeError("Muse Spark MoE needs Muse Spark experts (see muse_spark_expert_cls)")

    @maybe_skip_or_early_return_by_cudagraph("preprocess")
    def preprocess(self, hidden_states: torch.Tensor, probs: torch.Tensor, routing_map: torch.Tensor):
        """Upstream ``MoELayer.preprocess`` plus ``pre_expert_norm`` after ``fc1_latent_proj``.

        Same decorator and the same shared-expert-overlap check as upstream; the only
        change is the norm between the latent projection and the dispatch.
        """
        assert (
            not self.shared_expert_overlap
        ), "Shared expert overlap not supported when MoE latent projections are used."
        hidden_states, _ = self.fc1_latent_proj(hidden_states)
        hidden_states = self.pre_expert_norm(hidden_states)
        hidden_states, probs = self.token_dispatcher.dispatch_preprocess(hidden_states, routing_map, probs)
        return hidden_states, probs
