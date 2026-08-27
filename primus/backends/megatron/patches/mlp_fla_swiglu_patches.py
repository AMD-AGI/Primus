###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MLP FLA SwiGLU Patch
======================

Routes ``MLP``'s gated-linear-unit activation through
flash-linear-attention's (FLA) Triton-fused ``swiglu`` kernel (one fwd + one
bwd kernel) instead of Megatron's naive two-kernel ``silu(x_glu) * x_linear``,
saving ~20 ms/iter on GDN/KDA/Mamba hybrid training.

Only takes effect for plain, unclamped SwiGLU: ``config.gated_linear_unit`` with
``F.silu``, no ``use_te_activation_func`` module slot, no
``activation_func_clamp_value`` and no ``glu_linear_offset``. ``F.silu`` on its
own is not a sufficient test -- Kimi K3 keeps that config value only to satisfy
the whitelist in ``TransformerConfig.__post_init__`` while supplying the real,
soft-clamped ``SituActivation`` through the module slot, and DeepSeek-V4 style
configs reach the eager ``glu()`` with a clamp bound. FLA's kernels implement
neither, so those configs fall through to Megatron's own code, as do GeLU-based
MLPs and MoE.

Toggle: ``args.use_fla_fused_swiglu`` (resolved by ``fla_runtime_patches.py``
from ``PRIMUS_FLA_SWIGLU`` / YAML ``use_fla_fused_swiglu``, default True).

Memory variant: ``args.use_fla_fused_swiglu_linear`` (default False) goes one
step further and fuses the activation *into* ``linear_fc2`` via FLA's
``swiglu_linear``. Megatron computes ``swiglu`` as its own op and then feeds the
result to ``linear_fc2``, which saves that ffn-wide tensor for its weight
gradient; ``swiglu_linear`` instead saves only the two fc1 halves and recomputes
the activation in the backward. That removes one ffn-wide tensor per MLP layer
(measured 0.188 GiB per unit micro-batch on GDN-300M, ffn 4096, seq 2048 -- the
entire measured activation gap vs FLA).

It is a trade, not a free win, which is why it stays off by default: recomputing
the activation costs time, and the cost grows with the micro-batch. Measured on
MI355X, 300M, seq 2048, 30 iters, median of three A/B repeats -- loss matched to
four decimals in every pair, and peak memory was bit-identical across repeats.
Both columns are deltas against the unfused path, so a positive iteration time
means the fused path is *slower* and a negative memory figure means it saves:

    model  mbs   iteration time   peak memory
    GDN      2       ~0%            -0.3 GB
    GDN     32     +4.4% slower     -5.3 GB
    KDA      2     +2.2% slower     -0.6 GB
    KDA     32     +5.1% slower     -5.8 GB

The saving tracks ~0.18 GiB per unit micro-batch, as predicted by the size of
the tensor that is no longer stashed for the fc2 weight gradient.

Correctness note: this bypasses ``linear_fc2``'s
``linear_with_grad_accumulation_and_async_allreduce``, so the fc2 weight
gradient lands in ``param.grad`` instead of being fused straight into
``param.main_grad``. That is safe -- Megatron's DDP backward post-hook folds
``param.grad`` into ``main_grad`` whenever ``grad_added_to_main_grad`` is False
-- but it does forgo the wgrad-accumulation fusion, so the path is gated to the
simple case it was measured on: TP=1, PP=1, no sequence parallel, no bias, no
per-token scaling, non-expert MLPs, no FP8 / ``delay_wgrad_compute`` /
``activation_func_fp8_input_store``, and no ``overlap_param_gather``. Everything
it bypasses lives in ``linear_fc2``: FP8 and delayed-wgrad execution inside
``TERowParallelLinear``, the deferred-wgrad enqueue the zero-bubble pipeline
schedules pop in their W phase, and the parameter all-gather hook. A PEFT
adapter can also wrap ``linear_fc2`` *after* ``__init__``, so the forward
rechecks that the module still exposes a plain ``.weight``.

Source-string rewrite style: the injection points sit inside ``MLP.__init__``
and the non-fused branch of ``MLP.forward``, which a plain wrapper cannot
reach without duplicating the whole (large) method body. The eligibility rules
themselves live in ``_resolve_fla_swiglu`` rather than in the injected string,
so they can be read and exercised as ordinary Python.
"""

import torch.nn.functional as F

from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched
from primus.backends.megatron.patches._source_patch_utils import (
    patch_method_source,
    patch_method_source_multi,
)
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

_PATCH_KEY = "megatron.mlp.fla_swiglu"


# `activation_func == F.silu` alone does not mean "plain SwiGLU". Kimi K3 keeps
# that value only to satisfy the whitelist in TransformerConfig.__post_init__
# while routing the real, soft-clamped activation through the `activation_func`
# module slot (`use_te_activation_func`, mlp.py:226); DeepSeek-V4 style configs
# instead reach the eager `glu()` with a clamp bound and/or a non-zero
# `glu_linear_offset`. FLA's kernels implement none of that, so both fused paths
# below are restricted to the plain, unclamped, zero-offset case and everything
# else falls through to Megatron's own code.
def _plain_swiglu_reject(config) -> str:
    """Why FLA's swiglu kernel cannot stand in for this MLP's activation, if so."""
    if not config.gated_linear_unit:
        return "gated_linear_unit is off"
    if config.activation_func is not F.silu:
        return "activation_func is not F.silu"
    if getattr(config, "use_te_activation_func", False):
        return "activation_func comes from the TE module slot (may be clamped)"
    if getattr(config, "activation_func_clamp_value", None) is not None:
        return "activation_func_clamp_value is set"
    if getattr(config, "glu_linear_offset", 0):
        return "glu_linear_offset is non-zero"
    return ""


def _fused_fc2_reject(config, args, is_expert: bool) -> str:
    """Why swiglu cannot be fused into linear_fc2 here, if it cannot.

    Everything this path bypasses lives in ``linear_fc2``, so each gate below
    names the mechanism that would silently be skipped.
    """
    plain = _plain_swiglu_reject(config)
    if plain:
        return plain
    if is_expert:
        return "expert MLP"
    if config.tensor_model_parallel_size != 1:
        return "tensor_model_parallel_size > 1 (fc2 is row-parallel)"
    if config.sequence_parallel:
        return "sequence_parallel is on"
    if config.add_bias_linear:
        return "add_bias_linear is on"
    # Zero-bubble / custom pipeline schedules expect linear_fc2's backward to
    # enqueue a deferred wgrad (WeightGradStore.split_bw); calling FLA's
    # autograd directly never enqueues, so the W phase pops an empty queue.
    if config.pipeline_model_parallel_size != 1:
        return "pipeline_model_parallel_size > 1 (deferred-wgrad schedules)"
    # FP8 and delayed-wgrad execution live inside TERowParallelLinear.
    if config.fp8:
        return "fp8 is enabled"
    if getattr(config, "delay_wgrad_compute", False):
        return "delay_wgrad_compute is on"
    if getattr(config, "activation_func_fp8_input_store", False):
        return "activation_func_fp8_input_store is on"
    # Reading linear_fc2.weight directly never triggers that module's parameter
    # all-gather hook, so the kernel could consume a still-sharded parameter.
    if getattr(args, "overlap_param_gather", False):
        return "overlap_param_gather is on"
    return ""


_DECISIONS_LOGGED = set()


def _resolve_fla_swiglu(config, is_expert: bool):
    """Pick the FLA SwiGLU kernels this MLP may use.

    Returns ``(swiglu_fn_or_None, swiglu_linear_fn_or_None)``. Kept a plain
    function of the config -- rather than inlined into the injected source
    string -- so the eligibility rules stay readable, and so a rejected opt-in
    is reported instead of silently falling back.
    """
    from megatron.training import get_args

    args = get_args()

    swiglu_fn = None
    if getattr(args, "use_fla_fused_swiglu", True) and not _plain_swiglu_reject(config):
        try:
            from fla.modules.activations import swiglu

            swiglu_fn = swiglu
        except ImportError:
            # use_fla_fused_swiglu defaults to on, so a missing FLA install is a
            # normal configuration rather than an error: leave swiglu_fn as None
            # and let Megatron's own activation run. The opt-in fc2 path below
            # does report its ImportError, since that one was asked for.
            swiglu_fn = None

    swiglu_linear_fn = None
    if getattr(args, "use_fla_fused_swiglu_linear", False):
        reason = _fused_fc2_reject(config, args, is_expert)
        if not reason:
            try:
                from fla.modules.activations import swiglu_linear

                swiglu_linear_fn = swiglu_linear
            except ImportError:
                reason = "flash-linear-attention is not installed"
        if reason not in _DECISIONS_LOGGED:
            _DECISIONS_LOGGED.add(reason)
            log_rank_0(
                f"[Patch:{_PATCH_KEY}] fused SwiGLU+fc2 "
                + ("active" if not reason else f"disabled: {reason}")
            )

    return swiglu_fn, swiglu_linear_fn


# Inserted right before linear_fc2 construction in MLP.__init__, mirroring
# where megatron_patches/03-mlp-fla-swiglu.patch spliced in its detection.
_INIT_ORI = "self.linear_fc2 = submodules.linear_fc2("
_INIT_NEW = (
    "from primus.backends.megatron.patches.mlp_fla_swiglu_patches import "
    "_resolve_fla_swiglu as _fla_resolve\n"
    "        self._fla_swiglu_fn, self._fla_swiglu_linear_fn = _fla_resolve(self.config, is_expert)\n"
    "        self._use_fla_swiglu = self._fla_swiglu_fn is not None\n"
    "        self._use_fla_swiglu_linear = self._fla_swiglu_linear_fn is not None\n"
    "\n        " + _INIT_ORI
)

# Inserted at the top of MLP.forward's activation section: when the fused
# swiglu+fc2 path is active we short-circuit both the activation and linear_fc2,
# so the ffn-wide activation output is never saved for backward.
_FWD_FUSED_ORI = 'nvtx_range_push(suffix="activation")'
_FWD_FUSED_NEW = (
    "if (getattr(self, '_use_fla_swiglu_linear', False)\n"
    "                and per_token_scale is None and bias_parallel is None\n"
    # PEFT wraps linear_fc2 after __init__ (LoRALinear exposes the base module as
    # `.to_wrap` and adds the adapter output in its forward), so the eligibility
    # decision has to be rechecked here: the wrapper has no `.weight`, and reading
    # through it would drop the adapter contribution anyway.
    "                and hasattr(self.linear_fc2, 'weight')):\n"
    "            x_glu, x_linear = torch.chunk(intermediate_parallel, 2, dim=-1)\n"
    "            output = self._fla_swiglu_linear_fn(\n"
    "                x_glu, x_linear, self.linear_fc2.weight, None\n"
    "            )\n"
    "            return output, None\n"
    "        " + _FWD_FUSED_ORI
)

# Inserted in MLP.forward's non-fused-bias-activation branch, in place of the
# unconditional `glu(intermediate_parallel)` call.
_FORWARD_ORI = "intermediate_parallel = glu(intermediate_parallel)"
_FORWARD_NEW = (
    "if self._use_fla_swiglu:\n"
    "                    x_glu, x_linear = torch.chunk(intermediate_parallel, 2, dim=-1)\n"
    "                    intermediate_parallel = self._fla_swiglu_fn(x_glu, x_linear)\n"
    "                else:\n"
    "                    " + _FORWARD_ORI
)


def _install_mlp_fla_swiglu_patch() -> None:
    from megatron.core.transformer.mlp import MLP

    if is_patched(MLP, _PATCH_KEY):
        log_rank_0(f"[Patch:{_PATCH_KEY}] MLP already patched; skipping.")
        return

    # Apply both rewrites or neither. If an anchor drifts on a Megatron bump the
    # second call raises, and a half-patched MLP is worse than an unpatched one:
    # __init__ would set the _use_fla_* flags that only the patched forward
    # reads, and a retry cannot re-read the source of an already-exec'd method.
    original_init, original_forward = MLP.__init__, MLP.forward
    try:
        patch_method_source(MLP, "__init__", _INIT_ORI, _INIT_NEW)
        # Both forward rewrites must go in one pass: a method can only be
        # source-patched once (inspect.getsource cannot re-read an exec'd function).
        patch_method_source_multi(
            MLP,
            "forward",
            [(_FWD_FUSED_ORI, _FWD_FUSED_NEW), (_FORWARD_ORI, _FORWARD_NEW)],
        )
    except Exception:
        MLP.__init__, MLP.forward = original_init, original_forward
        raise

    mark_patched(MLP, _PATCH_KEY)
    log_rank_0(
        f"[Patch:{_PATCH_KEY}] Patched MLP.__init__/forward to use FLA's Triton "
        "swiglu kernel when use_fla_fused_swiglu is set, and to fuse swiglu into "
        "linear_fc2 when use_fla_fused_swiglu_linear is set."
    )


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description=(
        "Route MLP's gated-linear-unit activation through FLA's Triton-fused "
        "swiglu kernel instead of Megatron's naive silu*x implementation."
    ),
    # Runs after fla_runtime_knobs (priority=-100) has resolved args.use_fla_fused_swiglu.
    priority=50,
    condition=lambda ctx: getattr(get_args(ctx), "use_fla_fused_swiglu", False)
    or getattr(get_args(ctx), "use_fla_fused_swiglu_linear", False),
)
def patch_mlp_fla_swiglu(ctx: PatchContext) -> None:
    _install_mlp_fla_swiglu_patch()
