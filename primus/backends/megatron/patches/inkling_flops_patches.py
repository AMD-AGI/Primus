###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Inkling FLOPs reporting patch.

Megatron's ``num_floating_point_operations`` models every layer as GPT attention plus
Megatron MoE. For Inkling it gets these terms wrong:

* the sliding-window layers (``inkling_local_layer_ids``) use their own head / KV-group
  counts and score at most ``inkling_sliding_window`` keys, not ``seq_length / 2``;
* the relative-position projection (``r_proj``, hidden -> heads x ``inkling_d_rel``) and
  the relative logits (``d_rel`` x ``E`` per head, where ``E`` is the relative-bias table
  extent: ``inkling_rel_extent`` on global layers, ``inkling_sliding_window`` on local
  layers; every query computes its whole table row) are not counted;
* the ``inkling_num_shared_experts`` shared (sink) experts run on every token, and the
  router also scores them (``num_experts + inkling_num_shared_experts`` gate rows);
* the multi-token-prediction depths (``mtp_num_layers``: input projection, one Inkling block with
  a dense SwiGLU MLP, and one more output-layer pass each).

The closed form below counts FMACs per token (matmuls only; norms, short convolutions and softmax
are left out, as upstream does) and applies the same ``3 (fwd + bwd) x 2 (FMA)`` factor.
"""

from __future__ import annotations

import copy
from typing import Any, Dict

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.flops_utils import local_visible_pairs
from primus.core.utils.module_utils import log_rank_0

_FORWARD_BACKWARD_FACTOR = 3
_FMA_FACTOR = 2


def _mean_keys(seq_length: int, window: int) -> float:
    """Mean number of keys a causal query scores (inside ``window``; 0 = full causal); the pair
    count is the shared ``flops_utils.local_visible_pairs``."""
    s = int(seq_length)
    return local_visible_pairs(window, s) / s


def attention_fmac_per_token(args: Any, local: bool, mtp: bool = False) -> float:
    """``mtp``: an MTP sliding depth (its own head count and window, when set)."""
    h = int(args.hidden_size)
    if local:
        nh = int(args.inkling_swa_num_attention_heads)
        nkv = int(args.inkling_swa_num_query_groups)
        hd = int(args.inkling_swa_kv_channels)
        window = int(args.inkling_sliding_window)
        if mtp:
            nh = int(getattr(args, "inkling_mtp_swa_num_attention_heads", None) or nh)
            window = int(getattr(args, "inkling_mtp_local_extent", None) or window)
        extent = window
    else:
        nh, hd = int(args.num_attention_heads), int(args.kv_channels)
        # As upstream: with group_query_attention off, Megatron leaves num_query_groups at its
        # default and every head has its own K / V.
        nkv = int(args.num_query_groups) if getattr(args, "group_query_attention", False) else nh
        window = 0  # full causal
        extent = int(args.inkling_rel_extent)
    d_rel = int(args.inkling_d_rel)
    keys = _mean_keys(args.seq_length, window)
    proj = h * (nh * hd + 2 * nkv * hd + nh * d_rel) + nh * hd * h
    core = nh * keys * 2 * hd  # q.k and p.v over the keys each query scores
    rel_logits = nh * extent * d_rel  # r @ rel_proj: one full table row per query and head
    return proj + core + rel_logits


def moe_fmac_per_token(args: Any) -> int:
    h = int(args.hidden_size)
    ffn = int(args.moe_ffn_hidden_size)
    shared = int(args.inkling_num_shared_experts)
    active = int(args.moe_router_topk) + shared
    router = h * (int(args.num_experts) + shared)  # the gate scores routed and shared experts
    return router + active * 3 * h * ffn  # router + SwiGLU experts


def dense_mlp_fmac_per_token(args: Any) -> int:
    """The dense SwiGLU MLP of an MTP depth (gate, up, down)."""
    return 3 * int(args.hidden_size) * int(getattr(args, "inkling_dense_intermediate_size", None) or 3072)


def mtp_fmac_per_token(args: Any) -> Dict[str, float]:
    """Per MTP depth: the input projection (2h -> h), one Inkling block (attention + dense MLP)
    and one more pass through the shared output layer (its loss)."""
    depths = int(getattr(args, "mtp_num_layers", None) or 0)
    if not depths:
        return {}
    h = int(args.hidden_size)
    local_ids = {int(i) for i in (getattr(args, "inkling_mtp_local_layer_ids", None) or [])}
    n_local = sum(1 for d in range(depths) if d in local_ids)
    vocab = int(getattr(args, "padded_vocab_size", None) or args.vocab_size)
    return {
        "mtp_attn": (depths - n_local) * attention_fmac_per_token(args, local=False)
        + n_local * attention_fmac_per_token(args, local=True, mtp=True),
        "mtp_mlp": depths * (dense_mlp_fmac_per_token(args) + 2 * h * h),
        "mtp_logits": depths * h * vocab,
    }


def compute_inkling_flops(args: Any, batch_size: int):
    num_layers = int(args.num_layers)
    local_ids = {int(i) for i in (args.inkling_local_layer_ids or [])}
    n_local = sum(1 for i in range(num_layers) if i in local_ids)
    n_global = num_layers - n_local
    vocab = int(getattr(args, "padded_vocab_size", None) or args.vocab_size)
    per_token: Dict[str, float] = {
        "attn_global": n_global * attention_fmac_per_token(args, local=False),
        "attn_local": n_local * attention_fmac_per_token(args, local=True),
        "moe": num_layers * moe_fmac_per_token(args),
        "logits": int(args.hidden_size) * vocab,
        **mtp_fmac_per_token(args),
    }
    tokens = int(batch_size) * int(args.seq_length)
    breakdown = {k: v * tokens * _FORWARD_BACKWARD_FACTOR * _FMA_FACTOR for k, v in per_token.items()}
    return int(sum(breakdown.values())), breakdown


_REQUIRED_INT_ARGS = (
    "num_layers",
    "hidden_size",
    "seq_length",
    "num_attention_heads",
    "num_query_groups",
    "kv_channels",
    "inkling_swa_num_attention_heads",
    "inkling_swa_num_query_groups",
    "inkling_swa_kv_channels",
    "inkling_sliding_window",
    "inkling_rel_extent",
    "inkling_d_rel",
    "moe_ffn_hidden_size",
    "moe_router_topk",
    "inkling_num_shared_experts",
    "num_experts",
)


def _validate_flops_args(args: Any) -> None:
    """Raise ``ValueError`` when an arg the closed form reads is unset or not an int >= 1
    (``mtp_num_layers`` may be unset or 0; when set it must be an int >= 1).

    This mirrors InklingTransformerConfig, which also requires ``inkling_num_shared_experts``
    >= 1. The vocab size is not checked here: it is padded after the patches run, so the count
    reads it at call time."""
    bad = {}
    for k in _REQUIRED_INT_ARGS:
        v = getattr(args, k, None)
        try:
            ok = v is not None and int(v) >= 1
        except (TypeError, ValueError):
            ok = False
        if not ok:
            bad[k] = v
    mtp = getattr(args, "mtp_num_layers", None)
    if mtp is not None and (type(mtp) is not int or mtp < 0):
        bad["mtp_num_layers"] = mtp
    if bad:
        raise ValueError(
            "[Patch:megatron.inkling.flops_reporting] the Inkling FLOPs count cannot be computed "
            f"from these args: {bad}"
        )


def _make_wrapper(original_fn):
    """Wrap Megatron's ``num_floating_point_operations``; each wrapper logs its breakdown once."""
    logged = False

    def wrapped(args, batch_size):
        nonlocal logged
        total, breakdown = compute_inkling_flops(args, batch_size)
        if not logged:
            logged = True
            try:
                # On a copy: upstream may set args.num_query_groups (group_query_attention off).
                upstream = f"{original_fn(copy.copy(args), batch_size) / 1e12:.4f}"
            except Exception as exc:  # noqa: BLE001 - the upstream comparison is a logging extra only
                upstream = f"n/a ({type(exc).__name__}: {exc})"
            rows = ", ".join(f"{k}={v / 1e12:.4f}" for k, v in breakdown.items())
            log_rank_0(
                f"[Patch:megatron.inkling.flops_reporting] TFLOP / global batch: {rows}, "
                f"total={total / 1e12:.4f}, upstream={upstream}"
            )
        return total

    wrapped.__wrapped__ = original_fn
    wrapped._inkling_flops_wrapper = True
    return wrapped


@register_patch(
    "megatron.inkling.flops_reporting",
    backend="megatron",
    phase="before_train",
    description=(
        "Inkling: report FLOPs with a closed form that counts the sliding-window layers, "
        "the relative-position terms and the shared experts."
    ),
    condition=lambda ctx: getattr(get_args(ctx), "model_type", None) == "inkling",
)
def patch_inkling_flops_reporting(ctx: PatchContext):
    import megatron.training.training as training_module

    current = training_module.num_floating_point_operations
    if getattr(current, "_inkling_flops_wrapper", False):
        log_rank_0("[Patch:megatron.inkling.flops_reporting] already wrapped; not wrapping again")
        return
    # Validate the args the closed form reads once, here: a bad value (seq_length 0, an unset
    # inkling_* / moe_* field) then fails at start-up, not at the first throughput log.
    _validate_flops_args(get_args(ctx))
    training_module.num_floating_point_operations = _make_wrapper(current)
    log_rank_0("[Patch:megatron.inkling.flops_reporting] wrapped num_floating_point_operations")
