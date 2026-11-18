###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Reference harness: the official transformers Inkling next to the Megatron one.

The reference is ``transformers.models.inkling`` (``InklingForCausalLM``) with
``attn_implementation="eager"`` -- the only backend that applies the
relative-position bias; sdpa/flex silently drop it. transformers releases up to
5.10 do not ship Inkling, so the tests that need it skip unless it imports;
install a pinned main build into a private directory and put it first on
``PYTHONPATH``::

    pip install --no-deps --target /tmp/inkling_ref/site \\
        "git+https://github.com/huggingface/transformers@263c5eb913849847dcff1885b7b5e387c483f844" \\
        "tokenizers>=0.23.1,<0.24"
    PYTHONPATH=/tmp/inkling_ref/site pytest tests/unit_tests/backends/megatron/inkling

Weights and gradients cross over through the importer itself
(``transformers_to_release`` then ``release_to_megatron``), so the harness
compares the two implementations through the same path a real checkpoint
takes.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from primus.backends.megatron.core.models.inkling.inkling_hf_mapping import (
    InklingGeometry,
    release_to_megatron,
    transformers_to_release,
)


def transformers_inkling_available() -> bool:
    try:
        from transformers.models.inkling import modeling_inkling  # noqa: F401
    except Exception:
        return False
    return True


def reference_modeling():
    """``modeling_inkling`` with its short convolution on the plain-torch path.

    transformers swaps in the ``causal_conv1d`` package when it is installed
    (CUDA-only, and a different kernel from the one being specified); the
    reference uses the torch function it falls back to.
    """
    from transformers.models.inkling import modeling_inkling as m

    for name in ("causal_conv1d_fn", "causal_conv1d_update"):
        fn = getattr(m, name)
        setattr(m, name, getattr(fn, "__wrapped__", fn))
    return m


def hf_text_config(config, vocab_size: int, **overrides):
    """``InklingTextConfig`` equivalent to an :class:`InklingTransformerConfig`."""
    from transformers.models.inkling.configuration_inkling import InklingTextConfig

    n = int(config.num_layers)
    unpadded = config.unpadded_vocab_size
    kwargs = dict(
        vocab_size=vocab_size,
        unpadded_vocab_size=unpadded if unpadded and unpadded < vocab_size else None,
        hidden_size=config.hidden_size,
        num_hidden_layers=n,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_query_groups,
        head_dim=config.kv_channels,
        swa_num_attention_heads=config.swa_num_attention_heads,
        swa_num_key_value_heads=config.swa_num_query_groups,
        swa_head_dim=config.swa_kv_channels,
        sliding_window_size=config.sliding_window_size,
        d_rel=config.d_rel,
        rel_extent=config.rel_extent,
        log_scaling_n_floor=config.log_scaling_n_floor,
        log_scaling_alpha=config.log_scaling_alpha,
        local_layer_ids=list(config.local_layer_ids),
        mlp_layer_types=["sparse" if config.is_moe_layer(i) else "dense" for i in range(n)],
        rms_norm_eps=config.layernorm_epsilon,
        conv_kernel_size=config.short_conv_kernel_size,
        intermediate_size=config.ffn_hidden_size,
        moe_intermediate_size=config.moe_ffn_hidden_size,
        n_routed_experts=config.num_moe_experts,
        num_experts_per_tok=config.moe_router_topk,
        n_shared_experts=config.moe_num_shared_experts,
        route_scale=config.moe_router_topk_scaling_factor,
        logits_mup_width_multiplier=config.logits_mup_width_multiplier,
        rms_norm_eps_moe_gate=config.rms_norm_eps_moe_gate,
        initializer_range=config.init_method_std,
        attention_dropout=0.0,
    )
    kwargs.update(overrides)
    cfg = InklingTextConfig(**kwargs)
    cfg._attn_implementation = "eager"
    return cfg


def _numeric_scale(name: str):
    """``(mean, std)`` for :func:`build_reference_model` with ``numeric=True``."""
    if name.endswith("norm.weight") or name.endswith("global_scale"):
        return 1.0, 0.1
    if "sconv" in name:
        return 0.0, 0.2
    if "rel_logits_proj" in name:
        return 0.0, 0.5
    return 0.0, 0.05


def build_reference_model(
    config,
    vocab_size: int,
    *,
    dtype=torch.float32,
    device="cpu",
    seed: int = 0,
    numeric: bool = False,
    **config_overrides,
):
    """Official ``InklingForCausalLM`` (eager) with every tensor randomised.

    The stock init leaves norms, scales and the router bias at constants,
    which would hide a permuted mapping; every parameter and buffer gets
    distinct values instead. The default (``0.02 * randn`` everywhere) is for
    layout checks; ``numeric=True`` keeps norms and scales near 1 and gives
    the short convs and the relative bias enough weight that a forward
    comparison exercises them.
    """
    torch.manual_seed(seed)
    model = (
        reference_modeling()
        .InklingForCausalLM(hf_text_config(config, vocab_size, **config_overrides))
        .to(dtype=dtype, device=device)
    )
    with torch.no_grad():
        tensors = list(model.named_parameters()) + list(model.named_buffers())
        for name, t in tensors:
            if t.is_floating_point():
                mean, std = _numeric_scale(name) if numeric else (0.0, 0.02)
                t.copy_(mean + torch.randn_like(t) * std)
    return model.eval()


_RELEASE_TO_TRANSFORMERS = (
    (r"^model\.llm\.embed\.weight$", "model.embed_tokens.weight"),
    (r"^model\.llm\.embed_norm\.weight$", "model.embed_tokens.embed_norm.weight"),
    (r"^model\.llm\.unembed\.weight$", "lm_head.weight"),
    (r"^model\.llm\.", "model."),
    (r"\.attn\.wq_du\.", ".self_attn.q_proj."),
    (r"\.attn\.wk_dv\.", ".self_attn.k_proj."),
    (r"\.attn\.wv_dv\.", ".self_attn.v_proj."),
    (r"\.attn\.wr_du\.", ".self_attn.r_proj."),
    (r"\.attn\.wo_ud\.", ".self_attn.o_proj."),
    (r"\.attn\.(q_norm|k_norm)\.", r".self_attn.\1."),
    (r"\.attn\.(k_sconv|v_sconv)\.", r".self_attn.\1.conv1d."),
    (r"\.attn\.rel_logits_proj\.", ".self_attn.rel_logits_proj."),
    (r"\.(attn_sconv|mlp_sconv)\.weight$", r".\1.conv1d.weight"),
    (r"\.attn_norm\.", ".input_layernorm."),
    (r"\.mlp_norm\.", ".post_attention_layernorm."),
    (r"\.mlp\.gate\.bias$", ".mlp.gate.e_score_correction_bias"),
    (r"\.mlp\.w2_md\.weight$", ".mlp.down_proj.weight"),
    (r"\.mlp\.experts\.w2_weight$", ".mlp.experts.down_proj"),
    (r"\.shared_experts\.shared_w2_weight$", ".shared_experts.down_proj"),
)


def release_to_transformers(release: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Released text weights in ``InklingForCausalLM`` naming, independently of the importer.

    The renames follow transformers' ``inkling`` conversion mapping, and the
    gate/up splits run transformers' own ``Interleave`` / ``Chunk`` ops.
    """
    import re

    from transformers.core_model_loading import Interleave

    def deinterleave(t, dim):
        return Interleave(dim=dim).convert({"w": t}, ["w"], ["w"])["w"]

    out = {}
    for key, value in release.items():
        if not key.startswith("model.llm."):
            continue
        name = key
        for pattern, repl in _RELEASE_TO_TRANSFORMERS:
            name = re.sub(pattern, repl, name)
        if name.endswith(".mlp.w13_dn.weight"):
            stem = name[: -len("w13_dn.weight")]
            gate, up = deinterleave(value, 0).chunk(2, dim=0)
            out[stem + "gate_proj.weight"], out[stem + "up_proj.weight"] = gate, up
        elif name.endswith(".mlp.experts.w13_weight"):
            out[name[: -len("w13_weight")] + "gate_up_proj"] = deinterleave(value, 1)
        elif name.endswith(".shared_experts.shared_w13_weight"):
            stem = name[: -len("shared_w13_weight")]
            gate, up = deinterleave(value, 1).chunk(2, dim=1)
            out[stem + "gate_proj"], out[stem + "up_proj"] = gate, up
        else:
            out[name] = value
    return {k: v.contiguous() for k, v in out.items()}


def reference_weights_for_megatron(
    model, config, *, target_vocab_size: Optional[int] = None
) -> Dict[str, torch.Tensor]:
    """The reference model's weights in Megatron layout, via the importer."""
    release = transformers_to_release(model.state_dict())
    return release_to_megatron(
        release, InklingGeometry.from_megatron_config(config), target_vocab_size=target_vocab_size
    )


def reference_forward_backward(model, input_ids: torch.Tensor, config):
    """Run the reference on ``input_ids`` (next-token loss).

    Returns ``(logits, loss, grads)`` with ``grads`` mapped into Megatron
    layout by the same importer that maps the weights.
    """
    model.zero_grad(set_to_none=True)
    out = model(input_ids=input_ids, labels=input_ids)
    out.loss.backward()
    grads = {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}
    for name, buf in model.state_dict().items():
        grads.setdefault(name, torch.zeros_like(buf))
    mapped = release_to_megatron(transformers_to_release(grads), InklingGeometry.from_megatron_config(config))
    return out.logits.detach(), out.loss.detach(), mapped


def megatron_forward_backward(model, input_ids: torch.Tensor):
    """The Megatron model on the reference's objective: mean next-token loss.

    Returns ``(logits, loss, grads)``; ``logits`` come from a separate
    no-grad pass without labels, the loss through Megatron's vocab-parallel
    cross entropy.
    """
    with torch.no_grad():
        logits = model(input_ids, None, None)
    labels = torch.zeros_like(input_ids)
    labels[:, :-1] = input_ids[:, 1:]
    loss_mask = torch.ones_like(input_ids, dtype=torch.float32)
    loss_mask[:, -1] = 0
    model.zero_grad(set_to_none=True)
    per_token = model(input_ids, None, None, labels=labels)
    loss = (per_token.float() * loss_mask).sum() / loss_mask.sum()
    loss.backward()
    grads = {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}
    return logits, loss.detach(), grads
