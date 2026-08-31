###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""TEMPORARY verification-only patches for Gemma 4. Not for merge.

``PRIMUS_GEMMA4_DUMP_MODEL=1``
    After the provider builds the model, dump the instantiated module tree,
    per-parameter shapes, and the resolved per-layer attention geometry
    (head dim, KV groups, window, RoPE base, core-attention class). Used to
    check the built graph against the published HF config.

``PRIMUS_GEMMA4_PROFILE=<start>:<end>``
    Enable the built-in PyTorch profiler over that global-step range and write a
    chrome trace next to the tensorboard directory.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0

_DUMPED = {"done": False}


def _is_rank0() -> bool:
    try:
        import torch.distributed as dist

        return not dist.is_initialized() or dist.get_rank() == 0
    except Exception:
        return True


def _fmt(n: int) -> str:
    return f"{n:,}"


def _describe_attention(layer: Any, layer_number: int) -> str:
    attn = getattr(layer, "self_attention", None)
    if attn is None:
        for name in ("self_attention_sliding", "self_attention_global"):
            attn = getattr(layer, name, None)
            if attn is not None:
                break
    if attn is None:
        return f"    layer {layer_number:>3}: <no self_attention>"

    cfg = getattr(attn, "config", None)
    core = getattr(attn, "core_attention", None)
    window = getattr(cfg, "window_size", None) if cfg else None
    kvc = getattr(cfg, "kv_channels", None) if cfg else None
    nqg = getattr(cfg, "num_query_groups", None) if cfg else None
    nah = getattr(cfg, "num_attention_heads", None) if cfg else None

    qkv = getattr(attn, "linear_qkv", None)
    qkv_shape = tuple(qkv.weight.shape) if qkv is not None and hasattr(qkv, "weight") else None
    proj = getattr(attn, "linear_proj", None)
    proj_shape = tuple(proj.weight.shape) if proj is not None and hasattr(proj, "weight") else None

    return (
        f"    layer {layer_number:>3}: heads={nah} kv_groups={nqg} head_dim={kvc} "
        f"window={window} core={type(core).__name__} "
        f"qkv={qkv_shape} proj={proj_shape}"
    )


def _dump_model(model: Any, tag: str) -> None:
    if _DUMPED["done"] or not _is_rank0():
        return
    _DUMPED["done"] = True

    lines: list[str] = []
    lines.append("=" * 100)
    lines.append(f"GEMMA4 MODEL DUMP [{tag}] type={type(model).__name__}")
    lines.append("=" * 100)

    # 1. Top-level children, so extra/missing towers (vision, audio, PLE) are visible.
    lines.append("-- top-level modules --")
    for name, child in model.named_children():
        n_params = sum(p.numel() for p in child.parameters(recurse=True))
        lines.append(f"    {name:<28} {type(child).__name__:<38} params={_fmt(n_params)}")

    # 2. Per-layer attention geometry.
    decoder = getattr(model, "decoder", None)
    layers = list(getattr(decoder, "layers", [])) if decoder is not None else []
    lines.append(f"-- per-layer attention geometry ({len(layers)} layers built) --")
    for layer in layers:
        ln = getattr(layer, "layer_number", -1)
        lines.append(_describe_attention(layer, ln))

    # 3. Full module inventory of layer 0 and the first global layer, with shapes.
    for probe_idx, probe_label in ((0, "first layer"), (5, "6th layer (expected global)")):
        if probe_idx >= len(layers):
            continue
        layer = layers[probe_idx]
        lines.append(f"-- module tree: {probe_label} (layer_number={getattr(layer, 'layer_number', '?')}) --")
        for name, mod in layer.named_modules():
            if not name:
                continue
            own = [(pn, tuple(p.shape)) for pn, p in mod.named_parameters(recurse=False)]
            if own:
                shapes = " ".join(f"{pn}{list(s)}" for pn, s in own)
                lines.append(f"    {name:<52} {type(mod).__name__:<34} {shapes}")
            else:
                lines.append(f"    {name:<52} {type(mod).__name__}")

    # 4. Parameter totals by category.
    lines.append("-- parameter totals --")
    total = 0
    by_bucket: dict[str, int] = {}
    for pname, p in model.named_parameters():
        total += p.numel()
        if ".layers." in pname:
            bucket = "decoder.layers"
        elif "embedding" in pname:
            bucket = "embedding"
        elif "output_layer" in pname:
            bucket = "output_layer"
        else:
            bucket = "other"
        by_bucket[bucket] = by_bucket.get(bucket, 0) + p.numel()
    for k in sorted(by_bucket):
        lines.append(f"    {k:<24} {_fmt(by_bucket[k])}")
    lines.append(f"    {'TOTAL (this rank)':<24} {_fmt(total)}")

    # 5. Things that are easy to silently drop.
    lines.append("-- feature checks --")
    out_layer = getattr(model, "output_layer", None)
    lines.append(f"    output_layer class            : {type(out_layer).__name__}")
    lines.append(
        f"    rotary_pos_emb class          : {type(getattr(model, 'rotary_pos_emb', None)).__name__}"
    )
    lines.append(
        f"    share_embeddings_and_output   : {getattr(model, 'share_embeddings_and_output_weights', None)}"
    )
    lines.append(f"    has per_layer_embedding (PLE) : {hasattr(model, 'per_layer_embedding')}")
    if layers:
        l0 = layers[0]
        for attr in (
            "input_layernorm",
            "post_self_attn_layernorm",
            "pre_mlp_layernorm",
            "post_mlp_layernorm",
            "post_per_layer_input_norm",
        ):
            mod = getattr(l0, attr, None)
            lines.append(f"    layer0.{attr:<22} : {type(mod).__name__ if mod is not None else None}")
        attn0 = getattr(l0, "self_attention", None)
        for attr in ("q_layernorm", "k_layernorm"):
            mod = getattr(attn0, attr, None) if attn0 is not None else None
            lines.append(
                f"    layer0.self_attention.{attr:<8} : {type(mod).__name__ if mod is not None else None}"
            )

    lines.append("=" * 100)
    text = "\n".join(lines)
    log_rank_0(text)
    out = os.environ.get("PRIMUS_GEMMA4_DUMP_PATH")
    if out:
        try:
            with open(out, "w") as fh:
                fh.write(text + "\n")
        except OSError:
            pass


@register_patch(
    "gemma4.diag.dump_model",
    backend="megatron_bridge",
    phase="setup",
    description="Verification only: dump the instantiated Gemma 4 module tree and attention geometry",
)
def patch_gemma4_dump_model(ctx: PatchContext) -> None:
    if os.environ.get("PRIMUS_GEMMA4_DUMP_MODEL", "") != "1":
        return

    try:
        from megatron.bridge.models.gemma.gemma4_provider import (
            Gemma4DenseProvider,
            Gemma4ModelProvider,
        )
    except Exception:
        return

    for cls, method, tag in (
        (Gemma4DenseProvider, "build", "dense"),
        (Gemma4ModelProvider, "provide", "moe"),
    ):
        original = cls.__dict__.get(method)
        if original is None or getattr(original, "_primus_gemma4_dump", False):
            continue

        def make(original=original, tag=tag):
            def wrapper(self, *args, **kwargs):
                model = original(self, *args, **kwargs)
                try:
                    _dump_model(model, tag)
                except Exception as exc:  # diagnostics must never break training
                    log_rank_0(f"[Patch:gemma4.diag.dump_model] dump failed: {exc}")
                return model

            wrapper._primus_gemma4_dump = True
            return wrapper

        setattr(cls, method, make())
        log_rank_0(f"[Patch:gemma4.diag.dump_model] wrapped {cls.__name__}.{method}")


@register_patch(
    "gemma4.diag.profile",
    backend="megatron_bridge",
    phase="setup",
    description="Verification only: enable the PyTorch profiler over PRIMUS_GEMMA4_PROFILE=<start>:<end>",
)
def patch_gemma4_profile(ctx: PatchContext) -> None:
    spec = os.environ.get("PRIMUS_GEMMA4_PROFILE", "")
    if not spec:
        return
    try:
        start_s, end_s = spec.split(":")
        start, end = int(start_s), int(end_s)
    except ValueError:
        log_rank_0(f"[Patch:gemma4.diag.profile] bad PRIMUS_GEMMA4_PROFILE={spec!r}, expected <start>:<end>")
        return

    try:
        from megatron.bridge.training import pretrain as pretrain_mod
        from megatron.bridge.training.config import ProfilingConfig
    except Exception:
        return

    original = getattr(pretrain_mod, "pretrain", None)
    if original is None or getattr(original, "_primus_gemma4_profile", False):
        return

    def pretrain(cfg, *args, **kwargs):
        cfg.profiling = ProfilingConfig(
            use_pytorch_profiler=True,
            use_nsys_profiler=False,
            profile_step_start=start,
            profile_step_end=end,
            profile_ranks=[0],
            pytorch_profiler_collect_shapes=True,
        )
        log_rank_0(f"[Patch:gemma4.diag.profile] PyTorch profiler enabled for steps {start}..{end} on rank 0")
        return original(cfg, *args, **kwargs)

    pretrain._primus_gemma4_profile = True
    pretrain_mod.pretrain = pretrain
    log_rank_0("[Patch:gemma4.diag.profile] wrapped megatron.bridge.training.pretrain.pretrain")
