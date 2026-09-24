# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""
HuggingFace to Primus WAN checkpoint converter.

Converts HuggingFace Diffusers WAN (2.1 / 2.2) video-diffusion transformer
checkpoints to Primus/Megatron-Core compatible format, and back.

Key Conversion:
    - HuggingFace: split ``to_q`` / ``to_k`` / ``to_v`` projections, ``to_out.0``
    - Primus: fused ``linear_qkv`` (self-attention) / ``linear_kv`` (cross-attention)
      in Megatron's per-head interleaved layout, ``linear_proj``

    Every other parameter keeps its diffusers name (patch embedding, condition
    embedder, ``norm2``, ``scale_shift_table``, ``proj_out``); the feed-forward
    network is a pure rename onto Megatron-Core's ``MLP``
    (``ffn.net.0.proj`` -> ``ffn.linear_fc1``, ``ffn.net.2`` -> ``ffn.linear_fc2``).

WAN 2.2 dual-expert checkpoints are handled automatically: keys prefixed with
``transformer.`` and ``transformer_2.`` are each converted in place.

Both directions validate against the analytic key set for the target layout and
return a :class:`ConversionReport` recording ``mapped`` / ``dropped`` /
``missing`` keys. ``strict=True`` raises if anything is dropped or missing.

Usage:
    from primus.backends.megatron.core.models.diffusion.wan.checkpoint_converter import (
        convert_hf_checkpoint,
    )

    primus_state_dict = convert_hf_checkpoint(
        checkpoint_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers/transformer",
        wan_config=config,
        save_to="primus_wan21_t2v_1_3b.safetensors",
    )

Plus a CLI for both directions, distributed-checkpoint input, and diffusers
directory export:

    python -m primus.backends.megatron.core.models.diffusion.wan.checkpoint_converter \\
        --direction hf2primus --input <ckpt> --output <out.safetensors>

Reference:
    - HuggingFace Diffusers ``WanTransformer3DModel``
    - Megatron-Core fused attention (``linear_qkv`` / ``linear_kv``) layout
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
from torch import Tensor

from primus.backends.megatron.core.models.diffusion.wan.config import WanConfig

logger = logging.getLogger(__name__)

StateDict = Dict[str, Tensor]


# ---------------------------------------------------------------------------
# Conversion report
# ---------------------------------------------------------------------------


@dataclass
class ConversionReport:
    """Records exactly what happened during a conversion."""

    direction: str
    mapped: List[str] = field(default_factory=list)
    dropped: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)
        logger.warning("[wan-convert:%s] %s", self.direction, msg)

    def summary(self) -> str:
        return (
            f"WAN checkpoint conversion ({self.direction}): "
            f"{len(self.mapped)} mapped, {len(self.dropped)} dropped, "
            f"{len(self.missing)} missing, {len(self.warnings)} warnings."
        )

    def raise_if_lossy(self) -> None:
        """Raise when keys were dropped or missing (strict mode)."""
        if self.dropped or self.missing:
            raise RuntimeError(
                self.summary()
                + " strict=True forbids lossy conversion. "
                + f"dropped={self.dropped[:5]}... missing={self.missing[:5]}..."
            )


# ---------------------------------------------------------------------------
# Expected key sets
#
# The Primus WAN backbone keeps most non-attention keys identical to diffusers
# (patch_embedding, condition_embedder, norms, scale_shift_table, proj_out) and
# differs in the attention submodules and the feed-forward network:
#
#   self-attn  (attn1):
#       to_q / to_k / to_v [inner, dim]  ->  linear_qkv [3*inner, dim]  (fused)
#       to_out.0                         ->  linear_proj
#       norm_q / norm_k                  ->  q_layernorm / k_layernorm
#   cross-attn (attn2):
#       to_q                             ->  linear_q
#       to_k / to_v [inner, dim]         ->  linear_kv [2*inner, dim]   (fused)
#       to_out.0                         ->  linear_proj
#       norm_q / norm_k                  ->  q_layernorm / k_layernorm
#   ffn (diffusers nn.Linear FeedForward -> mcore MLP; pure 1:1 rename):
#       net.0.proj [ffn_hidden, dim]     ->  linear_fc1
#       net.2      [dim, ffn_hidden]     ->  linear_fc2
#
# The fused layout is Megatron's per-head interleave: rows are grouped by head as
# ``[q_head0(hn), k_head0(hn), v_head0(hn), q_head1(hn), ...]``, matching the
# ``view(S, B, heads, 3*head_dim)`` split in the backbone's self-attention (and
# ``2*head_dim`` for cross-attention ``linear_kv``). q/k RMSNorm is across-heads,
# so its weight is ``[inner]`` and maps 1:1 with diffusers ``norm_q`` / ``norm_k``.
# ---------------------------------------------------------------------------

_SHARED_ROOT_KEYS: Tuple[str, ...] = (
    "patch_embedding.weight",
    "patch_embedding.bias",
    "condition_embedder.time_embedder.linear_1.weight",
    "condition_embedder.time_embedder.linear_1.bias",
    "condition_embedder.time_embedder.linear_2.weight",
    "condition_embedder.time_embedder.linear_2.bias",
    "condition_embedder.time_proj.weight",
    "condition_embedder.time_proj.bias",
    "condition_embedder.text_embedder.linear_1.weight",
    "condition_embedder.text_embedder.linear_1.bias",
    "condition_embedder.text_embedder.linear_2.weight",
    "condition_embedder.text_embedder.linear_2.bias",
    "proj_out.weight",
    "proj_out.bias",
    "scale_shift_table",
)

_SHARED_BLOCK_SUFFIXES: Tuple[str, ...] = ("scale_shift_table",)

_MCORE_BLOCK_SUFFIXES: Tuple[str, ...] = (
    "attn1.linear_qkv.weight",
    "attn1.linear_qkv.bias",
    "attn1.q_layernorm.weight",
    "attn1.k_layernorm.weight",
    "attn1.linear_proj.weight",
    "attn1.linear_proj.bias",
    "attn2.linear_q.weight",
    "attn2.linear_q.bias",
    "attn2.linear_kv.weight",
    "attn2.linear_kv.bias",
    "attn2.q_layernorm.weight",
    "attn2.k_layernorm.weight",
    "attn2.linear_proj.weight",
    "attn2.linear_proj.bias",
    "ffn.linear_fc1.weight",
    "ffn.linear_fc1.bias",
    "ffn.linear_fc2.weight",
    "ffn.linear_fc2.bias",
)

_DIFFUSERS_BLOCK_SUFFIXES: Tuple[str, ...] = (
    "attn1.to_q.weight",
    "attn1.to_q.bias",
    "attn1.to_k.weight",
    "attn1.to_k.bias",
    "attn1.to_v.weight",
    "attn1.to_v.bias",
    "attn1.norm_q.weight",
    "attn1.norm_k.weight",
    "attn1.to_out.0.weight",
    "attn1.to_out.0.bias",
    "attn2.to_q.weight",
    "attn2.to_q.bias",
    "attn2.to_k.weight",
    "attn2.to_k.bias",
    "attn2.to_v.weight",
    "attn2.to_v.bias",
    "attn2.norm_q.weight",
    "attn2.norm_k.weight",
    "attn2.to_out.0.weight",
    "attn2.to_out.0.bias",
    "ffn.net.0.proj.weight",
    "ffn.net.0.proj.bias",
    "ffn.net.2.weight",
    "ffn.net.2.bias",
)


def _build_expected_keys(config: WanConfig, block_suffixes: Tuple[str, ...]) -> Set[str]:
    keys: Set[str] = set(_SHARED_ROOT_KEYS)
    cross_attn_norm_suffixes = ("norm2.weight", "norm2.bias") if config.cross_attn_norm else ()
    for n in range(config.num_dit_layers):
        prefix = f"blocks.{n}."
        for suffix in block_suffixes + _SHARED_BLOCK_SUFFIXES + cross_attn_norm_suffixes:
            keys.add(prefix + suffix)
    return keys


def expected_mcore_keys(config: WanConfig) -> Set[str]:
    """Analytic bare ``state_dict`` key set for one Primus WAN backbone.

    Built analytically (not by instantiating the model) because the backbone
    constructs TransformerEngine modules, which require CUDA / parallel init and
    cannot be built on the ``meta`` device. ``_extra_state`` keys are
    intentionally excluded (they are not real parameters).
    """
    return _build_expected_keys(config, _MCORE_BLOCK_SUFFIXES)


def expected_diffusers_keys(config: WanConfig) -> Set[str]:
    """Analytic bare ``state_dict`` key set for one diffusers ``WanTransformer3DModel``.

    Covers the T2V backbone only; I2V-specific extras
    (``condition_embedder.image_embedder.*``, ``attn2.add_k_proj``,
    ``norm_added_k``) are deliberately excluded and are reported as dropped.
    """
    return _build_expected_keys(config, _DIFFUSERS_BLOCK_SUFFIXES)


# ---------------------------------------------------------------------------
# Per-head fuse / split helpers
# ---------------------------------------------------------------------------


def _fuse_heads(tensors: List[Tensor], heads: int) -> Tensor:
    """Interleave same-shaped per-projection tensors into the fused layout.

    Each tensor is ``[inner, ...]`` (weight ``[inner, dim]`` or bias ``[inner]``)
    where ``inner = heads * head_dim``. Returns ``[len(tensors)*inner, ...]`` with
    rows grouped per head: ``[t0_head0, t1_head0, ..., t0_head1, ...]``.
    """
    heads_dim_split = []
    tail_shape = tensors[0].shape[1:]
    hd = tensors[0].shape[0] // heads
    for t in tensors:
        heads_dim_split.append(t.reshape(heads, hd, *tail_shape))
    fused = torch.stack(heads_dim_split, dim=1)  # [heads, n, hd, ...]
    return fused.reshape(heads * len(tensors) * hd, *tail_shape).contiguous()


def _split_heads(fused: Tensor, heads: int, n: int) -> List[Tensor]:
    """Inverse of :func:`_fuse_heads`: recover the ``n`` per-projection tensors."""
    tail_shape = fused.shape[1:]
    hd = fused.shape[0] // (heads * n)
    x = fused.reshape(heads, n, hd, *tail_shape)
    return [x[:, i].reshape(heads * hd, *tail_shape).contiguous() for i in range(n)]


# ---------------------------------------------------------------------------
# Distributed checkpoint (fsdp_dtensor / torch DCP) loader
# ---------------------------------------------------------------------------


def _normalize_wan_key(full_key: str, expected: Set[str]) -> Optional[str]:
    """Map a DCP key to a prefixed backbone key, or ``None`` if not a backbone weight.

    Training checkpoints store the WAN backbone under some wrapper (e.g.
    ``model.transformer.*``) alongside optimizer/RNG state. We locate the
    ``transformer.`` / ``transformer_2.`` boundary and keep the key only if the
    remainder is an actual backbone parameter (in ``expected``). This naturally
    excludes optimizer moments (``...weight.exp_avg``), RNG, scheduler, etc.
    """
    for tag in ("transformer_2.", "transformer."):
        idx = full_key.find(tag)
        if idx != -1:
            bare = full_key[idx + len(tag) :]
            if bare in expected:
                return f"{tag}{bare}"
    return None


def is_distributed_checkpoint(path: Path) -> bool:
    """True if ``path`` is a torch Distributed Checkpoint (``.metadata`` + shards)."""
    return path.is_dir() and (path / ".metadata").exists()


def load_distcp_backbone_state_dict(path: Path, config: WanConfig) -> StateDict:
    """Load only the WAN backbone weights from a torch DCP (``fsdp_dtensor``) dir.

    Reads the checkpoint metadata, selects just the keys that correspond to WAN
    backbone parameters (single or dual transformer), allocates matching target
    tensors, and loads those shards (skipping optimizer/RNG state entirely).

    Key selection is name-based against :func:`expected_mcore_keys`, so
    ``config`` must describe the model the checkpoint was trained with: a
    mismatched depth silently keeps only the layers the two shapes happen to
    share.

    Returns a state dict keyed by ``transformer.*`` (and ``transformer_2.*`` for
    WAN 2.2), ready for :func:`convert_primus_to_hf` / :func:`export_hf_directory`.
    """
    import torch.distributed.checkpoint as dcp

    reader = dcp.FileSystemReader(str(path))
    metadata = reader.read_metadata()
    expected = expected_mcore_keys(config)

    target: StateDict = {}
    keymap: Dict[str, str] = {}
    for full_key, tmeta in metadata.state_dict_metadata.items():
        norm = _normalize_wan_key(full_key, expected)
        if norm is None:
            continue
        size = getattr(tmeta, "size", None)
        props = getattr(tmeta, "properties", None)
        if size is None or props is None:
            continue  # non-tensor metadata (bytes/RNG)
        target[full_key] = torch.empty(tuple(size), dtype=props.dtype)
        keymap[full_key] = norm

    if not target:
        raise RuntimeError(
            f"No WAN backbone tensors found in DCP at {path}. The model keys did "
            "not match the expected layout (check --config-json matches the "
            "trained model)."
        )

    try:
        dcp.load(target, storage_reader=reader, no_dist=True)
    except TypeError:
        # Older torch signatures without the ``no_dist`` kwarg.
        dcp.load(target, storage_reader=reader)

    out = {keymap[k]: v for k, v in target.items()}
    logger.info(
        "Loaded %d backbone tensor(s) from DCP %s (prefixes: %s)",
        len(out),
        path,
        sorted({k.split(".")[0] for k in out}),
    )
    return out


# ---------------------------------------------------------------------------
# Single-backbone conversion (bare keys)
# ---------------------------------------------------------------------------


def _finalize_report(
    out: StateDict,
    src: StateDict,
    consumed: Set[str],
    expected: Set[str],
    report: ConversionReport,
) -> StateDict:
    """Record mapped / dropped / missing for an explicit-remap conversion."""
    report.mapped.extend(sorted(consumed))
    dropped = [k for k in src if k not in consumed]
    report.dropped.extend(dropped)
    missing = [k for k in expected if k not in out]
    report.missing.extend(missing)
    if dropped:
        report.add_warning(
            f"Dropped {len(dropped)} source key(s) with no counterpart in the target "
            f"layout (e.g. {dropped[:3]}); typically I2V-only extras or TE _extra_state."
        )
    if missing:
        report.add_warning(
            f"{len(missing)} expected key(s) not produced "
            f"(e.g. {missing[:3]}); source checkpoint may be incomplete."
        )
    return out


def _hf_to_primus_backbone(src: StateDict, config: WanConfig, report: ConversionReport) -> StateDict:
    """Convert a bare diffusers WAN backbone state dict to the Primus fused layout."""
    heads = config.num_attention_heads
    out: StateDict = {}
    consumed: Set[str] = set()

    def copy(key: str) -> None:
        if key in src:
            out[key] = src[key].clone()
            consumed.add(key)

    for key in _SHARED_ROOT_KEYS:
        copy(key)

    for n in range(config.num_dit_layers):
        b = f"blocks.{n}."

        for wb in ("weight", "bias"):
            q, k, v = f"{b}attn1.to_q.{wb}", f"{b}attn1.to_k.{wb}", f"{b}attn1.to_v.{wb}"
            if q in src and k in src and v in src:
                out[f"{b}attn1.linear_qkv.{wb}"] = _fuse_heads([src[q], src[k], src[v]], heads)
                consumed.update({q, k, v})
        for src_name, dst_name in (
            (f"{b}attn1.norm_q.weight", f"{b}attn1.q_layernorm.weight"),
            (f"{b}attn1.norm_k.weight", f"{b}attn1.k_layernorm.weight"),
            (f"{b}attn1.to_out.0.weight", f"{b}attn1.linear_proj.weight"),
            (f"{b}attn1.to_out.0.bias", f"{b}attn1.linear_proj.bias"),
        ):
            if src_name in src:
                out[dst_name] = src[src_name].clone()
                consumed.add(src_name)

        for wb in ("weight", "bias"):
            q = f"{b}attn2.to_q.{wb}"
            if q in src:
                out[f"{b}attn2.linear_q.{wb}"] = src[q].clone()
                consumed.add(q)
            k, v = f"{b}attn2.to_k.{wb}", f"{b}attn2.to_v.{wb}"
            if k in src and v in src:
                out[f"{b}attn2.linear_kv.{wb}"] = _fuse_heads([src[k], src[v]], heads)
                consumed.update({k, v})
        for src_name, dst_name in (
            (f"{b}attn2.norm_q.weight", f"{b}attn2.q_layernorm.weight"),
            (f"{b}attn2.norm_k.weight", f"{b}attn2.k_layernorm.weight"),
            (f"{b}attn2.to_out.0.weight", f"{b}attn2.linear_proj.weight"),
            (f"{b}attn2.to_out.0.bias", f"{b}attn2.linear_proj.bias"),
        ):
            if src_name in src:
                out[dst_name] = src[src_name].clone()
                consumed.add(src_name)

        for src_name, dst_name in (
            (f"{b}ffn.net.0.proj.weight", f"{b}ffn.linear_fc1.weight"),
            (f"{b}ffn.net.0.proj.bias", f"{b}ffn.linear_fc1.bias"),
            (f"{b}ffn.net.2.weight", f"{b}ffn.linear_fc2.weight"),
            (f"{b}ffn.net.2.bias", f"{b}ffn.linear_fc2.bias"),
        ):
            if src_name in src:
                out[dst_name] = src[src_name].clone()
                consumed.add(src_name)

        for key in (f"{b}norm2.weight", f"{b}norm2.bias", f"{b}scale_shift_table"):
            copy(key)

    return _finalize_report(out, src, consumed, expected_mcore_keys(config), report)


def _primus_to_hf_backbone(src: StateDict, config: WanConfig, report: ConversionReport) -> StateDict:
    """Convert a bare Primus fused WAN backbone state dict back to diffusers layout."""
    heads = config.num_attention_heads
    out: StateDict = {}
    consumed: Set[str] = set()

    def copy(key: str) -> None:
        if key in src:
            out[key] = src[key].clone()
            consumed.add(key)

    for key in _SHARED_ROOT_KEYS:
        copy(key)

    for n in range(config.num_dit_layers):
        b = f"blocks.{n}."

        for wb in ("weight", "bias"):
            qkv = f"{b}attn1.linear_qkv.{wb}"
            if qkv in src:
                q, k, v = _split_heads(src[qkv], heads, 3)
                out[f"{b}attn1.to_q.{wb}"] = q
                out[f"{b}attn1.to_k.{wb}"] = k
                out[f"{b}attn1.to_v.{wb}"] = v
                consumed.add(qkv)
        for src_name, dst_name in (
            (f"{b}attn1.q_layernorm.weight", f"{b}attn1.norm_q.weight"),
            (f"{b}attn1.k_layernorm.weight", f"{b}attn1.norm_k.weight"),
            (f"{b}attn1.linear_proj.weight", f"{b}attn1.to_out.0.weight"),
            (f"{b}attn1.linear_proj.bias", f"{b}attn1.to_out.0.bias"),
        ):
            if src_name in src:
                out[dst_name] = src[src_name].clone()
                consumed.add(src_name)

        for wb in ("weight", "bias"):
            q = f"{b}attn2.linear_q.{wb}"
            if q in src:
                out[f"{b}attn2.to_q.{wb}"] = src[q].clone()
                consumed.add(q)
            kv = f"{b}attn2.linear_kv.{wb}"
            if kv in src:
                k, v = _split_heads(src[kv], heads, 2)
                out[f"{b}attn2.to_k.{wb}"] = k
                out[f"{b}attn2.to_v.{wb}"] = v
                consumed.add(kv)
        for src_name, dst_name in (
            (f"{b}attn2.q_layernorm.weight", f"{b}attn2.norm_q.weight"),
            (f"{b}attn2.k_layernorm.weight", f"{b}attn2.norm_k.weight"),
            (f"{b}attn2.linear_proj.weight", f"{b}attn2.to_out.0.weight"),
            (f"{b}attn2.linear_proj.bias", f"{b}attn2.to_out.0.bias"),
        ):
            if src_name in src:
                out[dst_name] = src[src_name].clone()
                consumed.add(src_name)

        for src_name, dst_name in (
            (f"{b}ffn.linear_fc1.weight", f"{b}ffn.net.0.proj.weight"),
            (f"{b}ffn.linear_fc1.bias", f"{b}ffn.net.0.proj.bias"),
            (f"{b}ffn.linear_fc2.weight", f"{b}ffn.net.2.weight"),
            (f"{b}ffn.linear_fc2.bias", f"{b}ffn.net.2.bias"),
        ):
            if src_name in src:
                out[dst_name] = src[src_name].clone()
                consumed.add(src_name)

        for key in (f"{b}norm2.weight", f"{b}norm2.bias", f"{b}scale_shift_table"):
            copy(key)

    return _finalize_report(out, src, consumed, expected_diffusers_keys(config), report)


# ---------------------------------------------------------------------------
# Prefix helpers (WAN 2.2 dual-expert: transformer. / transformer_2.)
# ---------------------------------------------------------------------------


def _detect_backbone_prefixes(state_dict: StateDict) -> List[str]:
    prefixes = []
    for p in ("transformer.", "transformer_2."):
        if any(k.startswith(p) for k in state_dict):
            prefixes.append(p)
    return prefixes


def _strip_prefix(state_dict: StateDict, prefix: str) -> StateDict:
    return {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}


def _add_prefix(state_dict: StateDict, prefix: str) -> StateDict:
    return {f"{prefix}{k}": v for k, v in state_dict.items()}


def _convert_one_backbone(
    bare: StateDict,
    config: WanConfig,
    direction: str,
    report: ConversionReport,
) -> StateDict:
    """Convert a single bare backbone state dict in the requested direction."""
    if direction == "hf->primus":
        return _hf_to_primus_backbone(bare, config, report)
    if direction == "primus->hf":
        return _primus_to_hf_backbone(bare, config, report)
    raise ValueError(f"Unknown direction={direction!r}; expected 'hf->primus' or 'primus->hf'.")


def _convert(
    src: StateDict,
    config: WanConfig,
    direction: str,
    prefix: Optional[str],
    strict: bool,
) -> Tuple[StateDict, ConversionReport]:
    """Shared driver: dispatch over the detected backbone prefixes."""
    report = ConversionReport(direction=direction)
    detected = _detect_backbone_prefixes(src)

    if not detected:
        out = _convert_one_backbone(src, config, direction, report)
        if prefix:
            out = _add_prefix(out, prefix)
    else:
        out = {}
        for src_prefix in detected:
            bare = _strip_prefix(src, src_prefix)
            conv = _convert_one_backbone(bare, config, direction, report)
            out.update(_add_prefix(conv, prefix or src_prefix))

    logger.info("%s", report.summary())
    if strict:
        report.raise_if_lossy()
    return out, report


# ---------------------------------------------------------------------------
# Public API: state-dict level
# ---------------------------------------------------------------------------


def convert_hf_to_primus(
    hf_state_dict: StateDict,
    config: WanConfig,
    *,
    prefix: Optional[str] = None,
    strict: bool = False,
) -> Tuple[StateDict, ConversionReport]:
    """Convert a diffusers WAN transformer state dict to Primus format.

    Args:
        hf_state_dict: A bare backbone state dict (keys like
            ``patch_embedding.weight``) or a full model dict prefixed with
            ``transformer.`` / ``transformer_2.``.
        config: ``WanConfig`` describing the target Primus model.
        prefix: Output key prefix (e.g. ``transformer.``). ``None`` reuses the
            detected input prefix(es); bare inputs produce bare outputs (which
            is what the ``backbone_pretrained`` loaders expect).
        strict: Raise if any key is dropped or missing.

    Returns:
        ``(primus_state_dict, report)``.
    """
    return _convert(hf_state_dict, config, "hf->primus", prefix, strict)


def convert_primus_to_hf(
    primus_state_dict: StateDict,
    config: WanConfig,
    *,
    prefix: Optional[str] = None,
    strict: bool = False,
) -> Tuple[StateDict, ConversionReport]:
    """Convert a Primus WAN state dict to diffusers ``WanTransformer3DModel`` format.

    Args:
        primus_state_dict: A bare backbone state dict or a full model dict
            prefixed with ``transformer.`` / ``transformer_2.``.
        config: ``WanConfig`` describing the source Primus model.
        prefix: Output key prefix. ``None`` reuses the detected input prefix(es).
        strict: Raise if any key is dropped or missing.

    Returns:
        ``(hf_state_dict, report)``.
    """
    return _convert(primus_state_dict, config, "primus->hf", prefix, strict)


# ---------------------------------------------------------------------------
# Public API: checkpoint (path / HuggingFace repo id) level
# ---------------------------------------------------------------------------


def _get_hf_token(token_file: Optional[str] = None) -> Optional[str]:
    """Get HuggingFace token with fallback options.

    Thin wrapper around the shared
    :func:`...preprocessing.auth.setup_hf_authentication` so the token-resolution
    priority chain (file with permission check -> HF_TOKEN env -> HF CLI login ->
    None) lives in one place. Returns None instead of raising so the converter
    can fall back to public-only access.

    Args:
        token_file: Optional path to token file

    Returns:
        Token string if found, None otherwise
    """
    from primus.backends.megatron.data.diffusion.preprocessing.auth import (
        HFAuthError,
        setup_hf_authentication,
    )

    try:
        return setup_hf_authentication(token_file=token_file, use_env=True)
    except HFAuthError:
        # Preserve fallback-to-public behavior for the converter rather than
        # hard-failing on a bad/insecure token file.
        return None


def _resolve_checkpoint_path(checkpoint_path: Union[str, Path]) -> Path:
    """Resolve a local path, or download a HuggingFace repo id to the HF cache."""
    checkpoint_path_str = str(checkpoint_path)
    checkpoint_path_obj = Path(checkpoint_path)

    if checkpoint_path_obj.exists():
        return checkpoint_path_obj

    # HF repo ids: don't start with /, ./, ../, and don't contain ..
    looks_like_hf_repo = (
        "/" in checkpoint_path_str
        and not checkpoint_path_str.startswith("/")
        and not checkpoint_path_str.startswith(".")
        and ".." not in checkpoint_path_str
    )
    if not looks_like_hf_repo:
        raise FileNotFoundError(
            f"Checkpoint file or directory not found: {checkpoint_path_str}\n"
            f"If this is a HuggingFace repo ID, ensure it follows the format 'org/repo' or "
            f"'org/repo/subfolder'"
        )

    logger.info("Detected HuggingFace repo ID: %s", checkpoint_path_str)
    logger.info("Downloading from HuggingFace Hub...")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for downloading checkpoints. "
            "Install with: pip install huggingface_hub"
        )

    # Authentication: .hf_token at the repo root -> HF_TOKEN env -> HF CLI login.
    token_file = Path(__file__).parents[7] / ".hf_token"
    if token_file.exists():
        logger.info("Using HuggingFace token from project root: %s", token_file)
        hf_token = _get_hf_token(token_file=str(token_file))
    else:
        hf_token = _get_hf_token(token_file=None)

    parts = checkpoint_path_str.split("/")
    repo_id = "/".join(parts[:2])
    subfolder = "/".join(parts[2:]) if len(parts) > 2 else None

    cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))

    try:
        local_dir = snapshot_download(
            repo_id=repo_id,
            allow_patterns=[f"{subfolder}/*"] if subfolder else None,
            cache_dir=cache_dir,
            token=hf_token,
            resume_download=True,
        )
    except Exception as e:
        logger.error("Download failed: %s", e)
        if "401" in str(e) or "403" in str(e):
            raise RuntimeError(
                f"Authentication failed. Please set HuggingFace token using one of:\n"
                f"  1. Create .hf_token file: echo 'your_token' > .hf_token && chmod 600 .hf_token\n"
                f"  2. Set environment variable: export HF_TOKEN=your_token\n"
                f"  3. Login via CLI: huggingface-cli login\n"
                f"Get your token from: https://huggingface.co/settings/tokens\n"
                f"Accept the model license at: https://huggingface.co/{repo_id}"
            ) from e
        raise

    resolved = Path(local_dir) / subfolder if subfolder else Path(local_dir)
    logger.info("Downloaded to: %s", resolved)
    return resolved


def convert_hf_checkpoint(
    checkpoint_path: Union[str, Path],
    wan_config: WanConfig,
    save_to: Optional[Union[str, Path]] = None,
    *,
    prefix: Optional[str] = None,
    strict: bool = False,
) -> StateDict:
    """Convert a HuggingFace WAN checkpoint to Primus format.

    Supports both local paths and HuggingFace repo IDs. If a repo ID is provided
    (e.g. ``"Wan-AI/Wan2.1-T2V-1.3B-Diffusers/transformer"``), the checkpoint will
    be automatically downloaded from HuggingFace Hub.

    Args:
        checkpoint_path: Path to HF checkpoint OR HuggingFace repo ID
                        (e.g. ``"Wan-AI/Wan2.1-T2V-1.3B-Diffusers/transformer"``)
        wan_config: WanConfig instance with model architecture info
        save_to: Optional path to save converted checkpoint
        prefix: Optional output key prefix (e.g. ``"transformer."``)
        strict: Raise if any key is dropped or missing

    Returns:
        Dictionary of converted state dict (Primus format)

    Example:
        >>> from primus.backends.megatron.core.models.diffusion.wan import WanConfig
        >>> config = WanConfig(hidden_size=1536, num_attention_heads=12, num_dit_layers=30)
        >>> primus_sd = convert_hf_checkpoint(
        ...     "Wan-AI/Wan2.1-T2V-1.3B-Diffusers/transformer",
        ...     wan_config=config,
        ...     save_to="primus_wan21_t2v_1_3b.safetensors",
        ... )
    """
    resolved = _resolve_checkpoint_path(checkpoint_path)

    logger.info("Loading HuggingFace checkpoint from: %s", resolved)
    hf_state_dict = _load_state_dict(resolved)
    logger.info("Loaded %d keys from HuggingFace checkpoint", len(hf_state_dict))

    primus_state_dict, _ = convert_hf_to_primus(hf_state_dict, wan_config, prefix=prefix, strict=strict)
    logger.info("Conversion complete! Primus state dict has %d keys", len(primus_state_dict))

    if save_to:
        save_to = Path(save_to)
        logger.info("Saving to: %s", save_to)
        _save_state_dict(primus_state_dict, save_to)
        logger.info("Saved successfully!")

    return primus_state_dict


# ---------------------------------------------------------------------------
# Diffusers directory export (config.json + diffusion_pytorch_model.safetensors)
# ---------------------------------------------------------------------------


def wan_hf_config_dict(config: WanConfig) -> Dict[str, object]:
    """Build a diffusers ``WanTransformer3DModel`` ``config.json`` payload."""
    return {
        "_class_name": "WanTransformer3DModel",
        "_diffusers_version": "0.33.0.dev0",
        "patch_size": list(config.patch_size_3d),
        "num_attention_heads": config.num_attention_heads,
        "attention_head_dim": config.hidden_size // config.num_attention_heads,
        "in_channels": config.in_channels,
        "out_channels": config.out_channels or config.in_channels,
        "text_dim": config.text_embed_dim,
        "freq_dim": config.freq_dim,
        "ffn_dim": config.ffn_hidden_size,
        "num_layers": config.num_dit_layers,
        "cross_attn_norm": config.cross_attn_norm,
        "qk_norm": config.qk_norm,
        "eps": config.layernorm_epsilon,
        "image_dim": config.image_dim,
        "added_kv_proj_dim": config.added_kv_proj_dim,
        "rope_max_seq_len": config.rope_max_seq_len,
        "pos_embed_seq_len": config.pos_embed_seq_len,
    }


def export_hf_directory(
    primus_state_dict: StateDict,
    config: WanConfig,
    export_dir: str,
    *,
    config_template: Optional[str] = None,
    strict: bool = False,
) -> ConversionReport:
    """Write a diffusers-loadable export directory from a Primus WAN state dict.

    Produces, for each backbone present:

        <export_dir>/transformer/config.json
        <export_dir>/transformer/diffusion_pytorch_model.safetensors
        [<export_dir>/transformer_2/...]   # WAN 2.2 dual-expert

    These are exactly the files ``WanTransformer3DModel.from_pretrained(
    export_dir, subfolder="transformer")`` expects; the rest of the pipeline
    (vae / text_encoder / tokenizer / scheduler) is loaded from the base WAN repo.

    Args:
        primus_state_dict: Primus WAN state dict (bare or ``transformer.`` /
            ``transformer_2.`` prefixed).
        config: ``WanConfig`` describing the model.
        export_dir: Output directory root.
        config_template: Optional path to an existing diffusers ``config.json``
            to copy verbatim (e.g. the original HF transformer config). If
            ``None``, the config is derived from ``config``.
        strict: Raise if any key is dropped or missing.

    Returns:
        The :class:`ConversionReport` for the export.
    """
    from safetensors.torch import save_file as save_safetensors

    report = ConversionReport(direction="primus->hf")
    root = Path(export_dir)

    if config_template is not None:
        cfg_json = json.loads(Path(config_template).read_text())
    else:
        cfg_json = wan_hf_config_dict(config)

    detected = _detect_backbone_prefixes(primus_state_dict)
    # Bare input -> single "transformer" subfolder.
    prefixes = detected or [None]

    for prefix in prefixes:
        bare = _strip_prefix(primus_state_dict, prefix) if prefix else dict(primus_state_dict)
        conv = _convert_one_backbone(bare, config, "primus->hf", report)
        subfolder = prefix.rstrip(".") if prefix else "transformer"
        sub_dir = root / subfolder
        sub_dir.mkdir(parents=True, exist_ok=True)
        save_safetensors(
            {k: v.contiguous() for k, v in conv.items()},
            str(sub_dir / "diffusion_pytorch_model.safetensors"),
        )
        with open(sub_dir / "config.json", "w") as f:
            json.dump(cfg_json, f, indent=2)
        logger.info("Wrote diffusers backbone -> %s", sub_dir)

    logger.info("%s", report.summary())
    if strict:
        report.raise_if_lossy()
    return report


# ---------------------------------------------------------------------------
# I/O helpers + CLI
# ---------------------------------------------------------------------------


def _load_state_dict(path: Path) -> StateDict:
    """Load a state dict from .safetensors, a torch file, or a directory of shards."""
    from safetensors.torch import load_file as load_safetensors

    if path.is_dir():
        sd: StateDict = {}
        safes = sorted(path.glob("*.safetensors"))
        if safes:
            for f in safes:
                logger.info("  Loading %s...", f.name)
                sd.update(load_safetensors(str(f)))
            return sd
        pts = sorted([*path.glob("*.pt"), *path.glob("*.pth"), *path.glob("*.bin")])
        if not pts:
            raise FileNotFoundError(f"No checkpoint files under {path}")
        loaded = torch.load(str(pts[0]), map_location="cpu", weights_only=False)
        return loaded.get("state_dict", loaded)
    if path.suffix == ".safetensors":
        return load_safetensors(str(path))
    loaded = torch.load(str(path), map_location="cpu", weights_only=False)
    return loaded.get("state_dict", loaded)


def _save_state_dict(state_dict: StateDict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".safetensors":
        from safetensors.torch import save_file as save_safetensors

        save_safetensors({k: v.contiguous() for k, v in state_dict.items()}, str(path))
    else:
        torch.save(state_dict, str(path))


def _config_from_args(args: argparse.Namespace) -> WanConfig:
    if args.config_json:
        with open(args.config_json, "r") as f:
            overrides = json.load(f)
    else:
        overrides = {}
    if isinstance(overrides.get("patch_size_3d"), list):
        overrides["patch_size_3d"] = tuple(overrides["patch_size_3d"])
    return WanConfig(**overrides)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert WAN checkpoints between HF diffusers and Primus layouts."
    )
    parser.add_argument(
        "--direction",
        required=True,
        choices=["hf2primus", "primus2hf"],
        help="Conversion direction.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input checkpoint file, directory, distributed-checkpoint directory, "
        "or HuggingFace repo ID (hf2primus only).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output checkpoint file (.safetensors or .pt) for a raw state-dict conversion.",
    )
    parser.add_argument(
        "--hf-export-dir",
        default=None,
        help="(primus2hf only) Write a diffusers-loadable directory: "
        "<dir>/transformer/{config.json,diffusion_pytorch_model.safetensors} "
        "(+ transformer_2/ for WAN 2.2). Use this for diffusers inference.",
    )
    parser.add_argument(
        "--hf-config-template",
        default=None,
        help="(primus2hf export) Optional existing diffusers config.json to copy "
        "verbatim into each subfolder instead of deriving it from --config-json.",
    )
    parser.add_argument(
        "--config-json",
        default=None,
        help="Optional JSON file with WanConfig field overrides (hidden_size, num_dit_layers, ...).",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Optional output key prefix (e.g. 'transformer.').",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any key is dropped or missing.",
    )
    parser.add_argument(
        "--report-json",
        default=None,
        help="Optional path to write the conversion report as JSON.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.hf_export_dir and args.direction != "primus2hf":
        parser.error("--hf-export-dir is only valid with --direction primus2hf")
    if not args.hf_export_dir and not args.output:
        parser.error("either --output or --hf-export-dir is required")

    config = _config_from_args(args)

    input_path = Path(args.input)
    if is_distributed_checkpoint(input_path):
        if args.direction != "primus2hf":
            parser.error(
                "Distributed checkpoint (fsdp_dtensor) input is only supported " "with --direction primus2hf."
            )
        src = load_distcp_backbone_state_dict(input_path, config)
    else:
        src = _load_state_dict(_resolve_checkpoint_path(input_path))

    if args.hf_export_dir:
        report = export_hf_directory(
            src,
            config,
            args.hf_export_dir,
            config_template=args.hf_config_template,
            strict=args.strict,
        )
        print(report.summary())
        print(f"Wrote diffusers export -> {args.hf_export_dir}")
    else:
        if args.direction == "hf2primus":
            out, report = convert_hf_to_primus(src, config, prefix=args.prefix, strict=args.strict)
        else:
            out, report = convert_primus_to_hf(src, config, prefix=args.prefix, strict=args.strict)

        _save_state_dict(out, Path(args.output))
        print(report.summary())
        print(f"Wrote {len(out)} tensors -> {args.output}")

    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(
                {
                    "direction": report.direction,
                    "mapped": report.mapped,
                    "dropped": report.dropped,
                    "missing": report.missing,
                    "warnings": report.warnings,
                },
                f,
                indent=2,
            )
        print(f"Wrote conversion report -> {args.report_json}")

    return 0


__all__ = [
    "ConversionReport",
    "expected_mcore_keys",
    "expected_diffusers_keys",
    "is_distributed_checkpoint",
    "load_distcp_backbone_state_dict",
    "convert_hf_to_primus",
    "convert_primus_to_hf",
    "convert_hf_checkpoint",
    "wan_hf_config_dict",
    "export_hf_directory",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
