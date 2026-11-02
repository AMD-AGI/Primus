###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark waveform audio encoder and the audio placeholder merge.

HF ``MuseSparkAudioModel`` (transformers ``modeling_muse_spark.py``): raw mono 24 kHz
audio, no spectrogram. Four stages; stage ``i`` is

* a patch embedding: ``reshape(B, T / p_i, p_i * C)`` -> ``Linear`` (bias) -> RMSNorm
  (weight, ``eps``);
* ``depths[i]`` conv blocks: ``x + mlp(norm(causal_depthwise_conv(x)))``, the conv with
  kernel ``k`` and left padding ``k - 1`` (so output ``t`` sees inputs ``<= t`` only), the
  MLP ``fc1 -> act -> fc2`` with biases and width ``mlp_ratio * hidden``.

Then ``RMSNorm -> Linear(hidden[-1], out_hidden) -> RMSNorm``. One output token covers
``prod(patch_sizes)`` samples (1920 = 80 ms).

Module and parameter names follow HF, so the checkpoint maps one to one
(``model.audio_tower.<name>`` -> ``audio_tower.<name>``).

:func:`audio_features` is HF ``get_audio_features``: it drops the tokens of right padding
(``input_values_mask``) and applies the text embedding norm (weightless RMSNorm).
:func:`merge_audio` is the placeholder scatter of HF ``MuseSparkModel.forward``.
"""

import math
import os
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from primus.backends.megatron.core.transformer.muse_spark.norms import (
    MuseSparkRMSNorm,
    _rms,
)

__all__ = [
    "AUDIO_CONFIG_KEYS",
    "MuseSparkAudioEncoder",
    "audio_config_from_hf",
    "audio_features",
    "merge_audio",
    "samples_per_token",
]

# HF audio_config keys the encoder uses, with the HF defaults.
AUDIO_CONFIG_KEYS = {
    "hidden_sizes": [256, 512, 1024, 2048],
    "depths": [2, 2, 2, 2],
    "patch_sizes": [16, 6, 5, 4],
    "conv_kernel_size": 7,
    "mlp_ratio": 4.0,
    "hidden_act": "gelu",
    "rms_norm_eps": 1e-6,
    "out_hidden_size": 8192,
}

_ACT = {
    "gelu": F.gelu,
    "gelu_new": lambda x: F.gelu(x, approximate="tanh"),
    "gelu_pytorch_tanh": lambda x: F.gelu(x, approximate="tanh"),
    "silu": F.silu,
    "relu": F.relu,
}


def audio_config_from_hf(cfg: Optional[dict]) -> Optional[dict]:
    """The encoder settings from an HF ``audio_config`` dict (None when there is none).

    Raises on a value the encoder does not support (unknown activation, list lengths that
    differ)."""
    if not cfg:
        return None
    out = {k: cfg.get(k, v) for k, v in AUDIO_CONFIG_KEYS.items()}
    out["hidden_sizes"] = [int(x) for x in out["hidden_sizes"]]
    out["depths"] = [int(x) for x in out["depths"]]
    out["patch_sizes"] = [int(x) for x in out["patch_sizes"]]
    n = int(cfg.get("num_hidden_layers", len(out["hidden_sizes"])))
    lens = {len(out["hidden_sizes"]), len(out["depths"]), len(out["patch_sizes"]), n}
    if len(lens) != 1:
        raise ValueError(
            f"Muse Spark audio_config: hidden_sizes / depths / patch_sizes / num_hidden_layers "
            f"disagree on the stage count: {out}"
        )
    if out["hidden_act"] not in _ACT:
        raise ValueError(f"Muse Spark audio_config: hidden_act {out['hidden_act']!r} is not supported")
    return out


def samples_per_token(cfg: dict) -> int:
    return math.prod(cfg["patch_sizes"])


class MuseSparkAudioCausalConv1d(nn.Conv1d):
    """Conv1d with left padding only (``(k - 1) * dilation + 1 - stride``): causal.

    New math: Megatron has no causal depthwise convolution over time."""

    @property
    def left_pad(self) -> int:
        return (self.kernel_size[0] - 1) * self.dilation[0] + 1 - self.stride[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(x, (self.left_pad, 0)))


class MuseSparkAudioPatchEmbedding(nn.Module):
    """New math: Megatron has no waveform patch embedding (reshape of raw samples ->
    Linear -> RMSNorm)."""

    def __init__(self, cfg: dict, idx: int):
        super().__init__()
        in_ch = 1 if idx == 0 else cfg["hidden_sizes"][idx - 1]
        out_ch = cfg["hidden_sizes"][idx]
        self.patch_size = cfg["patch_sizes"][idx]
        self.proj = nn.Linear(in_ch * self.patch_size, out_ch, bias=True)
        self.norm = MuseSparkRMSNorm(hidden_size=out_ch, eps=cfg["rms_norm_eps"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        if t % self.patch_size:
            raise ValueError(f"audio length {t} is not a multiple of the patch size {self.patch_size}")
        return self.norm(self.proj(x.reshape(b, t // self.patch_size, self.patch_size * c)))


class MuseSparkAudioMLP(nn.Module):
    """``fc2(act(fc1(x)))`` with biases, at the per-stage audio width (256 .. 2048).

    New math: Megatron's MLP is the decoder MLP (TP-sharded, hidden_size from the
    TransformerConfig); the audio stages have their own widths and are not tensor parallel
    (the whole encoder is ~0.1 B params and runs replicated on the first stage)."""

    def __init__(self, cfg: dict, idx: int):
        super().__init__()
        h = cfg["hidden_sizes"][idx]
        inter = int(h * cfg["mlp_ratio"])
        self.fc1 = nn.Linear(h, inter, bias=True)
        self.fc2 = nn.Linear(inter, h, bias=True)
        self.act = _ACT[cfg["hidden_act"]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class MuseSparkAudioConvBlock(nn.Module):
    """``x + mlp(norm(causal_depthwise_conv(x)))``.

    New math: a convolution block with no attention; Megatron's TransformerLayer /
    TransformerBlock have no conv mixer."""

    def __init__(self, cfg: dict, idx: int):
        super().__init__()
        h = cfg["hidden_sizes"][idx]
        self.conv = MuseSparkAudioCausalConv1d(h, h, kernel_size=cfg["conv_kernel_size"], groups=h, bias=True)
        self.norm = MuseSparkRMSNorm(hidden_size=h, eps=cfg["rms_norm_eps"])
        self.mlp = MuseSparkAudioMLP(cfg, idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.mlp(self.norm(y)) + x


class MuseSparkAudioLayer(nn.Module):
    """One encoder stage: patch embedding (downsampling) then conv blocks.

    New math: a downsampling conv stage; Megatron's TransformerLayer has no such layer."""

    def __init__(self, cfg: dict, idx: int):
        super().__init__()
        self.patch_embedder = MuseSparkAudioPatchEmbedding(cfg, idx)
        self.conv_blocks = nn.ModuleList(MuseSparkAudioConvBlock(cfg, idx) for _ in range(cfg["depths"][idx]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedder(x)
        for block in self.conv_blocks:
            x = block(x)
        return x


class MuseSparkAudioEncoder(nn.Module):
    """``[B, samples]`` waveform -> ``[B, samples / prod(patch_sizes), out_hidden_size]``.

    New math: a causal convolutional pyramid encoder (no attention, no spectrogram);
    Megatron has no audio / conv encoder to build on. It is small (~0.1 B params), so it is
    replicated (not tensor / pipeline parallel) and lives on the first pipeline stage."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = dict(cfg)
        self.samples_per_token = samples_per_token(cfg)
        self.layers = nn.ModuleList(MuseSparkAudioLayer(cfg, i) for i in range(len(cfg["hidden_sizes"])))
        self.norm = MuseSparkRMSNorm(hidden_size=cfg["hidden_sizes"][-1], eps=cfg["rms_norm_eps"])
        self.proj = nn.Linear(cfg["hidden_sizes"][-1], cfg["out_hidden_size"], bias=True)
        self.post_proj_norm = MuseSparkRMSNorm(hidden_size=cfg["out_hidden_size"], eps=cfg["rms_norm_eps"])

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        if input_values.dim() != 2 or input_values.shape[1] % self.samples_per_token:
            raise ValueError(
                f"input_values must be [batch, samples] with samples a multiple of "
                f"{self.samples_per_token}; got {tuple(input_values.shape)}"
            )
        x = input_values.to(device=self.proj.weight.device, dtype=self.proj.weight.dtype).unsqueeze(-1)
        for layer in self.layers:
            x = layer(x)
        return self.post_proj_norm(self.proj(self.norm(x)))


def padded_audio_features(
    encoder: MuseSparkAudioEncoder,
    input_values: torch.Tensor,
    input_values_mask: Optional[torch.Tensor],
    embed_norm_eps: float,
):
    """``(feats, n_tok)``: the normed features of every token, padded ``[B, T, out_hidden]``,
    and the real token count of each clip (``[B]`` on the features' device; None without a
    mask: all ``T`` tokens are real). No data-dependent shape, so no device sync.

    ``input_values_mask`` ``[B, samples]`` marks real samples (right padding only); a clip
    keeps ``ceil(len / samples_per_token)`` tokens. The features pass the text embedding
    norm (weightless RMSNorm in fp32, eps = ``embed_norm_eps``), a per-token op, so norming
    the padded rows too changes nothing for the real ones."""
    feats = encoder(input_values)
    feats = _rms(feats.float(), embed_norm_eps).type_as(feats)
    if input_values_mask is None:
        return feats, None
    spt = encoder.samples_per_token
    n_tok = ((input_values_mask.sum(-1) + spt - 1) // spt).to(feats.device)
    return feats, n_tok


def audio_features(
    encoder: MuseSparkAudioEncoder,
    input_values: torch.Tensor,
    input_values_mask: Optional[torch.Tensor],
    embed_norm_eps: float,
) -> torch.Tensor:
    """HF ``get_audio_features``: ``[N, out_hidden]`` features of the real (unpadded) tokens,
    batch-major (:func:`padded_audio_features` with the padding dropped; the boolean index
    has a data-dependent size, a device sync: the model's forward uses
    :func:`merge_padded_audio` instead)."""
    feats, n_tok = padded_audio_features(encoder, input_values, input_values_mask, embed_norm_eps)
    if n_tok is None:
        return feats.reshape(-1, feats.shape[-1])
    pos = torch.arange(feats.shape[1], device=feats.device)
    return feats[pos[None, :] < n_tok[:, None]]


def merge_padded_audio(
    embeddings: torch.Tensor, audio_mask: torch.Tensor, feats: torch.Tensor, n_tok: Optional[torch.Tensor]
) -> torch.Tensor:
    """:func:`merge_audio` of the real rows of padded ``feats`` (``[B_a, T, H]``, from
    :func:`padded_audio_features`) without a device sync: the k-th audio token (batch-major
    over ``audio_mask``) takes the k-th real feature row, found on device with a cumsum
    of ``audio_mask`` and a searchsorted over the clips' token counts."""
    b_a, t, h = feats.shape
    if n_tok is None:
        n_tok = torch.full((b_a,), t, dtype=torch.long, device=feats.device)
    n_tok = n_tok.to(device=audio_mask.device, dtype=torch.long)
    if audio_mask.device.type == "cpu" or os.environ.get("MUSE_SPARK_CHECK_AUDIO") == "1":
        n, have = int(audio_mask.sum()), int(n_tok.sum())
        if n != have:
            raise ValueError(f"Audio features and audio tokens do not match, tokens: {n}, features: {have}")
    flat_mask = audio_mask.reshape(-1)
    k = torch.cumsum(flat_mask.long(), 0) - 1  # index of each audio token among all of them
    ends = torch.cumsum(n_tok, 0)  # end (exclusive) of each clip's real rows
    clip = torch.searchsorted(ends, k.clamp(min=0), right=True).clamp(max=b_a - 1)
    row = clip * t + (k.clamp(min=0) - (ends[clip] - n_tok[clip])).clamp(0, t - 1)
    picked = feats.reshape(-1, h).to(embeddings.device, embeddings.dtype)[row.to(embeddings.device)]
    bsh = embeddings.transpose(0, 1)  # [B, S, H]
    out = torch.where(audio_mask.to(embeddings.device).unsqueeze(-1), picked.view(bsh.shape), bsh)
    return out.transpose(0, 1).contiguous()


def merge_audio(embeddings: torch.Tensor, audio_mask: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
    """Write ``feats`` (``[N, H]``, batch-major order) into the rows of ``embeddings``
    (``[S, B, H]``, Megatron layout) where ``audio_mask`` (``[B, S]``) is True."""
    # the count check reads the mask on the host (a device sync): only when the mask is on
    # the CPU or MUSE_SPARK_CHECK_AUDIO=1; masked_scatter itself needs no count
    if audio_mask.device.type == "cpu" or os.environ.get("MUSE_SPARK_CHECK_AUDIO") == "1":
        n = int(audio_mask.sum())
        if n != feats.shape[0]:
            raise ValueError(
                f"Audio features and audio tokens do not match, tokens: {n}, features: {feats.shape[0]}"
            )
    out = embeddings.transpose(0, 1).masked_scatter(audio_mask.unsqueeze(-1), feats.to(embeddings.dtype))
    return out.transpose(0, 1).contiguous()
