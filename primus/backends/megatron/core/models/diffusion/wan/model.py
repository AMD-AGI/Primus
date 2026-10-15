# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Portions copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
WAN (2.1 / 2.2) video diffusion models on Megatron-Core.

Two model classes, selected by ``config.num_transformers``:

    - :class:`Wan` -- one denoiser. Covers all of WAN 2.1 and the WAN 2.2
      single-DiT TI2V-5B.
    - :class:`Wan2_2` -- two denoisers with per-sample routing on the timestep
      boundary. Covers WAN 2.2 A14B.

The backbone keeps the WAN scaffolding (Conv3d patch embed, time/text condition
embedders, ``scale_shift_table`` AdaLN modulation, output head) around a
``DiffusionTransformerBlock`` of :class:`~.layer_spec.WanLayer` blocks built
from the backend-resolved specs in ``layer_spec.py``, as Flux does.

The transformer stack runs sequence-major ``[S, B, dim]`` end to end, matching
Megatron-Bridge: AdaLN modulation, the LayerNorms, residuals, and the attention
core all operate on that layout, so the block GEMMs dispatch identical kernels.
Tokens are transposed into ``[S, B, dim]`` after the patch embed and back to
``[B, S, dim]`` before the final unpatchify.
"""

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.utils import sharded_state_dict_default
from safetensors.torch import load_file as load_safetensors
from torch import Tensor

from primus.backends.megatron.core.transformer.diffusion_transformer_block import (
    DiffusionTransformerBlock,
)

from ...common.diffusion_module.diffusion_module import DiffusionModule
from .config import WanConfig
from .layer_spec import get_wan_layer_spec
from .layers import WanConditionEmbedder, WanRotaryPosEmbed
from .utils import fold_patches_into_batch, patch_grid, unpatchify


class WanTransformer3D(nn.Module):
    """WAN video DiT backbone.

    Input  ``hidden_states``: ``[B, C_in, T, H, W]``
    Output velocity        : ``[B, C_out, T, H, W]``
    """

    def __init__(self, config: WanConfig):
        super().__init__()
        self.config = config
        self.patch_size_3d = tuple(config.patch_size_3d)
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels or config.in_channels
        self.hidden_size = config.hidden_size

        dim = config.hidden_size
        head_dim = dim // config.num_attention_heads

        self.rope = WanRotaryPosEmbed(head_dim, config.rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(
            self.in_channels, dim, kernel_size=self.patch_size_3d, stride=self.patch_size_3d
        )

        self.condition_embedder = WanConditionEmbedder(
            dim=dim,
            time_freq_dim=config.freq_dim,
            time_proj_dim=dim * 6,
            text_embed_dim=config.text_embed_dim,
        )

        # Activation recompute follows Megatron's recompute_granularity /
        # recompute_method / recompute_num_layers (and recompute_layer_ids) via
        # the block's checkpointed path. Named ``decoder`` as in Megatron-Bridge's
        # WAN: ``transformer.`` / ``transformer_2.`` already prefix the backbones.
        self.decoder = DiffusionTransformerBlock(
            config=config,
            spec=get_wan_layer_spec(config),
            post_layer_norm=False,
            pre_process=True,
            post_process=True,
        )

        self.norm_out = nn.LayerNorm(dim, eps=config.layernorm_epsilon, elementwise_affine=False)
        self.proj_out = nn.Linear(dim, self.out_channels * math.prod(self.patch_size_3d))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

        if config.adaln_zero_init:
            self._init_adaln_zero()

    def _init_adaln_zero(self) -> None:
        """AdaLN-zero init: identity blocks plus a zeroed output head."""
        for layer in self.decoder.layers:
            nn.init.zeros_(layer.scale_shift_table)
        nn.init.zeros_(self.condition_embedder.time_proj.weight)
        if self.condition_embedder.time_proj.bias is not None:
            nn.init.zeros_(self.condition_embedder.time_proj.bias)
        nn.init.zeros_(self.proj_out.weight)
        if self.proj_out.bias is not None:
            nn.init.zeros_(self.proj_out.bias)

    def forward(
        self,
        hidden_states: Tensor,
        timestep: Tensor,
        encoder_hidden_states: Tensor,
        attention_kwargs: Optional[Dict[str, Any]] = None,  # noqa: ARG002
        return_dict: bool = False,  # noqa: ARG002
    ) -> Tensor:
        batch_size = hidden_states.shape[0]
        grid = patch_grid(
            hidden_states.shape[2],
            hidden_states.shape[3],
            hidden_states.shape[4],
            self.patch_size_3d,
        )
        ppf, pph, ppw = grid

        rotary_freqs = self.rope(ppf, pph, ppw, hidden_states.device)

        x = fold_patches_into_batch(hidden_states, self.in_channels, self.patch_size_3d)
        x = self.patch_embedding(x)
        hs = x.reshape(batch_size, ppf * pph * ppw, -1)
        hs = hs.transpose(0, 1).contiguous()

        temb, timestep_proj, enc = self.condition_embedder(timestep, encoder_hidden_states)
        timestep_proj = timestep_proj.unflatten(1, (6, -1))
        enc = enc.transpose(0, 1).contiguous()

        hs = self.decoder(
            hidden_states=hs,
            attention_mask=None,
            context=enc,
            rotary_pos_emb=rotary_freqs,
            timestep_emb=timestep_proj,
        )
        # The checkpointed path returns hidden states alone; the plain path
        # also returns the (unchanged) text context.
        if isinstance(hs, tuple):
            hs = hs[0]

        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).transpose(0, 1).chunk(2, dim=0)
        hs = self.norm_out(hs) * (1 + scale) + shift
        hs = self.proj_out(hs)
        hs = hs.transpose(0, 1).contiguous()

        return unpatchify(hs, batch_size, grid, self.patch_size_3d)


def resolve_backbone_checkpoint(checkpoint_path: str, subfolder: Optional[str] = None) -> Path:
    """Locate one backbone's weights under ``backbone_pretrained``.

    ``backbone_pretrained`` may name the converted file itself, a directory
    holding one transformer's checkpoint, or a directory with one subfolder per
    transformer (the diffusers repo layout). ``subfolder`` applies only in the
    last case, so the file ``convert_wan_hf_to_primus.py`` writes loads as is.
    """
    path = Path(checkpoint_path)
    if subfolder and path.is_dir() and (path / subfolder).is_dir():
        return path / subfolder
    return path


def resolve_expert_checkpoints(checkpoint_path: str, subfolder: str, subfolder_2: str) -> Tuple[Path, Path]:
    """Locate both WAN 2.2 experts' weights, refusing to load one checkpoint twice."""
    first = resolve_backbone_checkpoint(checkpoint_path, subfolder)
    second = resolve_backbone_checkpoint(checkpoint_path, subfolder_2)
    if first == second:
        raise ValueError(
            f"Wan2_2 loads both experts from backbone_pretrained={checkpoint_path!r}, "
            f"so it must be a directory holding {subfolder!r} and {subfolder_2!r}; "
            "otherwise both experts would get the same weights"
        )
    return first, second


def _load_backbone_checkpoint(
    backbone: nn.Module, checkpoint_path: str, subfolder: Optional[str] = None
) -> None:
    """Load a fused-qkv WAN backbone checkpoint.

    Accepts a ``.safetensors`` file, a ``.pt``/``.pth`` file, or a directory of
    ``.safetensors`` shards, located by :func:`resolve_backbone_checkpoint`. The
    checkpoint must already be in the fused ``linear_qkv`` layout produced by
    ``checkpoint_converter``; a raw diffusers checkpoint with split
    ``to_q``/``to_k``/``to_v`` will not load.
    """
    path = resolve_backbone_checkpoint(checkpoint_path, subfolder)
    if not path.exists():
        raise FileNotFoundError(f"WAN backbone checkpoint not found: {path}")

    state_dict: Dict[str, Tensor] = {}
    if path.is_dir():
        shards = sorted(path.glob("*.safetensors"))
        if shards:
            for shard in shards:
                state_dict.update(load_safetensors(str(shard)))
        else:
            pt_files = sorted([*path.glob("*.pt"), *path.glob("*.pth")])
            if not pt_files:
                raise FileNotFoundError(f"No checkpoint files found under {path}")
            loaded = torch.load(str(pt_files[0]), map_location="cpu", weights_only=True)
            state_dict = loaded.get("state_dict", loaded)
    elif path.suffix == ".safetensors":
        state_dict = load_safetensors(str(path))
    else:
        loaded = torch.load(str(path), map_location="cpu", weights_only=True)
        state_dict = loaded.get("state_dict", loaded)

    missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
    missing = [k for k in missing if not k.endswith("_extra_state")]
    unexpected = [k for k in unexpected if not k.endswith("_extra_state")]
    if missing:
        raise RuntimeError(
            f"WAN backbone checkpoint is missing {len(missing)} keys "
            f"(first 5: {missing[:5]}). Expected a fused-qkv checkpoint from "
            "convert_wan_hf_to_primus.py."
        )
    if unexpected:
        raise RuntimeError(
            f"WAN backbone checkpoint has {len(unexpected)} unexpected keys " f"(first 5: {unexpected[:5]})."
        )


def build_wan_backbone(config: WanConfig, subfolder: str) -> nn.Module:
    """Build a WAN backbone and optionally load pretrained weights into it."""
    backbone = WanTransformer3D(config)
    if config.backbone_pretrained:
        _load_backbone_checkpoint(backbone, checkpoint_path=config.backbone_pretrained, subfolder=subfolder)
    return backbone


class Wan(DiffusionModule):
    """WAN with a single transformer denoiser.

    Covers WAN 2.1 at every size and the WAN 2.2 single-DiT TI2V-5B. Both share
    one call signature and one training objective (weighted flow matching).

    Args:
        config: ``WanConfig`` with ``num_transformers == 1``.
        encoder_configs: Optional encoder configs (Wan VAE, UMT5). Only the
            on-the-fly encoding path needs them; pre-encoded training does not.
        pg_collection: Process groups for distributed training. Under Strategy A
            this is effectively ``(DP, CP)`` with TP=PP=1.
    """

    def __init__(
        self,
        config: WanConfig,
        encoder_configs: Optional[Dict[str, Any]] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        if config.num_transformers != 1:
            raise ValueError(
                "Wan expects num_transformers=1; use Wan2_2 for dual-expert models "
                f"(got num_transformers={config.num_transformers})"
            )

        super().__init__(
            config=config,
            pg_collection=pg_collection,
            encoder_configs=encoder_configs,
        )

        self.transformer = build_wan_backbone(config, config.backbone_subfolder)

    def set_input_tensor(self, input_tensor) -> None:
        """Accept Megatron's pipeline input tensor.

        Diffusion models run with PP=1 so there is no inter-stage activation to
        wire in, but the forward-backward schedules call this on every model
        chunk, so the hook has to exist.
        """
        if isinstance(input_tensor, (list, tuple)):
            input_tensor = input_tensor[0] if len(input_tensor) > 0 else None
        self.input_tensor = input_tensor

    def forward(
        self,
        hidden_states: Tensor,
        timestep: Tensor,
        encoder_hidden_states: Tensor,
        *,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,  # noqa: ARG002
    ) -> Tensor:
        """Predict the flow-matching velocity.

        Args:
            hidden_states: ``[B, C, T, H, W]`` noisy latents.
            timestep: ``[B]`` timesteps on the ``[0, num_train_timesteps]`` axis.
            encoder_hidden_states: ``[B, S_txt, D_txt]`` UMT5 features.
            attention_kwargs: Forwarded to the backbone.

        Returns:
            ``[B, C, T, H, W]`` velocity prediction.
        """
        return self.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            attention_kwargs=attention_kwargs,
            return_dict=False,
        )

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """Sharded state dict under the diffusers-compatible ``transformer.`` prefix."""
        sharded: Dict[str, Any] = {}
        for name, module in self.named_children():
            sharded.update(sharded_state_dict_default(module, f"{prefix}{name}.", sharded_offsets, metadata))
        return sharded


class Wan2_2(DiffusionModule):
    """WAN 2.2 A14B: two experts routed per sample on the timestep boundary.

    Samples with ``timestep >= boundary`` go to ``transformer`` (high noise),
    the rest to ``transformer_2`` (low noise). Submodule names match the HF
    diffusers ``Wan-AI/Wan2.2-T2V-A14B-Diffusers`` layout.
    """

    def __init__(
        self,
        config: WanConfig,
        encoder_configs: Optional[Dict[str, Any]] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        if config.num_transformers != 2:
            raise ValueError(
                "Wan2_2 expects num_transformers=2; use Wan for single-transformer "
                f"models (got num_transformers={config.num_transformers})"
            )
        if config.boundary_ratio is None:
            raise ValueError("Wan2_2 requires boundary_ratio to route samples between experts")
        if config.backbone_pretrained:
            resolve_expert_checkpoints(
                config.backbone_pretrained, config.backbone_subfolder, config.backbone_subfolder_2
            )

        super().__init__(
            config=config,
            pg_collection=pg_collection,
            encoder_configs=encoder_configs,
        )

        self.default_boundary_timestep = config.boundary_timestep
        self.transformer = build_wan_backbone(config, config.backbone_subfolder)
        self.transformer_2 = build_wan_backbone(config, config.backbone_subfolder_2)

    def set_input_tensor(self, input_tensor) -> None:
        """Accept Megatron's pipeline input tensor (PP=1, so nothing to wire)."""
        if isinstance(input_tensor, (list, tuple)):
            input_tensor = input_tensor[0] if len(input_tensor) > 0 else None
        self.input_tensor = input_tensor

    def _resolve_boundary(self, timestep: Tensor, boundary_timestep: Optional[Tensor]) -> Tensor:
        """Broadcast the routing boundary to ``[B]``."""
        if boundary_timestep is None:
            return torch.full_like(timestep, fill_value=self.default_boundary_timestep)
        if isinstance(boundary_timestep, (float, int)):
            return torch.full_like(timestep, fill_value=float(boundary_timestep))

        boundary_timestep = boundary_timestep.to(device=timestep.device, dtype=timestep.dtype)
        if boundary_timestep.shape == timestep.shape:
            return boundary_timestep
        try:
            return boundary_timestep.expand_as(timestep)
        except RuntimeError as exc:
            raise ValueError(
                f"boundary_timestep shape {tuple(boundary_timestep.shape)} is not "
                f"broadcastable to timestep shape {tuple(timestep.shape)}"
            ) from exc

    def forward(
        self,
        hidden_states: Tensor,
        timestep: Tensor,
        encoder_hidden_states: Tensor,
        *,
        boundary_timestep: Optional[Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,  # noqa: ARG002
    ) -> Tensor:
        """Dual-expert forward with per-sample routing.

        Args:
            hidden_states: ``[B, C, T, H, W]`` noisy latents.
            timestep: ``[B]`` timesteps on the ``[0, num_train_timesteps]`` axis
                (post-shift during training, matching diffusers inference).
            encoder_hidden_states: ``[B, S_txt, D_txt]`` UMT5 features.
            boundary_timestep: Optional ``[B]`` or scalar override of the
                routing boundary, on the same axis as ``timestep``. Defaults to
                ``boundary_ratio * num_train_timesteps``.
            attention_kwargs: Forwarded to the backbones.

        Returns:
            ``[B, C, T, H, W]`` velocity prediction, reassembled in input order.
        """
        boundary = self._resolve_boundary(timestep, boundary_timestep)
        high_mask = timestep >= boundary

        def run(backbone, latents, ts, ctx):
            return backbone(
                hidden_states=latents,
                timestep=ts,
                encoder_hidden_states=ctx,
                attention_kwargs=attention_kwargs,
                return_dict=False,
            )

        # Routing is per rank, so ranks may run different experts; the trainer
        # rejects this model above one data-parallel rank.
        if bool(torch.all(high_mask)):
            return run(self.transformer, hidden_states, timestep, encoder_hidden_states)
        if not bool(torch.any(high_mask)):
            return run(self.transformer_2, hidden_states, timestep, encoder_hidden_states)

        high_idx = high_mask.nonzero(as_tuple=False).squeeze(-1)
        low_idx = (~high_mask).nonzero(as_tuple=False).squeeze(-1)

        out_high = run(
            self.transformer,
            hidden_states.index_select(0, high_idx),
            timestep.index_select(0, high_idx),
            encoder_hidden_states.index_select(0, high_idx),
        )
        out_low = run(
            self.transformer_2,
            hidden_states.index_select(0, low_idx),
            timestep.index_select(0, low_idx),
            encoder_hidden_states.index_select(0, low_idx),
        )

        output = hidden_states.new_empty((hidden_states.shape[0], *out_high.shape[1:]))
        output.index_copy_(0, high_idx, out_high.to(output.dtype))
        output.index_copy_(0, low_idx, out_low.to(output.dtype))
        return output

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """Sharded state dict over both experts.

        Prefixes stay ``{prefix}transformer.*`` and ``{prefix}transformer_2.*``
        so checkpoints remain interoperable with the diffusers A14B layout.
        """
        sharded: Dict[str, Any] = {}
        for name, module in self.named_children():
            sharded.update(sharded_state_dict_default(module, f"{prefix}{name}.", sharded_offsets, metadata))
        return sharded


__all__ = ["Wan", "Wan2_2", "WanTransformer3D", "build_wan_backbone"]
