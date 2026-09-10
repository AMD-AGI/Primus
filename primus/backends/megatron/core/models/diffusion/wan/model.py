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
embedders, ``scale_shift_table`` AdaLN modulation, output head) and builds its
attention and FFN from the backend-resolved specs in ``layer_spec.py``.

The transformer stack runs sequence-major ``[S, B, dim]`` end to end, matching
Megatron-Bridge: AdaLN modulation, the LayerNorms, residuals, and the attention
core all operate on that layout, so the block GEMMs dispatch identical kernels.
Tokens are transposed into ``[S, B, dim]`` after the patch embed and back to
``[B, S, dim]`` before the final unpatchify.
"""

import math
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.transformer.utils import sharded_state_dict_default
from safetensors.torch import load_file as load_safetensors
from torch import Tensor

from ...common.diffusion_module.diffusion_module import DiffusionModule
from .config import WanConfig
from .layer_spec import WanLayerSpec, get_wan_layer_spec
from .layers import WanConditionEmbedder, WanRotaryPosEmbed
from .utils import fold_patches_into_batch, patch_grid, unpatchify


class WanTransformerBlock(nn.Module):
    """One WAN DiT block: gated self-attention, cross-attention, gated FFN.

    AdaLN modulation, the LayerNorms, and the residual adds run in the tensor's
    native dtype with plain ``nn.LayerNorm`` -- Megatron-Bridge's
    normalize/modulate/scale-add math, not the diffusers explicit-fp32 path.
    """

    def __init__(self, config: WanConfig, spec: WanLayerSpec, layer_number: int):
        super().__init__()
        dim = config.hidden_size
        eps = config.layernorm_epsilon

        # Carried so the block satisfies the one TransformerLayer attribute
        # Megatron reads off an FSDP unit (see the registration below).
        self.layer_number = layer_number

        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.attn1 = build_module(
            spec.self_attention,
            config=spec.attention_config,
            submodules=spec.self_attention.submodules,
            layer_number=layer_number,
            rope_config=spec.rope_config,
            norm_config=spec.norm_config,
        )
        self.attn2 = build_module(
            spec.cross_attention,
            config=spec.attention_config,
            submodules=spec.cross_attention.submodules,
            layer_number=layer_number,
            norm_config=spec.norm_config,
        )
        self.norm2 = (
            nn.LayerNorm(dim, eps=eps, elementwise_affine=True) if config.cross_attn_norm else nn.Identity()
        )
        self.ffn = MLP(
            config=spec.ffn_config,
            submodules=spec.mlp,
            ffn_hidden_size=config.ffn_hidden_size,
        )
        self.norm3 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        temb: Tensor,
        rotary_freqs: Tensor,
    ) -> Tensor:
        # temb is [B, 6, dim]; transposing gives modulation terms of [1, B, dim]
        # so they broadcast over the sequence-major activations.
        mod = (self.scale_shift_table + temb).transpose(0, 1)
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = mod.chunk(6, dim=0)

        norm_h = self.norm1(hidden_states) * (1 + scale_msa) + shift_msa
        attn_out, attn_bias = self.attn1(norm_h, rotary_freqs)
        if attn_bias is not None:
            attn_out = attn_out + attn_bias
        hidden_states = hidden_states + gate_msa * attn_out

        norm_h = self.norm2(hidden_states)
        attn_out, attn_bias = self.attn2(norm_h, encoder_hidden_states)
        if attn_bias is not None:
            attn_out = attn_out + attn_bias
        hidden_states = hidden_states + attn_out

        norm_h = self.norm3(hidden_states) * (1 + c_scale_msa) + c_shift_msa
        # The mcore MLP returns (output, fc2_bias) under skip_bias_add; diffusers
        # folds that bias in before the gate scaling, so add it here.
        ff_out, ff_bias = self.ffn(norm_h)
        if ff_bias is not None:
            ff_out = ff_out + ff_bias
        return hidden_states + c_gate_msa * ff_out


# Make the block a Megatron-FSDP sharding unit.
#
# Megatron-FSDP gathers and releases parameters at the granularity of its "FSDP
# unit modules". When the caller passes none -- and ``megatron/training/
# training.py`` never does, it builds the wrapper with no ``fsdp_unit_modules``
# argument and ``DistributedDataParallelConfig`` has no field for one -- the
# adapter falls back to ``[TransformerLayer]`` for the ``optim_grads_params``
# strategy WAN runs under. ``WanTransformerBlock`` deliberately is not a
# Megatron ``TransformerLayer`` (see ``layer_spec.py`` on why WAN does not build
# on ``TransformerBlock``), so that default matched zero modules and left the
# unit list empty.
#
# An empty unit list costs more than coarse sharding. The hook-registration loop
# skips modules that live inside a registered unit; with nothing registered that
# skip never fires, so every norm, linear and attention in all 30 blocks takes a
# pre-forward unshard hook plus an fp8-transpose-cache post-hook that a bf16 run
# has no use for. Any hook forces ``nn.Module._call_impl`` off its no-hook fast
# path, so the Dynamo graph breaks at every submodule boundary inside a block
# instead of only at the block boundary.
#
# Registering as a virtual subclass makes Megatron's own default match the
# block. The alternatives were worse: Megatron-LM is an upstream submodule so
# the argument cannot be plumbed through ``training.py``, and real inheritance
# from ``TransformerLayer`` would change the checkpoint key layout.
#
# Eager trades a little step time for peak memory, because the per-block
# all-gathers cost more collective time than one whole-model gather. Compiled
# improves on both, as the de-fragmented graph lets Inductor drop casts it
# previously could not elide across a break. Loss is unchanged.
#
# ``register`` only affects ``isinstance``. The other Megatron sites that test
# for ``TransformerLayer`` are either unreachable for WAN (CUDA graphs, Mamba
# and hybrid blocks, GPT callables) or keyed on ``layers.N`` parameter names,
# and WAN's are ``blocks.N``; ``layer_number`` is set above for the one
# attribute the FSDP-DTensor checkpoint path would read.
TransformerLayer.register(WanTransformerBlock)


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

        layer_specs = get_wan_layer_spec(config)
        self.blocks = nn.ModuleList(
            [WanTransformerBlock(config, spec, layer_number=i + 1) for i, spec in enumerate(layer_specs)]
        )

        self.norm_out = nn.LayerNorm(dim, eps=config.layernorm_epsilon, elementwise_affine=False)
        self.proj_out = nn.Linear(dim, self.out_channels * math.prod(self.patch_size_3d))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

        self.gradient_checkpointing = getattr(config, "recompute_granularity", None) == "full"

        if config.adaln_zero_init:
            self._init_adaln_zero()

    def _init_adaln_zero(self) -> None:
        """AdaLN-zero init: identity blocks plus a zeroed output head."""
        for block in self.blocks:
            nn.init.zeros_(block.scale_shift_table)
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

        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            for block in self.blocks:
                hs = torch.utils.checkpoint.checkpoint(
                    block, hs, enc, timestep_proj, rotary_freqs, use_reentrant=False
                )
        else:
            for block in self.blocks:
                hs = block(hs, enc, timestep_proj, rotary_freqs)

        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).transpose(0, 1).chunk(2, dim=0)
        hs = self.norm_out(hs) * (1 + scale) + shift
        hs = self.proj_out(hs)
        hs = hs.transpose(0, 1).contiguous()

        return unpatchify(hs, batch_size, grid, self.patch_size_3d)


def _load_backbone_checkpoint(
    backbone: nn.Module, checkpoint_path: str, subfolder: Optional[str] = None
) -> None:
    """Load a fused-qkv WAN backbone checkpoint.

    Accepts a ``.safetensors`` file, a ``.pt``/``.pth`` file, or a directory of
    ``.safetensors`` shards. The checkpoint must already be in the fused
    ``linear_qkv`` layout produced by ``checkpoint_converter``; a raw diffusers
    checkpoint with split ``to_q``/``to_k``/``to_v`` will not load.
    """
    path = Path(checkpoint_path)
    if subfolder:
        path = path / subfolder

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

        output = torch.empty_like(hidden_states)
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


__all__ = ["Wan", "Wan2_2", "WanTransformer3D", "WanTransformerBlock", "build_wan_backbone"]
