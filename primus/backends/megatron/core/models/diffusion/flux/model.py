# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Portions copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Flux diffusion model implementation.

This module implements the complete Flux model, integrating all components:
- Input embeddings and position encodings
- MMDiT (joint image-text) transformer blocks
- Single (image-only) transformer blocks
- Output projection

API Note:
    Primus's Flux model uses Megatron-Core's TransformerBlock for unified
    layer management, enabling heterogeneous layer types (MMDiT + Single)
    in a single container.

    Architecture:
        - Input: Packed latents [S, B, C*4] (pre-processed)
        - Processing: Unified TransformerBlock with heterogeneous specs
        - Output: Packed predictions [S, B, C*4]
        - S = H*W/4 (2x2 spatial patches grouped into sequence tokens)

    Key Enhancement (vs traditional ModuleList approach):
        - Unified TransformerBlock with layer_specs for heterogeneous layers
        - Automatic pipeline parallelism splitting
        - Cleaner distributed checkpoint format

Reference:
    - Flux Paper: "Flux: A Scalable Diffusion Model for High-Resolution Image Synthesis"
    - Megatron-Core: Heterogeneous TransformerBlock patterns
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.utils import sharded_state_dict_default
from torch import Tensor

from primus.backends.megatron.core.models.common.diffusion_module.diffusion_module import (
    DiffusionModule,
)
from primus.backends.megatron.core.models.diffusion.common.embeddings import (
    MLPEmbedder,
    TimeStepEmbedder,
)
from primus.backends.megatron.core.models.diffusion.common.normalization import (
    AdaLNContinuous,
)
from primus.backends.megatron.core.models.diffusion.flux.config import FluxConfig
from primus.backends.megatron.core.models.diffusion.flux.layer_spec import (
    get_flux_layer_spec,
)
from primus.backends.megatron.core.models.diffusion.flux.layers import EmbedND
from primus.backends.megatron.core.transformer.diffusion_transformer_block import (
    DiffusionTransformerBlock,
)

_QK_RMSNORM_PARAM_ATTRS = (
    "sequence_parallel",
    "shared",
    "allreduce",
    "tensor_model_parallel",
    "partition_dim",
    "partition_stride",
)


class _TorchRMSNorm(nn.Module):
    """Compile-friendly RMSNorm replacement for TE's QK LayerNorm.

    TE's TENorm is decorated with @no_torch_dynamo, causing graph breaks
    inside compiled regions. This pure-PyTorch version is fully traceable.
    Matches NeMo's _TorchRMSNorm (custom_flux.py).
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, *, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.bias = None

    def reset_parameters(self):
        nn.init.ones_(self.weight)

    def forward(self, x: Tensor) -> Tensor:
        mean_sq = x.square().mean(dim=-1, keepdim=True, dtype=torch.float32)
        inv_rms = torch.rsqrt(mean_sq + self.eps).to(dtype=x.dtype)
        return x * inv_rms * self.weight


def _build_local_qk_rmsnorm(norm: nn.Module) -> "_TorchRMSNorm":
    """Build a _TorchRMSNorm from an existing TE norm, preserving weights and attrs."""
    weight = norm.weight
    local_norm = _TorchRMSNorm(
        weight.numel(),
        eps=getattr(norm, "eps", 1e-6),
        device=weight.device,
        dtype=weight.dtype,
    )
    local_norm.weight.data.copy_(weight.data)
    local_norm.weight.requires_grad_(weight.requires_grad)
    for attr_name in _QK_RMSNORM_PARAM_ATTRS:
        if hasattr(weight, attr_name):
            setattr(local_norm.weight, attr_name, getattr(weight, attr_name))
    return local_norm


class Flux(DiffusionModule):
    """
    Flux: Flow-based diffusion model with MMDiT architecture.

    Uses Megatron-Core's TransformerBlock with heterogeneous layer specs,
    combining MMDiT "double blocks" (joint image-text) with "single blocks"
    (image-only) for text-to-image generation.

    Model Variants:
        - Flux 535M: 1 joint + 1 single layer (~535M params, testing)
        - Flux 12B: 19 joint + 38 single layers (~12B params, production)

    Args:
        config: FluxConfig with all model parameters
        encoder_configs: Optional encoder configurations (VAE, T5, CLIP)
        pg_collection: ProcessGroupCollection for distributed training
        backend: Optional BackendSpecProvider for layer implementations (default: auto-selected)

    Forward args:
        img: Packed image latents [S_img, B, C*4] from VAE
        txt: Text embeddings [S_txt, B, D_txt] from T5-XXL
        y: CLIP pooled embeddings [B, D_pool]
        timesteps: Diffusion timesteps [B] in [0, 1]
        img_ids, txt_ids: Position IDs [B, S, 3]
        guidance: Optional guidance scale [B]

    Returns:
        Predicted velocity [S_img, B, C*4] in packed format

    torch.compile:
        Set enable_torch_compile=True in config. Compilation is applied AFTER
        FSDP/DDP wrapping by the trainer -- do NOT call compile_model() manually.

    Reference:
        - "Flux: A Scalable Diffusion Model for High-Resolution Image Synthesis"
        - Megatron-Core TransformerBlock
    """

    def __init__(
        self,
        config: FluxConfig,
        encoder_configs: Optional[Dict[str, Any]] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        backend: Optional["BackendSpecProvider"] = None,
    ):
        super().__init__(config, pg_collection=pg_collection, encoder_configs=encoder_configs)

        self.out_channels = config.in_channels
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.patch_size = config.patch_size
        self.in_channels = config.in_channels
        self.guidance_embed = config.guidance_embed

        # Position embedding (3D RoPE for image patches)
        # axes_dim=[16, 56, 56] for 1024x1024 images with 64 channels
        self.pos_embed = EmbedND(
            dim=self.hidden_size,
            theta=config.theta,
            axes_dim=list(config.axes_dim),
        )

        # Input embeddings
        # img_embed accepts packed format with in_channels already set to C*4 in config
        # Note: config.in_channels = 64 = 16 (VAE channels) * 4 (packing factor)
        self.img_embed = nn.Linear(config.in_channels, self.hidden_size)
        self.txt_embed = nn.Linear(config.context_dim, self.hidden_size)

        # Conditioning embeddings
        self.timestep_embedding = TimeStepEmbedder(
            embedding_dim=config.model_channels,
            hidden_dim=self.hidden_size,
        )
        self.vector_embedding = MLPEmbedder(
            in_dim=config.vec_in_dim,
            hidden_dim=self.hidden_size,
        )

        # Optional guidance embedding for classifier-free guidance
        if config.guidance_embed:
            self.guidance_embedding = MLPEmbedder(
                in_dim=config.model_channels,
                hidden_dim=self.hidden_size,
            )
        else:
            self.guidance_embedding = nn.Identity()

        # Create unified DiffusionTransformerBlock with heterogeneous layers
        # Layers 0-18: MMDiTLayer (joint processing)
        # Layers 19-56: FluxSingleTransformerBlock (single processing)
        self.transformer = DiffusionTransformerBlock(
            config=config,
            spec=get_flux_layer_spec(config, backend=backend),
            post_layer_norm=False,  # Flux uses AdaLN, not standard final layernorm
            pre_process=True,
            post_process=True,
        )

        # Output layers
        self.norm_out = AdaLNContinuous(
            config=config,
            conditioning_embedding_dim=self.hidden_size,
        )
        self.proj_out = nn.Linear(
            self.hidden_size,
            self.patch_size * self.patch_size * self.out_channels,
            bias=True,
        )

        # NOTE: torch.compile is NOT applied here. It must be applied AFTER
        # distributed wrapping (FSDP/DDP). The training framework will call
        # compile_model() at the appropriate time.

        # Stack runner references (replaced with compiled versions by compile_model)
        self._double_block_stack_runner = self._run_double_block_stack
        self._single_block_stack_runner = self._run_single_block_stack
        self._output_head_runner = self._run_output_head
        self._full_dit_runner = self._run_full_dit

        # Replace TE QK RMSNorm BEFORE DDP wrapping so that DDP's forward
        # pre-hook registration sees the final module tree.
        if config.torch_compile_replace_qk_rmsnorm and config.transformer_impl == "transformer_engine":
            try:
                from primus.core.utils.module_utils import log_rank_0 as _log
            except Exception:
                _log = print
            self._replace_qk_rmsnorm(_log)

        self.init_weights()

    def init_weights(self):
        """
        Custom weight initialization matching NVIDIA MLPerf Training v5.1.

        - img_embed, txt_embed: Xavier uniform
        - timestep_embedding, vector_embedding: Normal(std=0.02) for MLP layers
        - Per-block AdaLN modulations: re-zeroed after construction (construction
          now uses normal_ init_method to match NeMo's RNG sequence)
        - norm_out.adaLN_modulation: zero-init (last linear layer)
        - norm_out.norm: reset to default
        - proj_out: zero-init weight and bias
        """
        from primus.core.utils.module_utils import log_rank_0

        # Embedders: Xavier uniform
        nn.init.xavier_uniform_(self.img_embed.weight)
        nn.init.constant_(self.img_embed.bias, 0)
        nn.init.xavier_uniform_(self.txt_embed.weight)
        nn.init.constant_(self.txt_embed.bias, 0)

        # Timestep embedder MLP: Normal(std=0.02)
        self._init_mlpembedder(self.timestep_embedding.time_embedding)
        self._init_mlpembedder(self.vector_embedding)

        if self.guidance_embed and not isinstance(self.guidance_embedding, nn.Identity):
            self._init_mlpembedder(self.guidance_embedding)

        # Per-block AdaLN: zero modulation weights and reset LayerNorms.
        # Order matches NeMo: single blocks first, then double blocks.
        # These are deterministic ops (no RNG consumption).
        for layer in self.transformer.layers[self.config.num_joint_layers :]:
            nn.init.constant_(layer.adaln.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(layer.adaln.adaLN_modulation[-1].bias, 0)
            layer.adaln.ln.reset_parameters()

        for layer in self.transformer.layers[: self.config.num_joint_layers]:
            nn.init.constant_(layer.adaln.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(layer.adaln.adaLN_modulation[-1].bias, 0)
            layer.adaln.ln.reset_parameters()
            layer.adaln.ln2.reset_parameters()
            nn.init.constant_(layer.adaln_context.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(layer.adaln_context.adaLN_modulation[-1].bias, 0)
            layer.adaln_context.ln.reset_parameters()
            layer.adaln_context.ln2.reset_parameters()

        # Output layers: zero-init for stable training start
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

        # norm_out (AdaLNContinuous): zero-init modulation, reset norm
        nn.init.constant_(self.norm_out.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.norm_out.adaLN_modulation[-1].bias, 0)
        self.norm_out.norm.reset_parameters()

        log_rank_0("Applied custom weight initialization (MLPerf v5.1 aligned, NeMo RNG-matched)")

    @staticmethod
    def _init_mlpembedder(module, init_std: float = 0.02):
        """Initialize MLPEmbedder with Normal(std) for both linear layers."""
        nn.init.normal_(module.in_layer.weight, std=init_std)
        nn.init.constant_(module.in_layer.bias, 0)
        nn.init.normal_(module.out_layer.weight, std=init_std)
        nn.init.constant_(module.out_layer.bias, 0)

    def compile_model(self):
        """
        Apply torch.compile to the model using the configured strategy.

        CRITICAL: This must be called AFTER distributed wrapping (FSDP/DDP) is complete
        AND after ddp_config is set. Do NOT call this in __init__. The training
        framework will call this at the appropriate time.

        Strategies:
            per_block:   compile each transformer layer individually (default, recommended)
            whole_model: compile entire forward (incompatible with overlap_param_gather)
            double_stack: compile the double (joint) block loop
            single_stack: compile the single block loop
            stack:        compile both double and single block loops
            full_dit:     compile entire DiT as one region
        """
        import os

        if not self.config.enable_torch_compile:
            return

        try:
            from primus.core.utils.module_utils import log_rank_0

            log = log_rank_0
        except Exception:
            log = print

        strategy = self.config.torch_compile_strategy
        valid_strategies = {"whole_model", "per_block", "double_stack", "single_stack", "stack", "full_dit"}
        if strategy not in valid_strategies:
            raise ValueError(
                f"Invalid torch_compile_strategy='{strategy}'. " f"Must be one of: {sorted(valid_strategies)}"
            )

        is_te = self.config.transformer_impl == "transformer_engine"

        if not is_te and strategy not in ("whole_model", "per_block"):
            log(
                f"WARNING: torch_compile_strategy='{strategy}' is designed for TE spec. "
                f"Local spec modules are already torch.compile-friendly; "
                f"falling back to 'per_block' strategy."
            )
            strategy = "per_block"

        if (
            strategy not in ("whole_model", "per_block")
            and getattr(self.config, "recompute_granularity", None) == "full"
        ):
            raise ValueError(
                f"Activation checkpointing (recompute_granularity='full') is incompatible "
                f"with torch_compile_strategy='{strategy}'. Stack runners bypass the "
                f"checkpointed forward path."
            )

        if strategy == "whole_model":
            from megatron.training import get_args

            args = get_args()
            if getattr(args, "overlap_param_gather", False):
                raise ValueError(
                    "whole_model compile strategy is incompatible with overlap_param_gather. "
                    "DDP hooks are traced inside the compiled graph, causing ~20% convergence "
                    "degradation. Use 'per_block' strategy instead, or disable overlap_param_gather."
                )

        if strategy == "per_block" and getattr(self.config, "enable_cuda_graph", False):
            log(
                "WARNING: per_block compile strategy may conflict with per-layer "
                "CUDA graph __call__ overrides on MMDiTLayer/FluxSingleTransformerBlock."
            )

        compile_kwargs = {
            "backend": self.config.torch_compile_backend,
            "mode": self.config.torch_compile_mode,
            "fullgraph": self.config.torch_compile_fullgraph,
        }

        if is_te and self.config.torch_compile_disable_inductor_cudagraphs:
            torch._inductor.config.triton.cudagraphs = False
            torch._inductor.config.triton.cudagraph_trees = False
            os.environ["TORCHINDUCTOR_CUDAGRAPHS"] = "0"
            log("  Disabled Inductor CUDA graphs (TE FP8 compatibility)")

        if self.config.torch_compile_emulate_precision_casts:
            torch._inductor.config.emulate_precision_casts = True
            log(
                "  Enabled emulate_precision_casts "
                "(preserves eager BF16 precision semantics in fused Triton kernels)"
            )

        if not self.config.torch_compile_fused_ln_modulate:
            for m in self.modules():
                if hasattr(m, "use_fused_ln_modulate") and not getattr(m, "_adaln_plain_ops", False):
                    m.use_fused_ln_modulate = False
            log("  Disabled fused LN+modulate Triton kernels (using separate opaque ops)")

        if strategy == "whole_model":
            self._compile_whole_model(compile_kwargs, log)
        elif strategy == "per_block":
            log("=" * 80)
            log("Applying PER-BLOCK torch.compile on Flux...")
            for i, layer in enumerate(self.transformer.layers):
                layer.forward = torch.compile(layer.forward, **compile_kwargs)
            log(f"  Compiled {len(self.transformer.layers)} layers individually")
            log("=" * 80)
        elif strategy == "double_stack":
            log("=" * 80)
            log("Applying DOUBLE-STACK torch.compile on Flux...")
            self._double_block_stack_runner = torch.compile(self._run_double_block_stack, **compile_kwargs)
            log(f"  Compiled double block stack ({self.config.num_joint_layers} layers)")
            log("=" * 80)
        elif strategy == "single_stack":
            log("=" * 80)
            log("Applying SINGLE-STACK torch.compile on Flux...")
            self._single_block_stack_runner = torch.compile(self._run_single_block_stack, **compile_kwargs)
            log(f"  Compiled single block stack ({self.config.num_single_layers} layers)")
            log("=" * 80)
        elif strategy == "stack":
            log("=" * 80)
            log("Applying STACK torch.compile on Flux (double + single)...")
            self._double_block_stack_runner = torch.compile(self._run_double_block_stack, **compile_kwargs)
            self._single_block_stack_runner = torch.compile(self._run_single_block_stack, **compile_kwargs)
            log(
                f"  Compiled double ({self.config.num_joint_layers}) "
                f"+ single ({self.config.num_single_layers}) block stacks"
            )
            log("=" * 80)
        elif strategy == "full_dit":
            log("=" * 80)
            log("Applying FULL-DIT torch.compile on Flux...")
            self._full_dit_runner = torch.compile(self._run_full_dit, **compile_kwargs)
            log("  Compiled entire DiT as one region")
            log("=" * 80)

    def _compile_whole_model(self, compile_kwargs, log):
        """Apply a single torch.compile on self.forward (whole-model strategy)."""
        import torch

        log("=" * 80)
        log("Applying WHOLE-MODEL torch.compile on Flux...")
        log(f"  Backend: {compile_kwargs['backend']}")
        log(f"  Mode: {compile_kwargs['mode']}")
        log(f"  Fullgraph: {compile_kwargs['fullgraph']}")
        log("=" * 80)
        self.forward = torch.compile(self.forward, **compile_kwargs)
        log("  Done (actual compilation deferred to first forward pass)")
        log("=" * 80)

    def _replace_qk_rmsnorm(self, log):
        """Replace TE QK RMSNorm modules with compile-friendly _TorchRMSNorm.

        Only meaningful for transformer_impl="transformer_engine" where QK norms
        are TENorm instances (opaque to torch.compile). Local spec already uses
        torch.nn.RMSNorm which is fully traceable.
        """
        count = 0
        for layer in self.transformer.layers:
            attn = getattr(layer, "self_attention", None)
            if attn is None:
                continue
            for attr in ("q_layernorm", "k_layernorm", "added_q_layernorm", "added_k_layernorm"):
                old_norm = getattr(attn, attr, None)
                if old_norm is not None and not isinstance(old_norm, _TorchRMSNorm):
                    setattr(attn, attr, _build_local_qk_rmsnorm(old_norm))
                    count += 1
        log(f"  Replaced {count} TE QK RMSNorm modules with compile-friendly _TorchRMSNorm")

    # ------------------------------------------------------------------
    # Embedding computation: compilable helper that converts raw inputs
    # into hidden states, conditioning vectors, and RoPE frequencies.
    # ------------------------------------------------------------------

    def _compute_embeddings(self, img, txt, timesteps, img_ids, txt_ids, guidance, y):
        """Compute all embeddings and RoPE in a single compilable region."""
        hidden_states = self.img_embed(img)
        encoder_hidden_states = self.txt_embed(txt)

        if hidden_states.dim() == 3 and hidden_states.shape[0] < hidden_states.shape[1]:
            hidden_states = hidden_states.transpose(0, 1)
        if (
            encoder_hidden_states.dim() == 3
            and encoder_hidden_states.shape[0] < encoder_hidden_states.shape[1]
        ):
            encoder_hidden_states = encoder_hidden_states.transpose(0, 1)

        txt_seq_len = encoder_hidden_states.shape[0]

        timesteps = timesteps.to(hidden_states.dtype) * 1000.0
        vec_emb = self.timestep_embedding(timesteps)

        if guidance is not None:
            guidance_emb = self.guidance_embedding(self.timestep_embedding.time_proj(guidance * 1000.0))
            vec_emb = vec_emb + guidance_emb

        vec_emb = vec_emb + self.vector_embedding(y)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        rotary_pos_emb = self.pos_embed(ids)

        return hidden_states, encoder_hidden_states, vec_emb, rotary_pos_emb, txt_seq_len

    # ------------------------------------------------------------------
    # Stack runners: self-contained compilable callables that iterate
    # transformer layers directly (bypassing DiffusionTransformerBlock).
    # ------------------------------------------------------------------

    def _run_double_block_stack(self, hidden_states, encoder_hidden_states, rotary_pos_emb, vec_emb):
        """Run all double (joint MMDiT) blocks as one compilable region."""
        from megatron.core.utils import make_viewless_tensor

        hidden_states = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )
        for layer in self.transformer.layers[: self.config.num_joint_layers]:
            hidden_states, encoder_hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=None,
                context=encoder_hidden_states,
                context_mask=None,
                rotary_pos_emb=rotary_pos_emb,
                timestep_emb=vec_emb,
            )
        return hidden_states, encoder_hidden_states

    def _run_single_block_stack(self, hidden_states, rotary_pos_emb, vec_emb):
        """Run all single blocks as one compilable region.

        The caller must concatenate [context, hidden] before entering this runner.
        All single blocks receive context=None (concat already done).
        """
        for layer in self.transformer.layers[self.config.num_joint_layers :]:
            hidden_states, _ = layer(
                hidden_states=hidden_states,
                attention_mask=None,
                context=None,
                context_mask=None,
                rotary_pos_emb=rotary_pos_emb,
                timestep_emb=vec_emb,
            )
        return hidden_states

    def _run_output_head(self, hidden_states, txt_seq_len, vec_emb):
        """Run the output head (slice + norm_out + proj_out)."""
        hidden_states = hidden_states[txt_seq_len:, ...]
        hidden_states = self.norm_out(hidden_states, vec_emb)
        return self.proj_out(hidden_states)

    def _run_full_dit(self, hidden_states, encoder_hidden_states, rotary_pos_emb, vec_emb, txt_seq_len):
        """Run the entire DiT (double + cat + single + output) as one compiled region."""
        hidden_states, encoder_hidden_states = self._run_double_block_stack(
            hidden_states, encoder_hidden_states, rotary_pos_emb, vec_emb
        )
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=0)
        hidden_states = self._run_single_block_stack(hidden_states, rotary_pos_emb, vec_emb)
        if self.transformer.final_layernorm is not None:
            hidden_states = self.transformer.final_layernorm(hidden_states)
        return self._run_output_head(hidden_states, txt_seq_len, vec_emb)

    def set_input_tensor(self, input_tensor):
        """
        Set the input tensor on the underlying TransformerBlock.

        Required by the Megatron model interface (the scheduler calls this on
        every model). Pipeline parallelism is not supported for diffusion
        models (rejected at config construction; PP > 1 raises), so only the
        single-stage case is handled here.

        Args:
            input_tensor: Union[Tensor, List[Tensor], None]
        """
        if input_tensor is None:
            return

        # Handle list or single tensor
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        # Pass image hidden states to TransformerBlock
        self.transformer.set_input_tensor(input_tensor[0])

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """
        Generate sharded state dictionary for distributed checkpointing.

        With TransformerBlock, this delegates to the transformer which handles
        all layer numbering automatically for heterogeneous layers.

        Flux is architecturally heterogeneous: joint MMDiT blocks carry params
        (e.g. ``self_attention.added_linear_qkv.*``, ``context_mlp.*``) that the
        single DiT blocks intentionally omit (FluxSingleAttention rebuilds
        ``linear_proj`` with ``bias=False``). Megatron's heterogeneous, per-layer
        indexed sharded keys (``transformer.layers.<i>.*``) are required here:
        the homogeneous layer-stacked path would leave unclaimed slots for the
        params absent in single blocks and raise a CheckpointingException at
        save time. TransformerBlock.sharded_state_dict provides this
        automatically, so no config toggle is needed.

        Args:
            prefix: Prefix for state dict keys (e.g., 'module.')
            sharded_offsets: Pipeline parallel offsets
            metadata: Optional metadata for checkpoint conversion

        Returns:
            Dictionary mapping state dict keys to ShardedTensor objects
        """
        sharded_state_dict = {}

        # Delegate transformer layers to TransformerBlock
        # Handles heterogeneous layers automatically
        sharded_state_dict.update(
            self.transformer.sharded_state_dict(f"{prefix}transformer.", sharded_offsets, metadata)
        )

        # Handle other submodules (embeddings, projections, norms)
        for name, module in self.named_children():
            if module is not self.transformer:
                sharded_state_dict.update(
                    sharded_state_dict_default(module, f"{prefix}{name}.", sharded_offsets, metadata)
                )

        return sharded_state_dict

    def get_fp8_context(self):
        """
        Return FP8 context manager for FP8 training.

        When transformer_impl="local", FP8 is handled per-module at init time
        (inside Float8ColumnParallelLinear / Float8RowParallelLinear), so no
        global FP8 context manager is needed. This avoids mutable global state
        that would cause torch.compile graph breaks.

        Returns:
            Context manager for FP8 operations (or nullcontext if FP8 disabled
            or using local spec)
        """
        from contextlib import nullcontext

        if self.config.transformer_impl == "local":
            return nullcontext()

        from megatron.core.fp8_utils import get_fp8_context as get_fp8_context_util

        return get_fp8_context_util(self.config)

    def forward(
        self,
        img: Tensor,
        txt: Tensor,
        y: Tensor,
        timesteps: Tensor,
        img_ids: Tensor,
        txt_ids: Tensor,
        guidance: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass through Flux model.

        Args:
            img: Image latents [S_img, B, C*4] or [B, S_img, C*4] - PACKED format from prepare_flux_latents
                 where S_img = H*W/4 (2x2 patches grouped)
            txt: Text embeddings [S_txt, B, D_txt] or [B, S_txt, D_txt] from T5-XXL
            y: CLIP pooled embeddings [B, D_pool] from CLIP-L
            timesteps: Diffusion timesteps [B] in range [0, 1]
            img_ids: Position IDs for image patches [B, S_img, 3]
            txt_ids: Position IDs for text tokens [B, S_txt, 3]
            guidance: Optional guidance scale [B] for classifier-free guidance

        Returns:
            Predicted velocity [S_img, B, C*4] or [B, S_img, C*4] in packed format (matches input format)

        Note:
            The model accepts pre-packed
            latents from the wrapper/training code. The packing groups 2x2 spatial patches,
            reducing spatial dimensions by 4x and increasing channels by 4x.
        """
        hidden_states, encoder_hidden_states, vec_emb, rotary_pos_emb, txt_seq_len = self._compute_embeddings(
            img, txt, timesteps, img_ids, txt_ids, guidance, y
        )

        strategy = self.config.torch_compile_strategy

        if strategy in ("whole_model", "per_block"):
            # Original path: delegate to DiffusionTransformerBlock
            with self.get_fp8_context():
                hidden_states = self.transformer(
                    hidden_states=hidden_states,
                    attention_mask=None,
                    context=encoder_hidden_states,
                    context_mask=None,
                    rotary_pos_emb=rotary_pos_emb,
                    rotary_pos_cos=None,
                    rotary_pos_sin=None,
                    attention_bias=None,
                    timestep_emb=vec_emb,
                    packed_seq_params=None,
                )

            hidden_states = hidden_states[txt_seq_len:, ...]
            hidden_states = self.norm_out(hidden_states, vec_emb)
            output = self.proj_out(hidden_states)

        elif strategy == "full_dit":
            # One large compiled region covering double + cat + single + output
            with self.get_fp8_context():
                output = self._full_dit_runner(
                    hidden_states, encoder_hidden_states, rotary_pos_emb, vec_emb, txt_seq_len
                )

        else:
            # stack / double_stack / single_stack: compiled stack runners
            # with eager transitions between them
            with self.get_fp8_context():
                hidden_states, encoder_hidden_states = self._double_block_stack_runner(
                    hidden_states, encoder_hidden_states, rotary_pos_emb, vec_emb
                )

            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=0)

            with self.get_fp8_context():
                hidden_states = self._single_block_stack_runner(hidden_states, rotary_pos_emb, vec_emb)

            if self.transformer.final_layernorm is not None:
                hidden_states = self.transformer.final_layernorm(hidden_states)

            output = self._output_head_runner(hidden_states, txt_seq_len, vec_emb)

        return output

    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        convert_from_hf: bool = False,
        save_converted_to: Optional[Union[str, Path]] = None,
        strict: bool = False,
    ) -> Tuple[List[str], List[str]]:
        """
        Load checkpoint into Flux model.

        Supports both Primus native checkpoints and HuggingFace checkpoints
        with automatic conversion.

        Note: HuggingFace checkpoint conversion is handled by the checkpoint_converter module.

        Args:
            checkpoint_path: Path to checkpoint file or directory
            convert_from_hf: If True, convert from HuggingFace format
            save_converted_to: Optional path to save converted checkpoint
            strict: Whether to use strict state dict loading

        Returns:
            Tuple of (missing_keys, unexpected_keys)

        Example:
            >>> config = FluxConfig.flux_12b()
            >>> model = Flux(config=config)
            >>> # Load native Primus checkpoint
            >>> missing, unexpected = model.load_checkpoint("primus_flux_12b.safetensors")
        """
        from safetensors.torch import load_file as load_safetensors

        from primus.core.utils.module_utils import error_rank_0, log_rank_0, warning_rank_0

        if convert_from_hf:
            try:
                from .checkpoint_converter import convert_hf_checkpoint

                log_rank_0(f"Converting HuggingFace checkpoint: {checkpoint_path}")
                state_dict = convert_hf_checkpoint(
                    checkpoint_path,
                    flux_config=self.config,
                    save_to=save_converted_to,
                )
            except ImportError:
                raise NotImplementedError(
                    "HuggingFace checkpoint conversion requires the checkpoint_converter module. "
                    "Please use a pre-converted Primus checkpoint."
                )
        else:
            # Load native Primus checkpoint
            log_rank_0(f"Loading Primus checkpoint: {checkpoint_path}")
            checkpoint_path = Path(checkpoint_path)

            if checkpoint_path.is_dir():
                # Load all .safetensors files
                safetensor_files = list(checkpoint_path.glob("*.safetensors"))
                if not safetensor_files:
                    raise FileNotFoundError(f"No .safetensors files in {checkpoint_path}")

                state_dict = {}
                for file in safetensor_files:
                    state_dict.update(load_safetensors(str(file)))
            else:
                state_dict = load_safetensors(str(checkpoint_path))

        # Load state dict into model
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=strict)

        # Filter out _extra_state keys (Megatron-specific metadata, expected to be missing)
        missing_keys = [k for k in missing_keys if not k.endswith("_extra_state")]

        # Check for critical missing keys
        critical_patterns = ["timestep_embedding", "img_embed", "txt_embed", "vector_embedding"]
        critical_missing = [k for k in missing_keys if any(pattern in k for pattern in critical_patterns)]

        if critical_missing:
            variant = "flux_12b" if self.config.num_joint_layers == 19 else "flux_535m"
            err_lines = [
                "\n" + "=" * 80,
                "CRITICAL ERROR: Key model weights are missing!",
                "=" * 80,
                f"Missing {len(critical_missing)} critical keys:",
            ]
            err_lines += [f"  - {key}" for key in critical_missing[:10]]
            if len(critical_missing) > 10:
                err_lines.append(f"  ... and {len(critical_missing) - 10} more")
            err_lines += [
                "\nThis usually means:",
                "  1. Checkpoint format mismatch (HuggingFace vs Primus)",
                "  2. Checkpoint was converted with an older version",
                "  3. Checkpoint file is corrupted or incomplete",
                "\nTo fix, reconvert the checkpoint:",
                "    python tools/checkpoint_conversion/convert_flux_hf_to_primus.py \\",
                "      --input black-forest-labs/FLUX.1-dev/transformer \\",
                f"      --output {checkpoint_path} \\",
                f"      --variant {variant}",
                "=" * 80,
            ]
            error_rank_0("\n".join(err_lines))

            raise RuntimeError(
                f"Critical model weights missing ({len(critical_missing)} keys). "
                f"Checkpoint may be corrupted or incompatible. See details above."
            )

        if missing_keys:
            warning_rank_0(f"Missing keys ({len(missing_keys)}): {missing_keys[:5]}...")
        if unexpected_keys:
            warning_rank_0(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}...")

        log_rank_0("Checkpoint loaded successfully!")
        return missing_keys, unexpected_keys
