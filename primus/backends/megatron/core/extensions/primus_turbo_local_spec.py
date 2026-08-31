# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Primus Turbo Local Spec Provider

Clean implementation using native Megatron modules with Primus Turbo
optimizations. NO TransformerEngine dependencies.

Key Features:
- PrimusTurboLocalAttention: New class, inherits from MegatronModule (not TE)
- Native Megatron linear layers (ColumnParallelLinear, RowParallelLinear)
- Native LayerNorm (WrappedTorchNorm or FusedLayerNorm)
- Maximum torch.compile compatibility
- 52ms attention advantage preserved via pt.ops.flash_attn_func

Performance Target (vs Inductor baseline):
- Attention: Same (52ms advantage over Inductor's flash_attn)
- GEMM: -7ms (native PyTorch proven within 0.8% of TE)
- Elementwise: -157ms (torch.compile fusion)
- Reduce: -17ms (torch.compile fusion)
- Framework: -107ms (no TE overhead)
- Total: ~1,131ms vs Inductor's 1,415ms = 20% faster
"""

import math
import os
from typing import Optional

import primus_turbo.pytorch as pt
import torch
from primus_turbo.pytorch.ops.attention.flash_attn_interface import FlashAttnFunc

torch._dynamo.allow_in_graph(FlashAttnFunc)


@torch._dynamo.disable
def _advance_model_parallel_rng(batch_size: int, num_heads: int) -> None:
    """Simulate TE's CUDA RNG consumption inside
    PrimusTurboLocalAttention.

    TEDotProductAttention is instantiated with
    get_rng_state_tracker=get_cuda_rng_tracker, so TE's fused_attn_fwd kernel
    call runs inside tracker.fork() (the 'model-parallel-rng' generator state).
    The ASM kernel pulls a fresh philox state per call regardless of dropout,
    advancing model-parallel-rng by counter_offset = B*H*warp_size each
    attention call. Aiter's _ndropout path does not touch RNG, so PT and
    aiter attention see different downstream RNG sequences and diverge.

    This helper replays the same RNG advance. Decorated with @dynamo.disable
    so AOTAutograd never tries to trace through the generator manipulation.
    """
    warp_size = 64
    counter_offset = batch_size * num_heads * warp_size
    dev_idx = torch.cuda.current_device()
    gens = torch.cuda.default_generators
    if dev_idx >= len(gens):
        return
    gen = gens[dev_idx]

    def _advance_with(g):
        try:
            g.set_offset(g.get_offset() + counter_offset)
        except AttributeError:
            torch.empty(counter_offset, device="cuda", dtype=torch.int32).random_(generator=g)

    try:
        from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

        tracker = get_cuda_rng_tracker()
    except Exception:
        tracker = None

    if tracker is not None and getattr(tracker, "is_initialized", lambda: False)():
        with tracker.fork():
            _advance_with(gen)
    else:
        _advance_with(gen)


from megatron.core.models.backends import LocalSpecProvider
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import (
    get_context_parallel_group,
    get_tensor_model_parallel_group,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.global_vars import get_args
from torch import Tensor

from primus.backends.megatron.core.extensions.turbo_flex_attention import (
    build_turbo_flex_attention,
    reject_reset_attention_mask,
)


class PrimusTurboLocalAttention(MegatronModule):
    """
    Primus Turbo Flash Attention - Native Megatron Implementation.

    This is a COMPLETELY NEW class with ZERO TransformerEngine dependencies.

    Key differences from PrimusTurboAttention (in primus_turbo.py):
    - Inherits from MegatronModule (NOT te.pytorch.DotProductAttention)
    - No TE infrastructure or overhead
    - torch.compile friendly (no @no_torch_dynamo decorator)
    - Same 52ms performance advantage via pt.ops.flash_attn_func

    Used exclusively by: PrimusTurboLocalSpecProvider

    Performance:
    - 52ms faster than Inductor's flash_attn implementation
    - Same as existing PrimusTurboAttention but without TE overhead
    - torch.compile compatible for surrounding operations
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        k_channels: Optional[int] = None,
        v_channels: Optional[int] = None,
        cp_comm_type: str = "p2p",
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(config=config)

        self.config = config
        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type

        # Calculate softmax scale
        kv_channels = k_channels if k_channels is not None else config.kv_channels
        self.softmax_scale = softmax_scale or (1.0 / math.sqrt(kv_channels))

        # Setup process groups
        if pg_collection is None:
            pg_collection = ProcessGroupCollection(
                tp=get_tensor_model_parallel_group(check_initialized=False),
                cp=get_context_parallel_group(check_initialized=False),
            )

        # Select Primus Turbo flash attention variant
        args = get_args()
        if getattr(args, "use_turbo_flex_attention", False):
            # Same swap as PrimusTurboAttention: the compat layer mirrors
            # flash_attn_func's call signature, so forward() is unchanged. Wired here
            # too so the switch is never silently ignored under the local specs.
            self.attn_func = build_turbo_flex_attention(args=args, config=config)
        elif args.enable_turbo_attention_float8:
            self.attn_func = (
                pt.ops.flash_attn_fp8_usp_func
                if config.context_parallel_size > 1
                else pt.ops.flash_attn_fp8_func
            )
        else:
            self.attn_func = (
                pt.ops.flash_attn_usp_func if config.context_parallel_size > 1 else pt.ops.flash_attn_func
            )

        # The transpose in forward() produces a non-contiguous, SBHD-strided view.
        # aiter's v3 flash-attention backward mishandles that strided layout on
        # gfx942 (MI300X), corrupting gradients and causing grad-norm divergence
        # ~step 31; gfx950 (MI355X) handles it correctly. Feed contiguous BSHD on
        # gfx942 only, so gfx950 keeps the faster zero-copy strided path unchanged.
        # Computed here (not in forward) so torch.compile sees a constant guard
        # rather than a device query in the traced region.
        self.force_contiguous_qkv = torch.cuda.get_device_capability() < (9, 5)

        # Setup context parallel arguments
        self.attn_kwargs = {}
        if config.context_parallel_size > 1:
            self.attn_kwargs["ulysses_group"] = pg_collection.cp

        # Validate configuration
        if config.window_size is not None:
            raise ValueError("PrimusTurboLocalAttention does not support sliding window attention")

        # reset_attention_mask has nowhere to go here either: forward() reads no mask
        # tensor and none of the bound kernels take one. See the helper for why the
        # *default* causal mask is safe to drop but this one is not.
        reject_reset_attention_mask(
            args,
            is_flex=bool(getattr(args, "use_turbo_flex_attention", False)),
            where="PrimusTurboLocalAttention",
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tensor:
        """
        Forward pass using Primus Turbo flash attention.

        Args:
            query: Query tensor [seq_len, batch, num_heads, head_dim] (sbhd)
            key: Key tensor [seq_len, batch, num_heads, head_dim] (sbhd)
            value: Value tensor [seq_len, batch, num_heads, head_dim] (sbhd)
            attention_mask: unread. The bound kernels take the mask as a ``causal``
                flag, not a tensor. Sound only because reset_attention_mask -- the
                one setting under which Megatron's mask says more than ``causal``
                does -- is rejected in ``__init__``; otherwise the tensor is exactly
                ``torch.tril(...)``.
            attn_mask_type: Type of attention mask (causal, no_mask, etc.)
            attention_bias: rejected. Nothing below forwards it -- see the guard.
            packed_seq_params: rejected. Nothing below forwards it -- see the guard.

        Returns:
            Attention output [seq_len, batch, num_heads * head_dim] (merged heads)
        """
        if attention_bias is not None:
            # The kernel call below passes bias=None and alibi_slopes=None
            # unconditionally, and nothing between here and there reads this tensor.
            # Accepting it would train a model with no bias while the config says
            # there is one, and say nothing about it. Same refusal, same reason, as
            # PrimusTurboAttention.forward in primus_turbo.py.
            raise NotImplementedError(
                "PrimusTurboLocalAttention does not support attention_bias; the tensor "
                "would be silently dropped. This also covers ALiBi, which Megatron "
                "delivers as an attention bias -- use a position embedding this path "
                "implements."
            )

        if packed_seq_params is not None:
            # Unlike PrimusTurboAttention, this class has no thd branch at all: it
            # always calls the dense kernel with no cu_seqlens. A packed batch would
            # therefore be attended to as ONE long sequence, letting every document
            # see every earlier document -- wrong numbers, wrong gradients, clean log.
            # Refuse rather than approximate.
            raise NotImplementedError(
                "PrimusTurboLocalAttention does not support packed sequences; the "
                "document boundaries in packed_seq_params would be silently dropped and "
                "the batch attended to as one contiguous sequence. Use the "
                "TransformerEngine-based PrimusTurboAttention spec, which implements "
                "qkv_format='thd'."
            )

        query, key, value = [x.transpose(0, 1) for x in (query, key, value)]

        # gfx942: avoid aiter's broken strided-sbhd backward (see __init__).
        if self.force_contiguous_qkv:
            query, key, value = query.contiguous(), key.contiguous(), value.contiguous()

        causal = attn_mask_type == AttnMaskType.causal

        if os.environ.get("PRIMUS_PT_MIMIC_TE_RNG", "0") == "1":
            B = query.size(0)
            H = query.size(2)
            _advance_model_parallel_rng(B, H)

        output = self.attn_func(
            query,
            key,
            value,
            dropout_p=0.0,
            softmax_scale=self.softmax_scale,
            causal=causal,
            window_size=(-1, -1),
            bias=None,
            alibi_slopes=None,
            deterministic=False,
            return_lse=False,
            return_attn_probs=False,
            **self.attn_kwargs,
        )

        # Transpose back to Megatron format (bshd -> sbhd) and merge heads
        output = output.transpose(0, 1)
        output = output.reshape(output.shape[0], output.shape[1], -1)

        return output


class PrimusTurboMXFP4LocalSpecProvider(LocalSpecProvider):
    """
    Compile-friendly MXFP4 spec: Primus Turbo attention + MXFP4 linear layers.
    NO TransformerEngine. NO global FP4 state in forward path.
    Requires tensor_model_parallel_size=1.
    """

    def column_parallel_linear(self) -> type:
        from .primus_turbo_mxfp4_local import MXFP4ColumnParallelLinear

        return MXFP4ColumnParallelLinear

    def row_parallel_linear(self) -> type:
        from .primus_turbo_mxfp4_local import MXFP4RowParallelLinear

        return MXFP4RowParallelLinear

    def core_attention(self) -> type:
        return PrimusTurboLocalAttention


class PrimusTurboFloat8LocalSpecProvider(LocalSpecProvider):
    """
    Compile-friendly FP8 spec: Primus Turbo attention + FP8 linear layers.
    NO TransformerEngine. NO global FP8 state in forward path.
    NO changes to Primus-Turbo repo. Requires tensor_model_parallel_size=1.
    """

    def column_parallel_linear(self) -> type:
        from .primus_turbo_float8_local import Float8ColumnParallelLinear

        return Float8ColumnParallelLinear

    def row_parallel_linear(self) -> type:
        from .primus_turbo_float8_local import Float8RowParallelLinear

        return Float8RowParallelLinear

    def core_attention(self) -> type:
        return PrimusTurboLocalAttention


class PrimusTurboLocalSpecProvider(LocalSpecProvider):
    """
    Spec provider that extends LocalSpecProvider with Primus Turbo attention.

    Uses native Megatron modules everywhere except attention, where it uses
    PrimusTurboLocalAttention for the 52ms performance advantage.

    Key features:
    - NO TransformerEngine dependencies
    - Maximum torch.compile compatibility
    - Minimal custom code (~120 lines total)
    - Native PyTorch GEMM (proven within 0.8% of TE)
    - Primus Turbo flash attention (52ms advantage)

    Configuration:
        config.use_primus_turbo_local_spec = True

    Performance Target:
    - Attention: -52ms (Primus Turbo advantage over Inductor)
    - Operator fusion: -180ms (torch.compile)
    - Framework overhead: -107ms (no TE)
    - GEMM: ~same (native PyTorch excellent)
    - Total: ~1,131ms vs Inductor's 1,415ms (20% faster)

    vs Current TE implementation:
    - Total: ~1,131ms vs 1,588ms (28.8% faster)
    """

    def core_attention(self) -> type:
        """Return Primus Turbo local attention (NO TE dependency)"""
        return PrimusTurboLocalAttention

    # Everything else inherited from LocalSpecProvider:
    # - column_parallel_linear() -> ColumnParallelLinear (native Megatron)
    # - row_parallel_linear() -> RowParallelLinear (native Megatron)
    # - layer_norm() -> WrappedTorchNorm/FusedLayerNorm (compile-friendly)
    # - fuse_layernorm_and_linear() -> False (no fusion overhead)
    # - grouped_mlp_modules() -> SequentialMLP (native modules)
    # - activation_func() -> None (standard activation)
