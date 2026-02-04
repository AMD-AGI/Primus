"""
Zebra-Llama: Hybrid Mamba + Multi-Latent Attention (MLA) Model (HuggingFace).

This is a pragmatic HF implementation intended for:
- loading converted Megatron checkpoints
- running `generate()` / lm-eval

Notes / simplifications:
- KV cache is intentionally NOT supported (generation works, but is slower).
- Mamba mixer is a Megatron-shaped *simplified* implementation (not the full SSM scan).
- MLA follows Megatron’s tensorization and YaRN RoPE knobs, but uses PyTorch ops.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PretrainedConfig, PreTrainedModel
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import logging
from transformers.activations import ACT2FN

# from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
from mamba_ssm.ops.triton.ssd_combined import (
        mamba_chunk_scan_combined,
        mamba_split_conv1d_scan_combined,
    )
from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from einops import rearrange, repeat

logger = logging.get_logger(__name__)

try:
    # Optional: FlashAttention v2 (CUDA-only in most environments).
    from flash_attn import flash_attn_func  # type: ignore

    _HAVE_FLASH_ATTN = True
except Exception:
    flash_attn_func = None
    _HAVE_FLASH_ATTN = False

try:
    # Optional: Transformer Engine (used by Megatron for many fused ops).
    import transformer_engine.pytorch as te  # type: ignore

    _HAVE_TE = True
except Exception:
    te = None
    _HAVE_TE = False


# =============================================================================
# Config
# =============================================================================


class ZebraLlamaConfig(PretrainedConfig):
    model_type = "zebra_llama"

    def __init__(
        self,
        vocab_size: int = 128256,
        hidden_size: int = 2048,
        num_hidden_layers: int = 32,
        intermediate_size: int = 8192,
        num_attention_heads: int = 32,
        # MLA
        multi_latent_attention: bool = True,
        q_lora_rank: int = 1344,
        kv_lora_rank: int = 128,
        qk_head_dim: int = 32,
        qk_pos_emb_head_dim: int = 32,
        v_head_dim: int = 64,
        # Hybrid
        is_hybrid_model: bool = True,
        hybrid_attention_ratio: float = 0.25,
        # Mamba
        mamba_expand: int = 1,
        mamba_state_dim: int = 64,
        mamba_head_dim: int = 64,
        mamba_num_groups: int = 8,
        mamba_d_conv: int = 4,
        # Norm
        normalization: str = "RMSNorm",  # "LayerNorm" or "RMSNorm"
        layernorm_epsilon: float = 1e-5,
        # Dropout / residual
        hidden_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        residual_in_fp32: bool = False,
        bias_dropout_fusion: bool = True,
        # RoPE / YaRN
        rope_type: str = "yarn",  # "rope" or "yarn"
        rope_theta: float = 500000.0,  # megatron: rotary_base
        rotary_scaling_factor: float = 1.0,
        original_max_position_embeddings: int = 4096,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        mscale: float = 1.0,
        mscale_all_dim: float = 1.0,
        # HF misc
        max_position_embeddings: int = 131072,
        use_cache: bool = False,  # intentionally disabled
        tie_word_embeddings: bool = False,
        pad_token_id: Optional[int] = None,
        bos_token_id: int = 128000,
        eos_token_id: int = 128001,
        torch_dtype: str | torch.dtype | None = "bfloat16",
        # Optional: use Transformer Engine ops in HF model
        use_transformer_engine: bool = True,
        mamba_hidden_act: str = "silu",
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads

        self.multi_latent_attention = multi_latent_attention
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_head_dim = qk_head_dim
        self.qk_pos_emb_head_dim = qk_pos_emb_head_dim
        self.v_head_dim = v_head_dim

        self.is_hybrid_model = is_hybrid_model
        self.hybrid_attention_ratio = hybrid_attention_ratio

        self.mamba_expand = mamba_expand
        self.mamba_state_dim = mamba_state_dim
        self.mamba_head_dim = mamba_head_dim
        self.mamba_num_groups = mamba_num_groups
        self.mamba_d_conv = mamba_d_conv
        self.mamba_hidden_act = mamba_hidden_act

        self.normalization = normalization
        self.layernorm_epsilon = layernorm_epsilon

        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.residual_in_fp32 = residual_in_fp32
        self.bias_dropout_fusion = bias_dropout_fusion

        self.rope_type = rope_type
        self.rope_theta = rope_theta
        self.rotary_scaling_factor = rotary_scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim

        self.max_position_embeddings = max_position_embeddings
        self.use_cache = False  # force off
        self.torch_dtype = torch_dtype
        self.use_transformer_engine = use_transformer_engine

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


# =============================================================================
# Utils
# =============================================================================


def _params_dtype_from_config(config: ZebraLlamaConfig) -> torch.dtype:
    td = getattr(config, "torch_dtype", None)
    if td is torch.bfloat16 or td == "bfloat16":
        return torch.bfloat16
    if td is torch.float16 or td in ("float16", "fp16"):
        return torch.float16
    if td is torch.float32 or td == "float32":
        return torch.float32
    return torch.bfloat16


def _te_enabled(config: ZebraLlamaConfig) -> bool:
    return bool(_HAVE_TE and getattr(config, "use_transformer_engine", False))


def _filter_kwargs_for_init(cls, kwargs: dict) -> dict:
    """Filter kwargs to only those accepted by cls.__init__."""
    import inspect

    try:
        sig = inspect.signature(cls.__init__)
    except Exception:
        return kwargs
    allowed = set(sig.parameters.keys())
    # remove self
    allowed.discard("self")
    return {k: v for k, v in kwargs.items() if k in allowed}


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon  = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.variance_epsilon)
        return (self.weight.to(torch.float32) * x).to(orig_dtype)


def _get_norm_eps(norm: nn.Module, default: float = 1e-5) -> float:
    # Different norm impls use different attribute names.
    for k in ("variance_epsilon", "eps", "epsilon"):
        v = getattr(norm, k, None)
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass
    return float(default)


def build_norm(config: ZebraLlamaConfig, hidden_size: int) -> nn.Module:
    # Prefer Transformer Engine norms when explicitly enabled.
    if _te_enabled(config):
        eps = float(getattr(config, "layernorm_epsilon", 1e-5))
        if getattr(config, "normalization", "LayerNorm") == "RMSNorm":
            te_rms = getattr(te, "RMSNorm", None) if te is not None else None
            if te_rms is not None:
                return te_rms(**_filter_kwargs_for_init(te_rms, {"hidden_size": hidden_size, "eps": eps, "epsilon": eps}))
            # TE may not ship RMSNorm; fall back below.
        te_ln = getattr(te, "LayerNorm", None) if te is not None else None
        if te_ln is not None:
            return te_ln(**_filter_kwargs_for_init(te_ln, {"hidden_size": hidden_size, "eps": eps, "epsilon": eps}))

    if getattr(config, "normalization", "LayerNorm") == "RMSNorm":
        return RMSNorm(hidden_size, eps=getattr(config, "layernorm_epsilon", 1e-5))
    return nn.LayerNorm(hidden_size, eps=getattr(config, "layernorm_epsilon", 1e-5), elementwise_affine=True)


def build_linear(config: ZebraLlamaConfig, in_features: int, out_features: int, bias: bool = False):
    """
    Build a Linear module, optionally using Transformer Engine.
    Keeps parameter names compatible with `nn.Linear` so checkpoint loading works.
    """
    if _te_enabled(config):
        te_linear = getattr(te, "Linear", None) if te is not None else None
        if te_linear is not None:
            kwargs = {
                "in_features": in_features,
                "out_features": out_features,
                "bias": bias,
                # Try to align TE param dtype with config.
                "params_dtype": _params_dtype_from_config(config),
            }
            return te_linear(**_filter_kwargs_for_init(te_linear, kwargs))
    return nn.Linear(in_features, out_features, bias=bias)


def allocate_hybrid_layers(num_layers: int, attention_ratio: float) -> List[str]:
    """Return a list of 'mamba' / 'attention' with roughly the desired attention ratio."""
    num_attn = max(1, int(round(num_layers // 2 * attention_ratio)))
    spacing = max(1, int(round(num_layers // 2 / num_attn)))
    types: List[str] = []
    attn_used = 0
    for i in range(num_layers // 2):
        if (i % spacing == 0) and (attn_used < num_attn):
            types.append("attention")
            types.append("mlp")
            attn_used += 1
        else:
            types.append("mamba")
            types.append("mlp")
    return types


# =============================================================================
# YaRN RoPE (ported from Megatron)
# =============================================================================


def _yarn_find_correction_dim(
    num_rotations: float, dim: int, rotary_base: float = 10000.0, max_position_embeddings: int = 2048
) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(rotary_base)
    )


def _yarn_find_correction_range(
    low_rot: float,
    high_rot: float,
    dim: int,
    rotary_base: float = 10000.0,
    max_position_embeddings: int = 2048,
    round_to_int: bool = True,
) -> Tuple[int, int]:
    low = _yarn_find_correction_dim(low_rot, dim, rotary_base, max_position_embeddings)
    high = _yarn_find_correction_dim(high_rot, dim, rotary_base, max_position_embeddings)
    if round_to_int:
        low = math.floor(low)
        high = math.ceil(high)
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp_mask(min_v: float, max_v: float, dim: int) -> torch.Tensor:
    if min_v == max_v:
        max_v += 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32) - min_v) / (max_v - min_v)
    return torch.clamp(linear_func, 0.0, 1.0)


def _yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    if scale <= 1.0:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _yarn_get_concentration_factor(scaling_factor: float, mscale: float, mscale_all_dim: float) -> float:
    return float(_yarn_get_mscale(scaling_factor, mscale) / _yarn_get_mscale(scaling_factor, mscale_all_dim))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = float(base)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache = {}

    def forward(self, x: torch.Tensor, seq_len: int, offset: int = 0):
        key = (seq_len, x.device.type, x.device.index, x.dtype, offset)
        if key in self._cache:
            return self._cache[key]
        inv_freq = self.inv_freq.to(device=x.device)
        t = torch.arange(seq_len, device=x.device, dtype=inv_freq.dtype) + offset
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=x.dtype)
        sin = emb.sin().to(dtype=x.dtype)
        self._cache[key] = (cos, sin)
        return cos, sin


class YarnRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        rotary_base: float,
        scaling_factor: float,
        original_max_position_embeddings: int,
        beta_fast: float,
        beta_slow: float,
        mscale: float,
        mscale_all_dim: float,
        correction_range_round_to_int: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.rotary_base = float(rotary_base)
        self.scaling_factor = float(scaling_factor)
        self.original_max_position_embeddings = int(original_max_position_embeddings)
        self.beta_fast = float(beta_fast)
        self.beta_slow = float(beta_slow)
        self.mscale = float(mscale)
        self.mscale_all_dim = float(mscale_all_dim)
        self.correction_range_round_to_int = bool(correction_range_round_to_int)

        inv_freq_extra = 1.0 / (
            self.rotary_base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        inv_freq_inter = 1.0 / (
            self.scaling_factor
            * self.rotary_base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        self.register_buffer("inv_freq_extra", inv_freq_extra, persistent=False)
        self.register_buffer("inv_freq_inter", inv_freq_inter, persistent=False)
        self._cache = {}

    def _inv_freq(self, device: torch.device) -> torch.Tensor:
        low, high = _yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.dim,
            self.rotary_base,
            self.original_max_position_embeddings,
            self.correction_range_round_to_int,
        )
        inv_freq_mask = 1.0 - _yarn_linear_ramp_mask(low, high, self.dim // 2).to(device=device, dtype=torch.float32)
        inv_freq = self.inv_freq_inter.to(device=device) * (1.0 - inv_freq_mask) + self.inv_freq_extra.to(device=device) * inv_freq_mask
        return inv_freq

    def forward(self, x: torch.Tensor, seq_len: int, offset: int = 0):
        key = (seq_len, x.device.type, x.device.index, x.dtype, offset)
        if key in self._cache:
            return self._cache[key]

        inv_freq = self._inv_freq(device=x.device)
        t = torch.arange(seq_len, device=x.device, dtype=inv_freq.dtype) + offset
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        mscale = _yarn_get_concentration_factor(self.scaling_factor, self.mscale, self.mscale_all_dim)
        cos = (emb.cos() * mscale).to(dtype=x.dtype)
        sin = (emb.sin() * mscale).to(dtype=x.dtype)
        self._cache[key] = (cos, sin)
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _mla_rope_reorder(t: torch.Tensor) -> torch.Tensor:
    """
    Match Megatron's MLA RoPE path (`multi_latent_attention=True`) where features are
    reordered as [even dims..., odd dims...] before applying rotate-half.
    """
    return torch.cat((t[..., 0::2], t[..., 1::2]), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    multi_latent_attention: bool = False,
):
    # q,k: [B, H, L, D]; cos/sin: [S, D]
    cos = cos[position_ids].unsqueeze(1)  # [B,1,L,D]
    sin = sin[position_ids].unsqueeze(1)

    if multi_latent_attention:
        q = _mla_rope_reorder(q)
        k = _mla_rope_reorder(k)

    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


# =============================================================================
# MLA Attention
# =============================================================================


class IdentityNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class MLAAttention(nn.Module):
    def __init__(self, config: ZebraLlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_head_dim = config.qk_head_dim
        self.qk_pos_emb_head_dim = config.qk_pos_emb_head_dim
        self.v_head_dim = config.v_head_dim
        self.q_head_dim = self.qk_head_dim + self.qk_pos_emb_head_dim

        # Q LoRA
        self.linear_q_down_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
        self.linear_q_up_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)

        # KV LoRA (+ positional part)
        self.linear_kv_down_proj = nn.Linear(
            self.hidden_size, self.kv_lora_rank + self.qk_pos_emb_head_dim, bias=False
        )
        self.linear_kv_up_proj = nn.Linear(
            self.kv_lora_rank, self.num_heads * (self.qk_head_dim + self.v_head_dim), bias=False
        )

        # In Megatron spec for Zebra-Llama, these are IdentityOp (LN fused in TE up-proj).
        self.q_layernorm = RMSNorm(self.q_lora_rank, eps=config.layernorm_epsilon)
        self.kv_layernorm = RMSNorm(self.kv_lora_rank, eps=config.layernorm_epsilon)

        self.linear_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False)

        if getattr(config, "rope_type", "rope") == "yarn":
            self.rotary_emb = YarnRotaryEmbedding(
                dim=self.qk_pos_emb_head_dim,
                rotary_base=config.rope_theta,
                scaling_factor=config.rotary_scaling_factor,
                original_max_position_embeddings=config.original_max_position_embeddings,
                beta_fast=config.beta_fast,
                beta_slow=config.beta_slow,
                mscale=config.mscale,
                mscale_all_dim=config.mscale_all_dim,
            )
        else:
            self.rotary_emb = RotaryEmbedding(dim=self.qk_pos_emb_head_dim, base=config.rope_theta)
        
        # import pdb; pdb.set_trace()
        mscale = _yarn_get_mscale(config.rotary_scaling_factor, config.mscale_all_dim)
        self.softmax_scale = mscale * mscale / math.sqrt(self.q_head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,                # [B, L, D]
        attention_mask: Optional[torch.Tensor] = None,  # additive mask [B,1,1,S] with 0 or -inf
        position_ids: Optional[torch.Tensor] = None,    # [B, L]
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        # no cache support
        _ = past_key_value
        use_cache = False
        
        bsz, q_len, _ = hidden_states.shape
        if position_ids is None:
            position_ids = torch.arange(q_len, device=hidden_states.device, dtype=torch.long)[None, :]

        # Down projections
        q_comp = self.q_layernorm(self.linear_q_down_proj(hidden_states))
        kv_combined = self.linear_kv_down_proj(hidden_states)
        kv_comp = self.kv_layernorm(kv_combined[..., : self.kv_lora_rank])
        k_pos_emb = kv_combined[..., self.kv_lora_rank :]  # [B,L,pos]
        
        # Up projections
        q = self.linear_q_up_proj(q_comp).view(bsz, q_len, self.num_heads, self.q_head_dim)
        kv = self.linear_kv_up_proj(kv_comp).view(bsz, q_len, self.num_heads, self.qk_head_dim + self.v_head_dim)

        # Split
        q_no_pe = q[..., : self.qk_head_dim]
        q_pos = q[..., self.qk_head_dim :]
        k_no_pe = kv[..., : self.qk_head_dim]
        v = kv[..., self.qk_head_dim :]

        # RoPE on positional components only
        kv_seq_len = q_len
        cos, sin = self.rotary_emb(q_pos, seq_len=kv_seq_len)

        k_pos = k_pos_emb.unsqueeze(2).expand(bsz, q_len, self.num_heads, self.qk_pos_emb_head_dim)
        q_pos = q_pos.transpose(1, 2)  # [B,H,L,pos]
        k_pos = k_pos.transpose(1, 2)
        q_pos, k_pos = apply_rotary_pos_emb(
            q_pos, k_pos, cos, sin, position_ids, multi_latent_attention=True
        )

        # Concatenate and transpose to [B,H,L,D]
        q = torch.cat([q_no_pe.transpose(1, 2), q_pos], dim=-1)
        k = torch.cat([k_no_pe.transpose(1, 2), k_pos], dim=-1)
        v = v.transpose(1, 2)

        # Attention
        if output_attentions:
            attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.q_head_dim)
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype=q.dtype)
            attn_out = torch.matmul(attn_weights, v)
        else:
            dropout_p = float(getattr(self.config, "attention_dropout", 0.0)) if self.training else 0.0

            can_use_flash = (
                _HAVE_FLASH_ATTN
                and q.is_cuda
                and q.dtype in (torch.float16, torch.bfloat16)
                and self.q_head_dim == self.v_head_dim
            )
            if can_use_flash:
                # flash_attn expects [B,L,H,D]
                attn_out = flash_attn_func(
                    q.transpose(1, 2).contiguous(),
                    k.transpose(1, 2).contiguous(),
                    v.transpose(1, 2).contiguous(),
                    dropout_p=dropout_p,
                    softmax_scale=self.softmax_scale,
                    causal=True,
                ).transpose(1, 2).contiguous()
            else:
                sdpa_mask = None
                if attention_mask is not None:
                    sdpa_mask = attention_mask < 0  # boolean mask
                try:
                    attn_out = F.scaled_dot_product_attention(
                        q, k, v, attn_mask=sdpa_mask, dropout_p=dropout_p, is_causal=True
                    )
                except TypeError:
                    attn_out = F.scaled_dot_product_attention(
                        q, k, v, attn_mask=sdpa_mask, dropout_p=dropout_p, is_causal=False
                    )
            attn_weights = None

        # Project back to hidden size
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, self.num_heads * self.v_head_dim)
        attn_out = self.linear_proj(attn_out)

        return attn_out, attn_weights, None


class MLAAttentionLayer(nn.Module):
    """
    Legacy wrapper kept for compatibility.

    Our primary implementation uses `ZebraLlamaDecoderLayer` directly.
    This wrapper is not used by the current model, but keeping it as a valid class
    avoids syntax/import issues if referenced elsewhere.
    """

    def __init__(self, config: ZebraLlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.self_attention = MLAAttention(config, layer_idx=layer_idx or 0)
        self.hidden_dropout = float(getattr(config, "hidden_dropout", 0.0))
        self.dropout = nn.Dropout(self.hidden_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        # No KV cache support for now.
        _ = past_key_value
        use_cache = False

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, _ = self.self_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=output_attentions,
            use_cache=False,
            **kwargs,
        )
        hidden_states = self.dropout(hidden_states) + residual
        return hidden_states, self_attn_weights

# =============================================================================
# Simplified Mamba
# =============================================================================


def _bias_dropout_add(x_with_bias, residual: torch.Tensor, prob: float, training: bool) -> torch.Tensor:
    x, bias = x_with_bias
    residual = residual if residual.dtype == x.dtype else residual.to(x.dtype)
    if bias is not None:
        x = x + bias
    x = F.dropout(x, p=prob, training=training)
    return residual + x


class RMSNormGated(torch.nn.Module):
    def __init__(self, hidden_size, group_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.group_size = group_size

    # jan28b version
    def forward(self, hidden_states, gate=None):
        return rmsnorm_fn(x=hidden_states,
            weight=self.weight,
            bias=None, # No bias
            z=gate,
            eps=self.variance_epsilon,
            group_size=self.group_size,
            norm_before_gate=False
        )

class MambaMixer(nn.Module):
    """
    Megatron-shaped but simplified Mamba mixer.
    Produces the same logical splits as Megatron (z | x | B | C | dt),
    runs a depthwise causal conv on (x|B|C), and then applies a cheap per-head decay.
    """

    def __init__(self, config: ZebraLlamaConfig):
        super().__init__()
        self.config = config

        self.d_model = int(config.hidden_size)
        self.expand = int(getattr(config, "mamba_expand", 1))
        self.d_inner = int(self.expand * self.d_model)
        self.d_state = int(getattr(config, "mamba_state_dim", 64))
        self.headdim = int(getattr(config, "mamba_head_dim", 64))
        self.ngroups = int(getattr(config, "mamba_num_groups", 8))
        self.d_conv = int(getattr(config, "mamba_d_conv", 4))
        self.norm = RMSNormGated(
            self.d_inner,
            eps=config.layernorm_epsilon,
            group_size=self.d_inner // self.ngroups
        )
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim

        self.in_proj = nn.Linear(
            self.d_model,
            2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads,
            bias=False,
        )

        self.conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=self.d_conv,
            groups=self.conv_dim,
            padding=self.d_conv - 1,
            bias=True,
        )
        self.activation = config.mamba_hidden_act
        self.act = ACT2FN[config.mamba_hidden_act]

        # import pdb; pdb.set_trace()
        self.dt_bias = nn.Parameter(torch.ones(self.nheads))
        A = torch.arange(1, self.nheads + 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, hidden_states: torch.Tensor, inference_context=None, *, inference_params=None):

        bsz, seqlen, _ = hidden_states.shape

        zxBCdt = self.in_proj(hidden_states)
        
        # Get A from log scale
        A = -torch.exp(self.A_log.float())

        # 2-4. Fused kernel for conv1d, SSM, and the final projection
        out = mamba_split_conv1d_scan_combined(
            zxBCdt,
            self.conv1d.weight.squeeze(1),
            self.conv1d.bias,
            self.dt_bias,
            A,
            D=self.D,
            chunk_size=128,
            seq_idx=None,  # was seq_idx
            activation=self.activation,
            rmsnorm_weight=self.norm.weight,
            rmsnorm_eps=self.norm.variance_epsilon,
            outproj_weight=self.out_proj.weight,
            outproj_bias=self.out_proj.bias,
            headdim=self.headdim,
            ngroups=self.ngroups,
            norm_before_gate=False,
            return_final_states=False,
        )
        
        return out, None


class MambaLayer(nn.Module):
    def __init__(self, config: ZebraLlamaConfig):
        super().__init__()
        self.config = config
        self.residual_in_fp32 = bool(getattr(config, "residual_in_fp32", False))
        self.hidden_dropout = float(getattr(config, "hidden_dropout", 0.0))
        self.bias_dropout_fusion = bool(getattr(config, "bias_dropout_fusion", True))
        self.dropout = nn.Dropout(self.hidden_dropout)
        self.norm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.mixer = MambaMixer(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, inference_context=None, **kwargs):
        _ = attention_mask
        residual = hidden_states

        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        hidden_states, _ = self.mixer(hidden_states, inference_context=inference_context)
        hidden_states = self.dropout(hidden_states) + residual

        return hidden_states, None


# =============================================================================
# MLP
# =============================================================================


class SwiGLUMLP(nn.Module):
    def __init__(self, config: ZebraLlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.linear_fc1 = nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=False)
        self.linear_fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc1 = self.linear_fc1(x)
        x_glu = x_fc1[:, :, : self.intermediate_size]
        x_lin = x_fc1[:, :, self.intermediate_size :]
        return self.linear_fc2(F.silu(x_glu) * x_lin)


class MLPLayer(nn.Module):
    def __init__(self, config: ZebraLlamaConfig):
        super().__init__()
        self.config = config
        self.mlp = SwiGLUMLP(config)
        self.dropout = nn.Dropout(float(getattr(config, "hidden_dropout", 0.0)))
        self.pre_mlp_layernorm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, inference_context=None, **kwargs) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.pre_mlp_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.dropout(hidden_states) + residual
        return hidden_states, None


# =============================================================================
# Base model
# =============================================================================


class ZebraLlamaModel(PreTrainedModel):
    config_class = ZebraLlamaConfig

    def __init__(self, config: ZebraLlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        layer_types = allocate_hybrid_layers(config.num_hidden_layers, config.hybrid_attention_ratio)
        self.layers = nn.ModuleList()
        for i, t in enumerate(layer_types):
            if t == "mamba":
                self.layers.append(MambaLayer(config))
            elif t == "attention":
                self.layers.append(MLAAttentionLayer(config, i))
            elif t == "mlp":
                self.layers.append(MLPLayer(config))
        self.norm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # No cache support
        _ = past_key_values
        use_cache = False

        output_attentions = output_attentions if output_attentions is not None else bool(self.config.output_attentions)
        output_hidden_states = output_hidden_states if output_hidden_states is not None else bool(self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else bool(self.config.use_return_dict)

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Specify only one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        bsz, seq_len, _ = inputs_embeds.shape
        hidden_states = inputs_embeds

        # attention_mask: [B,L] -> additive [B,1,1,L]
        attn_mask = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attn_mask = attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
                attn_mask = (1.0 - attn_mask) * torch.finfo(hidden_states.dtype).min
            else:
                attn_mask = attention_mask

        if position_ids is None:
            device = hidden_states.device
            position_ids = torch.arange(seq_len, device=device, dtype=torch.long)[None, :]

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            hidden_states, attn = layer(
                hidden_states,
                attention_mask=attn_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
            )
            if output_attentions:
                all_attns += (attn,)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, None, all_hidden_states, all_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=all_attns,
        )


# =============================================================================
# Causal LM
# =============================================================================


class ZebraLlamaForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = ZebraLlamaConfig

    def __init__(self, config: ZebraLlamaConfig):
        super().__init__(config)
        self.model = ZebraLlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self._tied_weights_keys = ["lm_head.weight"]
        self.post_init()
        if getattr(config, "tie_word_embeddings", False):
            self.tie_weights()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states).float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1).to(shift_logits.device))

        if not return_dict:
            out = (logits,) + outputs[1:]
            return (loss,) + out if loss is not None else out

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, inputs_embeds=None, **kwargs):
        # No KV cache: do NOT slice to last token. Always recompute from scratch.
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)

        if inputs_embeds is not None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "use_cache": False,
                "past_key_values": None,
            }
        )
        return model_inputs


# =============================================================================
# Optional local registration (helps when importing this module directly)
# =============================================================================


from transformers import AutoConfig, AutoModel, AutoModelForCausalLM  # noqa: E402

AutoConfig.register("zebra_llama", ZebraLlamaConfig)
AutoModel.register(ZebraLlamaConfig, ZebraLlamaModel)
AutoModelForCausalLM.register(ZebraLlamaConfig, ZebraLlamaForCausalLM)

