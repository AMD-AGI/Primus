###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Backend-neutral model architecture records for the projection tool.

TorchTitan names a model ``(model.name, model.flavor)`` and keeps the shape in a
Python dataclass; MaxText names it ``model_name`` and keeps the shape in a YAML
file.  Neither spelling reaches the Primus experiment YAML -- an experiment says
``flavor: "8B"`` or ``model_name: "llama3-8b"`` and nothing else -- so a config
adapter has to recover the shape before it can describe the workload to the
projection.

The backend is the authority on its own flavors, so the adapters read it first
(:mod:`primus.core.projection.frameworks.torchtitan` imports the flavor table,
the MaxText adapter reads the model YAML).  This module is what answers when the
backend is not installed, which is the normal case for projection: sizing a
cluster is supposed to need neither a GPU nor a training checkout.  A ``ModelSpec``
is therefore a *transcription* of the backend's own definition, and
``tests/unit_tests/core/projection/test_projection_model_specs.py`` re-checks it
against the backend whenever the backend happens to be present, so a spec that
drifts from upstream fails a test rather than quietly mis-sizing a cluster.

Specs are shared across backends on purpose: a TorchTitan ``llama3/8B`` and a
MaxText ``llama3-8b`` are the same 32 layers of the same GEMMs, and the
projection has no reason to hold two opinions about that.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


def llama_ffn_hidden_size(dim: int, multiple_of: int, ffn_dim_multiplier: Optional[float] = None) -> int:
    """Llama-family SwiGLU hidden width, in the backends' own integer order.

    Every Llama derivative (and TorchTitan's port of it) derives the FFN width
    from ``dim`` rather than storing it, and the truncation order matters: the
    ``int()`` before the multiplier is what turns Llama 3 8B's 4096 into 14336
    rather than 14338.
    """
    hidden = int(2 * (4 * dim) / 3)
    if ffn_dim_multiplier is not None:
        hidden = int(ffn_dim_multiplier * hidden)
    return multiple_of * ((hidden + multiple_of - 1) // multiple_of)


def llama4_moe_ffn_hidden_size(
    dim: int,
    multiple_of: int,
    ffn_dim_multiplier: Optional[float],
    top_k: int,
    num_shared_experts: int,
    auto_scale_hidden_dim: bool = True,
) -> int:
    """Per-expert FFN width for Llama 4, which divides the dense width by the
    number of experts each token actually runs through."""
    hidden = int(2 * (4 * dim) / 3)
    if ffn_dim_multiplier is not None:
        hidden = int(ffn_dim_multiplier * hidden)
    if auto_scale_hidden_dim:
        hidden = int(hidden / (top_k + num_shared_experts))
    return hidden + (-hidden % multiple_of)


@dataclass(frozen=True)
class ModelSpec:
    """One model's architecture, in the projection's vocabulary."""

    num_layers: int
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int
    num_query_groups: int
    kv_channels: int
    vocab_size: int
    swiglu: bool = True
    tie_embeddings: bool = False
    qk_layernorm: bool = False

    # Mixture of experts.  ``num_experts == 0`` means the model is dense and the
    # MoE fields below are ignored.
    num_experts: int = 0
    moe_ffn_hidden_size: int = 0
    moe_router_topk: int = 0
    num_shared_experts: int = 0
    # Leading dense layers before the MoE stack starts (DeepSeek's warm-up
    # layers); MoE then applies to every ``moe_layer_step``-th remaining layer.
    num_dense_layers: int = 0
    moe_layer_step: int = 1

    # Multi-head latent attention (DeepSeek V2/V3).
    multi_latent_attention: bool = False
    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    qk_head_dim: int = 0
    qk_pos_emb_head_dim: int = 0
    v_head_dim: int = 0

    attn_sliding_window: int = 0

    @property
    def group_query_attention(self) -> bool:
        return self.num_query_groups != self.num_attention_heads

    @property
    def moe_shared_expert_intermediate_size(self) -> int:
        """Total shared-expert FFN width, which is what Megatron's field means."""
        return self.num_shared_experts * self.moe_ffn_hidden_size

    def moe_layer_pattern(self) -> List[int]:
        """Per-layer 0/1 MoE flags for the whole stack."""
        if not self.num_experts:
            return [0] * self.num_layers
        pattern = []
        for i in range(self.num_layers):
            if i < self.num_dense_layers:
                pattern.append(0)
            else:
                pattern.append(1 if (i - self.num_dense_layers) % self.moe_layer_step == 0 else 0)
        return pattern


def _llama3(dim, n_layers, n_heads, n_kv_heads, multiple_of, ffn_mult, vocab=128256) -> ModelSpec:
    return ModelSpec(
        num_layers=n_layers,
        hidden_size=dim,
        ffn_hidden_size=llama_ffn_hidden_size(dim, multiple_of, ffn_mult),
        num_attention_heads=n_heads,
        num_query_groups=n_kv_heads,
        kv_channels=dim // n_heads,
        vocab_size=vocab,
    )


def _llama4(n_layers, num_experts, moe_layer_step) -> ModelSpec:
    dim, n_heads, multiple_of, ffn_mult = 5120, 40, 2048, 1.2
    # TorchTitan leaves MoEArgs.top_k / num_shared_experts at their defaults for
    # Llama 4, which is also what the architecture uses: top-1 routing plus one
    # shared expert.
    top_k, num_shared = 1, 1
    return ModelSpec(
        num_layers=n_layers,
        hidden_size=dim,
        ffn_hidden_size=llama_ffn_hidden_size(dim, multiple_of, ffn_mult),
        num_attention_heads=n_heads,
        num_query_groups=8,
        kv_channels=dim // n_heads,
        vocab_size=202048,
        num_experts=num_experts,
        moe_ffn_hidden_size=llama4_moe_ffn_hidden_size(dim, multiple_of, ffn_mult, top_k, num_shared),
        moe_router_topk=top_k,
        num_shared_experts=num_shared,
        moe_layer_step=moe_layer_step,
    )


def _deepseek(
    n_layers,
    dim,
    inter_dim,
    moe_inter_dim,
    n_heads,
    n_dense_layers,
    num_experts,
    top_k,
    num_shared,
    q_lora_rank,
    vocab,
) -> ModelSpec:
    return ModelSpec(
        num_layers=n_layers,
        hidden_size=dim,
        ffn_hidden_size=inter_dim,
        num_attention_heads=n_heads,
        num_query_groups=n_heads,
        # MLA stores the compressed latent, so the projection reads the
        # non-positional query width here (matching the Megatron DeepSeek
        # presets) and the rope/value widths from their own fields.
        kv_channels=128,
        vocab_size=vocab,
        qk_layernorm=True,
        num_experts=num_experts,
        moe_ffn_hidden_size=moe_inter_dim,
        moe_router_topk=top_k,
        num_shared_experts=num_shared,
        num_dense_layers=n_dense_layers,
        multi_latent_attention=True,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=512,
        qk_head_dim=128,
        qk_pos_emb_head_dim=64,
        v_head_dim=128,
    )


def _qwen3(dim, n_layers, n_heads, n_kv_heads, hidden_dim, tie=False) -> ModelSpec:
    return ModelSpec(
        num_layers=n_layers,
        hidden_size=dim,
        ffn_hidden_size=hidden_dim,
        num_attention_heads=n_heads,
        num_query_groups=n_kv_heads,
        kv_channels=128,
        vocab_size=151936,
        tie_embeddings=tie,
        qk_layernorm=True,
    )


def _gpt_oss(n_layers, num_experts) -> ModelSpec:
    moe_inter_dim = 2880
    return ModelSpec(
        num_layers=n_layers,
        hidden_size=2880,
        # GPT-OSS is MoE in every layer; the dense width is never reached, so it
        # mirrors the expert width rather than inventing a number.
        ffn_hidden_size=moe_inter_dim,
        num_attention_heads=64,
        num_query_groups=8,
        kv_channels=64,
        vocab_size=201088,
        num_experts=num_experts,
        moe_ffn_hidden_size=moe_inter_dim,
        moe_router_topk=4,
        num_shared_experts=0,
        attn_sliding_window=128,
    )


def _mixtral(dim, n_layers, n_heads, ffn, vocab) -> ModelSpec:
    return ModelSpec(
        num_layers=n_layers,
        hidden_size=dim,
        ffn_hidden_size=ffn,
        num_attention_heads=n_heads,
        num_query_groups=8,
        kv_channels=128,
        vocab_size=vocab,
        num_experts=8,
        moe_ffn_hidden_size=ffn,
        moe_router_topk=2,
    )


BUILTIN_MODEL_SPECS: Dict[str, ModelSpec] = {
    # Llama 2 / 3 / 4
    "llama2-7b": ModelSpec(32, 4096, 11008, 32, 32, 128, 32000),
    "llama2-70b": ModelSpec(80, 8192, 28672, 64, 8, 128, 32000),
    "llama3-8b": _llama3(4096, 32, 32, 8, 1024, 1.3),
    "llama3-70b": _llama3(8192, 80, 64, 8, 4096, 1.3),
    "llama3.1-405b": _llama3(16384, 126, 128, 8, 4096, 1.2),
    "llama3.2-1b": ModelSpec(16, 2048, 8192, 32, 8, 64, 128256, tie_embeddings=True),
    "llama4-17bx16e": _llama4(48, 16, moe_layer_step=1),
    "llama4-17bx128e": _llama4(48, 128, moe_layer_step=2),
    # DeepSeek (TorchTitan files V2 under its deepseek_v3 spec; the shapes are
    # the published V2-Lite / V2 / V3 architectures)
    "deepseek-v2-16b": _deepseek(27, 2048, 10944, 1408, 16, 1, 64, 6, 2, 0, 102400),
    "deepseek-v2-236b": _deepseek(60, 5120, 12288, 1536, 128, 1, 160, 6, 2, 1536, 102400),
    "deepseek-v3-671b": _deepseek(61, 7168, 18432, 2048, 128, 3, 256, 8, 1, 1536, 129280),
    # Qwen 3
    "qwen3-0.6b": _qwen3(1024, 28, 16, 8, 3072, tie=True),
    "qwen3-1.7b": _qwen3(2048, 28, 16, 8, 6144, tie=True),
    "qwen3-4b": _qwen3(2560, 36, 32, 8, 9728, tie=True),
    "qwen3-8b": _qwen3(4096, 36, 32, 8, 12288),
    "qwen3-14b": _qwen3(5120, 40, 40, 8, 17408),
    "qwen3-32b": _qwen3(5120, 64, 64, 8, 25600),
    "qwen3-30b-a3b": ModelSpec(
        num_layers=48,
        hidden_size=2048,
        ffn_hidden_size=6144,
        num_attention_heads=32,
        num_query_groups=4,
        kv_channels=128,
        vocab_size=151936,
        qk_layernorm=True,
        num_experts=128,
        moe_ffn_hidden_size=768,
        moe_router_topk=8,
    ),
    "qwen3-235b-a22b": ModelSpec(
        num_layers=94,
        hidden_size=4096,
        ffn_hidden_size=12288,
        num_attention_heads=64,
        num_query_groups=4,
        kv_channels=128,
        vocab_size=151936,
        qk_layernorm=True,
        num_experts=128,
        moe_ffn_hidden_size=1536,
        moe_router_topk=8,
    ),
    # GPT-OSS
    "gpt-oss-20b": _gpt_oss(24, 32),
    "gpt-oss-120b": _gpt_oss(36, 128),
    # Mixtral / Grok
    "mixtral-8x7b": _mixtral(4096, 32, 32, 14336, 32000),
    "mixtral-8x22b": _mixtral(6144, 56, 48, 16384, 32768),
    "grok1": _mixtral(6144, 64, 48, 32768, 131072),
    # Gemma 4 (GeGLU is still a gated MLP, so it keeps the three-matrix shape)
    "gemma4-26b": ModelSpec(
        num_layers=30,
        hidden_size=2816,
        ffn_hidden_size=2112,
        num_attention_heads=16,
        num_query_groups=8,
        kv_channels=256,
        vocab_size=262144,
        tie_embeddings=True,
        num_experts=128,
        moe_ffn_hidden_size=704,
        moe_router_topk=8,
        num_shared_experts=1,
        attn_sliding_window=1024,
    ),
    "gemma4-31b": ModelSpec(
        num_layers=60,
        hidden_size=5376,
        ffn_hidden_size=21504,
        num_attention_heads=32,
        num_query_groups=16,
        kv_channels=256,
        vocab_size=262144,
        tie_embeddings=True,
        attn_sliding_window=1024,
    ),
}


# ``(model.name, model.flavor)`` as TorchTitan spells them, lowercased.
TORCHTITAN_FLAVOR_ALIASES: Dict[Tuple[str, str], str] = {
    ("llama3", "1b"): "llama3.2-1b",
    ("llama3", "8b"): "llama3-8b",
    ("llama3", "70b"): "llama3-70b",
    ("llama3", "405b"): "llama3.1-405b",
    ("llama4", "17bx16e"): "llama4-17bx16e",
    ("llama4", "17bx128e"): "llama4-17bx128e",
    ("deepseek_v3", "16b"): "deepseek-v2-16b",
    ("deepseek_v3", "236b"): "deepseek-v2-236b",
    ("deepseek_v3", "671b"): "deepseek-v3-671b",
    ("gpt_oss", "20b"): "gpt-oss-20b",
    ("gpt_oss", "120b"): "gpt-oss-120b",
    ("qwen3", "0.6b"): "qwen3-0.6b",
    ("qwen3", "1.7b"): "qwen3-1.7b",
    ("qwen3", "4b"): "qwen3-4b",
    ("qwen3", "8b"): "qwen3-8b",
    ("qwen3", "14b"): "qwen3-14b",
    ("qwen3", "32b"): "qwen3-32b",
    ("qwen3", "30b-a3b"): "qwen3-30b-a3b",
    ("qwen3", "235b-a22b"): "qwen3-235b-a22b",
}


# ``model_name`` as MaxText spells it, lowercased.
MAXTEXT_MODEL_ALIASES: Dict[str, str] = {
    "llama2-7b": "llama2-7b",
    "llama2-70b": "llama2-70b",
    "llama3-8b": "llama3-8b",
    "llama3.1-8b": "llama3-8b",
    "llama3-70b": "llama3-70b",
    "llama3.1-70b": "llama3-70b",
    "llama3.3-70b": "llama3-70b",
    "llama3.1-405b": "llama3.1-405b",
    "llama4-17b-16e": "llama4-17bx16e",
    "llama4-17b-128e": "llama4-17bx128e",
    "mixtral-8x7b": "mixtral-8x7b",
    "mixtral-8x22b": "mixtral-8x22b",
    "deepseek2-16b": "deepseek-v2-16b",
    "deepseek2-236b": "deepseek-v2-236b",
    "deepseek3-671b": "deepseek-v3-671b",
    "qwen3-14b": "qwen3-14b",
    "qwen3-30b-a3b": "qwen3-30b-a3b",
    "qwen3-235b-a22b": "qwen3-235b-a22b",
    "gpt-oss-20b": "gpt-oss-20b",
    "gpt-oss-120b": "gpt-oss-120b",
    "gemma4-26b": "gemma4-26b",
    "gemma4-31b": "gemma4-31b",
    # MaxText ships no config for Grok-1, so the Primus preset spells the
    # architecture out; the alias is here for an experiment that only names it.
    "grok-1": "grok1",
    "grok1": "grok1",
}


def get_builtin_spec(canonical_name: str) -> Optional[ModelSpec]:
    """Return the transcribed spec for *canonical_name*, or ``None``."""
    if not canonical_name:
        return None
    return BUILTIN_MODEL_SPECS.get(str(canonical_name).lower().strip())


def torchtitan_builtin_spec(name: str, flavor: str) -> Optional[ModelSpec]:
    """Return the spec for a TorchTitan ``(model.name, model.flavor)`` pair."""
    key = (str(name or "").lower().strip(), str(flavor or "").lower().strip())
    return get_builtin_spec(TORCHTITAN_FLAVOR_ALIASES.get(key, ""))


def maxtext_builtin_spec(model_name: str) -> Optional[ModelSpec]:
    """Return the spec for a MaxText ``model_name``."""
    key = str(model_name or "").lower().strip()
    return get_builtin_spec(MAXTEXT_MODEL_ALIASES.get(key, ""))


def spec_to_projection_fields(spec: ModelSpec) -> Dict[str, object]:
    """Flatten a spec into the argument names the projection config reads.

    ``num_experts`` is ``None`` rather than ``0`` for a dense model because that
    is the value the Megatron path produces, and downstream code distinguishes
    "no MoE" from "a one-expert MoE" by exactly that.
    """
    fields: Dict[str, object] = {
        "num_layers": spec.num_layers,
        "hidden_size": spec.hidden_size,
        "ffn_hidden_size": spec.ffn_hidden_size,
        "num_attention_heads": spec.num_attention_heads,
        "num_query_groups": spec.num_query_groups,
        "group_query_attention": spec.group_query_attention,
        "kv_channels": spec.kv_channels,
        "padded_vocab_size": spec.vocab_size,
        "swiglu": spec.swiglu,
        "qk_layernorm": spec.qk_layernorm,
        "untie_embeddings_and_output_weights": not spec.tie_embeddings,
        "multi_latent_attention": spec.multi_latent_attention,
        "q_lora_rank": spec.q_lora_rank,
        "kv_lora_rank": spec.kv_lora_rank,
        "qk_head_dim": spec.qk_head_dim,
        "qk_pos_emb_head_dim": spec.qk_pos_emb_head_dim,
        "v_head_dim": spec.v_head_dim,
        "attn_sliding_window": spec.attn_sliding_window,
        "num_experts": spec.num_experts or None,
        "moe_ffn_hidden_size": spec.moe_ffn_hidden_size,
        "moe_router_topk": spec.moe_router_topk,
        "moe_shared_expert_intermediate_size": spec.moe_shared_expert_intermediate_size,
        "moe_layer_freq": spec.moe_layer_pattern(),
    }
    return fields
