"""Subprocess body for test_granite_cpu_parity.py (CPU-only monkeypatches stay in this process).

Prints one ``GRANITE_PARITY_RESULT=<json>`` line; the parent test checks the tolerances."""

import contextlib
import json
import os
import socket
from types import SimpleNamespace

import torch

# GPTModel's default attention_backend (auto) asserts these are unset; some images set them.
# The model below uses Megatron's local attention, so they have no effect here.
for _var in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
    os.environ.pop(_var, None)

torch.cuda.current_device = lambda: "cpu"  # RotaryEmbedding.get_emb asks for the device

# granite_builders imports megatron.training, which imports the Mamba mixer; with mamba_ssm
# installed, its Triton layer norm queries the GPU at import. Megatron imports mamba_ssm under
# try/except ImportError and the GPT model does not use it, so hide it in this CPU process.
import sys as _sys  # noqa: E402

for _mod in ("mamba_ssm", "causal_conv1d"):
    _sys.modules[_mod] = None  # import raises ImportError

from megatron.core import parallel_state as ps
from megatron.core import tensor_parallel
from megatron.core.transformer import transformer_block
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules

# With Transformer Engine installed, TransformerBlock asks TE for a CPU-offload context,
# which creates a CUDA stream even when offloading is off (fails without a GPU). This test
# does not offload, so take the no-TE branch (a nullcontext).
transformer_block.get_cpu_offload_context = None

s = socket.socket()
s.bind(("127.0.0.1", 0))
port = s.getsockname()[1]
s.close()
torch.distributed.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", world_size=1, rank=0)
ps.initialize_model_parallel(1, 1)
tensor_parallel.get_cuda_rng_tracker().fork = lambda *a, **k: contextlib.nullcontext()

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

from primus.backends.megatron.checkpoint import native_loader_granite as conv
from primus.backends.megatron.core.models.granite.granite_transformer_config import (
    GraniteTransformerConfig,
)

torch.manual_seed(0)
H, NH, NG, D, FFN, V, L, T, B = 64, 4, 2, 16, 128, 128, 2, 7, 2
EPS, THETA = 1e-5, 10000.0
hf_cfg = {
    "architectures": ["GraniteForCausalLM"],
    "model_type": "granite",
    "hidden_size": H,
    "num_attention_heads": NH,
    "num_key_value_heads": NG,
    "intermediate_size": FFN,
    "num_hidden_layers": L,
    "vocab_size": V,
    "rms_norm_eps": EPS,
    "rope_theta": THETA,
    "max_position_embeddings": 64,
    "tie_word_embeddings": True,
    "hidden_act": "silu",
    "attention_bias": False,
    "mlp_bias": False,
    "attention_dropout": 0.0,
    "embedding_multiplier": 12.0,
    "logits_scaling": 10.0,
    "attention_multiplier": 0.05,
    "residual_multiplier": 0.22,
}
f = conv.derive_megatron_fields(hf_cfg)

# Random HF-layout weights (norm weights away from 1 so they matter).
w = {"model.embed_tokens.weight": torch.randn(V, H) * 0.5, "model.norm.weight": 1 + 0.1 * torch.randn(H)}
for i in range(L):
    p = f"model.layers.{i}."
    w[p + "input_layernorm.weight"] = 1 + 0.1 * torch.randn(H)
    w[p + "post_attention_layernorm.weight"] = 1 + 0.1 * torch.randn(H)
    w[p + "self_attn.q_proj.weight"] = torch.randn(NH * D, H) * 0.2
    w[p + "self_attn.k_proj.weight"] = torch.randn(NG * D, H) * 0.2
    w[p + "self_attn.v_proj.weight"] = torch.randn(NG * D, H) * 0.2
    w[p + "self_attn.o_proj.weight"] = torch.randn(H, NH * D) * 0.2
    w[p + "mlp.gate_proj.weight"] = torch.randn(FFN, H) * 0.2
    w[p + "mlp.up_proj.weight"] = torch.randn(FFN, H) * 0.2
    w[p + "mlp.down_proj.weight"] = torch.randn(H, FFN) * 0.2


def rms(x, g):
    return g * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)


def rot_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat([-b, a], dim=-1)


def hf_forward(ids):
    """A plain-torch transcription of GraniteForCausalLM.forward (modeling_granite.py).

    It is a second, independent reference; the real ``transformers`` model
    (``transformers_forward``) is the primary one when it is installed."""
    x = w["model.embed_tokens.weight"][ids] * hf_cfg["embedding_multiplier"]
    inv = 1.0 / THETA ** (torch.arange(0, D, 2).float() / D)
    fr = torch.outer(torch.arange(T).float(), inv)
    emb = torch.cat([fr, fr], dim=-1)
    cos, sin = emb.cos(), emb.sin()
    causal = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
    rm = hf_cfg["residual_multiplier"]
    for i in range(L):
        p = f"model.layers.{i}."
        hs = rms(x, w[p + "input_layernorm.weight"])
        q = (hs @ w[p + "self_attn.q_proj.weight"].T).view(B, T, NH, D).transpose(1, 2)
        k = (hs @ w[p + "self_attn.k_proj.weight"].T).view(B, T, NG, D).transpose(1, 2)
        v = (hs @ w[p + "self_attn.v_proj.weight"].T).view(B, T, NG, D).transpose(1, 2)
        q, k = q * cos + rot_half(q) * sin, k * cos + rot_half(k) * sin
        k, v = k.repeat_interleave(NH // NG, 1), v.repeat_interleave(NH // NG, 1)
        sc = (q @ k.transpose(-1, -2)) * hf_cfg["attention_multiplier"]
        a = sc.masked_fill(causal, float("-inf")).softmax(-1) @ v
        a = a.transpose(1, 2).reshape(B, T, NH * D) @ w[p + "self_attn.o_proj.weight"].T
        x = x + a * rm
        hs = rms(x, w[p + "post_attention_layernorm.weight"])
        m = torch.nn.functional.silu(hs @ w[p + "mlp.gate_proj.weight"].T) * (
            hs @ w[p + "mlp.up_proj.weight"].T
        )
        x = x + (m @ w[p + "mlp.down_proj.weight"].T) * rm
    x = rms(x, w["model.norm.weight"])
    return (x @ w["model.embed_tokens.weight"].T) / hf_cfg["logits_scaling"]


cfg = GraniteTransformerConfig(
    num_layers=L,
    hidden_size=H,
    num_attention_heads=NH,
    num_query_groups=NG,
    kv_channels=D,
    ffn_hidden_size=FFN,
    use_cpu_initialization=True,
    normalization="RMSNorm",
    gated_linear_unit=True,
    add_bias_linear=False,
    hidden_dropout=0.0,
    attention_dropout=0.0,
    layernorm_epsilon=EPS,
    use_mup=f["use_mup"],
    mup_embedding_mult=f["mup_embedding_mult"],
    mup_output_mult=f["mup_output_mult"],
    # attention_multiplier 0.05 (head dim 16) reaches softmax_scale through upstream MuP.
    mup_base_head_dim=f["mup_base_head_dim"],
    mup_attn_scale_power=f["mup_attn_scale_power"],
    residual_multiplier=f["residual_multiplier"],
    bias_activation_fusion=False,
    bias_dropout_fusion=False,
    masked_softmax_fusion=False,
    persist_layer_norm=False,
    apply_rope_fusion=False,
    activation_func=torch.nn.functional.silu,
)
# A block spec: every layer and the final norm use Megatron's local modules. A bare layer
# spec would make TransformerBlock pick TE's norm for the final norm when TE is installed,
# and TE's norm needs a GPU. It is given to the Granite builder as a user spec (``args.spec``),
# which the builder wraps with the scaled bdas, as in production.
_layer_spec = get_gpt_layer_local_spec(normalization="RMSNorm")
CPU_BLOCK_SPEC = TransformerBlockSubmodules(
    layer_specs=[_layer_spec] * L, layer_norm=_layer_spec.submodules.input_layernorm
)

# The model is built by the Granite builder itself (granite_builder), from Megatron args as
# the converter derives them (build args the way native_loader_llama.build_margs maps the
# HF config: tied embeddings, RoPE base, position embedding type, rotary percent).
# Any import error here fails the test: the GPU probes of the image are avoided above
# (mamba_ssm hidden), so a failure is a defect in the Granite code or its imports.
from primus.backends.megatron.core.models.granite.granite_builders import (  # noqa: E402
    granite_builder,
)

builder_args = SimpleNamespace(
    spec=("__main__", "CPU_BLOCK_SPEC"),
    padded_vocab_size=V,
    max_position_embeddings=hf_cfg["max_position_embeddings"],
    fp16_lm_cross_entropy=False,
    untie_embeddings_and_output_weights=f["untie_embeddings_and_output_weights"],
    position_embedding_type="rope",  # native_loader_llama.build_margs
    rotary_percent=1.0,  # Megatron's default (full RoPE)
    rotary_base=f["rotary_base"],  # rope_theta
    transformer_impl="local",
    normalization="RMSNorm",
    swiglu=True,
    add_bias_linear=False,
)
model = granite_builder(builder_args, pre_process=True, post_process=True, config=cfg).eval()
assert model.share_embeddings_and_output_weights, "tied embeddings (tie_word_embeddings: true)"
assert model.rotary_pos_emb.inv_freq.numel() == D // 2
torch.testing.assert_close(
    model.rotary_pos_emb.inv_freq, 1.0 / THETA ** (torch.arange(0, D, 2).float() / D), rtol=0, atol=0
)
assert abs(cfg.softmax_scale - hf_cfg["attention_multiplier"]) < 1e-12, cfg.softmax_scale
for _layer in model.decoder.layers:
    assert _layer.self_attention.core_attention.softmax_scale == cfg.softmax_scale
    assert _layer.self_attn_bda.residual_multiplier == _layer.mlp_bda.residual_multiplier == 0.22


class _Store:
    def get(self, name, dtype):
        return w[name].to(dtype)

    def has(self, name):
        return name in w


margs = SimpleNamespace(
    params_dtype=torch.float32,
    num_attention_heads=NH,
    num_query_groups=NG,
    kv_channels=D,
    hidden_size=H,
    untie_embeddings_and_output_weights=False,
    num_layers=L,
    add_bias_linear=False,
    add_qkv_bias=False,
)
sd, consumed = conv.build_mcore_state_dict(model, _Store(), margs, f, lambda *a: None)
assert consumed == set(w), set(w) ^ consumed
# The converter writes TE names (norms fused into the linears: production uses TE's
# layernorm-linear). TE needs a GPU, so this CPU model uses the local spec, which keeps the
# norms separate; only these two norm names are renamed. Loading the TE names into the TE
# model is covered by the GPU gate (l3-logprob-parity, real weights).
sd = {
    k.replace("self_attention.linear_qkv.layer_norm_weight", "input_layernorm.weight").replace(
        "mlp.linear_fc1.layer_norm_weight", "pre_mlp_layernorm.weight"
    ): v
    for k, v in sd.items()
}
missing, unexpected = model.load_state_dict(sd, strict=False)
missing = [k for k in missing if not k.endswith("_extra_state")]
assert not missing and not unexpected, (missing, unexpected)

ids = torch.randint(0, V, (B, T))
pos = torch.arange(T).unsqueeze(0).expand(B, T)
mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1).view(1, 1, T, T)


TRANSFORMERS_SKIP = None  # why the transformers reference did not run, or None


def transformers_forward(ids):
    """The real ``transformers`` GraniteForCausalLM (eager attention, fp32) on the same HF
    weights, or None when transformers is not installed or has no Granite model (the reason
    is in ``TRANSFORMERS_SKIP``). Any other import error (a broken install) raises."""
    global TRANSFORMERS_SKIP
    import importlib.util

    if importlib.util.find_spec("transformers") is None:
        TRANSFORMERS_SKIP = "transformers is not installed"
        return None
    import transformers

    if not hasattr(transformers, "GraniteForCausalLM"):
        TRANSFORMERS_SKIP = f"transformers {transformers.__version__} has no GraniteForCausalLM"
        return None
    from transformers import GraniteConfig, GraniteForCausalLM

    hf = GraniteForCausalLM(GraniteConfig(**{k: v for k, v in hf_cfg.items() if k != "architectures"}))
    hf.config._attn_implementation = "eager"
    hf = hf.float().eval()
    missing, unexpected = hf.load_state_dict(w, strict=False)
    # lm_head.weight is tied to embed_tokens (not shipped); nothing else may be missing.
    assert not unexpected and set(missing) <= {"lm_head.weight"}, (missing, unexpected)
    assert hf.lm_head.weight.data_ptr() == hf.model.embed_tokens.weight.data_ptr()
    return hf(input_ids=ids).logits.float()


with torch.no_grad():
    got = model(ids, pos, attention_mask=mask).float()
    ref = hf_forward(ids)
    hf_ref = transformers_forward(ids)

result = {
    "max_abs_err_plain_torch": (got - ref).abs().max().item(),
    "ref_scale": ref.abs().max().item(),
    "transformers_available": hf_ref is not None,
    "transformers_skip_reason": TRANSFORMERS_SKIP,
    "max_abs_err_transformers": None if hf_ref is None else (got - hf_ref).abs().max().item(),
}
# The parent test parses this line and checks the tolerance itself.
print("GRANITE_PARITY_RESULT=" + json.dumps(result))
