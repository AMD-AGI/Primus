###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Builder, config and tiny-model tests for Muse Spark.

The formula checks live in ``test_muse_spark_math.py``. This file covers the
training entry point ``get_model_provider("muse_spark")`` -> ``muse_spark_builder``
-> ``MuseSparkGPTModel``, and compares one decoder layer's attention, latent
MoE and gated residuals with fp32 references written from
transformers' ``modeling_muse_spark.py``, using the Megatron module weights.
"""

import math
import os
import socket
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.torch_norm import L2Norm
from megatron.core.transformer.utils import is_layer_window_attention

from primus.backends.megatron.core.models.muse_spark.muse_spark_builders import (
    model_provider,
    muse_spark_builder,
)
from primus.backends.megatron.core.models.muse_spark.muse_spark_model import (
    MuseSparkGPTModel,
)
from primus.backends.megatron.core.models.muse_spark.muse_spark_modules import (
    MuseSparkMoELayer,
    MuseSparkQueryNorm,
    MuseSparkSelfAttention,
    MuseSparkTopKRouter,
    MuseSparkTransformerLayer,
)
from primus.backends.megatron.core.models.muse_spark.muse_spark_ops import (
    input_scaled_rms_norm,
    residual_gate,
    softcap_logits,
)
from primus.backends.megatron.core.models.muse_spark.muse_spark_transformer_config import (
    MuseSparkTransformerConfig,
)
from primus.core.utils import logger as primus_logger
from primus.core.utils.import_utils import get_model_provider

SEQ = 16
BATCH = 2
VOCAB = 128


def _ensure_logger() -> None:
    if primus_logger._logger is not None:
        return
    primus_logger.setup_logger(
        primus_logger.LoggerConfig(
            exp_root_path=os.environ.get("UT_LOG_PATH", "ut_out"),
            work_group="develop",
            user_name="root",
            exp_name="unittest",
            module_name="UT-muse-spark",
            file_sink_level="DEBUG",
            stderr_sink_level="ERROR",
            node_ip="localhost",
            rank=0,
            world_size=1,
        ),
        is_head=False,
    )


def _tiny_config(**overrides) -> MuseSparkTransformerConfig:
    fields = dict(
        num_layers=4,
        hidden_size=64,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=32,
        ffn_hidden_size=128,
        normalization="RMSNorm",
        layernorm_zero_centered_gamma=True,
        layernorm_epsilon=1e-5,
        qk_l2_norm=True,
        attention_output_gate=True,
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        add_qkv_bias=False,
        bias_activation_fusion=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        num_moe_experts=4,
        moe_router_topk=2,
        moe_ffn_hidden_size=32,
        moe_latent_size=32,
        moe_grouped_gemm=True,
        moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        moe_router_dtype="fp32",
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0.0,
        moe_token_dispatcher_type="alltoall",
        # Short enough that a 16-token sequence exercises the window.
        muse_sliding_window=4,
        bf16=True,
        params_dtype=torch.bfloat16,
        attention_backend=AttnBackend.auto,
        sequence_parallel=False,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
    )
    fields.update(overrides)
    return MuseSparkTransformerConfig(**fields)


def _args(**overrides):
    args = SimpleNamespace(
        use_legacy_models=False,
        transformer_impl="transformer_engine",
        position_embedding_type="rope",
        spec=None,
        experimental_attention_variant=None,
        mtp_num_layers=None,
        padded_vocab_size=VOCAB,
        max_position_embeddings=64,
        fp16_lm_cross_entropy=False,
        untie_embeddings_and_output_weights=True,
        rotary_percent=1.0,
        rotary_base=10000,
        use_rope_scaling=False,
        record_memory_history=False,
        # Read by PrimusTopKRouter.routing.
        router_logit_softcapping=None,
        enable_primus_turbo=False,
        moe_use_fused_router_with_aux_score=False,
        moe_router_force_load_balancing=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


# ---------------------------------------------------------------------------
# Wiring, config and builder checks (no GPU)
# ---------------------------------------------------------------------------


def test_get_model_provider_wires_muse_spark_builder():
    _ensure_logger()
    provider = get_model_provider("muse_spark")
    assert provider.func is model_provider
    assert provider.args == (muse_spark_builder,)


def test_config_derives_hybrid_attention_from_num_layers():
    config = _tiny_config(num_layers=8)
    # Layers 3 and 7 (0-based) are global: full attention, no RoPE.
    expected_global = [False, False, False, True] * 2
    assert config.no_rope_freq == [int(g) for g in expected_global]
    assert config.window_size == (3, 0)
    assert config.window_attn_skip_freq == [int(not g) for g in expected_global]
    for layer_number, is_global in enumerate(expected_global, start=1):
        assert is_layer_window_attention(
            config.window_size, config.window_attn_skip_freq, layer_number
        ) is (not is_global)
    # The qk scale factor rides on the query norm, so attention keeps 1/sqrt(head_dim).
    assert config.softmax_scale is None


def test_config_without_sliding_window_is_full_attention():
    config = _tiny_config(muse_sliding_window=None)
    assert config.window_size is None
    assert config.window_attn_skip_freq is None
    assert config.no_rope_freq == [0, 0, 0, 1]


@pytest.mark.parametrize(
    "overrides, missing",
    [
        (dict(qk_l2_norm=False), "qk_l2_norm"),
        (dict(layernorm_zero_centered_gamma=False), "layernorm_zero_centered_gamma"),
        (dict(attention_output_gate=False), "attention_output_gate"),
        (dict(moe_latent_size=None), "moe_latent_size"),
        (dict(moe_router_score_function="softmax", moe_router_enable_expert_bias=False), "sigmoid"),
        (dict(tensor_model_parallel_size=2, sequence_parallel=True), "expert_tensor_parallel_size"),
    ],
)
def test_config_rejects_non_muse_architecture(overrides, missing):
    with pytest.raises(ValueError, match=missing):
        _tiny_config(**overrides)


def test_config_allows_tensor_parallel_with_whole_experts():
    config = _tiny_config(tensor_model_parallel_size=2, sequence_parallel=True, expert_tensor_parallel_size=1)
    assert config.expert_tensor_parallel_size == 1


@pytest.mark.parametrize(
    "overrides, env, match",
    [
        (dict(moe_use_legacy_grouped_gemm=True), {}, "moe_use_legacy_grouped_gemm"),
        (dict(use_turbo_mega_moe=True), {}, "use_turbo_mega_moe"),
        (dict(mtp_num_layers=1), {}, "mtp_num_layers"),
        (dict(enable_primus_turbo=True, moe_use_fused_router_with_aux_score=True), {}, "fused router"),
        (dict(enable_fused_linear_ce=True), {}, "fused linear cross entropy"),
        ({}, {"PRIMUS_FUSED_RESIDUAL_NORM": "1"}, "PRIMUS_FUSED_RESIDUAL_NORM"),
    ],
)
def test_builder_rejects_options_that_bypass_muse_modules(overrides, env, match, monkeypatch):
    _ensure_logger()
    for name in ("PRIMUS_FUSED_RESIDUAL_NORM", "PRIMUS_FUSED_RESIDUAL_NORM_V2", "FUSED_LINEAR_CE"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=match):
        muse_spark_builder(_args(**overrides), pre_process=True, post_process=True, config=_tiny_config())


def test_builder_rejects_a_plain_transformer_config():
    with pytest.raises(AssertionError, match="MuseSparkTransformerConfig"):
        muse_spark_builder(_args(), pre_process=True, post_process=True, config=object())


# ---------------------------------------------------------------------------
# Tiny model on one GPU
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def parallel_state():
    """Single-rank process groups so ``GPTModel`` can build its embedding, MoE and logits."""
    _ensure_logger()
    if not torch.cuda.is_available():
        pytest.skip("MuseSparkGPTModel builds Transformer Engine modules on CUDA.")

    import torch.distributed as dist
    from megatron.core import parallel_state as ps
    from megatron.core.tensor_parallel import random as tp_random

    torch.cuda.set_device(0)
    if not dist.is_initialized():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://127.0.0.1:{port}",
            world_size=1,
            rank=0,
        )
    if ps.model_parallel_is_initialized():
        ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        context_parallel_size=1,
    )
    try:
        tp_random.initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
    except (ImportError, AssertionError):
        tp_random.initialize_rng_tracker(use_cudagraphable_rng=True, force_reset=True)
    tp_random.model_parallel_cuda_manual_seed(42)
    yield
    if ps.model_parallel_is_initialized():
        ps.destroy_model_parallel()
    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.fixture
def model(parallel_state, monkeypatch):
    """A 4-layer tiny model with every zero-initialised Muse weight randomised."""
    del parallel_state
    # GPTModel sets the three NVTE_*_ATTN vars from attention_backend and
    # refuses values leaked by an earlier test; monkeypatch restores them.
    for name in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
        monkeypatch.delenv(name, raising=False)
    for name in ("PRIMUS_FUSED_RESIDUAL_NORM", "PRIMUS_FUSED_RESIDUAL_NORM_V2", "FUSED_LINEAR_CE"):
        monkeypatch.delenv(name, raising=False)
    from megatron.training.global_vars import set_args

    set_args(_args())
    torch.manual_seed(0)
    built = get_model_provider("muse_spark")(pre_process=True, post_process=True, config=_tiny_config())
    built.cuda()
    # Zero-initialised norm weights and a zero expert bias would hide a
    # missing (1 + w), a swapped norm or a bias leaking into the weights.
    with torch.no_grad():
        for name, param in built.named_parameters():
            if "norm" in name and param.dim() == 1:
                param.normal_(0.0, 0.2)
        for layer in built.decoder.layers:
            layer.mlp.router.expert_bias.normal_(0.0, 0.5)
    return built


def _inputs(device="cuda"):
    generator = torch.Generator(device="cpu").manual_seed(1)
    input_ids = torch.randint(0, VOCAB, (BATCH, SEQ), generator=generator).to(device)
    position_ids = torch.arange(SEQ, device=device).unsqueeze(0).expand(BATCH, -1)
    return input_ids, position_ids


def _rms(x, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def _rel_err(got, ref):
    return ((got.float() - ref).norm() / ref.norm()).item()


def _capture(module, store, key, *, pre=False):
    """Record a module's first positional (or ``hidden_states``) input, or its output."""
    if pre:

        def hook(_module, args, kwargs):
            store[key] = args[0] if args else kwargs["hidden_states"]

        return module.register_forward_pre_hook(hook, with_kwargs=True)

    def hook(_module, _inputs, output):
        store[key] = output[0] if isinstance(output, tuple) else output

    return module.register_forward_hook(hook)


def _ref_rope(x, positions, base):
    dim = x.shape[-1]
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim))
    freqs = positions.float()[:, None] * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)[:, None, None, :]
    x1, x2 = x.chunk(2, dim=-1)
    return x * emb.cos() + torch.cat((-x2, x1), dim=-1) * emb.sin()


def _ref_attention(attention, hidden_states, config, *, rope, window):
    """``MuseSparkTextAttention.forward`` in fp32 from Megatron's fused weights.

    ``linear_qkv`` carries the input norm and packs, per KV group,
    ``[q heads, gate heads, k, v]``.
    """
    seq, batch, _ = hidden_states.shape
    heads, groups, head_dim = config.num_attention_heads, config.num_query_groups, config.kv_channels
    per_group = heads // groups
    eps = config.layernorm_epsilon
    qkv = attention.linear_qkv

    normed = _rms(hidden_states.float(), eps) * (1.0 + qkv.layer_norm_weight.float())
    mixed = (normed @ qkv.weight.float().t()).view(seq, batch, groups, (2 * per_group + 2) * head_dim)
    query, gate, key, value = torch.split(
        mixed, [per_group * head_dim, per_group * head_dim, head_dim, head_dim], dim=-1
    )
    query = query.reshape(seq, batch, heads, head_dim)
    gate = gate.reshape(seq, batch, heads, head_dim)

    query = _rms(query, eps) * config.muse_qk_scale_factor
    key = _rms(key, eps)
    if rope:
        positions = torch.arange(seq, device=hidden_states.device)
        query = _ref_rope(query, positions, 10000)
        key = _ref_rope(key, positions, 10000)
    key = key.repeat_interleave(per_group, dim=2)
    value = value.repeat_interleave(per_group, dim=2)

    scores = torch.einsum("sbhd,tbhd->bhst", query, key) / math.sqrt(head_dim)
    q_idx = torch.arange(seq, device=hidden_states.device)[:, None]
    k_idx = torch.arange(seq, device=hidden_states.device)[None, :]
    allowed = k_idx <= q_idx
    if window is not None:
        # HF's sliding-window mask keeps keys with kv_idx > q_idx - sliding_window.
        allowed &= k_idx > q_idx - window
    scores = scores.masked_fill(~allowed, float("-inf"))
    out = torch.einsum("bhst,tbhd->sbhd", scores.softmax(dim=-1), value)

    out = _rms(out, eps) * torch.sigmoid(gate)
    return out.reshape(seq, batch, heads * head_dim) @ attention.linear_proj.weight.float().t()


def _ref_moe(moe, hidden_states, config):
    """``MuseSparkMoE.forward`` in fp32 from Megatron's module weights."""
    shape = hidden_states.shape
    x = hidden_states.float().reshape(-1, shape[-1])
    router = moe.router

    logits = (x @ router.weight.float().t()) * config.muse_output_multiplier
    scores = torch.sigmoid(logits)
    selected = torch.topk(scores + router.expert_bias.float(), config.moe_router_topk, dim=-1).indices
    weights = scores.gather(1, selected)
    weights = weights / weights.sum(dim=-1, keepdim=True) * (config.moe_router_topk_scaling_factor or 1.0)

    latent = x @ moe.fc1_latent_proj.weight.float().t()
    latent = _rms(latent, config.layernorm_epsilon) * (1.0 + moe.pre_expert_norm.weight.float())

    combined = torch.zeros_like(latent)
    post_weight = moe.post_expert_norm.weight.float()
    for expert in range(config.num_moe_experts):
        fc1 = getattr(moe.experts.linear_fc1, f"weight{expert}").float()
        fc2 = getattr(moe.experts.linear_fc2, f"weight{expert}").float()
        gate, up = (latent @ fc1.t()).chunk(2, dim=-1)
        out = (F.silu(gate) * up) @ fc2.t()
        out = _rms(out * (1.0 + post_weight), config.muse_post_norm_eps)
        coef = (weights * (selected == expert)).sum(dim=-1, keepdim=True)
        combined = combined + out * coef

    return (combined @ moe.fc2_latent_proj.weight.float().t()).reshape(shape)


def test_tiny_model_swaps_in_muse_modules(model):
    assert isinstance(model, MuseSparkGPTModel)
    assert model.share_embeddings_and_output_weights is False
    assert len(model.decoder.layers) == 4
    for layer in model.decoder.layers:
        assert isinstance(layer, MuseSparkTransformerLayer)
        assert isinstance(layer.self_attention, MuseSparkSelfAttention)
        assert isinstance(layer.self_attention.q_layernorm, MuseSparkQueryNorm)
        assert type(layer.self_attention.k_layernorm) is L2Norm
        assert isinstance(layer.mlp, MuseSparkMoELayer)
        assert isinstance(layer.mlp.router, MuseSparkTopKRouter)
        assert layer.mlp.fc1_latent_proj.weight.shape == (32, 64)
        assert layer.mlp.pre_expert_norm.weight.shape == (32,)
        assert layer.mlp.post_expert_norm.weight.shape == (32,)
        assert layer.self_attn_bda(True, True) == layer._apply_self_attn_gated_residual
        assert layer.mlp_bda(True, True) == layer._apply_mlp_gated_residual
        assert layer.post_self_attn_residual_gate.gate.shape == (64,)
        # The output gate doubles the q width of linear_qkv: 4 q + 4 gate + 2 k + 2 v heads.
        assert layer.self_attention.linear_qkv.weight.shape == ((4 + 4 + 2 + 2) * 32, 64)


@pytest.mark.parametrize("layer_index", [0, 3])
def test_layer_matches_reference(model, layer_index):
    """Layer 0 is sliding-window with RoPE, layer 3 is full attention with NoPE."""
    config = model.config
    layer = model.decoder.layers[layer_index]
    seen = {}
    handles = [
        _capture(layer, seen, "layer_in", pre=True),
        _capture(layer, seen, "layer_out"),
        _capture(layer.self_attention, seen, "attn_in", pre=True),
        _capture(layer.self_attention, seen, "attn_out"),
        _capture(layer.pre_mlp_layernorm, seen, "mlp_residual", pre=True),
        _capture(layer.mlp, seen, "moe_in", pre=True),
        _capture(layer.mlp, seen, "moe_out"),
    ]
    model.eval()
    try:
        with torch.no_grad():
            model(*_inputs(), attention_mask=None)
    finally:
        for handle in handles:
            handle.remove()

    is_global = layer_index == 3
    with torch.no_grad():
        attn_ref = _ref_attention(
            layer.self_attention,
            seen["attn_in"],
            config,
            rope=not is_global,
            window=None if is_global else config.muse_sliding_window,
        )
        # bf16 runs land near 1e-2; a one-token window shift, a missing o_norm
        # or probs applied before the post-expert norm all exceed 0.5.
        assert _rel_err(seen["attn_out"], attn_ref) < 5e-2

        moe_ref = _ref_moe(layer.mlp, seen["moe_in"], config)
        assert _rel_err(seen["moe_out"], moe_ref) < 5e-2

        # Gated residuals, recomputed from the captured sublayer outputs.
        gated_attn = residual_gate(
            seen["layer_in"],
            _rms(seen["attn_out"].float(), config.muse_post_norm_eps).to(seen["attn_out"].dtype),
            layer.post_self_attn_residual_gate.gate,
            config.muse_residual_gate_temperature,
            config.muse_residual_gate_eps,
        )
        torch.testing.assert_close(seen["mlp_residual"], gated_attn, rtol=2e-2, atol=2e-2)
        gated_mlp = residual_gate(
            seen["mlp_residual"],
            input_scaled_rms_norm(seen["moe_out"], layer.post_mlp_norm.weight, config.muse_post_norm_eps),
            layer.post_mlp_residual_gate.gate,
            config.muse_residual_gate_temperature,
            config.muse_residual_gate_eps,
        )
        torch.testing.assert_close(seen["layer_out"], gated_mlp, rtol=2e-2, atol=2e-2)


def test_normalized_embedding_and_capped_logits(model):
    config = model.config
    seen = {}
    handles = [
        _capture(model.embedding, seen, "embedding"),
        _capture(model.output_layer, seen, "raw_logits"),
    ]
    model.eval()
    try:
        with torch.no_grad():
            logits = model(*_inputs(), attention_mask=None)
    finally:
        for handle in handles:
            handle.remove()

    # The hook registered by MuseSparkGPTModel runs first, so this sees normalized rows.
    rms = seen["embedding"].float().pow(2).mean(dim=-1)
    torch.testing.assert_close(rms, torch.ones_like(rms), rtol=2e-2, atol=2e-2)

    assert logits.shape == (BATCH, SEQ, VOCAB)
    expected = softcap_logits(
        seen["raw_logits"].transpose(0, 1), config.muse_output_multiplier, config.final_logit_softcapping
    )
    torch.testing.assert_close(logits, expected)
    assert logits.float().abs().max() <= config.final_logit_softcapping


def test_tiny_model_trains(model):
    model.train()
    input_ids, position_ids = _inputs()
    labels = torch.roll(input_ids, shifts=-1, dims=1)
    loss = model(input_ids, position_ids, attention_mask=None, labels=labels)
    assert loss.shape == (BATCH, SEQ)
    assert torch.isfinite(loss).all()
    loss.mean().backward()

    layer = model.decoder.layers[0]
    for name, param in (
        ("embedding", model.embedding.word_embeddings.weight),
        ("attention residual gate", layer.post_self_attn_residual_gate.gate),
        ("mlp residual gate", layer.post_mlp_residual_gate.gate),
        ("post-mlp norm", layer.post_mlp_norm.weight),
        ("pre-expert norm", layer.mlp.pre_expert_norm.weight),
        ("post-expert norm", layer.mlp.post_expert_norm.weight),
        ("router", layer.mlp.router.weight),
    ):
        grad = getattr(param, "main_grad", None)
        grad = param.grad if grad is None else grad
        assert grad is not None, f"{name} got no gradient"
        assert torch.isfinite(grad).all(), f"{name} gradient is not finite"
        assert grad.abs().sum() > 0, f"{name} gradient is zero"
