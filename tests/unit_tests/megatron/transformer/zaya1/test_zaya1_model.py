###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Builder, wiring, and tiny-config forward tests for ZAYA1.

The mixer-formula checks live in ``test_zaya1_math.py``. This file covers the
training entry point: ``get_model_provider("zaya1")`` -> ``zaya1_builder`` ->
``Zaya1Model``, the alternating CCA/MoE stack, RMSNorm, and one forward/backward.
"""

import os
import socket

import pytest
import torch
from megatron.core.transformer.enums import AttnBackend
from torch import nn

from primus.backends.megatron.core.models.zaya1.zaya1_builders import (
    model_provider,
    zaya1_builder,
)
from primus.backends.megatron.core.models.zaya1.zaya1_model import Zaya1Model
from primus.backends.megatron.core.models.zaya1.zaya1_modules import RMSNorm, ZayaStack
from primus.backends.megatron.core.models.zaya1.zaya1_transformer_config import (
    Zaya1TransformerConfig,
)
from primus.core.utils import logger as primus_logger
from primus.core.utils.import_utils import get_model_provider


def _ensure_logger() -> None:
    if primus_logger._logger is not None:
        return
    primus_logger.setup_logger(
        primus_logger.LoggerConfig(
            exp_root_path=os.environ.get("UT_LOG_PATH", "ut_out"),
            work_group="develop",
            user_name="root",
            exp_name="unittest",
            module_name="UT-zaya1",
            file_sink_level="DEBUG",
            stderr_sink_level="ERROR",
            node_ip="localhost",
            rank=0,
            world_size=1,
        ),
        is_head=False,
    )


def _tiny_config(**overrides) -> Zaya1TransformerConfig:
    fields = dict(
        num_layers=4,
        hidden_size=32,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=8,
        ffn_hidden_size=16,
        moe_ffn_hidden_size=16,
        num_moe_experts=4,
        moe_router_topk=1,
        moe_router_pre_softmax=True,
        layernorm_epsilon=1e-5,
        normalization="RMSNorm",
        add_bias_linear=False,
        bias_activation_fusion=False,
        # This image sets NVTE_FLASH_ATTN=0 and NVTE_FUSED_ATTN=1. CCA does
        # not call TE attention; the flag only satisfies LanguageModule.
        attention_backend=AttnBackend.fused,
        sequence_parallel=False,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        zaya_mlp_expansion=8,
        zaya_use_mod=True,
        zaya_use_eda=True,
        zaya_high_prec=True,
        scale_residual_merge=True,
        cca_time0=2,
        cca_time1=2,
        partial_rotary_factor=0.5,
        zaya_balance_lr=1.0e-3,
    )
    fields.update(overrides)
    return Zaya1TransformerConfig(**fields)


def _args(**overrides):
    from types import SimpleNamespace

    args = SimpleNamespace(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        sequence_parallel=False,
        use_legacy_models=False,
        multi_latent_attention=False,
        position_embedding_type="rope",
        padded_vocab_size=64,
        max_position_embeddings=32,
        fp16_lm_cross_entropy=False,
        untie_embeddings_and_output_weights=False,
        record_memory_history=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


@pytest.fixture(scope="module")
def parallel_state():
    """TP=1 process group so ``Zaya1Model`` can build its embedding and logits."""
    _ensure_logger()
    if not torch.cuda.is_available():
        pytest.skip("Zaya1Model embedding and output layer allocate on CUDA.")

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


def test_get_model_provider_wires_zaya1_builder():
    _ensure_logger()
    provider = get_model_provider("zaya1")
    assert provider.func is model_provider
    assert provider.args == (zaya1_builder,)


def test_builder_rejects_non_data_parallel():
    args = _args(tensor_model_parallel_size=2)
    with pytest.raises(NotImplementedError, match="tensor_model_parallel_size=2"):
        zaya1_builder(args, pre_process=True, post_process=True, config=_tiny_config())


def test_builder_rejects_mla_legacy_and_non_rope():
    with pytest.raises(AssertionError, match="Zaya1TransformerConfig"):
        zaya1_builder(args=_args(), pre_process=True, post_process=True, config=object())
    with pytest.raises(AssertionError, match="use_legacy_models"):
        zaya1_builder(
            args=_args(use_legacy_models=True),
            pre_process=True,
            post_process=True,
            config=_tiny_config(),
        )
    with pytest.raises(AssertionError, match="multi_latent_attention"):
        zaya1_builder(
            args=_args(multi_latent_attention=True),
            pre_process=True,
            post_process=True,
            config=_tiny_config(),
        )
    with pytest.raises(ValueError, match="position_embedding_type"):
        zaya1_builder(
            args=_args(position_embedding_type="learned_absolute"),
            pre_process=True,
            post_process=True,
            config=_tiny_config(),
        )


def test_rmsnorm_matches_fp32_formula():
    hidden = 8
    eps = 1e-5
    norm = RMSNorm(hidden, eps)
    with torch.no_grad():
        norm.weight.uniform_(0.5, 1.5)
    hidden_states = torch.randn(3, 2, hidden)
    got = norm(hidden_states)
    variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
    ref = hidden_states.float() * torch.rsqrt(variance + eps) * norm.weight.float()
    torch.testing.assert_close(got, ref)
    assert got.dtype == hidden_states.dtype


def test_disabled_residual_scaling_skips_both_affines():
    config = _tiny_config(num_layers=2, scale_residual_merge=False)
    stack = ZayaStack(config)
    assert all(layer.res_scale is None for layer in stack.layers)
    assert stack.res_scale is None
    assert [layer.kind for layer in stack.layers] == ["a", "m"]
    assert isinstance(stack.layers[0].input_norm, RMSNorm)
    assert isinstance(stack.final_norm, RMSNorm)


def test_tiny_model_wires_stages_and_forwards(parallel_state):
    del parallel_state
    from megatron.training.global_vars import set_args

    config = _tiny_config()
    args = _args()
    set_args(args)
    model = get_model_provider("zaya1")(pre_process=True, post_process=True, config=config)
    assert isinstance(model, Zaya1Model)
    kinds = [layer.kind for layer in model.decoder.layers]
    assert kinds == ["a", "m", "a", "m"]
    assert model.decoder.layers[0].self_attn is not None
    assert model.decoder.layers[0].zaya_block is None
    assert model.decoder.layers[1].self_attn is None
    assert isinstance(model.decoder.layers[1].zaya_block, nn.Module)
    assert model.decoder.layers[0].res_scale.has_residual is False
    assert model.decoder.layers[1].res_scale.has_residual is True
    assert model.share_embeddings_and_output_weights is True

    device = torch.device("cuda")
    model.to(device)
    batch, seq, vocab = 2, 8, args.padded_vocab_size
    input_ids = torch.randint(0, vocab, (batch, seq), device=device)
    position_ids = torch.arange(seq, device=device).unsqueeze(0).expand(batch, -1)
    logits = model(input_ids, position_ids, attention_mask=None)
    assert logits.shape == (batch, seq, vocab)
    assert torch.isfinite(logits).all()

    labels = torch.randint(0, vocab, (batch, seq), device=device)
    loss = model(input_ids, position_ids, attention_mask=None, labels=labels)
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    assert torch.isfinite(model.embedding.word_embeddings.weight.grad).all()

    with pytest.raises(NotImplementedError, match="Packed document"):
        model(input_ids, position_ids, None, packed_seq_params=object())
