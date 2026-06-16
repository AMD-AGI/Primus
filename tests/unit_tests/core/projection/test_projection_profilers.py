###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for Primus Projection module profilers.

These tests exercise the *analytical* paths of the projection profilers
(``estimated_num_params`` / ``estimated_activation_memory``) and the
``LanguageModelProfiler`` integration built via ``build_profiler``.

They intentionally avoid the simulation path (``origami`` GEMM backend) and
the GPU benchmarking path, so they run on CPU with no extra dependencies.
"""

import pytest

from primus.core.projection.module_profilers.attention import AttentionProfiler
from primus.core.projection.module_profilers.dense_mlp import DenseMLPProfiler
from primus.core.projection.module_profilers.embedding import EmbeddingProfiler
from primus.core.projection.module_profilers.language_model import (
    build_profiler,
    get_language_model_profiler_spec,
)
from primus.core.projection.module_profilers.moe_mlp import MoEMLPProfiler
from primus.core.projection.module_profilers.output_layer import OutputLayerProfiler
from primus.core.projection.training_config import (
    ModelConfig,
    ModelParallelConfig,
    RuntimeConfig,
    TrainingConfig,
)

HIDDEN = 512
FFN = 1024
VOCAB = 32000
HEADS = 8
KV_CHANNELS = 64
N_LAYERS = 4
NUM_EXPERTS = 8
TOPK = 2


def _model_config(moe: bool = False, swiglu: bool = True, **overrides) -> ModelConfig:
    base = dict(
        num_layers=N_LAYERS,
        hidden_size=HIDDEN,
        padded_vocab_size=VOCAB,
        ffn_hidden_size=FFN,
        num_attention_heads=HEADS,
        kv_channels=KV_CHANNELS,
        num_query_groups=HEADS,
        group_query_attention=False,
        swiglu=swiglu,
        multi_latent_attention=False,
        use_flash_attn=True,
    )
    if moe:
        base.update(
            num_experts=NUM_EXPERTS,
            moe_ffn_hidden_size=FFN,
            moe_router_topk=TOPK,
            moe_pattern=[1] * N_LAYERS,
        )
    else:
        # Dense models leave num_experts unset (None) in real Megatron configs.
        base.update(num_experts=None, moe_pattern=[0] * N_LAYERS)
    base.update(overrides)
    return ModelConfig(**base)


def _training_config(moe: bool = False, swiglu: bool = True, **overrides) -> TrainingConfig:
    return TrainingConfig(
        model_config=_model_config(moe=moe, swiglu=swiglu, **overrides),
        runtime_config=RuntimeConfig(
            global_batch_size=8, micro_batch_size=1, sequence_length=128, data_parallel_size=1
        ),
        model_parallel_config=ModelParallelConfig(),
    )


# ----------------------- exact-formula param tests -----------------------


def test_dense_mlp_params_swiglu():
    p = DenseMLPProfiler(_training_config())
    assert p.estimated_num_params() == 3 * HIDDEN * FFN


def test_dense_mlp_params_no_swiglu():
    p = DenseMLPProfiler(_training_config(swiglu=False))
    assert p.estimated_num_params() == 2 * HIDDEN * FFN


def test_moe_mlp_params_swiglu():
    p = MoEMLPProfiler(_training_config(moe=True))
    assert p.estimated_num_params() == NUM_EXPERTS * 3 * HIDDEN * FFN


def test_embedding_and_output_params_equal_vocab_times_hidden():
    cfg = _training_config()
    assert EmbeddingProfiler(cfg).estimated_num_params() == VOCAB * HIDDEN
    assert OutputLayerProfiler(cfg).estimated_num_params() == VOCAB * HIDDEN


def test_attention_params_positive_and_scales_with_hidden():
    small = AttentionProfiler(_training_config(hidden_size=512)).estimated_num_params()
    big = AttentionProfiler(_training_config(hidden_size=1024)).estimated_num_params()
    assert small > 0
    assert big > small


# ----------------------- activation-memory sanity -----------------------


@pytest.mark.parametrize(
    "profiler_cls", [DenseMLPProfiler, AttentionProfiler, EmbeddingProfiler, OutputLayerProfiler]
)
def test_activation_memory_positive_dense(profiler_cls):
    cfg = _training_config()
    assert profiler_cls(cfg).estimated_activation_memory(batch_size=1, seq_len=128) > 0


def test_moe_activation_memory_positive_for_moe_config():
    assert MoEMLPProfiler(_training_config(moe=True)).estimated_activation_memory(1, 128) > 0


# ----------------------- LanguageModelProfiler integration -----------------------


def test_language_model_profiler_dense_totals_positive(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("NNODES", "1")
    monkeypatch.setenv("GPUS_PER_NODE", "8")
    prof = build_profiler(get_language_model_profiler_spec(_training_config()))
    assert prof.estimated_num_params(rank=None) > 0
    assert prof.estimated_activation_memory(batch_size=1, seq_len=128) > 0


def test_language_model_profiler_moe_has_more_params_than_dense(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("NNODES", "1")
    monkeypatch.setenv("GPUS_PER_NODE", "8")
    dense = build_profiler(get_language_model_profiler_spec(_training_config(moe=False)))
    moe = build_profiler(get_language_model_profiler_spec(_training_config(moe=True)))
    assert moe.estimated_num_params(rank=None) > dense.estimated_num_params(rank=None)


def test_language_model_param_count_includes_embedding_and_output(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("NNODES", "1")
    monkeypatch.setenv("GPUS_PER_NODE", "8")
    cfg = _training_config()
    prof = build_profiler(get_language_model_profiler_spec(cfg))
    total = prof.estimated_num_params(rank=None)
    # Embedding + output layer alone are 2 * VOCAB * HIDDEN; total must exceed that.
    assert total > 2 * VOCAB * HIDDEN


def test_single_gpu_config_does_not_divide_by_zero(monkeypatch):
    """Regression: TP*PP*EP*CP < GPUS_PER_NODE on a single node used to yield
    num_nodes=0 and a ZeroDivisionError in get_dp_size()."""
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("NNODES", "1")
    monkeypatch.setenv("GPUS_PER_NODE", "8")
    prof = build_profiler(get_language_model_profiler_spec(_training_config()))
    assert prof.get_dp_size() >= 1
    assert prof.estimated_activation_memory(batch_size=1, seq_len=128) > 0


def test_memory_print_profiler_hierarchy_runs(monkeypatch, capsys):
    """Exercise the memory-projection hierarchy walker end-to-end (no CLI)."""
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("NNODES", "1")
    monkeypatch.setenv("GPUS_PER_NODE", "8")
    from primus.core.projection.memory_projection.simulate import (
        print_profiler_hierarchy,
    )

    prof = build_profiler(get_language_model_profiler_spec(_training_config(moe=True)))
    print_profiler_hierarchy(prof, batch_size=1, seq_len=128, rank=0, name="LanguageModelProfiler", depth=0)
    out = capsys.readouterr().out
    assert "Total Number of Parameters" in out
    # The dense-config None bug previously surfaced here as an error line.
    assert "Error calculating metrics" not in out


# ----------------------- simulated path via a fake GEMM backend -----------------------
# These cover the ``_get_simulated_results`` branches without the origami
# dependency (the real GEMM backend), by injecting a tiny stub backend.


class _FakeSimResult:
    forward_time_ms = 0.1
    backward_time_ms = 0.2


class _FakeGemmBackend:
    hbm_bandwidth_gbps = 5300.0

    def simulate_gemm(self, *args, **kwargs):
        return _FakeSimResult()

    def simulate_mlp_gemms(self, *args, **kwargs):
        return _FakeSimResult()

    def simulate_attention_gemms(self, *args, **kwargs):
        return _FakeSimResult()


def test_dense_mlp_simulated_path():
    p = DenseMLPProfiler(_training_config())
    p.set_gemm_backend(_FakeGemmBackend())
    assert p.measured_forward_time(1, 128) >= 0
    assert p.measured_backward_time(1, 128) >= 0
    assert p.measured_activation_memory(1, 128) > 0


@pytest.mark.parametrize("turbo", [False, True])
def test_moe_mlp_simulated_path(turbo):
    cfg = _training_config(moe=True, enable_primus_turbo=turbo, use_turbo_grouped_mlp=turbo)
    p = MoEMLPProfiler(cfg)
    p.set_gemm_backend(_FakeGemmBackend())
    # routed experts + router + permute + activation overheads -> strictly positive
    assert p.measured_forward_time(1, 128) > 0
    assert p.measured_backward_time(1, 128) > 0


def test_output_layer_simulated_path():
    p = OutputLayerProfiler(_training_config())
    p.set_gemm_backend(_FakeGemmBackend())
    assert p.measured_forward_time(1, 128) >= 0
    assert p.measured_backward_time(1, 128) >= 0


def test_attention_simulated_gemm_path():
    p = AttentionProfiler(_training_config())
    p.set_gemm_backend(_FakeGemmBackend())
    assert p.measured_forward_time(1, 128) >= 0
    assert p.measured_backward_time(1, 128) >= 0


# ----------------------- dense must not build the MoE sub-tree -----------------------


def test_dense_model_omits_moe_subtree(monkeypatch):
    """Dense models (num_experts=None) must not instantiate the MoE sub-tree,
    so the hierarchy walker never evaluates MoE profilers on a dense model."""
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("NNODES", "1")
    monkeypatch.setenv("GPUS_PER_NODE", "8")
    dense = build_profiler(get_language_model_profiler_spec(_training_config(moe=False)))
    moe = build_profiler(get_language_model_profiler_spec(_training_config(moe=True)))
    assert "moe_transformer_layer" not in dense.sub_profilers
    assert "moe_transformer_layer" in moe.sub_profilers
    # Dense totals must still be correct without the MoE branch.
    assert dense.estimated_num_params(rank=None) > 0
