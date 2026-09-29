###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the projection's per-backend benchmark adapters.

The benchmark harness needs two things from a backend it has never seen: where
the transformer layers live, and what their ``forward`` wants.  These tests pin
both for Megatron and TorchTitan against stand-in module trees shaped like the
real ones, so a backend's conventions can be checked without a GPU, without the
backend installed, and without running a kernel.

Also covers the flat-to-native write-back, which is what makes the driver's
bench-shrinking edits (layer cap, EP rescale, PP flatten) actually reach the
model TorchTitan builds.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from primus.core.projection.bench_harness import (
    ATTENTION,
    EMBEDDING,
    LAYER,
    MLP,
    MOE,
    OUTPUT,
    BenchContext,
    BenchModelAdapter,
    available_bench_frameworks,
    get_model_adapter,
    register_model_adapter,
    resolve_model_adapter,
)
from primus.core.projection.bench_harness.megatron import MegatronBenchAdapter
from primus.core.projection.bench_harness.torchtitan import TorchTitanBenchAdapter
from primus.core.projection.frameworks.torchtitan import (
    torchtitan_apply_bench_overrides,
)

HIDDEN = 128
SEQ = 64
BATCH = 2


def _ctx(cp_size: int = 1) -> BenchContext:
    config = SimpleNamespace(
        model_config=SimpleNamespace(hidden_size=HIDDEN),
        model_parallel_config=SimpleNamespace(context_model_parallel_size=cp_size),
    )
    return BenchContext.build(config, batch_size=BATCH, seq_len=SEQ)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_builtin_backends_are_benchmarkable():
    frameworks = available_bench_frameworks()
    assert "megatron" in frameworks
    assert "torchtitan" in frameworks


def test_adapter_lookup_is_case_insensitive():
    assert isinstance(get_model_adapter("MegaTron"), MegatronBenchAdapter)
    assert isinstance(get_model_adapter("  TorchTitan "), TorchTitanBenchAdapter)


def test_unknown_framework_names_the_benchmarkable_ones():
    with pytest.raises(NotImplementedError) as excinfo:
        resolve_model_adapter("tensorflow")
    message = str(excinfo.value)
    assert "tensorflow" in message
    # The error has to be actionable: it should say what *can* be benchmarked.
    assert "megatron" in message and "torchtitan" in message


def test_out_of_tree_backend_can_register():
    class Stub(BenchModelAdapter):
        framework = "stub"

        def discover(self, model):
            raise NotImplementedError

        def layer_submodules(self, layer):
            raise NotImplementedError

        def inputs(self, kind, module, ctx):
            return None

    register_model_adapter("stub-backend", Stub())
    assert isinstance(resolve_model_adapter("stub-backend"), Stub)


def test_registry_rejects_non_adapters():
    with pytest.raises(TypeError):
        register_model_adapter("bogus", object())


# ---------------------------------------------------------------------------
# Megatron conventions
# ---------------------------------------------------------------------------


def _megatron_layer():
    return SimpleNamespace(self_attention=nn.Identity(), mlp=nn.Identity())


def _megatron_model(num_layers=3, wrapped=False):
    language_model = SimpleNamespace(
        embedding=SimpleNamespace(word_embeddings=nn.Embedding(8, HIDDEN)),
        output_layer=nn.Linear(HIDDEN, 8),
        decoder=SimpleNamespace(layers=[_megatron_layer() for _ in range(num_layers)]),
    )
    model = SimpleNamespace(language_model=language_model)
    return SimpleNamespace(module=model) if wrapped else model


def test_megatron_discovers_layers_embedding_and_head():
    parts = MegatronBenchAdapter().discover(_megatron_model(num_layers=3))
    assert len(parts.layers) == 3
    assert parts.embedding is not None
    assert parts.output_layer is not None


def test_megatron_unwraps_ddp_wrappers():
    parts = MegatronBenchAdapter().discover(_megatron_model(num_layers=2, wrapped=True))
    assert len(parts.layers) == 2


def test_megatron_concatenates_pipeline_chunks_in_order():
    chunks = [_megatron_model(num_layers=2), _megatron_model(num_layers=3)]
    parts = MegatronBenchAdapter().discover(chunks)
    assert len(parts.layers) == 5


def test_megatron_missing_layers_is_an_error():
    with pytest.raises(ValueError, match="Cannot find transformer layers"):
        MegatronBenchAdapter().discover(SimpleNamespace())


def test_megatron_layer_halves_use_its_own_names():
    layer = _megatron_layer()
    parts = MegatronBenchAdapter().layer_submodules(layer)
    assert parts.attention is layer.self_attention
    assert parts.mlp is layer.mlp


@pytest.mark.parametrize("kind", [LAYER, MLP, MOE])
def test_megatron_hidden_states_are_sequence_first(kind):
    """Megatron carries activations as [S, B, D], not [B, S, D]."""
    inputs = MegatronBenchAdapter().inputs(kind, nn.Identity(), _ctx())
    assert list(inputs.args) == [(SEQ, BATCH, HIDDEN)]


def test_megatron_attention_gets_a_boolean_mask():
    inputs = MegatronBenchAdapter().inputs(ATTENTION, nn.Identity(), _ctx())
    hidden, mask = inputs.args
    assert hidden == (SEQ, BATCH, HIDDEN)
    assert mask == ((1, 1, SEQ, SEQ), torch.bool)


def test_megatron_context_parallel_shards_the_query_sequence():
    """Under CP a rank holds seq_len/cp of the query, attending to the whole key."""
    inputs = MegatronBenchAdapter().inputs(ATTENTION, nn.Identity(), _ctx(cp_size=4))
    _, mask = inputs.args
    assert mask == ((1, 1, SEQ // 4, SEQ), torch.bool)


def test_megatron_embedding_takes_token_ids():
    inputs = MegatronBenchAdapter().inputs(EMBEDDING, nn.Identity(), _ctx())
    assert list(inputs.args) == [((BATCH, SEQ), torch.int64)]


def test_megatron_moe_decomposition_needs_dispatch_and_combine():
    adapter = MegatronBenchAdapter()
    bare = nn.Identity()
    assert not adapter.supports_moe_decomposition(bare)

    decomposable = SimpleNamespace(dispatch=lambda *a: None, combine=lambda *a: None)
    assert adapter.supports_moe_decomposition(decomposable)


# ---------------------------------------------------------------------------
# TorchTitan conventions
# ---------------------------------------------------------------------------


class _TitanBlock(nn.Module):
    """Shaped like torchtitan's TransformerBlock: one of feed_forward / moe."""

    def __init__(self, moe: bool = False):
        super().__init__()
        self.attention = nn.Linear(HIDDEN, HIDDEN)
        if moe:
            self.moe = nn.Linear(HIDDEN, HIDDEN)
        else:
            self.feed_forward = nn.Linear(HIDDEN, HIDDEN)


class _TitanModel(nn.Module):
    """Shaped like torchtitan's Transformer: layers is a ModuleDict."""

    def __init__(self, num_layers=3, rope_attr="freqs_cis", first_moe=None):
        super().__init__()
        self.tok_embeddings = nn.Embedding(8, HIDDEN)
        self.layers = nn.ModuleDict(
            {str(i): _TitanBlock(moe=(first_moe is not None and i >= first_moe)) for i in range(num_layers)}
        )
        self.norm = nn.LayerNorm(HIDDEN)
        self.output = nn.Linear(HIDDEN, 8)
        if rope_attr:
            setattr(self, rope_attr, torch.zeros(SEQ, 8))


def test_torchtitan_discovers_layers_embedding_and_head():
    parts = TorchTitanBenchAdapter().discover(_TitanModel(num_layers=3))
    assert len(parts.layers) == 3
    assert isinstance(parts.embedding, nn.Embedding)
    assert isinstance(parts.output_layer, nn.Linear)


def test_torchtitan_orders_layers_numerically_not_lexically():
    """layers is keyed by global index as a string; '10' must not sort before '2'."""
    model = _TitanModel(num_layers=12)
    # Give each block a marker so order is observable.
    for key, block in model.layers.items():
        block.marker = int(key)

    parts = TorchTitanBenchAdapter().discover(model)
    assert [block.marker for block in parts.layers] == list(range(12))


def test_torchtitan_accepts_a_list_of_model_parts():
    parts = TorchTitanBenchAdapter().discover([_TitanModel(num_layers=2)])
    assert len(parts.layers) == 2


def test_torchtitan_missing_layers_is_an_error():
    with pytest.raises(ValueError, match="layers"):
        TorchTitanBenchAdapter().discover(nn.Identity())


def test_torchtitan_dense_block_mlp_is_feed_forward():
    block = _TitanBlock(moe=False)
    parts = TorchTitanBenchAdapter().layer_submodules(block)
    assert parts.attention is block.attention
    assert parts.mlp is block.feed_forward


def test_torchtitan_moe_block_mlp_is_moe():
    block = _TitanBlock(moe=True)
    parts = TorchTitanBenchAdapter().layer_submodules(block)
    assert parts.mlp is block.moe


@pytest.mark.parametrize("kind", [LAYER, ATTENTION])
def test_torchtitan_hidden_states_are_batch_first(kind):
    """TorchTitan carries activations as [B, S, D] -- the opposite of Megatron."""
    adapter = TorchTitanBenchAdapter()
    adapter.discover(_TitanModel())
    inputs = adapter.inputs(kind, nn.Linear(HIDDEN, HIDDEN), _ctx())
    assert inputs.args[0][0] == (BATCH, SEQ, HIDDEN)


def test_torchtitan_passes_the_models_own_rope_table():
    """The rope values decide whether the kernel is valid, so they are not faked."""
    adapter = TorchTitanBenchAdapter()
    model = _TitanModel(rope_attr="freqs_cis")
    adapter.discover(model)

    inputs = adapter.inputs(LAYER, nn.Linear(HIDDEN, HIDDEN), _ctx())
    rope = inputs.args[1]
    assert isinstance(rope, torch.Tensor)
    assert rope.shape == model.freqs_cis.shape


def test_torchtitan_finds_rope_under_either_name():
    """Qwen 3 and GPT-OSS call it rope_cache; the rest call it freqs_cis."""
    adapter = TorchTitanBenchAdapter()
    adapter.discover(_TitanModel(rope_attr="rope_cache"))
    inputs = adapter.inputs(LAYER, nn.Linear(HIDDEN, HIDDEN), _ctx())
    assert isinstance(inputs.args[1], torch.Tensor)


def test_torchtitan_layer_forward_gets_mask_and_positions():
    adapter = TorchTitanBenchAdapter()
    adapter.discover(_TitanModel())
    inputs = adapter.inputs(LAYER, nn.Linear(HIDDEN, HIDDEN), _ctx())
    # None selects the dense causal path the projection models.
    assert inputs.kwargs == {"attention_masks": None, "positions": None}


def test_torchtitan_input_dtype_follows_the_module_parameters():
    """TorchTitan builds some models in fp32 and reaches bf16 via autocast."""
    adapter = TorchTitanBenchAdapter()
    adapter.discover(_TitanModel())

    fp32 = nn.Linear(HIDDEN, HIDDEN).to(torch.float32)
    assert adapter.inputs(MLP, fp32, _ctx()).args[0][1] == torch.float32

    bf16 = nn.Linear(HIDDEN, HIDDEN).to(torch.bfloat16)
    assert adapter.inputs(MLP, bf16, _ctx()).args[0][1] == torch.bfloat16


def test_torchtitan_embedding_takes_token_ids():
    inputs = TorchTitanBenchAdapter().inputs(EMBEDDING, nn.Embedding(8, HIDDEN), _ctx())
    assert list(inputs.args) == [((BATCH, SEQ), torch.int64)]


def test_torchtitan_output_head_is_batch_first():
    inputs = TorchTitanBenchAdapter().inputs(OUTPUT, nn.Linear(HIDDEN, 8), _ctx())
    assert inputs.args[0][0] == (BATCH, SEQ, HIDDEN)


def test_torchtitan_declines_to_measure_attention_without_a_rope_table():
    """Better no measurement than a measurement of the wrong kernel."""
    adapter = TorchTitanBenchAdapter()
    adapter.discover(_TitanModel(rope_attr=None))
    assert adapter.inputs(LAYER, nn.Linear(HIDDEN, HIDDEN), _ctx()) is None


def test_torchtitan_moe_is_not_decomposed():
    """Expert all-to-all lives in the parallel style, not in methods on MoE."""
    module = SimpleNamespace(dispatch=lambda *a: None, combine=lambda *a: None)
    assert not TorchTitanBenchAdapter().supports_moe_decomposition(module)


# ---------------------------------------------------------------------------
# Benchmark write-back (flat projection fields -> TorchTitan's own config)
# ---------------------------------------------------------------------------


def _bench_shaped_args(**overrides):
    """A config as the performance driver leaves it: shrunk onto the bench node."""
    args = SimpleNamespace(
        framework="torchtitan",
        num_layers=2,
        moe_layer_freq=[0, 1],
        num_experts=8,
        tensor_model_parallel_size=2,
        context_model_parallel_size=1,
        expert_model_parallel_size=4,
        pipeline_model_parallel_size=1,
        micro_batch_size=3,
        seq_length=4096,
        recompute_granularity=None,
        parallelism=SimpleNamespace(
            tensor_parallel_degree=8,
            expert_parallel_degree=64,
            pipeline_parallel_degree=4,
            data_parallel_shard_degree=-1,
        ),
        training=SimpleNamespace(local_batch_size=1, seq_len=8192),
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_writeback_moves_parallel_degrees_into_torchtitan_namespaces():
    args = _bench_shaped_args()
    torchtitan_apply_bench_overrides(args)

    assert args.parallelism.tensor_parallel_degree == 2
    assert args.parallelism.expert_parallel_degree == 4
    assert args.parallelism.pipeline_parallel_degree == 1


def test_writeback_unshards_data_parallelism():
    """FSDP2/DDP wrap params in DTensors the layer bench cannot feed."""
    args = _bench_shaped_args()
    torchtitan_apply_bench_overrides(args)

    assert args.parallelism.data_parallel_shard_degree == 1
    assert args.parallelism.data_parallel_replicate_degree == 1


def test_writeback_moves_batch_shape_into_torchtitan_namespaces():
    args = _bench_shaped_args()
    torchtitan_apply_bench_overrides(args)

    assert args.training.local_batch_size == 3
    assert args.training.seq_len == 4096
    # One gradient accumulation step: the bench measures a single microbatch.
    assert args.training.global_batch_size == 3


def test_writeback_stages_the_layer_cap_as_a_model_args_override():
    """n_layers comes from the flavor registry, which no config field reaches."""
    args = _bench_shaped_args(num_layers=2)
    torchtitan_apply_bench_overrides(args)

    overrides = args.primus_projection.model_args_overrides
    assert overrides["n_layers"] == 2
    assert overrides["moe_args.num_experts"] == 8


def test_writeback_translates_a_dense_then_moe_stack():
    """[dense, moe] is DeepSeek's n_dense_layers=1 and Llama 4's stride of 2."""
    args = _bench_shaped_args(moe_layer_freq=[0, 1])
    torchtitan_apply_bench_overrides(args)

    overrides = args.primus_projection.model_args_overrides
    assert overrides["n_dense_layers"] == 1
    assert overrides["interleave_moe_layer_step"] == 2


def test_writeback_translates_an_all_moe_stack():
    args = _bench_shaped_args(moe_layer_freq=[1, 1])
    torchtitan_apply_bench_overrides(args)

    overrides = args.primus_projection.model_args_overrides
    assert overrides["n_dense_layers"] == 0
    assert overrides["interleave_moe_layer_step"] == 1


@pytest.mark.parametrize(
    "granularity,expected",
    [("full", "full"), ("selective", "selective")],
)
def test_writeback_translates_recompute_granularity(granularity, expected):
    args = _bench_shaped_args(recompute_granularity=granularity)
    torchtitan_apply_bench_overrides(args)
    assert args.activation_checkpoint.mode == expected


def test_writeback_creates_missing_namespaces():
    """A config that never mentioned activation_checkpoint still gets one."""
    args = SimpleNamespace(
        framework="torchtitan",
        num_layers=1,
        micro_batch_size=1,
        seq_length=1024,
        recompute_granularity="full",
    )
    torchtitan_apply_bench_overrides(args)

    assert args.activation_checkpoint.mode == "full"
    assert args.parallelism.tensor_parallel_degree == 1
    assert args.training.seq_len == 1024


# ---------------------------------------------------------------------------
# End-to-end: the profiler tree driving a backend's modules
# ---------------------------------------------------------------------------
#
# The measurement itself needs a GPU -- CUDA events, peak-memory stats, real
# kernels -- so these intercept the one call that touches silicon and assert on
# what the harness was about to measure: which module, with what shaped input.
# That is the part a backend gets wrong, and it is checkable on CPU.


@pytest.fixture
def recorded_benchmarks(monkeypatch):
    """Intercept every benchmark call, recording module and inputs instead."""
    calls = []

    def fake_benchmark_layer(module, input_shapes, **kwargs):
        calls.append(
            {
                "module": module,
                "input_shapes": list(input_shapes),
                "forward_kwargs": kwargs.get("forward_kwargs"),
            }
        )
        return (1.0, 2.0, 4096)

    for target in (
        "primus.core.projection.module_profilers.transformer_layer",
        "primus.core.projection.module_profilers.attention",
        "primus.core.projection.module_profilers.dense_mlp",
        "primus.core.projection.module_profilers.moe_mlp",
        "primus.core.projection.module_profilers.embedding",
        "primus.core.projection.module_profilers.output_layer",
    ):
        monkeypatch.setattr(f"{target}.benchmark_layer", fake_benchmark_layer)
    return calls


def _language_model_profiler(monkeypatch, num_layers=2):
    from primus.core.projection.module_profilers.language_model import (
        build_profiler,
        get_language_model_profiler_spec,
    )
    from primus.core.projection.training_config import (
        ModelConfig,
        ModelParallelConfig,
        RuntimeConfig,
        TrainingConfig,
    )

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("NNODES", "1")
    monkeypatch.setenv("GPUS_PER_NODE", "1")

    config = TrainingConfig(
        model_config=ModelConfig(
            num_layers=num_layers,
            hidden_size=HIDDEN,
            padded_vocab_size=8,
            ffn_hidden_size=HIDDEN * 2,
            num_attention_heads=8,
            kv_channels=16,
            num_query_groups=8,
            group_query_attention=False,
            swiglu=True,
            multi_latent_attention=False,
            use_flash_attn=True,
            num_experts=None,
            moe_pattern=[0] * num_layers,
        ),
        runtime_config=RuntimeConfig(
            global_batch_size=BATCH,
            micro_batch_size=BATCH,
            sequence_length=SEQ,
            data_parallel_size=1,
        ),
        model_parallel_config=ModelParallelConfig(),
    )
    return build_profiler(get_language_model_profiler_spec(config))


def test_torchtitan_model_drives_the_profiler_tree(monkeypatch, recorded_benchmarks):
    """A TorchTitan model should be measurable without touching the profilers."""
    profiler = _language_model_profiler(monkeypatch)
    profiler.set_bench_adapter(TorchTitanBenchAdapter())

    model = _TitanModel(num_layers=2)
    results = profiler.run_layer_benchmark(model=model, batch_size=BATCH, seq_len=SEQ)

    assert recorded_benchmarks, "nothing was benchmarked"
    # The layers, embedding and head all got measured.
    assert "embedding" in results and "output" in results
    assert 0 in results

    measured = {id(call["module"]) for call in recorded_benchmarks}
    assert id(model.tok_embeddings) in measured
    assert id(model.output) in measured
    block = model.layers["0"]
    assert id(block.attention) in measured
    assert id(block.feed_forward) in measured


def test_torchtitan_modules_are_fed_batch_first_inputs(monkeypatch, recorded_benchmarks):
    profiler = _language_model_profiler(monkeypatch)
    profiler.set_bench_adapter(TorchTitanBenchAdapter())
    model = _TitanModel(num_layers=2)

    profiler.run_layer_benchmark(model=model, batch_size=BATCH, seq_len=SEQ)

    hidden_state_calls = [
        call
        for call in recorded_benchmarks
        if call["input_shapes"] and call["input_shapes"][0] != ((BATCH, SEQ), torch.int64)
    ]
    assert hidden_state_calls
    for call in hidden_state_calls:
        assert call["input_shapes"][0][0] == (BATCH, SEQ, HIDDEN)


def test_megatron_model_still_gets_sequence_first_inputs(monkeypatch, recorded_benchmarks):
    """The default adapter must not have changed Megatron's conventions."""
    profiler = _language_model_profiler(monkeypatch)
    model = _megatron_model(num_layers=2)

    profiler.run_layer_benchmark(model=model, batch_size=BATCH, seq_len=SEQ)

    layer_calls = [call for call in recorded_benchmarks if call["input_shapes"] == [(SEQ, BATCH, HIDDEN)]]
    assert layer_calls, "expected [S, B, D] inputs for Megatron modules"


def test_a_backend_that_cannot_build_inputs_fails_loudly(monkeypatch, recorded_benchmarks):
    """Reporting an analytical estimate as a measurement would be worse."""
    profiler = _language_model_profiler(monkeypatch)
    adapter = TorchTitanBenchAdapter()
    profiler.set_bench_adapter(adapter)

    # No rope table -> the attention kernel cannot be fed truthfully.
    model = _TitanModel(num_layers=2, rope_attr=None)
    with pytest.raises(RuntimeError, match="cannot build benchmark inputs"):
        profiler.run_layer_benchmark(model=model, batch_size=BATCH, seq_len=SEQ)
