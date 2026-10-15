###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the projection's training-framework config adapters.

Covers the adapter registry and the TorchTitan adapter (flavor resolution, mesh
degrees, pipeline layout, precision, recompute), plus the idempotency the adapter
owes the performance driver.  Pure CPU, no backend checkouts.
"""

import argparse
from types import SimpleNamespace

import pytest

from primus.core.projection.frameworks import (
    NORMALIZED_FLAG,
    available_frameworks,
    framework_of,
    get_config_adapter,
    is_normalized,
    register_config_adapter,
    resolve_config_adapter,
)
from primus.core.projection.frameworks.model_specs import (
    BUILTIN_MODEL_SPECS,
    ModelSpec,
    llama_ffn_hidden_size,
    torchtitan_builtin_spec,
)
from primus.core.projection.frameworks.torchtitan import torchtitan_derive_default_args


@pytest.fixture(autouse=True)
def _single_node_env(monkeypatch):
    monkeypatch.setenv("NNODES", "1")
    monkeypatch.setenv("GPUS_PER_NODE", "8")


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


def test_registry_ships_every_supported_framework():
    names = available_frameworks()
    for expected in ("megatron", "torchtitan", "dlrm"):
        assert expected in names


def test_registry_lookup_is_case_insensitive():
    assert get_config_adapter("TorchTitan") is get_config_adapter("torchtitan")
    assert get_config_adapter("  TORCHTITAN  ") is get_config_adapter("torchtitan")


def test_unknown_framework_names_the_supported_ones():
    with pytest.raises(NotImplementedError) as excinfo:
        resolve_config_adapter("tensorflow")
    message = str(excinfo.value)
    assert "tensorflow" in message
    assert "torchtitan" in message


def test_out_of_tree_backends_can_register():
    register_config_adapter("my_backend", lambda args: args)
    try:
        assert "my_backend" in available_frameworks()
        sentinel = argparse.Namespace()
        assert resolve_config_adapter("MY_BACKEND")(sentinel) is sentinel
    finally:
        from primus.core.projection.frameworks import _CONFIG_ADAPTER_REGISTRY

        _CONFIG_ADAPTER_REGISTRY.pop("my_backend", None)


@pytest.mark.parametrize("bad", ["", None, 123])
def test_register_rejects_bad_names(bad):
    with pytest.raises(ValueError):
        register_config_adapter(bad, lambda args: args)


def test_register_rejects_non_callable():
    with pytest.raises(TypeError):
        register_config_adapter("not_callable", object())


def test_framework_defaults_to_megatron():
    assert framework_of(argparse.Namespace()) == "megatron"
    assert framework_of(argparse.Namespace(framework=None)) == "megatron"
    assert framework_of(argparse.Namespace(framework=" TorchTitan ")) == "torchtitan"


# --------------------------------------------------------------------------- #
# TorchTitan
# --------------------------------------------------------------------------- #


def _titan_args(**overrides):
    """A TorchTitan trainer namespace shaped like a loaded experiment YAML."""
    model = SimpleNamespace(name="llama3", flavor="8B", converters=[])
    parallelism = SimpleNamespace(
        tensor_parallel_degree=1,
        pipeline_parallel_degree=1,
        context_parallel_degree=1,
        expert_parallel_degree=1,
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=-1,
        pipeline_parallel_schedule="1F1B",
        pipeline_parallel_microbatch_size=1,
    )
    args = SimpleNamespace(
        framework="torchtitan",
        model=model,
        parallelism=parallelism,
        training=SimpleNamespace(local_batch_size=2, seq_len=8192, global_batch_size=-1),
        activation_checkpoint=SimpleNamespace(mode="none"),
        optimizer=SimpleNamespace(name="AdamW"),
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_torchtitan_resolves_the_llama3_8b_flavor():
    args = torchtitan_derive_default_args(_titan_args())

    assert args.num_layers == 32
    assert args.hidden_size == 4096
    assert args.ffn_hidden_size == 14336
    assert args.num_attention_heads == 32
    assert args.num_query_groups == 8
    assert args.group_query_attention is True
    assert args.kv_channels == 128
    assert args.padded_vocab_size == 128256
    # Llama 3 is dense, and the projection spells "dense" as a falsy num_experts.
    assert not args.num_experts


def test_torchtitan_reads_cross_entropy_fusion_off_the_compiled_components():
    """TorchTitan fuses softmax+CE by compiling the loss, not by a flag.

    Every shipped TorchTitan config compiles it, so assuming otherwise
    over-predicts the loss module on large-vocabulary models.
    """
    compile_cfg = SimpleNamespace(enable=True, components=["model", "loss"])
    args = torchtitan_derive_default_args(_titan_args(compile=compile_cfg))
    assert args.cross_entropy_loss_fusion is True


def test_torchtitan_compiling_only_the_model_leaves_cross_entropy_unfused():
    compile_cfg = SimpleNamespace(enable=True, components=["model"])
    args = torchtitan_derive_default_args(_titan_args(compile=compile_cfg))
    assert args.cross_entropy_loss_fusion is False


def test_torchtitan_cross_entropy_is_unfused_when_compile_is_off():
    compile_cfg = SimpleNamespace(enable=False, components=["model", "loss"])
    args = torchtitan_derive_default_args(_titan_args(compile=compile_cfg))
    assert args.cross_entropy_loss_fusion is False


def test_torchtitan_cross_entropy_is_unfused_without_a_compile_section():
    assert torchtitan_derive_default_args(_titan_args()).cross_entropy_loss_fusion is False


def test_torchtitan_reads_the_job_batch_and_sequence():
    args = torchtitan_derive_default_args(_titan_args())
    assert args.seq_length == 8192
    assert args.micro_batch_size == 2
    # No explicit global batch: one local batch per data-parallel rank.
    assert args.global_batch_size == 2 * args.data_parallel_size


def test_torchtitan_fills_the_leftover_ranks_into_dp_shard(monkeypatch):
    monkeypatch.setenv("GPUS_PER_NODE", "8")
    par = SimpleNamespace(
        tensor_parallel_degree=2,
        pipeline_parallel_degree=1,
        context_parallel_degree=2,
        expert_parallel_degree=1,
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=-1,
        pipeline_parallel_schedule="1F1B",
        pipeline_parallel_microbatch_size=1,
    )
    args = torchtitan_derive_default_args(_titan_args(parallelism=par))

    assert args.tensor_model_parallel_size == 2
    assert args.context_parallel_size == 2
    # 8 ranks, TP=2 and CP=2 consume 4 of them.
    assert args.data_parallel_size == 2
    assert args.use_torch_fsdp2 is True


def test_torchtitan_explicit_dp_shard_is_left_alone():
    par = SimpleNamespace(
        tensor_parallel_degree=1,
        pipeline_parallel_degree=1,
        context_parallel_degree=1,
        expert_parallel_degree=1,
        data_parallel_replicate_degree=2,
        data_parallel_shard_degree=1,
        pipeline_parallel_schedule="1F1B",
        pipeline_parallel_microbatch_size=1,
    )
    args = torchtitan_derive_default_args(_titan_args(parallelism=par))
    assert args.data_parallel_size == 2
    assert args.use_torch_fsdp2 is False


@pytest.mark.parametrize(
    "schedule,expected_vpp",
    [
        ("1F1B", 1),
        ("GPipe", 1),
        ("Interleaved1F1B", 2),
        ("ZBVZeroBubble", 2),
    ],
)
def test_torchtitan_infers_virtual_stages_from_the_schedule(schedule, expected_vpp):
    par = SimpleNamespace(
        tensor_parallel_degree=1,
        pipeline_parallel_degree=2,
        context_parallel_degree=1,
        expert_parallel_degree=1,
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=1,
        pipeline_parallel_schedule=schedule,
        pipeline_parallel_microbatch_size=1,
    )
    args = torchtitan_derive_default_args(_titan_args(parallelism=par))
    assert args.virtual_pipeline_model_parallel_size == expected_vpp


def test_torchtitan_marks_zero_bubble_schedules():
    par = SimpleNamespace(
        tensor_parallel_degree=1,
        pipeline_parallel_degree=2,
        context_parallel_degree=1,
        expert_parallel_degree=1,
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=1,
        pipeline_parallel_schedule="ZBVZeroBubble",
        pipeline_parallel_microbatch_size=1,
    )
    args = torchtitan_derive_default_args(_titan_args(parallelism=par))
    assert args.enable_zero_bubble is True


def test_torchtitan_pipeline_layout_charges_embedding_and_head_to_the_ends():
    par = SimpleNamespace(
        tensor_parallel_degree=1,
        pipeline_parallel_degree=4,
        context_parallel_degree=1,
        expert_parallel_degree=1,
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=1,
        pipeline_parallel_schedule="1F1B",
        pipeline_parallel_microbatch_size=1,
    )
    args = torchtitan_derive_default_args(_titan_args(parallelism=par))

    layout = args.pipeline_model_parallel_layout
    assert layout is not None
    # The embedding rides on the first stage and the head on the last, so those
    # stages hold fewer transformer layers than the middle ones.
    counts = [int(stage.split("t*")[1].split(",")[0]) for stage in layout.split("|")]
    assert sum(counts) == 32
    assert counts[0] < max(counts)
    assert counts[-1] < max(counts)
    assert layout.startswith("E")
    assert layout.endswith(",L")


def test_torchtitan_even_pipeline_split_needs_no_layout():
    par = SimpleNamespace(
        tensor_parallel_degree=1,
        pipeline_parallel_degree=1,
        context_parallel_degree=1,
        expert_parallel_degree=1,
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=1,
        pipeline_parallel_schedule="1F1B",
        pipeline_parallel_microbatch_size=1,
    )
    args = torchtitan_derive_default_args(_titan_args(parallelism=par))
    assert args.pipeline_model_parallel_layout is None


def test_torchtitan_pipelined_batch_is_the_microbatch():
    par = SimpleNamespace(
        tensor_parallel_degree=1,
        pipeline_parallel_degree=2,
        context_parallel_degree=1,
        expert_parallel_degree=1,
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=1,
        pipeline_parallel_schedule="1F1B",
        pipeline_parallel_microbatch_size=1,
    )
    args = torchtitan_derive_default_args(
        _titan_args(
            parallelism=par,
            training=SimpleNamespace(local_batch_size=8, seq_len=4096, global_batch_size=-1),
        )
    )
    # The local batch is fed to the pipeline as microbatches, and a stage only
    # ever holds one of them.
    assert args.micro_batch_size == 1


@pytest.mark.parametrize(
    "converters,expected_recipe",
    [
        ([], None),
        (["float8"], "tensorwise"),
        (["mxfp8"], "mxfp8"),
    ],
)
def test_torchtitan_precision_comes_from_the_converters(converters, expected_recipe):
    args = torchtitan_derive_default_args(
        _titan_args(model=SimpleNamespace(name="llama3", flavor="8B", converters=converters))
    )
    assert args.fp8_recipe == expected_recipe
    assert args.fp8 == ("hybrid" if expected_recipe else None)


def test_torchtitan_float8_recipe_name_is_honoured():
    args = _titan_args(model=SimpleNamespace(name="llama3", flavor="8B", converters=["float8"]))
    args.quantize = SimpleNamespace(linear=SimpleNamespace(float8=SimpleNamespace(recipe_name="rowwise")))
    args = torchtitan_derive_default_args(args)
    assert args.fp8_recipe == "rowwise"


@pytest.mark.parametrize(
    "mode,granularity,method",
    [
        ("none", None, None),
        ("full", "full", "uniform"),
        ("selective", "selective", None),
    ],
)
def test_torchtitan_activation_checkpoint_maps_to_recompute(mode, granularity, method):
    args = torchtitan_derive_default_args(_titan_args(activation_checkpoint=SimpleNamespace(mode=mode)))
    assert args.recompute_granularity == granularity
    assert args.recompute_method == method
    assert args.recompute_num_layers == (32 if granularity == "full" else 0)


def test_torchtitan_moe_flavor_carries_its_expert_shape():
    args = torchtitan_derive_default_args(
        _titan_args(model=SimpleNamespace(name="deepseek_v3", flavor="671B", converters=[]))
    )
    assert args.num_experts == 256
    assert args.moe_router_topk == 8
    assert args.multi_latent_attention is True
    # DeepSeek V3 runs three dense layers before the MoE stack starts.
    assert args.moe_layer_freq[:3] == [0, 0, 0]
    assert set(args.moe_layer_freq[3:]) == {1}


def test_torchtitan_yaml_model_overrides_beat_the_flavor_table():
    model = SimpleNamespace(name="deepseek_v3", flavor="671B", converters=[], n_layers=4)
    args = torchtitan_derive_default_args(_titan_args(model=model))

    assert args.num_layers == 4
    # A shortened stack has to shorten the MoE pattern with it, or the profiler
    # would index past the end of it.
    assert len(args.moe_layer_freq) == 4
    assert args.moe_layer_freq == [0, 0, 0, 1]


def test_torchtitan_unknown_flavor_says_what_to_do():
    with pytest.raises(ValueError) as excinfo:
        torchtitan_derive_default_args(
            _titan_args(model=SimpleNamespace(name="llama3", flavor="9000B", converters=[]))
        )
    assert "9000B" in str(excinfo.value)
    assert "model_specs" in str(excinfo.value)


def test_torchtitan_turbo_flags_are_read():
    args = _titan_args()
    args.primus_turbo = SimpleNamespace(enable_primus_turbo=True, use_turbo_grouped_gemm=True)
    args.parallelism.expert_parallel_comm_backend = "deepep"
    args = torchtitan_derive_default_args(args)

    assert args.enable_primus_turbo is True
    assert args.use_turbo_grouped_gemm is True
    assert args.use_turbo_grouped_mlp is True
    assert args.use_turbo_deepep is True


# --------------------------------------------------------------------------- #
# Idempotency: the performance driver edits the normalized config and reconverts
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "adapter,make_args",
    [
        (torchtitan_derive_default_args, _titan_args),
    ],
)
def test_adapters_normalize_in_place_and_mark_the_namespace(adapter, make_args):
    args = make_args()
    assert not is_normalized(args)

    returned = adapter(args)
    # The driver keeps a handle on the namespace it passed in, so an adapter that
    # returned a fresh object would strand every edit the driver makes later.
    assert returned is args
    assert getattr(args, NORMALIZED_FLAG) is True


@pytest.mark.parametrize(
    "adapter,make_args",
    [
        (torchtitan_derive_default_args, _titan_args),
    ],
)
def test_second_pass_keeps_the_drivers_edits(adapter, make_args):
    args = adapter(make_args())
    assert args.num_layers == 32

    # Stand in for what the performance driver does between conversions: cap the
    # stack, flatten the pipeline, shrink the mesh onto the bench node.
    args.num_layers = 2
    args.moe_layer_freq = [0, 0]
    args.pipeline_model_parallel_size = 1
    args.tensor_model_parallel_size = 1

    args = adapter(args)

    assert args.num_layers == 2
    assert args.pipeline_model_parallel_size == 1
    assert args.tensor_model_parallel_size == 1


def test_performance_driver_layer_limiting_survives_reconversion():
    from primus.core.projection.performance_projection.projection import (
        _limit_layers_for_projection,
    )

    args = torchtitan_derive_default_args(
        _titan_args(model=SimpleNamespace(name="deepseek_v3", flavor="671B", converters=[]))
    )
    full_layers = args.num_layers
    assert full_layers > 8

    _limit_layers_for_projection(args)
    limited = args.num_layers
    assert limited < full_layers

    # This is the conversion the driver does after limiting.  Re-deriving the
    # architecture here would project the full 61-layer model instead.
    args = torchtitan_derive_default_args(args)
    assert args.num_layers == limited
    assert len(args.moe_layer_freq) == limited


def test_alias_tables_only_point_at_specs_that_exist():
    from primus.core.projection.frameworks.model_specs import TORCHTITAN_FLAVOR_ALIASES

    for key, canonical in TORCHTITAN_FLAVOR_ALIASES.items():
        assert torchtitan_builtin_spec(*key) is not None, key
        assert canonical in BUILTIN_MODEL_SPECS


def test_llama_ffn_width_truncates_the_way_the_backends_do():
    # Llama 3 8B: the int() before the multiplier is what gives 14336, and the
    # 70B's 1.3 multiplier is what gives 28672.
    assert llama_ffn_hidden_size(4096, 1024, 1.3) == 14336
    assert llama_ffn_hidden_size(8192, 4096, 1.3) == 28672


@pytest.mark.parametrize("name,spec", sorted(BUILTIN_MODEL_SPECS.items()))
def test_every_builtin_spec_is_self_consistent(name, spec):
    assert isinstance(spec, ModelSpec)
    assert spec.num_layers > 0
    assert spec.hidden_size > 0
    assert spec.ffn_hidden_size > 0
    assert spec.num_attention_heads > 0
    assert spec.vocab_size > 0
    assert spec.num_attention_heads % spec.num_query_groups == 0
    assert len(spec.moe_layer_pattern()) == spec.num_layers

    if spec.num_experts:
        assert spec.moe_ffn_hidden_size > 0
        assert 0 < spec.moe_router_topk <= spec.num_experts
        # A MoE model has to have at least one MoE layer, or it is just dense
        # with a router attached.
        assert any(spec.moe_layer_pattern())
    if spec.multi_latent_attention:
        assert spec.kv_lora_rank > 0
        assert spec.qk_head_dim > 0
        assert spec.v_head_dim > 0
