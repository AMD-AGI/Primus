###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the projection's training-framework config adapters.

Covers the adapter registry, the TorchTitan adapter (flavor resolution, mesh
degrees, pipeline layout, precision, recompute) and the MaxText/JAX adapter
(ICI/DCN axis fill, batch derivation, remat, quantization), plus the idempotency
both adapters owe the performance driver.  Pure CPU, no backend checkouts.
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
from primus.core.projection.frameworks.jax import maxtext_derive_default_args
from primus.core.projection.frameworks.model_specs import (
    BUILTIN_MODEL_SPECS,
    ModelSpec,
    llama_ffn_hidden_size,
    maxtext_builtin_spec,
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
    for expected in ("megatron", "torchtitan", "maxtext", "jax", "dlrm"):
        assert expected in names


def test_registry_lookup_is_case_insensitive():
    assert get_config_adapter("TorchTitan") is get_config_adapter("torchtitan")
    assert get_config_adapter("  JAX  ") is get_config_adapter("jax")


def test_maxtext_and_jax_share_one_adapter():
    assert get_config_adapter("jax") is get_config_adapter("maxtext")


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
# MaxText / JAX
# --------------------------------------------------------------------------- #


def _maxtext_args(**overrides):
    """A MaxText trainer namespace shaped like a loaded experiment YAML."""
    args = argparse.Namespace(
        framework="maxtext",
        model_name="llama3-8b",
        per_device_batch_size=2,
        max_target_length=8192,
        remat_policy="full",
        attention="flash",
        opt_type="adamw",
        quantization="",
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


@pytest.fixture
def _no_maxtext_checkout(monkeypatch):
    """Resolve models from the transcribed table, not from a local checkout."""
    monkeypatch.setattr(
        "primus.core.projection.frameworks.jax._read_maxtext_model_config",
        lambda model_name: None,
    )


def test_maxtext_resolves_llama3_8b(_no_maxtext_checkout):
    args = maxtext_derive_default_args(_maxtext_args())

    assert args.num_layers == 32
    assert args.hidden_size == 4096
    assert args.ffn_hidden_size == 14336
    assert args.num_attention_heads == 32
    assert args.num_query_groups == 8
    assert args.padded_vocab_size == 128256
    assert args.seq_length == 8192


def test_maxtext_fsdp_absorbs_the_unspecified_ranks(_no_maxtext_checkout):
    # base.yml leaves ici_fsdp at -1, so it takes whatever the node has left.
    args = maxtext_derive_default_args(_maxtext_args(ici_tensor_parallelism=2))

    assert args.tensor_model_parallel_size == 2
    assert args.data_parallel_size == 4
    assert args.use_torch_fsdp2 is True


def test_maxtext_folds_ici_and_dcn_into_one_degree(monkeypatch, _no_maxtext_checkout):
    monkeypatch.setenv("NNODES", "4")
    monkeypatch.setenv("GPUS_PER_NODE", "8")
    args = maxtext_derive_default_args(
        _maxtext_args(
            ici_tensor_parallelism=8,
            ici_fsdp_parallelism=1,
            dcn_pipeline_parallelism=2,
            dcn_data_parallelism=2,
        )
    )
    assert args.tensor_model_parallel_size == 8
    assert args.pipeline_model_parallel_size == 2
    assert args.data_parallel_size == 2


def test_maxtext_tensor_sequence_counts_as_tensor_parallel(_no_maxtext_checkout):
    args = maxtext_derive_default_args(
        _maxtext_args(ici_tensor_parallelism=2, ici_tensor_sequence_parallelism=2)
    )
    assert args.tensor_model_parallel_size == 4


def test_maxtext_context_axes_fold_into_cp(_no_maxtext_checkout):
    args = maxtext_derive_default_args(_maxtext_args(ici_context_parallelism=2, ici_sequence_parallelism=2))
    assert args.context_parallel_size == 4


def test_maxtext_rejects_two_unspecified_axes_in_a_group(_no_maxtext_checkout):
    with pytest.raises(ValueError) as excinfo:
        maxtext_derive_default_args(_maxtext_args(ici_fsdp_parallelism=-1, ici_tensor_parallelism=-1))
    assert "at most one" in str(excinfo.value)


def test_maxtext_rejects_a_mesh_that_does_not_divide(_no_maxtext_checkout):
    with pytest.raises(ValueError) as excinfo:
        maxtext_derive_default_args(_maxtext_args(ici_tensor_parallelism=3))
    assert "do not divide evenly" in str(excinfo.value)


def test_maxtext_batch_is_per_device_times_the_mesh(_no_maxtext_checkout):
    # 8 devices x 2 per device = 16 sequences a step; TP=2 mirrors the batch, so
    # each of the 4 data-parallel ranks carries 4 of them.
    args = maxtext_derive_default_args(_maxtext_args(per_device_batch_size=2, ici_tensor_parallelism=2))
    assert args.micro_batch_size == 4
    assert args.global_batch_size == 16


def test_maxtext_gradient_accumulation_scales_the_global_batch(_no_maxtext_checkout):
    args = maxtext_derive_default_args(_maxtext_args(per_device_batch_size=1, gradient_accumulation_steps=4))
    assert args.global_batch_size == 32


@pytest.mark.parametrize(
    "policy,granularity,method",
    [
        ("full", "full", "uniform"),
        ("save_all", None, None),
        ("minimal", "selective", None),
        ("save_dot_except_mlp", "selective", None),
    ],
)
def test_maxtext_remat_policy_maps_to_recompute(policy, granularity, method, _no_maxtext_checkout):
    args = maxtext_derive_default_args(_maxtext_args(remat_policy=policy))
    assert args.recompute_granularity == granularity
    assert args.recompute_method == method


@pytest.mark.parametrize("quantization", ["fp8", "nanoo_fp8"])
def test_maxtext_quantization_turns_on_fp8(quantization, _no_maxtext_checkout):
    args = maxtext_derive_default_args(_maxtext_args(quantization=quantization))
    assert args.fp8 == "hybrid"
    assert args.fp8_recipe == "tensorwise"


def test_maxtext_bf16_leaves_fp8_off(_no_maxtext_checkout):
    args = maxtext_derive_default_args(_maxtext_args(quantization=""))
    assert args.fp8 is None


def test_maxtext_moe_model_carries_its_expert_shape(_no_maxtext_checkout):
    args = maxtext_derive_default_args(_maxtext_args(model_name="mixtral-8x7b"))
    assert args.num_experts == 8
    assert args.moe_router_topk == 2


def test_maxtext_experiment_may_spell_the_architecture_out(_no_maxtext_checkout):
    # Grok-1 is shipped this way: MaxText has no config for it, so the Primus
    # experiment carries the architecture itself.
    args = maxtext_derive_default_args(
        _maxtext_args(
            model_name="grok-1",
            base_num_decoder_layers=64,
            base_emb_dim=6144,
            base_mlp_dim=32768,
            base_num_query_heads=48,
            base_num_kv_heads=8,
            head_dim=128,
            vocab_size=131072,
            num_experts=8,
            num_experts_per_tok=2,
        )
    )
    assert args.num_layers == 64
    assert args.hidden_size == 6144
    assert args.num_experts == 8
    assert args.moe_router_topk == 2


def test_maxtext_reads_a_real_checkout_when_there_is_one(tmp_path, monkeypatch):
    models = tmp_path / "configs" / "models"
    models.mkdir(parents=True)
    (models / "tiny.yml").write_text(
        "base_num_decoder_layers: 6\n"
        "base_emb_dim: 512\n"
        "base_mlp_dim: 1024\n"
        "base_num_query_heads: 8\n"
        "base_num_kv_heads: 4\n"
        "head_dim: 64\n"
        "vocab_size: 1000\n"
    )
    monkeypatch.setenv("PRIMUS_MAXTEXT_PATH", str(tmp_path))

    args = maxtext_derive_default_args(_maxtext_args(model_name="tiny"))
    assert args.num_layers == 6
    assert args.hidden_size == 512
    assert args.ffn_hidden_size == 1024
    assert args.num_query_groups == 4
    assert args.padded_vocab_size == 1000


def test_maxtext_unknown_model_says_what_to_do(_no_maxtext_checkout):
    with pytest.raises(ValueError) as excinfo:
        maxtext_derive_default_args(_maxtext_args(model_name="no-such-model"))
    message = str(excinfo.value)
    assert "no-such-model" in message
    assert "PRIMUS_MAXTEXT_PATH" in message


def test_maxtext_rejects_global_parameter_scale(_no_maxtext_checkout):
    with pytest.raises(ValueError) as excinfo:
        maxtext_derive_default_args(_maxtext_args(global_parameter_scale=4))
    assert "global_parameter_scale" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Idempotency: the performance driver edits the normalized config and reconverts
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "adapter,make_args",
    [
        (torchtitan_derive_default_args, _titan_args),
        (maxtext_derive_default_args, _maxtext_args),
    ],
)
def test_adapters_normalize_in_place_and_mark_the_namespace(adapter, make_args, monkeypatch):
    monkeypatch.setattr(
        "primus.core.projection.frameworks.jax._read_maxtext_model_config",
        lambda model_name: None,
    )
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
        (maxtext_derive_default_args, _maxtext_args),
    ],
)
def test_second_pass_keeps_the_drivers_edits(adapter, make_args, monkeypatch):
    monkeypatch.setattr(
        "primus.core.projection.frameworks.jax._read_maxtext_model_config",
        lambda model_name: None,
    )
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


# --------------------------------------------------------------------------- #
# Cross-backend agreement
# --------------------------------------------------------------------------- #


def test_the_same_model_projects_the_same_shape_on_either_backend(_no_maxtext_checkout):
    titan = torchtitan_derive_default_args(_titan_args())
    maxtext = maxtext_derive_default_args(_maxtext_args())

    for field in (
        "num_layers",
        "hidden_size",
        "ffn_hidden_size",
        "num_attention_heads",
        "num_query_groups",
        "kv_channels",
        "padded_vocab_size",
    ):
        assert getattr(titan, field) == getattr(maxtext, field), field


def test_alias_tables_only_point_at_specs_that_exist():
    from primus.core.projection.frameworks.model_specs import (
        MAXTEXT_MODEL_ALIASES,
        TORCHTITAN_FLAVOR_ALIASES,
    )

    for key, canonical in TORCHTITAN_FLAVOR_ALIASES.items():
        assert torchtitan_builtin_spec(*key) is not None, key
        assert canonical in BUILTIN_MODEL_SPECS
    for model_name, canonical in MAXTEXT_MODEL_ALIASES.items():
        assert maxtext_builtin_spec(model_name) is not None, model_name
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
