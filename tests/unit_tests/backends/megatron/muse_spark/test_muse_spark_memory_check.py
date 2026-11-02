###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fail-fast memory estimate for Muse Spark layouts (pure Python)."""

from types import SimpleNamespace

import pytest

from primus.backends.megatron.core.models.muse_spark.memory_check import (
    DEFAULT_BUDGET,
    DEFAULT_MAX_NODES,
    RAISE_FRACTION,
    WARN_FRACTION,
    _stage_params,
    param_parts,
    param_split,
    stage_layers,
    state_bytes_per_gpu,
    suggest_layout,
)

GIB = 2**30
FULL = dict(
    hidden_size=8192,
    num_layers=62,
    kv_channels=64,
    num_attention_heads=128,
    num_query_groups=16,
    moe_latent_size=4096,
    num_experts=256,
    moe_ffn_hidden_size=4096,
    attention_output_gate=True,
    untie_embeddings_and_output_weights=True,
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    expert_model_parallel_size=8,
    use_distributed_optimizer=True,
)


def _a(**kw):
    return SimpleNamespace(**{**FULL, **kw})


def test_full_model_count_is_816b():
    assert 800e9 < sum(param_split(_a())) < 840e9


def test_single_node_defaults_cannot_fit_and_documented_layout_can():
    assert state_bytes_per_gpu(_a(), 8) > 1000 * GIB  # example defaults: TP1 PP1 EP8 on one node
    multi = state_bytes_per_gpu(_a(expert_model_parallel_size=64, pipeline_model_parallel_size=2), 128)
    assert multi < 200 * GIB  # 16 MI355X nodes, EP64 PP2 (288 GB GPUs)


def test_gate_sized_runs_pass():
    # 4 layers, 16 experts, 1 GPU: measured 134 GB max allocated on MI355X (round 2 proxy log)
    one = state_bytes_per_gpu(_a(num_layers=4, num_experts=16, expert_model_parallel_size=1), 1)
    assert 100 * GIB < one < 150 * GIB


def test_fsdp_shards_weights_and_grads():
    base = dict(num_layers=4, num_experts=16, expert_model_parallel_size=1, use_distributed_optimizer=False)
    plain = state_bytes_per_gpu(_a(**base), 2)
    fsdp = state_bytes_per_gpu(_a(**base, use_torch_fsdp2=True), 2)
    assert fsdp < plain / 1.9


def test_required_state_bytes_or_raise_raises_with_the_layout_hint():
    import pytest

    from primus.backends.megatron.patches.muse_spark_memory_check_patches import (
        required_state_bytes_or_raise,
    )

    with pytest.raises(ValueError, match=r">= 16 node\(s\).*PRIMUS_PP=2 PRIMUS_EP=64"):
        required_state_bytes_or_raise(_a(), 8, 288 * 10**9)  # MI355X
    with pytest.raises(ValueError, match=r">= 16 node\(s\).*PRIMUS_TP=8 PRIMUS_PP=2 PRIMUS_EP=64"):
        required_state_bytes_or_raise(_a(), 8, 192 * 10**9)  # MI300X: the hint follows the GPU memory
    assert (
        required_state_bytes_or_raise(
            _a(num_layers=4, num_experts=16, expert_model_parallel_size=1), 1, 288 * 10**9
        )
        > 0
    )


def test_context_parallel_does_not_raise_the_estimate():
    # the distributed optimizer / FSDP shard dense states over DP x CP (with_context_parallel=True)
    base = dict(expert_model_parallel_size=64, pipeline_model_parallel_size=2)
    cp1 = state_bytes_per_gpu(_a(**base), 128)
    cp8 = state_bytes_per_gpu(_a(**base, context_parallel_size=8), 128)
    assert cp8 == cp1


def test_uses_the_padded_vocab_from_args():
    h2 = 2 * 8192
    small = param_split(_a(padded_vocab_size=32768))[0]
    padded = param_split(_a(padded_vocab_size=202240))[0]  # an explicit, larger padded vocab
    assert padded - small == (202240 - 32768) * h2
    assert param_split(_a())[0] == param_split(_a(padded_vocab_size=202048))[0]  # default: released vocab


def test_missing_expert_ffn_size_does_not_pass_vacuously():
    import pytest

    # Megatron's fallback: moe_ffn_hidden_size unset -> ffn_hidden_size
    assert param_split(_a(moe_ffn_hidden_size=None, ffn_hidden_size=4096)) == param_split(_a())
    with pytest.raises(ValueError, match="moe_ffn_hidden_size"):
        param_split(_a(moe_ffn_hidden_size=None))


def test_suggested_layouts_fit_with_room_for_activations():
    for total in (288 * 10**9, 192 * 10**9):
        hint = suggest_layout(_a(), total)
        assert hint and hint["bytes"] <= DEFAULT_BUDGET * total


def test_no_phantom_latent_projections():
    # without moe_latent_size there are no pre/post expert projections (experts at width h)
    h, L = 8192, 62
    dense_latent = param_split(_a())[0]
    dense_plain = param_split(_a(moe_latent_size=None))[0]
    assert dense_latent - dense_plain == L * 2 * h * 4096
    assert param_split(_a(moe_latent_size=None))[1] == L * 256 * 3 * h * 4096


def test_output_gate_and_tied_head_are_counted_only_when_set():
    h, L, q = 8192, 62, 128 * 64
    assert param_split(_a())[0] - param_split(_a(attention_output_gate=False))[0] == L * h * q
    no_flags = {
        k: v
        for k, v in FULL.items()
        if k not in ("attention_output_gate", "untie_embeddings_and_output_weights")
    }
    tied = param_split(SimpleNamespace(**no_flags))[0]  # Megatron defaults: no gate, tied head
    assert param_split(_a())[0] - tied == L * h * q + 202048 * h


def test_optimizer_args_change_the_bytes():
    base = dict(num_layers=4, num_experts=16, expert_model_parallel_size=1, use_distributed_optimizer=False)
    adam = state_bytes_per_gpu(_a(**base), 1)
    n = sum(param_split(_a(**base)))
    assert abs(adam - 18 * n) < 1  # bf16 weight + fp32 grad + fp32 master + 2 fp32 moments
    assert abs(state_bytes_per_gpu(_a(**base, optimizer="sgd"), 1) - 14 * n) < 1
    half = dict(
        use_precision_aware_optimizer=True,
        main_params_dtype="fp32",
        exp_avg_dtype="bf16",
        exp_avg_sq_dtype="bf16",
    )
    assert abs(state_bytes_per_gpu(_a(**base, **half), 1) - 14 * n) < 1
    # dtypes are ignored without the precision-aware optimizer
    assert state_bytes_per_gpu(_a(**base, exp_avg_dtype="bf16"), 1) == adam
    off = state_bytes_per_gpu(_a(**base, optimizer_cpu_offload=True, optimizer_offload_fraction=0.5), 1)
    assert abs(off - 12 * n) < 1
    full_off = state_bytes_per_gpu(_a(**base, optimizer_cpu_offload=True, optimizer_offload_fraction=1.0), 1)
    assert abs(full_off - 6 * n) < 1
    assert abs(state_bytes_per_gpu(_a(**base, main_grads_dtype="bf16"), 1) - 16 * n) < 1


def test_counts_param_keeps_the_result():
    a = _a(expert_model_parallel_size=64, pipeline_model_parallel_size=2)
    assert state_bytes_per_gpu(a, 128, parts=param_parts(a)) == state_bytes_per_gpu(a, 128)


def test_suggested_tp_follows_gpus_per_node():
    hint = suggest_layout(_a(), 192 * 10**9, n_gpus_per_node=4)
    assert hint and hint["tp"] in (1, 4) and hint["world"] == hint["nodes"] * 4
    one = suggest_layout(_a(num_layers=4, num_experts=16), 288 * 10**9, n_gpus_per_node=1)
    assert one and one["tp"] == 1


def test_messages_come_from_the_constants():
    import pytest

    from primus.backends.megatron.patches.muse_spark_memory_check_patches import (
        required_state_bytes_or_raise,
    )

    huge = _a(hidden_size=8192 * 64, moe_latent_size=4096 * 64)
    msg = rf"No layout up to {DEFAULT_MAX_NODES} nodes .* under {DEFAULT_BUDGET:.0%} of a GPU"
    with pytest.raises(ValueError, match=msg):
        required_state_bytes_or_raise(huge, 8, 288 * 10**9)
    assert 0 < DEFAULT_BUDGET < WARN_FRACTION < 1


# ---- the before_train patch itself (torch.cuda / get_args / log_rank_0 stubbed) -----------


def _patch_env(monkeypatch, *, total, args, gpus=8, cuda=True):
    import torch

    from primus.backends.megatron.patches import muse_spark_memory_check_patches as mod

    calls = {"props": 0, "logs": []}

    def props(_dev):
        calls["props"] += 1
        return SimpleNamespace(total_memory=total)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: gpus)
    monkeypatch.setattr(torch.cuda, "get_device_properties", props)
    monkeypatch.setattr(mod, "get_args", lambda _ctx: args)
    monkeypatch.setattr(mod, "log_rank_0", lambda msg, *a, **k: calls["logs"].append(msg))
    monkeypatch.delenv("GPUS_PER_NODE", raising=False)
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    return mod, calls


SMALL = dict(num_layers=4, num_experts=16, expert_model_parallel_size=1)


@pytest.mark.parametrize("value", [True, "1", "true", "TRUE", "yes", " on "])
def test_patch_skip_arg_does_nothing(monkeypatch, value):
    args = _a()
    setattr(args, "muse_spark_skip_memory_check", value)
    mod, calls = _patch_env(monkeypatch, total=1, args=args)  # would raise if checked
    mod.patch_muse_spark_memory_check(None)
    assert calls == {"props": 0, "logs": []}


@pytest.mark.parametrize(
    "value, warned",
    [
        (False, False),
        (None, False),
        ("0", False),
        ("off", False),
        ("no", False),
        ("", False),
        ("maybe", True),
        ("2", True),
    ],
)
def test_patch_skip_arg_false_or_unknown_runs_the_check(monkeypatch, value, warned):
    args = _a()
    setattr(args, "muse_spark_skip_memory_check", value)
    mod, calls = _patch_env(monkeypatch, total=1, args=args)
    with pytest.raises(ValueError, match="muse_spark_skip_memory_check: true skips this check"):
        mod.patch_muse_spark_memory_check(None)
    assert any("is not a boolean" in m for m in calls["logs"]) == warned


def test_the_error_gives_the_cut_advice_once():
    from primus.backends.megatron.patches.muse_spark_memory_check_patches import (
        required_state_bytes_or_raise,
    )

    with pytest.raises(ValueError) as err:
        required_state_bytes_or_raise(_a(), 8, 288 * 10**9)
    assert str(err.value).count("cut the model") == 1
    huge = _a(hidden_size=8192 * 64, moe_latent_size=4096 * 64)
    with pytest.raises(ValueError) as err:
        required_state_bytes_or_raise(huge, 8, 288 * 10**9)
    assert str(err.value).count("cut the model") == 1


def test_patch_without_a_gpu_does_nothing(monkeypatch):
    mod, calls = _patch_env(monkeypatch, total=1, args=_a(), cuda=False)
    mod.patch_muse_spark_memory_check(None)
    assert calls == {"props": 0, "logs": []}


def test_patch_reads_world_size_from_the_env(monkeypatch):
    import pytest

    need1 = state_bytes_per_gpu(_a(**SMALL), 1)
    need2 = state_bytes_per_gpu(_a(**SMALL), 2)
    assert need2 < need1
    # memory between the 2-GPU and the 1-GPU need: passes only with world size 2
    total = (need1 + need2) // 2
    args = _a(**SMALL, world_size=None)
    mod, calls = _patch_env(monkeypatch, total=total, args=args)
    monkeypatch.setenv("WORLD_SIZE", "2")
    mod.patch_muse_spark_memory_check(None)
    assert len(calls["logs"]) == 1
    monkeypatch.setenv("WORLD_SIZE", "1")
    with pytest.raises(ValueError, match="world=1"):
        mod.patch_muse_spark_memory_check(None)
    # args.world_size wins over WORLD_SIZE
    args.world_size = 2
    mod.patch_muse_spark_memory_check(None)
    assert len(calls["logs"]) == 2


def test_patch_warns_when_little_room_is_left(monkeypatch):
    need = state_bytes_per_gpu(_a(**SMALL), 1)
    near_full = int(need / ((WARN_FRACTION + RAISE_FRACTION) / 2))  # between WARN and RAISE
    mod, calls = _patch_env(monkeypatch, total=near_full, args=_a(**SMALL, world_size=1))
    mod.patch_muse_spark_memory_check(None)
    assert "WARNING" in calls["logs"][-1]
    mod, calls = _patch_env(monkeypatch, total=need * 10, args=_a(**SMALL, world_size=1))
    mod.patch_muse_spark_memory_check(None)
    assert "WARNING" not in calls["logs"][-1]


def test_patch_uses_gpus_per_node_for_the_hint(monkeypatch):
    import pytest

    mod, _ = _patch_env(monkeypatch, total=288 * 10**9, args=_a(world_size=8), gpus=8)
    monkeypatch.setenv("GPUS_PER_NODE", "4")
    with pytest.raises(ValueError, match="node\\(s\\) of 4 such GPUs"):
        mod.patch_muse_spark_memory_check(None)


@pytest.mark.parametrize("raw", ["abc", "0", "-2", "4.5"])
def test_patch_falls_back_to_the_device_count_for_a_bad_gpus_per_node(monkeypatch, raw):
    mod, calls = _patch_env(monkeypatch, total=288 * 10**9, args=_a(world_size=8), gpus=2)
    monkeypatch.setenv("GPUS_PER_NODE", raw)
    with pytest.raises(ValueError, match="node\\(s\\) of 2 such GPUs"):
        mod.patch_muse_spark_memory_check(None)
    assert any("ignoring GPUS_PER_NODE" in m for m in calls["logs"])


def test_embedding_and_output_layer_are_not_spread_over_stages():
    """The most loaded stage holds the word embedding (first) or the untied output layer
    (last); dividing them by PP would undercount by ~PP x vocab x h on those stages."""
    # stage 0 gets 1 layer, the others 2: the last stage (2 layers + head) is strictly the
    # heaviest, so the head's placement decides the max
    a = _a(
        num_layers=7,
        num_experts=0,
        moe_latent_size=None,
        expert_model_parallel_size=1,
        pipeline_model_parallel_size=4,
        decoder_first_pipeline_num_layers=1,
        use_distributed_optimizer=False,
    )
    parts = param_parts(a)
    assert stage_layers(a) == [1, 2, 2, 2]
    ld, emb, out = parts["layer_dense"], parts["embedding"], parts["output"]
    assert out > 0
    assert [d for d, _ in _stage_params(a, parts)] == [ld + emb, 2 * ld, 2 * ld, 2 * ld + out]
    wg_opt = 2 + 4 + 12  # bf16 weight, fp32 grad, Adam fp32 states (no distributed optimizer)
    expect = (2 * parts["layer_dense"] + parts["output"]) * wg_opt  # last stage: 2 layers + head
    assert expect > (ld + emb) * wg_opt
    assert state_bytes_per_gpu(a, 4) == expect
    dense, _ = param_split(a)
    assert state_bytes_per_gpu(a, 4) > dense / 4 * wg_opt  # the old even split was lower


def test_tied_embedding_is_copied_to_the_last_stage():
    a = _a(
        num_layers=8,
        num_experts=0,
        moe_latent_size=None,
        expert_model_parallel_size=1,
        untie_embeddings_and_output_weights=False,
        use_distributed_optimizer=False,
    )
    parts = param_parts(a)
    assert parts["output"] == 0
    ld, emb = parts["layer_dense"], parts["embedding"]
    # stage 0 made lighter (1 layer, the others 2): the last stage (2 layers + the tied copy) is strictly the heaviest
    a4 = _a(
        **{
            **vars(a),
            "num_layers": 7,
            "pipeline_model_parallel_size": 4,
            "decoder_first_pipeline_num_layers": 1,
        }
    )
    assert [d for d, _ in _stage_params(a4, parts)] == [ld + emb, 2 * ld, 2 * ld, 2 * ld + emb]
    assert state_bytes_per_gpu(a4, 4) == (2 * ld + emb) * 18
    # PP 1: one stage, no copy
    assert [d for d, _ in _stage_params(a, parts)] == [8 * ld + emb]


def test_stage_layers_follow_megatron_split_options():
    assert stage_layers(_a(num_layers=62, pipeline_model_parallel_size=1)) == [62]
    assert stage_layers(_a(num_layers=62, pipeline_model_parallel_size=2)) == [31, 31]
    assert stage_layers(
        _a(
            num_layers=62,
            pipeline_model_parallel_size=4,
            decoder_first_pipeline_num_layers=14,
            decoder_last_pipeline_num_layers=14,
        )
    ) == [14, 17, 17, 14]
    assert stage_layers(
        _a(num_layers=62, pipeline_model_parallel_size=4, decoder_last_pipeline_num_layers=20)
    ) == [14, 14, 14, 20]
    assert stage_layers(
        _a(
            num_layers=62,
            pipeline_model_parallel_size=4,
            account_for_embedding_in_pipeline_split=True,
            account_for_loss_in_pipeline_split=True,
        )
    ) == [15, 16, 16, 15]


def test_uneven_last_stage_is_the_most_loaded():
    base = dict(pipeline_model_parallel_size=4, expert_model_parallel_size=64)
    even = state_bytes_per_gpu(_a(num_layers=64, **base), 256)
    heavy = state_bytes_per_gpu(_a(num_layers=64, decoder_last_pipeline_num_layers=22, **base), 256)
    assert heavy > even


def test_mtp_layers_are_counted_on_the_last_stage():
    a = _a(pipeline_model_parallel_size=2, expert_model_parallel_size=64)
    assert param_parts(a)["mtp"] == 0
    with_mtp = _a(pipeline_model_parallel_size=2, expert_model_parallel_size=64, mtp_num_layers=1)
    p = param_parts(with_mtp)
    assert p["mtp"] == p["layer_dense"] + 2 * 8192 * 8192 and p["mtp_expert"] == p["layer_expert"]
    assert state_bytes_per_gpu(with_mtp, 128) > state_bytes_per_gpu(a, 128)


def test_required_state_bytes_or_raise_leaves_room_for_workspaces_and_buffers():
    import pytest

    from primus.backends.megatron.patches.muse_spark_memory_check_patches import (
        required_state_bytes_or_raise,
    )

    a = _a(num_layers=4, num_experts=16, expert_model_parallel_size=1)
    need = state_bytes_per_gpu(a, 1)
    assert required_state_bytes_or_raise(a, 1, int(need / RAISE_FRACTION) + 1) == need
    with pytest.raises(ValueError, match="limit 95%"):
        required_state_bytes_or_raise(
            a, 1, int(need / 0.97)
        )  # states at 97% of the GPU: raise, not a warning


@pytest.mark.parametrize("first, last, pp", [(3, 3, 2), (40, 30, 4), (30, 30, 2), (31, None, 1)])
def test_stage_layers_rejects_a_split_that_does_not_cover_num_layers(first, last, pp):
    if pp == 1:  # PP 1 ignores the split options
        assert stage_layers(
            _a(num_layers=62, pipeline_model_parallel_size=1, decoder_first_pipeline_num_layers=first)
        ) == [62]
        return
    with pytest.raises(ValueError, match="do not split num_layers=62"):
        stage_layers(
            _a(
                num_layers=62,
                pipeline_model_parallel_size=pp,
                decoder_first_pipeline_num_layers=first,
                decoder_last_pipeline_num_layers=last,
            )
        )


def test_suggested_layout_ignores_the_users_uneven_split():
    # the user's first/last-stage counts belong to the user's PP=4; with them kept, a
    # candidate PP=2 would count only 3 + 3 of the 60 layers
    user = dict(
        num_layers=60,
        num_experts=128,
        pipeline_model_parallel_size=4,
        decoder_first_pipeline_num_layers=3,
        decoder_last_pipeline_num_layers=3,
        account_for_embedding_in_pipeline_split=True,
        expert_model_parallel_size=8,
    )
    s = suggest_layout(_a(**user), 192 * GIB)
    assert s is not None
    even = _a(
        **{
            **user,
            "decoder_first_pipeline_num_layers": None,
            "decoder_last_pipeline_num_layers": None,
            "account_for_embedding_in_pipeline_split": False,
            "tensor_model_parallel_size": s["tp"],
            "pipeline_model_parallel_size": s["pp"],
            "expert_model_parallel_size": s["ep"],
            "expert_tensor_parallel_size": 1,
        }
    )
    assert s["bytes"] == state_bytes_per_gpu(even, s["world"])
    assert s["bytes"] <= DEFAULT_BUDGET * 192 * GIB


def test_gpus_per_node_is_one_shared_reader(monkeypatch):
    """The DeepEP node-span check and the memory-check hint read GPUs per node the same way."""
    from primus.backends.megatron.core.models.muse_spark import memory_check as mc
    from primus.backends.megatron.patches.args import (
        muse_spark_recompute_cp_patches as rc,
    )

    assert rc.gpus_per_node is mc.gpus_per_node
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    monkeypatch.delenv("GPUS_PER_NODE", raising=False)
    assert mc.gpus_per_node() == mc.DEFAULT_GPUS_PER_NODE
    assert mc.gpus_per_node(3) == 3
    monkeypatch.setenv("GPUS_PER_NODE", "4")
    assert mc.gpus_per_node() == 4
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    assert mc.gpus_per_node() == 2  # torchrun's per-node count first
    warns = []
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "x")
    assert mc.gpus_per_node(warn=warns.append) == 4
    assert warns == ["ignoring LOCAL_WORLD_SIZE='x' (not a positive int)"]


@pytest.mark.parametrize(
    "dtype, expected",
    [("fp8", 1), ("torch.uint8", 1), ("bf16", 2), ("torch.float16", 2), ("fp32", 4), (None, 4)],
)
def test_dtype_bytes_counts_8bit_optimizer_states(dtype, expected):
    from primus.backends.megatron.core.models.muse_spark.memory_check import (
        _dtype_bytes,
    )

    assert _dtype_bytes(dtype) == expected


def test_fp8_moments_are_one_byte_each():
    from primus.backends.megatron.core.models.muse_spark.memory_check import (
        _bytes_per_param,
    )

    a = SimpleNamespace(
        use_precision_aware_optimizer=True,
        main_params_dtype="bf16",
        exp_avg_dtype="fp8",
        exp_avg_sq_dtype="fp8",
    )
    assert _bytes_per_param(a)[1] == 2 + 1 + 1
