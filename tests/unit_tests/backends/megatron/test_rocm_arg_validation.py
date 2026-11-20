###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

from types import SimpleNamespace

import pytest

from primus.backends.megatron.patches.args import rocm_arg_validation
from primus.backends.megatron.patches.args.rocm_arg_validation import (
    ALLOC_CONF_ENVS,
    validate_cross_entropy_exclusivity,
    validate_moe_replica,
    validate_moe_routing_replay,
    validate_turbo_fp8_permute,
    validate_turbo_grouped_gemm_with_muon,
    validate_turbo_grouped_gemm_without_padding,
    warn_turbo_grouped_gemm_without_expandable_segments,
)


def _no_padding_args(**overrides):
    values = {
        "turbo_grouped_gemm_without_padding": True,
        "enable_primus_turbo": True,
        "use_turbo_grouped_gemm": True,
        "fp8": "e4m3",
        "fp8_recipe": "tensorwise",
        "fp4": False,
        "moe_router_padding_for_quantization": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_cross_entropy_exclusivity_rejects_turbo_with_deferred():
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_cross_entropy_exclusivity(
            SimpleNamespace(deferred_cross_entropy=True, use_turbo_cross_entropy=True)
        )


@pytest.mark.parametrize(("deferred", "turbo"), [(True, False), (False, True), (False, False)])
def test_cross_entropy_exclusivity_allows_one(deferred, turbo):
    validate_cross_entropy_exclusivity(
        SimpleNamespace(deferred_cross_entropy=deferred, use_turbo_cross_entropy=turbo)
    )


def test_turbo_grouped_gemm_without_padding_accepts_tensorwise_fp8():
    validate_turbo_grouped_gemm_without_padding(_no_padding_args())


@pytest.mark.parametrize(
    "overrides",
    [
        {"fp8_recipe": "blockwise"},
        {"fp8_recipe": "mxfp8"},
        {"fp8": None, "fp8_recipe": None, "fp4": "e2m1", "fp4_recipe": "mxfp4"},
        {"fp8": None, "fp8_recipe": None, "fp4": False},
    ],
)
def test_turbo_grouped_gemm_without_padding_accepts_all_supported_precision_modes(overrides):
    validate_turbo_grouped_gemm_without_padding(_no_padding_args(**overrides))


def test_turbo_grouped_gemm_without_padding_disabled_is_noop():
    validate_turbo_grouped_gemm_without_padding(SimpleNamespace())


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"enable_primus_turbo": False}, "requires enable_primus_turbo"),
        ({"use_turbo_grouped_gemm": False}, "requires enable_primus_turbo"),
        (
            {"moe_router_padding_for_quantization": True},
            "requires moe_router_padding_for_quantization=False",
        ),
    ],
)
def test_turbo_grouped_gemm_without_padding_rejects_unsupported_config(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_turbo_grouped_gemm_without_padding(_no_padding_args(**overrides))


def _muon_args(**overrides):
    values = dict(
        optimizer="muon",
        enable_primus_turbo=True,
        use_turbo_grouped_gemm=True,
        moe_grouped_gemm=True,
        num_experts=8,
        turbo_sync_free_moe_stage=0,
    )
    return SimpleNamespace(**{**values, **overrides})


@pytest.fixture
def turbo_installed(monkeypatch):
    monkeypatch.setattr(rocm_arg_validation, "_primus_turbo_installed", lambda: True)


@pytest.mark.parametrize("optimizer", ["muon", "dist_muon", "adaptive_muon"])
def test_turbo_grouped_gemm_rejects_muon(turbo_installed, optimizer):
    with pytest.raises(ValueError, match="use_turbo_grouped_gemm=false.$"):
        validate_turbo_grouped_gemm_with_muon(_muon_args(optimizer=optimizer))


@pytest.mark.parametrize("freq", [1, 2, [0, 1, 0, 0]])
def test_turbo_grouped_gemm_with_muon_rejects_any_moe_layer(turbo_installed, freq):
    with pytest.raises(ValueError, match="use_turbo_grouped_gemm=false.$"):
        validate_turbo_grouped_gemm_with_muon(_muon_args(moe_layer_freq=freq))


@pytest.mark.parametrize("stage", [2, 3])
def test_turbo_grouped_gemm_with_muon_advises_sync_free_stage(turbo_installed, stage):
    with pytest.raises(ValueError, match="turbo_sync_free_moe_stage below 2"):
        validate_turbo_grouped_gemm_with_muon(_muon_args(turbo_sync_free_moe_stage=stage))


@pytest.mark.parametrize(
    "overrides",
    [
        {"optimizer": "adam"},
        {"use_turbo_grouped_gemm": False},
        {"enable_primus_turbo": False},
        {"num_experts": None},
        {"moe_grouped_gemm": False},
        {"moe_layer_freq": [0, 0, 0, 0]},
    ],
)
def test_turbo_grouped_gemm_with_muon_allows_other_configs(turbo_installed, overrides):
    validate_turbo_grouped_gemm_with_muon(_muon_args(**overrides))


def test_turbo_grouped_gemm_with_muon_allows_missing_turbo(monkeypatch):
    """Without primus_turbo the experts fall back to TE, which Muon trains."""
    monkeypatch.setattr(rocm_arg_validation, "_primus_turbo_installed", lambda: False)
    validate_turbo_grouped_gemm_with_muon(_muon_args())


def _fp8_permute_args(**overrides):
    values = {
        "turbo_fp8_permute": True,
        "enable_primus_turbo": True,
        "use_turbo_deepep": True,
        "turbo_fused_grouped_gemm": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"recompute_granularity": "full", "recompute_modules": ["moe_act"]},
        {"recompute_granularity": "selective", "recompute_modules": ["core_attn"]},
        {"recompute_granularity": "selective", "recompute_modules": None},
    ],
)
def test_turbo_fp8_permute_accepts_supported_config(overrides):
    validate_turbo_fp8_permute(_fp8_permute_args(**overrides))


def test_turbo_fp8_permute_disabled_is_noop():
    validate_turbo_fp8_permute(SimpleNamespace())
    validate_turbo_fp8_permute(SimpleNamespace(turbo_fp8_permute=False, use_turbo_deepep=False))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"enable_primus_turbo": False}, "requires enable_primus_turbo=True"),
        ({"use_turbo_deepep": False}, "requires use_turbo_deepep=True"),
        ({"turbo_fused_grouped_gemm": False}, "requires turbo_fused_grouped_gemm=True"),
        (
            {"recompute_granularity": "selective", "recompute_modules": ["core_attn", "moe_act"]},
            "does not support recompute of moe_act",
        ),
    ],
)
def test_turbo_fp8_permute_rejects_unsupported_config(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_turbo_fp8_permute(_fp8_permute_args(**overrides))


@pytest.fixture
def captured_warnings(monkeypatch):
    for env in ALLOC_CONF_ENVS:
        monkeypatch.delenv(env, raising=False)
    messages = []
    monkeypatch.setattr(rocm_arg_validation, "warning_rank_0", messages.append)
    return messages


def test_turbo_grouped_gemm_without_expandable_segments_warns(captured_warnings):
    warn_turbo_grouped_gemm_without_expandable_segments(SimpleNamespace(use_turbo_grouped_gemm=True))
    assert len(captured_warnings) == 1
    assert "PYTORCH_HIP_ALLOC_CONF=expandable_segments:True" in captured_warnings[0]


@pytest.mark.parametrize("env", ALLOC_CONF_ENVS)
@pytest.mark.parametrize(
    "conf", ["expandable_segments:True", "max_split_size_mb:512, expandable_segments:True"]
)
def test_turbo_grouped_gemm_with_expandable_segments_is_silent(captured_warnings, monkeypatch, env, conf):
    monkeypatch.setenv(env, conf)
    warn_turbo_grouped_gemm_without_expandable_segments(SimpleNamespace(use_turbo_grouped_gemm=True))
    assert captured_warnings == []


def test_expandable_segments_false_still_warns(captured_warnings, monkeypatch):
    monkeypatch.setenv("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:False")
    warn_turbo_grouped_gemm_without_expandable_segments(SimpleNamespace(use_turbo_grouped_gemm=True))
    assert len(captured_warnings) == 1


@pytest.mark.parametrize("args", [SimpleNamespace(), SimpleNamespace(use_turbo_grouped_gemm=False)])
def test_no_turbo_grouped_gemm_does_not_warn(captured_warnings, args):
    warn_turbo_grouped_gemm_without_expandable_segments(args)
    assert captured_warnings == []


@pytest.mark.parametrize(
    ("backend", "ok"),
    [
        ("default", True),
        ("triton", True),
        ("TRITON", True),
        ("bf16:triton,fp8:ck", True),
        ("other:triton", True),
        ("fp8:ck", True),  # BF16 left to the default choice
        ("ck", False),
        ("bf16:hipblaslt,fp8:triton", False),
        ("fp8:triton,other:ck", False),
    ],
)
def test_moe_replica_bf16_backend(backend, ok):
    args = _replica_args(turbo_grouped_gemm_backend=backend)
    if ok:
        validate_moe_replica(args)
    else:
        with pytest.raises(ValueError, match="BF16 backend"):
            validate_moe_replica(args)


def _dsv3_replica_args(**overrides):
    # DeepSeek-V3: 3 dense layers and 58 MoE layers, hidden 7168, expert FFN 2048 with SwiGLU, BF16 main_grad
    values = dict(
        hidden_size=7168,
        moe_ffn_hidden_size=2048,
        swiglu=True,
        num_layers=61,
        moe_layer_freq=[0] * 3 + [1] * 58,
        params_dtype="torch.bfloat16",
        main_grads_dtype="bf16",
    )
    values.update(overrides)
    return _replica_args(**values)


def test_moe_replica_slot_memory():
    per_slot_layer = (2 * 2048 * 7168 + 2048 * 7168) * (2 + 2)  # fc1 + fc2, BF16 weights and gradients
    assert rocm_arg_validation.moe_replica_slot_bytes(_dsv3_replica_args()) == 58 * 4 * per_slot_layer
    assert rocm_arg_validation.moe_replica_slot_bytes(_dsv3_replica_args(moe_replica_slots=32)) == (
        58 * 32 * per_slot_layer
    )


def test_moe_replica_slots_that_do_not_fit_are_an_error(monkeypatch):
    cuda = rocm_arg_validation.torch.cuda
    monkeypatch.setattr(cuda, "is_available", lambda: True)
    monkeypatch.setattr(cuda, "current_device", lambda: 0)
    monkeypatch.setattr(cuda, "get_device_properties", lambda _: SimpleNamespace(total_memory=288 * 10**9))
    validate_moe_replica(_dsv3_replica_args())  # about 38 GiB: a warning
    with pytest.raises(ValueError, match="more than the 268 GiB"):
        validate_moe_replica(_dsv3_replica_args(moe_replica_planner="moonep", moe_replica_slots=32))


def _replica_args(**overrides):
    values = {
        "moe_replica_planner": "ultraep",
        "moe_replica_placement": "per_step",
        "moe_replica_slots": 4,
        "moe_replica_min_rows": 1024,
        "moe_replica_min_gain": 0.1,
        "moe_replica_cost": 0,
        "moe_replica_block": 1,
        "moe_replica_replan_above": 0.0,
        "moe_replica_log_interval": 0,
        "enable_primus_turbo": True,
        "use_turbo_deepep": True,
        "use_turbo_grouped_gemm": True,
        "moe_grouped_gemm": True,
        "expert_model_parallel_size": 8,
        "num_experts": 256,
        "turbo_grouped_gemm_backend": "default",
        "tensor_model_parallel_size": 1,
        "expert_tensor_parallel_size": None,
        "pipeline_model_parallel_size": 1,
        "moe_expert_capacity_factor": None,
        "overlap_grad_reduce": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_moe_replica_off_is_noop():
    validate_moe_replica(SimpleNamespace())
    validate_moe_replica(SimpleNamespace(moe_replica_planner=None))


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"moe_replica_planner": "moonep", "moe_replica_slots": 32},
        {"moe_replica_placement": "per_microbatch", "moe_replica_replan_above": 1.08},
        {"turbo_grouped_gemm_backend": "triton"},
    ],
)
def test_moe_replica_accepts_supported(overrides):
    validate_moe_replica(_replica_args(**overrides))


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"moe_replica_planner": "fixed"}, "ultraep or moonep"),
        ({"moe_replica_placement": "per_layer"}, "per_step or per_microbatch"),
        ({"moe_replica_slots": 0}, "moe_replica_slots must be >= 1"),
        ({"moe_replica_min_gain": 1.0}, "moe_replica_min_gain must be in"),
        ({"moe_replica_block": 0}, "moe_replica_block must be >= 1"),
        ({"moe_replica_placement": "per_microbatch", "moe_replica_replan_above": 0.5}, "0 or >= 1"),
        ({"moe_replica_replan_above": 1.08}, "requires moe_replica_placement=per_microbatch"),
        ({"use_turbo_deepep": False}, "requires use_turbo_deepep=True"),
        ({"expert_model_parallel_size": 1}, "expert_model_parallel_size > 1"),
        ({"moe_replica_planner": "moonep"}, "moe_replica_slots >= num_experts"),
        ({"turbo_grouped_gemm_backend": "ck"}, "turbo_grouped_gemm_backend=ck"),
        ({"tensor_model_parallel_size": 2}, "tensor_model_parallel_size=1"),
        ({"expert_tensor_parallel_size": 2}, "expert_tensor_parallel_size=1"),
        ({"pipeline_model_parallel_size": 2}, "pipeline_model_parallel_size=1"),
        ({"use_turbo_mega_moe": True}, "use_turbo_mega_moe"),
        ({"fp8": "e4m3"}, "does not support fp8"),
        ({"turbo_fused_grouped_gemm": True}, "turbo_fused_grouped_gemm"),
        ({"overlap_moe_expert_parallel_comm": True}, "overlap_moe_expert_parallel_comm"),
        ({"delay_wgrad_compute": True}, "delay_wgrad_compute"),
        ({"use_torch_fsdp2": True}, "use_torch_fsdp2"),
        ({"moe_expert_capacity_factor": 1.0}, "token dropping"),
        (
            {"moe_router_force_load_balancing": True, "moe_router_force_load_balancing_type": "even"},
            "force_load_balancing_type=even",
        ),
        ({"enable_cuda_graph": True}, "CUDA graphs"),
        ({"overlap_grad_reduce": True}, "overlap_grad_reduce=True"),
    ],
)
def test_moe_replica_rejects_unsupported(overrides, match):
    with pytest.raises(ValueError, match=match):
        validate_moe_replica(_replica_args(**overrides))


def _replay_args(**overrides):
    values = {
        "moe_routing_replay_path": "/traces",
        "moe_routing_replay_layers": None,
        "moe_router_fusion": False,
        "moe_router_force_load_balancing": False,
        "moe_use_fused_router_with_aux_score": False,
        "moe_router_force_biased": None,
        "pipeline_model_parallel_size": 1,
        "mtp_num_layers": None,
        "recompute_granularity": None,
        "recompute_layer_ids": None,
        "recompute_modules": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"moe_routing_replay_layers": [10, 30, 44, 23]},
        # moe_act recompute does not route again
        {"recompute_granularity": "selective", "recompute_modules": ["moe_act"]},
    ],
)
def test_moe_routing_replay_accepts_supported(overrides):
    validate_moe_routing_replay(_replay_args(**overrides))
    validate_moe_routing_replay(SimpleNamespace())


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        (
            {"moe_routing_replay_path": None, "moe_routing_replay_layers": [0]},
            "requires moe_routing_replay_path",
        ),
        ({"moe_router_fusion": True}, "moe_router_fusion=False"),
        ({"moe_use_fused_router_with_aux_score": True}, "moe_use_fused_router_with_aux_score=False"),
        ({"moe_router_force_load_balancing": True}, "moe_router_force_load_balancing=False"),
        ({"moe_router_force_biased": 0.3}, "moe_router_force_biased=None"),
        ({"pipeline_model_parallel_size": 2}, "pipeline_model_parallel_size=1"),
        ({"mtp_num_layers": 1}, "MTP"),
        ({"recompute_granularity": "full"}, "recomputing MoE layers"),
        ({"recompute_layer_ids": [0, 1]}, "recomputing MoE layers"),
        ({"recompute_granularity": "selective", "recompute_modules": ["moe"]}, "recomputing MoE layers"),
    ],
)
def test_moe_routing_replay_rejects_unsupported(overrides, match):
    with pytest.raises(ValueError, match=match):
        validate_moe_routing_replay(_replay_args(**overrides))
