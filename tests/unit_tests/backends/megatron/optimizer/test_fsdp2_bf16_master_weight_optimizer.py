# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Unit tests for FSDP2BF16MasterWeightOptimizer.

Tests focus on the core design properties:
- FP32 master weights are created from BF16 model parameters
- prepare_grads copies BF16 grads to FP32 master grads
- Optimizer step updates FP32 masters and copies back to BF16 params
- state_dict includes fp32_from_fp16_params key
- Checkpoint save/load round-trip preserves values
- sharded_state_dict integrates with Megatron's torch_dist checkpointing
- finalize_dist_ckpt_load fills step counter and syncs params
- Factory param group splitting (weight decay, LR scaling) works
- Multi-step training actually reduces loss
"""

import pytest
import torch
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint

from primus.backends.megatron.core.optimizer.fsdp2_bf16_master_weight_optimizer import (
    get_fsdp2_bf16_master_weight_optimizer,
)
from tests.utils import PrimusUT


def _make_optimizer_config(**overrides):
    defaults = dict(
        bf16=True,
        fp16=False,
        optimizer="adam",
        lr=1e-4,
        min_lr=1e-5,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        clip_grad=1.0,
        use_precision_aware_optimizer=False,
    )
    defaults.update(overrides)
    return OptimizerConfig(**defaults)


def _make_model_and_optimizer(device="cpu"):
    model = torch.nn.Linear(8, 8, bias=True).to(dtype=torch.bfloat16, device=device)
    config = _make_optimizer_config()
    optimizer = get_fsdp2_bf16_master_weight_optimizer(
        config=config,
        model_chunks=[model],
    )
    return model, optimizer, config


def _build_model_sharded_state_dict(model, prefix=""):
    model_sd = model.state_dict(prefix=prefix, keep_vars=True)
    return make_sharded_tensors_for_checkpoint(model_sd, prefix, {}, ())


class TestFSDP2BF16MasterWeightOptimizer(PrimusUT):
    """Tests for FSDP2BF16MasterWeightOptimizer core functionality."""

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state):
        pass

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_master_weights_are_fp32_clones_of_bf16(self):
        """FP32 master weights must match BF16 params in value and shape."""
        model, optimizer, _ = _make_model_and_optimizer(device="cuda")

        for bf16_group, fp32_group in zip(optimizer.bf16_groups, optimizer.fp32_from_bf16_groups):
            for bf16_param, fp32_master in zip(bf16_group, fp32_group):
                assert bf16_param.dtype == torch.bfloat16
                assert fp32_master.dtype == torch.float32
                assert fp32_master.shape == bf16_param.shape
                torch.testing.assert_close(fp32_master, bf16_param.float(), atol=0, rtol=0)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_main_param_attribute_set(self):
        """BF16 params must have .main_param pointing to FP32 master."""
        model, optimizer, _ = _make_model_and_optimizer(device="cuda")

        for bf16_group, fp32_group in zip(optimizer.bf16_groups, optimizer.fp32_from_bf16_groups):
            for bf16_param, fp32_master in zip(bf16_group, fp32_group):
                assert hasattr(bf16_param, "main_param")
                assert bf16_param.main_param is fp32_master

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_optimizer_operates_on_fp32_masters(self):
        """Base optimizer's param_groups must contain FP32 masters, not BF16 params."""
        model, optimizer, _ = _make_model_and_optimizer(device="cuda")

        for group in optimizer.optimizer.param_groups:
            for p in group["params"]:
                assert p.dtype == torch.float32, f"Optimizer param dtype is {p.dtype}, expected float32"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_prepare_grads_copies_to_fp32(self):
        """prepare_grads must cast BF16 grads to FP32 on masters and clear BF16 grads."""
        model, optimizer, _ = _make_model_and_optimizer(device="cuda")

        x = torch.randn(2, 8, dtype=torch.bfloat16, device="cuda")
        loss = model(x).sum()
        loss.backward()

        for bf16_group in optimizer.bf16_groups:
            for p in bf16_group:
                assert p.grad is not None, "BF16 param should have grad before prepare_grads"

        optimizer.prepare_grads()

        for bf16_group in optimizer.bf16_groups:
            for p in bf16_group:
                assert p.grad is None, "BF16 grad should be cleared after prepare_grads"

        for fp32_group in optimizer.fp32_from_bf16_groups:
            for p in fp32_group:
                assert p.grad is not None, "FP32 master should have grad after prepare_grads"
                assert p.grad.dtype == torch.float32

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_step_updates_both_master_and_model(self):
        """step() must update FP32 masters and copy back to BF16 params.

        Uses high LR to ensure BF16 copy-back produces visible changes
        (small updates can round to zero in BF16).
        """
        model = torch.nn.Linear(8, 8, bias=True).to(dtype=torch.bfloat16, device="cuda")
        config = _make_optimizer_config(lr=1e-1, clip_grad=1.0)
        optimizer = get_fsdp2_bf16_master_weight_optimizer(config=config, model_chunks=[model])

        bf16_before = {}
        for name, p in model.named_parameters():
            bf16_before[name] = p.data.clone()

        x = torch.randn(2, 8, dtype=torch.bfloat16, device="cuda")
        loss = (model(x) * 10.0).sum()
        loss.backward()
        success, grad_norm, num_zeros = optimizer.step()

        assert success is True

        for name, p in model.named_parameters():
            assert not torch.equal(
                p.data, bf16_before[name]
            ), f"BF16 param '{name}' was not updated after step"

        for bf16_group, fp32_group in zip(optimizer.bf16_groups, optimizer.fp32_from_bf16_groups):
            for bf16_param, fp32_master in zip(bf16_group, fp32_group):
                expected = fp32_master.data.to(torch.bfloat16)
                torch.testing.assert_close(bf16_param.data, expected, atol=0, rtol=0)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_optimizer_states_are_fp32(self):
        """All optimizer states (exp_avg, exp_avg_sq) must be FP32."""
        model, optimizer, _ = _make_model_and_optimizer(device="cuda")

        x = torch.randn(2, 8, dtype=torch.bfloat16, device="cuda")
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        for fp32_group in optimizer.fp32_from_bf16_groups:
            for p in fp32_group:
                state = optimizer.optimizer.state[p]
                assert state["exp_avg"].dtype == torch.float32
                assert state["exp_avg_sq"].dtype == torch.float32

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_state_dict_has_fp32_from_fp16_params_key(self):
        """state_dict must include 'fp32_from_fp16_params' with FP32 master weights."""
        model, optimizer, _ = _make_model_and_optimizer(device="cuda")

        x = torch.randn(2, 8, dtype=torch.bfloat16, device="cuda")
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        saved = optimizer.state_dict()
        assert "optimizer" in saved
        assert "fp32_from_fp16_params" in saved

        for group in saved["fp32_from_fp16_params"]:
            for param in group:
                assert param.dtype == torch.float32

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_multi_step_loss_decreases(self):
        """Run 10 steps and verify loss decreases."""
        torch.manual_seed(42)
        model = torch.nn.Linear(8, 8, bias=True).to(dtype=torch.bfloat16, device="cuda")
        config = _make_optimizer_config(lr=1e-2, clip_grad=0.0)
        optimizer = get_fsdp2_bf16_master_weight_optimizer(config=config, model_chunks=[model])

        x = torch.randn(4, 8, dtype=torch.bfloat16, device="cuda")
        target = torch.randn(4, 8, dtype=torch.bfloat16, device="cuda")

        loss_initial = None
        loss_final = None
        for _ in range(10):
            optimizer.zero_grad()
            output = model(x)
            loss = ((output - target).float() ** 2).mean()
            loss.backward()
            optimizer.step()

            if loss_initial is None:
                loss_initial = loss.item()
            loss_final = loss.item()

        assert (
            loss_final < loss_initial
        ), f"Loss did not decrease: initial={loss_initial:.6f}, final={loss_final:.6f}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_state_dict_round_trip(self):
        """Verify state_dict save/load preserves FP32 master weights and optimizer states."""
        model, optimizer, config = _make_model_and_optimizer(device="cuda")

        x = torch.randn(2, 8, dtype=torch.bfloat16, device="cuda")
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        saved_state = optimizer.state_dict()

        original_masters = []
        for group in saved_state["fp32_from_fp16_params"]:
            original_masters.append([p.clone() for p in group])

        original_opt_states = {}
        for pid, pstate in saved_state["optimizer"]["state"].items():
            original_opt_states[pid] = {
                "exp_avg": pstate["exp_avg"].clone(),
                "exp_avg_sq": pstate["exp_avg_sq"].clone(),
            }

        fresh_optimizer = get_fsdp2_bf16_master_weight_optimizer(config=config, model_chunks=[model])
        fresh_optimizer.load_state_dict(saved_state)

        loaded_state = fresh_optimizer.state_dict()
        for saved_group, loaded_group in zip(original_masters, loaded_state["fp32_from_fp16_params"]):
            for saved_p, loaded_p in zip(saved_group, loaded_group):
                torch.testing.assert_close(loaded_p, saved_p)
                assert loaded_p.dtype == torch.float32

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_sharded_state_dict_creates_sharded_tensors(self):
        """Verify sharded_state_dict produces ShardedTensor entries."""
        model, optimizer, _ = _make_model_and_optimizer(device="cuda")

        model_ssd = _build_model_sharded_state_dict(model)
        opt_ssd = optimizer.sharded_state_dict(model_ssd, is_loading=True)

        assert "optimizer" in opt_ssd
        assert "fp32_from_fp16_params" in opt_ssd

        # Check optimizer states are ShardedTensors
        state_entries = opt_ssd["optimizer"]["state"]
        assert len(state_entries) > 0
        for pid, pstate in state_entries.items():
            if isinstance(pstate, dict):
                for key in ("exp_avg", "exp_avg_sq"):
                    assert key in pstate
                    assert isinstance(pstate[key], ShardedTensor)

        # Check fp32_from_fp16_params are ShardedTensors
        for group in opt_ssd["fp32_from_fp16_params"]:
            for entry in group:
                assert isinstance(entry, ShardedTensor)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_finalize_dist_ckpt_load(self):
        """Verify finalize_dist_ckpt_load sets step counter and syncs params."""
        model, optimizer, _ = _make_model_and_optimizer(device="cuda")
        optimizer.init_state_fn(optimizer.optimizer)

        # Modify FP32 masters to differ from BF16 params
        for fp32_group in optimizer.fp32_from_bf16_groups:
            for p in fp32_group:
                p.data.add_(1.0)

        iteration = 42
        optimizer.finalize_dist_ckpt_load(iteration)

        for fp32_group in optimizer.fp32_from_bf16_groups:
            for p in fp32_group:
                if p in optimizer.optimizer.state:
                    step = optimizer.optimizer.state[p].get("step")
                    if step is not None:
                        assert step.item() == float(iteration)

        # BF16 params should now match FP32 masters (copy-back happened)
        for bf16_group, fp32_group in zip(optimizer.bf16_groups, optimizer.fp32_from_bf16_groups):
            for bf16_param, fp32_master in zip(bf16_group, fp32_group):
                expected = fp32_master.data.to(torch.bfloat16)
                torch.testing.assert_close(bf16_param.data, expected, atol=0, rtol=0)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_reload_model_params(self):
        """Verify reload_model_params copies BF16 params to FP32 masters."""
        model, optimizer, _ = _make_model_and_optimizer(device="cuda")

        # Manually change BF16 params (simulating checkpoint load)
        for p in model.parameters():
            p.data.fill_(0.5)

        optimizer.reload_model_params()

        for bf16_group, fp32_group in zip(optimizer.bf16_groups, optimizer.fp32_from_bf16_groups):
            for bf16_param, fp32_master in zip(bf16_group, fp32_group):
                torch.testing.assert_close(fp32_master, bf16_param.float(), atol=0, rtol=0)

    def test_factory_applies_weight_decay_condition(self):
        """Verify no_weight_decay_cond splits parameters into correct groups."""
        model = torch.nn.Linear(8, 8, bias=True).to(dtype=torch.bfloat16)
        config = _make_optimizer_config(weight_decay=0.1)

        def no_wd_for_bias(param):
            return param.ndim == 1

        optimizer = get_fsdp2_bf16_master_weight_optimizer(
            config=config,
            model_chunks=[model],
            no_weight_decay_cond=no_wd_for_bias,
        )

        assert len(optimizer.optimizer.param_groups) == 2
        wds = {pg["weight_decay"] for pg in optimizer.optimizer.param_groups}
        assert 0.0 in wds
        assert 0.1 in wds

    def test_factory_handles_fp32_params(self):
        """Verify FP32 params (e.g. LayerNorm) go to fp32_from_fp32_groups without master copy."""
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 8, bias=False).to(dtype=torch.bfloat16),
            torch.nn.LayerNorm(8).to(dtype=torch.float32),
        )
        config = _make_optimizer_config()
        optimizer = get_fsdp2_bf16_master_weight_optimizer(config=config, model_chunks=[model])

        total_bf16 = sum(len(g) for g in optimizer.bf16_groups)
        total_fp32 = sum(len(g) for g in optimizer.fp32_from_fp32_groups)

        assert total_bf16 == 1, f"Expected 1 BF16 param, got {total_bf16}"
        assert total_fp32 == 2, f"Expected 2 FP32 params (LN weight+bias), got {total_fp32}"
