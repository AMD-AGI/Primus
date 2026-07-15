# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Unit tests for FSDP2FP32Optimizer.

Tests focus on the core design properties of this optimizer:
- Optimizer states are FP32 (the whole point)
- state_dict format is raw PyTorch (not Megatron-wrapped)
- Gradient clipping actually clips gradients
- Optimizer step actually updates parameters and reduces loss
- Checkpoint save/load round-trip preserves state values and dtypes
- sharded_state_dict integrates with Megatron's torch_dist checkpointing
- finalize_dist_ckpt_load fills step counter (FSDP2-specific)
- Factory param group splitting (weight decay, LR scaling) works
"""

import pytest
import torch
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint

from primus.backends.megatron.core.optimizer.fsdp2_fp32_optimizer import (
    get_fsdp2_fp32_optimizer,
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
    model = torch.nn.Linear(8, 8, bias=True).to(dtype=torch.float32, device=device)
    config = _make_optimizer_config()
    optimizer = get_fsdp2_fp32_optimizer(
        config=config,
        model_chunks=[model],
    )
    return model, optimizer, config


def _build_model_sharded_state_dict(model, prefix=""):
    model_sd = model.state_dict(prefix=prefix, keep_vars=True)
    return make_sharded_tensors_for_checkpoint(model_sd, prefix, {}, ())


def _compute_grad_norm(model):
    """Compute L2 gradient norm across all model parameters."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.float().norm().item() ** 2
    return total**0.5


class TestFSDP2FP32Optimizer(PrimusUT):
    """Tests for FSDP2FP32Optimizer core functionality."""

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state):
        pass

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_optimizer_states_are_fp32(self):
        """The core invariant: all optimizer states must be FP32."""
        model, optimizer, _ = _make_model_and_optimizer(device="cuda")

        x = torch.randn(2, 8, dtype=torch.float32, device="cuda")
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        for p in optimizer.get_parameters():
            state = optimizer.optimizer.state[p]
            assert (
                state["exp_avg"].dtype == torch.float32
            ), f"exp_avg dtype is {state['exp_avg'].dtype}, expected float32"
            assert (
                state["exp_avg_sq"].dtype == torch.float32
            ), f"exp_avg_sq dtype is {state['exp_avg_sq'].dtype}, expected float32"
            assert state["step"].dtype == torch.float32

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_state_dict_format_is_raw(self):
        """state_dict must return raw PyTorch format, NOT a Megatron-wrapped
        format ({"optimizer": {...}}).

        This distinction is critical for checkpoint compatibility.
        """
        model, optimizer, _ = _make_model_and_optimizer(device="cuda")

        x = torch.randn(2, 8, dtype=torch.float32, device="cuda")
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        saved = optimizer.state_dict()
        assert "state" in saved, "state_dict missing 'state' key"
        assert "param_groups" in saved, "state_dict missing 'param_groups' key"
        assert "optimizer" not in saved, (
            "state_dict has 'optimizer' wrapper key -- this is the BFloat16Optimizer "
            "format, not the raw format expected by FSDP2FP32Optimizer"
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_clip_grad_norm_clips_gradients(self):
        """Verify clipping reduces grad norm to at most clip_value."""
        model, optimizer, _ = _make_model_and_optimizer(device="cuda")

        x = torch.randn(2, 8, dtype=torch.float32, device="cuda")
        loss = (model(x) * 100.0).sum()
        loss.backward()

        grad_norm_before = _compute_grad_norm(model)
        assert grad_norm_before > 1.0, "Gradients too small to test clipping"

        clip_value = 0.01
        returned_norm = optimizer.clip_grad_norm(clip_value)
        assert isinstance(
            returned_norm, torch.Tensor
        ), "clip_grad_norm should return a GPU tensor (deferred .item())"
        returned_norm_f = returned_norm.item()

        grad_norm_after = _compute_grad_norm(model)

        assert (
            abs(returned_norm_f - grad_norm_before) < 1e-3
        ), f"Returned norm {returned_norm_f} should approximate pre-clip norm {grad_norm_before}"
        assert (
            grad_norm_after <= clip_value + 1e-6
        ), f"Post-clip norm {grad_norm_after} exceeds clip_value {clip_value}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_step_actually_updates_parameters(self):
        """Verify optimizer.step() changes parameter values and returns
        the (success, grad_norm, num_zeros) 3-tuple that Megatron's
        training loop unpacks."""
        model, optimizer, _ = _make_model_and_optimizer(device="cuda")

        params_before = {name: p.clone() for name, p in model.named_parameters()}

        x = torch.randn(2, 8, dtype=torch.float32, device="cuda")
        loss = model(x).sum()
        loss.backward()
        success, grad_norm, num_zeros = optimizer.step()

        assert success is True
        assert isinstance(
            grad_norm, (float, torch.Tensor)
        ), f"grad_norm should be float or Tensor, got {type(grad_norm)}"
        grad_norm_f = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        assert grad_norm_f > 0.0

        for name, p in model.named_parameters():
            assert not torch.equal(
                p, params_before[name]
            ), f"Parameter '{name}' was not updated by optimizer.step()"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_multi_step_loss_decreases(self):
        """Run 10 steps and verify loss decreases, catching silent update failures."""
        torch.manual_seed(42)
        model = torch.nn.Linear(8, 8, bias=True).to(dtype=torch.float32, device="cuda")
        config = _make_optimizer_config(lr=1e-2, clip_grad=0.0)
        optimizer = get_fsdp2_fp32_optimizer(config=config, model_chunks=[model])

        x = torch.randn(4, 8, dtype=torch.float32, device="cuda")
        target = torch.randn(4, 8, dtype=torch.float32, device="cuda")

        loss_initial = None
        loss_final = None
        for _ in range(10):
            optimizer.zero_grad()
            output = model(x)
            loss = ((output - target) ** 2).mean()
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
        """Verify state_dict save/load preserves values and FP32 dtypes."""
        model, optimizer, config = _make_model_and_optimizer(device="cuda")

        x = torch.randn(2, 8, dtype=torch.float32, device="cuda")
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        saved_state = optimizer.state_dict()

        original_states = {}
        for pid, pstate in saved_state["state"].items():
            original_states[pid] = {
                "exp_avg": pstate["exp_avg"].clone(),
                "exp_avg_sq": pstate["exp_avg_sq"].clone(),
            }

        fresh_optimizer = get_fsdp2_fp32_optimizer(
            config=config,
            model_chunks=[model],
        )
        fresh_optimizer.load_state_dict(saved_state)

        loaded_state = fresh_optimizer.state_dict()
        for pid in original_states:
            assert pid in loaded_state["state"]
            torch.testing.assert_close(
                loaded_state["state"][pid]["exp_avg"],
                original_states[pid]["exp_avg"],
            )
            torch.testing.assert_close(
                loaded_state["state"][pid]["exp_avg_sq"],
                original_states[pid]["exp_avg_sq"],
            )
            assert loaded_state["state"][pid]["exp_avg"].dtype == torch.float32
            assert loaded_state["state"][pid]["exp_avg_sq"].dtype == torch.float32

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_sharded_state_dict_creates_sharded_tensors(self):
        """Verify sharded_state_dict produces ShardedTensor entries for torch_dist checkpointing."""
        model, optimizer, _ = _make_model_and_optimizer(device="cuda")

        model_ssd = _build_model_sharded_state_dict(model)
        opt_ssd = optimizer.sharded_state_dict(model_ssd, is_loading=True)

        assert "state" in opt_ssd
        state_entries = opt_ssd["state"]
        assert len(state_entries) > 0

        for pid, pstate in state_entries.items():
            if isinstance(pstate, dict):
                for key in ("exp_avg", "exp_avg_sq"):
                    assert key in pstate, f"Key '{key}' missing from state[{pid}]"
                    assert isinstance(pstate[key], ShardedTensor)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_finalize_dist_ckpt_load(self):
        """Verify finalize_dist_ckpt_load sets step counter from iteration."""
        model, optimizer, config = _make_model_and_optimizer(device="cuda")

        optimizer.init_state_fn(optimizer.optimizer)

        iteration = 42
        optimizer.finalize_dist_ckpt_load(iteration)

        for p in optimizer.get_parameters():
            if p in optimizer.optimizer.state:
                step = optimizer.optimizer.state[p].get("step")
                if step is not None:
                    assert step.item() == float(iteration)

    def test_init_state_fn_creates_fp32_states(self):
        """Verify init_state_fn creates FP32 placeholders for checkpoint loading."""
        model, optimizer, _ = _make_model_and_optimizer()

        base_opt = optimizer.optimizer
        for group in base_opt.param_groups:
            for p in group["params"]:
                assert len(base_opt.state[p]) == 0

        optimizer.init_state_fn(base_opt)

        for group in base_opt.param_groups:
            for p in group["params"]:
                state = base_opt.state[p]
                assert state["exp_avg"].dtype == torch.float32
                assert state["exp_avg_sq"].dtype == torch.float32
                assert state["exp_avg"].shape == p.data.shape

    def test_factory_applies_weight_decay_condition(self):
        """Verify no_weight_decay_cond splits parameters into correct groups."""
        model = torch.nn.Linear(8, 8, bias=True).to(dtype=torch.float32)
        config = _make_optimizer_config(weight_decay=0.1)

        def no_wd_for_bias(param):
            return param.ndim == 1

        optimizer = get_fsdp2_fp32_optimizer(
            config=config,
            model_chunks=[model],
            no_weight_decay_cond=no_wd_for_bias,
        )

        assert len(optimizer.optimizer.param_groups) == 2
        wds = {pg["weight_decay"] for pg in optimizer.optimizer.param_groups}
        assert 0.0 in wds
        assert 0.1 in wds
