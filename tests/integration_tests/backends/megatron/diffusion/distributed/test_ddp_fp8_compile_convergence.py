# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Multi-GPU convergence test: Megatron DDP + FP8 + torch.compile scope.

Tests whether torch.compile scope (whole_model vs inner-only) causes
convergence divergence when real NCCL all-reduce + overlapped gradient
reduction is active. Phase 1 and Phase 2 proved that single-GPU FP8 +
compile and single-GPU mock DDP + compile are bit-identical. This test
adds real NCCL communication to isolate the remaining hypothesis.

Run with:
    torchrun --nproc_per_node 2 --master_port 29502 -m pytest -xvs \
        tests/integration_tests/backends/megatron/diffusion/distributed/test_ddp_fp8_compile_convergence.py
"""

import copy
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.core.low_precision import (
    ScalingGranularity,
    float8_e4m3,
    float8_e5m2,
)
from primus_turbo.pytorch.ops.quantization import quantize_fp8

from primus.backends.megatron.core.extensions.primus_turbo_float8_local import (
    OpaqueFP8LinearTensorwiseFunction,
)

_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 0))

requires_torchrun = pytest.mark.skipif(
    _WORLD_SIZE < 1,
    reason="Requires torchrun (WORLD_SIZE env var not set)",
)

_DIM = 256
_OUT_DIM = 512
_BATCH = 16
_N_STEPS = 100
_GRAN_VALUE = ScalingGranularity.TENSORWISE.value
_BACKEND_VALUE = BackendType.HIPBLASLT.value


# ---------------------------------------------------------------------------
# Distributed setup / teardown (follows FSDP2 integration test pattern)
# ---------------------------------------------------------------------------


def _init_distributed():
    from megatron.core import parallel_state as ps

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())

    import primus.core.utils.logger as primus_logger

    if primus_logger._logger is None:
        cfg = primus_logger.LoggerConfig(
            exp_root_path="/tmp/primus_test_logs",
            work_group="test",
            user_name="test",
            exp_name="ddp_fp8_compile_convergence",
            module_name="test",
            rank=rank,
            world_size=world_size,
        )
        primus_logger.setup_logger(cfg, is_head=(rank == 0))

    from primus.modules.module_utils import set_logging_rank

    set_logging_rank(rank, world_size)

    if ps.model_parallel_is_initialized():
        ps.destroy_model_parallel()

    ps.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
    )

    from megatron.core.tensor_parallel import random as tp_random

    try:
        tp_random.initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
    except (ImportError, AssertionError):
        tp_random.initialize_rng_tracker(use_cudagraphable_rng=True, force_reset=True)
    tp_random.model_parallel_cuda_manual_seed(42)

    from megatron.training.global_vars import set_args

    set_args(
        SimpleNamespace(
            enable_turbo_attention_float8=False,
            data_parallel_replicate_degree=1,
            use_fsdp2_fp8_all_gather=False,
            use_fsdp2_bf16_master_weight_optimizer=False,
            use_fsdp2_fp32_param_optimizer=False,
            torch_fsdp2_reshard_after_forward=False,
        )
    )


def _cleanup_distributed():
    from megatron.core import parallel_state as ps

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    if ps.model_parallel_is_initialized():
        ps.destroy_model_parallel()

    import megatron.training.global_vars as gvars

    gvars._GLOBAL_ARGS = None


# ---------------------------------------------------------------------------
# Model (same as Phase 1/2 _OpaqueModule)
# ---------------------------------------------------------------------------


class _OpaqueModule(nn.Module):
    """FP8 module using OpaqueFP8LinearTensorwiseFunction with pre-extracted weights."""

    def __init__(self, dim, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.weight = nn.Parameter(torch.randn(out_dim, dim, dtype=torch.bfloat16))

    def forward(self, x):
        x = self.norm(x)
        w_fp8, w_scale = quantize_fp8(self.weight, float8_e4m3, ScalingGranularity.TENSORWISE)
        result = OpaqueFP8LinearTensorwiseFunction.apply(
            x,
            self.weight,
            w_fp8,
            w_scale,
            float8_e4m3,
            float8_e5m2,
            _GRAN_VALUE,
            _BACKEND_VALUE,
        )
        return result[0]


# ---------------------------------------------------------------------------
# Optimizer that reads main_grad (reused from Phase 2)
# ---------------------------------------------------------------------------


class _MainGradAdamW(torch.optim.AdamW):
    """AdamW that reads from param.main_grad instead of param.grad."""

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if hasattr(p, "main_grad") and p.main_grad is not None:
                    p.grad = p.main_grad.clone()
        result = super().step(closure=closure)
        for group in self.param_groups:
            for p in group["params"]:
                p.grad = None
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_transformer_config():
    from megatron.core.transformer.transformer_config import TransformerConfig

    return TransformerConfig(
        num_layers=1,
        hidden_size=_DIM,
        num_attention_heads=1,
        gradient_accumulation_fusion=False,
    )


def _wrap_with_megatron_ddp(module, overlap_grad_reduce=True):
    from megatron.core.distributed import DistributedDataParallel
    from megatron.core.distributed.distributed_data_parallel_config import (
        DistributedDataParallelConfig,
    )

    config = _make_transformer_config()
    ddp_config = DistributedDataParallelConfig(
        overlap_grad_reduce=overlap_grad_reduce,
        use_distributed_optimizer=False,
        average_in_collective=False,
    )
    ddp = DistributedDataParallel(
        config=config,
        ddp_config=ddp_config,
        module=module,
    )
    return ddp


def _make_inputs(n, dim, batch, seed=42):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return [torch.randn(batch, dim, dtype=torch.bfloat16, generator=g).cuda() for _ in range(n)]


def _clone_state(model):
    return copy.deepcopy(model.state_dict())


def _run_ddp_training(ddp, inputs, n_steps, compile_target=None, lr=1e-3):
    """Run training with real Megatron DDP.

    Args:
        ddp: Megatron DistributedDataParallel-wrapped model.
        inputs: List of input tensors.
        n_steps: Number of training steps.
        compile_target: None="eager", "whole_model", or "inner_only".
        lr: Learning rate.
    """
    optimizer = _MainGradAdamW(ddp.module.parameters(), lr=lr)

    if compile_target == "whole_model":
        ddp.forward = torch.compile(ddp.forward)
    elif compile_target == "inner_only":
        ddp.module.forward = torch.compile(ddp.module.forward)

    losses = []
    for step in range(n_steps):
        ddp.zero_grad_buffer()
        out = ddp(inputs[step % len(inputs)])
        loss = out.sum()
        loss.backward()
        ddp.finish_grad_sync()
        optimizer.step()
        losses.append(loss.item())
    return losses


def _print_comparison(label_a, losses_a, label_b, losses_b, milestones=None):
    if milestones is None:
        milestones = [0, 1, 5, 10, 20, 50]
        milestones += [i for i in range(100, max(len(losses_a), len(losses_b)), 50)]
        if len(losses_a) - 1 not in milestones:
            milestones.append(len(losses_a) - 1)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank != 0:
        return
    print(f"\n{'Step':>6} | {label_a:>20} | {label_b:>20} | {'Rel Diff':>10}")
    print("-" * 65)
    for step in milestones:
        if step < len(losses_a) and step < len(losses_b):
            a, b = losses_a[step], losses_b[step]
            rel = abs(a - b) / max(abs(a), abs(b), 1e-12)
            print(f"{step:>6} | {a:>20.6f} | {b:>20.6f} | {rel:>10.6f}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDDPFP8CompileConvergence:
    """Real Megatron DDP + NCCL + FP8 + torch.compile scope convergence tests.

    Tests whether whole_model compile (failing local spec config) vs
    inner_only compile (converging TE spec config) diverge when real
    NCCL all-reduce with overlap_grad_reduce=True is active.
    """

    def setup_method(self):
        _init_distributed()
        torch._dynamo.reset()

    def teardown_method(self):
        _cleanup_distributed()

    @requires_torchrun
    def test_whole_model_compiled_vs_eager(self):
        """whole_model compile + real DDP should track eager + real DDP."""
        torch._dynamo.reset()
        torch.manual_seed(42)
        inputs = _make_inputs(20, _DIM, _BATCH)

        torch.manual_seed(123)
        eager_model = _OpaqueModule(_DIM, _OUT_DIM).to(dtype=torch.bfloat16, device="cuda")
        init_state = _clone_state(eager_model)
        eager_ddp = _wrap_with_megatron_ddp(eager_model)
        eager_losses = _run_ddp_training(eager_ddp, inputs, _N_STEPS)

        torch._dynamo.reset()
        torch.manual_seed(123)
        compiled_model = _OpaqueModule(_DIM, _OUT_DIM).to(dtype=torch.bfloat16, device="cuda")
        compiled_model.load_state_dict(init_state)
        compiled_ddp = _wrap_with_megatron_ddp(compiled_model)
        compiled_losses = _run_ddp_training(compiled_ddp, inputs, _N_STEPS, compile_target="whole_model")

        _print_comparison(
            "DDP Eager",
            eager_losses,
            "DDP WholeModel",
            compiled_losses,
        )

        rank = torch.distributed.get_rank()
        final_rel = abs(eager_losses[-1] - compiled_losses[-1]) / max(abs(eager_losses[-1]), 1e-12)
        if rank == 0:
            print(f"\nFinal loss relative diff: {final_rel:.6f}")
        assert final_rel < 0.05, (
            f"[Rank {rank}] DDP whole_model compile diverged from eager: "
            f"eager={eager_losses[-1]:.6f}, compiled={compiled_losses[-1]:.6f}, "
            f"rel_diff={final_rel:.6f}"
        )

    @requires_torchrun
    def test_inner_only_compiled_vs_eager(self):
        """inner-only compile + real DDP should track eager + real DDP."""
        torch._dynamo.reset()
        torch.manual_seed(42)
        inputs = _make_inputs(20, _DIM, _BATCH)

        torch.manual_seed(123)
        eager_model = _OpaqueModule(_DIM, _OUT_DIM).to(dtype=torch.bfloat16, device="cuda")
        init_state = _clone_state(eager_model)
        eager_ddp = _wrap_with_megatron_ddp(eager_model)
        eager_losses = _run_ddp_training(eager_ddp, inputs, _N_STEPS)

        torch._dynamo.reset()
        torch.manual_seed(123)
        compiled_model = _OpaqueModule(_DIM, _OUT_DIM).to(dtype=torch.bfloat16, device="cuda")
        compiled_model.load_state_dict(init_state)
        compiled_ddp = _wrap_with_megatron_ddp(compiled_model)
        compiled_losses = _run_ddp_training(compiled_ddp, inputs, _N_STEPS, compile_target="inner_only")

        _print_comparison(
            "DDP Eager",
            eager_losses,
            "DDP InnerOnly",
            compiled_losses,
        )

        rank = torch.distributed.get_rank()
        final_rel = abs(eager_losses[-1] - compiled_losses[-1]) / max(abs(eager_losses[-1]), 1e-12)
        if rank == 0:
            print(f"\nFinal loss relative diff: {final_rel:.6f}")
        assert final_rel < 0.05, (
            f"[Rank {rank}] DDP inner-only compile diverged from eager: "
            f"eager={eager_losses[-1]:.6f}, compiled={compiled_losses[-1]:.6f}, "
            f"rel_diff={final_rel:.6f}"
        )

    @requires_torchrun
    def test_whole_model_vs_inner_only(self):
        """Direct comparison of whole_model vs inner_only compile with real DDP."""
        torch._dynamo.reset()
        torch.manual_seed(42)
        inputs = _make_inputs(20, _DIM, _BATCH)

        torch.manual_seed(123)
        whole_model = _OpaqueModule(_DIM, _OUT_DIM).to(dtype=torch.bfloat16, device="cuda")
        init_state = _clone_state(whole_model)
        whole_ddp = _wrap_with_megatron_ddp(whole_model)
        whole_losses = _run_ddp_training(whole_ddp, inputs, _N_STEPS, compile_target="whole_model")

        torch._dynamo.reset()
        torch.manual_seed(123)
        inner_model = _OpaqueModule(_DIM, _OUT_DIM).to(dtype=torch.bfloat16, device="cuda")
        inner_model.load_state_dict(init_state)
        inner_ddp = _wrap_with_megatron_ddp(inner_model)
        inner_losses = _run_ddp_training(inner_ddp, inputs, _N_STEPS, compile_target="inner_only")

        _print_comparison(
            "DDP WholeModel",
            whole_losses,
            "DDP InnerOnly",
            inner_losses,
        )

        rank = torch.distributed.get_rank()
        final_rel = abs(whole_losses[-1] - inner_losses[-1]) / max(abs(whole_losses[-1]), 1e-12)
        if rank == 0:
            print(f"\nFinal loss relative diff: {final_rel:.6f}")
        assert final_rel < 0.05, (
            f"[Rank {rank}] DDP whole_model vs inner-only diverged: "
            f"whole={whole_losses[-1]:.6f}, inner={inner_losses[-1]:.6f}, "
            f"rel_diff={final_rel:.6f}"
        )
