# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Integration tests for Flux model with FSDP2 FP8 all-gather (Approach A).

Single-GPU tests verify wrapping, forward, and forward+backward using
the init_parallel_state fixture (world_size=1, all-reduce is a no-op).

Multi-GPU tests (torchrun) verify real FP8 all-gather communication,
training steps, numerical parity, and torch.compile compatibility.

Run multi-GPU tests with:
    torchrun --nproc_per_node 2 --master_port 29501 -m pytest -xvs \\
        tests/integration_tests/backends/megatron/diffusion/distributed/test_flux_fsdp2_fp8_all_gather_integration.py
"""

import os
from types import SimpleNamespace

import pytest
import torch
from megatron.core.distributed.distributed_data_parallel_config import (
    DistributedDataParallelConfig,
)

from primus.backends.megatron.core.distributed.fsdp2_fp8_all_gather import (
    WeightWithFP8AllGatherTensor,
)
from primus.backends.megatron.core.distributed.torch_fully_sharded_data_parallel import (
    PrimusTorchFullyShardedDataParallel,
)
from primus.backends.megatron.core.models.diffusion.flux.config import FluxConfig
from primus.backends.megatron.core.models.diffusion.flux.model import Flux
from primus.backends.megatron.core.models.diffusion.flux.utils import (
    generate_image_position_ids,
    pack_latents,
)
from tests.unit_tests.backends.megatron.diffusion.constants import (
    CLIP_L_EMBEDDING_DIM,
    IMG_SIZE_TINY,
    T5_XXL_EMBEDDING_DIM,
    TEXT_SEQ_LEN_SHORT,
    VAE_LATENT_CHANNELS,
)
from tests.utils import PrimusUT

_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 0))

requires_torchrun = pytest.mark.skipif(
    _WORLD_SIZE < 1,
    reason="Requires torchrun (WORLD_SIZE env var not set)",
)


def _make_flux_inputs(batch_size, device):
    height, width = IMG_SIZE_TINY, IMG_SIZE_TINY
    img = torch.randn(
        batch_size,
        VAE_LATENT_CHANNELS,
        height,
        width,
        dtype=torch.bfloat16,
        device=device,
    )
    packed_img = pack_latents(img).transpose(0, 1)
    txt = torch.randn(
        batch_size,
        TEXT_SEQ_LEN_SHORT,
        T5_XXL_EMBEDDING_DIM,
        dtype=torch.bfloat16,
        device=device,
    ).transpose(0, 1)
    y = torch.randn(batch_size, CLIP_L_EMBEDDING_DIM, dtype=torch.bfloat16, device=device)
    timesteps = torch.rand(batch_size, dtype=torch.bfloat16, device=device)
    img_ids = generate_image_position_ids(batch_size, height, width, device=device)
    txt_ids = torch.zeros(batch_size, TEXT_SEQ_LEN_SHORT, 3, dtype=torch.bfloat16, device=device)
    return dict(img=packed_img, txt=txt, y=y, timesteps=timesteps, img_ids=img_ids, txt_ids=txt_ids)


# ============================================================================
# Single-GPU tests (init_parallel_state fixture, world_size=1)
# ============================================================================


class TestFluxFSDP2FP8AllGather(PrimusUT):

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state):
        from megatron.training.global_vars import set_args

        set_args(
            SimpleNamespace(
                enable_turbo_attention_float8=False,
                data_parallel_replicate_degree=1,
                use_fsdp2_fp8_all_gather=True,
                use_fsdp2_bf16_master_weight_optimizer=True,
                use_fsdp2_fp32_param_optimizer=False,
                torch_fsdp2_reshard_after_forward=False,
            )
        )

    def _make_fsdp_model(self):
        config = FluxConfig.flux_535m(
            transformer_impl="local",
            fp8="e4m3",
            fp8_recipe="tensorwise",
        )
        model = Flux(config).cuda().to(torch.bfloat16)
        ddp_config = DistributedDataParallelConfig()
        fsdp_wrapper = PrimusTorchFullyShardedDataParallel(
            config=config,
            ddp_config=ddp_config,
            module=model,
        )
        return model, fsdp_wrapper

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_fsdp2_fp8_allgather_wrapping(self):
        """Verify only weight params of FP8-aware modules are wrapped."""
        from torch.distributed._tensor import DTensor

        model, _ = self._make_fsdp_model()

        expected_fp8_weights = sum(
            1
            for m in model.modules()
            if hasattr(m, "_fp8_config") and hasattr(m, "weight") and m.weight is not None
        )

        wrapped_count = 0
        unwrapped_count = 0
        for m in model.modules():
            if not hasattr(m, "weight") or m.weight is None:
                continue
            w = m.weight
            local_t = w._local_tensor if isinstance(w, DTensor) else w
            if isinstance(local_t, WeightWithFP8AllGatherTensor):
                wrapped_count += 1
                assert local_t._precomputed_scale is not None, "precomputed scale not set after FSDP init"
                assert hasattr(m, "_fp8_config"), f"Non-FP8 module {type(m).__name__} should not be wrapped"
            elif w.requires_grad:
                unwrapped_count += 1

        assert (
            wrapped_count == expected_fp8_weights
        ), f"Expected {expected_fp8_weights} wrapped FP8 weights, got {wrapped_count}"
        assert unwrapped_count > 0, "Expected some unwrapped params (biases, non-FP8 layers)"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_fsdp2_fp8_allgather_forward(self):
        """Forward pass produces valid output with correct shape."""
        model, _ = self._make_fsdp_model()
        model.eval()
        inputs = _make_flux_inputs(batch_size=2, device="cuda")
        with torch.no_grad():
            output = model(**inputs)
        assert output is not None
        assert len(output.shape) == 3
        assert output.shape[1] == 2
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_fsdp2_fp8_allgather_forward_backward(self):
        """Forward + backward: completes without error, grads allocated for both param types."""
        from torch.distributed._tensor import DTensor

        model, _ = self._make_fsdp_model()
        model.train()
        inputs = _make_flux_inputs(batch_size=2, device="cuda")
        output = model(**inputs)
        loss = output.float().sum()
        loss.backward()

        has_any_nonzero_grad = False
        fp8_grad_allocated = False
        non_fp8_grad_allocated = False
        for param in model.parameters():
            if param.grad is None:
                continue
            local_t = param._local_tensor if isinstance(param, DTensor) else param
            if isinstance(local_t, WeightWithFP8AllGatherTensor):
                fp8_grad_allocated = True
            else:
                non_fp8_grad_allocated = True
            if param.grad.abs().sum() > 0:
                has_any_nonzero_grad = True
        assert has_any_nonzero_grad, "No non-zero gradients found after backward"
        assert fp8_grad_allocated, "No grad tensors allocated for FP8-wrapped params"
        assert non_fp8_grad_allocated, "No grad tensors allocated for non-wrapped params"


# ============================================================================
# Multi-GPU tests (torchrun, real FP8 all-gather communication)
# ============================================================================


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
            exp_name="fp8_allgather_integration",
            module_name="test",
            rank=rank,
            world_size=world_size,
        )
        primus_logger.setup_logger(cfg, is_head=(rank == 0))

    from primus.core.utils.module_utils import set_logging_rank

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
            use_fsdp2_fp8_all_gather=True,
            use_fsdp2_bf16_master_weight_optimizer=True,
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


def _make_optimizer_config():
    from megatron.core.optimizer.optimizer_config import OptimizerConfig

    return OptimizerConfig(
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


def _create_model_and_optimizer(seed=42, fp8_all_gather=True):
    from primus.backends.megatron.core.optimizer.fsdp2_bf16_master_weight_optimizer import (
        get_fsdp2_bf16_master_weight_optimizer,
    )

    torch.manual_seed(seed)
    flux_config = FluxConfig.flux_535m(
        transformer_impl="local",
        fp8="e4m3" if fp8_all_gather else None,
        fp8_recipe="tensorwise" if fp8_all_gather else None,
    )
    model = Flux(flux_config).cuda().to(torch.bfloat16)

    ddp_config = DistributedDataParallelConfig()

    # Temporarily set the fp8 all-gather flag for FSDP wrapping
    from megatron.training import get_args

    args = get_args()
    original_flag = getattr(args, "use_fsdp2_fp8_all_gather", False)
    args.use_fsdp2_fp8_all_gather = fp8_all_gather

    fsdp_wrapper = PrimusTorchFullyShardedDataParallel(
        config=flux_config,
        ddp_config=ddp_config,
        module=model,
    )

    args.use_fsdp2_fp8_all_gather = original_flag

    opt_config = _make_optimizer_config()
    optimizer = get_fsdp2_bf16_master_weight_optimizer(
        config=opt_config,
        model_chunks=[model],
    )
    return model, fsdp_wrapper, optimizer


def _train_steps(model, optimizer, n_steps, batch_size=2, device="cuda", fp8_all_gather=True):
    from primus.backends.megatron.core.distributed.fsdp2_fp8_all_gather import (
        precompute_fp8_scales_for_fsdp,
    )

    losses = []
    for _ in range(n_steps):
        optimizer.zero_grad()
        inputs = _make_flux_inputs(batch_size, device)
        output = model(**inputs)
        loss = output.float().sum()
        loss.backward()
        optimizer.step()
        if fp8_all_gather:
            precompute_fp8_scales_for_fsdp(model)
        losses.append(loss.item())
    return losses


class TestFluxFSDP2FP8AllGatherMultiGPU:

    def setup_method(self):
        _init_distributed()

    def teardown_method(self):
        _cleanup_distributed()

    @requires_torchrun
    def test_fsdp2_fp8_allgather_train_steps(self):
        """Train 5 steps with FP8 all-gather, verify all losses are finite."""
        model, _, optimizer = _create_model_and_optimizer(seed=42)
        losses = _train_steps(model, optimizer, n_steps=5)
        assert len(losses) == 5
        for i, loss_val in enumerate(losses):
            assert torch.isfinite(torch.tensor(loss_val)), f"Non-finite loss at step {i}: {loss_val}"

    @requires_torchrun
    def test_fsdp2_fp8_allgather_numerical_parity(self):
        """Compare FP8-all-gather loss vs BF16-only baseline over 3 steps."""
        torch.manual_seed(42)
        model_fp8, _, opt_fp8 = _create_model_and_optimizer(
            seed=42,
            fp8_all_gather=True,
        )
        losses_fp8 = _train_steps(
            model_fp8,
            opt_fp8,
            n_steps=3,
            fp8_all_gather=True,
        )

        torch.manual_seed(42)
        model_bf16, _, opt_bf16 = _create_model_and_optimizer(
            seed=42,
            fp8_all_gather=False,
        )
        losses_bf16 = _train_steps(
            model_bf16,
            opt_bf16,
            n_steps=3,
            fp8_all_gather=False,
        )

        for i, (fp8_loss, bf16_loss) in enumerate(zip(losses_fp8, losses_bf16)):
            torch.testing.assert_close(
                torch.tensor(fp8_loss),
                torch.tensor(bf16_loss),
                rtol=0.05,
                atol=1.0,
                msg=f"Loss diverged at step {i}: FP8={fp8_loss}, BF16={bf16_loss}",
            )

    @requires_torchrun
    def test_fsdp2_fp8_allgather_with_compile(self):
        """Train steps with torch.compile + FP8 all-gather."""
        model, _, optimizer = _create_model_and_optimizer(seed=42)
        model.compile_model()
        losses = _train_steps(model, optimizer, n_steps=3)
        assert len(losses) == 3
        for i, loss_val in enumerate(losses):
            assert torch.isfinite(torch.tensor(loss_val)), f"Non-finite loss at step {i}: {loss_val}"
