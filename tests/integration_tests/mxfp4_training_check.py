###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""torchrun preflight for mixed-bucket MXFP4 training communication.

GPU mode tests the production dispatch hook and prepare_weights gradient bridge,
two optimizer updates, exact quantized bytes, and BF16 materialization. CPU mode
uses a deterministic stand-in quantizer to test ownership and cache lifecycle;
it makes no claim about MXFP4 arithmetic. Neither mode needs training datasets.
"""

import argparse
import faulthandler
import importlib.util
import os
import sys
import types
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check_shared_tile_wire(wire, device, rank):
    """Validate compact reconstruction against the original dual quantizer."""
    shape = (3, 96, 160)  # Both orientations require kernel padding.
    coordinates = torch.arange(torch.Size(shape).numel(), device=device, dtype=torch.float32)
    weight = (torch.sin(coordinates * 0.037) * torch.exp2((coordinates // 1024) % 31 - 15)).to(torch.bfloat16)
    weight[::97] = -0.0
    weight = weight.view(shape)
    for mode in (0, 1, 2):
        reference = wire.MXFP4WireLayout(shape, mode)
        expected = reference.assemble(reference.quantize(weight).unsqueeze(0))
        compact = wire.MXFP4WireLayout(shape, mode, shared_2d=True)
        payload = compact.quantize(weight)
        plan = wire.MXFP4StripGatherPlan(
            compact, [(0, shape[0] * shape[1] // 32)], [0], payload.numel(), device
        )
        for actual, full_dual in zip(plan.assemble(payload), expected):
            torch.testing.assert_close(actual, full_dual, rtol=0, atol=0)
    if rank == 0:
        print("[MXFP4-COMM-PREFLIGHT] shared_tile_padding_scale_modes=PASS", flush=True)


def main():
    faulthandler.enable()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--turbo-root", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    device = torch.device("cpu" if args.cpu else f"cuda:{os.environ['LOCAL_RANK']}")
    if not args.cpu:
        torch.cuda.set_device(device)
    dist.init_process_group("gloo" if args.cpu else "nccl", timeout=timedelta(seconds=180))
    rank, world = dist.get_rank(), dist.get_world_size()
    if args.cpu:
        if args.turbo_root is None:
            parser.error("--cpu requires --turbo-root")
        wire = load(
            "primus_turbo.pytorch.core.mxfp4_comm",
            args.turbo_root / "primus_turbo/pytorch/core/mxfp4_comm.py",
        )
        namespace = types.ModuleType("mxfp4_test_runtime")
        namespace.__path__ = [str(root / "primus/backends/megatron/core/distributed")]
        sys.modules[namespace.__name__] = namespace
        runtime = load("mxfp4_test_runtime.mxfp4_training", Path(namespace.__path__[0]) / "mxfp4_training.py")

        def fake_quantize(layout, weight):
            assert torch.isfinite(weight).all(), "a remote/stale fragment reached quantization"
            torch.testing.assert_close(
                weight.flatten(),
                reference[weight.storage_offset() : weight.storage_offset() + weight.numel()],
                rtol=0,
                atol=0,
            )
            # A deterministic nibble matrix and shared tile scales support both
            # wire formats while keeping stale-source/ownership checks exact.
            values = (weight.float() * 11).to(torch.uint8) & 15
            transposed = values.transpose(-2, -1)
            g, n, k = weight.shape
            scales = torch.full((g, n // 32, k // 32), 127, dtype=torch.uint8)
            components = (
                values[..., ::2] | (values[..., 1::2] << 4),
                scales.repeat_interleave(32, dim=1),
                transposed[..., ::2] | (transposed[..., 1::2] << 4),
                scales.transpose(1, 2).repeat_interleave(32, dim=1),
            )
            return layout.pack([part.contiguous() for part in components])

        wire.MXFP4WireLayout.quantize = fake_quantize
        wire.MXFP4WireLayout.wrap_components = lambda _self, components, shape=None: components
    else:
        from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
        from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBucketGroup
        from primus_turbo.pytorch.core import mxfp4_comm as wire
        from primus_turbo.pytorch.core.low_precision import Float4QuantConfig
        from primus_turbo.pytorch.ops.grouped_gemm_fp4 import grouped_gemm_fp4
        from primus_turbo.pytorch.ops.grouped_mlp_fp4 import grouped_mlp_fp4

        from primus.backends.megatron.core.distributed import mxfp4_training as runtime
        from primus.backends.megatron.core.extensions import primus_turbo as extension
        from primus.backends.megatron.patches.parallelism.rccl_sdma_param_all_gather_patches import (
            make_ddp_init,
            make_start_param_sync,
        )

        os.environ["MEGATRON_MXFP4_PARAM_GATHER"] = "1"
        os.environ["PRIMUS_TURBO_MXFP4_SCALE_ROUNDING"] = "2"
        config = Float4QuantConfig(scale_rounding_mode=2, use_gradient_sr=False)
        manager = extension.PrimusTurboLowPrecisionGlobalStateManager
        manager.PRIMUS_TURBO_FP4_ENABLED = True
        manager.PRIMUS_TURBO_QUANT_CONFIG = types.SimpleNamespace(
            data=lambda: config,
            mxfp4_scaling=lambda: True,
        )
        if os.getenv("MEGATRON_MXFP4_PARAM_GATHER_FORMAT", "dual") == "shared_2d":
            check_shared_tile_wire(wire, device, rank)

    # Non-aligned offsets guarantee strips cross rank boundaries, including an
    # expert boundary. Ordinary BF16 parameters share the same bucket.
    # Use a supported fused-MLP shape in GPU mode. Its activation-quantization
    # epilogue needs at least four K iterations and a trailing 128-element tile;
    # the smaller CPU fixture tests ownership without invoking that kernel.
    shapes = [(3, 256, 128), (3, 128, 128)] if args.cpu else [(4, 1024, 896), (4, 896, 512)]
    starts = [127, 127 + torch.Size(shapes[0]).numel()]
    last = starts[1] + torch.Size(shapes[1]).numel()
    # A substantial ordinary tail also exercises ranks with no expert strips,
    # as in a training bucket that contains embedding or output weights.
    ordinary_tail = max(259, last // 4)
    numel = ((last + ordinary_tail + world * 128 - 1) // (world * 128)) * world * 128
    shard_size = numel // world
    reference_master = torch.sin(torch.arange(numel, device=device).float() * 0.037)
    reference = reference_master.to(torch.bfloat16)
    local_master = reference_master[rank * shard_size : (rank + 1) * shard_size].clone()
    raw_workspace = None
    if args.cpu:
        storage = torch.empty(numel, dtype=torch.bfloat16, device=device)
    else:
        raw_workspace = runtime.ByteAllGather(numel * 2 // world, dist.group.WORLD, device, True)
        storage = raw_workspace.storage.view(torch.bfloat16)
    params, mapping = [], {}
    for shape, start in zip(shapes, starts):
        param = torch.nn.Parameter(storage[start : start + torch.Size(shape).numel()].view(shape))
        param._primus_mxfp4_comm_candidate = True
        params.append(param)
        mapping[param] = (start, start + param.numel())
    for start, end in ((0, 127), (last, numel)):
        mapping[torch.nn.Parameter(storage[start:end])] = (start, end)
    bucket = types.SimpleNamespace(
        param_data=storage, param_to_index=mapping, params=set(mapping), params_list=list(mapping)
    )
    state = None
    try:
        if args.cpu:
            state = runtime.PackedExpertBucket(bucket, dist.group.WORLD, use_sdma=False)
        else:

            class BucketGroup(_ParamAndGradBucketGroup):
                start_param_sync = make_start_param_sync(_ParamAndGradBucketGroup.start_param_sync)

            ddp_config = DistributedDataParallelConfig(
                use_distributed_optimizer=True, overlap_param_gather=True
            )
            bucket_group = BucketGroup([bucket], ddp_config, dist.group.WORLD, world)

            class DDPBinding:
                @make_ddp_init
                def __init__(self):
                    self.ddp_config = ddp_config
                    self.overlap_param_gather_with_optimizer_step = False
                    self.param_to_bucket_group = {param: bucket_group for param in mapping}

            DDPBinding()
            # Match DistributedOptimizer's later partition_buckets call over
            # the same buffers. These bookkeeping groups must not own hooks.
            bookkeeping_group = BucketGroup([bucket], ddp_config, dist.group.WORLD, world)

            class NextBucketProbe:
                param_gather_dispatched = False
                dispatches = 0

                def start_param_sync(self):
                    self.param_gather_dispatched = True
                    self.dispatches += 1

            next_bucket = NextBucketProbe()
            bucket_group.next_param_gather_bucket_group = next_bucket
        for step in range(3):
            storage.fill_(float("nan"))
            storage[rank * shard_size : (rank + 1) * shard_size].copy_(local_master.to(torch.bfloat16))
            if args.cpu:
                state.dispatch().wait()
            else:
                bucket_group.param_gather_dispatched = False
                next_bucket.param_gather_dispatched = False
                next_bucket.dispatches = 0
                # Step 1 exercises the path used by a fused MLP that bypasses
                # the Linear module's DDP forward hook entirely.
                if step == 1:
                    params[0]._primus_mxfp4_ensure_ready()
                else:
                    bucket_group.start_param_sync(force_sync=step == 2)
                    params[0]._primus_mxfp4_ensure_ready()
                    bucket_group.finish_param_sync()
                state = bucket._primus_mxfp4_gather
                assert not bookkeeping_group.param_gather_dispatched
                assert state.generation == step + 1, "duplicate parameter gather"
                assert next_bucket.dispatches == (1 if step < 2 else 0), "lost next-bucket prefetch"
            # Ordinary BF16 weights must be ready before any forward consumer,
            # independently of the explicit full-BF16 materialization below.
            for param, (start, end) in mapping.items():
                if not getattr(param, "_primus_mxfp4_comm_candidate", False):
                    torch.testing.assert_close(param.detach(), reference[start:end], rtol=0, atol=0)
            actual_grad = torch.zeros_like(storage)
            expected_grad = torch.zeros_like(storage)
            for param, start, shape in zip(params, starts, shapes):
                layout = wire.MXFP4WireLayout(shape, 2)
                full = reference[start : start + param.numel()].view(shape)
                expected_bytes = layout.assemble(layout.quantize(full).unsqueeze(0))
                weight_state = param._primus_mxfp4_comm_state
                pair = weight_state.get_pair(2)
                assert pair is weight_state.get_pair(2), "cache was not reused within an update"
                actual_bytes = (
                    pair
                    if args.cpu
                    else (pair.data.qdata, pair.data.scale_inv, pair.data_t.qdata, pair.data_t.scale_inv)
                )
                for actual, expected in zip(actual_bytes, expected_bytes):
                    torch.testing.assert_close(actual.view(torch.uint8), expected, rtol=0, atol=0)
                if args.cpu:
                    actual_grad[start : start + param.numel()].fill_(rank + 1 + step)
                    expected_grad[start : start + param.numel()].fill_(rank + 1 + step)
                else:
                    # Exercise the actual single-microbatch consumer and weight
                    # gradient bridge, both fused and unfused across updates.
                    module = extension.PrimusTurboGroupedLinear.__new__(extension.PrimusTurboGroupedLinear)
                    torch.nn.Module.__init__(module)
                    module.weights = param
                    module.config = types.SimpleNamespace(gradient_accumulation_fusion=step != 0)
                    module.is_first_microbatch = True
                    module._weight_views_registered = True
                    param.main_grad = actual_grad[start : start + param.numel()].view(shape)
                    param.grad_added_to_main_grad = False
                    generator = torch.Generator(device=device).manual_seed(1701 + rank + step)
                    x = torch.randn(
                        shape[0] * 128, shape[-1], device=device, dtype=torch.bfloat16, generator=generator
                    ).requires_grad_(True)
                    xr, wr = x.detach().clone().requires_grad_(True), full.detach().clone().requires_grad_(
                        True
                    )
                    lens = torch.full((shape[0],), 128, device=device, dtype=torch.int64)
                    bridged_x, cached_weight, pattern = module.prepare_weights(x)
                    out = grouped_gemm_fp4(
                        bridged_x, cached_weight, lens, config=config, fuse_bgrad_accum_pattern=pattern
                    )
                    expected_out = grouped_gemm_fp4(xr, wr, lens, config=config)
                    cotangent = torch.randn(out.shape, device=device, dtype=out.dtype, generator=generator)
                    out.backward(cotangent)
                    expected_out.backward(cotangent)
                    torch.testing.assert_close(out, expected_out, rtol=0, atol=0)
                    torch.testing.assert_close(x.grad, xr.grad, rtol=0, atol=0)
                    torch.testing.assert_close(param.main_grad, wr.grad, rtol=0, atol=0)
                    assert param.grad_added_to_main_grad
                    expected_grad[start : start + param.numel()].copy_(wr.grad.flatten())
                    param.grad = None
            if not args.cpu:
                # GPT-OSS uses this fused FC1/SwiGLU/FC2 path. Exercise its
                # gradient bridge separately from the individual GEMM checks.
                actual_grad.zero_()
                expected_grad.zero_()
                modules = []
                for param in params:
                    module = extension.PrimusTurboGroupedLinear.__new__(extension.PrimusTurboGroupedLinear)
                    torch.nn.Module.__init__(module)
                    module.weights = param
                    module.config = types.SimpleNamespace(gradient_accumulation_fusion=step != 0)
                    module.is_first_microbatch = True
                    module._weight_views_registered = True
                    param.grad_added_to_main_grad = False
                    modules.append(module)
                groups, _, hidden = shapes[0]
                tokens_per_group = 512
                tokens = groups * tokens_per_group
                x = torch.randn(
                    tokens, hidden, device=device, dtype=torch.bfloat16, generator=generator
                ).requires_grad_(True)
                probs = torch.rand(
                    tokens, device=device, dtype=torch.float32, generator=generator
                ).requires_grad_(True)
                xr = x.detach().clone().requires_grad_(True)
                pr = probs.detach().clone().requires_grad_(True)
                refs = [
                    reference[start : start + param.numel()].view(shape).detach().clone().requires_grad_(True)
                    for param, start, shape in zip(params, starts, shapes)
                ]
                bridged_x, w1, pattern = modules[0].prepare_weights(x)
                bridged_x, w2, pattern2 = modules[1].prepare_weights(bridged_x)
                assert pattern == pattern2
                lens = torch.full((groups,), tokens_per_group, device=device, dtype=torch.int64)
                out = grouped_mlp_fp4(
                    bridged_x,
                    w1,
                    w2,
                    lens,
                    probs=probs,
                    config=config,
                    trans_w1=True,
                    trans_w2=True,
                    activation="silu",
                    clamp_limit=7.0,
                    fuse_wgrad_accum_pattern=pattern,
                )
                expected_out = grouped_mlp_fp4(
                    xr,
                    refs[0],
                    refs[1],
                    lens,
                    probs=pr,
                    config=config,
                    trans_w1=True,
                    trans_w2=True,
                    activation="silu",
                    clamp_limit=7.0,
                )
                cotangent = torch.randn(out.shape, device=device, dtype=out.dtype, generator=generator)
                out.backward(cotangent)
                expected_out.backward(cotangent)
                for actual, expected in ((out, expected_out), (x.grad, xr.grad), (probs.grad, pr.grad)):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for param, ref, start in zip(params, refs, starts):
                    torch.testing.assert_close(param.main_grad, ref.grad, rtol=0, atol=0)
                    expected_grad[start : start + param.numel()].copy_(ref.grad.flatten())
                    param.grad = None
                if rank == 0:
                    print(f"[MXFP4-COMM-PREFLIGHT] step={step} fused_mlp_parity=PASS", flush=True)
            # Saving a checkpoint must see exact BF16, never dequantized MXFP4.
            state.materialize_bf16()
            torch.testing.assert_close(storage, reference, rtol=0, atol=0)
            if rank == 0:
                print(f"[MXFP4-COMM-PREFLIGHT] step={step} bf16_materialization=PASS", flush=True)
            reduced = torch.empty(shard_size, device=device, dtype=torch.bfloat16)
            expected_reduced = torch.empty_like(reduced)
            # Match production's ProcessGroupNCCL stream path. Synchronous CE
            # calls can enter RCCL's null-stream fallback, which does not safely
            # handle the imported ROCr VMM pointers used by this transport.
            dist.reduce_scatter_tensor(reduced, actual_grad, group=dist.group.WORLD, async_op=True).wait()
            # Match the baseline's reduction operation: BF16 AllReduce may
            # associate additions differently from ReduceScatter. That rounding
            # difference is unrelated to quantized parameter communication.
            dist.reduce_scatter_tensor(
                expected_reduced, expected_grad, group=dist.group.WORLD, async_op=True
            ).wait()
            torch.testing.assert_close(reduced, expected_reduced, rtol=0, atol=0)
            dist.all_gather_into_tensor(
                expected_grad, expected_reduced, group=dist.group.WORLD, async_op=True
            ).wait()
            local_master.add_(reduced.float(), alpha=-0.01 / world)
            reference_master.add_(expected_grad.float(), alpha=-0.01 / world)
            reference = reference_master.to(torch.bfloat16)
            if rank == 0:
                print(
                    f"[MXFP4-COMM-PREFLIGHT] step={step} PASS backend={'gloo-simulated-quantizer' if args.cpu else 'rccl-sdma'}",
                    flush=True,
                )
        if rank == 0:
            print(
                "[MXFP4-COMM-PREFLIGHT] PASS updates=2 cache_refresh=3 bf16_materialization=exact", flush=True
            )
    finally:
        if state is not None:
            state.close()
        bucket.param_data = None
        # Ordinary BF16 parameters also alias the raw symmetric allocation.
        # Release every parameter view before destroying its private pool.
        for param in mapping:
            param.data = torch.empty(0, device=device, dtype=torch.bfloat16)
        params.clear()
        mapping.clear()
        storage = None
        if raw_workspace is not None:
            raw_workspace.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
