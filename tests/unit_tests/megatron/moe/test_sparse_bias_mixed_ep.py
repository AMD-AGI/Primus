# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Run with torchrun --standalone --nproc-per-node=8 -m pytest -q <this file>."""

import os
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist


@pytest.mark.skipif(
    torch.version.hip is None
    or os.environ.get("LOCAL_WORLD_SIZE") != "8"
    or os.environ.get("WORLD_SIZE") != "8",
    reason="requires an eight-GPU ROCm torchrun",
)
def test_mixed_ep_sparse_bias_collectives_and_gradients(monkeypatch):
    from primus_turbo.pytorch.ops.moe.fused_mega_moe_fp8 import (
        FusedMegaMoEFP8Stage2Function,
    )

    from primus.backends.megatron.core.extensions import mega_moe
    from primus.backends.megatron.core.extensions.sparse_expert_bias import (
        SparseExpertBias,
        sparse_expert_bias_fallback_reason,
    )

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    owns_group = not dist.is_initialized()
    if owns_group:
        dist.init_process_group("nccl", timeout=timedelta(minutes=3))
    ep = dist.group.WORLD
    precision = torch.get_float32_matmul_precision()
    try:
        arch = getattr(torch.cuda.get_device_properties(torch.cuda.current_device()), "gcnArchName", "")
        ready = torch.tensor(int(arch.split(":")[0] == "gfx950"), device="cuda")
        dist.all_reduce(ready, op=dist.ReduceOp.MIN, group=ep)
        if not ready.item():
            pytest.skip("requires gfx950 on every rank")
        torch.set_float32_matmul_precision("highest")
        rank = ep.rank()
        tokens = 73 if rank % 2 == 0 else 0
        gen = torch.Generator(device="cuda").manual_seed(8754 + rank)
        x = torch.randn(tokens, 2880, dtype=torch.bfloat16, generator=gen, device="cuda", requires_grad=True)
        dy = torch.randn(tokens, 2880, dtype=x.dtype, generator=gen, device="cuda")
        bias = torch.randn(16, 2880, dtype=x.dtype, generator=gen, device="cuda", requires_grad=True)
        logits = torch.randn(tokens, 4, generator=gen, device="cuda", requires_grad=True)
        # Nonempty ranks route to every owner, including the owners with no local tokens.
        ids = (torch.arange(tokens * 4, device="cuda").reshape(tokens, 4) + 11 * rank) % 128

        # Isolate the FC2 bias edge: pass a real BF16 prebias leaf through the
        # expert matmul, leaving the public Turbo stage2 and its bias operation intact.
        monkeypatch.setattr(mega_moe, "fused_mega_moe_fp8_stage1", lambda x, *args: (x, None, (), None))
        monkeypatch.setattr(FusedMegaMoEFP8Stage2Function, "apply", staticmethod(lambda x, *args: x))
        monkeypatch.setattr(mega_moe, "_log_once", lambda msg: None)
        monkeypatch.setattr(mega_moe, "log_rank_all", lambda msg: None)
        layer = SimpleNamespace(
            ep_group=ep,
            training=True,
            activation=None,
            fc1_weight=lambda: (None, None),
            fc2_weight=lambda: (None, bias),
            _sparse_bias_ops=None,
        )
        calls = []
        for name in ("all_gather_into_tensor", "reduce_scatter_tensor"):
            original = getattr(dist, name)

            def record(out, src, *args, name=name, original=original, **kwargs):
                calls.append((name, src.shape, out.shape, src.dtype, out.dtype, kwargs.get("group") is ep))
                return original(out, src, *args, **kwargs)

            monkeypatch.setattr(dist, name, record)

        results, traces, paths = [], [], []
        for mixed in (False, True):
            layer._sparse_bias_ops = (
                (sparse_expert_bias_fallback_reason, SparseExpertBias.apply) if mixed else None
            )
            probs = logits.softmax(1)
            y = mega_moe.MegaMoEFP8Experts.forward(layer, x, ids, probs)
            paths.append(type(y.grad_fn).__name__)
            results.append((y.detach(), *torch.autograd.grad(y, (x, probs, logits, bias), dy)))
            traces.append(calls.copy())
            calls.clear()

        expected = [
            ("all_gather_into_tensor", (16, 2880), (128, 2880), torch.bfloat16, torch.bfloat16, True),
            ("reduce_scatter_tensor", (128, 2880), (16, 2880), torch.float32, torch.float32, True),
        ]
        assert traces == [expected, expected]
        assert paths[0] != "SparseExpertBiasBackward"
        assert paths[1] == ("SparseExpertBiasBackward" if tokens else paths[0])
        y, dx, dp, dl, db = results[1]
        ry, rdx, rdp, rdl, rdb = results[0]
        torch.testing.assert_close(dx, rdx, rtol=0, atol=0)
        torch.testing.assert_close(dx, dy, rtol=0, atol=0)
        torch.testing.assert_close(db, rdb, rtol=0, atol=0)
        assert (db != 0).any(dim=1).all()
        for got, ref, rel, peak in ((y, ry, 5e-4, 0.06), (dp, rdp, 2e-5, 2.5e-4), (dl, rdl, 2e-5, 2.5e-4)):
            assert got.shape == ref.shape
            if got.numel():
                error = (got.float() - ref.float()).abs()
                assert error.norm() <= rel * ref.float().norm()
                assert error.max() <= peak * ref.float().square().mean().sqrt()
    finally:
        torch.set_float32_matmul_precision(precision)
        if owns_group:
            dist.destroy_process_group()
