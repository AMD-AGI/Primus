###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for TorchTitan's TritonFusedAdamW and the patch that installs it.

``torch.optim.AdamW(fused=True)`` (what TorchTitan builds by default) is the
reference: same updates, interchangeable state dicts, DTensor shards updated
locally, unsupported configurations falling back to the stock step.
"""

import copy
import os
from types import SimpleNamespace

import pytest
import torch

from primus.backends.torchtitan.patches import triton_fused_adamw_patches as patch_mod
from primus.core.patches import PatchContext
from primus.core.patches.patch_registry import PatchRegistry

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")

HPARAMS = dict(lr=7.2e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
# Small tensors take the batched kernel, (2048, 4096) the per-tensor one.
SHAPES = [(4096,), (1000, 37), (3, 5), (2048, 4096), (1,)] + [(64, 128)] * 40


def _ctx(enabled: bool) -> PatchContext:
    params = SimpleNamespace(optimizer=SimpleNamespace(use_triton_fused_adam=enabled))
    return PatchContext(
        backend="torchtitan", phase="setup", extra={"module_config": SimpleNamespace(params=params)}
    )


def test_patch_registration_and_condition():
    patch = PatchRegistry.get(patch_mod._PATCH_ID)
    assert patch is not None
    assert patch.backend == "torchtitan"
    assert patch.applies_to(_ctx(True))
    assert not patch.applies_to(_ctx(False))


def test_job_config_keeps_primus_optimizer_flag():
    pytest.importorskip("torchtitan")
    from primus.backends.torchtitan.config_utils import build_job_config_from_namespace

    job_config = build_job_config_from_namespace(
        SimpleNamespace(optimizer=SimpleNamespace(name="AdamW", use_triton_fused_adam=True))
    )
    assert job_config.optimizer.name == "AdamW"
    assert job_config.optimizer.use_triton_fused_adam is True


def _params(shapes, dtype=torch.float32, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return [torch.nn.Parameter(torch.randn(s, device="cuda", generator=gen).to(dtype)) for s in shapes]


def _set_grads(param_lists, step):
    gen = torch.Generator(device="cuda").manual_seed(1000 + step)
    for group in zip(*param_lists):
        g = torch.randn(group[0].shape, device="cuda", generator=gen).to(group[0].dtype)
        for p in group:
            p.grad = g.clone()


def _steps(opt):
    """Per-parameter ``step`` values as a checkpoint sees them (absent: 0)."""
    state = opt.state_dict()["state"]
    n = len(opt.param_groups[0]["params"])
    return [float(state[i]["step"]) if i in state else 0.0 for i in range(n)]


def _run_pair(dtype=torch.float32, steps=4, **kwargs):
    from primus.backends.torchtitan.components.optimizer.triton_adamw import (
        TritonFusedAdamW,
    )

    ref_params = _params(SHAPES, dtype)
    tri_params = [torch.nn.Parameter(p.detach().clone()) for p in ref_params]
    ref = torch.optim.AdamW(ref_params, fused=True, **HPARAMS, **kwargs)
    tri = TritonFusedAdamW(tri_params, fused=True, **HPARAMS, **kwargs)
    for step in range(steps):
        _set_grads([ref_params, tri_params], step)
        ref.step()
        tri.step()
    torch.cuda.synchronize()
    return ref, ref_params, tri, tri_params


@requires_gpu
def test_fp32_matches_torch_fused_adamw():
    ref, ref_params, tri, tri_params = _run_pair()
    tri_steps = _steps(tri)
    for i, (pr, pt) in enumerate(zip(ref_params, tri_params)):
        torch.testing.assert_close(pt, pr, rtol=1e-5, atol=1e-6)
        for name in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(tri.state[pt][name], ref.state[pr][name], rtol=1e-5, atol=1e-7)
        assert tri.state[pt]["step"].device.type == "cpu"
        assert tri_steps[i] == ref.state[pr]["step"].item() == 4


@requires_gpu
def test_bf16_params_match_torch_fused_adamw():
    _, ref_params, _, tri_params = _run_pair(dtype=torch.bfloat16)
    for pr, pt in zip(ref_params, tri_params):
        torch.testing.assert_close(pt.float(), pr.float(), rtol=1.6e-2, atol=1e-2)


@requires_gpu
def test_state_dict_is_interchangeable_with_torch_adamw():
    from primus.backends.torchtitan.components.optimizer.triton_adamw import (
        TritonFusedAdamW,
    )

    ref, ref_params, tri, tri_params = _run_pair(steps=2)

    # load_state_dict aliases same-device tensors, so copy to get independent optimizers.
    stock = torch.optim.AdamW([torch.nn.Parameter(p.detach().clone()) for p in tri_params], **HPARAMS)
    stock.load_state_dict(copy.deepcopy(tri.state_dict()))
    restored = TritonFusedAdamW([torch.nn.Parameter(p.detach().clone()) for p in ref_params], **HPARAMS)
    restored.load_state_dict(copy.deepcopy(ref.state_dict()))

    for src, dst in ((tri, stock), (ref, restored)):
        for p_src, p_dst in zip(src.param_groups[0]["params"], dst.param_groups[0]["params"]):
            torch.testing.assert_close(dst.state[p_dst]["exp_avg"], src.state[p_src]["exp_avg"])
            assert dst.state[p_dst]["step"].item() == 2
    # A fused checkpoint keeps ``step`` on the device; loading must still use the Triton path.
    assert all(s["step"].device.type == "cpu" for s in restored.state.values())

    _set_grads([restored.param_groups[0]["params"], ref_params], 99)
    assert all(restored._locals(p, s) is not None for p, s in restored.state.items())
    restored.step()
    ref.step()
    for pr, pt in zip(ref_params, restored.param_groups[0]["params"]):
        torch.testing.assert_close(pt, pr, rtol=1e-5, atol=1e-6)


@requires_gpu
@pytest.mark.parametrize("kwargs", [dict(amsgrad=True), dict(maximize=True)])
def test_unsupported_configs_fall_back_to_torch(kwargs, monkeypatch):
    from primus.backends.torchtitan.components.optimizer import triton_adamw

    calls = []
    for name in ("triton_adam_step_", "triton_multi_tensor_adam_step_"):
        monkeypatch.setattr(triton_adamw, name, lambda *a, **k: calls.append(1))
    ref, ref_params, tri, tri_params = _run_pair(steps=2, **kwargs)
    assert calls == []
    for pr, pt in zip(ref_params, tri_params):
        torch.testing.assert_close(pt, pr, rtol=1e-5, atol=1e-6)


@requires_gpu
def test_unsupported_tensors_use_torch_within_the_same_step():
    from primus.backends.torchtitan.components.optimizer.triton_adamw import (
        TritonFusedAdamW,
    )

    shapes = [(4096,), (300, 7), (2048, 4096)]
    ref_params = _params(shapes) + [torch.nn.Parameter(torch.randn(500, device="cuda", dtype=torch.float64))]
    tri_params = [torch.nn.Parameter(p.detach().clone()) for p in ref_params]
    ref = torch.optim.AdamW(ref_params, foreach=False, **HPARAMS)
    tri = TritonFusedAdamW(tri_params, **HPARAMS)
    for step in range(3):
        _set_grads([ref_params, tri_params], step)
        ref.step()
        tri.step()
    assert tri._locals(tri_params[-1], tri.state[tri_params[-1]]) is None
    for pr, pt in zip(ref_params, tri_params):
        torch.testing.assert_close(pt, pr, rtol=1e-5, atol=1e-6)
    assert _steps(tri) == [3.0] * len(tri_params)


@requires_gpu
def test_step_counts_match_torch_when_grads_are_missing():
    from primus.backends.torchtitan.components.optimizer.triton_adamw import (
        TritonFusedAdamW,
    )

    ref_params = _params(SHAPES)
    tri_params = [torch.nn.Parameter(p.detach().clone()) for p in ref_params]
    ref = torch.optim.AdamW(ref_params, foreach=False, **HPARAMS)
    tri = TritonFusedAdamW(tri_params, **HPARAMS)
    # Parameters without a gradient in a step: none, first only, second and a late
    # starter, every one (no update at all), none.
    missing = [set(), {0}, {1, 7}, set(range(len(SHAPES))), set(), {0, 1}]
    for step, skip in enumerate(missing):
        _set_grads([ref_params, tri_params], step)
        for i in skip | ({7} if step < 2 else set()):
            ref_params[i].grad = tri_params[i].grad = None
        ref.step()
        tri.step()
        assert _steps(tri) == _steps(ref)
    for pr, pt in zip(ref_params, tri_params):
        torch.testing.assert_close(pt, pr, rtol=1e-5, atol=1e-6)


@requires_gpu
def test_closure_step_keeps_step_counts():
    from primus.backends.torchtitan.components.optimizer.triton_adamw import (
        TritonFusedAdamW,
    )

    ref_params = _params(SHAPES)
    tri_params = [torch.nn.Parameter(p.detach().clone()) for p in ref_params]
    ref = torch.optim.AdamW(ref_params, foreach=False, **HPARAMS)
    tri = TritonFusedAdamW(tri_params, **HPARAMS)
    for step in range(5):
        _set_grads([ref_params, tri_params], step)
        ref.step()
        # A closure takes the stock step, which reads and advances the ``step`` tensors.
        tri.step(closure=(lambda: None) if step == 2 else None)
    assert _steps(tri) == [5.0] * len(SHAPES)
    for pr, pt in zip(ref_params, tri_params):
        torch.testing.assert_close(pt, pr, rtol=1e-5, atol=1e-6)


@requires_gpu
def test_dcp_state_dict_round_trip():
    from torch.distributed.checkpoint.state_dict import (
        get_optimizer_state_dict,
        set_optimizer_state_dict,
    )

    from primus.backends.torchtitan.components.optimizer.triton_adamw import (
        TritonFusedAdamW,
    )

    def model():
        torch.manual_seed(0)
        return torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.Linear(64, 8)).cuda()

    def train(m, opt, steps):
        for step in range(steps):
            gen = torch.Generator(device="cuda").manual_seed(step)
            for p in m.parameters():
                p.grad = torch.randn(p.shape, device="cuda", generator=gen)
            if step == 1:
                m[1].bias.grad = None
            opt.step()

    ref_model, tri_model = model(), model()
    ref = torch.optim.AdamW(ref_model.parameters(), foreach=False, **HPARAMS)
    tri = TritonFusedAdamW(tri_model.parameters(), **HPARAMS)
    train(ref_model, ref, 3)
    train(tri_model, tri, 3)
    saved = copy.deepcopy(get_optimizer_state_dict(tri_model, tri))
    assert {k: float(v["step"]) for k, v in saved["state"].items()} == {
        k: float(v["step"]) for k, v in get_optimizer_state_dict(ref_model, ref)["state"].items()
    }

    resumed = TritonFusedAdamW(tri_model.parameters(), **HPARAMS)
    set_optimizer_state_dict(tri_model, resumed, saved)
    train(ref_model, ref, 2)
    train(tri_model, resumed, 2)
    for pr, pt in zip(ref_model.parameters(), tri_model.parameters()):
        torch.testing.assert_close(pt, pr, rtol=1e-5, atol=1e-6)


@pytest.fixture
def single_rank_mesh():
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU")
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    created = not dist.is_initialized()
    if created:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29517")
        try:
            dist.init_process_group("nccl", rank=0, world_size=1)
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"cannot initialize a single-rank process group: {exc}")
    yield init_device_mesh("cuda", (1,))
    if created:
        dist.destroy_process_group()


def test_dtensor_params_update_local_shards(single_rank_mesh, monkeypatch):
    from torch.distributed.tensor import Shard, distribute_tensor

    from primus.backends.torchtitan.components.optimizer.triton_adamw import (
        TritonFusedAdamW,
    )

    ref_params = _params(SHAPES[:5])
    tri_params = [
        torch.nn.Parameter(distribute_tensor(p.detach().clone(), single_rank_mesh, [Shard(0)]))
        for p in ref_params
    ]
    ref = torch.optim.AdamW(ref_params, fused=True, **HPARAMS)
    tri = TritonFusedAdamW(tri_params, **HPARAMS)
    monkeypatch.setattr(tri, "_torch_step", lambda *a: pytest.fail("DTensor shards must use the kernels"))
    for step in range(3):
        gen = torch.Generator(device="cuda").manual_seed(step)
        for pr, pt in zip(ref_params, tri_params):
            g = torch.randn(pr.shape, device="cuda", generator=gen)
            pr.grad = g.clone()
            pt.grad = distribute_tensor(g.clone(), single_rank_mesh, [Shard(0)])
        ref.step()
        tri.step()
    for pr, pt in zip(ref_params, tri_params):
        torch.testing.assert_close(pt.full_tensor(), pr, rtol=1e-5, atol=1e-6)


@requires_gpu
def test_patch_swaps_adamw_in_optimizers_container():
    pytest.importorskip("torchtitan")
    from torchtitan.components.optimizer import OptimizersContainer

    from primus.backends.torchtitan.components.optimizer.triton_adamw import (
        TritonFusedAdamW,
    )

    original_init = OptimizersContainer.__init__
    try:
        patch_mod.patch_torchtitan_triton_fused_adamw(_ctx(True))
        model = torch.nn.Linear(16, 16).cuda()
        container = OptimizersContainer(
            [model], torch.optim.AdamW, dict(fused=True, foreach=False, **HPARAMS)
        )
        assert all(type(opt) is TritonFusedAdamW for opt in container.optimizers)
        other = OptimizersContainer([model], torch.optim.Adam, dict(lr=1e-3))
        assert all(type(opt) is torch.optim.Adam for opt in other.optimizers)
    finally:
        OptimizersContainer.__init__ = original_init
