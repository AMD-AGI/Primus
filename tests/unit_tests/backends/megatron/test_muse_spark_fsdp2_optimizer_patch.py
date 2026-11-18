###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The FSDP2 fp32-grad patch, run against upstream's real ``_copy_model_grads_to_main_grads``.

Upstream is not stubbed: the test checks the gradients that reach the optimizer's
main params. A one-rank gloo group gives real ``DTensor`` params (CPU).
"""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

if not dist.is_available():
    pytest.skip("torch.distributed is not available in this build", allow_module_level=True)
pytest.importorskip("megatron.core")
from megatron.core.optimizer import optimizer as opt_module  # noqa: E402
from torch.distributed.device_mesh import init_device_mesh  # noqa: E402
from torch.distributed.tensor import DTensor, Shard, distribute_tensor  # noqa: E402

from primus.backends.megatron.core.models.muse_spark.model_types import (  # noqa: E402
    mark_muse_spark_param,
)
from primus.backends.megatron.patches import (  # noqa: E402
    muse_spark_fsdp2_patches as patch,
)
from primus.backends.megatron.patches import (  # noqa: E402
    muse_spark_optimizer_ckpt_patches as patch_ckpt,
)

CLS = opt_module.Float16OptimizerWithFloat16Params


@pytest.fixture(scope="module")
def mesh(tmp_path_factory):
    # a FileStore: no MASTER_ADDR / MASTER_PORT written to os.environ, no fixed TCP port
    # that could collide with another job on the host
    created = False
    if not dist.is_initialized():
        store = dist.FileStore(str(tmp_path_factory.mktemp("pg") / "store"), 1)
        dist.init_process_group("gloo", store=store, rank=0, world_size=1)
        created = True
    yield init_device_mesh("cpu", (1,))
    if created:
        dist.destroy_process_group()


@pytest.fixture
def patched(monkeypatch):
    # the patch replaces a class attribute: restore upstream after each test
    monkeypatch.setattr(CLS, "_copy_model_grads_to_main_grads", CLS._copy_model_grads_to_main_grads)
    patch.patch_muse_spark_fsdp2_fp32_param_grads(SimpleNamespace())
    return CLS._copy_model_grads_to_main_grads


def _dparam(mesh, value, dtype=torch.float32):
    p = torch.nn.Parameter(distribute_tensor(torch.full((4,), value, dtype=dtype), mesh, [Shard(0)]))
    assert isinstance(p, DTensor)
    return p


def test_grads_reach_the_optimizer(mesh, patched):
    # FSDP2 fp32 island param: reduce-scattered grad in .grad, no main_grad
    fsdp_fp32 = mark_muse_spark_param(_dparam(mesh, 1.0))  # e.g. a Muse Spark router weight
    fsdp_fp32.grad = distribute_tensor(torch.full((4,), 3.0), mesh, [Shard(0)])
    # a DTensor that does have main_grad must still take it (only the no-main_grad half is hidden)
    dt_main = _dparam(mesh, 1.0)
    dt_main.main_grad = distribute_tensor(torch.full((4,), 5.0), mesh, [Shard(0)])
    # DDP-style plain fp32 param with main_grad
    ddp = torch.nn.Parameter(torch.zeros(4))
    ddp.main_grad = torch.full((4,), 7.0)
    # FSDP2 bf16 param (float16 group): upstream copies .grad into the fp32 main param
    bf16 = _dparam(mesh, 1.0, torch.bfloat16)
    bf16.grad = distribute_tensor(torch.full((4,), 2.0, dtype=torch.bfloat16), mesh, [Shard(0)])
    main_bf16 = _dparam(mesh, 1.0)

    groups = [[fsdp_fp32, dt_main], [ddp]]
    opt = SimpleNamespace(
        float16_groups=[[bf16]], fp32_from_float16_groups=[[main_bf16]], fp32_from_fp32_groups=groups
    )
    patched(opt)

    assert torch.equal(fsdp_fp32.grad.full_tensor(), torch.full((4,), 3.0))  # kept
    assert dt_main.grad is dt_main.main_grad
    assert ddp.grad is ddp.main_grad
    assert main_bf16.grad.dtype == torch.float32
    assert torch.equal(main_bf16.grad.full_tensor(), torch.full((4,), 2.0)) and bf16.grad is None
    assert opt.fp32_from_fp32_groups is groups  # restored


def test_plain_param_without_main_grad_still_fails(mesh, patched):
    plain = torch.nn.Parameter(torch.zeros(4))  # not FSDP2: the patch must not hide it
    groups = [[plain, mark_muse_spark_param(_dparam(mesh, 1.0))]]
    opt = SimpleNamespace(float16_groups=[], fp32_from_float16_groups=[], fp32_from_fp32_groups=groups)
    with pytest.raises(AttributeError, match="main_grad"):
        patched(opt)
    assert opt.fp32_from_fp32_groups is groups  # restored on the error path too (finally)


def test_a_non_muse_optimizer_takes_upstreams_path(mesh, patched):
    # the class method stays replaced for the process: another model's FSDP2 fp32 DTensor
    # param without main_grad must still fail in upstream's loop, not be skipped
    other = _dparam(mesh, 1.0)
    other.grad = distribute_tensor(torch.full((4,), 3.0), mesh, [Shard(0)])
    groups = [[other]]
    opt = SimpleNamespace(float16_groups=[], fp32_from_float16_groups=[], fp32_from_fp32_groups=groups)
    with pytest.raises(AttributeError, match="main_grad"):
        patched(opt)
    assert opt.fp32_from_fp32_groups is groups


def test_the_muse_check_runs_once_per_optimizer(mesh, patched, monkeypatch):
    calls = []
    real = patch_ckpt.holds_muse_spark_params
    # the name the fsdp2 patch imported (module-level import)
    monkeypatch.setattr(patch, "holds_muse_spark_params", lambda o: calls.append(o) or real(o))
    p = mark_muse_spark_param(_dparam(mesh, 1.0))
    p.grad = distribute_tensor(torch.full((4,), 3.0), mesh, [Shard(0)])
    groups = [[p]]
    opt = SimpleNamespace(float16_groups=[], fp32_from_float16_groups=[], fp32_from_fp32_groups=groups)
    for _ in range(3):  # one check, then the cached groups
        patched(opt)
        assert opt.fp32_from_fp32_groups is groups
    assert calls == [opt]
    opt.fp32_from_fp32_groups = new = [[p]]  # new groups object: checked again
    patched(opt)
    assert calls == [opt, opt] and opt.fp32_from_fp32_groups is new


def test_patch_is_idempotent(patched):
    assert patched._primus_fsdp2_fp32
    patch.patch_muse_spark_fsdp2_fp32_param_grads(SimpleNamespace())
    assert CLS._copy_model_grads_to_main_grads is patched


@pytest.mark.parametrize(
    "model_type, fsdp2, expected",
    [
        ("muse_spark", True, True),
        ("muse_spark_text", True, False),  # the HF text_config model_type is not a Primus one
        ("muse_spark", False, False),
        ("muse_spark", None, False),  # flag absent
        ("muse_spark", "false", False),  # env-substituted strings: bool("false") would be True
        ("muse_spark", "0", False),
        ("muse_spark", "true", True),
        ("gpt", True, False),
    ],
)
@pytest.mark.parametrize(
    "name", ["megatron.muse_spark.fsdp2_fp32_param_grads", "megatron.muse_spark.fsdp2_dtype_islands"]
)
def test_registered_condition_needs_muse_spark_and_fsdp2(name, model_type, fsdp2, expected):
    # both FSDP2 patches share one condition (muse_spark_fsdp2_patches._is_muse_spark_fsdp2)
    from primus.backends.megatron.patches import muse_spark_fsdp2_patches  # noqa: F401
    from primus.core.patches import PatchContext
    from primus.core.patches.patch_registry import PatchRegistry

    registered = PatchRegistry.get(name)
    assert registered is not None and registered.phase == "before_train"
    params = SimpleNamespace(model_type=model_type)
    if fsdp2 is not None:
        params.use_torch_fsdp2 = fsdp2
    ctx = PatchContext(
        backend="megatron", phase="before_train", extra={"module_config": SimpleNamespace(params=params)}
    )
    assert bool(registered.condition(ctx)) is expected
