###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
import functools
from contextlib import contextmanager
from copy import deepcopy

import torch
import torch.distributed as dist
import transformer_engine as te
import transformer_engine_torch as tex
from megatron.core.utils import is_te_min_version
from torch._inductor.utils import run_and_get_triton_code
from torch.distributed._symmetric_memory import enable_symm_mem_for_group
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard
from torch.distributed.tensor.parallel import parallelize_module
from torch.distributed.tensor.parallel.style import ColwiseParallel, RowwiseParallel
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from transformer_engine.pytorch import LayerNormLinear, Linear, fp8_autocast

from primus.backends.transformer_engine import transformer_engine_torch as ptex
from primus.backends.transformer_engine.pytorch.module.base import (
    get_workspace,
    initialize_ub,
)
from primus.core.utils import logger
from primus.modules.module_utils import set_logging_rank
from primus.modules.trainer.torchtitan.pre_trainer import TorchTitanPretrainTrainer


@contextmanager
def custom_te_patch():

    prev_CommOverlap = tex.CommOverlap
    prev_CommOverlapP2P = tex.CommOverlapP2P
    if is_te_min_version("2.0"):
        prev_general_gemm = te.pytorch.cpp_extensions.general_gemm
    else:
        prev_CommOverlapAlgo = tex.CommOverlapAlgo
        prev_gemm = te.pytorch.cpp_extensions.gemm
        prev_fp8_gemm = te.pytorch.cpp_extensions.fp8_gemm
    prev_initialize_ub = te.pytorch.module.base.initialize_ub
    prev_get_workspace = te.pytorch.module.base.get_workspace
    try:
        tex.CommOverlap = ptex.CommOverlap
        tex.CommOverlapP2P = ptex.CommOverlapP2P
        tex.CommOverlapType = ptex.CommOverlapType
        if is_te_min_version("2.0"):
            from primus.backends.transformer_engine.pytorch.cpp_extensions.gemm import (
                general_gemm,
            )

            te.pytorch.cpp_extensions.general_gemm = functools.partial(
                general_gemm, orig_func=prev_general_gemm
            )
            te.pytorch.module.linear.general_gemm = functools.partial(
                general_gemm, orig_func=prev_general_gemm
            )
            te.pytorch.module.layernorm_linear.general_gemm = functools.partial(
                general_gemm, orig_func=prev_general_gemm
            )
        else:
            from primus.backends.transformer_engine.pytorch.cpp_extensions.gemm import (
                fp8_gemm,
                gemm,
            )

            tex.CommOverlapAlgo = ptex.CommOverlapAlgo
            te.pytorch.cpp_extensions.CommOverlapAlgo = ptex.CommOverlapAlgo
            te.pytorch.cpp_extensions.gemm = functools.partial(gemm, orig_func=prev_gemm)
            te.pytorch.module.linear.gemm = functools.partial(gemm, orig_func=prev_gemm)
            te.pytorch.cpp_extensions.fp8_gemm = functools.partial(fp8_gemm, orig_func=prev_fp8_gemm)
            te.pytorch.module.linear.fp8_gemm = functools.partial(fp8_gemm, orig_func=prev_fp8_gemm)
        te.pytorch.module.base.initialize_ub = initialize_ub
        te.pytorch.module.base.get_workspace = get_workspace

        te.pytorch.cpp_extensions.CommOverlapType = ptex.CommOverlapType

        yield
    finally:
        tex.CommOverlap = prev_CommOverlap
        tex.CommOverlapP2P = prev_CommOverlapP2P
        if is_te_min_version("2.0"):
            te.pytorch.cpp_extensions.general_gemm = prev_general_gemm
            te.pytorch.module.linear.general_gemm = prev_general_gemm
            te.pytorch.module.layernorm_linear.general_gemm = prev_general_gemm
        else:
            tex.CommOverlapAlgo = prev_CommOverlapAlgo
            te.pytorch.cpp_extensions.CommOverlapAlgo = prev_CommOverlapAlgo
            te.pytorch.cpp_extensions.gemm = prev_gemm
            te.pytorch.module.linear.gemm = prev_gemm
            te.pytorch.cpp_extensions.fp8_gemm = prev_fp8_gemm
            te.pytorch.module.linear.fp8_gemm = prev_fp8_gemm
        te.pytorch.module.base.initialize_ub = prev_initialize_ub
        te.pytorch.module.base.get_workspace = prev_get_workspace


def te_linear(
    seed,
    batch_size,
    seqlen,
    in_features,
    out_features,
    ub_overlap_ag,
    ub_overlap_rs,
    ub_overlap_rs_dgrad=False,
    ub_bulk_wgrad=False,
    ub_bulk_dgrad=False,
    enable_fp8=False,
    **cfg
):
    torch.manual_seed(seed)
    tp_size = cfg["tp_size"]
    dtype = cfg.get("params_dtype", torch.bfloat16)
    parallel_mode = cfg.get("parallel_mode", "row")

    if ub_overlap_ag or ub_overlap_rs:
        te.pytorch.module.base._ub_communicators = None
        input_shape = [seqlen * batch_size, in_features]
        te.pytorch.module.base.initialize_ub(
            shape=input_shape,
            tp_size=tp_size,
            use_fp8=enable_fp8,
            dtype=torch.uint8 if enable_fp8 else dtype,
        )

    if parallel_mode == "column":
        inp_shape = (batch_size * seqlen // tp_size, in_features)
        grad_out_shape = (batch_size * seqlen, out_features // tp_size)
        model = LayerNormLinear(
            in_features,
            out_features,
            ub_overlap_ag=ub_overlap_ag,
            ub_overlap_rs_dgrad=ub_overlap_rs_dgrad,
            ub_bulk_wgrad=ub_bulk_wgrad,
            ub_bulk_dgrad=ub_bulk_dgrad,
            **cfg
        )
    else:
        inp_shape = (batch_size * seqlen, in_features // tp_size)
        grad_out_shape = (batch_size * seqlen // tp_size, in_features)
        model = Linear(
            in_features, out_features, ub_overlap_ag=ub_overlap_ag, ub_overlap_rs=ub_overlap_rs, **cfg
        )

    inp = torch.rand(inp_shape, dtype=dtype, device="cuda", requires_grad=True)
    grad_output = torch.rand(grad_out_shape, dtype=dtype, device="cuda")

    out = model(inp)
    out.backward(grad_output)

    return (out, inp.grad, model.weight.grad)


class ToyModel(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(ToyModel, self).__init__()
        self.w1 = torch.nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        return self.w1(x)


def torch_linear(seed, batch_size, seqlen, in_features, out_features, enable_fp8=False, **cfg):
    torch.manual_seed(seed)
    tp_size = cfg["tp_size"]
    dtype = cfg.get("dtype", torch.bfloat16)
    parallel_mode = cfg.get("parallel_mode", "row")
    patch = cfg.get("patch", False)
    tp_mesh = init_device_mesh("cuda", (tp_size,))
    model = ToyModel(in_features, out_features).to("cuda")

    if enable_fp8:

        from torchao.float8 import Float8LinearConfig, convert_to_float8_training
        from torchao.float8.float8_tensor_parallel import (
            Float8ColwiseParallel,
            Float8RowwiseParallel,
        )

        rowwise_parallel, colwise_parallel = (
            Float8RowwiseParallel,
            Float8ColwiseParallel,
        )

        model = convert_to_float8_training(
            model,
            config=Float8LinearConfig(emulate=False),
        )
    else:
        rowwise_parallel, colwise_parallel = (
            RowwiseParallel,
            ColwiseParallel,
        )
        model = model.to(dtype)

    if parallel_mode == "column":
        inp_shape = (batch_size * seqlen // tp_size, in_features)
        grad_out_shape = (batch_size * seqlen, out_features // tp_size)
        sharded_parallel = {"w1": colwise_parallel(input_layouts=Shard(0))}
    else:
        inp_shape = (batch_size * seqlen, in_features // tp_size)
        grad_out_shape = (batch_size * seqlen // tp_size, in_features)
        sharded_parallel = {"w1": rowwise_parallel(output_layouts=Shard(0))}
    parallelize_mod = parallelize_module(deepcopy(model), tp_mesh, sharded_parallel)

    # torch._dynamo.reset()
    parallelize_mod = torch.compile(parallelize_mod)

    inp = torch.rand(inp_shape, dtype=dtype, device="cuda", requires_grad=True)
    grad_output = torch.rand(grad_out_shape, dtype=dtype, device="cuda")
    code = run_and_get_triton_code(parallelize_mod, inp)
    if patch and parallel_mode == "column":
        assert (
            "fused_all_gather_scaled_matmul" in code or "fused_all_gather_matmul" in code
        ), "Async TP not applied in columnwise linear"
    elif patch and parallel_mode == "row":
        assert (
            "fused_scaled_matmul_reduce_scatter" in code or "fused_matmul_reduce_scatter" in code
        ), "Async TP not applied in rowwise linear"

    out = parallelize_mod(inp)
    out.backward(grad_output)

    return (out, inp.grad, parallelize_mod.w1.weight.grad)


@instantiate_parametrized_tests
class TPOverlapTestCase(MultiProcessTestCase):
    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    @property
    def world_size(self) -> int:
        return torch.cuda.device_count()

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    def _init_process(self):
        torch.cuda.set_device(self.device)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )

        logger_cfg = logger.LoggerConfig(
            exp_root_path="/tmp",
            work_group="",
            user_name="",
            exp_name="test",
            module_name="linear",
            rank=self.rank,
            world_size=self.world_size,
        )

        logger.setup_logger(logger_cfg)
        set_logging_rank(rank=self.rank, world_size=self.world_size)

    @skip_if_lt_x_gpu(2)
    @parametrize("out_features", [4096])
    @parametrize("in_features", [4096])
    @parametrize("seqlen", [8192])
    @parametrize("batch_size", [1])
    @parametrize("ub_name", ["qkv", "proj"])
    @parametrize("ub_overlap_ag", [True])
    @parametrize("ub_overlap_rs", [True])
    @parametrize("ub_overlap_rs_dgrad", [False])
    @parametrize("ub_bulk_wgrad", [True])
    @parametrize("ub_bulk_dgrad", [True])
    def test_te_linear(
        self,
        batch_size,
        seqlen,
        in_features,
        out_features,
        ub_name,
        ub_overlap_ag,
        ub_overlap_rs,
        ub_overlap_rs_dgrad,
        ub_bulk_wgrad,
        ub_bulk_dgrad,
    ) -> None:
        self._init_process()
        group = dist.group.WORLD
        rank = self.rank
        seed = 42 + rank

        cfg = {
            "tp_group": group,
            "tp_size": self.world_size,
            "parallel_mode": "column" if ub_name in ["qkv", "fc1"] else "row",
            "sequence_parallel": True,
            "bias": False,
            "ub_name": ub_name,
            "params_dtype": torch.bfloat16,
        }

        base_outputs = te_linear(
            seed,
            batch_size,
            seqlen,
            in_features,
            out_features,
            ub_overlap_ag=False,
            ub_overlap_rs=False,
            **cfg
        )

        with custom_te_patch():
            patch_outputs = te_linear(
                seed,
                batch_size,
                seqlen,
                in_features,
                out_features,
                ub_overlap_ag=ub_overlap_ag,
                ub_overlap_rs=ub_overlap_rs,
                ub_overlap_rs_dgrad=ub_overlap_rs_dgrad,
                ub_bulk_wgrad=ub_bulk_wgrad,
                ub_bulk_dgrad=ub_bulk_dgrad,
                **cfg
            )

        for base_out, patch_out in zip(base_outputs, patch_outputs):
            torch.testing.assert_close(base_out, patch_out, atol=3e-2, rtol=1e-2)

    @skip_if_lt_x_gpu(2)
    @parametrize("out_features", [1024])
    @parametrize("in_features", [1024])
    @parametrize("seqlen", [4096])
    @parametrize("batch_size", [1])
    @parametrize("ub_name", ["qkv", "proj"])
    @parametrize("ub_overlap_ag", [True])
    @parametrize("ub_overlap_rs", [True])
    @parametrize("dtype", [torch.bfloat16])
    def test_fp8_te_linear(
        self,
        batch_size,
        seqlen,
        in_features,
        out_features,
        ub_name,
        ub_overlap_ag,
        ub_overlap_rs,
        dtype,
    ) -> None:
        self._init_process()
        group = dist.group.WORLD
        rank = self.rank
        seed = 42 + rank

        cfg = {
            "tp_group": group,
            "tp_size": self.world_size,
            "parallel_mode": "column" if ub_name in ["qkv", "fc1"] else "row",
            "sequence_parallel": True,
            "bias": False,
            "ub_name": ub_name,
            "params_dtype": dtype,
        }

        with fp8_autocast(enabled=True):
            base_outputs = te_linear(
                seed,
                batch_size,
                seqlen,
                in_features,
                out_features,
                ub_overlap_ag=False,
                ub_overlap_rs=False,
                enable_fp8=True,
                **cfg
            )

        with custom_te_patch(), fp8_autocast(enabled=True):
            patch_outputs = te_linear(
                seed,
                batch_size,
                seqlen,
                in_features,
                out_features,
                ub_overlap_ag=ub_overlap_ag,
                ub_overlap_rs=ub_overlap_rs,
                enable_fp8=True,
                **cfg
            )

        for base_out, patch_out in zip(base_outputs, patch_outputs):
            torch.testing.assert_close(base_out, patch_out, atol=1e-2, rtol=1e-2)

    @skip_if_lt_x_gpu(2)
    @parametrize("out_features", [4096])
    @parametrize("in_features", [4096])
    @parametrize("seqlen", [8192])
    @parametrize("batch_size", [1])
    @parametrize("parallel_mode", ["column", "row"])
    @parametrize("dtype", [torch.bfloat16])
    @parametrize("enable_fp8", [True, False])
    def test_torch_linear(
        self,
        batch_size,
        seqlen,
        in_features,
        out_features,
        parallel_mode,
        enable_fp8,
        dtype,
    ):
        self._init_process()
        group = dist.group.WORLD
        rank = self.rank
        seed = 42 + rank

        cfg = {
            "tp_size": self.world_size,
            "parallel_mode": parallel_mode,
            "dtype": dtype,
            "tp_group": group,
            "patch": False,
        }

        base_outputs = torch_linear(seed, batch_size, seqlen, in_features, out_features, enable_fp8, **cfg)
        cfg["patch"] = True

        TorchTitanPretrainTrainer.patch_torch_async_tp(True)
        enable_symm_mem_for_group(group.group_name)
        torch._inductor.config._micro_pipeline_tp = True
        patch_outputs = torch_linear(seed, batch_size, seqlen, in_features, out_features, enable_fp8, **cfg)

        for base_out, patch_out in zip(base_outputs, patch_outputs):
            torch.testing.assert_close(base_out, patch_out, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    run_tests()
