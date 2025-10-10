import functools

import torch.distributed._symmetric_memory as symm_module
import transformer_engine as te
import transformer_engine_torch as tex
from megatron.core.utils import is_te_min_version

from primus.backends.transformer_engine import transformer_engine_torch as ptex
from primus.backends.transformer_engine.pytorch.module.base import (
    get_workspace,
    initialize_ub,
)
from primus.modules.trainer.torchtitan.pre_trainer import TorchTitanPretrainTrainer

original_fused_all_gather_matmul_impl = None
original_fused_matmul_reduce_scatter_impl = None
original_fused_scaled_matmul_reduce_scatter_impl = None


def patch_async_tp():
    global original_fused_all_gather_matmul_impl
    global original_fused_matmul_reduce_scatter_impl
    global original_fused_scaled_matmul_reduce_scatter_impl

    original_fused_all_gather_matmul_impl = symm_module._fused_all_gather_matmul_impl
    original_fused_matmul_reduce_scatter_impl = symm_module._fused_matmul_reduce_scatter_impl
    original_fused_scaled_matmul_reduce_scatter_impl = symm_module._fused_scaled_matmul_reduce_scatter_impl

    TorchTitanPretrainTrainer.patch_torch_async_tp(True)


def restore_async_tp():
    global original_fused_all_gather_matmul_impl
    global original_fused_matmul_reduce_scatter_impl
    global original_fused_scaled_matmul_reduce_scatter_impl

    if original_fused_all_gather_matmul_impl is not None:
        symm_module._fused_all_gather_matmul_impl = original_fused_all_gather_matmul_impl
    if original_fused_matmul_reduce_scatter_impl is not None:
        symm_module._fused_matmul_reduce_scatter_impl = original_fused_matmul_reduce_scatter_impl
    if original_fused_scaled_matmul_reduce_scatter_impl is not None:
        symm_module._fused_scaled_matmul_reduce_scatter_impl = (
            original_fused_scaled_matmul_reduce_scatter_impl
        )

    print(f"Restored original Async TP functions\n")


prev_CommOverlap = None
prev_CommOverlapP2P = None
prev_general_gemm = None
prev_CommOverlapAlgo = None
prev_gemm = None
prev_fp8_gemm = None
prev_initialize_ub = None
prev_get_workspace = None


def te_patch():
    global prev_CommOverlap
    global prev_CommOverlapP2P
    global prev_general_gemm
    global prev_CommOverlapAlgo
    global prev_gemm
    global prev_fp8_gemm
    global prev_initialize_ub
    global prev_get_workspace

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

    tex.CommOverlap = ptex.CommOverlap
    tex.CommOverlapP2P = ptex.CommOverlapP2P
    tex.CommOverlapType = ptex.CommOverlapType
    if is_te_min_version("2.0"):
        from primus.backends.transformer_engine.pytorch.cpp_extensions.gemm import (
            general_gemm,
        )

        te.pytorch.cpp_extensions.general_gemm = functools.partial(general_gemm, orig_func=prev_general_gemm)
        te.pytorch.module.linear.general_gemm = functools.partial(general_gemm, orig_func=prev_general_gemm)
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


def te_restore():
    global prev_CommOverlap
    global prev_CommOverlapP2P
    global prev_general_gemm
    global prev_CommOverlapAlgo
    global prev_gemm
    global prev_fp8_gemm
    global prev_initialize_ub
    global prev_get_workspace

    if prev_CommOverlap is not None:
        tex.CommOverlap = prev_CommOverlap
    if prev_CommOverlapP2P is not None:
        tex.CommOverlapP2P = prev_CommOverlapP2P
    if is_te_min_version("2.0"):
        if prev_general_gemm is not None:
            te.pytorch.cpp_extensions.general_gemm = prev_general_gemm
            te.pytorch.module.linear.general_gemm = prev_general_gemm
            te.pytorch.module.layernorm_linear.general_gemm = prev_general_gemm
    else:
        if prev_gemm is not None:
            tex.CommOverlapAlgo = prev_CommOverlapAlgo
            te.pytorch.cpp_extensions.CommOverlapAlgo = prev_CommOverlapAlgo
            te.pytorch.cpp_extensions.gemm = prev_gemm
            te.pytorch.module.linear.gemm = prev_gemm
            te.pytorch.cpp_extensions.fp8_gemm = prev_fp8_gemm
            te.pytorch.module.linear.fp8_gemm = prev_fp8_gemm

    if prev_initialize_ub is not None:
        te.pytorch.module.base.initialize_ub = prev_initialize_ub
        te.pytorch.module.base.get_workspace = prev_get_workspace
