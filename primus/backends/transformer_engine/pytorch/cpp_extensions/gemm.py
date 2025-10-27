###############################################################################
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


from typing import Callable, Iterable, Optional, Tuple, Union

import torch
import transformer_engine_torch as tex
from megatron.core.utils import is_te_min_version
from transformer_engine.pytorch.cpp_extensions.gemm import _empty_tensor
from transformer_engine.pytorch.utils import assert_dim_for_fp8_exec

import primus.backends.transformer_engine.transformer_engine_torch as ptex

if is_te_min_version("2.4"):
    from ...debug.pytorch.debug_quantization import DebugQuantizer
    from ..tensor._internal.float8_blockwise_tensor_base import (
        Float8BlockwiseQTensorBase,
    )

if is_te_min_version("2.0"):

    # TE version >= 2.0 and <= 2.3
    if not is_te_min_version("2.3", check_equality=False):
        from transformer_engine.pytorch.cpp_extensions.gemm import (
            reset_swizzled_inputs,
            swizzle_inputs,
        )
    from transformer_engine.pytorch.tensor.quantized_tensor import Quantizer

    def general_gemm(
        A: torch.Tensor,
        B: torch.Tensor,
        workspace: torch.Tensor,
        out_dtype: Optional[torch.dtype] = None,
        quantization_params: Optional[Quantizer] = None,
        gelu: bool = False,
        gelu_in: torch.Tensor = None,
        accumulate: bool = False,
        layout: str = "TN",
        out: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        use_split_accumulator: bool = False,
        grad: bool = False,
        ub: Union[ptex.CommOverlap, ptex.CommOverlapP2P] = None,
        ub_type: ptex.CommOverlapType = None,
        extra_output: Optional[torch.Tensor] = None,
        bulk_overlap: bool = False,
        orig_func: Callable = None,
    ) -> Iterable[Optional[torch.Tensor]]:
        """GEMM supporting fp8 inputs."""
        if ub is None or ub_type is None and orig_func is not None:
            return orig_func(
                A,
                B,
                workspace,
                out_dtype,
                quantization_params,
                gelu,
                gelu_in,
                accumulate,
                layout,
                out,
                bias,
                use_split_accumulator,
                grad,
                ub,
                ub_type,
                extra_output,
                bulk_overlap,
            )

        assert layout in ("TN", "NN", "NT"), f"GEMM layout {layout} not supported."
        transa = layout[0] == "T"
        transb = layout[1] == "T"
        # assert quantization_params is None, "FP8 output not supported yet"

        if ub_type is not None:
            assert ub is not None, (
                f"{'AG+GEMM' if ub_type == ptex.CommOverlapType.AG else 'GEMM+RS'} overlap requires"
                + "a valid `ub` communicator object."
            )

        if ub is not None:
            assert ub_type is not None, "Comm+GEMM overlap requires a valid `comm_type` argument."
            if ub_type == ptex.CommOverlapType.RS:
                if not (bulk_overlap and not ub.is_fp8_ubuf()):
                    assert extra_output is not None, "GEMM+RS overlap requires extra output tensor."

        if out is not None:
            if not out.is_contiguous():
                raise ValueError("Output tensor is not contiguous.")

        if is_te_min_version("2.3", check_equality=False):
            debug_quantizer = None
            if isinstance(quantization_params, DebugQuantizer):
                debug_quantizer = quantization_params
                quantization_params = quantization_params.parent_quantizer
                A = A.get_tensor(not transa)
                B = B.get_tensor(transb)
            if isinstance(A, Float8BlockwiseQTensorBase) or isinstance(B, Float8BlockwiseQTensorBase):
                # There is not use_split_accumulator == False
                # implementation for Float8BlockwiseQTensorBase GEMM
                use_split_accumulator = True

        # Use bfloat16 as default bias_dtype
        bias_dtype = torch.bfloat16 if bias is None else bias.dtype

        args = (
            A,
            transa,  # transa
            B,
            transb,  # transb
            out,
            quantization_params,
            out_dtype,
            bias,
            bias_dtype,
            gelu,
            gelu_in,
            grad,  # grad
            workspace,
            workspace.shape[0],
            accumulate,
            use_split_accumulator,
        )
        kwargs = {
            "comm_overlap": ub,
            "comm_type": ub_type,
            "extra_output": extra_output,
            "bulk_overlap": bulk_overlap,
        }

        # TE version >= 2.0 and <= 2.3
        if not is_te_min_version("2.3", check_equality=False):
            original_scale_inverses = swizzle_inputs(A, B, layout)

        out, bias_grad, gelu_input, extra_output = ptex.generic_gemm(*args, **kwargs)

        # TE version >= 2.0 and <= 2.3
        if not is_te_min_version("2.3", check_equality=False):
            reset_swizzled_inputs(A, B, original_scale_inverses)
        elif debug_quantizer is not None:
            # TE version >= 2.4
            out = debug_quantizer.process_gemm_output(out)

        return out, bias_grad, gelu_input, extra_output

else:

    def fp8_gemm(
        A: torch.Tensor,
        A_scale_inv: torch.Tensor,
        A_fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
        A_dtype: tex.DType,
        B: torch.Tensor,
        B_scale_inv: torch.Tensor,
        B_fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
        B_dtype: tex.DType,
        out_dtype: torch.dtype,
        workspace: torch.Tensor,
        gelu: bool = False,
        accumulate: bool = False,
        out: Optional[torch.Tensor] = None,
        out_index=None,
        fp8_meta_tensor: tex.FP8TensorMeta = None,
        bias: Optional[torch.Tensor] = None,
        use_bias: bool = False,
        use_split_accumulator: bool = False,
        D_dtype: Optional[tex.DType] = None,
        ub_algo: ptex.CommOverlapAlgo = None,
        ub: Union[ptex.CommOverlap, ptex.CommOverlapP2P] = None,
        extra_output_tensor: torch.Tensor = None,
        orig_func: Callable = None,
    ) -> torch.Tensor:
        """TN layout GEMM with fp8 inputs."""
        if ub_algo is None or ub is None and orig_func is not None:
            return orig_func(
                A,
                A_scale_inv,
                A_fp8_tensor,
                A_dtype,
                B,
                B_scale_inv,
                B_fp8_tensor,
                B_dtype,
                out_dtype,
                workspace,
                gelu,
                accumulate,
                out,
                out_index,
                fp8_meta_tensor,
                bias,
                use_bias,
                use_split_accumulator,
                D_dtype,
                ub_algo,
                ub,
                extra_output_tensor,
            )

        if not use_bias:
            bias = None

        empty_tensor = _empty_tensor()
        if D_dtype is not None and D_dtype in [
            tex.DType.kFloat8E4M3,
            tex.DType.kFloat8E5M2,
        ]:
            assert fp8_meta_tensor is not None and out_index is not None
        assert_dim_for_fp8_exec(A)
        assert_dim_for_fp8_exec(B)
        assert A.dtype == torch.uint8
        assert B.dtype == torch.uint8

        if A_scale_inv.numel() > 0:
            A_scale_inv = A_scale_inv[A_fp8_tensor]
        if B_scale_inv.numel() > 0:
            B_scale_inv = B_scale_inv[B_fp8_tensor]

        if out is None:
            out = torch.empty(
                B.shape[0],
                A.shape[0],
                dtype=out_dtype,
                device="cuda",
            )
        else:
            if not out.is_contiguous():
                raise ValueError("Output tensor is not contiguous.")

        # Use bfloat16 as default bias_dtype
        bias_dtype = torch.bfloat16 if bias is None else bias.dtype
        if gelu:
            gelu_input = torch.empty_like(out, dtype=bias_dtype)
        else:
            gelu_input = empty_tensor

        if gelu or accumulate:
            raise NotImplementedError(f"not impl for async tp, gelu: {gelu}, accumulate: {accumulate}")

        out_dtype = out.dtype if D_dtype is None else D_dtype

        D_scale = None if out_index is None else fp8_meta_tensor.scale[out_index]

        A = ptex.comm_overlap.view_as_torch_dtype(A, A_dtype)
        B = ptex.comm_overlap.view_as_torch_dtype(B, B_dtype)

        is_out_uint8 = out.dtype == torch.uint8
        out_dtype = ptex.comm_overlap.te_to_torch_dtype(out_dtype)
        out = out.view(out_dtype)

        args = (B, A.T)
        kwargs = {
            "scale_a": B_scale_inv,
            "scale_b": A_scale_inv,
            "scale_result": D_scale,
            "out_dtype": out_dtype,
            "bias": bias,
            "use_fast_accum": not use_split_accumulator,
        }

        fn = torch._scaled_mm

        if ub_algo is not None:
            assert ub is not None, "ub object is None!"
            args = args + ("NN", out)
            if ub_algo == ptex.CommOverlapAlgo.BULK_OVERLAP_AG:
                fn = ub.bulk_overlap
                args = tuple(args + (ptex.CommOverlapType.AG, kwargs))
            elif ub_algo == ptex.CommOverlapAlgo.BULK_OVERLAP_RS:
                fn = ub.bulk_overlap
                args = tuple(args + (ptex.CommOverlapType.RS, kwargs))
            elif ub_algo == ptex.CommOverlapAlgo.SPLIT_PIPELINED_AG_P2P:
                fn = ub.split_overlap_ag
                args = tuple(args + (extra_output_tensor, kwargs))
            elif ub_algo == ptex.CommOverlapAlgo.SPLIT_PIPELINED_RS:
                fn = ub.split_overlap_rs
            elif ub_algo == ptex.CommOverlapAlgo.SPLIT_PIPELINED_RS_P2P:
                fn = ub.split_overlap_rs

            else:
                raise NotImplementedError(f"not impl ub_algo: {ub_algo}!")
            fn(*args)
        else:
            fn(*args, out=out, **kwargs)
        if is_out_uint8:
            out = out.view(torch.uint8)
        return out, gelu_input

    def gemm(
        A: torch.Tensor,
        B: torch.Tensor,
        dtype: torch.dtype,
        workspace: torch.Tensor,
        gelu: bool = False,
        gelu_input: Optional[torch.Tensor] = None,
        grad: bool = False,
        accumulate: bool = False,
        layout: str = "TN",
        out: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        use_bias: bool = False,
        ub_algo: ptex.CommOverlapAlgo = None,
        ub: Union[ptex.CommOverlap, ptex.CommOverlapP2P] = None,
        extra_output_tensor: torch.Tensor = None,
        orig_func: Callable = None,
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        """Non FP8 GEMM."""
        if ub_algo is None or ub is None and orig_func is not None:
            return orig_func(
                A,
                B,
                dtype,
                workspace,
                gelu,
                gelu_input,
                grad,
                accumulate,
                layout,
                out,
                bias,
                use_bias,
                ub_algo,
                ub,
                extra_output_tensor,
            )

        assert layout in ("TN", "NN", "NT"), f"GEMM layout {layout} not supported."
        transa = layout[0] == "T"
        transb = layout[1] == "T"
        empty_tensor = _empty_tensor()

        if out is None:
            out = torch.empty(
                B.shape[1] if transb else B.shape[0],
                A.shape[0] if transa else A.shape[1],
                dtype=dtype,
                device="cuda",
            )
        else:
            if not out.is_contiguous():
                raise ValueError("Output tensor is not contiguous.")

        if gelu and not grad:
            gelu_input = torch.empty_like(out, dtype=dtype)
        elif not gelu:
            gelu_input = empty_tensor

        if grad and use_bias:
            grad_bias = torch.empty(B.shape[1], dtype=out.dtype, device="cuda")
        else:
            grad_bias = empty_tensor

        bias = bias if use_bias else empty_tensor

        if gelu or accumulate or use_bias:
            raise NotImplementedError

        assert (
            A.dtype == dtype and B.dtype == dtype
        ), f"Expected dtype={dtype}, but found A.dtype={A.dtype} and B.dtype={B.dtype}"

        B = B.T if layout[1] == "T" else B
        if (ub_algo is not None) and (ub_algo == ptex.CommOverlapAlgo.SPLIT_PIPELINED_RS):
            layout = "N" + layout[0]
        else:
            A = A.T if layout[0] == "T" else A
            layout = "NN"

        args = (B, A)
        if ub_algo is not None:
            assert ub is not None, "ub object is None!"

            args = args + (layout, out)
            if ub_algo == ptex.CommOverlapAlgo.BULK_OVERLAP_AG:
                fn = ub.bulk_overlap
                args = tuple(args + (ptex.CommOverlapType.AG,))
            elif ub_algo == ptex.CommOverlapAlgo.BULK_OVERLAP_RS:
                fn = ub.bulk_overlap
                args = tuple(args + (ptex.CommOverlapType.RS,))
            elif ub_algo == ptex.CommOverlapAlgo.SPLIT_PIPELINED_AG_P2P:
                fn = ub.split_overlap_ag
                args = tuple(args + (extra_output_tensor,))
            elif ub_algo == ptex.CommOverlapAlgo.SPLIT_PIPELINED_RS:
                fn = ub.split_overlap_rs
                assert extra_output_tensor is not None, "SPLIT_PIPELINED_RS requires extra output tensor"
                args = tuple(args + (extra_output_tensor,))
            elif ub_algo == ptex.CommOverlapAlgo.SPLIT_PIPELINED_RS_P2P:
                fn = ub.split_overlap_rs
                assert extra_output_tensor is not None, "SPLIT_PIPELINED_RS_P2P requires extra output tensor"
                args = tuple(args + (extra_output_tensor,))
            fn(*args)

        else:
            fn = torch.mm
            if use_bias:
                fn = torch.addmm
                args = args + (bias,)

            fn(*args, out=out)
        return out, grad_bias, gelu_input
