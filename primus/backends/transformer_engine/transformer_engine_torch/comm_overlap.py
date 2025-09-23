###############################################################################
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import operator
from functools import reduce
from typing import Any, Dict, List, Optional, Union

import primus_turbo.pytorch as pt
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
import transformer_engine_torch as tex
from hip import hip
from megatron.core.utils import is_te_min_version

from .comm_overlap_type import CommOverlapType

_backend_streams: Dict[int, List[torch.cuda.Stream]] = {}
_stream_priorities: Dict[int, tuple] = {}


def get_backend_stream(size=1, priority=0, prefix=""):
    global _backend_streams

    key = (priority, prefix)
    if key not in _backend_streams or len(_backend_streams[key]) < size:
        _backend_streams[key] = [torch.cuda.Stream(priority=priority) for _ in range(size)]

    return _backend_streams[key][:size]


def get_stream_priority_range(device_id=-1):
    global _stream_priorities

    if device_id < 0:
        device_id = hip_check(hip.hipGetDevice())

    if device_id not in _stream_priorities.keys():
        hip_check(hip.hipSetDevice(device_id))
        _stream_priorities[device_id] = hip_check(hip.hipDeviceGetStreamPriorityRange())

    return _stream_priorities[device_id]


def hip_check(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    return result


def te_to_torch_dtype(dtype: Union[tex.DType, torch.dtype]) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype

    if dtype == tex.DType.kByte:
        return torch.uint8
    elif dtype == tex.DType.kInt32:
        return torch.int32
    elif dtype == tex.DType.kFloat32:
        return torch.float32
    elif dtype == tex.DType.kFloat16:
        return torch.float16
    elif dtype == tex.DType.kBFloat16:
        return torch.bfloat16
    elif dtype == tex.DType.kFloat8E4M3:
        return pt.float8_e4m3
    elif dtype == tex.DType.kFloat8E5M2:
        return pt.float8_e5m2
    raise ValueError(f"not support dtype: {dtype}")


def view_as_torch_dtype(tensor: torch.Tensor, dtype: tex.DType):
    torch_dtype = te_to_torch_dtype(dtype)
    if tensor.dtype != torch_dtype:
        return tensor.view(torch_dtype)
    return tensor


if is_te_min_version("2.0"):
    import warnings

    from transformer_engine.pytorch.tensor._internal.float8_tensor_base import (
        Float8TensorBase,
    )
    from transformer_engine.pytorch.tensor._internal.mxfp8_tensor_base import (
        MXFP8TensorBase,
    )
    from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
    from transformer_engine.pytorch.tensor.quantized_tensor import (
        QuantizedTensor,
        Quantizer,
    )

    class CommOverlapBase:
        def __init__(self, buffer_shape: List[int], buffer_dtype: torch.dtype, group_name: str, tp_size: int):

            group = c10d._resolve_process_group(group_name)
            assert tp_size == group.size(), f"tp_size {tp_size} is difference with group size: {group.size()}"

            alloc_size = reduce(operator.mul, buffer_shape, 1) * buffer_dtype.itemsize
            self.buf = torch.empty((alloc_size,), dtype=torch.uint8, device="cuda")
            self.buf_size = self.buf.nbytes
            self.tp_size = tp_size
            self.group = group
            self.rank = group.rank()
            self.buf_dtype = buffer_dtype
            self.buf_shape = buffer_shape
            self.group_name = group_name
            self.scale_inv_initialized = False

        def is_atomic_gemm(self) -> bool: ...

        def is_p2p_overlap(self) -> bool: ...

        def is_fp8_ubuf(self) -> bool:
            return self.buf_dtype.itemsize == 1

        def copy_into_buffer(self, input: torch.Tensor, quantizer: Quantizer, local_chunk: bool = False):
            """copy input to local buffer

            Args:
                input (torch.Tensor): ...
                quantizer (Quantizer): input_quantizer

                if comm_type is CommOverlapType.AG, copy input to tp_size chunk of local buffer;
                if comm_type is CommOverlapType.RS, copy input to local_buffer
            """
            src_data = self._quantize_input(input, quantizer)
            dst_data = self._get_buffer_without_quantizer(local_chunk=local_chunk)

            if src_data.numel() != dst_data.numel() or src_data.element_size() != dst_data.element_size():
                raise ValueError(f"input and ubuf size do not match!")

            self._copy_inp_to_buffer(src_data, dst_data)

        def _quantize_input(self, input, quantizer):
            if quantizer is not None:
                if (
                    not isinstance(input, QuantizedTensor)
                    and not isinstance(input, Float8TensorBase)
                    and not isinstance(input, MXFP8TensorBase)
                    and not (isinstance(quantizer, MXFP8Quantizer) and not quantizer.is_quantizable(input))
                ):
                    input = quantizer(input)
                elif isinstance(input, MXFP8TensorBase) and (
                    input._rowwise_data is None
                    and quantizer.rowwise_usage
                    or input._columnwise_data is None
                    and quantizer.columnwise_usage
                ):
                    warnings.warn(
                        "Input and quantizer do not have matching usages. "
                        "Dequantizing and requantizing to MXFP8."
                    )
                    input = quantizer(input.dequantize())

            if isinstance(input, Float8TensorBase):
                data = input._data
            elif isinstance(input, MXFP8TensorBase) and quantizer.rowwise_usage:
                data = input._rowwise_data
            elif isinstance(input, MXFP8TensorBase) and quantizer.columnwise_usage:
                data = input._columnwise_data
            else:
                data = input
            return data

        def _copy_inp_to_buffer(self, src_data, dst_data):
            hip_check(
                hip.hipMemcpyAsync(
                    dst_data.data_ptr(),
                    src_data.data_ptr(),
                    src_data.nbytes,
                    hip.hipMemcpyKind.hipMemcpyDeviceToDevice,
                    torch.cuda.current_stream().cuda_stream,
                )
            )

        def _get_buffer_without_quantizer(self, local_chunk: bool = False, shape=None):
            out_shape = shape or self.buf_shape

            if shape is None and local_chunk:
                out_shape = [out_shape[0] // self.tp_size] + list(out_shape)[1:]

            request_size = reduce(operator.mul, out_shape, 1) * self.buf_dtype.itemsize

            if local_chunk:
                buf = self.buf.chunk(self.tp_size)[self.rank]
            else:
                buf = self.buf

            buffer = buf[0:request_size].view(self.buf_dtype).view(*out_shape)
            return buffer

        def get_buffer(self, quantizer: Quantizer = None, local_chunk: bool = False, shape=None):
            if is_te_min_version("2.1"):
                from transformer_engine.pytorch.tensor.float8_tensor import (
                    Float8CurrentScalingQuantizer,
                )

                per_tensor_quantizers = (Float8Quantizer, Float8CurrentScalingQuantizer)
            else:
                per_tensor_quantizers = Float8Quantizer

            buffer = self._get_buffer_without_quantizer(local_chunk, shape)
            if quantizer is not None and isinstance(quantizer, per_tensor_quantizers):
                return quantizer.create_tensor_from_data(data=buffer, fake_dtype=self.buf_dtype)
            return buffer

        def set_buffer_params(self, quantizer: Quantizer):
            if quantizer is not None:
                raise ValueError("Not supported for fp8")
            self.scale_inv_initialized = True

        def bulk_overlap(
            self, A: torch.Tensor, B: torch.Tensor, layout: str, D: torch.Tensor, comm_type: CommOverlapType
        ):

            with torch.profiler.record_function("torch_native_bulk_overlap"):
                output = self.get_buffer(local_chunk=comm_type == CommOverlapType.RS)
                local_buf = self.get_buffer(local_chunk=comm_type == CommOverlapType.AG)

                if comm_type == CommOverlapType.AG:
                    handle = dist.all_gather_into_tensor(output, local_buf, group=self.group, async_op=True)
                else:
                    handle = dist.reduce_scatter_tensor(output, local_buf, group=self.group, async_op=True)

                A = A.T if layout[0] == "T" else A
                B = B.T if layout[1] == "T" else B

                torch.mm(A, B, out=D)

                handle.wait()

else:

    class CommOverlapBase:
        def __init__(self, buffer_shape: List[int], buffer_dtype: torch.dtype, group_name: str, tp_size: int):

            group = c10d._resolve_process_group(group_name)
            assert tp_size == group.size(), f"tp_size {tp_size} is difference with group size: {group.size()}"

            alloc_size = reduce(operator.mul, buffer_shape, 1) * buffer_dtype.itemsize
            self.buf = torch.empty((alloc_size,), dtype=torch.uint8, device="cuda")
            self.buf_size = self.buf.nbytes
            self.tp_size = tp_size
            self.group = group
            self.rank = group.rank()
            self.buf_dtype = buffer_dtype
            self.buf_shape = buffer_shape
            self.group_name = group_name

            self.scale_inv = None
            self.scale_inv_initialized = False

        def is_atomic_gemm(self) -> bool: ...

        def is_p2p_overlap(self) -> bool: ...

        def is_fp8_ubuf(self) -> bool:
            return self.buf_dtype.itemsize == 1

        def set_ubuf_scale_inv(self, scale_inv):
            self.scale_inv = scale_inv
            self.scale_inv_initialized = True

        def copy_input_to_ubuf(self, input: torch.Tensor, comm_type: Union[bool, int]) -> None:
            """copy input to local buffer

            Args:
                input (torch.Tensor): ...
                comm_type (int): 0 or 1

                if comm_type is CommOverlapType.AG, copy input to tp_size chunk of local buffer;
                if comm_type is CommOverlapType.RS, copy input to local_buffer
            """
            comm_type = CommOverlapType(int(comm_type))

            if comm_type == CommOverlapType.AG:
                if (
                    input.numel() * self.tp_size != self.buf.nbytes // self.buf_dtype.itemsize
                    or input.element_size() != self.buf_dtype.itemsize
                ):
                    raise ValueError(f"input and ubuf size do not match!")
            else:
                if (
                    input.numel() != self.buf.nbytes // self.buf_dtype.itemsize
                    or input.element_size() != self.buf_dtype.itemsize
                ):
                    raise ValueError(f"input and ubuf size do not match!")

            local_chunk = comm_type == CommOverlapType.AG

            self.copy_into_buffer(input, local_chunk=local_chunk)

        def copy_into_buffer(self, input, local_chunk: bool = False):
            buf = self.get_buffer(local_chunk=local_chunk)
            hip_check(
                hip.hipMemcpyAsync(
                    buf.data_ptr(),
                    input.data_ptr(),
                    input.nbytes,
                    hip.hipMemcpyKind.hipMemcpyDeviceToDevice,
                    torch.cuda.current_stream().cuda_stream,
                )
            )

        def get_buffer(self, local_chunk: bool = False, shape=None):
            out_shape = shape or self.buf_shape

            if shape is None and local_chunk:
                out_shape = [out_shape[0] // self.tp_size] + list(out_shape)[1:]

            request_size = reduce(operator.mul, out_shape, 1) * self.buf_dtype.itemsize

            if local_chunk:
                buf = self.buf.chunk(self.tp_size)[self.rank]
            else:
                buf = self.buf

            buffer = buf[0:request_size].view(self.buf_dtype).view(*out_shape)
            return buffer

        def get_ubuf_output(self, comm_type: int) -> torch.Tensor:
            """return local buffer as output.
            Args:
                comm_type (int): CommOverlapType.AG or CommOverlapType.RS

            Returns:
                torch.Tensor: if comm_type is CommOverlapType.AG, return the total buffer as output;
                            if comm_type is CommOverlapType.RS, return the tp_size chunk of local buffer as output;
            """
            buffer = self.get_buffer(local_chunk=comm_type == 0)
            return buffer

        def bulk_overlap(
            self, A: torch.Tensor, B: torch.Tensor, layout: str, D: torch.Tensor, comm_type: CommOverlapType
        ):

            with torch.profiler.record_function("torch_native_bulk_overlap"):
                output = self.get_ubuf_output(comm_type.value)
                local_buf = self.get_buffer(local_chunk=comm_type == CommOverlapType.AG)

                if comm_type == CommOverlapType.AG:
                    handle = dist.all_gather_into_tensor(output, local_buf, group=self.group, async_op=True)
                else:
                    handle = dist.reduce_scatter_tensor(output, local_buf, group=self.group, async_op=True)

                A = A.T if layout[0] == "T" else A
                B = B.T if layout[1] == "T" else B

                torch.mm(A, B, out=D)

                handle.wait()


class CommOverlap(CommOverlapBase):
    def __init__(
        self,
        buffer_shape: List[int],
        buffer_dtype: torch.dtype,
        group_name: str,
        tp_size: int,
        num_splits: int = 2,
        num_max_streams: int = 3,
        comm_cga_size: int = 2,
        num_comm_sm: int = 16,
        set_sm_margin: bool = True,
        atomic_gemm: bool = False,
    ):

        super().__init__(buffer_shape, buffer_dtype, group_name, tp_size)

        self.num_splits = num_splits
        self.atomic_gemm = atomic_gemm

    def is_atomic_gemm(self) -> bool:
        return self.atomic_gemm

    def is_p2p_overlap(self) -> bool:
        return False

    def split_overlap_rs(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        layout: str,
        D: torch.Tensor,
        rs_out: torch.Tensor,
        comm_method: str = "pipeline",
        A_scale: torch.Tensor = None,
        scaled_mm_kwargs: Optional[Dict] = None,
    ):
        if comm_method == "pipeline":
            gemm_streams = [torch.cuda.current_stream()]
            comm_streams = get_backend_stream(size=self.tp_size, priority=0, prefix="comm")
        elif comm_method == "tile":
            gemm_streams = []
            comm_streams = []
        else:
            raise ValueError(f"Only pipeline and tile supported, but {comm_method} provided")

        if scaled_mm_kwargs is None:
            pt.ops.fused_matmul_reduce_scatter(
                A,
                B,
                layout,
                reduce_op="sum",
                scatter_dim=0,
                group_name=self.group_name,
                gemm_streams=gemm_streams,
                comm_streams=comm_streams,
                comm_method=comm_method,
                num_splits=self.num_splits,
                enable_sdma=True,
                output=D,
                rs_out=rs_out,
            )
        else:
            pt.ops.fused_scaled_matmul_reduce_scatter(
                A,
                B,
                layout="NN",
                A_scale=A_scale,
                kwargs=scaled_mm_kwargs,
                reduce_op="sum",
                orig_scatter_dim=0,
                scatter_dim_after_maybe_reshape=0,
                group_name=self.group_name,
                output_shape=list(D.shape),
                out_dtype=scaled_mm_kwargs["out_dtype"],
                gemm_streams=gemm_streams,
                comm_streams=comm_streams,
                comm_method=comm_method,
                num_splits=self.num_splits,
                enable_sdma=True,
                output=D,
                rs_out=rs_out,
            )

    def split_overlap_ag(
        self,
        A_out: torch.Tensor,
        B: torch.Tensor,
        layout: str,
        D: torch.Tensor,
        A_copy: Optional[torch.Tensor] = None,
        A_scale: torch.Tensor = None,
        scaled_mm_kwargs_list: list[dict[str, Any]] = None,
    ):
        local_A = self.get_buffer(local_chunk=True)
        gemm_streams = [torch.cuda.current_stream()]
        comm_streams = get_backend_stream(size=self.tp_size - 1, priority=0, prefix="comm")

        copy_streams = get_backend_stream(size=1, priority=0, prefix="copy")
        if A_copy is not None:
            if A_copy.shape != local_A.shape:
                raise ValueError("A_copy shape is different with local_A")
            A_copy.copy_(local_A)

        if scaled_mm_kwargs_list is None:
            pt.ops.fused_all_gather_matmul(
                local_A,
                [B],
                [layout],
                gather_dim=0,
                group_name=self.group_name,
                gemm_streams=gemm_streams,
                comm_streams=comm_streams,
                copy_streams=copy_streams,
                comm_method="pipeline",
                num_splits=self.num_splits,
                skip_copy_local_ag_out=True,
                return_A=True,
                A_out=A_out,
                outputs=[D],
            )
        else:
            local_A = local_A.view(A_out.dtype)
            pt.ops.fused_all_gather_scaled_matmul(
                local_A,
                [B],
                [layout],
                A_scale,
                scaled_mm_kwargs_list,
                gather_dim=0,
                group_name=self.group_name,
                gemm_streams=gemm_streams,
                comm_streams=comm_streams,
                copy_streams=copy_streams,
                out_dtypes=[scaled_mm_kwargs["out_dtype"] for scaled_mm_kwargs in scaled_mm_kwargs_list],
                skip_copy_local_ag_out=True,
                A_out=A_out,
                mm_out=[D],
            )


class CommOverlapP2P(CommOverlapBase):
    def __init__(
        self,
        buffer_shape: List[int],
        buffer_dtype: torch.dtype,
        group_name: str,
        tp_size: int,
        comm_type: CommOverlapType,
        num_max_streams: int = 3,
        comm_cga_size: int = 1,
        gemm_priority: int = 0,
        comm_priority: int = 0,
        num_comm_sm: int = 1,
        set_sm_margin: bool = True,
        atomic_gemm: bool = False,
        use_ce: bool = True,
        aggregate: bool = False,
    ):
        super().__init__(buffer_shape, buffer_dtype, group_name, tp_size)

    def is_atomic_gemm(self) -> bool: ...

    def is_p2p_overlap(self) -> bool:
        return True

    def is_fp8_ubuf(self) -> bool:
        return False

    def split_overlap_rs(self, A_out: torch.Tensor, B: torch.Tensor, layout: str, D: torch.Tensor):
        raise NotImplementedError("not support for now!")

    def split_overlap_ag(self, A_out: torch.Tensor, B: torch.Tensor, layout: str, D: torch.Tensor):
        raise NotImplementedError("not support for now!")

    def copy_input_to_ubuf(self, input: torch.Tensor, comm_type: int) -> None:
        raise NotImplementedError("not support for now!")
