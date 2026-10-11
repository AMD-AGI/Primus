###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Exact-shape BF16 head wgrad on the qualified hipBLASLt runtime.

Unsupported shapes use Apex. Each CUDA stream owns its context/workspace;
updates retain beta=1 and the original BF16 main_grad accumulation.
Set PRIMUS_LLAMA8B_HEAD_WGRAD=0 to disable.
"""

import ctypes as C
import hashlib
import os
from pathlib import Path

from primus.backends.megatron.patches.llama8b_ce_patches import enabled
from primus.core.patches import register_patch

WORKSPACE = 32 * 1024 * 1024
HIPBLASLT_SHA256 = "59cf9d003b4dd9209df04c9c4edddae320218241bca862af0074126fcdb78c41"
APEX_SOURCE = (
    "/opt/venv/lib/python3.12/site-packages/apex/csrc/megatron/fused_weight_gradient_dense_16bit_prec_hip.hip"
)
APEX_SHA256 = "c75b664b6db1b4098057fa2fa116f7891ff5916ea035a9d5f2d6299d44ff5b54"


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def assert_tensor_signature(x, dy, out, device):
    import torch

    expected = (
        (x, (32768, 4096), (4096, 1), torch.bfloat16),
        (dy, (32768, 128256), (128256, 1), torch.bfloat16),
        (out, (128256, 4096), (4096, 1), torch.bfloat16),
    )
    spans = []
    for tensor, shape, stride, dtype in expected:
        if (
            tuple(tensor.shape) != shape
            or tuple(tensor.stride()) != stride
            or tensor.dtype != dtype
            or tensor.device != torch.device("cuda", device)
            or not tensor.is_contiguous()
        ):
            raise ValueError("Exact head shape/stride/dtype/device assertion failed")
        if tensor.requires_grad:
            raise ValueError("This native adapter has no autograd; pass detached tensors")
        start = tensor.data_ptr()
        if start % 16:
            raise ValueError("Require 16-byte aligned operand pointers")
        spans.append((start, start + tensor.numel() * tensor.element_size()))
    for i, (lo, hi) in enumerate(spans):
        for lo2, hi2 in spans[:i]:
            if lo < hi2 and lo2 < hi:
                raise ValueError("X, dY and output must not overlap")


class NativeHead:
    """Thin ctypes ABI, serial host use only; construction DOES use HIP."""

    def __init__(self, library, device):
        self.path = Path(library).resolve()
        self.device = int(device)
        self.lib = C.CDLL(str(self.path))
        p, i, u = C.c_void_p, C.c_int, C.c_uint64
        pi, pu, pp = C.POINTER(i), C.POINTER(u), C.POINTER(p)
        signatures = {
            "hw_create": ([i, pp], i),
            "hw_candidate": ([p, i, pi, pu, pi], i),
            "hw_load_index": ([p, i], i),
            "hw_name": ([p, i, i, p, u, pu], i),
            "hw_run": ([p, i, p, p, p, p, u, p], i),
            "hw_destroy": ([p], i),
            "hw_last_error": ([], C.c_char_p),
            "hw_workspace_limit": ([], u),
            "hw_output_dtype_code": ([], C.c_uint32),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes = args
            fn.restype = result
        if self.lib.hw_output_dtype_code() != 1:
            raise RuntimeError("Native library must have BF16 C/D contract")
        self.context = p()
        self.check(self.lib.hw_create(self.device, C.byref(self.context)))
        if self.lib.hw_workspace_limit() != WORKSPACE:
            self.close()
            raise RuntimeError("Unexpected native workspace limit")

    def check(self, status):
        if status:
            message = self.lib.hw_last_error()
            raise RuntimeError(f'native status {status}: {message.decode() if message else ""}')

    def close(self):
        if self.context.value:
            status = self.lib.hw_destroy(self.context)
            # A device-mismatch error retains context; other cleanup attempts consume it.
            if status != -2:
                self.context = C.c_void_p()
            self.check(status)

    def name(self, slot, which):
        needed = C.c_uint64()
        self.check(self.lib.hw_name(self.context, slot, which, None, 0, C.byref(needed)))
        text = C.create_string_buffer(needed.value)
        self.check(self.lib.hw_name(self.context, slot, which, text, len(text), C.byref(needed)))
        return text.value.decode()

    def metadata(self, slot):
        index, rank, workspace = C.c_int(), C.c_int(), C.c_uint64()
        self.check(
            self.lib.hw_candidate(self.context, slot, C.byref(index), C.byref(workspace), C.byref(rank))
        )
        return dict(
            slot=slot,
            index=index.value,
            workspace_bytes=workspace.value,
            heuristic_rank=rank.value,
            solution=self.name(slot, 0),
            kernel=self.name(slot, 1),
        )

    def load(self, index):
        self.check(self.lib.hw_load_index(self.context, int(index)))
        return self.metadata(0)

    def run(self, slot, x, dy, out, workspace, stream):
        self.check(
            self.lib.hw_run(
                self.context,
                slot,
                x.data_ptr(),
                dy.data_ptr(),
                out.data_ptr(),
                workspace.data_ptr(),
                workspace.numel(),
                stream.cuda_stream,
            )
        )

    def identity(self):
        libraries = set()
        for line in Path("/proc/self/maps").read_text().splitlines():
            parts = line.split(maxsplit=5)
            if len(parts) == 6 and "/libhipblaslt.so" in parts[5]:
                libraries.add(str(Path(parts[5]).resolve()))
        if not libraries:
            raise RuntimeError("Cannot identify loaded hipBLASLt library")
        return sorted({digest(p) for p in libraries})


def build_native():
    """Use PyTorch's locked extension cache; concurrent ranks share one build."""
    from torch.utils.cpp_extension import load

    sdk = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
    return load(
        name="primus_llama8b_head_wgrad",
        sources=[str(Path(__file__).with_name("llama8b_head_wgrad.cpp"))],
        extra_cflags=["-O2", "-std=c++17", "-D__HIP_PLATFORM_AMD__"],
        extra_include_paths=[str(sdk / "include")],
        extra_ldflags=[
            f"-L{sdk / 'lib'}",
            f"-Wl,-rpath,{sdk / 'lib'}",
            "-Wl,-z,defs",
            "-lhipblaslt",
            "-lamdhip64",
        ],
        with_cuda=False,
        is_python_module=False,
        verbose=False,
    )


@register_patch(
    "megatron.llama8b.head_wgrad",
    backend="megatron",
    phase="before_train",
    condition=lambda ctx: enabled(ctx, "llama8b_head_wgrad", "PRIMUS_LLAMA8B_HEAD_WGRAD"),
    description="Use qualified hipBLASLt algorithm 12109 for Llama 8B head wgrad",
)
def install_head_wgrad(ctx):
    import torch
    from megatron.core.tensor_parallel import layers

    apex = layers.fused_weight_gradient_mlp_cuda
    original = apex.wgrad_gemm_accum_fp16
    if getattr(original, "_llama8b_head_wgrad", False):
        return
    if (
        str(torch.__version__) != "2.12.0+rocm7.15.0a20260727"
        or torch.version.hip != "7.15.0"
        or digest(APEX_SOURCE) != APEX_SHA256
        or torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(":")[0] != "gfx950"
    ):
        raise RuntimeError("Llama 8B head wgrad requires the qualified MI355X runtime")
    library = build_native()
    contexts = {}

    def wrapped(x, dy, out):
        if (
            not (x.is_cuda and dy.is_cuda and out.is_cuda)
            or x.device != dy.device
            or x.device != out.device
            or x.dtype != torch.bfloat16
            or dy.dtype != x.dtype
            or out.dtype != x.dtype
            or tuple(x.shape) != (32768, 4096)
            or tuple(dy.shape) != (32768, 128256)
            or tuple(out.shape) != (128256, 4096)
            or not (x.is_contiguous() and dy.is_contiguous() and out.is_contiguous())
            or torch.cuda.is_current_stream_capturing()
        ):
            return original(x, dy, out)
        x, dy, out = x.detach(), dy.detach(), out.detach()
        device = x.device.index
        assert_tensor_signature(x, dy, out, device)
        if torch.cuda.current_device() != device:
            raise ValueError("Select the head wgrad device before invocation")
        stream = torch.cuda.current_stream(device)
        key = (device, stream.cuda_stream)
        if key not in contexts:
            native = NativeHead(library, device)
            try:
                if native.identity() != [HIPBLASLT_SHA256]:
                    raise RuntimeError("Unqualified hipBLASLt library for algorithm 12109")
                meta = native.load(12109)
                if meta["index"] != 12109 or meta["workspace_bytes"] > WORKSPACE:
                    raise RuntimeError("Unexpected head wgrad algorithm or workspace")
                workspace = torch.empty(WORKSPACE, dtype=torch.uint8, device=x.device)
            except BaseException:
                native.close()
                raise
            contexts[key] = (native, workspace, stream)
        native, workspace, owned_stream = contexts[key]
        native.run(0, x, dy, out, workspace, owned_stream)
        return None

    wrapped._llama8b_head_wgrad = True
    apex.wgrad_gemm_accum_fp16 = wrapped
