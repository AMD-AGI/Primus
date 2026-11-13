# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: Apache-2.0
"""FP8 NN GEMM through the hipBLASLt C API (ctypes), for weight gradients whose
first operand is stored one token per row.

The FP8 output-projection wgrad is ``dW[N, C] = dout^T @ y``, contracted over the
tokens. ``y`` is stored row-major ``[M, C]`` (tokens are rows). ``torch._scaled_mm``
only accepts row-major A with column-major B (both operands K-contiguous), so with
``y`` as stored it needs a token-major copy of y. hipBLASLt itself also runs the NN
layout, which reads ``y`` exactly as stored:

    column-major view: dW^T (C x N) = y^T (C x M) @ dout (M x N)
      A = y        (op N, lda = C:   element (c, t) at c + t*C   -> row-major y)
      B = dout^T   (op N, ldb = Mp:  element (t, n) at t + n*Mp  -> row-major [N, Mp])
      D = dW       (ld = C:          element (c, n) at c + n*C   -> row-major [N, C])

That removes the y^T copy (a full extra write and read of y per layer). The NN
kernel itself is slower than the TN kernel at this shape, but cheaper than TN
plus the y^T copy it would need.

All calls are async on the current torch stream; the algo comes from the library
heuristic and is cached per (device, shape, ldb). Returns None when the library
offers no algorithm, so callers keep a fallback.
"""

import ctypes
import os
import threading
from typing import Dict, Optional, Tuple

import torch

# hipBLASLt / HIP enum values (hipblaslt.h, hipblas-common.h, library_types.h of
# the ROCm 7.2 image; checked against the image's installed headers).
_HIPBLAS_OP_N = 111
_HIPBLAS_COMPUTE_32F = 2
_HIP_R_32F = 0
_HIP_R_16BF = 14
_HIP_R_8F_E4M3 = 28
_HIP_R_8F_E5M2 = 29
_DESC_TRANSA = 0
_DESC_TRANSB = 1
_DESC_A_SCALE_POINTER = 5
_DESC_B_SCALE_POINTER = 6
_PREF_MAX_WORKSPACE_BYTES = 1

_TORCH_TO_HIP = {
    torch.float8_e4m3fn: _HIP_R_8F_E4M3,
    torch.float8_e5m2: _HIP_R_8F_E5M2,
    torch.float32: _HIP_R_32F,
    torch.bfloat16: _HIP_R_16BF,
}

# "exact" (default): one heuristic query per distinct (M, ldb), so the algo always
# matches the problem it runs (a host-side query on the first call of each token
# count; the step is host-sync free, so it stays off the GPU critical path).
# "shape": one query per (C, N), reused for every M -- correct at the token
# counts probed but not guaranteed, since solutions can carry K predicates.
_ALGO_CACHE_MODE = os.environ.get("DLRM_HSTU_FP8_HBLT_NN_ALGO_CACHE", "exact")

_WORKSPACE_BYTES = int(
    os.environ.get("DLRM_HSTU_FP8_HBLT_NN_WORKSPACE_MB", "64")
) * (1 << 20)


class _Algo(ctypes.Structure):
    _fields_ = [("data", ctypes.c_uint8 * 16), ("max_workspace_bytes", ctypes.c_size_t)]


class _HeuristicResult(ctypes.Structure):
    _fields_ = [
        ("algo", _Algo),
        ("workspaceSize", ctypes.c_size_t),
        ("state", ctypes.c_int),
        ("wavesCount", ctypes.c_float),
        ("reserved", ctypes.c_int * 4),
    ]


_lock = threading.Lock()
_lib: Optional[ctypes.CDLL] = None
_handles: Dict[int, ctypes.c_void_p] = {}
_workspaces: Dict[int, torch.Tensor] = {}
_algos: Dict[Tuple, Optional[Tuple[_HeuristicResult, int]]] = {}


def _library() -> ctypes.CDLL:
    """The libhipblaslt torch itself loaded (same copy, same kernel database)."""
    global _lib
    if _lib is not None:
        return _lib
    path = None
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    for line in open("/proc/self/maps"):
        p = line.split()[-1]
        if "/libhipblaslt.so" in p:
            path = p
            break
    if path is None:
        cand = os.path.join(torch_lib, "libhipblaslt.so")
        path = cand if os.path.exists(cand) else "libhipblaslt.so"
    lib = ctypes.CDLL(path)
    vp, sz = ctypes.c_void_p, ctypes.c_size_t
    lib.hipblasLtCreate.argtypes = [ctypes.POINTER(vp)]
    lib.hipblasLtMatrixLayoutCreate.argtypes = [
        ctypes.POINTER(vp), ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int64
    ]
    lib.hipblasLtMatrixLayoutDestroy.argtypes = [vp]
    lib.hipblasLtMatmulDescCreate.argtypes = [ctypes.POINTER(vp), ctypes.c_int, ctypes.c_int]
    lib.hipblasLtMatmulDescDestroy.argtypes = [vp]
    lib.hipblasLtMatmulDescSetAttribute.argtypes = [vp, ctypes.c_int, vp, sz]
    lib.hipblasLtMatmulPreferenceCreate.argtypes = [ctypes.POINTER(vp)]
    lib.hipblasLtMatmulPreferenceDestroy.argtypes = [vp]
    lib.hipblasLtMatmulPreferenceSetAttribute.argtypes = [vp, ctypes.c_int, vp, sz]
    lib.hipblasLtMatmulAlgoGetHeuristic.argtypes = [
        vp, vp, vp, vp, vp, vp, vp, ctypes.c_int,
        ctypes.POINTER(_HeuristicResult), ctypes.POINTER(ctypes.c_int),
    ]
    lib.hipblasLtMatmul.argtypes = [
        vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp,
        ctypes.POINTER(_Algo), vp, sz, vp,
    ]
    _lib = lib
    return lib


def _check(status: int, what: str) -> None:
    if status != 0:
        raise RuntimeError(f"hipBLASLt {what} failed with status {status}")


class _Desc:
    """Matmul descriptor + layouts for one call; destroyed after the launch (the
    library copies everything it needs at hipblasLtMatmul enqueue time)."""

    def __init__(self, lib, a_type, b_type, d_type, m, n, k, lda, ldb, ldd, sa, sb):
        self.lib = lib
        vp = ctypes.c_void_p
        self.desc, self.la, self.lb, self.ld = vp(), vp(), vp(), vp()
        _check(lib.hipblasLtMatmulDescCreate(ctypes.byref(self.desc), _HIPBLAS_COMPUTE_32F, _HIP_R_32F), "DescCreate")
        op = ctypes.c_int(_HIPBLAS_OP_N)
        for attr in (_DESC_TRANSA, _DESC_TRANSB):
            _check(lib.hipblasLtMatmulDescSetAttribute(self.desc, attr, ctypes.byref(op), 4), "SetTrans")
        self._sa = ctypes.c_void_p(sa)
        self._sb = ctypes.c_void_p(sb)
        _check(lib.hipblasLtMatmulDescSetAttribute(self.desc, _DESC_A_SCALE_POINTER, ctypes.byref(self._sa), 8), "SetScaleA")
        _check(lib.hipblasLtMatmulDescSetAttribute(self.desc, _DESC_B_SCALE_POINTER, ctypes.byref(self._sb), 8), "SetScaleB")
        _check(lib.hipblasLtMatrixLayoutCreate(ctypes.byref(self.la), a_type, m, k, lda), "LayoutA")
        _check(lib.hipblasLtMatrixLayoutCreate(ctypes.byref(self.lb), b_type, k, n, ldb), "LayoutB")
        _check(lib.hipblasLtMatrixLayoutCreate(ctypes.byref(self.ld), d_type, m, n, ldd), "LayoutD")

    def close(self) -> None:
        self.lib.hipblasLtMatrixLayoutDestroy(self.la)
        self.lib.hipblasLtMatrixLayoutDestroy(self.lb)
        self.lib.hipblasLtMatrixLayoutDestroy(self.ld)
        self.lib.hipblasLtMatmulDescDestroy(self.desc)


def _handle_and_workspace(device: torch.device) -> Tuple[ctypes.c_void_p, torch.Tensor]:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    with _lock:
        if idx not in _handles:
            lib = _library()
            h = ctypes.c_void_p()
            with torch.cuda.device(idx):
                _check(lib.hipblasLtCreate(ctypes.byref(h)), "Create")
                _workspaces[idx] = torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device=f"cuda:{idx}")
            _handles[idx] = h
        return _handles[idx], _workspaces[idx]


def fp8_wgrad_nn(
    y: torch.Tensor,
    dout_t: torch.Tensor,
    scale_y: torch.Tensor,
    scale_d: torch.Tensor,
    out_dtype: torch.dtype = torch.float32,
) -> Optional[torch.Tensor]:
    """``dW[N, C] = scale_y * scale_d * dout_t[:, :M] @ y`` in one hipBLASLt FP8 NN
    GEMM. y: [M, C] row-major FP8. dout_t: [N, Mp] row-major FP8 (Mp >= M; columns
    beyond M are not read). Scales: fp32 device scalars (dequant). Returns None if
    the library has no algorithm for the problem."""
    assert y.dim() == 2 and dout_t.dim() == 2 and y.stride(1) == 1 and dout_t.stride(1) == 1
    m_tok, c = y.shape
    n, mp = dout_t.shape
    assert mp >= m_tok and y.stride(0) == c
    a_type = _TORCH_TO_HIP[y.dtype]
    b_type = _TORCH_TO_HIP[dout_t.dtype]
    d_type = _TORCH_TO_HIP[out_dtype]
    ldb = dout_t.stride(0)
    try:
        lib = _library()
        handle, ws = _handle_and_workspace(y.device)
    except Exception:
        # Library load / symbol lookup / handle creation failed on this host:
        # degrade to the caller's transpose+TN fallback instead of crashing backward.
        return None
    out = torch.empty((n, c), dtype=out_dtype, device=y.device)
    try:
        desc = _Desc(lib, a_type, b_type, d_type, c, n, m_tok, c, ldb, c,
                     scale_y.data_ptr(), scale_d.data_ptr())
    except Exception:
        # Descriptor setup refused: take the caller's transpose+TN fallback.
        return None
    try:
        key = (y.device.index, a_type, b_type, d_type, c, n)
        if _ALGO_CACHE_MODE == "exact":
            key = key + (m_tok, ldb)
        hit = _algos.get(key)
        if key not in _algos:
            pref = ctypes.c_void_p()
            _check(lib.hipblasLtMatmulPreferenceCreate(ctypes.byref(pref)), "PrefCreate")
            wsb = ctypes.c_uint64(ws.numel())
            _check(lib.hipblasLtMatmulPreferenceSetAttribute(pref, _PREF_MAX_WORKSPACE_BYTES, ctypes.byref(wsb), 8), "PrefWs")
            res = (_HeuristicResult * 1)()
            count = ctypes.c_int(0)
            st = lib.hipblasLtMatmulAlgoGetHeuristic(
                handle, desc.desc, desc.la, desc.lb, desc.ld, desc.ld, pref, 1, res, ctypes.byref(count)
            )
            lib.hipblasLtMatmulPreferenceDestroy(pref)
            hit = (res[0], count.value) if st == 0 and count.value > 0 and res[0].state == 0 else None
            if len(_algos) > 4096:
                _algos.clear()
            _algos[key] = hit
        if hit is None:
            return None
        result = hit[0]
        if result.workspaceSize > ws.numel():
            return None
        alpha = ctypes.c_float(1.0)
        beta = ctypes.c_float(0.0)
        stream = torch.cuda.current_stream(y.device).cuda_stream
        status = (
            lib.hipblasLtMatmul(
                handle, desc.desc, ctypes.byref(alpha),
                ctypes.c_void_p(y.data_ptr()), desc.la,
                ctypes.c_void_p(dout_t.data_ptr()), desc.lb,
                ctypes.byref(beta),
                ctypes.c_void_p(out.data_ptr()), desc.ld,
                ctypes.c_void_p(out.data_ptr()), desc.ld,
                ctypes.byref(result.algo),
                ctypes.c_void_p(ws.data_ptr()), ctypes.c_size_t(ws.numel()),
                ctypes.c_void_p(stream),
            )
        )
        if status != 0:
            # The library refused the algo for this problem: drop it and let
            # the caller take its fallback (no partial output is used).
            _algos.pop(key, None)
            return None
    finally:
        desc.close()
    # Everything (workspace, inputs, output) is used on the current stream only.
    return out
