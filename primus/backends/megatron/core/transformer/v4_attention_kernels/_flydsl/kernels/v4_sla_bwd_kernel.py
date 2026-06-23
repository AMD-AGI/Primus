# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""v4_sla_bwd_kernel: V4 SWA attention backward kernels (FlyDSL, STEP 1).

Forked from /workspace/FlyDSL-amd/kernels/sla_bwd_preprocess.py.

STEP 1 SCOPE
============
Only the preprocess kernel is implemented in this file. dq / dkv are still
handled by Triton in the v4_attention_bwd_flydsl_mqa wrapper (see that file
for the rationale). The hooks defined here will be the landing spots for
the dq / dkv kernels in STEP 1b / STEP 1c.

PREPROCESS KERNEL (D scalar)
---------------------------
Computes ``delta[b, h, m] = sum_d (out[b, h, m, d] * dout[b, h, m, d])`` in
fp32 for every query row. This is the standard FA-2 pre-pass and is dense
(no SWA / sink dependency) so it is the simplest piece to lift to FlyDSL
first.

Inputs (flat views):
    OS, DOS  shape (N_ROWS, D) where N_ROWS = B*HQ*Sq
    DELTAS   shape (N_ROWS,)    fp32

Grid: (N_ROWS / BLOCK_ROWS, 1, 1).
Block: BLOCK_ROWS * THREADS_PER_ROW threads.

Each row uses THREADS_PER_ROW = D // VEC_WIDTH threads. Cross-thread row
reduction uses ``shuffle_xor`` at offsets THREADS_PER_ROW/2, /4, ..., 1
which stays inside a THREADS_PER_ROW-lane subgroup (XOR never carries out
of a power-of-2 subgroup).

Requires: head_dim % VEC_WIDTH == 0, N_ROWS % BLOCK_ROWS == 0.
"""

import math
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, buffer_ops, range_constexpr
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T, Int32
from flydsl.expr.numeric import Float32
from flydsl.expr.vector import ReductionOp
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from kernels.kernels_common import dtype_to_elem_type

KERNEL_NAME_PRE = "v4_swa_bwd_preprocess_kernel"

# Wider VEC_WIDTH than the SLA kernel (D=512 vs D=128). At D=512 with
# VEC_WIDTH=8 we would need 64 threads per row (== a full warp), which is
# fine but uses one wave per BLOCK_ROWS rows. We use VEC_WIDTH=16 so
# THREADS_PER_ROW=32 and a 256-thread block processes 8 rows = 2 waves
# (more parallel tile generators per CU at the cost of slightly wider
# loads). Either is correct.
VEC_WIDTH = 8
WARP_SIZE = 64


def build_v4_swa_bwd_preprocess_module(
    head_dim,
    dtype_str="bf16",
    block_rows=None,
):
    """Build the V4 SWA backward preprocess launcher.

    Inputs (flattened to 2D by the launcher):
        OS, DOS  shape (N_ROWS, D) where N_ROWS = B*H*L
        DELTAS   shape (N_ROWS,)    f32

    Args:
        head_dim: head dimension (must be divisible by VEC_WIDTH).
        dtype_str: "bf16" or "f16" for O_S / DO_S; DELTAS is always f32.
        block_rows: rows processed per block. Default 8.
    """
    gpu_arch = get_hip_arch()

    if block_rows is None:
        BLOCK_ROWS = 8
    else:
        BLOCK_ROWS = block_rows

    assert head_dim % VEC_WIDTH == 0, f"head_dim {head_dim} must be % {VEC_WIDTH}"
    assert head_dim >= VEC_WIDTH, f"head_dim {head_dim} < VEC_WIDTH {VEC_WIDTH}"
    assert dtype_str in ("bf16", "f16"), f"unsupported dtype {dtype_str}"

    D = head_dim
    THREADS_PER_ROW = D // VEC_WIDTH  # 32 for D=512, VEC_WIDTH=16
    BLOCK_THREADS = BLOCK_ROWS * THREADS_PER_ROW
    assert BLOCK_THREADS <= 1024, f"BLOCK_THREADS {BLOCK_THREADS} > 1024"
    assert THREADS_PER_ROW & (THREADS_PER_ROW - 1) == 0, \
        "THREADS_PER_ROW must be power of 2 for shfl_xor reduction"

    elem_bits = 16  # bf16 / f16

    @flyc.kernel
    def v4_swa_bwd_preprocess_kernel(
        OS: fx.Tensor,   # (N_ROWS, D)
        DOS: fx.Tensor,  # (N_ROWS, D)
        DELTAS: fx.Tensor,  # (N_ROWS,)
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_type = dtype_to_elem_type(dtype_str)
        compute_type = T.f32
        fm_fast = arith.FastMathFlags.fast

        # ---- Thread decomposition ----
        row_in_block_i32 = tid // Int32(THREADS_PER_ROW)
        col_in_row_i32 = tid % Int32(THREADS_PER_ROW)

        global_row_i32 = bid * Int32(BLOCK_ROWS) + row_in_block_i32
        global_row_idx = ArithValue(global_row_i32).index_cast(T.index)

        # ---- Buffer-backed 2D tensors for loads ----
        OS_buf = fx.rocdl.make_buffer_tensor(OS)
        DOS_buf = fx.rocdl.make_buffer_tensor(DOS)
        delta_rsrc = buffer_ops.create_buffer_resource(DELTAS, max_size=True)

        row_os = fx.slice(OS_buf, (global_row_i32, None))
        row_dos = fx.slice(DOS_buf, (global_row_i32, None))

        in_div_os = fx.logical_divide(row_os, fx.make_layout(VEC_WIDTH, 1))
        in_div_dos = fx.logical_divide(row_dos, fx.make_layout(VEC_WIDTH, 1))

        copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
        vec_reg_ty = fx.MemRefType.get(
            elem_type, fx.LayoutType.get(VEC_WIDTH, 1), fx.AddressSpace.Register
        )
        vec_reg_lay = fx.make_layout(VEC_WIDTH, 1)

        def _load_vec(div_tensor, col_idx):
            r = fx.memref_alloca(vec_reg_ty, vec_reg_lay)
            fx.copy_atom_call(copy_atom, fx.slice(div_tensor, (None, col_idx)), r)
            return fx.memref_load_vec(r)

        os_vec = _load_vec(in_div_os, col_in_row_i32)
        dos_vec = _load_vec(in_div_dos, col_in_row_i32)

        os_f32 = os_vec.to(Float32)
        dos_f32 = dos_vec.to(Float32)
        prod_f32 = os_f32 * dos_f32
        local_sum = prod_f32.reduce(ReductionOp.ADD, fastmath=fm_fast)

        width_i32 = Int32(WARP_SIZE)
        val = local_sum
        num_rounds = int(math.log2(THREADS_PER_ROW))
        for _sh_exp in range_constexpr(num_rounds):
            off = Int32(THREADS_PER_ROW // (2 << _sh_exp))
            peer = val.shuffle_xor(off, width_i32)
            val = val.addf(peer, fastmath=fm_fast)

        if col_in_row_i32 == Int32(0):
            delta_off_i32 = arith.index_cast(T.i32, global_row_idx)
            val_ir = val.ir_value() if hasattr(val, "ir_value") else val
            buffer_ops.buffer_store(val_ir, delta_rsrc, delta_off_i32)

    @flyc.jit
    def launch_v4_swa_bwd_preprocess(
        OS: fx.Tensor,
        DOS: fx.Tensor,
        DELTAS: fx.Tensor,
        n_rows: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        """Launch grid: (n_rows / BLOCK_ROWS, 1, 1).

        Expects OS, DOS to be views of shape (n_rows, D). DELTAS is (n_rows,).
        """
        blocks_i32 = n_rows // Int32(BLOCK_ROWS)
        blocks_idx = ArithValue(blocks_i32).index_cast(T.index)
        launcher = v4_swa_bwd_preprocess_kernel(OS, DOS, DELTAS)
        launcher.launch(
            grid=(blocks_idx, arith.index(1), arith.index(1)),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_v4_swa_bwd_preprocess


# Convenience alias used by the wrapper.
def build_v4_swa_bwd_preprocess(*args, **kwargs):
    return build_v4_swa_bwd_preprocess_module(*args, **kwargs)



# ---------------------------------------------------------------------------
# STEP 1b: dq kernel re-export (forked from kernels/sla_bwd_dq.py).
# Lives in v4_sla_bwd_dq_kernel.py for file-size reasons; re-exported here
# so the wrapper can import a single module.
# ---------------------------------------------------------------------------
import os as _os
import sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)
from v4_sla_bwd_dq_kernel import (  # noqa: E402,F401
    build_v4_swa_bwd_dq_module,
    build_v4_swa_bwd_dq_module_primary,
)
