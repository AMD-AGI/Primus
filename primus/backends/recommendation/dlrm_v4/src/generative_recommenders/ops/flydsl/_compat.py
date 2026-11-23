# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The FlyDSL 0.2 API the kernels here are written against, on FlyDSL 0.2 or >= 0.3.

    from generative_recommenders.ops.flydsl._compat import buffer_ops, rocdl_raw, vector

FlyDSL 0.2.x: the library's own `flydsl.expr.buffer_ops`, `flydsl.expr.vector`
and `flydsl._mlir.dialects.rocdl`, unchanged.

FlyDSL >= 0.3 (rocm/primus:v26.9 ships 0.3.4.1) removed `expr.buffer_ops` and
`expr.vector`, and its raw ROCDL op builders changed:
  * buffer_ops / vector: 0.2.4's modules (_compat_buffer_ops.py, _compat_vector.py);
  * RawPtrBufferStoreOp / RawPtrBufferAtomicFaddOp take the cache-policy `aux`
    as an attribute (0.2: an i32 SSA value), and the atomic now has a result;
  * mfma_f32_16x16x32_{bf16,fp8_fp8} take `blgp` as #rocdl<mfma_perm_b ...>,
    mfma_scale_f32_16x16x128_f8f6f4 takes cbsz / blgp as #rocdl<matrix_format ...>
    (0.2: i32 attributes; same encodings, so the same instructions).
`rocdl_raw` forwards everything else to `flydsl._mlir.dialects.rocdl`.
"""

from types import ModuleType

from flydsl._mlir import ir
from flydsl._mlir.dialects import rocdl as _rocdl

try:  # FlyDSL 0.2.x
    from flydsl.expr import buffer_ops, vector  # noqa: F401

    FLYDSL_02 = True
except ImportError:  # FlyDSL >= 0.3
    FLYDSL_02 = False
    from generative_recommenders.ops.flydsl import _compat_buffer_ops as buffer_ops  # noqa: F401
    from generative_recommenders.ops.flydsl import _compat_vector as vector  # noqa: F401

# int -> keyword of the ROCDL enum attributes (FlyDSL 0.3.4.1 expr/rocdl/__init__.py).
_MFMA_PERM_B = {0: "none", 1: "bcast_first_32", 2: "bcast_second_32", 4: "bcast_first_16", 5: "bcast_second_16"}
_MATRIX_FORMAT = {0: "fp8_e4m3", 1: "fp8_e5m2", 2: "fp6_e2m3", 3: "fp6_e3m2", 4: "fp4_e2m1"}


def _enum_attr(value, table, name):
    if isinstance(value, ir.Attribute):
        return value
    return ir.Attribute.parse(f"#rocdl<{name} {table[int(value)]}>")


def _aux_attr(aux):
    """0.2's aux operand (a Python int or an i32 arith.constant) as the i32 attribute 0.3 wants."""
    if aux is None or isinstance(aux, ir.Attribute):
        return aux
    if not isinstance(aux, int):
        value = aux.owner.attributes["value"]  # the arith.constant the 0.2 call sites pass
        aux = ir.IntegerAttr(value).value
    return ir.IntegerAttr.get(ir.IntegerType.get_signless(32), int(aux))


class _RocdlRaw(ModuleType):
    """`flydsl._mlir.dialects.rocdl` with the 0.2 signatures of the ops above."""

    def __getattr__(self, name):
        return getattr(_rocdl, name)

    @staticmethod
    def RawPtrBufferStoreOp(vdata, rsrc, offset, soffset, aux):
        return _rocdl.RawPtrBufferStoreOp(vdata, rsrc, offset, soffset, aux=_aux_attr(aux))

    @staticmethod
    def RawPtrBufferAtomicFaddOp(vdata, rsrc, offset, soffset, aux):
        return _rocdl.RawPtrBufferAtomicFaddOp(vdata.type, vdata, rsrc, offset, soffset, aux=_aux_attr(aux))

    @staticmethod
    def mfma_f32_16x16x32_bf16(res, a, b, c, cbsz, abid, blgp):
        return _rocdl.mfma_f32_16x16x32_bf16(res, a, b, c, cbsz, abid, _enum_attr(blgp, _MFMA_PERM_B, "mfma_perm_b"))

    @staticmethod
    def mfma_f32_16x16x32_fp8_fp8(res, a, b, c, cbsz, abid, blgp):
        return _rocdl.mfma_f32_16x16x32_fp8_fp8(
            res, a, b, c, cbsz, abid, _enum_attr(blgp, _MFMA_PERM_B, "mfma_perm_b")
        )

    @staticmethod
    def mfma_scale_f32_16x16x128_f8f6f4(res, a, b, c, cbsz, blgp, opsel_a, scale_a, opsel_b, scale_b):
        return _rocdl.mfma_scale_f32_16x16x128_f8f6f4(
            res,
            a,
            b,
            c,
            _enum_attr(cbsz, _MATRIX_FORMAT, "matrix_format"),
            _enum_attr(blgp, _MATRIX_FORMAT, "matrix_format"),
            opsel_a,
            scale_a,
            opsel_b,
            scale_b,
        )


rocdl_raw = _rocdl if FLYDSL_02 else _RocdlRaw("rocdl_raw")
