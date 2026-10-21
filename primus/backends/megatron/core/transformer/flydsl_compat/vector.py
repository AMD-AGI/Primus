###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025 FlyDSL Project Contributors
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Adapted from FlyDSL 0.2.4 (https://github.com/ROCm/FlyDSL)
#   python/flydsl/expr/vector.py
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus (see LICENSE).
###############################################################################

"""FlyDSL 0.2.4 ``flydsl.expr.vector`` helpers, kept for Primus kernels on 0.3.x.

FlyDSL 0.3 dropped ``flydsl.expr.vector`` from the wheel. The MLIR dialect
(``flydsl._mlir.dialects.vector``) remains; this module re-exports it and keeps
the thin wrappers (``load_op``, ``extract``, …) the Primus MSA / DSA / KDA
kernels call.
"""

from __future__ import annotations

from flydsl._mlir import ir
from flydsl._mlir.dialects import vector as _vector

# Re-export upstream dialect for ``vector.broadcast(...)`` etc.
from flydsl._mlir.dialects.vector import *  # noqa: F401,F403,E402
from flydsl.expr.meta import dsl_loc_tracing
from flydsl.expr.typing import (  # noqa: F401
    ReductionOp,
    Vector,
    empty_like,
    full,
    full_like,
    ones_like,
    zeros_like,
)


@dsl_loc_tracing
def from_elements(*args, **kwargs):
    """Construct a vector from scalar elements, auto-unwrapping ArithValue wrappers."""
    from flydsl.expr import arith as _arith_ext

    if len(args) >= 2:
        args = list(args)
        elems = args[1]
        if isinstance(elems, (list, tuple)):
            args[1] = [_arith_ext.unwrap(v) for v in elems]
        return _vector.from_elements(*args, **kwargs)

    return _vector.from_elements(*args, **kwargs)


@dsl_loc_tracing
def store(value, memref, indices, **kwargs):
    """Vector store wrapper that accepts ArithValue/wrappers for value/indices."""
    from flydsl.expr import arith as _arith_ext

    return _vector.store(
        _arith_ext.unwrap(value),
        _arith_ext.unwrap(memref),
        [_arith_ext.unwrap(i, index=True) for i in indices],
        **kwargs,
    )


@dsl_loc_tracing
def extract(vector, static_position=None, dynamic_position=None):
    """Wrapper around ``vector.ExtractOp(...).result``."""
    from flydsl.expr import arith as _arith_ext

    if static_position is None:
        static_position = []
    if dynamic_position is None:
        dynamic_position = []
    dynamic_position = [_arith_ext.unwrap(i, index=True) for i in dynamic_position]

    n_static = len(static_position)
    n_dynamic = len(dynamic_position)
    if n_dynamic > 0 and n_static < n_dynamic:
        kDynamic = ir.ShapedType.get_dynamic_size()
        static_position = list(static_position) + [kDynamic] * (n_dynamic - n_static)

    return _vector.ExtractOp(
        _arith_ext.unwrap(vector),
        static_position=static_position,
        dynamic_position=dynamic_position,
    ).result


@dsl_loc_tracing
def load_op(result_type, memref, indices):
    """Wrapper around ``vector.LoadOp(...).result``."""
    from flydsl.expr import arith as _arith_ext

    return _vector.LoadOp(
        result_type,
        _arith_ext.unwrap(memref),
        [_arith_ext.unwrap(i, index=True) for i in indices],
    ).result


@dsl_loc_tracing
def bitcast(result_type, source):
    """Wrapper around ``vector.BitCastOp(...).result``."""
    from flydsl.expr import arith as _arith_ext

    return _vector.BitCastOp(
        result_type,
        _arith_ext.unwrap(source),
    ).result
