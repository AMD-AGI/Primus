###############################################################################
# Copyright 2023–2025 Google LLC. All rights reserved.
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from dataclasses import dataclass

import jax.numpy as jnp
from MaxText.common_types import DType
from MaxText.layers.quantizations import Fp8Einsum, NANOOFp8Quantization


@dataclass
class PrimusNANOOFp8Quantization(NANOOFp8Quantization):
    def einsum(self, dtype: DType = jnp.float32):
        return Fp8Einsum(dtype=dtype, e4m3_dtype=jnp.float8_e4m3fnuz, e5m2_dtype=jnp.float8_e5m2fnuz)
