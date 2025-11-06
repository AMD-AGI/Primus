
from dataclasses import dataclass
import jax.numpy as jnp

from MaxText.common_types import DType
from MaxText.layers.quantizations import Fp8Einsum, NANOOFp8Quantization


@dataclass
class PrimusNANOOFp8Quantization(NANOOFp8Quantization):
    def einsum(self, dtype: DType = jnp.float32):
        return Fp8Einsum(dtype=dtype, e4m3_dtype=jnp.float8_e4m3fnuz, e5m2_dtype=jnp.float8_e5m2fnuz)
