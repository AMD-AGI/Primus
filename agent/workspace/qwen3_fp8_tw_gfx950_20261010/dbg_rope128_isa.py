"""Compile only the qk_rmsnorm_rope backward kernel (for an ISA dump)."""

import sys

import torch

sys.path.insert(0, "/home/xiaompen/turbo-opt")
import primus_turbo.flydsl.rope.qk_rmsnorm_rope_kernel as K  # noqa: E402
from tests.pytorch.ops.test_qk_rmsnorm_rope import _inputs  # noqa: E402

S, B, NG, NPG, D = (int(v) for v in sys.argv[1:6])
qkv, qg, kg, freqs, split = _inputs(S=S, B=B, NG=NG, NPG=NPG, seed=321, D=D)
q, k, v, qr, kr = K.flydsl_qkv_rmsnorm_rope_forward(qkv, qg, kg, freqs, split, 1e-5)
K._make_fold_kernel = None  # stop before the fold kernel compiles over the dump
try:
    K.flydsl_qkv_rmsnorm_rope_backward(
        torch.randn_like(q), torch.randn_like(k), torch.randn_like(v), qkv, qg, kg, freqs, qr, kr, split
    )
except TypeError:
    pass
torch.cuda.synchronize()
print("bwd compiled")
