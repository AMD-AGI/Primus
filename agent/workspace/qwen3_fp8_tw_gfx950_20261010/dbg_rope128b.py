import sys

import torch

sys.path.insert(0, "/home/xiaompen/turbo-opt")
import primus_turbo.flydsl.rope.qk_rmsnorm_rope_kernel as K  # noqa: E402
from tests.pytorch.ops.test_qk_rmsnorm_rope import _inputs  # noqa: E402

S, B, NG, NPG, D = (int(v) for v in sys.argv[1:6])
qkv, qg, kg, freqs, split = _inputs(S=S, B=B, NG=NG, NPG=NPG, seed=321, D=D)
q, k, v, qr, kr = K.flydsl_qkv_rmsnorm_rope_forward(qkv, qg, kg, freqs, split, 1e-5)

orig_empty_like = torch.empty_like
K.torch.empty_like = lambda t, **kw: torch.full_like(t, 123.0)
try:
    dqkv, _, _ = K.flydsl_qkv_rmsnorm_rope_backward(
        torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v), qkv, qg, kg, freqs, qr, kr, split
    )
finally:
    K.torch.empty_like = orig_empty_like
torch.cuda.synchronize()
g = dqkv.float()
bad = g != 0
print("nonzero", bad.sum().item(), "==123:", (g == 123).sum().item())
if bad.any():
    idx = bad.nonzero()
    col = idx[:, 3]
    print("heads", (col // D).unique().tolist(), "elems", (col % D).unique().tolist()[:40])
    vals = g[bad]
    print("sample values", vals[:8].tolist())
