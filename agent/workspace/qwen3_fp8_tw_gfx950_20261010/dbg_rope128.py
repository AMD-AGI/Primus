import sys

import torch

sys.path.insert(0, "/home/xiaompen/turbo-opt")
from tests.pytorch.ops.test_qk_rmsnorm_rope import _inputs, _reference  # noqa: E402

from primus_turbo.pytorch.ops.rope import fused_qkv_rmsnorm_rope  # noqa: E402

S, B, NG, NPG, D = (int(v) for v in sys.argv[1:6])
qkv, qg, kg, freqs, split = _inputs(S=S, B=B, NG=NG, NPG=NPG, seed=321, D=D)
qkv.requires_grad_()
ref_in = qkv.detach().clone().requires_grad_()
out = fused_qkv_rmsnorm_rope(qkv, qg, kg, freqs, split, 1e-5)
ref = _reference(ref_in, qg, kg, freqs, split, 1e-5)
grads = tuple(torch.randn_like(t) for t in out)
torch.autograd.backward(out, grads)
torch.autograd.backward(ref, grads)
g, r = qkv.grad.float(), ref_in.grad.float()
bad = ~torch.isclose(g, r, rtol=2e-2, atol=2e-2)
print("bad", bad.sum().item(), "of", bad.numel(), "nan", g.isnan().sum().item())
if bad.any():
    idx = bad.nonzero()
    print("seq range", idx[:, 0].min().item(), idx[:, 0].max().item())
    print("batch values", idx[:, 1].unique().tolist())
    print("group values", idx[:, 2].unique().tolist())
    col = idx[:, 3]
    print("heads in packed group", (col // D).unique().tolist())
    print("elem-in-head values", (col % D).unique().tolist()[:80])
    s0, b0, g0, c0 = idx[0].tolist()
    print("first bad", (s0, b0, g0, c0), "got", g[s0, b0, g0, c0 - 4 : c0 + 4].tolist())
    print("            ref", r[s0, b0, g0, c0 - 4 : c0 + 4].tolist())
    first = g.clone()
    qkv.grad = None
    torch.autograd.backward(fused_qkv_rmsnorm_rope(qkv, qg, kg, freqs, split, 1e-5), grads)
    g2 = qkv.grad.float()
    print("rerun identical:", torch.equal(first.nan_to_num(7.0), g2.nan_to_num(7.0)))
    bad2 = ~torch.isclose(g2, r, rtol=2e-2, atol=2e-2)
    print("bad sets identical:", torch.equal(bad, bad2))
