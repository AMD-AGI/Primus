###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Run the L2 op spec (``l2_inkling_ops.OPS``) under pytest on a GPU: every fused Inkling kernel
(relative-bias flash attention, causal_conv1d short conv with ``seq_idx``) against its eager
reference, forward and backward, at every shape and dtype the spec lists.

The reference runs on fp32 copies of the same inputs; the error is the max abs difference over
the max abs of the reference, per output and per gradient, checked against the spec's ``tol``."""

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("the fused Inkling kernels need a GPU", allow_module_level=True)

from . import l2_inkling_ops as spec  # noqa: E402


def _shapes(op):
    shapes = list(op.get("shapes", []))
    if "fixed" in op:
        shapes.insert(0, op["fixed"])
    return shapes


def _rel_err(a, b):
    a, b = a.float(), b.float()
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-12)).item()


def _cases():
    for op in spec.OPS:
        for dtype in op["dtypes"]:
            for shape in _shapes(op):
                key = "-".join(f"{k}{v}" for k, v in shape.items())
                yield pytest.param(op, dtype, shape, id=f"{op['name']}-{dtype}-{key}")


@pytest.mark.parametrize("op,dtype,shape", list(_cases()))
def test_kernel_matches_reference(op, dtype, shape):
    g = torch.Generator(device="cuda").manual_seed(1234)
    x = op["inputs"](shape, dtype, "cuda", g)
    grads = op.get("grad", [])
    k_in = {
        n: (v.detach().clone().requires_grad_(n in grads) if torch.is_tensor(v) else v) for n, v in x.items()
    }
    r_in = {
        n: (
            (v.detach().float() if v.is_floating_point() else v.detach()).clone().requires_grad_(n in grads)
            if torch.is_tensor(v)
            else v
        )
        for n, v in x.items()
    }
    out = op["kernel"](k_in)
    ref = op["reference"](r_in)
    tol = op["tol"][dtype]
    assert out.shape == ref.shape
    assert torch.isfinite(out.float()).all()
    assert _rel_err(out, ref) <= tol, f"forward: {_rel_err(out, ref):.3e} > {tol}"
    if grads:
        dy = torch.randn(ref.shape, generator=g, device="cuda")
        out.backward(dy.to(out.dtype))
        ref.backward(dy)
        for n in grads:
            err = _rel_err(k_in[n].grad, r_in[n].grad)
            assert err <= tol, f"d{n}: {err:.3e} > {tol}"
