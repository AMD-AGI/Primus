"""mxfp6_adaln_gemm_backend "aiter": AdaLNLinearFunction on aiter's AdaLN GEMM kernels vs the hipBLASLt path -- wgrad
(into main_grad) bitwise, forward / dgrad within bf16 GEMM tolerance, eager and under torch.compile(fullgraph=True)."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _setup(backend):
    from primus.backends.megatron.core.models.diffusion.common import mxfp6_gates
    from primus.backends.megatron.core.models.diffusion.common.mxfp6_gates import Mxfp6Gates

    g = Mxfp6Gates(adaln_wgrad_main_grad=True, adaln_gemm_backend=backend)
    g.validate()
    mxfp6_gates.reset(g)
    if backend == "aiter":
        from primus.backends.megatron.core.models.diffusion.common import normalization as N

        N.prime_adaln_aiter()


@pytest.mark.parametrize("n", [18432, 9216])
@pytest.mark.parametrize("compiled", [False, True])
def test_adaln_aiter_matches(n, compiled):
    from primus.backends.megatron.core.models.diffusion.common import normalization as N

    try:
        from aiter.ops.adaln_gemm import adaln_gemm_supported
    except ImportError:
        pytest.skip("no aiter adaln_gemm")
    if not adaln_gemm_supported("wgrad", n, 3072):
        pytest.skip("no AdaLN kernels for this shape")
    g = torch.Generator(device="cuda").manual_seed(n)
    x0 = torch.randn(32, 3072, device="cuda", dtype=torch.bfloat16, generator=g)
    w = torch.randn(n, 3072, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
    b = torch.randn(n, device="cuda", dtype=torch.bfloat16, generator=g)
    go = torch.randn(32, n, device="cuda", dtype=torch.bfloat16, generator=g)
    res = {}
    for backend in ("hipblaslt", "aiter"):
        _setup(backend)
        wp = torch.nn.Parameter(w.clone()); bp = torch.nn.Parameter(b.clone())
        wp.main_grad = torch.zeros_like(w); bp.main_grad = torch.zeros(n, device="cuda", dtype=torch.float32)
        x = x0.clone().requires_grad_(True)
        f = (lambda x_, w_, b_: N.AdaLNLinearFunction.apply(x_, w_, b_))
        if compiled:
            torch._dynamo.reset()
            f = torch.compile(f, fullgraph=True)
        y = f(x, wp, bp)
        y.backward(go)
        res[backend] = (y.detach(), x.grad, wp.main_grad.clone())
    (y0, dx0, dw0), (y1, dx1, dw1) = res["hipblaslt"], res["aiter"]
    assert torch.equal(dw0.view(torch.int16), dw1.view(torch.int16))
    torch.testing.assert_close(y1, y0, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(dx1, dx0, rtol=2e-2, atol=2e-2)
    from primus.backends.megatron.core.models.diffusion.common import mxfp6_gates

    mxfp6_gates.reset()
