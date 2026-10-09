"""mxfp6_adaln_gemm_backend "turbo": AdaLNLinearFunction on Primus-Turbo's GEMM dispatcher pinned to its AITER backend
vs the hipBLASLt path, eager and under torch.compile(fullgraph=True) -- wgrad (into main_grad) and dgrad bitwise, the
forward bitwise to torch.addmm, each GEMM on the AITER backend -- plus the gate's validation: "aiter" is a deprecated
alias, and a backend without shapes is a configuration error."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _setup(backend):
    from primus.backends.megatron.core.models.diffusion.common import mxfp6_gates
    from primus.backends.megatron.core.models.diffusion.common.mxfp6_gates import Mxfp6Gates

    g = Mxfp6Gates(adaln_wgrad_main_grad=True, adaln_gemm_backend=backend)
    g.validate()
    mxfp6_gates.reset(g)


def _table():
    try:
        from primus_turbo.pytorch.kernels.gemm.gemm_impl import GEMMAiterBackend
    except ImportError:
        return frozenset()
    return GEMMAiterBackend.shapes()


def _count_aiter_calls(monkeypatch):
    """Count the GEMMs the dispatcher runs on the AITER backend (its execute runs eagerly inside the custom op)."""
    from primus_turbo.pytorch.kernels.gemm.gemm_impl import GEMMAiterBackend

    calls = []
    execute = GEMMAiterBackend.execute

    def counted(*args, **kwargs):
        calls.append(kwargs.get("bias") is not None)
        return execute(*args, **kwargs)

    monkeypatch.setattr(GEMMAiterBackend, "execute", counted)
    return calls


def _bits(t):
    return t.view(torch.int16)


@pytest.fixture(autouse=True)
def _restore_gates():
    yield
    from primus.backends.megatron.core.models.diffusion.common import mxfp6_gates

    mxfp6_gates.reset()


@pytest.mark.parametrize("n", [18432, 9216])
@pytest.mark.parametrize("compiled", [False, True])
def test_adaln_turbo_matches(n, compiled, monkeypatch):
    from primus.backends.megatron.core.models.diffusion.common import normalization as N

    if any((p, n, 3072) not in _table() for p in ("fwd", "dgrad", "wgrad")):
        pytest.skip("no AdaLN kernels for this shape")
    g = torch.Generator(device="cuda").manual_seed(n)
    x0 = torch.randn(32, 3072, device="cuda", dtype=torch.bfloat16, generator=g)
    w = torch.randn(n, 3072, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
    b = torch.randn(n, device="cuda", dtype=torch.bfloat16, generator=g)
    go = torch.randn(32, n, device="cuda", dtype=torch.bfloat16, generator=g)
    res = {}
    calls = _count_aiter_calls(monkeypatch)
    for backend in ("hipblaslt", "turbo"):
        _setup(backend)
        wp = torch.nn.Parameter(w.clone())
        bp = torch.nn.Parameter(b.clone())
        wp.main_grad = torch.zeros_like(w)
        bp.main_grad = torch.zeros(n, device="cuda", dtype=torch.float32)
        x = x0.clone().requires_grad_(True)
        f = lambda x_, w_, b_: N.AdaLNLinearFunction.apply(x_, w_, b_)  # noqa: E731
        if compiled:
            torch._dynamo.reset()
            f = torch.compile(f, fullgraph=True)
        y = f(x, wp, bp)
        y.backward(go)
        res[backend] = (y.detach(), x.grad, wp.main_grad.clone(), bp.main_grad.clone())
        # fwd (with bias), dgrad, wgrad
        assert len(calls) == (3 if backend == "turbo" else 0), (backend, calls)
    (y0, dx0, dw0, db0), (y1, dx1, dw1, db1) = res["hipblaslt"], res["turbo"]
    assert torch.equal(_bits(dw1), _bits(dw0))
    assert torch.equal(_bits(dx1), _bits(dx0))
    assert torch.equal(db1, db0)
    assert torch.equal(_bits(y1), _bits(torch.addmm(b, x0, w.t())))
    # the hipBLASLt path's forward adds the bias separately (eager: a second rounding)
    torch.testing.assert_close(y1, y0, rtol=2e-2, atol=2e-2)


def test_adaln_turbo_off_its_shapes_stays_on_torch(monkeypatch):
    """A GEMM the AITER backend cannot handle (here M = 64) takes the hipBLASLt path, not the pinned backend."""
    from primus.backends.megatron.core.models.diffusion.common import normalization as N

    if not _table():
        pytest.skip("no Primus-Turbo AITER GEMM shapes")
    n = 9216
    g = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn(64, 3072, device="cuda", dtype=torch.bfloat16, generator=g).requires_grad_(True)
    wp = torch.nn.Parameter(torch.randn(n, 3072, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02)
    bp = torch.nn.Parameter(torch.randn(n, device="cuda", dtype=torch.bfloat16, generator=g))
    wp.main_grad = torch.zeros_like(wp)
    bp.main_grad = torch.zeros(n, device="cuda", dtype=torch.float32)
    calls = _count_aiter_calls(monkeypatch)
    _setup("turbo")
    y = N.AdaLNLinearFunction.apply(x, wp, bp)
    y.backward(torch.ones_like(y))
    assert calls == []
    assert torch.equal(y, torch.matmul(x.detach(), wp.detach().t()) + bp.detach())


def test_aiter_is_a_deprecated_alias():
    from primus.backends.megatron.core.models.diffusion.common.mxfp6_gates import Mxfp6Gates

    g = Mxfp6Gates(adaln_wgrad_main_grad=True, adaln_gemm_backend="aiter")
    with pytest.warns(FutureWarning, match="deprecated"):
        g.validate()
    assert g.adaln_gemm_backend == "turbo"


def test_backend_validation():
    from primus.backends.megatron.core.models.diffusion.common.mxfp6_gates import Mxfp6Gates

    with pytest.raises(ValueError, match="must be one of"):
        Mxfp6Gates(adaln_wgrad_main_grad=True, adaln_gemm_backend="cublas").validate()
    with pytest.raises(ValueError, match="enable it"):
        Mxfp6Gates(adaln_gemm_backend="turbo").validate()


def test_empty_table_is_a_configuration_error(monkeypatch):
    from primus.backends.megatron.core.models.diffusion.common import normalization as N

    monkeypatch.setattr(N, "prime_adaln_turbo", lambda: frozenset())
    with pytest.raises(RuntimeError, match="no shapes"):
        _setup("turbo")


def test_configure_logs_the_table(monkeypatch):
    from types import SimpleNamespace

    from primus.backends.megatron.core.models.diffusion.common import mxfp6_gates
    from primus.core.utils import module_utils

    if not _table():
        pytest.skip("no Primus-Turbo AITER GEMM shapes")
    lines = []
    monkeypatch.setattr(module_utils, "log_rank_0", lambda m, *a, **k: lines.append(m))
    cfg = SimpleNamespace(mxfp6_adaln_wgrad_main_grad=True, mxfp6_adaln_gemm_backend="turbo")
    assert mxfp6_gates.configure(cfg).adaln_gemm_backend == "turbo"
    assert any("adaln_gemm_shapes: [(" in m for m in lines), lines
