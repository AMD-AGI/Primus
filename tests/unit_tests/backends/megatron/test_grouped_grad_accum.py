###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Grouped gradient accumulation against the DDP hook's ``main_grad.add_(grad)`` loop."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels need a GPU")


def _bits(t):
    return t.view({2: torch.int16, 4: torch.int32}[t.element_size()])


@pytest.fixture
def clean_checks():
    from primus.backends.megatron.core.fusions import bitwise_check

    saved = dict(bitwise_check._results)
    bitwise_check._results.clear()
    yield bitwise_check
    bitwise_check._results.clear()
    bitwise_check._results.update(saved)


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("main_dtype", [torch.bfloat16, torch.float32])
def test_kernel_bitwise(require_bitwise, k, main_dtype):
    # the kernel itself; accumulate_ would run the add_ loop after a failed self-check
    from primus.backends.megatron.core.fusions.grouped_grad_accum import (
        _add_loop,
        _kernel,
        _probe,
    )

    main, grads = _probe(k, main_dtype, "cuda")
    got, want = main.clone(), main.clone()
    _kernel(got, grads)
    _add_loop(want, grads)
    nan = want.isnan()
    ok = torch.equal(got.isnan(), nan) and torch.equal(_bits(got)[~nan], _bits(want)[~nan])
    require_bitwise(ok, f"grouped_grad_accum k={k} {main_dtype}")


@pytest.mark.parametrize("scale", [1.0, 1.0e-3, 1.0e-38, 1.0e38])
@pytest.mark.parametrize("shape", [(6144, 512), (7,), (3, 5, 4099)])
def test_kernel_bitwise_on_gradient_like_values(require_bitwise, scale, shape):
    from primus.backends.megatron.core.fusions.grouped_grad_accum import (
        _add_loop,
        _kernel,
    )

    gen = torch.Generator(device="cuda").manual_seed(17)
    ok = True
    for k in (2, 4, 8):
        grads = [(torch.randn(shape, device="cuda", generator=gen) * scale).bfloat16() for _ in range(k)]
        got = torch.zeros(shape, device="cuda", dtype=torch.bfloat16)
        want = got.clone()
        for _ in range(2):  # a second group lands on the first one's sums
            _kernel(got, grads)
            _add_loop(want, grads)
        ok = ok and torch.equal(_bits(got), _bits(want))
    require_bitwise(ok, f"grouped_grad_accum scale={scale} shape={shape}")


def test_kernel_is_stable_over_repeats(require_bitwise):
    from primus.backends.megatron.core.fusions.grouped_grad_accum import (
        _add_loop,
        _kernel,
    )

    gen = torch.Generator(device="cuda").manual_seed(23)
    grads = [torch.randn(2_000_003, device="cuda", generator=gen).bfloat16() for _ in range(4)]
    want = torch.zeros(2_000_003, device="cuda", dtype=torch.bfloat16)
    _add_loop(want, grads)
    ok = True
    for _ in range(200):
        got = torch.zeros_like(want)
        _kernel(got, grads)
        ok = ok and torch.equal(_bits(got), _bits(want))
    require_bitwise(ok, "grouped_grad_accum over 200 repeats")


def test_accumulate_splits_long_lists_and_checks_each_count(require_bitwise, clean_checks):
    from primus.backends.megatron.core.fusions.grouped_grad_accum import accumulate_

    gen = torch.Generator(device="cuda").manual_seed(29)
    grads = [torch.randn(1000, 33, device="cuda", generator=gen).bfloat16() for _ in range(11)]
    got = torch.zeros(1000, 33, device="cuda", dtype=torch.bfloat16)
    want = got.clone()
    accumulate_(got, grads)
    for g in grads:
        want.add_(g)
    assert torch.equal(_bits(got), _bits(want))  # either path gives the loop's bits
    assert sorted(clean_checks._results) == [
        ("grouped_grad_accum", (3, "torch.bfloat16")),
        ("grouped_grad_accum", (8, "torch.bfloat16")),
    ]
    require_bitwise(all(clean_checks._results.values()), "grouped_grad_accum")


@pytest.mark.parametrize("limit", [4096, 5000, 100_000])
@pytest.mark.parametrize("main_dtype", [torch.bfloat16, torch.float32])
def test_large_tensors_take_several_launches(require_bitwise, monkeypatch, limit, main_dtype):
    """A buffer above ``MAX_ELEMS`` is accumulated slice by slice (every launch has an int32
    element count, the kernel variant the self-check ran); here with the limit lowered."""
    from primus.backends.megatron.core.fusions import grouped_grad_accum

    assert grouped_grad_accum.MAX_ELEMS < 2**31
    launches = []
    kernel = grouped_grad_accum._accum

    class Counting:
        def __getitem__(self, grid):
            launches.append(grid)
            return kernel[grid]

    monkeypatch.setattr(grouped_grad_accum, "MAX_ELEMS", limit)
    monkeypatch.setattr(grouped_grad_accum, "_accum", Counting())
    gen = torch.Generator(device="cuda").manual_seed(17)
    ok = True
    for shape in [(700, 300), (limit,), (limit + 1,), (3,)]:
        main = torch.randn(shape, device="cuda", generator=gen).to(main_dtype)
        grads = [torch.randn(shape, device="cuda", generator=gen).bfloat16() for _ in range(3)]
        want = main.clone()
        for g in grads:
            want.add_(g)
        launches.clear()
        grouped_grad_accum._kernel(main, grads)
        ok = ok and torch.equal(_bits(main), _bits(want))
        assert len(launches) == -(-main.numel() // limit)
    require_bitwise(ok, f"grouped_grad_accum in slices of {limit} into {main_dtype}")


def test_accumulate_keeps_add_for_other_layouts(clean_checks):
    from primus.backends.megatron.core.fusions.grouped_grad_accum import (
        accumulate_,
        can_fuse,
    )

    main = torch.zeros(64, 8, device="cuda", dtype=torch.bfloat16)
    g = torch.randn(64, 8, device="cuda").bfloat16()
    assert can_fuse(main, g)
    assert not can_fuse(main, g.float())  # FP32 gradient
    assert not can_fuse(main.double(), g)
    assert not can_fuse(main[:, ::2], g[:, ::2])
    assert not can_fuse(main, g[:32])
    assert not can_fuse(main.cpu(), g.cpu())
    g32 = torch.randn(64, 8, device="cuda")
    want = main.clone().add_(g32).add_(g32)
    accumulate_(main, [g32, g32])
    assert torch.equal(_bits(main), _bits(want)) and clean_checks._results == {}


def test_accumulate_keeps_add_when_the_self_check_fails(monkeypatch, clean_checks):
    from primus.backends.megatron.core.fusions import grouped_grad_accum as gga

    monkeypatch.setattr(gga, "_kernel", lambda main, grads: main.zero_())
    gen = torch.Generator(device="cuda").manual_seed(31)
    grads = [torch.randn(500, device="cuda", generator=gen).bfloat16() for _ in range(2)]
    got = torch.zeros(500, device="cuda", dtype=torch.bfloat16)
    want = got.clone().add_(grads[0]).add_(grads[1])
    gga.accumulate_(got, grads)
    assert torch.equal(_bits(got), _bits(want))
    assert clean_checks._results == {("grouped_grad_accum", (2, "torch.bfloat16")): False}


class _FakeDDP:
    """What the patched hook reads of DistributedDataParallel, with Megatron's hook body."""

    def __init__(self, params, overlap=False):
        self.ddp_config = SimpleNamespace(overlap_grad_reduce=overlap)
        group = SimpleNamespace(is_last_microbatch=True, registered=[])
        self.bucket_group = group
        self.param_to_bucket_group = {p: group for p in params}
        self.original_calls = 0

    def megatron_hook_factory(self, param):
        def hook(*unused):
            self.original_calls += 1
            if self.ddp_config.overlap_grad_reduce:
                assert param.grad is not None
            if param.grad is not None and (
                not param.grad_added_to_main_grad or getattr(param, "zero_out_wgrad", False)
            ):
                param.main_grad.add_(param.grad.data)
            param.grad = None
            if self.ddp_config.overlap_grad_reduce and self.bucket_group.is_last_microbatch:
                # register_grad_ready: the bucket's reduction may read main_grad from here on
                self.bucket_group.registered.append(param.main_grad.clone())

        return hook


def _param(n, gen, main_dtype=torch.bfloat16):
    p = torch.nn.Parameter(torch.randn(n, 7, device="cuda", generator=gen).bfloat16())
    p.main_grad = torch.zeros(n, 7, device="cuda", dtype=main_dtype)
    p.grad_added_to_main_grad = False
    return p


def _run_microbatches(ddp, factory, params, grads, flush):
    hooks = [factory(ddp, p) for p in params]
    for i, mb in enumerate(grads):
        # Megatron's no_sync: only the last micro-batch may start a reduction (overlap_grad_reduce)
        ddp.bucket_group.is_last_microbatch = not ddp.ddp_config.overlap_grad_reduce or i == len(grads) - 1
        for p, h, g in zip(params, hooks, mb):
            p.grad = g.clone()
            h()
            assert p.grad is None
    flush(ddp)
    return [p.main_grad.clone() for p in params]


@pytest.mark.parametrize("group", [2, 3, 4, 8])
@pytest.mark.parametrize("microbatches", [1, 2, 5, 8])
@pytest.mark.parametrize("main_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("overlap", [False, True])
def test_patched_hook_matches_megatron(group, microbatches, main_dtype, overlap):
    from primus.backends.megatron.patches.grouped_grad_accum_patches import (
        _PENDING,
        flush_pending,
        make_hook_factory,
    )

    def build():
        gen = torch.Generator(device="cuda").manual_seed(37)
        params = [_param(n, gen, main_dtype) for n in (5, 1000, 4097)]
        grads = [
            [torch.randn(p.shape, device="cuda", generator=gen).bfloat16() for p in params]
            for _ in range(microbatches)
        ]
        return params, grads

    params, grads = build()
    ref = _FakeDDP(params, overlap)
    want = _run_microbatches(ref, _FakeDDP.megatron_hook_factory, params, grads, lambda d: None)
    params, grads = build()
    new = _FakeDDP(params, overlap)
    factory = make_hook_factory(_FakeDDP.megatron_hook_factory, group, lambda: False)
    got = _run_microbatches(new, factory, params, grads, flush_pending)
    assert all(torch.equal(_bits(a), _bits(b)) for a, b in zip(got, want))
    # Megatron's hook ran for every gradient (its bookkeeping), and nothing stays held
    assert new.original_calls == ref.original_calls == 3 * microbatches
    assert not any(getattr(new, _PENDING).values())
    assert all(p.grad_added_to_main_grad is False for p in params)
    # under overlap_grad_reduce the buffer is complete when the parameter is registered
    assert len(new.bucket_group.registered) == len(ref.bucket_group.registered) == (3 if overlap else 0)
    for a, b in zip(new.bucket_group.registered, ref.bucket_group.registered):
        assert torch.equal(_bits(a), _bits(b))


def test_patched_hook_holds_at_most_group_minus_one():
    from primus.backends.megatron.patches.grouped_grad_accum_patches import (
        _PENDING,
        make_hook_factory,
    )

    gen = torch.Generator(device="cuda").manual_seed(41)
    p = _param(100, gen)
    ddp = _FakeDDP([p])
    hook = make_hook_factory(_FakeDDP.megatron_hook_factory, 3, lambda: False)(ddp, p)
    held = []
    for _ in range(7):
        p.grad = torch.randn(p.shape, device="cuda", generator=gen).bfloat16()
        hook()
        held.append(len(getattr(ddp, _PENDING).get(p, [])))
    assert held == [1, 2, 0, 1, 2, 0, 1]


def test_patched_hook_leaves_other_cases_to_megatron():
    from primus.backends.megatron.patches.grouped_grad_accum_patches import (
        _PENDING,
        make_hook_factory,
    )

    gen = torch.Generator(device="cuda").manual_seed(43)

    def grad(p):
        return torch.randn(p.shape, device="cuda", generator=gen).bfloat16()

    # a gradient already in main_grad (gradient accumulation fusion): held ones are added first
    p = _param(64, gen)
    ddp = _FakeDDP([p])
    hook = make_hook_factory(_FakeDDP.megatron_hook_factory, 4, lambda: False)(ddp, p)
    g1 = grad(p)
    p.grad = g1.clone()
    hook()
    assert len(getattr(ddp, _PENDING)[p]) == 1 and not p.main_grad.any()
    direct = grad(p)
    p.main_grad.copy_(direct)  # what a fused wgrad would have written... after g1 in eager order
    want = torch.zeros_like(p.main_grad).copy_(direct).add_(g1)
    p.grad_added_to_main_grad = True
    p.grad = grad(p)  # placeholder
    hook()
    assert ddp.original_calls == 2 and not getattr(ddp, _PENDING)[p]
    assert torch.equal(_bits(p.main_grad), _bits(want))

    # (torch refuses a param.grad of another dtype than the parameter's, so no FP32-gradient case)
    for case in ("capturing", "no_grad", "zero_out"):
        p = _param(64, gen)
        ddp = _FakeDDP([p])
        factory = make_hook_factory(_FakeDDP.megatron_hook_factory, 4, lambda: case == "capturing")
        hook = factory(ddp, p)
        g = grad(p)
        p.grad = None if case == "no_grad" else g.clone()
        if case == "zero_out":
            p.zero_out_wgrad = True
        if case == "no_grad":
            ddp.ddp_config.overlap_grad_reduce = False
        hook()
        assert ddp.original_calls == 1 and not getattr(ddp, _PENDING), case
        if case != "no_grad":
            assert torch.equal(_bits(p.main_grad), _bits(torch.zeros_like(p.main_grad).add_(g))), case


def test_flush_and_zero(clean_checks):
    from primus.backends.megatron.core.fusions.grouped_grad_accum import HELD
    from primus.backends.megatron.patches.grouped_grad_accum_patches import (
        _PENDING,
        drop_pending,
        flush_pending,
    )

    gen = torch.Generator(device="cuda").manual_seed(47)
    p = _param(50, gen)
    g = [torch.randn(p.shape, device="cuda", generator=gen).bfloat16() for _ in range(3)]
    ddp = SimpleNamespace()
    flush_pending(ddp)  # nothing held, nothing imported
    drop_pending(ddp)
    held = list(g)
    setattr(p, HELD, held)
    setattr(ddp, _PENDING, {p: held})
    flush_pending(ddp)
    want = torch.zeros_like(p.main_grad).add_(g[0]).add_(g[1]).add_(g[2])
    assert torch.equal(_bits(p.main_grad), _bits(want)) and not held
    held.extend(g)
    drop_pending(ddp)  # zero_grad_buffer: held gradients go with the sums
    assert not held and torch.equal(_bits(p.main_grad), _bits(want))


def test_hook_refuses_a_reused_gradient_buffer():
    from primus.backends.megatron.patches.grouped_grad_accum_patches import (
        make_hook_factory,
    )

    gen = torch.Generator(device="cuda").manual_seed(53)
    p = _param(64, gen)
    ddp = _FakeDDP([p])
    hook = make_hook_factory(_FakeDDP.megatron_hook_factory, 4, lambda: False)(ddp, p)
    workspace = torch.randn(p.shape, device="cuda", generator=gen).bfloat16()
    p.grad = workspace
    hook()
    workspace.normal_()  # a Function that writes its next gradient into the same buffer
    p.grad = workspace
    with pytest.raises(RuntimeError, match="reuses its"):
        hook()


def test_rows_grad_embedding_adds_after_the_held_dense_gradients(require_bitwise, clean_checks):
    """embedding_rows_grad after a dense-route micro-batch whose gradient is still held."""
    import torch.nn.functional as F

    from primus.backends.megatron.core.fusions import embedding_rows_grad as erg
    from primus.backends.megatron.patches.grouped_grad_accum_patches import (
        flush_pending,
        make_hook_factory,
    )

    gen = torch.Generator(device="cuda").manual_seed(59)
    vocab, hidden, n = 700, 48, 512
    ids = [torch.randint(0, vocab, (1, n), device="cuda", generator=gen) for _ in range(4)]
    grads = [torch.randn(1, n, hidden, device="cuda", generator=gen).bfloat16() for _ in range(4)]
    routes = "ddrd"  # dense, dense (both held at group 4), rows, dense

    def step(use_rows, wrap):
        w = torch.nn.Parameter(torch.randn(vocab, hidden, device="cuda", generator=gen).bfloat16())
        w.main_grad = torch.zeros(vocab, hidden, device="cuda", dtype=torch.bfloat16)
        w.grad_added_to_main_grad = False
        ddp = _FakeDDP([w])
        factory = make_hook_factory(_FakeDDP.megatron_hook_factory, 4, lambda: False) if wrap else None
        hook = factory(ddp, w) if wrap else _FakeDDP.megatron_hook_factory(ddp, w)
        for route, i, g in zip(routes, ids, grads):
            w.grad_added_to_main_grad = False if route == "d" else w.grad_added_to_main_grad
            out = erg.embedding(i, w) if (route == "r" and use_rows) else F.embedding(i, w)
            out.backward(g)
            hook()
        flush_pending(ddp)
        return w.main_grad

    want = step(False, False)
    got = step(True, True)
    require_bitwise(torch.equal(_bits(got), _bits(want)), "embedding_rows_grad with grad_accum_group")


@pytest.mark.parametrize("value, want", [(0, 0), (1, 0), (2, 2), (8, 8), (None, 0), ("x", 0), (-3, 0)])
def test_group_size(value, want):
    from primus.backends.megatron.patches.grouped_grad_accum_patches import group_size

    assert group_size(SimpleNamespace(grad_accum_group=value)) == want
    assert group_size(SimpleNamespace()) == 0


def test_switch_defaults_to_off():
    import os

    import yaml

    root = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
    with open(os.path.join(root, "primus/configs/modules/megatron/primus_megatron_module.yaml")) as f:
        assert yaml.safe_load(f)["grad_accum_group"] == 0
