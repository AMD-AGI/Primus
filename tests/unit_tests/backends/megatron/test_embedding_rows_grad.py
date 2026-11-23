###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The row-wise word-embedding gradient against the dense gradient + the DDP hook's ``add_``."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels need a GPU")


def _ids(kind, n, vocab, gen):
    if kind == "uniform":
        return torch.randint(0, vocab, (n,), device="cuda", generator=gen)
    if kind == "constant":
        return torch.full((n,), vocab - 1, device="cuda", dtype=torch.long)
    # Zipf-like token ids, as natural text has: many repeats of a few ids
    exponent = {"zipf1.1": 1.1, "zipf1.5": 1.5}[kind]
    ranks = torch.arange(1, vocab + 1, device="cuda", dtype=torch.float64)
    probs = ranks.pow(-exponent)
    perm = torch.randperm(vocab, device="cuda", generator=gen)
    return perm[torch.multinomial(probs / probs.sum(), n, replacement=True, generator=gen)]


def _bits(t):
    return t.view({2: torch.int16, 4: torch.int32}[t.element_size()])


def _dense_reference(main, grad_output, ids, vocab):
    weight = torch.zeros(
        vocab, grad_output.shape[-1], dtype=torch.bfloat16, device="cuda", requires_grad=True
    )
    out = F.embedding(ids, weight)
    (dense,) = torch.autograd.grad(out, weight, grad_output)
    main.add_(dense)
    return main


@pytest.mark.parametrize("kind", ["uniform", "zipf1.1", "zipf1.5", "constant"])
@pytest.mark.parametrize(
    "n, vocab, hidden", [(4096, 20011, 384), (8192, 3000, 96), (100, 517, 1030), (3072, 9000, 64)]
)
@pytest.mark.parametrize("main_dtype", [torch.bfloat16, torch.float32])
def test_rows_add_bitwise(require_bitwise, kind, n, vocab, hidden, main_dtype):
    # the kernel path itself; the patched forward would run Megatron's after a failed check
    from primus.backends.megatron.core.fusions.embedding_rows_grad import (
        add_embedding_grad_,
    )

    gen = torch.Generator(device="cuda").manual_seed(n + vocab)
    ids = _ids(kind, n, vocab, gen).view(1, n)
    main0 = torch.zeros(vocab, hidden, device="cuda", dtype=main_dtype)
    got, want = main0.clone(), main0.clone()
    ok = True
    for _ in range(3):  # micro-batches accumulate into the same buffer
        g = torch.randn(1, n, hidden, device="cuda", generator=gen).bfloat16()
        g[0, 1::5, : hidden // 3] = -0.0
        add_embedding_grad_(got, g, ids)
        _dense_reference(want, g, ids, vocab)
        ok = ok and torch.equal(_bits(got), _bits(want))
        ids = _ids(kind, n, vocab, gen).view(1, n)
    require_bitwise(ok, f"embedding_rows_grad {kind} n={n} vocab={vocab} {main_dtype}")


def test_rows_add_is_stable_over_repeats(require_bitwise):
    from primus.backends.megatron.core.fusions.embedding_rows_grad import (
        add_embedding_grad_,
    )

    gen = torch.Generator(device="cuda").manual_seed(13)
    n, vocab, hidden = 4096, 30000, 1536
    ids = _ids("zipf1.1", n, vocab, gen).view(1, n)
    g = torch.randn(1, n, hidden, device="cuda", generator=gen).bfloat16()
    main0 = torch.randn(vocab, hidden, device="cuda", generator=gen).bfloat16()
    want = _dense_reference(main0.clone(), g, ids, vocab)
    ok = True
    for _ in range(30):
        got = main0.clone()
        add_embedding_grad_(got, g, ids)
        ok = ok and torch.equal(_bits(got), _bits(want))
    require_bitwise(ok, "embedding_rows_grad over 30 repeats")


def test_compact_ids_is_an_order_preserving_renumbering():
    from primus.backends.megatron.core.fusions.embedding_rows_grad import compact_ids

    gen = torch.Generator(device="cuda").manual_seed(5)
    for kind in ("uniform", "zipf1.1", "constant"):
        ids = _ids(kind, 777, 1000, gen).view(7, 111)
        inverse, unique, count = compact_ids(ids)
        c = int(count)
        assert c == ids.unique().numel() and unique.numel() == ids.numel()
        assert torch.equal(unique[:c], ids.unique(sorted=True)) and not unique[c:].any()
        assert torch.equal(unique[inverse], ids)


def test_self_check_passes(require_bitwise):
    from primus.backends.megatron.core.fusions import bitwise_check
    from primus.backends.megatron.core.fusions import embedding_rows_grad as erg

    saved = dict(bitwise_check._results)
    bitwise_check._results.clear()
    try:
        weight = torch.zeros(1000, 256, device="cuda", dtype=torch.bfloat16)
        ids = torch.zeros(4096, device="cuda", dtype=torch.long)
        for dtype in (torch.bfloat16, torch.float32):
            weight.main_grad = torch.zeros(1000, 256, device="cuda", dtype=dtype)
            require_bitwise(erg.checked(ids, weight), f"embedding_rows_grad self-check {dtype}")
        assert len(bitwise_check._results) == 2  # one verdict per (ids, width, dtypes)
    finally:
        bitwise_check._results.clear()
        bitwise_check._results.update(saved)


class _FakeEmbedding(torch.nn.Module):
    """What the patched forward reads of VocabParallelEmbedding, with Megatron's lookup."""

    def __init__(self, vocab, hidden, main_dtype=torch.bfloat16, tp=1):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(vocab, hidden, device="cuda").bfloat16())
        self.weight.main_grad = torch.zeros(vocab, hidden, device="cuda", dtype=main_dtype)
        self.weight.grad_added_to_main_grad = False
        self.tp_group = SimpleNamespace(size=lambda: tp)
        self.gtp_remat_size = 1
        self.deterministic_mode = False
        self.reduce_scatter_embeddings = False
        self.original_calls = 0

    def megatron_forward(self, input_):
        self.original_calls += 1
        return F.embedding(input_, self.weight)


def _ddp_hook(param):
    """Megatron DDP's backward hook for one parameter."""
    if param.grad is not None and not param.grad_added_to_main_grad:
        param.main_grad.add_(param.grad.data)
    param.grad = None


def _step(module, forward, ids, grad_outputs):
    for g in grad_outputs:
        out = forward(module, ids)
        out.backward(g)
        _ddp_hook(module.weight)
    return module.weight.main_grad


@pytest.mark.parametrize("main_dtype", [torch.bfloat16, torch.float32])
def test_patched_forward_matches_megatron(require_bitwise, main_dtype):
    from primus.backends.megatron.core.fusions import bitwise_check
    from primus.backends.megatron.patches.embedding_rows_grad_patches import (
        _UNTIED,
        make_forward,
    )

    saved = dict(bitwise_check._results)
    bitwise_check._results.clear()
    try:
        gen = torch.Generator(device="cuda").manual_seed(9)
        vocab, hidden, n = 5000, 128, 2048
        ids = _ids("zipf1.1", n, vocab, gen).view(1, n)
        grads = [torch.randn(1, n, hidden, device="cuda", generator=gen).bfloat16() for _ in range(4)]
        patched = make_forward(_FakeEmbedding.megatron_forward)

        ref = _FakeEmbedding(vocab, hidden, main_dtype)
        want = _step(ref, _FakeEmbedding.megatron_forward, ids, grads)
        new = _FakeEmbedding(vocab, hidden, main_dtype)
        new.weight.data.copy_(ref.weight.data)
        setattr(new, _UNTIED, True)
        out_new, out_ref = patched(new, ids), _FakeEmbedding.megatron_forward(ref, ids)
        assert torch.equal(_bits(out_new), _bits(out_ref))
        got = _step(new, patched, ids, grads)
        assert torch.equal(_bits(got), _bits(want))  # either path gives the dense path's bits
        require_bitwise(all(bitwise_check._results.values()), "embedding_rows_grad")
        assert new.original_calls == 0 and new.weight.grad_added_to_main_grad is True
    finally:
        bitwise_check._results.clear()
        bitwise_check._results.update(saved)


@pytest.mark.parametrize("order", ["rows_then_dense", "dense_then_rows", "rows_dense_rows"])
def test_route_may_change_within_a_step(require_bitwise, order):
    """Megatron resets grad_added_to_main_grad once per step: a micro-batch that falls back to
    the dense forward after a row-wise one must still get its gradient added."""
    from primus.backends.megatron.core.fusions import bitwise_check
    from primus.backends.megatron.patches.embedding_rows_grad_patches import (
        _UNTIED,
        make_forward,
    )

    saved = dict(bitwise_check._results)
    bitwise_check._results.clear()
    try:
        gen = torch.Generator(device="cuda").manual_seed(19)
        vocab, hidden, n = 3000, 96, 1024
        routes = {"rows_then_dense": "rd", "dense_then_rows": "dr", "rows_dense_rows": "rdr"}[order]
        ids = [_ids("zipf1.1", n, vocab, gen).view(1, n) for _ in routes]
        grads = [torch.randn(1, n, hidden, device="cuda", generator=gen).bfloat16() for _ in routes]
        patched = make_forward(_FakeEmbedding.megatron_forward)

        ref = _FakeEmbedding(vocab, hidden)
        new = _FakeEmbedding(vocab, hidden)
        new.weight.data.copy_(ref.weight.data)
        setattr(new, _UNTIED, True)
        for route, i, g in zip(routes, ids, grads):
            _step(ref, _FakeEmbedding.megatron_forward, i, [g])
            new.deterministic_mode = route == "d"  # one of the cases that keep Megatron's forward
            _step(new, patched, i, [g])
        assert new.original_calls == routes.count("d")
        assert torch.equal(_bits(new.weight.main_grad), _bits(ref.weight.main_grad))
        require_bitwise(all(bitwise_check._results.values()), "embedding_rows_grad")
    finally:
        bitwise_check._results.clear()
        bitwise_check._results.update(saved)


def test_backward_refuses_a_gradient_it_was_not_checked_for():
    from primus.backends.megatron.core.fusions.embedding_rows_grad import (
        _RowsGradEmbedding,
    )

    weight = torch.nn.Parameter(torch.zeros(50, 16, device="cuda", dtype=torch.bfloat16))
    weight.main_grad = torch.zeros(50, 16, device="cuda", dtype=torch.bfloat16)
    weight.grad_added_to_main_grad = False
    ids = torch.arange(8, device="cuda").view(1, 8)
    ctx = SimpleNamespace(saved_tensors=(ids,), weight=weight)
    with pytest.raises(RuntimeError, match="embedding_rows_grad"):
        _RowsGradEmbedding.backward(ctx, torch.zeros(1, 8, 16, device="cuda"))  # FP32 gradient
    with pytest.raises(RuntimeError, match="embedding_rows_grad"):
        _RowsGradEmbedding.backward(ctx, torch.zeros(8, 16, device="cuda", dtype=torch.bfloat16))
    assert not weight.main_grad.any() and weight.grad_added_to_main_grad is False


def test_patched_forward_keeps_megatron_where_it_must():
    from primus.backends.megatron.patches.embedding_rows_grad_patches import (
        _UNTIED,
        make_forward,
    )

    patched = make_forward(_FakeEmbedding.megatron_forward)
    ids = torch.arange(64, device="cuda").view(1, 64)

    def uses_original(module, **kwargs):
        before = module.original_calls
        with kwargs.get("ctx", torch.enable_grad()):
            patched(module, ids)
        return module.original_calls == before + 1

    tied = _FakeEmbedding(100, 32)  # not marked: the weight may feed an output layer too
    assert uses_original(tied)
    tp = _FakeEmbedding(100, 32, tp=2)
    setattr(tp, _UNTIED, True)
    assert uses_original(tp)
    for attr, value in (
        ("deterministic_mode", True),
        ("reduce_scatter_embeddings", True),
        ("gtp_remat_size", 2),
    ):
        m = _FakeEmbedding(100, 32)
        setattr(m, _UNTIED, True)
        setattr(m, attr, value)
        assert uses_original(m)
    hooked = _FakeEmbedding(100, 32)
    setattr(hooked, _UNTIED, True)
    hooked.register_forward_pre_hook(lambda mod, args: None)
    assert uses_original(hooked)
    no_buffer = _FakeEmbedding(100, 32)
    setattr(no_buffer, _UNTIED, True)
    del no_buffer.weight.main_grad
    assert uses_original(no_buffer)
    frozen = _FakeEmbedding(100, 32)
    setattr(frozen, _UNTIED, True)
    assert uses_original(frozen, ctx=torch.no_grad())
    frozen.weight.requires_grad_(False)
    assert uses_original(frozen)


def test_patched_forward_keeps_megatron_when_the_self_check_fails(monkeypatch):
    from primus.backends.megatron.core.fusions import bitwise_check
    from primus.backends.megatron.patches.embedding_rows_grad_patches import (
        _UNTIED,
        make_forward,
    )

    saved = dict(bitwise_check._results)
    bitwise_check._results.clear()
    try:
        monkeypatch.setattr(bitwise_check, "same_bits", lambda a, b: False)
        patched = make_forward(_FakeEmbedding.megatron_forward)
        module = _FakeEmbedding(300, 64)
        setattr(module, _UNTIED, True)
        ids = torch.arange(128, device="cuda").view(1, 128)
        out = patched(module, ids)
        out.sum().backward()
        assert module.original_calls == 1 and module.weight.grad is not None
        assert module.weight.grad_added_to_main_grad is False
        assert list(bitwise_check._results.values()) == [False]
    finally:
        bitwise_check._results.clear()
        bitwise_check._results.update(saved)


def test_only_untied_gpt_embeddings_are_marked():
    from primus.backends.megatron.patches.embedding_rows_grad_patches import (
        _UNTIED,
        mark_untied,
    )

    def model(shared, mtp_process=False, mtp_num_layers=None):
        word = SimpleNamespace()
        return (
            SimpleNamespace(
                embedding=SimpleNamespace(word_embeddings=word),
                share_embeddings_and_output_weights=shared,
                mtp_process=mtp_process,
                config=SimpleNamespace(mtp_num_layers=mtp_num_layers),
            ),
            word,
        )

    m, word = model(False)
    mark_untied(m)
    assert getattr(word, _UNTIED) is True
    m, word = model(True)
    mark_untied(m)
    assert not hasattr(word, _UNTIED)
    # multi-token prediction looks the embedding up again: the rows would be added lookup by lookup
    for kwargs in ({"mtp_process": True, "mtp_num_layers": 1}, {"mtp_num_layers": 2}):
        m, word = model(False, **kwargs)
        mark_untied(m)
        assert not hasattr(word, _UNTIED)
    mark_untied(SimpleNamespace())  # no embedding on this stage / model class


def test_switch_defaults_to_off():
    import os

    import yaml

    root = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
    with open(os.path.join(root, "primus/configs/modules/megatron/primus_megatron_module.yaml")) as f:
        assert yaml.safe_load(f)["embedding_rows_grad"] is False
