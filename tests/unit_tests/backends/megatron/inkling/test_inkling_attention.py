###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Inkling attention pieces: the skewed relative bias, the chunked / windowed core
attention against a dense reference, the sliding-window edge, the attention mask and
short-conv causality."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip(
    "megatron.core.transformer.module", reason="megatron.core is not importable in this environment"
)
if not torch.distributed.is_available():
    pytest.skip("torch.distributed is not available", allow_module_level=True)

import torch.nn.functional as F  # noqa: E402

from primus.backends.megatron.core.transformer.inkling import (  # noqa: E402
    inkling_layers,
)
from primus.backends.megatron.core.transformer.inkling.inkling_layers import (  # noqa: E402
    InklingAttention,
    InklingRMSNorm,
    InklingShortConv,
    add_relative_bias_,
    relative_bias,
)

from ._helpers import parallel, small_config  # noqa: E402,F401


def _gather_bias(rel_logits, q_start, k_start, k_end):
    """Dense reference: bias[i, j] = rel_logits[i, q - k] for 0 <= q - k < E, else 0."""
    *lead, c, E = rel_logits.shape
    q = torch.arange(q_start, q_start + c)[:, None]
    k = torch.arange(k_start, k_end)[None]
    d = q - k
    ok = (d >= 0) & (d < E)
    idx = d.clamp(0, E - 1).expand(*lead, c, k.shape[-1])
    return torch.gather(rel_logits, -1, idx) * ok


@pytest.mark.parametrize(
    "E,q_start,c,k_start", [(5, 0, 7, 0), (5, 9, 4, 0), (5, 9, 4, 7), (16, 3, 6, 0), (3, 10, 5, 8)]
)
def test_relative_bias_matches_gather(E, q_start, c, k_start):
    torch.manual_seed(0)
    rel = torch.randn(2, 3, c, E)
    got = relative_bias(rel, q_start, k_start)
    want = _gather_bias(rel, q_start, k_start, q_start + c)
    torch.testing.assert_close(got, want)


@pytest.mark.parametrize(
    "E,q_start,c,k_start", [(5, 0, 7, 0), (5, 9, 4, 0), (5, 9, 4, 7), (16, 3, 6, 0), (3, 10, 5, 8)]
)
def test_add_relative_bias_in_place_matches_the_padded_bias(E, q_start, c, k_start):
    """The in-place add touches only the keys the table reaches; values and gradients equal
    adding the full (zero-padded) bias."""
    torch.manual_seed(0)
    rel = torch.randn(2, 3, c, E, requires_grad=True)
    base = torch.randn(2, 3, c, q_start + c - k_start, requires_grad=True)
    got = add_relative_bias_(base * 1.0, rel, q_start, k_start)
    want = base + _gather_bias(rel, q_start, k_start, q_start + c)
    torch.testing.assert_close(got, want)
    g = torch.randn_like(want)
    got_grads = torch.autograd.grad(got, (base, rel), g)
    want_grads = torch.autograd.grad(want, (base, rel), g)
    for a, b in zip(got_grads, want_grads):
        torch.testing.assert_close(a, b)


def _dense_reference(attn, q, k, v, rel, rel_proj, attention_mask=None):
    """Full [s, s] attention written independently of the chunked kernel."""
    cfg = attn.config
    b, n, s, d = q.shape
    rel_logits = rel @ rel_proj
    if not attn.is_sliding:
        pos = torch.arange(s).float()
        tau = 1 + cfg.inkling_log_scaling_alpha * torch.log(
            ((pos + 1) / cfg.inkling_log_scaling_n_floor).clamp(min=1)
        )
        q = q * tau.view(1, 1, -1, 1)
        rel_logits = rel_logits * tau.view(1, 1, -1, 1)
    rep = n // k.shape[1]
    k = k.repeat_interleave(rep, 1)
    v = v.repeat_interleave(rep, 1)
    scores = q @ k.transpose(2, 3) * attn.scaling + _gather_bias(rel_logits, 0, 0, s)
    dist = torch.arange(s)[:, None] - torch.arange(s)[None]
    masked = dist < 0
    if attn.window is not None:
        masked |= dist >= attn.window
    if attention_mask is not None:
        masked = masked | attention_mask
    scores = scores.masked_fill(masked, float("-inf"))
    out = F.softmax(scores, -1) @ v
    return out.permute(2, 0, 1, 3).reshape(s, b, n * d)


def _inputs(attn, s=21, b=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    n, kv = attn.num_attention_heads_per_partition, attn.num_query_groups_per_partition
    d, r = attn.hidden_size_per_attention_head, attn.d_rel
    q = torch.randn(b, n, s, d, generator=g)
    k = torch.randn(b, kv, s, d, generator=g)
    v = torch.randn(b, kv, s, d, generator=g)
    rel = torch.randn(b, n, s, r, generator=g)
    return q, k, v, rel


@pytest.mark.parametrize("sliding", [False, True])
@pytest.mark.parametrize("chunk", [1024, 5])
def test_core_attention_matches_dense(parallel, monkeypatch, sliding, chunk):  # noqa: F811
    monkeypatch.setattr(inkling_layers, "ATTN_QUERY_CHUNK", chunk)
    cfg = small_config()
    attn = InklingAttention(cfg, 1, 0 if sliding else 1, sliding).eval()
    q, k, v, rel = _inputs(attn)
    with torch.no_grad():
        attn.rel_proj.normal_()
        got = attn._core_attention(q, k, v, rel, attn.rel_proj)
        want = _dense_reference(attn, q, k, v, rel, attn.rel_proj)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_sliding_window_edge(parallel):  # noqa: F811
    """A key exactly ``window`` positions back is invisible; ``window - 1`` back is visible."""
    cfg = small_config()
    attn = InklingAttention(cfg, 1, 0, True).eval()
    w = attn.window
    s = w + 3
    q, k, v, rel = _inputs(attn, s=s, b=1)
    with torch.no_grad():
        base = attn._core_attention(q, k, v, rel, attn.rel_proj)
        for key, changes in ((2, False), (3, True)):  # last query is s-1; s-1-w = 2
            v2 = v.clone()
            v2[:, :, key] += 100.0
            out = attn._core_attention(q, k, v2, rel, attn.rel_proj)
            assert bool((out[-1] - base[-1]).abs().max() > 1e-3) == changes


def test_attention_mask_is_applied(parallel):  # noqa: F811
    cfg = small_config()
    attn = InklingAttention(cfg, 1, 1, False).eval()
    s = 9
    q, k, v, rel = _inputs(attn, s=s)
    mask = torch.zeros(1, 1, s, s, dtype=torch.bool)
    mask[..., 3] = True  # hide key 3 from every query
    mask[..., 3, 3] = False  # (query 3 keeps itself)
    with torch.no_grad():
        got = attn._core_attention(q, k, v, rel, attn.rel_proj, mask)
        want = _dense_reference(attn, q, k, v, rel, attn.rel_proj, mask)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_per_head_attention_mask_under_gqa(parallel):  # noqa: F811
    """A ``[b, heads, s, s]`` mask reaches the right query head when k / v are shared (GQA)."""
    cfg = small_config()
    attn = InklingAttention(cfg, 1, 1, False).eval()
    assert attn.num_attention_heads_per_partition > attn.num_query_groups_per_partition
    s = 9
    q, k, v, rel = _inputs(attn, s=s)
    b, n = q.shape[:2]
    g = torch.Generator().manual_seed(5)
    mask = torch.rand(b, n, s, s, generator=g) > 0.7
    mask &= ~torch.eye(s, dtype=torch.bool)  # every query keeps itself
    with torch.no_grad():
        got = attn._core_attention(q, k, v, rel, attn.rel_proj, mask)
        want = _dense_reference(attn, q, k, v, rel, attn.rel_proj, mask)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_attention_forward_and_norm_modules_run(parallel):  # noqa: F811
    """q_norm / k_norm are called as modules (their forward hooks fire)."""
    cfg = small_config()
    attn = InklingAttention(cfg, 1, 1, False)
    fired = []
    attn.q_norm.register_forward_pre_hook(lambda *_: fired.append("q"))
    attn.k_norm.register_forward_pre_hook(lambda *_: fired.append("k"))
    x = torch.randn(7, 2, cfg.hidden_size)
    out, bias = attn(x)
    assert out.shape == x.shape and bias is None
    assert fired == ["q", "k"]


def test_short_conv_is_causal_and_fp32():
    torch.manual_seed(0)
    conv = InklingShortConv(6, 4)
    with torch.no_grad():
        conv.weight.normal_()
    x = torch.randn(10, 2, 6, dtype=torch.bfloat16)
    y = conv(x)
    assert y.dtype == torch.bfloat16
    x2 = x.clone()
    x2[5] += 1.0
    y2 = conv(x2)
    assert torch.equal(y[:5], y2[:5])  # no output before position 5 depends on it
    assert not torch.equal(y[5:9], y2[5:9])
    # explicit reference: y[t] = x[t] + sum_j w[:, j] * x[t - (k - 1) + j]
    xf = x.float()
    ref = xf.clone()
    for t in range(10):
        for j in range(4):
            src = t - 3 + j
            if src >= 0:
                ref[t] += conv.weight[:, j].float() * xf[src]
    torch.testing.assert_close(y.float(), ref.to(torch.bfloat16).float(), rtol=1e-2, atol=1e-2)


def test_short_conv_starts_as_identity():
    conv = InklingShortConv(3, 4)
    x = torch.randn(5, 1, 3)
    torch.testing.assert_close(conv(x), x)


def _tp_attention_worker(rank, world, sliding, store_path, out_dir):
    """One TP rank (gloo): the TP = 1 module's weights, sharded as Megatron shards them."""
    from ._helpers import init_parallel

    torch.cuda.current_device = lambda: "cpu"  # gloo, CPU tensors (spawned process: the patch dies with it)
    init_parallel(store_path, rank=rank, world_size=world, tensor_parallel_size=world)
    ref = torch.load(f"{out_dir}/ref.pt", weights_only=False)
    attn = InklingAttention(small_config(tensor_model_parallel_size=world), 1, 0 if sliding else 1, sliding)
    with torch.no_grad():
        for name, p in attn.named_parameters():
            full = ref["params"][name]
            if name.startswith(("q_proj", "k_proj", "v_proj", "r_proj", "k_sconv", "v_sconv")):
                full = full.chunk(world, 0)[rank]  # column-parallel rows / this rank's kv channels
            elif name.startswith("o_proj"):
                full = full.chunk(world, 1)[rank]  # row-parallel input columns
            p.copy_(full)
    x = ref["x"].clone().requires_grad_()
    out, _ = attn.train()(x)
    (out * ref["g"]).sum().backward()
    grads = {n: p.grad for n, p in attn.named_parameters()}
    torch.save({"out": out.detach(), "x_grad": x.grad, "grads": grads}, f"{out_dir}/rank{rank}.pt")
    torch.distributed.destroy_process_group()


@pytest.mark.parametrize("sliding", [False, True])
def test_tensor_parallel_attention_matches_single_rank(parallel, tmp_path, sliding):  # noqa: F811
    """TP = 2 (two gloo processes, Megatron's column / row parallel projections) against TP = 1:
    the same output and input gradient; the sharded weights get the matching slices of the
    TP = 1 gradient, and the replicated per-head q / k norms and ``rel_proj`` (applied to each
    rank's heads only) get the full gradient, summed over TP (``_replicated_for_heads``)."""
    import torch.multiprocessing as mp

    if not torch.distributed.is_gloo_available():
        pytest.skip("gloo is not available")
    world = 2
    torch.manual_seed(0)
    attn = InklingAttention(small_config(), 1, 0 if sliding else 1, sliding).train()
    with torch.no_grad():
        for p in attn.parameters():  # non-trivial values everywhere (the convs start at zero)
            p.copy_(torch.randn_like(p) * 0.3 + (1.0 if p.dim() == 1 else 0.0))
    s, b = 11, 2
    x = torch.randn(s, b, attn.config.hidden_size, requires_grad=True)
    g = torch.randn(s, b, attn.config.hidden_size)
    params = {n: p.detach().clone() for n, p in attn.named_parameters()}
    torch.save({"params": params, "x": x.detach(), "g": g}, tmp_path / "ref.pt")
    out, _ = attn(x)
    (out * g).sum().backward()

    mp.spawn(
        _tp_attention_worker, args=(world, sliding, str(tmp_path / "store"), str(tmp_path)), nprocs=world
    )
    for rank in range(world):
        got = torch.load(tmp_path / f"rank{rank}.pt", weights_only=False)
        torch.testing.assert_close(got["out"], out.detach(), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(got["x_grad"], x.grad, rtol=1e-5, atol=1e-5)
        for name, p in attn.named_parameters():
            want = p.grad
            if name.startswith(("q_proj", "k_proj", "v_proj", "r_proj", "k_sconv", "v_sconv")):
                want = want.chunk(world, 0)[rank]
            elif name.startswith("o_proj"):
                want = want.chunk(world, 1)[rank]
            torch.testing.assert_close(got["grads"][name], want, rtol=1e-4, atol=1e-5, msg=name)


def _hf_rms_norm(x, weight, eps):
    """HF InklingRMSNorm: normalize in fp32, cast back, then scale."""
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return weight * h.to(x.dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rms_norm_matches_hf(dtype):
    """InklingRMSNorm (the shared fused_rms_norm with mid_cast) has HF's rounding."""
    torch.manual_seed(0)
    norm = InklingRMSNorm(64, eps=1e-6).to(dtype)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(64))
    x = torch.randn(5, 3, 64, dtype=dtype)
    out = norm(x)
    assert out.dtype == dtype
    torch.testing.assert_close(out, _hf_rms_norm(x, norm.weight, 1e-6), rtol=0, atol=0)


@pytest.mark.parametrize("shape", ["b11s", "b1s1"])
def test_broadcast_mask_over_query_chunks(parallel, monkeypatch, shape):  # noqa: F811
    """A mask with a size-1 query (``[b, 1, 1, s]`` padding) or key axis applies to every query
    chunk, not only the first."""
    monkeypatch.setattr(inkling_layers, "ATTN_QUERY_CHUNK", 4)
    cfg = small_config()
    attn = InklingAttention(cfg, 1, 1, False).eval()
    s = 11
    q, k, v, rel = _inputs(attn, s=s)
    b = q.shape[0]
    if shape == "b11s":
        mask = torch.zeros(b, 1, 1, s, dtype=torch.bool)
        mask[..., 2] = True  # key 2 hidden from every query (query 2 keeps nothing else? keeps 0, 1)
    else:
        mask = torch.zeros(b, 1, s, 1, dtype=torch.bool)  # no query fully masked
    with torch.no_grad():
        got = attn._core_attention(q, k, v, rel, attn.rel_proj, mask)
        want = _dense_reference(attn, q, k, v, rel, attn.rel_proj, mask.expand(b, 1, s, s))
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


_GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused (Triton) attention needs a GPU")


@_GPU
@pytest.mark.parametrize("sliding", [False, True])
@pytest.mark.parametrize("with_mask", [False, True])
def test_fused_attention_matches_eager_with_grads(parallel, sliding, with_mask):  # noqa: F811
    """inkling_attention_impl=fused gives the eager path's output and gradients (q, k, v, rel,
    rel_proj), with log scaling on the global layer, the window on the local one and a mask."""
    pytest.importorskip("triton")
    out, grads = {}, {}
    s = 70  # > the window (8) and not a multiple of the kernel's block
    for impl in ("eager", "fused"):
        cfg = small_config(inkling_attention_impl=impl)
        attn = InklingAttention(cfg, 1, 0 if sliding else 1, sliding).eval()
        torch.manual_seed(0)
        with torch.no_grad():
            attn.rel_proj.normal_()
        x = [t.cuda().requires_grad_() for t in _inputs(attn, s=s)]
        rel_proj = attn.rel_proj.detach().cuda().requires_grad_()
        mask = None
        if with_mask:
            mask = torch.zeros(2, 1, 1, s, dtype=torch.bool, device="cuda")
            mask[1, ..., -5:] = True
        y = attn._core_attention(*x, rel_proj, mask)
        g = torch.autograd.grad(y, x + [rel_proj], torch.ones_like(y))
        out[impl], grads[impl] = y.detach(), g
    torch.testing.assert_close(out["fused"], out["eager"], rtol=1e-4, atol=1e-4)
    for a, b in zip(grads["fused"], grads["eager"]):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)


def test_fused_attention_refuses_dropout():
    with pytest.raises(Exception, match="attention_dropout"):
        small_config(inkling_attention_impl="fused", attention_dropout=0.1)
