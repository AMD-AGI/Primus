###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Numerics of the FLUX kernel ops against the PyTorch code they replace.

WHAT THESE ARE DEFENDING:
  Each op replaces a stretch of a FLUX block that diffusers computes with several PyTorch ops: the
  MLP's up-projection + GELU and its bias gradient, the q/k RMSNorm + RoPE (with both backward
  kernels), the single-stream adaLN, and the dual-stream gated residual + second adaLN. A wrong
  index, stride or rounding in a kernel still produces plausible numbers, so every output and
  every gradient is compared with an FP32 computation of the replaced code, and must be as close to
  it as the same code computed in BF16 is (twice its distance, plus a small floor). Calls the
  kernels do not serve must take the PyTorch path and give its result.

The kernels are specialized to the training's layout (hidden size 3072, head dim 128, token counts
in multiples of 128), so the tests run a small batch of it. They need a gfx950 GPU and FlyDSL and
skip elsewhere.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("flydsl")
transformer_flux = pytest.importorskip("diffusers.models.transformers.transformer_flux")
embeddings = pytest.importorskip("diffusers.models.embeddings")


def _gfx950() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950")


pytestmark = pytest.mark.skipif(not _gfx950(), reason="the kernels are written for gfx950")

BF16, DEV = torch.bfloat16, "cuda"
B, S_TXT, S_IMG, C, H, D, MLP, EPS = 2, 256, 256, 3072, 24, 128, 12288, 1e-6


def _rel(a, ref):
    a, ref = a.double(), ref.double()
    return ((a - ref).abs().max() / ref.abs().max().clamp_min(1e-30)).item()


def _check(name, got, ref32, ref16):
    err, err16 = _rel(got, ref32), _rel(ref16, ref32)
    assert err <= 2 * err16 + 2e-3, f"{name}: {err:.2e} from FP32, the BF16 reference {err16:.2e}"


def _leaves(*tensors):
    return [t.detach().clone().requires_grad_() for t in tensors]


def _grads(outputs, upstream, inputs):
    torch.autograd.backward(outputs, upstream)
    return [t.grad for t in inputs]


def _randn(*shape, scale=1.0, gen):
    return (torch.randn(*shape, generator=gen, device=DEV) * scale).to(BF16)


@pytest.fixture
def gen():
    return torch.Generator(device=DEV).manual_seed(0)


class TestGeluLinear:
    @staticmethod
    def _reference(x, w, b, prefix, dtype):
        x, w, b = (t.to(dtype) for t in (x, w, b))
        y = torch.nn.functional.gelu(x @ w.t() + b, approximate="tanh")
        return y if prefix is None else torch.cat([prefix.to(dtype), y], dim=1)

    @pytest.mark.parametrize("bias_grad", [False, True])
    @pytest.mark.parametrize(
        "rows,width,kernel", [(2048, C, True), (2048, 0, True), (1024, C, False), (1000, C, False)]
    )
    def test_against_pytorch(self, rows, width, kernel, bias_grad, gen, monkeypatch):
        """Single-stream (prefix) and dual-stream (none) rows, and row counts the kernel does not serve."""
        from primus.backends.automodel.models.flux import gelu_flydsl, mlp_gelu

        assert gelu_flydsl.supported(rows, width, MLP) is kernel
        monkeypatch.setitem(mlp_gelu._state, "bias_grad", bias_grad)
        x, w, b = (
            _randn(rows, C, gen=gen),
            _randn(MLP, C, scale=C**-0.5, gen=gen),
            _randn(MLP, scale=0.1, gen=gen),
        )
        prefix = _randn(rows, width, gen=gen) if width else None
        g = _randn(rows, width + MLP, gen=gen)
        results = {}
        for dtype in (None, torch.float32, BF16):
            leaves = _leaves(x, w, b) + ([] if prefix is None else _leaves(prefix))
            pre = leaves[3] if prefix is not None else None
            if dtype is None:
                out = torch.ops.primus_flux.gelu_linear(*leaves[:3], pre)[0]
            else:
                out = self._reference(*leaves[:3], pre, dtype)
            results[dtype] = [out.detach()] + _grads(out, g.to(out.dtype), leaves)
        names = ["out", "x.grad", "w.grad", "b.grad", "prefix.grad"]
        for name, got, r32, r16 in zip(names, results[None], results[torch.float32], results[BF16]):
            _check(name, got, r32, r16)


class TestQkNormRope:
    @staticmethod
    def _tables(dtype):
        ids = torch.zeros(S_TXT + S_IMG, 3)
        side = int(S_IMG**0.5)
        ids[S_TXT:, 1] = torch.arange(S_IMG) // side
        ids[S_TXT:, 2] = torch.arange(S_IMG) % side
        cos, sin = transformer_flux.FluxPosEmbed(theta=10000, axes_dim=[16, 56, 56])(ids.to(DEV))
        return cos.to(dtype).contiguous(), sin.to(dtype).contiguous()

    @staticmethod
    def _reference(qkv, txt_qkv, weights, cos, sin, dtype):
        def rms(t, w):
            t32 = t.float()
            return (t32 * torch.rsqrt(t32.pow(2).mean(-1, keepdim=True) + EPS)).to(dtype) * w.to(dtype)

        w_q, w_k, w_tq, w_tk = weights
        q, k, v = (t.to(dtype).unflatten(-1, (-1, D)) for t in qkv.chunk(3, dim=-1))
        q, k = rms(q, w_q), rms(k, w_k)
        if txt_qkv is not None:
            tq, tk, tv = (t.to(dtype).unflatten(-1, (-1, D)) for t in txt_qkv.chunk(3, dim=-1))
            q, k, v = (
                torch.cat([rms(tq, w_tq), q], 1),
                torch.cat([rms(tk, w_tk), k], 1),
                torch.cat([tv, v], 1),
            )
        rope = (cos.float(), sin.float())
        q = embeddings.apply_rotary_emb(q, rope, sequence_dim=1).to(dtype)
        k = embeddings.apply_rotary_emb(k, rope, sequence_dim=1).to(dtype)
        return q, k, v

    @pytest.mark.parametrize("flydsl_bwd", [False, True])
    @pytest.mark.parametrize("dual", [False, True])
    def test_against_pytorch(self, dual, flydsl_bwd, gen, monkeypatch):
        from primus.backends.automodel.models.flux import (
            qk_norm_rope,
            qk_norm_rope_flydsl,
        )

        monkeypatch.setitem(qk_norm_rope._state, "flydsl_bwd", flydsl_bwd)
        seq = S_IMG if dual else S_TXT + S_IMG
        qkv = _randn(B, seq, 3 * H * D, gen=gen)
        txt_qkv = _randn(B, S_TXT, 3 * H * D, gen=gen) if dual else None
        weights = [(1 + 0.1 * torch.randn(D, generator=gen, device=DEV)).to(BF16) for _ in range(4)]
        # The FlyDSL backward serves BF16 tables, which FSDP2's mixed precision passes in training.
        cos, sin = self._tables(BF16 if flydsl_bwd else torch.float32)
        assert qk_norm_rope_flydsl.supported(qkv, txt_qkv, cos, H) is flydsl_bwd
        upstream = [_randn(B, S_TXT + S_IMG, H, D, gen=gen) for _ in range(3)]
        results = {}
        for dtype in (None, torch.float32, BF16):
            leaves = _leaves(qkv, *weights[:2]) + (_leaves(txt_qkv, *weights[2:]) if dual else [])
            if dual:
                a, wq, wk, t, wtq, wtk = leaves
            else:
                (a, wq, wk), t, wtq, wtk = leaves, None, None, None
            if dtype is None:
                outs = torch.ops.primus_flux.qk_norm_rope(a, t, wq, wk, wtq, wtk, cos, sin, EPS, H)[:3]
            else:
                outs = self._reference(a, t, (wq, wk, wtq, wtk), cos, sin, dtype)
            results[dtype] = [o.detach() for o in outs] + _grads(
                outs, [u.to(outs[0].dtype) for u in upstream], leaves
            )
        names = [
            "q",
            "k",
            "v",
            "qkv.grad",
            "w_q.grad",
            "w_k.grad",
            "txt_qkv.grad",
            "w_txt_q.grad",
            "w_txt_k.grad",
        ]
        for name, got, r32, r16 in zip(names, results[None], results[torch.float32], results[BF16]):
            _check(name, got, r32, r16)


class TestAdaLNSingle:
    @staticmethod
    def _reference(x_txt, x_img, scale, shift, dtype):
        x = torch.cat([x_txt, x_img], 1).to(dtype)
        y = torch.nn.functional.layer_norm(x.float(), (C,), eps=EPS).to(dtype)
        return (
            x_txt.to(dtype) * 1,
            x_img.to(dtype) * 1,
            y * (1 + scale.to(dtype)[:, None]) + shift.to(dtype)[:, None],
        )

    def test_against_pytorch(self, gen):
        from primus.backends.automodel.models.flux import adaln_flydsl

        x_txt, x_img = _randn(B, S_TXT, C, scale=2, gen=gen), _randn(B, S_IMG, C, scale=2, gen=gen)
        emb = _randn(B, 3 * C, scale=0.5, gen=gen)
        upstream = [
            _randn(B, S_TXT, C, gen=gen),
            _randn(B, S_IMG, C, gen=gen),
            _randn(B, S_TXT + S_IMG, C, gen=gen),
        ]
        assert adaln_flydsl.supported(x_txt, x_img, *reversed(emb.chunk(3, dim=1)[:2]))
        results = {}
        for dtype in (None, torch.float32, BF16):
            a, b, e = _leaves(x_txt, x_img, emb)
            shift, scale, _ = e.chunk(3, dim=1)
            if dtype is None:
                outs = torch.ops.primus_flux.adaln_single(a, b, scale, shift, EPS)[:3]
            else:
                outs = self._reference(a, b, scale, shift, dtype)
            results[dtype] = [o.detach() for o in outs] + _grads(
                outs, [u.to(outs[0].dtype) for u in upstream], [a, b, e]
            )
        names = ["r_txt", "r_img", "y", "x_txt.grad", "x_img.grad", "emb.grad"]
        for name, got, r32, r16 in zip(names, results[None], results[torch.float32], results[BF16]):
            _check(name, got, r32, r16)


class TestDualResNorm:
    @staticmethod
    def _reference(x, a, g, s, h, dtype):
        x1 = x.to(dtype) + g.to(dtype)[:, None] * a.to(dtype)
        y = torch.nn.functional.layer_norm(x1.float(), (C,), eps=EPS).to(dtype)
        return x1, y * (1 + s.to(dtype)[:, None]) + h.to(dtype)[:, None]

    def test_against_pytorch(self, gen):
        from primus.backends.automodel.models.flux import (  # noqa: F401  (registers the op)
            adaln_dual_flydsl,
        )

        streams = [
            (
                _randn(B, n, C, scale=2, gen=gen),
                _randn(B, n, C, scale=0.7, gen=gen),
                _randn(B, 6 * C, scale=0.5, gen=gen),
            )
            for n in (S_IMG, S_TXT)
        ]
        upstream = [_randn(B, n, C, gen=gen) for n in (S_IMG, S_IMG, S_TXT, S_TXT)]
        results = {}
        for dtype in (None, torch.float32, BF16):
            leaves = [t for stream in streams for t in _leaves(*stream)]
            vecs = []
            for x, a, emb in (leaves[:3], leaves[3:]):
                _, _, gate_msa, shift_mlp, scale_mlp, _ = emb.chunk(6, dim=1)
                vecs.append((x, a, gate_msa, scale_mlp, shift_mlp))
            if dtype is None:
                outs = torch.ops.primus_flux.dual_res_norm(*vecs[0], *vecs[1], EPS)[:4]
            else:
                outs = self._reference(*vecs[0], dtype) + self._reference(*vecs[1], dtype)
            results[dtype] = [o.detach() for o in outs] + _grads(
                outs, [u.to(outs[0].dtype) for u in upstream], leaves
            )
        names = [
            "x1_img",
            "y_img",
            "x1_txt",
            "y_txt",
            "x_img.grad",
            "a_img.grad",
            "emb_img.grad",
            "x_txt.grad",
            "a_txt.grad",
            "emb_txt.grad",
        ]
        for name, got, r32, r16 in zip(names, results[None], results[torch.float32], results[BF16]):
            _check(name, got, r32, r16)
