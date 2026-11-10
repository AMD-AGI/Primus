###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""MiniMax-H3's AdaLN modulation gather with FP32-accumulated backward (``models/minimax_h3/modulation.py``).

The forward must be unchanged; the backward must reduce the per-row gradients into the few
table rows more accurately than BF16 ``index_add`` does, and deterministically. The swapped-in
forwards must equal diffusers' own, and a change to diffusers' code must stop the swap.
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers.models.transformers.transformer_minimax_h3")
from diffusers.models.transformers import transformer_minimax_h3 as h3  # noqa: E402

from primus.backends.automodel.models.minimax_h3 import modulation  # noqa: E402

TINY = dict(
    num_attention_heads=2,
    attention_head_dim=96,
    hidden_size=64,
    num_layers=2,
    num_refiner_layers=1,
    ffn_dim=128,
    text_dim=48,
    freq_dim=32,
    time_embed_hidden_dim=64,
    time_embed_dim=32,
    rope_freq_dim=16,
)


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


@pytest.fixture
def restore_forwards():
    saved = {cls: cls.forward for cls in (h3.MiniMaxH3TransformerBlock, h3.MiniMaxH3AdaLayerNormOut)}
    yield
    for cls, fn in saved.items():
        cls.forward = fn


def _table_and_rows(dtype, rows=20000, table_rows=6, width=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    table = torch.randn(table_rows, width, generator=g).to(dtype).requires_grad_()
    index = torch.randint(0, table_rows, (rows,), generator=g)
    grad = torch.randn(rows, width, generator=g).to(dtype)
    return table, index, grad


class TestGatherRows:
    def test_the_forward_is_index_select(self):
        table, index, _ = _table_and_rows(torch.bfloat16)
        assert torch.equal(modulation.gather_rows(table, index), table.index_select(0, index))

    def test_fp32_tables_take_the_plain_path(self):
        table, index, grad = _table_and_rows(torch.float32)
        modulation.gather_rows(table, index).backward(grad)
        ours = table.grad.clone()
        table.grad = None
        table.index_select(0, index).backward(grad)
        assert torch.equal(ours, table.grad)

    def test_the_bf16_reduction_is_the_exact_sum_rounded_once(self):
        table, index, grad = _table_and_rows(torch.bfloat16)
        exact = torch.zeros(table.shape, dtype=torch.float64).index_add_(0, index, grad.double())
        modulation.gather_rows(table, index).backward(grad)
        # One BF16 rounding of the FP32-accumulated sum: at most half an ulp per element.
        assert ((table.grad.double() - exact).abs() <= exact.abs() * 2**-8 + 1e-6).all()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="the BF16 atomics drift is a GPU effect")
    def test_on_gpu_it_is_far_closer_than_stock_index_add(self):
        """THE POINT. On GPU, index_add accumulates in BF16 atomics; on CPU it already accumulates in FP32."""
        table, index, grad = _table_and_rows(torch.bfloat16, rows=40000)
        table = table.detach().cuda().requires_grad_()
        index, grad = index.cuda(), grad.cuda()
        exact = torch.zeros(table.shape, dtype=torch.float64, device="cuda").index_add_(0, index, grad.double())

        modulation.gather_rows(table, index).backward(grad)
        ours = table.grad.double()
        table.grad = None
        table.index_select(0, index).backward(grad)
        stock = table.grad.double()

        err = lambda x: ((x - exact).norm() / exact.norm()).item()  # noqa: E731
        assert err(ours) < 5e-3
        assert err(ours) * 5 < err(stock)

    def test_the_backward_is_deterministic(self):
        table, index, grad = _table_and_rows(torch.bfloat16)
        grads = []
        for _ in range(2):
            table.grad = None
            modulation.gather_rows(table, index).backward(grad)
            grads.append(table.grad.clone())
        assert torch.equal(grads[0], grads[1])

    def test_a_table_without_grad_takes_the_plain_path(self):
        table, index, _ = _table_and_rows(torch.bfloat16)
        table = table.detach()
        assert torch.equal(modulation.gather_rows(table, index), table.index_select(0, index))


class TestInstall:
    def test_the_fingerprints_match_the_installed_diffusers(self):
        """Fails when a diffusers bump changes either forward: the copies must be reviewed against it."""
        for name, expected in modulation.UPSTREAM_FINGERPRINTS.items():
            cls_name, fn = name.split(".")
            assert modulation.fingerprint(getattr(getattr(h3, cls_name), fn)) == expected, name

    def test_the_swapped_forwards_equal_diffusers_own(self, restore_forwards):
        """Same outputs, and in FP32 the same gradients (gather_rows takes the plain path there)."""
        torch.manual_seed(0)
        model = h3.MiniMaxH3Transformer3DModel(**TINY)
        kwargs = _tiny_inputs(model)

        def run():
            model.zero_grad(set_to_none=True)
            out = model(**kwargs, return_dict=False)
            (out[0].square().mean() + out[1].square().mean()).backward()
            return [o.detach() for o in out[:2]], {n: p.grad.clone() for n, p in model.named_parameters()}

        before, grads_before = run()
        assert modulation.install() is True
        after, grads_after = run()
        for a, b in zip(before, after):
            assert torch.equal(a, b)
        for n in grads_before:
            assert torch.allclose(grads_before[n], grads_after[n], rtol=0, atol=0), n

    def test_install_is_idempotent(self, restore_forwards):
        assert modulation.install() is True
        first = h3.MiniMaxH3TransformerBlock.forward
        assert modulation.install() is True
        assert h3.MiniMaxH3TransformerBlock.forward is first

    def test_a_changed_upstream_is_left_alone_and_reported(self, restore_forwards, monkeypatch, caplog):
        monkeypatch.setitem(modulation.UPSTREAM_FINGERPRINTS, "MiniMaxH3TransformerBlock.forward", "0" * 64)
        original = h3.MiniMaxH3TransformerBlock.forward
        with caplog.at_level("WARNING", logger=modulation.logger.name):
            assert modulation.install() is False
        assert h3.MiniMaxH3TransformerBlock.forward is original
        assert "changed since this patch was written" in caplog.text


def _tiny_inputs(model):
    from primus.backends.automodel.models.minimax_h3 import layout as L

    c = L.constants()
    frames, height, width, audio_len, text = 3, 4, 4, 3, 5
    g = torch.Generator().manual_seed(1)
    lay = L.build_layout(text, frames, height, width, audio_len, torch.device("cpu"))
    return {
        "hidden_states": L.patchify_video(torch.randn(1, 24, frames, height, width, generator=g), c.patch_size),
        "audio_hidden_states": L.pack_audio(torch.randn(1, 2, 32, audio_len, generator=g)),
        "encoder_hidden_states": torch.randn(1, text, TINY["text_dim"], generator=g),
        "timestep": torch.tensor([0.4, 0.7]),
        **lay.model_kwargs(),
    }
