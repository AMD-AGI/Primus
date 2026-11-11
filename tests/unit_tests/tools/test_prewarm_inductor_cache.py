import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from primus.backends.diffusion.models.flux import fused_norm_rope, model
from primus.backends.diffusion.models.flux.fused_norm_rope import RopeTables

SCRIPT = Path(__file__).resolve().parents[3] / "examples/mlperf/flux1/prewarm_inductor_cache.py"
spec = importlib.util.spec_from_file_location("prewarm_inductor_cache", SCRIPT)
prewarm_inductor_cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prewarm_inductor_cache)


@pytest.mark.parametrize("fused", [False, True])
def test_main_supplies_training_conditioning_and_rope_to_both_blocks(fused, monkeypatch):
    # Simulate the imports performed when the fusion flag is on at startup.
    monkeypatch.setattr(model, "rope_tables", fused_norm_rope.rope_tables, raising=False)
    monkeypatch.setattr(model, "RopeTables", RopeTables, raising=False)
    calls = []

    def check_inputs(vec, pe):
        assert vec.requires_grad
        assert isinstance(pe, RopeTables if fused else torch.Tensor)
        calls.append((vec, pe))

    def double_forward(img, txt, vec, pe):
        assert img.requires_grad and txt.requires_grad
        check_inputs(vec, pe)
        return img + vec, txt + vec

    def single_forward(x, vec, pe):
        assert x.requires_grad
        check_inputs(vec, pe)
        return x + vec

    double = Mock(side_effect=double_forward)
    single = Mock(side_effect=single_forward)
    double.to.return_value = double
    single.to.return_value = single
    built_model = SimpleNamespace(dit=SimpleNamespace(double_blocks=[double], single_blocks=[single]))
    pe = torch.zeros(1, 512, 1, 64, 2, 2)
    embedder = Mock(return_value=pe)
    embedder.to.return_value = embedder
    tables = Mock(wraps=prewarm_inductor_cache.build_rope_tables)
    monkeypatch.setattr(prewarm_inductor_cache, "build_rope_tables", tables)
    monkeypatch.setattr(prewarm_inductor_cache, "EmbedND", Mock(return_value=embedder))
    monkeypatch.setattr(prewarm_inductor_cache, "build_flux_model", Mock(return_value=built_model))
    monkeypatch.setattr(prewarm_inductor_cache, "set_attention_backend", Mock())
    monkeypatch.setattr(prewarm_inductor_cache.layers, "FUSED_NORM_ROPE", fused)
    # Retain real autograd without allocating the fixed GPU training shapes.
    monkeypatch.setattr(
        prewarm_inductor_cache,
        "torch",
        SimpleNamespace(
            manual_seed=Mock(),
            device=Mock(return_value="cpu"),
            bfloat16=torch.bfloat16,
            zeros=Mock(return_value=torch.zeros(1, 512, 3)),
            randn=lambda shape, **kwargs: torch.ones(2, requires_grad=kwargs.get("requires_grad", False)),
            cuda=SimpleNamespace(synchronize=Mock(), empty_cache=Mock()),
        ),
    )
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", "/unused/mock-cache")
    prewarm_inductor_cache.main()
    assert len(calls) == 2
    assert calls[0][0] is calls[1][0] and calls[0][1] is calls[1][1]
    if fused:
        tables.assert_called_once_with(pe, 256, 256)
        assert calls[0][1].txt_cos.shape[0] == 256
        assert calls[0][1].img_cos.shape[0] == 256
    else:
        tables.assert_not_called()
        assert calls[0][1] is pe
    double.compile.assert_called_once()
    single.compile.assert_called_once()
