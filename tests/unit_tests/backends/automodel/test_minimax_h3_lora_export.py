###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""``tools/automodel/export_minimax_h3_lora.py``: AutoModel's PEFT-format LoRA to diffusers' format.

The round trip is the test: a PEFT-format adapter for a tiny real MiniMax-H3 transformer is
exported, then loaded back with diffusers' own ``load_lora_adapter``, and the factors and the
rank/alpha scaling must survive. AutoModel's file itself fails to load in diffusers, which is
why the tool exists.
"""
import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("diffusers.models.transformers.transformer_minimax_h3")
from diffusers.models.transformers.transformer_minimax_h3 import (  # noqa: E402
    MiniMaxH3Transformer3DModel,
)
from safetensors.torch import save_file  # noqa: E402

_TOOL = Path(__file__).resolve().parents[4] / "tools" / "automodel" / "export_minimax_h3_lora.py"
_spec = importlib.util.spec_from_file_location("export_minimax_h3_lora", _TOOL)
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)

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
RANK, ALPHA = 4, 8
SUFFIXES = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2")


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def model():
    torch.manual_seed(0)
    return MiniMaxH3Transformer3DModel(**TINY)


def automodel_adapter(tmp_path, base):
    """An adapter laid out as AutoModel saves it: PEFT keys, adapter_config.json beside the weights."""
    modules = {
        n: m for n, m in base.named_modules() if isinstance(m, torch.nn.Linear) and n.endswith(SUFFIXES)
    }
    g = torch.Generator().manual_seed(1)
    state = {}
    for name, linear in modules.items():
        state[f"base_model.model.{name}.lora_A.weight"] = torch.randn(RANK, linear.in_features, generator=g)
        state[f"base_model.model.{name}.lora_B.weight"] = torch.randn(linear.out_features, RANK, generator=g)
    save_file(state, str(tmp_path / "adapter_model.safetensors"))
    config = {
        "peft_type": "LORA",
        "r": RANK,
        "lora_alpha": ALPHA,
        "bias": "none",
        "target_modules": sorted(modules),
    }
    (tmp_path / "adapter_config.json").write_text(json.dumps(config))
    return state, sorted(modules)


def test_automodel_s_own_file_does_not_load_in_diffusers(tmp_path):
    """Why the tool exists: diffusers reads neither the file name nor the key prefix."""
    base = model()
    automodel_adapter(tmp_path, base)
    with pytest.raises(Exception):
        base.load_lora_adapter(str(tmp_path), prefix="transformer")


def test_the_export_loads_with_its_factors_and_scaling(tmp_path):
    base = model()
    (tmp_path / "ckpt").mkdir()
    state, targets = automodel_adapter(tmp_path / "ckpt", base)
    out = tmp_path / "out"
    assert tool.main(["--checkpoint", str(tmp_path / "ckpt"), "--out", str(out)]) == 0
    assert (out / "pytorch_lora_weights.safetensors").is_file()

    loaded = model()
    loaded.load_lora_adapter(
        str(out), weight_name="pytorch_lora_weights.safetensors", prefix="transformer", adapter_name="h3"
    )
    params = dict(loaded.named_parameters())
    for key, tensor in state.items():
        name = key.removeprefix("base_model.model.")
        module, factor = name.rsplit(".lora_", 1)
        loaded_key = f"{module}.lora_{factor.replace('.weight', '')}.h3.weight"
        assert torch.equal(params[loaded_key], tensor), loaded_key
    lora_modules = [
        m for n, m in loaded.named_modules() if hasattr(m, "scaling") and isinstance(m.scaling, dict)
    ]
    assert len(lora_modules) == len(targets)
    assert all(m.scaling["h3"] == pytest.approx(ALPHA / RANK) for m in lora_modules)


def test_the_export_changes_the_output_as_the_adapter_says(tmp_path):
    """W' = W + (alpha / r) B A on one targeted linear, checked through diffusers' loaded model."""
    base = model()
    (tmp_path / "ckpt").mkdir()
    state, targets = automodel_adapter(tmp_path / "ckpt", base)
    tool.main(["--checkpoint", str(tmp_path / "ckpt"), "--out", str(tmp_path / "out")])
    loaded = model()
    loaded.load_lora_adapter(
        str(tmp_path / "out"),
        weight_name="pytorch_lora_weights.safetensors",
        prefix="transformer",
        adapter_name="h3",
    )
    name = targets[0]
    linear = dict(loaded.named_modules())[name]
    x = torch.randn(3, linear.in_features)
    a = state[f"base_model.model.{name}.lora_A.weight"]
    b = state[f"base_model.model.{name}.lora_B.weight"]
    plain = dict(base.named_modules())[name]
    expected = plain(x) + (ALPHA / RANK) * (x @ a.T @ b.T)
    assert torch.allclose(linear(x), expected, atol=1e-5)


def test_a_key_that_is_not_a_lora_factor_is_refused():
    with pytest.raises(ValueError, match="not a LoRA factor"):
        tool.convert({"base_model.model.x.weight": torch.zeros(1)}, {"r": 4, "target_modules": ["x"]})


def test_a_factor_of_an_untargeted_module_is_refused():
    with pytest.raises(ValueError, match="does not target"):
        tool.convert({"base_model.model.y.lora_A.weight": torch.zeros(1)}, {"r": 4, "target_modules": ["x"]})


def test_a_missing_adapter_is_reported(tmp_path, capsys):
    assert tool.main(["--checkpoint", str(tmp_path), "--out", str(tmp_path / "out")]) == 2
    assert "adapter_model.safetensors" in capsys.readouterr().err
