###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""native_loader_llama.convert: the default (Llama) path and the ``family`` module check.
The heavy common.* steps (process-group init, model build, save) are faked; build_margs runs
Megatron's real argument parsing."""

import types
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("megatron.core")

from primus.backends.megatron.checkpoint import (  # noqa: E402
    native_convert_common as common,
)
from primus.backends.megatron.checkpoint import (  # noqa: E402
    native_loader_llama as llama,
)
from tests.unit_tests.backends.megatron.checkpoint.native_loader_test_utils import (  # noqa: E402
    llama_layout_hf_tensors,
    require_megatron_training,
    stub_convert_pipeline,
)

L, H, F, NH, KV, V = 2, 64, 128, 4, 2, 200
CFG = {
    "hidden_size": H,
    "intermediate_size": F,
    "num_hidden_layers": L,
    "num_attention_heads": NH,
    "num_key_value_heads": KV,
    "max_position_embeddings": 128,
    "rms_norm_eps": 1e-6,
    "rope_theta": 500000.0,
    "tie_word_embeddings": False,
    "vocab_size": V,
}


def test_llama_convert_default_path(tmp_path, monkeypatch, capsys):
    require_megatron_training()
    tensors = llama_layout_hf_tensors(L, H, F, NH, KV, V, lm_head=True)
    seen = {}

    def build_model(margs):
        seen["margs"] = margs
        sd = {
            "embedding.word_embeddings.weight": torch.zeros(256, H),
            "output_layer.weight": torch.zeros(256, H),
        }
        return SimpleNamespace(config=None, state_dict=lambda: sd)

    def save(model, sd, consumed, store, margs, save_dir, log):
        seen["consumed"], seen["save_dir"] = consumed, save_dir

    stub_convert_pipeline(monkeypatch, CFG, tensors, build_model, save)

    out = llama.convert(str(tmp_path), str(tmp_path / "out"), dtype="fp32", seq_length=128)
    assert out == str(tmp_path / "out") == seen["save_dir"]
    assert seen["consumed"] == set(tensors)
    margs = seen["margs"]
    assert margs.layernorm_epsilon == 1e-6 and margs.untie_embeddings_and_output_weights is True
    printed = capsys.readouterr().out
    assert "[llama-convert] arch: L=2" in printed and "layernorm_epsilon=1e-06" in printed


def test_convert_rejects_an_incomplete_family_module(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "ensure_megatron_on_path", lambda: pytest.fail("ran past the check"))
    family = types.ModuleType("half_family")
    family.LOG_NAME = "half"
    family.derive_megatron_fields = llama.derive_megatron_fields
    with pytest.raises(TypeError, match="extra_log_fields.*build_margs.*build_mcore_state_dict"):
        llama.convert(str(tmp_path), str(tmp_path / "out"), family=family)
