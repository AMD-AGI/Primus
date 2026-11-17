###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Per-PR (not weekly) tests of the Granite training path. They need a GPU (the
runtime sets up the device; they skip on CPU). The GPU-free model-provider checks are in
``test_granite_model_provider.py``. The last builds and trains a small Granite model with Transformer Engine on one GPU.

``MegatronArgBuilder.update`` keeps only keys Megatron's argparse knows, so the Granite
keys ``residual_multiplier`` and ``model_type`` (which selects the model provider) are
dropped there; ``PrimusRuntime._initialize_trainer`` merges
the remaining module params back into the backend args (``merge_namespace``). The test runs
the real config loading, CLI overrides, ``MegatronAdapter.convert_config`` and that merge on
the Granite example recipe, then checks the args the trainer receives and the provider
``get_model_provider(args.model_type)`` returns for them. Only the trainer class is a stub.
"""

import argparse
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from primus.backends.megatron.megatron_adapter import MegatronAdapter
from primus.core.runtime.train_runtime import PrimusRuntime

# Module-wide: the fixtures and Primus' runtime need a GPU before a test body can skip.
pytestmark = pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="the Granite runtime tests need a GPU"
)

EXP = "examples/megatron/configs/MI325X/granite_4_1_3b-BF16-pretrain.yaml"


class _CaptureTrainer:
    """Stands in for MegatronPretrainTrainer: records the backend args it is given."""

    captured = None

    def __init__(self, backend_args=None, **kwargs):
        type(self).captured = backend_args


@pytest.fixture(autouse=True)
def _workspace(tmp_path, monkeypatch):
    # The recipe writes its experiment dirs under ${PRIMUS_WORKSPACE:./output}.
    monkeypatch.setenv("PRIMUS_WORKSPACE", str(tmp_path))


def _granite_backend_args(overrides):
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5))
    runtime = PrimusRuntime(args=argparse.Namespace(config=os.path.join(root, EXP), backend_path=None))
    runtime._initialize_configuration(module_name="pre_trainer", overrides=overrides)
    adapter = MegatronAdapter()
    runtime.ctx.adapter = adapter
    converted = []
    real_convert = adapter.convert_config

    def convert(params):
        converted.append(dict(vars(real_convert(params))))
        return SimpleNamespace(**converted[-1])

    _CaptureTrainer.captured = None
    with patch.object(runtime, "_run_phase_patches"), patch.object(adapter, "prepare_backend"), patch.object(
        adapter, "load_trainer_class", return_value=_CaptureTrainer
    ), patch.object(adapter, "convert_config", side_effect=convert):
        runtime._initialize_trainer()
    assert _CaptureTrainer.captured is not None, "the trainer was not built"
    # The MuP keys reach the converted Megatron args.
    assert len(converted) == 1
    assert converted[0]["use_mup"] is True and converted[0]["mup_embedding_mult"] == 12.0
    return _CaptureTrainer.captured


def test_granite_args_reach_the_trainer():
    args = _granite_backend_args(["mup_base_head_dim=1.6384", "residual_multiplier=0.5", "num_layers=4"])
    # Not a Megatron argparse key: dropped by MegatronArgBuilder.update, merged back by the runtime.
    assert args.residual_multiplier == 0.5  # CLI override wins over the preset's 0.22
    # Megatron keys (upstream MuP fields), from the preset and the CLI.
    assert args.model_type == "granite"
    assert args.use_mup is True
    assert args.mup_embedding_mult == 12.0
    assert args.mup_output_mult == 0.1
    assert args.mup_base_head_dim == pytest.approx(1.6384)
    assert args.mup_base_hidden_size is None
    assert args.num_layers == 4
    assert args.kv_channels == 64
    assert args.untie_embeddings_and_output_weights is False


def test_granite_args_route_to_upstream_model_provider_with_granite_builder(monkeypatch):
    import megatron.training
    import model_provider as upstream  # third_party/Megatron-LM/model_provider.py

    from primus.backends.megatron.core.models.granite import granite_builders
    from primus.core.utils import import_utils

    args = _granite_backend_args([])
    assert args.residual_multiplier == 0.22
    # The pretrain trainer passes args.model_type to get_model_provider.
    # Upstream model_provider is replaced by a recorder before get_model_provider binds it,
    # so the test calls the provider production returns, whatever its form (partial, closure).
    calls = []
    monkeypatch.setattr(
        upstream,
        "model_provider",
        lambda model_builder, *a, **kw: calls.append((model_builder, a, kw)) or "model",
    )
    provider = import_utils.get_model_provider(model_type=args.model_type)
    monkeypatch.setattr(megatron.training, "get_args", lambda: args)
    # Calling the provider calls upstream model_provider with the Granite builder.
    assert provider(True, True, vp_stage=0) == "model"
    assert len(calls) == 1 and calls[0][0] is granite_builders.granite_builder
    assert calls[0][2]["pre_process"] is True and calls[0][2]["vp_stage"] == 0

    # ModelOpt would swap in upstream's ModelOpt builder and drop residual_multiplier: refused.
    monkeypatch.setattr(
        megatron.training, "get_args", lambda: SimpleNamespace(**vars(args), modelopt_enabled=True)
    )
    with pytest.raises(ValueError, match="ModelOpt"):
        provider(True, True)
    assert len(calls) == 1


GRANITE_SMALL_HF = {
    "architectures": ["GraniteForCausalLM"],
    "model_type": "granite",
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000000.0,
    "vocab_size": 256,
    "tie_word_embeddings": True,
    "attention_bias": False,
    "mlp_bias": False,
    "embedding_multiplier": 12.0,
    "logits_scaling": 10.0,
    "residual_multiplier": 0.22,
    "attention_multiplier": 1 / 64,
}


def test_granite_recipe_overrides_reach_te_layers_and_train(init_parallel_state, monkeypatch):
    """Per PR, on a GPU: the recipe's Granite keys with CLI overrides (residual_multiplier
    0.5, mup_base_head_dim 1.6384 -> softmax_scale 0.02) go through get_model_provider into
    a model built with Transformer Engine. Checked on the built model, not on log lines:

    * every layer's core attention is TE's, and computing it on random q/k/v equals
      softmax(0.02 * q k^T) v (a TE path that ignored softmax_scale would use 1/8);
    * both bda of every layer scale by 0.5;
    * one forward/backward gives a finite loss and gradients.

    The model shape is small (from a tiny HF config through the converter's args); the
    multipliers and model_type are the recipe's, read through the real runtime path.
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Transformer Engine layers need a GPU")
    from megatron.core.extensions.transformer_engine import TEDotProductAttention
    from megatron.core.transformer.enums import AttnMaskType
    from megatron.training import global_vars

    from primus.backends.megatron.checkpoint import native_loader_granite as conv
    from primus.core.utils.import_utils import get_model_provider

    recipe = _granite_backend_args(["mup_base_head_dim=1.6384", "residual_multiplier=0.5"])
    hf_args = SimpleNamespace(
        hf_dir="/nonexistent",
        save_dir="/tmp/x",
        dtype="fp32",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        expert_parallel_size=1,
        seq_length=16,
        make_vocab_size_divisible_by=128,
    )
    margs = conv.build_margs(conv.derive_megatron_fields(GRANITE_SMALL_HF), hf_args)
    for key in (
        "model_type",
        "residual_multiplier",
        "use_mup",
        "mup_embedding_mult",
        "mup_output_mult",
        "mup_base_head_dim",
        "mup_base_hidden_size",
        "mup_attn_scale_power",
    ):
        setattr(margs, key, getattr(recipe, key))
    margs.padded_vocab_size = 256
    margs.params_dtype = torch.float32
    # The converter args skip weight init (weights are loaded after); this model trains.
    margs.perform_initialization = True
    assert margs.transformer_impl == "transformer_engine"
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", margs)
    # attention_backend "auto" requires these unset (the image may set them).
    for var in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
        monkeypatch.delenv(var, raising=False)

    model = get_model_provider(model_type=recipe.model_type)(pre_process=True, post_process=True).cuda()
    assert model.config.softmax_scale == pytest.approx(0.02)
    layers = list(model.decoder.layers)
    assert len(layers) == 2
    for layer in layers:
        assert layer.self_attn_bda.residual_multiplier == pytest.approx(0.5)
        assert layer.mlp_bda.residual_multiplier == pytest.approx(0.5)
        assert isinstance(layer.self_attention.core_attention, TEDotProductAttention)

    # TE's attention itself applies 0.02: compare with a plain softmax reference.
    torch.manual_seed(0)
    s, b, heads, groups, d = 8, 2, 4, 2, 64
    q = torch.randn(s, b, heads, d, device="cuda")
    k = torch.randn(s, b, groups, d, device="cuda")
    v = torch.randn(s, b, groups, d, device="cuda")
    model.eval()  # no attention dropout
    with torch.no_grad():
        got = layers[0].self_attention.core_attention(q, k, v, None, attn_mask_type=AttnMaskType.causal)
    kk = k.repeat_interleave(heads // groups, dim=2)
    vv = v.repeat_interleave(heads // groups, dim=2)
    scores = torch.einsum("sbhd,tbhd->bhst", q, kk) * 0.02
    scores = scores.masked_fill(torch.ones(s, s, dtype=torch.bool, device="cuda").triu(1), float("-inf"))
    want = torch.einsum("bhst,tbhd->sbhd", scores.softmax(-1), vv).reshape(s, b, heads * d)
    assert torch.allclose(got.float(), want, atol=1e-4), (got.float() - want).abs().max()
    wrong = torch.einsum("bhst,tbhd->sbhd", (scores / 0.02 / 8).softmax(-1), vv).reshape(s, b, heads * d)
    assert not torch.allclose(got.float(), wrong, atol=1e-3)  # 1/sqrt(d) would be told apart

    # One training step through the built model.
    model.train()
    ids = torch.randint(0, 256, (b, s), device="cuda")
    pos = torch.arange(s, device="cuda").unsqueeze(0).expand(b, s)
    loss = model(ids, pos, None, labels=ids).float().mean()
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
