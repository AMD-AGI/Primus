###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Inkling multi-token prediction: the dense MLP, the per-depth block layout (global / sliding
with the MTP window), the MTP forward against a hand-written reference of SGLang's
``InklingMTPLayer`` (hidden-first concat, trunk embed_norm then MTP embed_norm, no final norm,
logits / mup), the MTP loss on the gradients, the converter mapping of ``model.mtp.*`` and the
sharded keys. Runs on CPU, except the two training-mode loss tests (GPU only)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip(
    "megatron.core.transformer.module", reason="megatron.core is not importable in this environment"
)
if not torch.distributed.is_available():
    pytest.skip("torch.distributed is not available", allow_module_level=True)

from types import SimpleNamespace  # noqa: E402

import torch.nn.functional as F  # noqa: E402

from primus.backends.megatron.checkpoint import (  # noqa: E402
    native_loader_inkling_mm_model as loader,
)
from primus.backends.megatron.core.models.inkling.inkling_model import (  # noqa: E402
    InklingModel,
)
from primus.backends.megatron.core.transformer.inkling.inkling_dense_mlp import (  # noqa: E402
    InklingDenseMLP,
)

from ._helpers import parallel, small_config  # noqa: E402,F401
from .test_inkling_model import (  # noqa: E402,F401
    _hf_tensors,
    _TensorStore,
    default_attention_env,
    no_te_offload_stream,
)

VOCAB = 96
needs_model = pytest.mark.usefixtures("no_te_offload_stream")
# A training-mode forward logs the MTP loss through Megatron's MTPLossLoggingHelper, which
# calls torch.cuda.current_device(); the eval-mode tests run on CPU.
needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="the MTP loss logging needs a GPU")
MTP = dict(
    mtp_num_layers=2,
    inkling_mtp_local_layer_ids=[1],
    inkling_mtp_local_extent=6,
    inkling_dense_intermediate_size=48,
)


def _model(**overrides):
    torch.manual_seed(0)
    return InklingModel(small_config(**{**MTP, **overrides}), vocab_size=VOCAB, max_sequence_length=64)


def _randomize(model):
    """Ones-initialized norms / scales would hide a swapped norm; give every parameter noise."""
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g, dtype=p.dtype))


def test_dense_mlp_matches_swiglu(parallel):  # noqa: F811
    cfg = small_config(**MTP)
    mlp = InklingDenseMLP(cfg, inkling_layer_idx=cfg.num_layers)
    with torch.no_grad():
        mlp.global_scale.fill_(1.7)
    x = torch.randn(5, 2, cfg.hidden_size)
    out, bias = mlp(x)
    assert bias is None and out.shape == x.shape
    gate, up = mlp.linear_fc1.weight.chunk(2, 0)
    ref = 1.7 * F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), mlp.linear_fc2.weight)
    torch.testing.assert_close(out, ref)


@needs_model
def test_depth_layout(parallel):  # noqa: F811
    model = _model()
    layers = [layer.mtp_model_layer for layer in model.mtp.layers]
    assert [layer.layer_number for layer in layers] == [1, 2]
    assert [layer.self_attention.is_sliding for layer in layers] == [False, True]
    assert layers[1].self_attention.window == 6
    assert all(isinstance(layer.mlp, InklingDenseMLP) for layer in layers)
    assert all(layer.mlp.linear_fc2.weight.shape == (64, 48) for layer in layers)
    # the sliding depth's relative-bias table spans its own window
    assert layers[0].self_attention.rel_proj.shape[-1] == model.config.inkling_rel_extent
    assert layers[1].self_attention.rel_proj.shape[-1] == 6
    # init seeds after the decoder's: the depths differ from each other and from the trunk
    assert not torch.equal(layers[0].self_attention.q_proj.weight, layers[1].self_attention.q_proj.weight)
    assert not torch.equal(
        layers[0].self_attention.q_proj.weight, model.decoder.layers[1].self_attention.q_proj.weight
    )


def _reference_logits(model, x):
    """SGLang's InklingMTPLayer chain, written out by hand with the model's modules."""
    cfg, mup = model.config, model.config.inkling_logits_mup_width_multiplier
    emb = model.embedding(input_ids=x, position_ids=None)
    h = model.embed_norm(emb)
    pre_norm = h
    for layer in model.decoder.layers:
        pre_norm, _ = layer(pre_norm, attention_mask=None)
    W = model.output_layer.weight
    out = [F.linear(model.decoder.final_layernorm(pre_norm) / mup, W)]
    ids, h = x, pre_norm
    for d, depth in enumerate(model.mtp.layers):
        ids = torch.roll(ids, -1, dims=-1)
        ids[:, -1] = 0  # Megatron's roll_tensor zero-fills the wrapped position
        e = depth.enorm(model.embed_norm(model.embedding(input_ids=ids, position_ids=None)))
        z = F.linear(torch.cat((depth.hnorm(h), e), -1), depth.eh_proj.weight)  # hidden first
        h, _ = depth.mtp_model_layer(z, attention_mask=None)
        out.append(F.linear(h / mup, W))  # chain_hidden_post_norm=False: no norm
    return out


@needs_model
def test_mtp_forward_matches_reference(parallel):  # noqa: F811
    model = _model()
    _randomize(model)
    model.eval()
    x = torch.randint(0, 90, (2, 12))
    with torch.no_grad():
        ref = _reference_logits(model, x)
        logits = model(x, None, None)
        torch.testing.assert_close(logits[..., :90], ref[0].transpose(0, 1)[..., :90])
        hidden = model.decoder(model._embed(x), attention_mask=None, extract_layer_indices={1})[1][0]
        mtp_hidden = model.mtp(
            input_ids=x,
            position_ids=torch.arange(12).expand(2, -1),
            hidden_states=hidden,
            attention_mask=None,
            embedding=model._embed,
        )
        chunks = mtp_hidden.chunk(3, 0)
        mup = model.config.inkling_logits_mup_width_multiplier
        for d in (1, 2):
            torch.testing.assert_close(F.linear(chunks[d] / mup, model.output_layer.weight), ref[d])


@needs_gpu
@needs_model
def test_mtp_loss_reaches_every_mtp_parameter(parallel):  # noqa: F811
    """Training: the returned loss is the trunk's; the MTP loss enters the backward pass through
    MTPLossAutoScaler and reaches every MTP parameter (and the shared embedding / unembed)."""
    from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler

    model = _model()
    model.train()
    x = torch.randint(0, 90, (2, 12))
    labels = torch.randint(0, 90, (2, 12))
    MTPLossAutoScaler.set_loss_scale(torch.tensor(1.0))
    loss = model(x, None, None, labels=labels, loss_mask=torch.ones(2, 12))
    assert loss.shape == (2, 12)
    loss.mean().backward()
    for name, p in model.mtp.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name
    # without labels the logits are the trunk's (the MTP chunks are dropped)
    model.zero_grad()
    model.eval()
    with torch.no_grad():
        assert model(x, None, None).shape == (2, 12, VOCAB)


@needs_gpu
@needs_model
def test_mtp_loss_scaling(parallel):  # noqa: F811
    """The MTP gradient scales with mtp_loss_scaling_factor (0 = trunk-only gradients)."""
    from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler

    x = torch.randint(0, 90, (1, 10))
    labels = torch.randint(0, 90, (1, 10))
    grads = []
    for factor in (0.0, 0.1):
        model = _model(mtp_loss_scaling_factor=factor)
        MTPLossAutoScaler.set_loss_scale(torch.tensor(1.0))
        model(x, None, None, labels=labels, loss_mask=torch.ones(1, 10)).sum().backward()
        grads.append(model.mtp.layers[0].eh_proj.weight.grad)
    assert grads[0] is None or grads[0].abs().sum() == 0
    assert grads[1].abs().sum() > 0


@needs_model
def test_sharded_keys(parallel):  # noqa: F811
    # checkpoint keys (``.key``; the dict keys keep the module path ``mtp_model_layer.``)
    keys = {v.key for v in _model().sharded_state_dict().values() if hasattr(v, "key")}
    for k in (
        "mtp.layers.0.enorm.weight",
        "mtp.layers.0.hnorm.weight",
        "mtp.layers.0.eh_proj.weight",
        "mtp.layers.0.transformer_layer.mlp.linear_fc1.weight",
        "mtp.layers.1.transformer_layer.self_attention.rel_proj",
        "mtp.layers.1.transformer_layer.mlp.global_scale",
    ):
        assert k in keys, k


def test_converter_mtp_fields():
    from .test_inkling_model import HF_TEXT

    cfg = {
        "text_config": HF_TEXT,
        "mtp_config": {
            "num_nextn_predict_layers": 2,
            "chain_hidden_post_norm": False,
            "local_layer_ids": [1],
        },
    }
    f = loader.derive_megatron_fields(cfg)
    assert f["mtp_num_layers"] == 2 and f["inkling_mtp_local_layer_ids"] == [1]
    assert f["inkling_mtp_local_extent"] is None and f["inkling_dense_intermediate_size"] == 3072
    assert loader.derive_megatron_fields({"text_config": HF_TEXT})["mtp_num_layers"] is None
    with pytest.raises(ValueError, match="chain_hidden_post_norm"):
        loader.derive_megatron_fields(
            {**cfg, "mtp_config": {"num_nextn_predict_layers": 2, "chain_hidden_post_norm": True}}
        )
    with pytest.raises(ValueError, match="local_layer_ids"):
        loader.derive_megatron_fields(
            {**cfg, "mtp_config": {"num_nextn_predict_layers": 2, "local_layer_ids": [2]}}
        )


def _hf_mtp_tensors(model, g):
    """HF-layout MTP tensors for ``model`` (w13 interleaved), and the expected gate / up."""
    msd, t, plan = model.state_dict(), {}, []
    K = model.config.inkling_sconv_kernel_size

    def rnd(*shape):
        return torch.randn(*shape, generator=g)

    for d in range(model.config.mtp_num_layers):
        h, m = f"{loader.MTP_PREFIX}{d}.", f"mtp.layers.{d}."
        b, ml = h + "transformer_block.", m + "mtp_model_layer."
        t[h + "embed_norm.weight"] = rnd(*msd[m + "enorm.weight"].shape)
        t[h + "hidden_norm.weight"] = rnd(*msd[m + "hnorm.weight"].shape)
        t[h + "input_proj.weight"] = rnd(*msd[m + "eh_proj.weight"].shape)
        t[b + "attn_norm.weight"] = rnd(*msd[ml + "input_layernorm.weight"].shape)
        t[b + "mlp_norm.weight"] = rnd(*msd[ml + "pre_mlp_layernorm.weight"].shape)
        for name in ("attn_sconv", "mlp_sconv"):
            t[b + name + ".weight"] = rnd(msd[ml + name + ".weight"].shape[0], 1, K)
        a, ma = b + "attn.", ml + "self_attention."
        for src, dst in (
            ("wq_du", "q_proj"),
            ("wk_dv", "k_proj"),
            ("wv_dv", "v_proj"),
            ("wo_ud", "o_proj"),
            ("wr_du", "r_proj"),
            ("q_norm", "q_norm"),
            ("k_norm", "k_norm"),
        ):
            t[a + src + ".weight"] = rnd(*msd[ma + dst + ".weight"].shape)
        for name in ("k_sconv", "v_sconv"):
            t[a + name + ".weight"] = rnd(msd[ma + name + ".weight"].shape[0], 1, K)
        t[a + "rel_logits_proj.proj"] = rnd(*msd[ma + "rel_proj"].shape)
        F_, H = model.config.inkling_dense_intermediate_size, model.config.hidden_size
        gate, up = rnd(F_, H), rnd(F_, H)
        t[b + "mlp.w13_dn.weight"] = torch.stack((gate, up), 1).flatten(0, 1)
        t[b + "mlp.w2_md.weight"] = rnd(H, F_)
        t[b + "mlp.global_scale"] = rnd(1)
        plan.append((gate, up))
    return t, plan


@needs_model
def test_converter_maps_every_mtp_tensor(parallel):  # noqa: F811
    model = _model()
    tensors, _ = _hf_tensors(model)
    mtp, plan = _hf_mtp_tensors(model, torch.Generator().manual_seed(3))
    tensors.update(mtp)
    margs = SimpleNamespace(num_layers=model.config.num_layers, params_dtype=torch.float32, mtp_num_layers=2)
    sd, consumed = loader.build_mcore_state_dict(
        model, _TensorStore(tensors), margs, {"inkling_unpadded_vocab_size": 90}, lambda *a: None
    )
    assert consumed == set(tensors)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert unexpected == [] and [k for k in missing if not k.endswith("_extra_state")] == []
    msd = model.state_dict()
    for d, (gate, up) in enumerate(plan):
        h, m = f"{loader.MTP_PREFIX}{d}.", f"mtp.layers.{d}."
        ml = m + "mtp_model_layer."
        torch.testing.assert_close(msd[ml + "mlp.linear_fc1.weight"], torch.cat((gate, up), 0))
        direct = {
            m + "enorm.weight": h + "embed_norm.weight",
            m + "hnorm.weight": h + "hidden_norm.weight",
            m + "eh_proj.weight": h + "input_proj.weight",
            ml + "mlp.linear_fc2.weight": h + "transformer_block.mlp.w2_md.weight",
            ml + "mlp.global_scale": h + "transformer_block.mlp.global_scale",
            ml + "self_attention.k_proj.weight": h + "transformer_block.attn.wk_dv.weight",
            ml + "self_attention.v_proj.weight": h + "transformer_block.attn.wv_dv.weight",
            ml + "self_attention.rel_proj": h + "transformer_block.attn.rel_logits_proj.proj",
        }
        for dst, src in direct.items():
            torch.testing.assert_close(msd[dst], tensors[src], msg=dst)
        torch.testing.assert_close(
            msd[ml + "attn_sconv.weight"], tensors[h + "transformer_block.attn_sconv.weight"].squeeze(1)
        )


@needs_model
def test_converter_refuses_missing_mtp_depths(parallel):  # noqa: F811
    model = _model()
    tensors, _ = _hf_tensors(model)
    margs = SimpleNamespace(num_layers=model.config.num_layers, params_dtype=torch.float32, mtp_num_layers=2)
    with pytest.raises(ValueError, match="MTP depths"):
        loader.build_mcore_state_dict(model, _TensorStore(tensors), margs, {}, lambda *a: None)


@needs_model
def test_converter_skips_mtp_without_mtp_model(parallel):  # noqa: F811
    model = _model(mtp_num_layers=None)
    tensors, _ = _hf_tensors(model)
    tensors.update(_hf_mtp_tensors(_model(), torch.Generator().manual_seed(3))[0])
    margs = SimpleNamespace(
        num_layers=model.config.num_layers, params_dtype=torch.float32, mtp_num_layers=None
    )
    logs = []
    _, consumed = loader.build_mcore_state_dict(model, _TensorStore(tensors), margs, {}, logs.append)
    assert consumed == set(tensors) and any("MTP tensors" in m for m in logs)
