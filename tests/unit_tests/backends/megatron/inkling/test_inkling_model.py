###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""InklingModel end to end: 1-based layer numbers and distributed-checkpoint keys, a
dist-ckpt save / load round trip, padded-vocab masking, config validation and the HF
config translation of the converter.

The model tests run on CPU too: without a GPU the ``no_te_offload_stream`` fixture switches
TransformerBlock to its no-offload path (TE's offload context would create a CUDA stream)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip(
    "megatron.core.transformer.module", reason="megatron.core is not importable in this environment"
)
if not torch.distributed.is_available():
    pytest.skip("torch.distributed is not available", allow_module_level=True)

import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import torch.nn.functional as F  # noqa: E402

from primus.backends.megatron.checkpoint import (  # noqa: E402
    native_loader_inkling_mm_model as loader,
)
from primus.backends.megatron.core.models.inkling.inkling_model import (  # noqa: E402
    InklingModel,
)

from ._helpers import parallel, small_config  # noqa: E402,F401

VOCAB = 96
needs_model = pytest.mark.usefixtures("no_te_offload_stream")


@pytest.fixture(autouse=True)
def default_attention_env(monkeypatch):
    """LanguageModule checks (and then sets) the NVTE_* attention-backend variables against
    ``attention_backend`` (auto); a CI image may preset them. Start each test without them
    (monkeypatch restores the image's values afterwards)."""
    for name in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def no_te_offload_stream(monkeypatch):
    """With Transformer Engine installed, TransformerBlock builds TE's activation-offload context,
    which creates a ``torch.cuda.Stream``; Inkling refuses CPU offloading anyway. On a host without
    a GPU, use the no-offload path (``nullcontext``) so the model tests run on CPU too."""
    if not torch.cuda.is_available():
        from megatron.core.transformer import transformer_block

        monkeypatch.setattr(transformer_block, "get_cpu_offload_context", None)


def _model(**overrides):
    torch.manual_seed(0)
    return InklingModel(small_config(**overrides), vocab_size=VOCAB, max_sequence_length=64)


@needs_model
def test_layer_numbers_and_sharded_keys(parallel):  # noqa: F811
    model = _model()
    layers = model.decoder.layers
    assert [layer.layer_number for layer in layers] == [1, 2]
    assert [layer.self_attention.is_sliding for layer in layers] == [True, False]
    sd = model.sharded_state_dict()
    keys = set(sd)
    # per-layer keys: sliding and global layers differ in shape (see InklingBlock.sharded_state_dict)
    assert "decoder.layers.0.self_attention.rel_proj" in keys
    assert "decoder.layers.1.self_attention.rel_proj" in keys
    assert not any(".-1." in k for k in keys)
    up = sd["decoder.layers.0.mlp.gate_up_proj"]
    # [expert axis (EP-sharded over global experts), 2I, H]
    assert up.global_shape[0] == model.config.num_moe_experts
    conv = next(v for k, v in sd.items() if k.endswith("self_attention.k_sconv.weight"))
    assert conv.global_shape[-1] == model.config.inkling_sconv_kernel_size


@needs_model
def test_dist_checkpoint_round_trip(parallel, tmp_path, monkeypatch):  # noqa: F811
    from megatron.core import dist_checkpointing

    if not torch.cuda.is_available():  # the torch_dist writer syncs CUDA before writing
        monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")

    model = _model()
    x = torch.randint(0, 90, (1, 12))
    with torch.no_grad():
        ref = model(x, None, None)
    dist_checkpointing.save(model.sharded_state_dict(), str(tmp_path))
    fresh = _model()
    with torch.no_grad():
        for p in fresh.parameters():
            p.add_(0.5)
        for layer in fresh.decoder.layers:  # persistent buffers must round-trip too
            layer.mlp.gate.expert_bias.add_(0.25)
    assert not torch.equal(
        fresh.decoder.layers[0].mlp.gate.expert_bias, model.decoder.layers[0].mlp.gate.expert_bias
    )
    loaded = dist_checkpointing.load(fresh.sharded_state_dict(), str(tmp_path))
    missing, unexpected = fresh.load_state_dict(loaded, strict=False)
    assert unexpected == [] and [k for k in missing if not k.endswith("_extra_state")] == []
    for (k, a), b in zip(model.state_dict().items(), fresh.state_dict().values()):
        if torch.is_tensor(a):
            torch.testing.assert_close(b, a, msg=k)
    with torch.no_grad():
        torch.testing.assert_close(fresh(x, None, None), ref)


@needs_model
def test_padded_vocab_masked_in_scoring_and_loss(parallel):  # noqa: F811
    model = _model().eval()
    x = torch.randint(0, 90, (2, 6))
    with torch.no_grad():
        logits = model(x, None, None)
        loss = model(x, None, None, labels=x)
    assert logits.shape == (2, 6, VOCAB)
    assert (logits[..., 90:] == torch.finfo(logits.dtype).min).all()
    assert torch.isfinite(logits[..., :90]).all()
    assert loss.shape == (2, 6) and torch.isfinite(loss).all()
    # the loss is the cross entropy over the 90 real ids only (padded columns carry no mass)
    want = F.cross_entropy(logits[..., :90].reshape(-1, 90).float(), x.reshape(-1), reduction="none")
    torch.testing.assert_close(loss.reshape(-1).float(), want, rtol=1e-4, atol=1e-4)
    # a padded-id label masked by loss_mask scores 0; the caller's loss_mask is not changed
    y = x.clone()
    y[0, 0] = 93
    mask = torch.ones(2, 6)
    mask[0, 0] = 0
    with torch.no_grad():
        loss2 = model(x, None, None, labels=y, loss_mask=mask)
    assert loss2[0, 0] == 0 and torch.isfinite(loss2).all() and mask.sum() == 11
    torch.testing.assert_close(loss2[1], loss[1])
    torch.testing.assert_close(loss2[0, 1:], loss[0, 1:])
    # an unmasked padded-id label (tokenizer / vocab mismatch) is refused, with or without a mask
    with torch.no_grad(), pytest.raises(ValueError, match="not masked by loss_mask"):
        model(x, None, None, labels=y, loss_mask=torch.ones(2, 6))
    with torch.no_grad(), pytest.raises(ValueError, match="not masked by loss_mask"):
        model(x, None, None, labels=y)
    # NullTokenizer mock data (inkling_ignored_label_id = its EOD id): that one id scores 0 even
    # unmasked; any other padded id is still refused
    model.config.inkling_ignored_label_id = 93
    try:
        with torch.no_grad():
            loss3 = model(x, None, None, labels=y, loss_mask=torch.ones(2, 6))
        torch.testing.assert_close(loss3, loss2)
        y[1, 2] = 94
        with torch.no_grad(), pytest.raises(ValueError, match="1 label"):
            model(x, None, None, labels=y, loss_mask=torch.ones(2, 6))
    finally:
        model.config.inkling_ignored_label_id = None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the deferred check needs a GPU")
def test_padded_label_check_is_deferred_on_gpu(monkeypatch):
    """On a GPU the padded-label check reads nothing from the device in the call: the counts are
    queued, and flush_padded_label_checks raises with their sum."""
    from primus.backends.megatron.core.models.inkling import inkling_model

    monkeypatch.setattr(inkling_model, "_LIVE_CHECKS", inkling_model.weakref.WeakSet())
    check = inkling_model._PaddedLabelCheck()
    clean = torch.zeros(4, dtype=torch.bool, device="cuda")
    with inkling_model.deferring_padded_label_checks():
        check(clean, 90)
    inkling_model.flush_padded_label_checks(90)
    with inkling_model.deferring_padded_label_checks():
        check(torch.tensor([False, True, True], device="cuda"), 90)  # queued, not raised here
        check(torch.tensor([True], device="cuda"), 90)
    assert len(check.pending) == 2
    with pytest.raises(ValueError, match=r"^Inkling: 3 label"):
        inkling_model.flush_padded_label_checks(90)
    assert check.pending == []
    # outside the forward-backward pass nothing is queued: the call checks at once
    with pytest.raises(ValueError, match=r"^Inkling: 1 label"):
        check(torch.tensor([True], device="cuda"), 90)
    assert check.pending == []


@needs_model
def test_backward_reaches_every_parameter(parallel):  # noqa: F811
    model = _model().train()
    x = torch.randint(0, 90, (2, 10))
    model(x, None, None, labels=x).mean().backward()
    assert [n for n, p in model.named_parameters() if p.grad is None] == []


@needs_model
@pytest.mark.parametrize(
    "modules", [["core_attn"], ["layernorm"], ["moe"], ["core_attn", "layernorm", "moe", "shared_experts"]]
)
def test_selective_recompute_matches(parallel, monkeypatch, modules):  # noqa: F811
    """Every recompute module engages its checkpoint (spied: the functions passed to
    ``tensor_parallel.checkpoint`` and ``CheckpointWithoutOutput.checkpoint``) and gives the
    no-recompute loss and gradients. Without a GPU, the CUDA part of Megatron's checkpoint RNG
    save / restore is a no-op (the model draws no random numbers: dropout is 0)."""
    from megatron.core import tensor_parallel
    from megatron.core.tensor_parallel import random as tp_random

    if not torch.cuda.is_available():
        monkeypatch.setattr(tp_random, "_get_cuda_rng_state", lambda *a, **k: None)
        monkeypatch.setattr(tp_random, "_set_cuda_rng_state", lambda *a, **k: None)
    seen = []
    real_checkpoint = tensor_parallel.checkpoint

    def checkpoint(fn, *a, **k):
        seen.append(fn.__name__)
        return real_checkpoint(fn, *a, **k)

    class SpyCheckpointWithoutOutput(tensor_parallel.CheckpointWithoutOutput):
        def checkpoint(self, fn, *a, **k):
            seen.append(getattr(fn, "__name__", type(fn).__name__))
            return super().checkpoint(fn, *a, **k)

    monkeypatch.setattr(tensor_parallel, "checkpoint", checkpoint)
    monkeypatch.setattr(tensor_parallel, "CheckpointWithoutOutput", SpyCheckpointWithoutOutput)
    x = torch.randint(0, 90, (2, 10))
    losses, grads, engaged = [], [], []
    for kw in ({}, {"recompute_granularity": "selective", "recompute_modules": modules}):
        model = _model(**kw).train()
        seen.clear()
        loss = model(x, None, None, labels=x).mean()
        engaged.append(set(seen))
        loss.backward()
        losses.append(loss.detach())
        grads.append({n: p.grad.clone() for n, p in model.named_parameters()})
    assert engaged[0] == set()
    want = {"core_attn": "_core_attention", "moe": "_forward", "shared_experts": "_shared"}
    for mod in modules:
        if mod == "layernorm":
            assert engaged[1] - set(want.values()), engaged[1]  # a norm checkpointed
        elif mod != "shared_experts" or "moe" not in modules:  # moe recompute covers _shared
            assert want[mod] in engaged[1], (mod, engaged[1])
    torch.testing.assert_close(losses[1], losses[0])
    for n in grads[0]:
        torch.testing.assert_close(grads[1][n], grads[0][n], msg=n)


@pytest.mark.parametrize(
    "override,match",
    [
        ({"inkling_logits_mup_width_multiplier": 0.0}, "mup"),
        ({"inkling_swa_num_query_groups": 3, "inkling_swa_num_attention_heads": 4}, "multiple"),
        ({"inkling_dense_mlp_idx": 1}, "inkling_dense_mlp_idx=1 must be 0"),
        ({"inkling_dense_mlp_idx": -1}, "inkling_dense_mlp_idx=-1 must be 0"),
        ({"mtp_num_layers": 2, "inkling_mtp_local_layer_ids": [-1]}, "inkling_mtp_local_layer_ids"),
        ({"mtp_num_layers": 2, "inkling_mtp_local_layer_ids": [1, 1]}, "inkling_mtp_local_layer_ids"),
        ({"mtp_num_layers": 1, "inkling_sliding_window": 0}, "inkling_mtp_local_extent is unset"),
        ({"mtp_num_layers": 2, "inkling_mtp_chain_hidden_post_norm": True}, "chain_hidden_post_norm"),
        ({"mtp_num_layers": 1, "mtp_use_repeated_layer": True}, "mtp_use_repeated_layer"),
        ({"inkling_dense_intermediate_size": 0}, "inkling_dense_intermediate_size"),
        ({"inkling_local_layer_ids": [-1]}, "non-negative"),
        ({"inkling_local_layer_ids": ["0", 1]}, "non-negative ints"),
        ({"inkling_local_layer_ids": [True]}, "non-negative ints"),
        ({"inkling_local_layer_ids": [0, 0]}, "duplicates"),
        ({"inkling_swa_num_query_groups": 0}, "inkling_swa_num_query_groups=0 must be a positive int"),
        ({"inkling_sconv_kernel_size": 0}, "inkling_sconv_kernel_size"),
        ({"inkling_sliding_window": -8}, "inkling_sliding_window"),
        ({"inkling_rel_extent": 0}, "inkling_rel_extent"),
        ({"inkling_d_rel": 0}, "inkling_d_rel"),
        ({"inkling_log_scaling_n_floor": 0}, "inkling_log_scaling_n_floor"),
        ({"inkling_log_scaling_n_floor": "128000"}, "inkling_log_scaling_n_floor='128000' must be a number"),
        (
            {"inkling_ignored_label_id": 5},
            "inkling_ignored_label_id=5 must be an int >= inkling_unpadded_vocab_size",
        ),
        ({"inkling_log_scaling_n_floor": True}, "inkling_log_scaling_n_floor=True must be a number"),
        # the MoE is replicated over TP: ETP != TP would mis-scale the expert gradients
        ({"tensor_model_parallel_size": 2, "expert_tensor_parallel_size": 1}, "expert_tensor_parallel_size"),
    ],
)
def test_config_validation(override, match):
    with pytest.raises(ValueError, match=match):
        small_config(**override)


@needs_model
def test_unpadded_vocab_must_fit(parallel):  # noqa: F811
    with pytest.raises(ValueError, match="unpadded"):
        InklingModel(
            small_config(inkling_unpadded_vocab_size=VOCAB + 1), vocab_size=VOCAB, max_sequence_length=64
        )


HF_TEXT = {
    "hidden_size": 768,
    "num_attention_heads": 8,
    "num_hidden_layers": 8,
    "num_key_value_heads": 2,
    "head_dim": 128,
    "rms_norm_eps": 1e-6,
    "intermediate_size": 384,
    "n_routed_experts": 256,
    "n_shared_experts": 2,
    "num_experts_per_tok": 6,
    "route_scale": 8.0,
    "dense_mlp_idx": 0,
    "d_rel": 16,
    "rel_extent": 1024,
    "sliding_window_size": 512,
    "local_layer_ids": [0, 2, 4, 6],
    "swa_num_attention_heads": 8,
    "swa_num_key_value_heads": 4,
    "swa_head_dim": 128,
    "log_scaling_n_floor": 128000,
    "log_scaling_alpha": 0.1,
    "sconv_kernel_size": 4,
    "use_embed_norm": True,
    "logits_mup_width_multiplier": 3.0,
    "unpadded_vocab_size": 200058,
    "vocab_size": 201024,
    "gate_activation": "sigmoid",
    "shared_expert_sink": True,
    "use_gate_bias": True,
    "use_global_scale": True,
    "dense_intermediate_size": 3072,
    "model_max_length": 1048576,
}


def test_converter_fields():
    f = loader.derive_megatron_fields({"text_config": HF_TEXT})
    assert f["inkling_route_scale"] == 8.0 and f["inkling_log_scaling_alpha"] == 0.1
    assert f["inkling_log_scaling_n_floor"] == 128000
    assert f["inkling_logits_mup_width_multiplier"] == 3.0
    assert f["inkling_local_layer_ids"] == [0, 2, 4, 6]
    assert f["inkling_swa_num_query_groups"] == 4 and f["inkling_swa_kv_channels"] == 128
    assert set(loader.INKLING_KEYS) <= set(f)


@pytest.mark.parametrize(
    "drop", ["route_scale", "local_layer_ids", "swa_head_dim", "logits_mup_width_multiplier"]
)
def test_converter_requires_keys(drop):
    cfg = {k: v for k, v in HF_TEXT.items() if k != drop}
    with pytest.raises(ValueError, match=drop):
        loader.derive_megatron_fields({"text_config": cfg})


@pytest.mark.parametrize(
    "extra,match",
    [
        ({"dense_mlp_idx": 1}, "dense_mlp_idx"),
        ({"attention_bias": True}, "attention_bias"),
        ({"tie_word_embeddings": True}, "tie_word_embeddings"),
        ({"rope_theta": 10000.0}, "RoPE"),
    ],
)
def test_converter_refuses_unported_variants(extra, match):
    with pytest.raises(ValueError, match=match):
        loader.derive_megatron_fields({"text_config": {**HF_TEXT, **extra}})


def test_converter_context_length():
    assert loader.derive_megatron_fields({"text_config": HF_TEXT})["max_position_embeddings"] == 1048576
    cfg = {k: v for k, v in HF_TEXT.items() if k != "model_max_length"}
    with pytest.raises(ValueError, match="model_max_length"):
        loader.derive_megatron_fields({"text_config": cfg})
    # a key that is present but null falls back like an absent one
    cfg = {**HF_TEXT, "max_position_embeddings": None, "model_max_length": 4096}
    assert loader.derive_megatron_fields({"text_config": cfg})["max_position_embeddings"] == 4096


class _Store:
    def __init__(self, keys):
        self.keys = set(keys)

    def all_keys(self):
        return set(self.keys)


def test_converter_skips_only_non_backbone_keys():
    store = _Store(
        [
            "model.llm.layers.0.attn_norm.weight",
            "model.llm.layers.1.attn_norm.weight",  # past num_layers (truncated build)
            "model.audio.encoder.weight",
            "model.visual.final_norm.weight",
            "model.llm.extra.weight",  # an unmapped backbone key: must stay unconsumed
        ]
    )
    consumed = {"model.llm.layers.0.attn_norm.weight"}
    loader._skip(store, consumed, SimpleNamespace(num_layers=1), lambda *a: None)
    assert store.all_keys() - consumed == {"model.llm.extra.weight"}


def test_converter_refuses_parallel_sizes():
    f = loader.derive_megatron_fields({"text_config": HF_TEXT})
    args = SimpleNamespace(tensor_parallel_size=2, pipeline_parallel_size=1, expert_parallel_size=1)
    with pytest.raises(ValueError, match="TP=PP=EP=1"):
        loader.build_margs(f, args)


def test_deinterleave():
    t = torch.arange(8.0).view(1, 8, 1)  # (g0, u0, g1, u1, ...)
    out = loader.deinterleave(t, 1).view(-1).tolist()
    assert out == [0.0, 2.0, 4.0, 6.0, 1.0, 3.0, 5.0, 7.0]


def test_init_seed_keeps_zero():
    from primus.backends.megatron.core.models.inkling.inkling_transformer_config import (
        INKLING_DEFAULT_INIT_SEED,
    )

    assert loader._init_seed(SimpleNamespace(seed=0)) == 0
    assert loader._init_seed(SimpleNamespace(seed=7)) == 7
    assert loader._init_seed(SimpleNamespace(seed=None)) == INKLING_DEFAULT_INIT_SEED
    assert loader._init_seed(SimpleNamespace()) == INKLING_DEFAULT_INIT_SEED


def test_builder_refuses_reset_position_ids():
    from primus.backends.megatron.core.models.inkling import inkling_builders

    args = SimpleNamespace(multi_latent_attention=False, use_legacy_models=False, reset_position_ids=True)
    with pytest.raises(ValueError, match="reset_position_ids"):
        inkling_builders.inkling_builder(args, True, True, config=small_config())


def test_builder_ignores_the_null_tokenizer_eod_label_only_for_mock_data(monkeypatch):
    """NullTokenizer mock data puts its EOD id (= vocab_size, past the HF vocab) in the labels:
    the builder builds the model with inkling_ignored_label_id = that id, on a validated copy (the
    caller's config is unchanged); it keeps None otherwise (a real tokenizer, a NullTokenizer on
    real data, or a mock vocab whose EOD id is a real id)."""
    from primus.backends.megatron.core.models.inkling import inkling_builders

    monkeypatch.setattr(inkling_builders, "InklingModel", lambda **kw: kw["config"])
    for tokenizer, mock, vocab, ignored in (
        ("NullTokenizer", True, 96, 96),
        ("NullTokenizer", True, 50, None),
        ("NullTokenizer", False, 96, None),
        ("HuggingFaceTokenizer", False, 96, None),
    ):
        cfg = small_config(inkling_unpadded_vocab_size=90)
        args = SimpleNamespace(
            multi_latent_attention=False,
            use_legacy_models=False,
            reset_position_ids=False,
            tokenizer_type=tokenizer,
            mock_data=mock,
            vocab_size=vocab,
            padded_vocab_size=96,
            max_position_embeddings=64,
            fp16_lm_cross_entropy=False,
            untie_embeddings_and_output_weights=True,
        )
        built = inkling_builders.inkling_builder(args, True, True, config=cfg)
        assert built.inkling_ignored_label_id == ignored
        assert cfg.inkling_ignored_label_id is None


def _hook_module():
    """The posttrain convert hook (a script, not a package module)."""
    import importlib.util as iu

    root = Path(__file__).resolve().parents[5]
    path = root / "runner/helpers/hooks/train/posttrain/megatron/01_convert_checkpoints.py"
    spec = iu.spec_from_file_location("_inkling_convert_hook", path)
    mod = iu.module_from_spec(spec)
    sys.path.insert(0, str(root))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(root))
    return mod


@pytest.mark.parametrize(
    "cfg,family",
    [
        ({"architectures": ["InklingForConditionalGeneration"], "model_type": "inkling_mm_model"}, "inkling"),
        ({"model_type": "inkling_mm_model"}, "inkling"),
        ({"architectures": ["LlamaForCausalLM"], "model_type": "llama"}, "llama"),
        ({"model_type": "something_else"}, None),
    ],
)
def test_hook_detects_inkling_family(tmp_path, cfg, family):
    import json

    (tmp_path / "config.json").write_text(json.dumps(cfg))
    assert _hook_module()._detect_native_family(str(tmp_path)) == family


@pytest.mark.parametrize(
    "layout,refused",
    [
        ({}, None),
        ({"tensor_model_parallel_size": 1, "expert_model_parallel_size": 1, "num_layers": 8}, None),
        ({"expert_model_parallel_size": 2}, "expert_model_parallel_size"),
        ({"tensor_model_parallel_size": "2"}, "tensor_model_parallel_size"),
        ({"pipeline_model_parallel_size": 2}, "pipeline_model_parallel_size"),
        ({"num_layers": 4}, "the checkpoint has 8"),
        ({"expert_tensor_parallel_size": 2}, "expert_tensor_parallel_size"),
        ({"num_layers_per_virtual_pipeline_stage": 2}, "num_layers_per_virtual_pipeline_stage"),
        ({"decoder_first_pipeline_num_layers": None, "expert_tensor_parallel_size": 1}, None),
        ({"tensor_model_parallel_size": "${TP}"}, "tensor_model_parallel_size='.*not an integer"),
        # int() would truncate these into a passing value: refuse them instead
        ({"tensor_model_parallel_size": 1.5}, "tensor_model_parallel_size=1.5 .float. is not an integer"),
        ({"expert_model_parallel_size": True}, "expert_model_parallel_size=True .bool. is not an integer"),
        ({"num_layers": 8.0}, "num_layers=8.0 .float. is not an integer"),
        ({"tensor_model_parallel_size": "1.0"}, "not an integer"),
        ({"tensor_model_parallel_size": "1", "num_layers": "8"}, None),  # CLI strings
        # virtual / custom pipeline splits the unsharded checkpoint cannot load
        ({"virtual_pipeline_model_parallel_size": 2}, "virtual_pipeline_model_parallel_size"),
        ({"virtual_pipeline_model_parallel_size": 1}, None),
        ({"pipeline_model_parallel_layout": "Et|tL"}, "pipeline_model_parallel_layout"),
        # with PP = 1 these flags change neither the split nor the files: accepted in any spelling
        ({"account_for_embedding_in_pipeline_split": True, "pipeline_model_parallel_size": 1}, None),
        ({"account_for_loss_in_pipeline_split": "true"}, None),
        ({"account_for_loss_in_pipeline_split": 0, "account_for_embedding_in_pipeline_split": "no"}, None),
        (
            {"account_for_loss_in_pipeline_split": True, "pipeline_model_parallel_size": 2},
            "pipeline_model_parallel_size",
        ),
        (
            {"num_layers": True},
            "num_layers=True .bool. is not an integer .a command-line --num_layers without a value",
        ),
    ],
)
def test_hook_refuses_a_layout_the_inkling_converter_cannot_write(tmp_path, layout, refused):
    """The converter writes TP = PP = EP = ETP = 1 at the full depth: the hook refuses any other
    training layout before converting."""
    import json

    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"num_hidden_layers": 8}}))
    hook = _hook_module()
    if refused is None:
        hook.check_native_layout("inkling", str(tmp_path), layout)
    else:
        with pytest.raises(ValueError, match=refused):
            hook.check_native_layout("inkling", str(tmp_path), layout)


def test_hook_main_logs_a_refused_layout_and_exits_1(tmp_path, monkeypatch):
    """_native_main reports a refused layout like its other refusals: log_error, then exit 1."""
    import json

    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "inkling_mm_model", "text_config": {"num_hidden_layers": 8}})
    )
    hook = _hook_module()
    logged = []
    monkeypatch.setattr(hook, "log_error", logged.append)
    with pytest.raises(SystemExit) as e:
        hook._native_main(str(tmp_path), {"train_layout": {"expert_model_parallel_size": 2}})
    assert e.value.code == 1
    assert len(logged) == 1 and "expert_model_parallel_size" in logged[0]


@pytest.mark.parametrize(
    "cfg, refused",
    [
        ({"text_config": {}}, "has no num_hidden_layers"),
        ({}, "has no num_hidden_layers"),
        ({"num_hidden_layers": 8}, None),  # a flat (text-only) config.json
    ],
)
def test_hook_refuses_when_the_checkpoint_depth_cannot_be_read(tmp_path, cfg, refused):
    """A config.json without num_hidden_layers cannot confirm num_layers: refuse, not skip."""
    import json

    (tmp_path / "config.json").write_text(json.dumps(cfg))
    hook = _hook_module()
    if refused is None:
        hook.check_native_layout("inkling", str(tmp_path), {"num_layers": 8})
    else:
        with pytest.raises(ValueError, match=refused):
            hook.check_native_layout("inkling", str(tmp_path), {"num_layers": 8})


def test_cli_layout_overrides_are_read_like_the_training_command():
    hook = _hook_module()
    argv = [
        "--tensor_model_parallel_size",
        "8",
        "--num-layers=4",
        "--lr",
        "1e-4",
        "--expert_model_parallel_size",
        "--decoder_first_pipeline_num_layers",
        "null",
    ]
    # parse_cli_overrides (the training command's parser): ints are parsed, a bare flag is True
    # (refused by the layout check as not an integer), "null" is kept and read as unset
    assert hook.cli_layout_overrides(argv) == {
        "tensor_model_parallel_size": 8,
        "num_layers": 4,
        "expert_model_parallel_size": True,
        "decoder_first_pipeline_num_layers": "null",
    }


@pytest.mark.parametrize("value", [None, "", "null", "None", " NULL "])
@pytest.mark.parametrize(
    "key", ["decoder_first_pipeline_num_layers", "tensor_model_parallel_size", "num_layers"]
)
def test_an_unset_value_from_any_source_is_unset(tmp_path, key, value):
    """YAML, env-substituted (${PP_FIRST:null}) and CLI values agree: null / none / empty is unset."""
    import json

    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"num_hidden_layers": 8}}))
    _hook_module().check_native_layout("inkling", str(tmp_path), {key: value})


@pytest.mark.parametrize("value", ["+-1", "1.5", 8.0, True, "abc"])
def test_a_non_integer_size_is_refused_with_the_layout_error(tmp_path, value):
    with pytest.raises(ValueError, match="is not an integer"):
        _hook_module().check_native_layout("inkling", str(tmp_path), {"tensor_model_parallel_size": value})


def test_hook_main_checks_the_cli_layout_over_the_yaml(tmp_path, monkeypatch):
    """A layout the YAML allows but the command line overrides (--num_layers 4) is refused."""
    import json

    hf = tmp_path / "hf"
    hf.mkdir()
    (hf / "config.json").write_text(
        json.dumps({"model_type": "inkling_mm_model", "text_config": {"num_hidden_layers": 8}})
    )
    hook = _hook_module()
    monkeypatch.setattr(hook, "get_checkpoint_config", lambda cfg: (str(hf), None))
    monkeypatch.setattr(
        hook,
        "read_native_opts",
        lambda cfg: {"enabled": None, "dtype": None, "out": None, "train_layout": {"num_layers": 8}},
    )
    logged = []
    monkeypatch.setattr(hook, "log_error", logged.append)
    monkeypatch.setattr(sys, "argv", ["hook", "--config", str(tmp_path / "x.yaml"), "--num_layers", "4"])
    with pytest.raises(SystemExit) as e:
        hook.main()
    assert e.value.code == 1
    assert len(logged) == 1 and "num_layers" in logged[0] and "the checkpoint has 8" in logged[0]


@pytest.mark.parametrize("key", ["expert_tensor_parallel_size", "num_layers", "tensor_model_parallel_size"])
def test_hook_treats_a_cli_null_as_unset(tmp_path, key):
    """`--key null` unsets the key, as the YAML null and the training command do: no refusal,
    and it also clears a YAML value the CLI overrides."""
    import json

    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"num_hidden_layers": 8}}))
    hook = _hook_module()
    layout = {key: 4}  # the YAML value, refused alone
    with pytest.raises(ValueError):
        hook.check_native_layout("inkling", str(tmp_path), dict(layout))
    layout.update(hook.cli_layout_overrides([f"--{key}", "null"]))
    hook.check_native_layout("inkling", str(tmp_path), layout)


def test_hook_reads_a_top_level_depth_when_text_config_has_null(tmp_path):
    import json

    (tmp_path / "config.json").write_text(
        json.dumps({"text_config": {"num_hidden_layers": None}, "num_hidden_layers": 4})
    )
    hook = _hook_module()
    hook.check_native_layout("inkling", str(tmp_path), {"num_layers": 4})
    with pytest.raises(ValueError, match="the checkpoint has 4"):
        hook.check_native_layout("inkling", str(tmp_path), {"num_layers": 8})


def test_hook_parses_the_cli_only_for_inkling(tmp_path, monkeypatch):
    """Other families never see the layout parser (and its warnings on tokens it does not know)."""
    import json

    hf = tmp_path / "hf"
    hf.mkdir()
    (hf / "config.json").write_text(json.dumps({"model_type": "llama"}))
    hook = _hook_module()
    monkeypatch.setattr(hook, "get_checkpoint_config", lambda cfg: (str(hf), None))
    monkeypatch.setattr(
        hook,
        "read_native_opts",
        lambda cfg: {"enabled": None, "dtype": None, "out": None, "train_layout": {}},
    )
    monkeypatch.setattr(hook, "cli_layout_overrides", lambda argv: pytest.fail("parsed for llama"))
    seen = []
    monkeypatch.setattr(hook, "_native_main", lambda path, opts: seen.append(opts["train_layout"]))
    monkeypatch.setattr(
        sys, "argv", ["hook", "--config", str(tmp_path / "x.yaml"), "positional", "--num_layers", "4"]
    )
    hook.main()
    assert seen == [{}]


def test_hook_does_not_check_the_layout_of_other_families(tmp_path):
    _hook_module().check_native_layout("llama", str(tmp_path), {"expert_model_parallel_size": 2})


def test_hook_reads_the_training_layout_from_the_config(tmp_path, monkeypatch):
    """read_native_opts takes the layout from the post_trainer module's settings; fed to
    check_native_layout, an EP=2 / truncated config is refused."""
    import json

    import yaml

    root = Path(__file__).resolve().parents[5]
    exp = yaml.safe_load(
        (root / "examples/megatron/configs/MI355X/llama3_8B-BF16-sft-packed-squad.yaml").read_text()
    )
    exp["modules"]["post_trainer"]["overrides"].update(
        {"expert_model_parallel_size": 2, "num_layers": 4, "tensor_model_parallel_size": 1}
    )
    cfg_file = tmp_path / "exp.yaml"
    cfg_file.write_text(yaml.safe_dump(exp))
    hook = _hook_module()
    monkeypatch.chdir(root)  # the experiment's model / module presets resolve from the repo root
    layout = hook.read_native_opts(str(cfg_file))["train_layout"]
    assert layout["expert_model_parallel_size"] == 2 and layout["num_layers"] == 4
    assert layout["tensor_model_parallel_size"] == 1 and "pipeline_model_parallel_size" in layout
    hf = tmp_path / "hf"
    hf.mkdir()
    (hf / "config.json").write_text(json.dumps({"text_config": {"num_hidden_layers": 8}}))
    with pytest.raises(ValueError, match="expert_model_parallel_size.*num_layers"):
        hook.check_native_layout("inkling", str(hf), layout)


def test_hook_reads_layout_keys_set_at_module_level(monkeypatch):
    """Like native_ckpt_convert / _dtype / _out, a layout key outside ``params`` is read from the
    module itself (``params`` first)."""
    hook = _hook_module()
    module = SimpleNamespace(
        params=SimpleNamespace(tensor_model_parallel_size=1),
        tensor_model_parallel_size=4,
        expert_model_parallel_size=2,
        num_layers=4,
    )
    monkeypatch.setattr(hook, "load_primus_config", lambda *a: SimpleNamespace())
    monkeypatch.setattr(
        hook, "get_module_config", lambda cfg, name: module if name == "post_trainer" else None
    )
    layout = hook.read_native_opts("unused.yaml")["train_layout"]
    assert layout == {"tensor_model_parallel_size": 1, "expert_model_parallel_size": 2, "num_layers": 4}


class _TensorStore(_Store):
    """A fake SafetensorsStore over an in-memory {name: tensor} dict."""

    def __init__(self, tensors):
        super().__init__(tensors)
        self.tensors = tensors

    def has(self, k):
        return k in self.tensors

    def get(self, k, dtype):
        return self.tensors[k].to(dtype)


def _hf_tensors(model):
    """A random HF-layout text checkpoint for ``model``'s shapes (gate / up stored interleaved)."""
    cfg, g = model.config, torch.Generator().manual_seed(11)
    msd = model.state_dict()

    def rnd(*shape):
        return torch.randn(*shape, generator=g)

    def interleave(gate, up, dim):  # inverse of loader.deinterleave
        return torch.stack((gate, up), dim + 1).flatten(dim, dim + 1)

    P, t, plan = loader.PREFIX, {}, SimpleNamespace(routed=[], shared=[])
    t[P + "embed.weight"] = rnd(90, cfg.hidden_size)
    t[P + "embed_norm.weight"] = rnd(cfg.hidden_size)
    t[P + "norm.weight"] = rnd(cfg.hidden_size)
    t[P + "unembed.weight"] = rnd(90, cfg.hidden_size)
    for L in range(cfg.num_layers):
        h, m = f"{P}layers.{L}.", f"decoder.layers.{L}."
        a, ma = h + "attn.", m + "self_attention."
        for src, dst in (("attn_norm", "input_layernorm"), ("mlp_norm", "pre_mlp_layernorm")):
            t[h + src + ".weight"] = rnd(*msd[m + dst + ".weight"].shape)
        for name in ("attn_sconv", "mlp_sconv"):
            t[h + name + ".weight"] = rnd(
                msd[m + name + ".weight"].shape[0], 1, cfg.inkling_sconv_kernel_size
            )
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
            t[a + name + ".weight"] = rnd(
                msd[ma + name + ".weight"].shape[0], 1, cfg.inkling_sconv_kernel_size
            )
        t[a + "rel_logits_proj.proj"] = rnd(*msd[ma + "rel_proj"].shape)
        g_ = h + "mlp."
        t[g_ + "gate.weight"] = rnd(*msd[m + "mlp.gate.weight"].shape)
        t[g_ + "gate.global_scale"] = rnd(())
        t[g_ + "gate.bias"] = rnd(cfg.num_moe_experts)
        E, I, H = msd[m + "mlp.gate_up_proj"].shape[0], cfg.moe_ffn_hidden_size, cfg.hidden_size
        gate, up = rnd(E, I, H), rnd(E, I, H)
        t[g_ + "experts.w13_weight"] = interleave(gate, up, 1)
        t[g_ + "experts.w2_weight"] = rnd(E, H, I)
        S = cfg.inkling_num_shared_experts
        sgate, sup = rnd(S, I, H), rnd(S, I, H)
        t[g_ + "shared_experts.shared_w13_weight"] = interleave(sgate, sup, 1)
        t[g_ + "shared_experts.shared_w2_weight"] = rnd(S, H, I)
        plan.routed.append((gate, up))
        plan.shared.append((sgate, sup))
    t["model.visual.final_norm.weight"] = rnd(4)  # skipped tower
    return t, plan


@needs_model
def test_converter_maps_every_hf_tensor(parallel):  # noqa: F811
    """build_mcore_state_dict: every mapped tensor by value (a swapped same-shape pair fails),
    routed / shared de-interleave, sconv squeeze, fp32 router bias, vocab padding; every key is
    consumed or skipped."""
    model = _model()
    tensors, plan = _hf_tensors(model)
    store = _TensorStore(tensors)
    margs = SimpleNamespace(num_layers=model.config.num_layers, params_dtype=torch.float32)
    logs = []
    sd, consumed = loader.build_mcore_state_dict(
        model, store, margs, {"inkling_unpadded_vocab_size": 90}, logs.append
    )
    assert any("checkpoint keys consumed (mapped + skipped)" in m for m in logs)
    assert consumed == set(tensors)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert unexpected == [] and [k for k in missing if not k.endswith("_extra_state")] == []
    P, msd = loader.PREFIX, model.state_dict()
    torch.testing.assert_close(msd["embedding.word_embeddings.weight"][:90], tensors[P + "embed.weight"])
    # padded rows replicate the last real row (common.pad_vocab); the model masks those logits
    unembed = tensors[P + "unembed.weight"]
    torch.testing.assert_close(msd["output_layer.weight"], torch.cat((unembed, unembed[-1:].expand(6, -1))))
    for L in range(model.config.num_layers):
        h, m = f"{P}layers.{L}.", f"decoder.layers.{L}."
        gate, up = plan.routed[L]
        torch.testing.assert_close(msd[m + "mlp.gate_up_proj"], torch.cat((gate, up), 1))
        sgate, sup = plan.shared[L]
        torch.testing.assert_close(msd[m + "mlp.shared_gate_up_proj"], torch.cat([sgate, sup], 1))
        # every directly copied tensor, by value (same-shape pairs: k / v, q_norm / k_norm, the norms)
        direct = {
            "input_layernorm.weight": "attn_norm.weight",
            "pre_mlp_layernorm.weight": "mlp_norm.weight",
            "self_attention.q_proj.weight": "attn.wq_du.weight",
            "self_attention.k_proj.weight": "attn.wk_dv.weight",
            "self_attention.v_proj.weight": "attn.wv_dv.weight",
            "self_attention.o_proj.weight": "attn.wo_ud.weight",
            "self_attention.r_proj.weight": "attn.wr_du.weight",
            "self_attention.q_norm.weight": "attn.q_norm.weight",
            "self_attention.k_norm.weight": "attn.k_norm.weight",
            "mlp.gate.weight": "mlp.gate.weight",
            "mlp.down_proj": "mlp.experts.w2_weight",
            "mlp.shared_down_proj": "mlp.shared_experts.shared_w2_weight",
        }
        for mk, hk_ in direct.items():
            torch.testing.assert_close(msd[m + mk], tensors[h + hk_], msg=mk)
        for mk, hk_ in {
            "attn_sconv.weight": "attn_sconv.weight",
            "self_attention.v_sconv.weight": "attn.v_sconv.weight",
        }.items():
            torch.testing.assert_close(msd[m + mk], tensors[h + hk_][:, 0], msg=mk)
        torch.testing.assert_close(
            msd[m + "self_attention.rel_proj"], tensors[h + "attn.rel_logits_proj.proj"]
        )
        torch.testing.assert_close(
            msd[m + "self_attention.k_sconv.weight"], tensors[h + "attn.k_sconv.weight"][:, 0]
        )
        torch.testing.assert_close(msd[m + "mlp_sconv.weight"], tensors[h + "mlp_sconv.weight"][:, 0])
        torch.testing.assert_close(msd[m + "mlp.gate.expert_bias"], tensors[h + "mlp.gate.bias"])
        assert msd[m + "mlp.gate.expert_bias"].dtype == torch.float32
        torch.testing.assert_close(
            msd[m + "mlp.gate.global_scale"], tensors[h + "mlp.gate.global_scale"].reshape(1)
        )


@needs_model
def test_converter_refuses_shape_mismatch(parallel):  # noqa: F811
    model = _model()
    tensors, _ = _hf_tensors(model)
    k = loader.PREFIX + "layers.0.attn.wr_du.weight"
    tensors[k] = torch.randn(tensors[k].shape[0] + 1, tensors[k].shape[1])
    margs = SimpleNamespace(num_layers=model.config.num_layers, params_dtype=torch.float32)
    with pytest.raises(ValueError, match=r"r_proj.weight has shape .* model expects"):
        loader.build_mcore_state_dict(model, _TensorStore(tensors), margs, {}, lambda *a: None)


@needs_model
def test_converter_refuses_more_layers_than_the_checkpoint(parallel):  # noqa: F811
    model = _model()
    tensors, _ = _hf_tensors(model)
    last = model.config.num_layers - 1
    tensors = {k: v for k, v in tensors.items() if not k.startswith(f"{loader.PREFIX}layers.{last}.")}
    margs = SimpleNamespace(num_layers=model.config.num_layers, params_dtype=torch.float32)
    with pytest.raises(ValueError, match="the checkpoint has"):
        loader.build_mcore_state_dict(model, _TensorStore(tensors), margs, {}, lambda *a: None)


@needs_model
def test_converter_refuses_a_layer_gap(parallel):  # noqa: F811
    model = _model()
    tensors, _ = _hf_tensors(model)
    tensors = {k: v for k, v in tensors.items() if not k.startswith(f"{loader.PREFIX}layers.0.")}
    margs = SimpleNamespace(num_layers=model.config.num_layers - 1, params_dtype=torch.float32)
    with pytest.raises(ValueError, match=r"not contiguous \(missing layer\(s\) \[0\]"):
        loader.build_mcore_state_dict(model, _TensorStore(tensors), margs, {}, lambda *a: None)


@needs_model
def test_converter_truncation_only_when_allowed(parallel):  # noqa: F811
    """Fewer model layers than the checkpoint: a WARNING for a proxy, refused for a conversion."""
    model = _model()
    tensors, _ = _hf_tensors(model)
    extra = model.config.num_layers
    for k in [k for k in tensors if k.startswith(f"{loader.PREFIX}layers.0.")]:
        tensors[k.replace("layers.0.", f"layers.{extra}.", 1)] = tensors[k].clone()
    margs = SimpleNamespace(num_layers=model.config.num_layers, params_dtype=torch.float32)
    with pytest.raises(ValueError, match="a conversion must cover every layer"):
        loader.build_mcore_state_dict(
            model, _TensorStore(tensors), margs, {}, lambda *a: None, allow_truncated=False
        )
    logs = []
    loader.build_mcore_state_dict(model, _TensorStore(tensors), margs, {}, logs.append)
    assert any("truncated proxy" in str(line) for line in logs)


@needs_model
def test_converter_embed_norm_follows_the_model(parallel):  # noqa: F811
    """embed_norm.weight is mapped because the model has embed_norm (use_embed_norm), and a
    checkpoint that disagrees with the flag is refused in either direction."""
    model = _model()
    assert model.embed_norm is not None
    tensors, _ = _hf_tensors(model)
    margs = SimpleNamespace(num_layers=model.config.num_layers, params_dtype=torch.float32)
    del tensors[loader.PREFIX + "embed_norm.weight"]
    with pytest.raises(ValueError, match="use_embed_norm: true"):
        loader.build_mcore_state_dict(model, _TensorStore(tensors), margs, {}, lambda *a: None)
    no_norm = _model(inkling_use_embed_norm=False)
    assert no_norm.embed_norm is None
    tensors, _ = _hf_tensors(no_norm)
    tensors[loader.PREFIX + "embed_norm.weight"] = torch.ones(no_norm.config.hidden_size)
    with pytest.raises(ValueError, match="use_embed_norm: false"):
        loader.build_mcore_state_dict(no_norm, _TensorStore(tensors), margs, {}, lambda *a: None)


@pytest.mark.parametrize("dtype", ["bfloat16", "float32", "", None])
def test_converter_refuses_unknown_dtype(dtype):
    """build_margs refuses a dtype other than bf16 / fp16 / fp32 (no silent fp32 fallback),
    before it reads the HF config fields."""
    args = SimpleNamespace(
        dtype=dtype, tensor_parallel_size=1, pipeline_parallel_size=1, expert_parallel_size=1
    )
    with pytest.raises(ValueError, match="is not one of 'bf16', 'fp16', 'fp32'"):
        loader.build_margs({}, args)
    assert loader._params_dtype("bf16") is torch.bfloat16
    assert loader._params_dtype("fp32") is torch.float32
