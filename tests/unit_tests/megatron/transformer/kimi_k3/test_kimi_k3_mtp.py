###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Kimi K3 Multi-Token Prediction: spec construction, wiring, gradients, causality.

Three things can go wrong here and only one of them is loud.

1. **The spec refuses to build.** Upstream's
   :class:`MultiTokenPredictionLayer` validates the inner layer's
   ``attn_mask_type`` and calls ``enorm`` / ``hnorm`` / ``layer_norm`` as raw
   classes. Both failures raise, so they need one test each and no negative
   control.
2. **The attention-residual contract is violated quietly.** The MTP layer
   sits after :class:`AttentionResidualHead` has already collapsed the
   checkpoint set. If it kept ``use_attn_residuals`` on it would either trip
   the drift assert (loud) *or*, on a shape where
   ``num_layers % attn_res_block_size == 0``, silently append a spurious
   checkpoint and mix a candidate set the model was never meant to see. That
   second case is what this file is really about, so it is tested from both
   ends: what is built, and what the forward actually touches.
3. **The MTP loss does not reach the MTP parameters.**
   ``mtp_loss_scaling_factor`` multiplies straight through to every MTP
   gradient (``multi_token_prediction.py:693``, ``:705-710``), and the MTP loss
   never appears in the reported lm loss, so a disconnected MTP block trains a
   perfectly healthy-looking curve while its parameters sit at initialisation
   forever.

Every claim in group 2 and 3 is stated as a **predicate** in
:data:`_PREDICATES`, checked once on the real wiring and once under each
deliberate bug in :func:`_injections`. A predicate that cannot fail proves
nothing, so both halves are required.

Numerics: fp32 throughout, unlike ``test_kimi_k3_block.py``'s bf16. Two
assertions here are equalities rather than tolerances -- the MTP label shift
(:func:`_mtp_predicts_two_ahead`) and the causality certificate -- and at
hidden 256 the bf16 rounding on a 128-token mean is the same order as the
signal those tests look for. The shape is small enough that fp32 costs
nothing.

Shape: the ``test_kimi_k3_block.py`` geometry -- 8 layers, hidden 256,
``attn_res_block_size=4`` -- because that is the smallest shape with a
*growing* ``block_residual`` axis, and because ``8 % 4 == 0`` makes the MTP
layer's index an *appending* one, which is precisely the silent failure mode.
"""

from __future__ import annotations

import contextlib
from typing import List, Optional

import pytest
import torch

from tests.unit_tests.megatron.transformer.kimi_k3.test_flops_closed_form import (
    _args as _flops_args,
)
from tests.unit_tests.megatron.transformer.kimi_k3.test_kimi_k3_block import (  # noqa: F401
    ATTN_RES_BLOCK_SIZE,
    BATCH,
    HIDDEN,
    NUM_LAYERS,
    SEQ,
    _make_config,
    mpu_tp1,
)

VOCAB = 512


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _unset_nvte_attention_env(monkeypatch):
    """Clear the TE attention-backend env vars.

    This is the first Kimi K3 unit test to build a whole
    :class:`KimiK3Model` rather than just the decoder block, and
    ``LanguageModule._set_attention_backend`` (reached from
    ``setup_embeddings_and_output_layer``) asserts that ``NVTE_FLASH_ATTN`` is
    unset-or-1 under the default ``auto`` backend
    (``language_module.py:99-105``). ``rocm/primus:v26.4`` bakes
    ``NVTE_FLASH_ATTN=0`` because it targets the fused/CK path, so every model
    construction fails without this. Production is fine -- Primus's
    ``megatron.attention_backend`` patch handles it -- but unit tests apply no
    patches.

    ``monkeypatch.delenv`` rather than ``os.environ.pop``:
    ``_set_attention_backend`` writes the vars back after its check, and
    monkeypatch's teardown is what contains that leak. Same fixture as
    ``tests/unit_tests/backends/megatron/diffusion/conftest.py``.
    """
    for var in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def _clear_loss_trackers():
    """Both aux-loss trackers are process-wide globals; empty them per test.

    ``save_to_aux_losses_tracker`` sizes its buffer ``num_layers +
    mtp_num_layers`` on first use and keeps it (``moe_utils.py:953-957``), so a
    ``mtp_num_layers=1`` test followed by a ``mtp_num_layers=2`` test indexes
    past the end and raises ``IndexError`` from inside the router --
    a cross-test artefact that looks exactly like a wiring bug.
    ``MTPLossLoggingHelper.tracker`` is the same shape of global and
    :func:`_mtp_predicts_two_ahead` reads it.
    """
    from megatron.core.transformer.moe.moe_utils import (
        get_moe_layer_wise_logging_tracker,
    )
    from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper

    def reset():
        # ``clear_aux_losses_tracker()`` only zeroes the buffers and keeps their
        # length, which is the very thing that has to change between a
        # 1-depth and a 2-depth test.
        get_moe_layer_wise_logging_tracker().clear()
        MTPLossLoggingHelper.tracker.clear()

    reset()
    yield
    reset()


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _mtp_config(
    *,
    mtp_depths: Optional[int] = 1,
    mtp_layer_type: str = "mirror_last",
    num_layers: int = NUM_LAYERS,
    attn_res_block_size: Optional[int] = ATTN_RES_BLOCK_SIZE,
    mtp_loss_scaling_factor: float = 0.1,
    use_nextn_name: bool = True,
    dense_ffn_only: bool = False,
    **kwargs,
):
    """The block test's config with MTP turned on.

    The MTP fields are set after construction and the resolver re-run, rather
    than threading five more keywords through ``_make_config``: the resolver
    normalises to a canonical pair so it is idempotent, and this keeps the
    shared geometry defined in exactly one place.

    ``use_nextn_name`` selects which of the two aliases the caller sets, so the
    reconciliation is exercised from both sides.

    ``dense_ffn_only`` blanks ``moe_layer_freq`` so every FFN -- including the
    MTP layer's -- is the dense ``situ`` MLP. It is done here rather than by
    unsetting ``num_moe_experts``, which upstream rejects while
    ``moe_ffn_hidden_size`` is set (``transformer_config.py:1178-1181``), and
    the spec builders read the *pattern* rather than the expert count
    (``kimi_k3_layer_specs.get_kimi_k3_moe_layer_pattern``).
    """
    kwargs.setdefault("params_dtype", torch.float32)
    config = _make_config(
        num_layers=num_layers, attn_res_block_size=attn_res_block_size, **kwargs
    )
    if dense_ffn_only:
        config.moe_layer_freq = [0] * num_layers
    if use_nextn_name:
        config.num_nextn_predict_layers = mtp_depths
        config.mtp_num_layers = None
    else:
        config.num_nextn_predict_layers = None
        config.mtp_num_layers = mtp_depths
    config.mtp_layer_type = mtp_layer_type
    config.mtp_loss_scaling_factor = mtp_loss_scaling_factor
    config._resolve_mtp_fields()
    config.vocab_size = VOCAB
    config.padded_vocab_size = VOCAB
    return config


def _build_model(config, *, pre_process: bool = True, post_process: bool = True):
    # Upstream's MTP loss logger allocates its tracker on
    # ``torch.cuda.current_device()`` unconditionally
    # (``multi_token_prediction.py:362``), which is reached by every
    # training-mode forward with MTP on. Skipping here rather than decorating
    # twenty tests keeps the gate in one place and impossible to forget; the
    # config-only tests above still run on CPU.
    if not torch.cuda.is_available():
        pytest.skip(
            "MTP's loss tracker allocates on torch.cuda.current_device() "
            "(multi_token_prediction.py:362), so the MTP training path needs a GPU"
        )

    from megatron.core.process_groups_config import ProcessGroupCollection

    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_layer_specs import (
        get_kimi_k3_runtime_decoder_spec,
    )
    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_model import KimiK3Model

    model = KimiK3Model(
        config=config,
        transformer_layer_spec=get_kimi_k3_runtime_decoder_spec(config),
        vocab_size=VOCAB,
        max_sequence_length=SEQ,
        pre_process=pre_process,
        post_process=post_process,
        # ``True`` is what ``kimi_k3_builder`` passes in production, and with
        # MTP it is not merely a preference. At ``parallel_output=False`` the
        # output layer's closing ``gather_from_tensor_model_parallel_region``
        # returns its input unchanged at TP=1, and
        # ``VocabParallelCrossEntropy`` then rewrites those logits in place --
        # which PyTorch forbids for a tensor a custom ``Function`` returned
        # as-is, and rejects when the saved tensor is unpacked in the backward
        # (``cross_entropy.py:196``). One such site is tolerated; MTP adds a
        # second cross-entropy over a second set of gathered logits and the
        # backward raises. ``validate/t2_causality.py`` uses
        # ``parallel_output=False`` and never hit it because it has no MTP.
        # At TP=1 the logits are full-width either way, so nothing is lost.
        parallel_output=True,
        share_embeddings_and_output_weights=False,
        position_embedding_type="none",
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
    )
    return model.to(_device())


def _causal_mask(seq: int = SEQ, batch: int = BATCH):
    """``True`` == masked out, the polarity ``GPTDataset`` produces."""
    mask = torch.tril(torch.ones(seq, seq, device=_device())) < 0.5
    return mask.view(1, 1, seq, seq).expand(batch, 1, seq, seq).contiguous()


def _batch(seq: int = SEQ, batch: int = BATCH):
    ids = torch.randint(1, VOCAB, (batch, seq), device=_device(), dtype=torch.long)
    labels = torch.roll(ids, shifts=-1, dims=1)
    loss_mask = torch.ones_like(labels)
    return ids, labels, loss_mask


def _mtp_spec_inner(config):
    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_mtp_specs import (
        get_kimi_k3_mtp_block_spec,
    )

    return (
        get_kimi_k3_mtp_block_spec(config).submodules.layer_specs[0].submodules.mtp_model_layer
    )


# ---------------------------------------------------------------------------
# Config reconciliation -- num_nextn_predict_layers stops being a no-op
# ---------------------------------------------------------------------------


def test_nextn_name_drives_mtp_num_layers():
    """``num_nextn_predict_layers`` is the K3-native name and must reach MTP."""
    config = _mtp_config(mtp_depths=1, use_nextn_name=True)
    assert config.mtp_num_layers == 1
    assert config.num_nextn_predict_layers == 1
    assert config.mtp_enabled is True


def test_mtp_num_layers_name_also_works_and_both_agree():
    config = _mtp_config(mtp_depths=2, use_nextn_name=False)
    assert config.num_nextn_predict_layers == 2
    assert config.mtp_num_layers == 2


def test_disagreeing_aliases_raise():
    config = _make_config()
    config.num_nextn_predict_layers = 1
    config.mtp_num_layers = 2
    with pytest.raises(ValueError, match="disagrees"):
        config._resolve_mtp_fields()


def test_zero_depths_is_normalised_to_none_not_left_as_zero():
    """``0`` is not "off" upstream, and that is the whole trap.

    ``mtp_on_this_rank`` tests ``config.mtp_num_layers is not None``
    (``multi_token_prediction.py:508``), so a literal 0 makes it return True on
    the last pipeline stage; ``get_mtp_num_layers_to_build`` then returns 0 and
    ``MultiTokenPredictionBlock.__init__`` asserts on the empty layer list. The
    failure lands at model construction with a message about layer counts,
    nowhere near the yaml key that caused it.
    """
    for nextn, mtp in ((0, None), (None, 0), (0, 0)):
        config = _make_config()
        config.num_nextn_predict_layers = nextn
        config.mtp_num_layers = mtp
        config._resolve_mtp_fields()
        assert config.mtp_num_layers is None, (nextn, mtp)
        assert config.num_nextn_predict_layers is None, (nextn, mtp)
        assert config.mtp_enabled is False


def test_negative_depths_rejected():
    config = _make_config()
    config.num_nextn_predict_layers = -1
    with pytest.raises(ValueError, match=">= 0"):
        config._resolve_mtp_fields()


def test_zero_loss_weight_is_rejected_because_it_kills_every_mtp_gradient():
    """A zero weight would build the MTP block and never train it.

    ``process_mtp_loss`` folds ``mtp_loss_scaling_factor / mtp_num_layers *
    loss`` into the tensor ``MTPLossAutoScaler`` differentiates, so the factor
    multiplies straight through to every MTP parameter's gradient.
    """
    config = _make_config()
    config.num_nextn_predict_layers = 1
    for bad in (0.0, None):
        config.mtp_loss_scaling_factor = bad
        with pytest.raises(ValueError, match="mtp_loss_scaling_factor"):
            config._resolve_mtp_fields()


def test_bad_mtp_layer_type_rejected():
    config = _make_config()
    config.mtp_layer_type = "transformer"
    with pytest.raises(ValueError, match="mtp_layer_type"):
        config._resolve_mtp_fields()


@pytest.mark.parametrize(
    "layer_type,expect_kda", [("kda", True), ("mla", False), ("mirror_last", False)]
)
def test_mtp_layer_type_resolution(layer_type, expect_kda):
    """``mirror_last`` follows the final backbone layer.

    The debug interleave is ``K K K F K K K F``, so the last layer is full
    attention -- which is the point: report §2.1 guarantees "an additional
    Gated MLA layer is placed at the end of the backbone", so mirroring the
    last layer is what turns §4.1.4's "mirrors the structure of a backbone
    layer" into a determinate choice.
    """
    assert _mtp_config(mtp_layer_type=layer_type).mtp_layer_is_kda() is expect_kda


def test_mirror_last_follows_a_kda_tail():
    """Not a constant: flip the tail and ``mirror_last`` flips with it."""
    config = _mtp_config(mtp_layer_type="mirror_last", kda_pattern=[1] * NUM_LAYERS)
    assert config.mtp_layer_is_kda() is True


# ---------------------------------------------------------------------------
# Spec construction
# ---------------------------------------------------------------------------


def test_mtp_block_spec_structure(mpu_tp1):
    from megatron.core.transformer.multi_token_prediction import (
        MultiTokenPredictionBlock,
        MultiTokenPredictionBlockSubmodules,
        MultiTokenPredictionLayer,
    )

    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_block import KimiK3Layer
    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_mtp_specs import (
        get_kimi_k3_mtp_block_spec,
    )

    spec = get_kimi_k3_mtp_block_spec(_mtp_config(mtp_depths=1))

    assert spec.module is MultiTokenPredictionBlock
    assert isinstance(spec.submodules, MultiTokenPredictionBlockSubmodules)
    assert len(spec.submodules.layer_specs) == 1
    layer_spec = spec.submodules.layer_specs[0]
    assert layer_spec.module is MultiTokenPredictionLayer
    assert layer_spec.submodules.mtp_model_layer.module is KimiK3Layer


@pytest.mark.parametrize("depths", [1, 2, 3])
def test_mtp_block_spec_has_one_layer_spec_per_depth(mpu_tp1, depths):
    """One spec per depth, and they must be distinct objects.

    ``MultiTokenPredictionBlock._build_layers`` builds one module per entry
    (``multi_token_prediction.py:1336-1343``), so a short list silently builds
    fewer depths than ``mtp_num_layers`` and then
    ``MultiTokenPredictionBlock.forward`` indexes off the end.
    """
    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_mtp_specs import (
        get_kimi_k3_mtp_block_spec,
    )

    specs = get_kimi_k3_mtp_block_spec(_mtp_config(mtp_depths=depths)).submodules.layer_specs
    assert len(specs) == depths
    assert len({id(s) for s in specs}) == depths


def test_mtp_block_spec_rejects_disabled_mtp(mpu_tp1):
    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_mtp_specs import (
        get_kimi_k3_mtp_block_spec,
    )

    with pytest.raises(ValueError, match="mtp_num_layers >= 1"):
        get_kimi_k3_mtp_block_spec(_mtp_config(mtp_depths=None))


def test_mtp_norm_slots_are_raw_classes_not_module_specs(mpu_tp1):
    """``MultiTokenPredictionLayer`` calls these three slots directly.

    ``self.enorm = self.submodules.enorm(config=..., hidden_size=..., eps=...)``
    (``multi_token_prediction.py:782-792``, ``:843-847``) -- no
    ``build_module``. A ``ModuleSpec`` would be *called* with those kwargs and
    return another spec, and the failure would surface much later as a missing
    ``forward``. The decoder tree's ``_build_norm_spec`` returns exactly the
    wrapper that must not be used here, so this is a live hazard.
    """
    from megatron.core.transformer.spec_utils import ModuleSpec

    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_mtp_specs import (
        get_kimi_k3_mtp_block_spec,
    )

    submodules = get_kimi_k3_mtp_block_spec(_mtp_config()).submodules.layer_specs[0].submodules
    for slot in ("enorm", "hnorm", "layer_norm"):
        value = getattr(submodules, slot)
        assert not isinstance(value, ModuleSpec), slot
        assert isinstance(value, type), (slot, type(value))


@pytest.mark.parametrize("layer_type", ["mla", "kda"])
def test_inner_attention_spec_declares_a_supported_attn_mask_type(mpu_tp1, layer_type):
    """Both attention variants must satisfy upstream's mask-type gate.

    ``MultiTokenPredictionLayer.__init__`` reads
    ``self_attention.params['attn_mask_type']`` and asserts membership in
    ``SUPPORTED_ATTN_MASK`` (``multi_token_prediction.py:773-780``).
    ``ModuleSpec.params`` defaults to ``{}`` (``spec_utils.py:29``), so the KDA
    spec had to start declaring it -- the same cross-cutting change DeepSeek-V4
    made at its P16.
    """
    from megatron.core.transformer.multi_token_prediction import SUPPORTED_ATTN_MASK

    inner = _mtp_spec_inner(_mtp_config(mtp_layer_type=layer_type))
    assert inner.submodules.self_attention.params.get("attn_mask_type") in SUPPORTED_ATTN_MASK


def test_inner_layer_submodules_extend_transformer_layer_submodules(mpu_tp1):
    """Otherwise upstream's validation takes its Mamba branch and raises.

    ``MultiTokenPredictionLayer.__init__:761-772`` isinstance-checks the inner
    spec's submodules against ``MambaStackSubmodules`` then
    ``TransformerLayerSubmodules``, and its ``else`` raises ``ValueError``.
    ``KimiK3LayerSubmodules`` already extends the latter; this pins it, because
    a refactor that made it a standalone dataclass would break MTP only.
    """
    from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules

    inner = _mtp_spec_inner(_mtp_config())
    assert isinstance(inner.submodules, TransformerLayerSubmodules)


def test_inner_layer_mirrors_the_last_backbone_layers_ffn(mpu_tp1):
    """MoE, not the dense ``situ`` MLP.

    ``first_k_dense_replace`` makes the *leading* layers dense, so reading the
    MoE pattern's head -- the obvious mistake -- would give the MTP layer a
    dense FFN on every real K3 shape.
    """
    from primus.backends.megatron.core.transformer.kimi_k3.moe.k3_stable_latent_moe import (
        StableLatentMoE,
    )

    assert _mtp_spec_inner(_mtp_config()).submodules.mlp.module is StableLatentMoE


def test_inner_layer_fills_the_activation_func_slot(mpu_tp1):
    """``use_te_activation_func`` makes the module slot the only live hook.

    With it empty, ``MLP.__init__`` falls back to ``config.activation_func`` --
    ``F.silu`` -- applied to the fused ``[gate | up]`` tensor, which is both the
    wrong activation and double the width ``linear_fc2`` expects. The decoder
    specs are already checked for this; the MTP layer is a new MLP site and
    would be the one place ``situ`` silently disappeared.
    """
    from primus.backends.megatron.core.transformer.kimi_k3.situ_activation import (
        SituActivation,
    )

    model = _build_model(_mtp_config())
    experts = model.mtp.layers[0].mtp_model_layer.mlp.experts
    assert isinstance(experts.activation_func, SituActivation), type(experts.activation_func)


# ---------------------------------------------------------------------------
# The attention-residual contract at the MTP boundary
# ---------------------------------------------------------------------------


def test_inner_layer_spec_disables_attention_residuals(mpu_tp1):
    """The spec must say so explicitly, not rely on the layer index.

    ``use_attn_residuals`` is otherwise derived from the shared
    ``config.attn_res_block_size``, which is set -- so without the override the
    MTP layer inherits the mechanism.
    """
    inner = _mtp_spec_inner(_mtp_config())
    assert inner.params["use_attn_residuals"] is False
    assert inner.submodules.attn_res_mixer is None
    assert inner.submodules.mlp_res_mixer is None
    # And ``is_mtp_layer`` must *not* be a spec param: upstream passes it as a
    # build_module keyword (``multi_token_prediction.py:835-841``) and
    # build_module unpacks params and kwargs into one call, so declaring it
    # here raises "got multiple values for keyword argument". The built layer
    # still has it set -- see
    # test_built_mtp_layer_has_no_mixers_and_appends_no_checkpoint.
    assert "is_mtp_layer" not in inner.params


def test_built_mtp_layer_has_no_mixers_and_appends_no_checkpoint(mpu_tp1):
    """Read off the built module, not the spec.

    ``appends_checkpoint`` is the quiet failure mode: on a shape where
    ``num_layers % attn_res_block_size == 0`` -- which this one is -- an MTP
    layer that kept the mechanism would land on an appending index and grow
    ``block_residual`` a ninth time *after* the head had already consumed it.
    Nothing would raise.
    """
    from primus.backends.megatron.core.transformer.kimi_k3.attention_residual import (
        AttentionResidualMixer,
    )

    # The premise of the paragraph above, asserted rather than assumed.
    assert NUM_LAYERS % ATTN_RES_BLOCK_SIZE == 0

    model = _build_model(_mtp_config())
    assert model.mtp_process is True
    mtp_layer = model.mtp.layers[0].mtp_model_layer

    assert mtp_layer.use_attn_residuals is False
    assert mtp_layer.appends_checkpoint is False
    assert mtp_layer.num_blocks_in == 0
    assert mtp_layer.attn_res_mixer is None
    assert mtp_layer.mlp_res_mixer is None
    assert mtp_layer.is_mtp_layer is True

    mixers = [
        name for name, m in model.mtp.named_modules() if isinstance(m, AttentionResidualMixer)
    ]
    assert mixers == [], mixers


def test_attn_res_head_stays_on_the_decoder_and_runs_once_before_mtp(mpu_tp1):
    """The head is the decoder's, and MTP reads its output.

    Report §2.2: "the final output layer then aggregates all N block
    representations" -- that aggregation is :class:`AttentionResidualHead`, it
    lives on the ``post_process`` stage only, and §4.1.4 says the MTP layer was
    pre-trained on the resulting high-level feature. So the order is
    ``head -> final_layernorm -> mtp`` and the head runs exactly once.
    """
    from primus.backends.megatron.core.transformer.kimi_k3.attention_residual import (
        AttentionResidualHead,
    )

    model = _build_model(_mtp_config())
    heads = [n for n, m in model.named_modules() if isinstance(m, AttentionResidualHead)]
    assert heads == ["decoder.attn_res_head"], heads

    trace: List[str] = []
    captured = {}
    head, norm = model.decoder.attn_res_head, model.decoder.final_layernorm
    orig_head, orig_norm, orig_mtp = head.forward, norm.forward, model.mtp.forward

    def traced_head(prefix_sum, block_residual, _o=orig_head):
        trace.append("head")
        return _o(prefix_sum, block_residual)

    def traced_norm(x, _o=orig_norm):
        trace.append("final_layernorm")
        captured["norm_out"] = _o(x)
        return captured["norm_out"]

    def traced_mtp(*args, _o=orig_mtp, **kwargs):
        trace.append("mtp")
        captured["mtp_in"] = kwargs["hidden_states"]
        return _o(*args, **kwargs)

    head.forward, norm.forward, model.mtp.forward = traced_head, traced_norm, traced_mtp
    try:
        ids, labels, loss_mask = _batch()
        model.train()
        model(ids, None, _causal_mask(), labels=labels, loss_mask=loss_mask)
    finally:
        head.forward, norm.forward, model.mtp.forward = orig_head, orig_norm, orig_mtp

    assert trace == ["head", "final_layernorm", "mtp"], trace
    # Value equality rather than identity: the block's closing
    # make_viewless_tensor may hand on a different python object with the same
    # storage (kimi_k3_block.py:829).
    assert torch.equal(captured["mtp_in"], captured["norm_out"])


def test_no_block_residual_crosses_into_the_mtp_layer(mpu_tp1):
    """Observed at the call, not inferred from construction."""
    model = _build_model(_mtp_config())
    assert _no_block_residual_at_mtp(model) is True


def test_decoder_checkpoint_trace_is_unchanged_by_enabling_mtp(mpu_tp1):
    """MTP must not perturb the backbone's append schedule at all."""
    without = _build_model(_mtp_config(mtp_depths=None))
    with_mtp = _build_model(_mtp_config(mtp_depths=1))

    assert without.decoder.attn_res_block_count_trace() == [0, 1, 1, 1, 1, 2, 2, 2]
    assert (
        with_mtp.decoder.attn_res_block_count_trace()
        == without.decoder.attn_res_block_count_trace()
    )
    assert [layer.appends_checkpoint for layer in with_mtp.decoder.layers] == [
        layer.appends_checkpoint for layer in without.decoder.layers
    ]


def test_mtp_layer_is_a_plain_residual_layer(mpu_tp1):
    """``x + attn(x)`` then ``x + mlp(x)``, i.e. it keeps its input.

    What the layer index alone would have produced instead: index 8 with
    ``attn_res_block_size=4`` appends, and appending *resets* ``prefix_sum`` to
    ``None`` (``kimi_k3_block.py:498-500``), dropping the incoming hidden state
    from the output stream. On an MTP layer, whose input is the whole point --
    it is ``eh_proj(cat(enorm(emb), hnorm(h)))`` -- that would silently make
    the layer non-residual.
    """
    config = _mtp_config()
    model = _build_model(config)
    layer = model.mtp.layers[0].mtp_model_layer

    hidden = torch.randn(SEQ, BATCH, HIDDEN, dtype=config.params_dtype, device=_device())
    captured = {}
    orig_attn, orig_mlp = layer.self_attention.forward, layer.mlp.forward

    def capture(*args, _o=orig_attn, **kwargs):
        out = _o(*args, **kwargs)
        captured["attn"] = out[0]
        return out

    layer.self_attention.forward = capture
    layer.mlp.forward = lambda x, *a, **k: (torch.zeros_like(x), None)
    try:
        out, blocks = layer(hidden, None)
    finally:
        layer.self_attention.forward = orig_attn
        layer.mlp.forward = orig_mlp

    assert blocks is None
    torch.testing.assert_close(out, hidden + captured["attn"], atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Structure / gradients
# ---------------------------------------------------------------------------


def _mtp_param_names(model) -> List[str]:
    return [
        name
        for name, p in model.named_parameters()
        if name.startswith("mtp.") and p.requires_grad
    ]


def _train_step(model):
    ids, labels, loss_mask = _batch()
    model.zero_grad(set_to_none=True)
    model.train()
    loss = model(ids, None, _causal_mask(), labels=labels, loss_mask=loss_mask)
    loss.float().mean().backward()
    return float(loss.detach().float().mean())


def test_mtp_block_is_built_and_carries_the_expected_parameters(mpu_tp1):
    model = _build_model(_mtp_config())
    names = _mtp_param_names(model)
    assert names, "no MTP parameters were created"
    for expected in ("enorm", "hnorm", "eh_proj", "mtp_model_layer", "final_layernorm"):
        assert any(expected in n for n in names), expected

    # eh_proj is [2h] -> [h] (multi_token_prediction.py:799-810).
    eh = model.mtp.layers[0].eh_proj
    assert tuple(eh.weight.shape) == (HIDDEN, 2 * HIDDEN), eh.weight.shape


@pytest.mark.parametrize("depths", [1, 2])
def test_mtp_router_layer_number_is_inside_the_aux_loss_tracker(mpu_tp1, depths):
    """The MTP router's ``layer_number`` must be 1-based *within the MTP block*.

    ``TopKRouter`` offsets an MTP layer's number by ``config.num_layers`` before
    indexing a tracker sized ``num_layers + mtp_num_layers``
    (``router.py:464-477`` -> ``moe_utils.py:953-957``). So depth ``d`` needs
    ``layer_number == d + 1``; anything else indexes past the end and raises
    ``IndexError`` from inside the router on the first MoE forward.

    ``KimiK3Layer`` would default ``layer_number`` to ``layer_idx + 1``, and the
    MTP layer's ``layer_idx`` is ``num_layers + depth`` -- which would be 9 and
    10 here, i.e. out of range for both depths. The value that saves it is the
    one ``MultiTokenPredictionLayer`` passes explicitly
    (``multi_token_prediction.py:749``, ``:839``), and ``_build_mlp`` forwarding
    it via ``set_layer_number``.
    """
    model = _build_model(_mtp_config(mtp_depths=depths))
    num_layers = int(model.config.num_layers)
    tracker_size = num_layers + depths

    for depth in range(depths):
        inner = model.mtp.layers[depth].mtp_model_layer
        assert inner.layer_number == depth + 1, (depth, inner.layer_number)
        # The layer index still continues the backbone's numbering, which is
        # what would have been used had layer_number not been passed.
        assert inner.layer_idx == num_layers + depth
        router = inner.mlp.router
        assert router.is_mtp_layer is True
        offset_index = router.layer_number + num_layers - 1
        assert 0 <= offset_index < tracker_size, (offset_index, tracker_size)

    # And it actually runs, which is the only proof the arithmetic above is the
    # arithmetic upstream does.
    _train_step(model)


def test_mtp_loss_gives_every_mtp_parameter_a_finite_gradient(mpu_tp1):
    """The load-bearing test.

    The MTP loss is folded into the *gradient* by ``MTPLossAutoScaler`` and
    never into the reported lm loss, so this is the only observable that
    distinguishes a wired MTP block from a decorative one.

    Routed-expert weights are held to a weaker bar than everything else: with
    top-2 of 8 experts over 128 tokens an individual expert can legitimately
    receive none, and grouped GEMM allocates an exactly-zero gradient for it
    rather than leaving ``.grad`` unset. So experts must be finite and
    non-``None``, and *some* expert must be non-zero; every other MTP
    parameter must be strictly non-zero.
    """
    model = _build_model(_mtp_config())
    _train_step(model)

    params = dict(model.named_parameters())
    names = _mtp_param_names(model)
    assert names

    missing = [n for n in names if params[n].grad is None]
    assert not missing, f"{len(missing)} MTP parameters got no gradient: {missing[:10]}"

    nonfinite = [n for n in names if not torch.isfinite(params[n].grad).all()]
    assert not nonfinite, f"non-finite MTP gradients: {nonfinite[:10]}"

    non_expert = [n for n in names if ".experts." not in n]
    zero = [n for n in non_expert if float(params[n].grad.float().abs().sum()) == 0.0]
    assert not zero, f"{len(zero)} MTP parameters got an exactly-zero gradient: {zero[:10]}"

    experts = [n for n in names if ".experts." in n]
    if experts:
        assert any(
            float(params[n].grad.float().abs().sum()) > 0.0 for n in experts
        ), "no routed expert in the MTP layer received any gradient"


def test_no_parameter_in_the_whole_model_is_left_without_a_gradient(mpu_tp1):
    """Turning MTP on must not orphan anything, MTP or backbone."""
    model = _build_model(_mtp_config())
    _train_step(model)

    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"{len(missing)} parameters got no gradient: {missing[:10]}"

    nonfinite = [
        n
        for n, p in model.named_parameters()
        if p.grad is not None and not torch.isfinite(p.grad).all()
    ]
    assert not nonfinite, f"non-finite gradients: {nonfinite[:10]}"


def test_reported_loss_is_the_main_loss_only(mpu_tp1):
    """MTP's contribution is invisible in the loss value, by design.

    ``process_mtp_loss`` returns the *main* chunk, so the returned lm loss is
    the same tensor it would be with MTP off; only the gradient differs. Stated
    explicitly so nobody later "fixes" the MTP loss into the reported number
    and breaks the loss-curve comparability the A/B in ``mtp/FINDINGS.md``
    depends on.
    """
    model = _build_model(_mtp_config())
    ids, labels, loss_mask = _batch()
    model.eval()
    with torch.no_grad():
        loss = model(ids, None, _causal_mask(), labels=labels, loss_mask=loss_mask)
        logits = model(ids, None, _causal_mask())
        manual = model.compute_language_model_loss(labels, logits.transpose(0, 1).contiguous())
    assert tuple(loss.shape) == (BATCH, SEQ)
    torch.testing.assert_close(loss.float(), manual.float(), atol=1e-4, rtol=1e-4)


def test_mtp_off_builds_no_mtp_attributes(mpu_tp1):
    """The disabled path must be genuinely inert.

    ``hasattr(model, 'mtp')`` is what Megatron's cudagraph
    ``set_current_microbatch`` probes before iterating ``model.mtp.layers``, so
    a ``self.mtp = None`` placeholder would crash cudagraph runs with MTP off.
    """
    model = _build_model(_mtp_config(mtp_depths=None))
    assert model.mtp_process is False
    assert model.mtp_block_spec is None
    assert not hasattr(model, "mtp")
    assert not _mtp_param_names(model)


@pytest.mark.parametrize("layer_type", ["mla", "kda"])
def test_both_mtp_layer_types_train(mpu_tp1, layer_type):
    """Both variants must run, not just build.

    KDA is the one that needed a constructor change to tolerate
    ``attn_mask_type``, so an untested ``kda`` option would be a latent
    ``TypeError``.
    """
    from primus.backends.megatron.core.transformer.kimi_k3.kimi_delta_attention import (
        KimiDeltaAttention,
    )
    from primus.backends.megatron.core.transformer.kimi_k3.kimi_k3_mla_attention import (
        KimiK3MLASelfAttention,
    )

    model = _build_model(_mtp_config(mtp_layer_type=layer_type))
    attention = model.mtp.layers[0].mtp_model_layer.self_attention
    expected = KimiDeltaAttention if layer_type == "kda" else KimiK3MLASelfAttention
    assert isinstance(attention, expected), type(attention)

    loss = _train_step(model)
    assert torch.isfinite(torch.tensor(loss))
    grad = model.mtp.layers[0].eh_proj.weight.grad
    assert grad is not None and float(grad.float().abs().sum()) > 0.0


@pytest.mark.parametrize("depths", [1, 2])
def test_multiple_depths_all_get_gradient(mpu_tp1, depths):
    model = _build_model(_mtp_config(mtp_depths=depths))
    assert len(model.mtp.layers) == depths
    _train_step(model)
    for depth in range(depths):
        grad = model.mtp.layers[depth].eh_proj.weight.grad
        assert grad is not None, depth
        assert float(grad.float().abs().sum()) > 0.0, depth


# ---------------------------------------------------------------------------
# Predicates -- the assertions the bug injections have to break
# ---------------------------------------------------------------------------


def _mtp_gradient_is_live(model) -> bool:
    """Same bar as :func:`test_mtp_loss_gives_every_mtp_parameter_a_finite_gradient`."""
    _train_step(model)
    params = dict(model.named_parameters())
    names = _mtp_param_names(model)
    if not names:
        return False
    for name in names:
        grad = params[name].grad
        if grad is None or not torch.isfinite(grad).all():
            return False
        if ".experts." not in name and float(grad.float().abs().sum()) == 0.0:
            return False
    return True


def _backbone_receives_mtp_gradient(model) -> bool:
    """Does the MTP loss reach the decoder?

    Measured by differencing rather than by inspection: run one step with the
    MTP depths live and one with their inputs detached, and compare a decoder
    parameter's gradient. If the MTP loss shapes the backbone the two differ.

    The probe is the last decoder layer's router weight -- a parameter the main
    loss certainly reaches, so a difference can only come from the MTP term.
    """
    probe = "decoder.layers.7.mlp.router.weight"
    params = dict(model.named_parameters())
    assert probe in params, sorted(n for n in params if "router" in n)[:5]

    torch.manual_seed(11)
    ids, labels, loss_mask = _batch()

    def grad_of(detach_mtp: bool):
        model.zero_grad(set_to_none=True)
        model.train()
        ctx = _inject_detached_mtp_layer_input(model) if detach_mtp else contextlib.nullcontext()
        with ctx:
            out = model(ids, None, _causal_mask(), labels=labels, loss_mask=loss_mask)
            out.float().mean().backward()
        grad = params[probe].grad
        assert grad is not None, (
            f"{probe} received no gradient at all (detach_mtp={detach_mtp}); the "
            "difference measurement would be meaningless"
        )
        return grad.detach().float().clone()

    return not torch.equal(grad_of(False), grad_of(True))


def _mtp_predicts_two_ahead(model) -> bool:
    """Is depth 0's loss scored against labels rolled one *further* than the main loss?

    This is an equality test, not a liveness test -- the lesson
    ``DECISIONS.md`` records from the router flag. The depth-0 hidden state is
    captured from the model's own forward, pushed through the model's own
    shared output layer and the model's own ``compute_language_model_loss``,
    and the resulting per-token loss is reduced exactly the way
    ``process_mtp_loss`` reduces it (``multi_token_prediction.py:675-710``).
    The only thing recomputed independently is **the shift**, which is the
    thing that can be wrong.

    Both candidate shifts are evaluated and the predicate requires a match to
    the two-ahead one *and* a mismatch to the one-ahead one, so it cannot pass
    by the two happening to coincide.
    """
    from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper

    depths = int(model.config.mtp_num_layers)
    captured = {}
    orig = model.mtp.forward

    def grab(*args, _o=orig, **kwargs):
        out = _o(*args, **kwargs)
        captured["chunks"] = list(torch.chunk(out, 1 + depths, dim=0))
        return out

    torch.manual_seed(17)
    ids, labels, loss_mask = _batch()

    MTPLossLoggingHelper.tracker.pop("values", None)
    model.mtp.forward = grab
    try:
        model.zero_grad(set_to_none=True)
        model.train()
        model(ids, None, _causal_mask(), labels=labels, loss_mask=loss_mask)
    finally:
        model.mtp.forward = orig

    logged = float(MTPLossLoggingHelper.tracker["values"][0])
    mtp_logits, _ = model.output_layer(captured["chunks"][1], weight=None)

    def reduced(shifted: bool) -> float:
        if shifted:
            lbl = torch.roll(labels, -1, dims=-1)
            msk = torch.roll(loss_mask, -1, dims=-1)
            lbl[:, -1] = 0
            msk[:, -1] = 0
        else:
            lbl, msk = labels, loss_mask
        with torch.no_grad():
            per_token = model.compute_language_model_loss(lbl, mtp_logits)
        return float((msk * per_token).sum() / msk.sum())

    two_ahead, one_ahead = reduced(True), reduced(False)
    # Self-check: if the two candidates coincide the predicate is vacuous.
    assert abs(two_ahead - one_ahead) > 1e-2, (two_ahead, one_ahead)
    return abs(logged - two_ahead) < 1e-3 and abs(logged - one_ahead) > 1e-2


def _no_block_residual_at_mtp(model) -> bool:
    """No checkpoint tensor enters or leaves the MTP layer."""
    layer = model.mtp.layers[0].mtp_model_layer
    seen = {}
    orig = layer.forward

    def traced(hidden_states, attention_mask=None, *, block_residual=None, _o=orig, **kwargs):
        out_hidden, out_blocks = _o(
            hidden_states, attention_mask, block_residual=block_residual, **kwargs
        )
        seen["arg"] = block_residual
        seen["ret"] = out_blocks
        return out_hidden, out_blocks

    layer.forward = traced
    try:
        ids, labels, loss_mask = _batch()
        model.train()
        model(ids, None, _causal_mask(), labels=labels, loss_mask=loss_mask)
    finally:
        layer.forward = orig

    if "arg" not in seen or seen["arg"] is not None:
        return False
    returned = seen["ret"]
    # The plain-residual branch hands the argument straight back; an appending
    # layer would return a grown tensor instead.
    return returned is None or returned.shape[-2] == 0


_PREDICATES = {
    "mtp_gradient": _mtp_gradient_is_live,
    "backbone_gradient": _backbone_receives_mtp_gradient,
    "two_ahead": _mtp_predicts_two_ahead,
    "no_block_residual": _no_block_residual_at_mtp,
}


# ---------------------------------------------------------------------------
# Bug injection
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _inject_detached_mtp_layer_input(model):
    """Cut each MTP depth off the decoder's graph.

    The realistic version of "someone added a ``.detach()`` to save memory":
    the block still runs and its own parameters still get gradient, but no MTP
    gradient reaches the backbone, so the auxiliary objective shapes nothing.

    Patched on the **depth layers**, not on ``model.mtp``. Detaching the
    block's own input would also detach the main path, because
    ``MultiTokenPredictionBlock.forward`` puts that same tensor in
    ``hidden_states_list[0]`` (``multi_token_prediction.py:1377-1378``) and
    ``process_mtp_loss`` then hands it to the LM head -- so the main loss would
    reach nothing either and the measurement would be about the wrong thing.
    """
    originals = [layer.forward for layer in model.mtp.layers]

    def make(layer, orig):
        def patched(*args, **kwargs):
            kwargs["hidden_states"] = kwargs["hidden_states"].detach()
            return orig(*args, **kwargs)

        return patched

    for layer, orig in zip(model.mtp.layers, originals):
        layer.forward = make(layer, orig)
    try:
        yield
    finally:
        for layer, orig in zip(model.mtp.layers, originals):
            layer.forward = orig


@contextlib.contextmanager
def _inject_dropped_mtp_chunks(model):
    """Return the main hidden state in every chunk slot.

    What a wrong ``torch.cat`` looks like: shapes stay right, so nothing
    raises, but ``process_mtp_loss`` scores the *main* state at every depth and
    the MTP layers are orphaned from the graph entirely.
    """
    orig = model.mtp.forward

    def patched(*args, _o=orig, **kwargs):
        out = _o(*args, **kwargs)
        depths = int(model.config.mtp_num_layers)
        main = torch.chunk(out, 1 + depths, dim=0)[0]
        return torch.cat([main] * (1 + depths), dim=0)

    model.mtp.forward = patched
    try:
        yield
    finally:
        model.mtp.forward = orig


@contextlib.contextmanager
def _inject_zero_loss_scale(model):
    """Set ``mtp_loss_scaling_factor`` to 0 behind the config validation's back.

    The config rejects this at construction; a runtime mutation is how it would
    actually happen -- a patch, a sweep script, a resumed checkpoint. It leaves
    every MTP parameter with an exactly-zero gradient rather than a missing
    one, which is why the gradient predicate tests for both.
    """
    orig = model.config.mtp_loss_scaling_factor
    model.config.mtp_loss_scaling_factor = 0.0
    try:
        yield
    finally:
        model.config.mtp_loss_scaling_factor = orig


@contextlib.contextmanager
def _inject_unrolled_mtp_labels(model):
    """Stop rolling, so depth 0 predicts t+1 instead of t+2.

    Not a gradient bug -- everything still gets gradient and the loss still
    falls -- it just makes MTP a duplicate of the main objective, which is the
    most expensive possible no-op.

    Patched in ``process_mtp_loss.__globals__`` rather than by ``setattr`` on
    the module: ``process_mtp_loss`` resolves ``roll_tensor`` from its own
    globals, and this reaches that namespace directly whichever module object
    the import machinery happened to bind.
    """
    from megatron.core.transformer.multi_token_prediction import process_mtp_loss

    namespace = process_mtp_loss.__globals__
    orig = namespace["roll_tensor"]

    def patched(tensor, shifts=-1, dims=-1, cp_group=None, packed_seq_params=None):
        return tensor, tensor.sum()

    namespace["roll_tensor"] = patched
    try:
        yield
    finally:
        namespace["roll_tensor"] = orig


@contextlib.contextmanager
def _inject_attn_residuals_on_the_mtp_layer(model):
    """Re-enable the mechanism on the built MTP layer.

    Exactly what forgetting ``use_attn_residuals=False`` in the spec would do.
    On this 8-layer / block-size-4 shape ``layer_idx = 8`` is an *appending*
    index, so the layer appends a spurious ninth checkpoint after the head has
    already consumed the set -- and resets ``prefix_sum``, dropping its own
    input. ``num_blocks_in`` is pinned to what the layer is actually handed so
    the drift assert stays quiet and the bug stays silent, which is the case
    worth testing.
    """
    layer = model.mtp.layers[0].mtp_model_layer
    saved = (
        layer.use_attn_residuals,
        layer.appends_checkpoint,
        layer.num_blocks_in,
        layer.mlp_res_mixer,
    )
    from primus.backends.megatron.core.transformer.kimi_k3.attention_residual import (
        AttentionResidualMixer,
    )

    layer.use_attn_residuals = True
    layer.appends_checkpoint = layer.layer_idx % layer.attn_res_block_size == 0
    layer.num_blocks_in = 0
    layer.mlp_res_mixer = AttentionResidualMixer(config=layer.config).to(
        next(layer.parameters()).device
    )
    try:
        yield
    finally:
        (
            layer.use_attn_residuals,
            layer.appends_checkpoint,
            layer.num_blocks_in,
            layer.mlp_res_mixer,
        ) = saved


def _injections():
    """``(name, context manager, which predicate must fail)``."""
    return [
        ("detached_mtp_layer_input", _inject_detached_mtp_layer_input, "backbone_gradient"),
        ("dropped_mtp_chunks", _inject_dropped_mtp_chunks, "mtp_gradient"),
        ("zero_loss_scale", _inject_zero_loss_scale, "mtp_gradient"),
        ("unrolled_mtp_labels", _inject_unrolled_mtp_labels, "two_ahead"),
        (
            "attn_residuals_on_mtp_layer",
            _inject_attn_residuals_on_the_mtp_layer,
            "no_block_residual",
        ),
    ]


@pytest.mark.parametrize("predicate_name", sorted(_PREDICATES))
def test_predicate_holds_on_the_clean_model(mpu_tp1, predicate_name):
    """Baseline half of the bug-injection argument.

    Each predicate has to pass on the real wiring, or its failure under
    injection would prove nothing.
    """
    torch.manual_seed(3)
    model = _build_model(_mtp_config())
    assert _PREDICATES[predicate_name](model) is True


@pytest.mark.parametrize(
    "name,injector,predicate_name", _injections(), ids=[c[0] for c in _injections()]
)
def test_injected_bug_is_caught(mpu_tp1, name, injector, predicate_name):
    """Break the wiring on purpose; the matching predicate must fail.

    A test that passes on broken code is worthless, and every failure mode here
    is silent: none of these five injections raises, and four of the five leave
    the reported lm loss numerically plausible.
    """
    torch.manual_seed(3)
    model = _build_model(_mtp_config())
    predicate = _PREDICATES[predicate_name]

    with injector(model):
        assert predicate(model) is False, (
            f"injection {name!r} was not detected by predicate {predicate_name!r}; "
            "the test has no discrimination power"
        )

    # And the model must be clean again once the injection is removed, so a
    # leaked patch cannot make a later test pass or fail spuriously.
    torch.manual_seed(3)
    assert predicate(model) is True


# ---------------------------------------------------------------------------
# Causality
# ---------------------------------------------------------------------------


def _causality_sweep(model, mask, ids, probes=(1, 17, 33, 63)):
    """Max ``|dlogit|`` strictly before ``t``, for each probe ``t``.

    ``validate/t2_causality.py``'s B2 certificate, narrowed to what this file
    needs. Returns ``(leaks_before_t, signals_at_and_after_t)``.
    """
    with torch.no_grad():
        base = model(ids, None, mask)
    leaks, signals = [], []
    for t in probes:
        ids2 = ids.clone()
        ids2[:, t] = (ids[:, t] + 137) % (VOCAB - 1) + 1
        with torch.no_grad():
            pert = model(ids2, None, mask)
        leaks.append((base[:, :t, :].float() - pert[:, :t, :].float()).abs().max().item())
        signals.append((base[:, t:, :].float() - pert[:, t:, :].float()).abs().max().item())
    return leaks, signals


@contextlib.contextmanager
def _injected_mtp_future_leak(model):
    """Leak 1 % of depth 0's hidden state into the main chunk.

    The realistic way MTP breaks causality: depth 0 legitimately depends on
    token ``t + 1`` (its ``input_ids`` are rolled one position left), so a
    ``process_mtp_loss`` that returned a contaminated main chunk instead of
    ``hidden_states_list[0]`` would hand the LM head genuine future
    information. 1 % is the subtlest realistic version, so passing bounds the
    sweep's sensitivity from below -- the convention
    ``validate/t4_causality_controls.py`` uses.
    """
    depths = int(model.config.mtp_num_layers)
    orig = model.mtp.forward

    def leaky(*args, _o=orig, **kwargs):
        out = _o(*args, **kwargs)
        chunks = list(torch.chunk(out, 1 + depths, dim=0))
        chunks[0] = chunks[0] + 0.01 * chunks[1]
        return torch.cat(chunks, dim=0)

    model.mtp.forward = leaky
    try:
        yield
    finally:
        model.mtp.forward = orig


def test_causality_is_bit_identical_with_mtp_enabled(mpu_tp1):
    """Perturbing the input at ``t`` leaves every logit at ``< t`` bit-identical.

    MTP is a real risk here rather than a formality:
    :class:`MultiTokenPredictionLayer` rolls ``input_ids`` one position **left**
    and embeds them, so position ``i`` of the MTP path legitimately sees token
    ``i + 1``. If that path leaked back into the main hidden states -- the
    tensor the LM head reads -- the model would be acausal in the one way a
    next-token objective cannot detect, because the loss would simply get
    better. ``process_mtp_loss`` returning ``hidden_states_list[0]`` is what
    keeps them separate, and ``MTPLossAutoScaler.forward`` is the identity, so
    the main logits must not move **at all**.

    Run with every FFN dense (``dense_ffn_only``, so no routed experts run).
    That is not a way of dodging a failure -- see
    :func:`test_causality_with_moe_residual_is_expert_grouping_noise`, which
    keeps the MoE and explains the 1e-6 that appears there -- it is what makes
    *bit*-identity a meaningful assertion: with MoE on, changing one token
    changes the global token-to-expert assignment and therefore the grouped
    GEMM's reduction order, which perturbs the arithmetic at every position
    without changing the mathematics at any of them.

    Everything MTP could break is still live here: attention residuals, the
    ``attn_res_head``, the MTP seam and the shared output layer.
    """
    torch.manual_seed(5)
    model = _build_model(_mtp_config(dense_ffn_only=True)).eval()
    ids, _, _ = _batch()

    leaks, signals = _causality_sweep(model, _causal_mask(), ids)
    assert max(leaks) == 0.0, f"future information leaked into the past: {leaks}"
    assert min(signals) > 0.0, f"the perturbation did nothing at or after t: {signals}"


def test_causality_sweep_catches_an_injected_future_leak(mpu_tp1):
    """Positive control for the sweep, at the MTP seam specifically.

    A sweep whose failure mode has never been demonstrated is not evidence.
    """
    torch.manual_seed(5)
    model = _build_model(_mtp_config(dense_ffn_only=True)).eval()
    ids, _, _ = _batch()

    with _injected_mtp_future_leak(model):
        leaks, _ = _causality_sweep(model, _causal_mask(), ids, probes=(17, 33, 63))
    assert max(leaks) > 0.0, (
        "the injected future leak was invisible to the causality sweep; "
        "test_causality_is_bit_identical_with_mtp_enabled proves nothing"
    )

    # Clean again once the injection is removed, so a leaked patch cannot make
    # a later test pass or fail spuriously.
    leaks_after, _ = _causality_sweep(model, _causal_mask(), ids, probes=(17, 33, 63))
    assert max(leaks_after) == 0.0, leaks_after


def test_causality_with_moe_residual_is_expert_grouping_noise(mpu_tp1):
    """With routed experts the pre-``t`` diff is ~1e-6, and it is not a leak.

    Diagnosed rather than tolerated. Changing ``input_ids[t]`` changes which
    expert token ``t`` routes to, which changes the permutation and the
    per-expert group sizes the grouped GEMM sees, which changes the reduction
    order for **every** token -- including those before ``t``. The value at
    each position is unchanged; only its rounding is.

    The test that establishes this is the third arm:
    ``moe_router_force_load_balancing`` makes ``TopKRouter`` discard its gating
    output and return reseeded ``normal_()`` draws (``router.py:696-698`` ->
    ``moe_utils.py:1177-1198``), so routing becomes independent of the token
    ids. If the residual really is grouping-order noise it must then vanish
    exactly -- and it does. (This is also why
    ``validate/t2_causality.py`` measured exactly 0.0 with MoE live: phase 1 ran
    with that flag on throughout.)

    The injected 1 % leak is two to three orders of magnitude above the
    residual, so the sweep still discriminates with MoE on.
    """
    from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed

    def sweep(force_load_balancing: bool, inject: bool):
        torch.manual_seed(5)
        config = _mtp_config()
        config.moe_router_force_load_balancing = force_load_balancing
        model = _build_model(config).eval()
        ids, _, _ = _batch()

        def reseed():
            torch.manual_seed(1234)
            torch.cuda.manual_seed_all(1234)
            model_parallel_cuda_manual_seed(1234)

        # RandomSTE draws from the expert-parallel RNG tracker, so two forwards
        # are only comparable if the tracker is reset between them.
        original = model.forward

        def reseeded(*args, _o=original, **kwargs):
            reseed()
            return _o(*args, **kwargs)

        model.forward = reseeded
        ctx = _injected_mtp_future_leak(model) if inject else contextlib.nullcontext()
        with ctx:
            leaks, _ = _causality_sweep(model, _causal_mask(), ids, probes=(17, 33, 63))
        return max(leaks)

    real_router = sweep(force_load_balancing=False, inject=False)
    ids_independent_router = sweep(force_load_balancing=True, inject=False)
    injected = sweep(force_load_balancing=False, inject=True)

    # 1e-4 is two orders below the injected signal and two above fp32 epsilon on
    # logits of this magnitude; the point of the assertion is the ordering, not
    # the constant.
    assert real_router < 1e-4, real_router
    assert ids_independent_router == 0.0, (
        "the residual did not vanish when routing was made independent of the "
        f"token ids, so it is not expert-grouping noise: {ids_independent_router:.3e}"
    )
    assert injected > 100 * max(real_router, 1e-12), (real_router, injected)


# ---------------------------------------------------------------------------
# FLOPs accounting
# ---------------------------------------------------------------------------


def test_flops_closed_form_charges_the_mtp_layer():
    """Without this the MTP arm's reported TFLOP/s is understated.

    ``mtp_num_layers`` used to affect only the logits term, so an MTP run paid
    for an extra LM head and nothing else -- while actually running a whole
    extra decoder layer. That flatters MTP in exactly the throughput comparison
    this work package reports.
    """
    from primus.backends.megatron.patches.kimi_k3_flops_patches import (
        compute_kimi_k3_flops,
    )

    off_total, off = compute_kimi_k3_flops(_flops_args(mtp_num_layers=None), 1)
    on_total, on = compute_kimi_k3_flops(_flops_args(mtp_num_layers=1), 1)

    assert off.mtp == 0
    assert off.num_mtp_layers == 0
    assert on.num_mtp_layers == 1
    assert on.mtp > 0
    assert on.logits == 2 * off.logits
    assert on_total > off_total
    # Every non-MTP component is untouched.
    for field in ("kda_proj", "kda_core", "mla", "dense_mlp", "moe", "attn_res"):
        assert getattr(on, field) == getattr(off, field), field

    # The MTP layer mirrors the last backbone layer -- MLA + MoE on this shape
    # -- so its body costs one such layer plus the [2h] -> [h] eh_proj.
    args = _flops_args(mtp_num_layers=1)
    tokens = int(args.seq_length) * 1
    hidden = int(args.hidden_size)
    one_mla_moe = (on.mla // on.num_full_attn_layers) + (on.moe // on.num_moe_layers)
    assert on.mtp == one_mla_moe + tokens * 2 * hidden * hidden, on.mtp


def test_flops_mtp_layer_type_selects_the_charged_layer():
    """``mtp_layer_type: kda`` must charge a KDA layer, not an MLA one."""
    from primus.backends.megatron.patches.kimi_k3_flops_patches import (
        compute_kimi_k3_flops,
    )

    _, mla = compute_kimi_k3_flops(_flops_args(mtp_num_layers=1, mtp_layer_type="mla"), 1)
    _, kda = compute_kimi_k3_flops(_flops_args(mtp_num_layers=1, mtp_layer_type="kda"), 1)
    _, mirror = compute_kimi_k3_flops(
        _flops_args(mtp_num_layers=1, mtp_layer_type="mirror_last"), 1
    )

    assert mla.mtp != kda.mtp
    # The debug interleave ends in a full-attention layer, so mirror_last == mla.
    assert mirror.mtp == mla.mtp


@pytest.fixture
def _silence_primus_logger(monkeypatch):
    """``log_rank_0`` needs a runtime-initialised global logger.

    Outside a real launcher ``primus.core.utils.logger._logger`` is ``None``
    and any log call raises. Same fixture as ``test_flops_closed_form.py``.
    """
    import primus.backends.megatron.patches.kimi_k3_flops_patches as mod

    monkeypatch.setattr(mod, "log_rank_0", lambda *a, **k: None)
    return mod


def test_flops_args_patch_mirrors_num_nextn_predict_layers(_silence_primus_logger):
    """``num_nextn_predict_layers`` has to reach the args layer too.

    ``training.py`` reads ``args.mtp_num_layers`` for FLOPs (``:347-351``),
    parameter counts (``:594-623``) and the per-depth MTP loss logging
    (``:2062-2065``), and never the config -- the same split ``DECISIONS.md``
    records for ``moe_latent_size``.
    """
    import types

    from primus.core.patches import PatchContext

    patch = _silence_primus_logger.patch_k3_args_mtp_num_layers

    def run(nextn, mtp):
        args = types.SimpleNamespace(num_nextn_predict_layers=nextn, mtp_num_layers=mtp)
        patch(PatchContext(backend="megatron", phase="build_args", extra={"backend_args": args}))
        return args.mtp_num_layers

    assert run(1, None) == 1
    assert run(2, 2) == 2
    # 0 is normalised away from either side, because 0 is not "off" upstream.
    assert run(0, None) is None
    assert run(None, 0) is None
    # Untouched when neither is set.
    assert run(None, None) is None

    args = types.SimpleNamespace(num_nextn_predict_layers=1, mtp_num_layers=2)
    with pytest.raises(ValueError, match="disagrees"):
        patch(PatchContext(backend="megatron", phase="build_args", extra={"backend_args": args}))
