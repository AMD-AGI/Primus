###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the OLMo-3 Megatron port (config, window, RoPE, QK-norm, converter)."""

import importlib.util
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

# The torch-free validation tests are in test_olmo3_validation.py (never skipped).
pytest.importorskip("megatron.core")

_REPO_ROOT = Path(__file__).resolve().parents[5]
_HOOK_DIR = _REPO_ROOT / "runner/helpers/hooks/train/posttrain/megatron"
_CONVERT_HOOK = _HOOK_DIR / "01_convert_checkpoints.py"
_INSTALL_HOOK = _HOOK_DIR / "00_install_requirements.sh"


def _cfg(**kw):
    from primus.backends.megatron.core.models.olmo3.olmo3_transformer_config import (
        Olmo3TransformerConfig,
    )

    base = dict(
        num_layers=4,
        hidden_size=256,
        num_attention_heads=4,
        num_query_groups=4,
        kv_channels=64,
        ffn_hidden_size=512,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        # the OLMo-3 settings (the shared refusal table rejects the TransformerConfig defaults)
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        add_bias_linear=False,
        olmo3_sliding_window=4096,
        window_attn_skip_freq=4,
        use_cpu_initialization=True,
    )
    base.update(kw)
    return Olmo3TransformerConfig(**base)


def test_yarn_attention_factor():
    from primus.backends.megatron.core.models.olmo3.olmo3_transformer_config import (
        yarn_attention_factor,
    )

    assert yarn_attention_factor(1.0) == 1.0
    assert yarn_attention_factor(8.0) == pytest.approx(0.1 * math.log(8.0) + 1.0)
    assert yarn_attention_factor(8.0, 1.5) == 1.5
    # HF applies the attention factor for rope_type yarn only: YaRN off -> 1.0
    assert yarn_attention_factor(1.0, 1.5) == 1.0


def test_config_window_tuple_and_mscale():
    cfg = _cfg(olmo3_yarn_attention_factor=1.2079441541679836)
    # window_size is Megatron / TE (W - 1, 0); the HF W is olmo3_sliding_window
    assert cfg.window_size == (4095, 0) and cfg.olmo3_sliding_window == 4096
    # neutral defaults: YaRN off, so the factor is not applied
    assert cfg.olmo3_yarn_rotary_scaling_factor == 1.0
    assert cfg.olmo3_yarn_mscale == 1.0
    cfg = _cfg(olmo3_yarn_rotary_scaling_factor=8.0, olmo3_yarn_original_max_position_embeddings=8192)
    assert cfg.olmo3_yarn_mscale == pytest.approx(0.1 * math.log(8.0) + 1.0)
    with pytest.raises(ValueError, match="original_max_position_embeddings"):
        _cfg(olmo3_yarn_rotary_scaling_factor=8.0)


def test_explicit_yarn_attention_factor_that_disagrees_is_used_and_warned(caplog):
    """An explicit olmo3_yarn_attention_factor equal to 0.1*ln(factor)+1 is silent; one that
    differs (a stale value after a factor override) is used, with a warning naming both."""
    yarn = dict(olmo3_yarn_original_max_position_embeddings=8192)
    with caplog.at_level("WARNING"):
        cfg = _cfg(
            olmo3_yarn_rotary_scaling_factor=8.0, olmo3_yarn_attention_factor=1.2079441541679836, **yarn
        )
    assert cfg.olmo3_yarn_mscale == pytest.approx(1.2079441541679836) and "differs" not in caplog.text
    with caplog.at_level("WARNING"):
        cfg = _cfg(
            olmo3_yarn_rotary_scaling_factor=4.0, olmo3_yarn_attention_factor=1.2079441541679836, **yarn
        )
    assert cfg.olmo3_yarn_mscale == pytest.approx(1.2079441541679836)
    assert "differs from 0.1*ln(factor)+1" in caplog.text


def test_layer_types_match_hf_pattern():
    from primus.backends.megatron.core.transformer.olmo3.olmo3_attention import (
        olmo3_layer_is_sliding,
        olmo3_te_window,
    )

    cfg = _cfg(num_layers=8)
    hf = ["full_attention" if (i + 1) % 4 == 0 else "sliding_attention" for i in range(8)]
    ours = ["sliding_attention" if olmo3_layer_is_sliding(cfg, i + 1) else "full_attention" for i in range(8)]
    assert ours == hf
    # HF sliding_window W -> query sees W keys (itself + W-1 back) -> TE (W-1, 0)
    assert olmo3_te_window(4096) == (4095, 0)


def test_full_width_rmsnorm_matches_hf():
    from primus.backends.megatron.core.transformer.olmo3.olmo3_attention import (
        FullWidthRMSNorm,
    )

    torch.manual_seed(0)
    heads, hd = 4, 8
    norm = FullWidthRMSNorm(heads * hd, heads * hd, 1e-6)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(heads * hd))
    x = torch.randn(5, 2, heads, hd, dtype=torch.bfloat16)
    flat = x.reshape(5, 2, -1).float()
    ref = flat * torch.rsqrt(flat.pow(2).mean(-1, keepdim=True) + 1e-6)
    ref32 = (norm.weight * ref).reshape_as(x)
    got = norm(x)
    assert got.dtype == torch.bfloat16
    # against the fp32 reference, with bf16 output rounding as the only error
    torch.testing.assert_close(got.float(), ref32, atol=0, rtol=2**-8)
    # and bit-equal to HF Olmo3RMSNorm, which rounds the fp32 result to the input dtype once
    assert torch.equal(got, ref32.to(torch.bfloat16))


_HF_CFG = {
    "model_type": "olmo3",
    "hidden_act": "silu",
    "attention_bias": False,
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 32,
    "num_hidden_layers": 32,
    "intermediate_size": 11008,
    "max_position_embeddings": 65536,
    "rms_norm_eps": 1e-6,
    "rope_theta": 500000,
    "sliding_window": 4096,
    "tie_word_embeddings": False,
    "layer_types": ["full_attention" if (i + 1) % 4 == 0 else "sliding_attention" for i in range(32)],
    "rope_scaling": {
        "rope_type": "yarn",
        "factor": 8.0,
        "original_max_position_embeddings": 8192,
        "beta_fast": 32,
        "beta_slow": 1,
        "attention_factor": 1.2079441541679836,
    },
}


def test_converter_fields_from_hf_config():
    from primus.backends.megatron.checkpoint.native_loader_olmo3 import (
        derive_megatron_fields,
    )

    f = derive_megatron_fields(dict(_HF_CFG), rope_layout="per_type")
    assert f["window_size"] == (4095, 0) and f["olmo3_sliding_window"] == 4096
    assert f["window_attn_skip_freq"] == 4
    assert f["kv_channels"] == 128
    assert f["rotary_base"] == 500000.0
    assert f["olmo3_yarn_rotary_scaling_factor"] == 8.0
    assert f["olmo3_yarn_original_max_position_embeddings"] == 8192
    assert f["olmo3_yarn_attention_factor"] == pytest.approx(1.2079441541679836)
    assert f["untie_embeddings_and_output_weights"] is True
    assert f["olmo3_rope_layout"] == "per_type"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="rotary tables live on GPU")
def test_dual_rope_matches_hf_formulas():
    from primus.backends.megatron.core.models.olmo3.olmo3_rotary import (
        Olmo3DualRotaryEmbedding,
    )

    cfg = _cfg(olmo3_yarn_rotary_scaling_factor=8.0, olmo3_yarn_original_max_position_embeddings=8192)
    rope = Olmo3DualRotaryEmbedding(cfg, rotary_base=500000.0)
    emb = rope(16)
    assert emb.shape == (2, 16, 1, 1, 64)
    dim, base = 64, 500000.0
    pos_freq = base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    inv = 1.0 / pos_freq
    pos = torch.arange(16, dtype=torch.float32)
    plain = torch.outer(pos, inv)
    torch.testing.assert_close(emb[0, :, 0, 0, : dim // 2].cpu(), plain)

    # HF _compute_yarn_parameters
    def corr(n):
        return (dim * math.log(8192 / (n * 2 * math.pi))) / (2 * math.log(base))

    low, high = max(math.floor(corr(32)), 0), min(math.ceil(corr(1)), dim - 1)
    ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low)).clamp(0, 1)
    extra = 1.0 - ramp
    hf_inv = (1.0 / (8.0 * pos_freq)) * (1 - extra) + inv * extra
    torch.testing.assert_close(emb[1, :, 0, 0, : dim // 2].cpu(), torch.outer(pos, hf_inv))


# ---------------------------------------------------------------------------
# RoPE layout resolver
# ---------------------------------------------------------------------------
def test_rope_layout_default_and_explicit():
    from primus.backends.megatron.core.models.olmo3.olmo3_transformer_config import (
        resolve_olmo3_rope_layout,
    )

    assert resolve_olmo3_rope_layout() == "per_type"
    with pytest.raises(ValueError, match="got None"):
        resolve_olmo3_rope_layout(None)  # never implicit
    assert resolve_olmo3_rope_layout("yarn_all") == "yarn_all"
    assert _cfg().olmo3_rope_layout == "per_type"
    with pytest.raises(ValueError, match="rope_layout"):
        resolve_olmo3_rope_layout("bogus")


def test_rope_layout_auto_versions(monkeypatch):
    """auto: too old for Olmo3 and a pre-release at a bound are refused; the rest map to a layout."""
    transformers = pytest.importorskip("transformers")

    from primus.backends.megatron.core.models.olmo3.olmo3_transformer_config import (
        resolve_olmo3_rope_layout,
    )

    for ver, want in (
        ("4.57.0", "per_type"),
        ("5.10.0", "yarn_all"),
        ("5.0.0", "yarn_all"),
        ("5.13.1", "per_type"),
        ("5.10.0.dev0", "yarn_all"),
        ("4.58.0rc1", "per_type"),
    ):
        monkeypatch.setattr(transformers, "__version__", ver)
        assert resolve_olmo3_rope_layout("auto") == want, ver
    for ver, why in (
        ("4.55.0", "no Olmo3"),
        ("5.0.0rc1", "pre-release"),
        ("5.13.0.dev0", "pre-release"),
        ("garbage", "cannot parse"),
    ):
        monkeypatch.setattr(transformers, "__version__", ver)
        with pytest.raises(ValueError, match=why):
            resolve_olmo3_rope_layout("auto")


def test_attention_layer_picks_rope_table():
    from primus.backends.megatron.core.transformer.olmo3.olmo3_attention import (
        Olmo3SelfAttention,
    )

    emb = torch.stack([torch.zeros(3, 1, 1, 8), torch.ones(3, 1, 1, 8)])
    for use_yarn, idx in ((False, 0), (True, 1)):
        layer = SimpleNamespace(olmo3_use_yarn=use_yarn)
        assert torch.equal(Olmo3SelfAttention._select_rotary(layer, emb), emb[idx])
    layer = SimpleNamespace(olmo3_use_yarn=True)
    for bad in (emb[0], (emb, emb), torch.zeros(3, 2, 1, 1, 8)):
        with pytest.raises(ValueError, match="stacked"):
            Olmo3SelfAttention._select_rotary(layer, bad)
    assert Olmo3SelfAttention._select_rotary(layer, None) is None


# ---------------------------------------------------------------------------
# Args patch
# ---------------------------------------------------------------------------
def _run_args_patch(**kw):
    from primus.backends.megatron.patches.olmo3_args_patches import patch_olmo3_args
    from primus.core.patches import PatchContext

    if "olmo3_sliding_window" not in kw:
        kw.setdefault("olmo3_window_size_form", "hf")  # as the model preset
    params = SimpleNamespace(model_type="olmo3", **kw)
    ctx = PatchContext(
        backend="megatron", phase="before_train", extra={"module_config": SimpleNamespace(params=params)}
    )
    patch_olmo3_args(ctx)
    return params


def test_preset_window_override_is_not_reinterpreted():
    """The preset states the window once (window_size [4096, 0], form hf; no olmo3_sliding_window):
    the args patch makes it (4095, 0) + W 4096, and an override of window_size alone changes the
    window. A converted value that disagrees with a W that is also set is still refused."""
    import yaml

    preset = yaml.safe_load(open("primus/configs/models/megatron/olmo_3_1025_7B.yaml"))
    assert "olmo3_sliding_window" not in preset and preset["window_size"] == [4096, 0]
    assert preset["olmo3_window_size_form"] == "hf"
    # derived from the factor (0.1 * ln(8) + 1 = the HF attention_factor), so a factor override moves it
    assert "olmo3_yarn_attention_factor" not in preset and preset["olmo3_yarn_rotary_scaling_factor"] == 8.0
    # run switches live in the example configs, not in the model preset
    assert "log_reset_first_iteration" not in preset and "keep_absolute_save_path" not in preset
    assert preset["norm_epsilon"] == preset["layernorm_epsilon"] == 1e-6
    base = dict(olmo3_window_size_form="hf", window_attn_skip_freq=4)
    a = _run_args_patch(**{**base, "window_size": [4096, 0]})
    assert a.window_size == (4095, 0) and a.olmo3_window_size_form == "megatron"
    assert a.olmo3_sliding_window == 4096
    a = _run_args_patch(**{**base, "window_size": [2048, 0]})
    assert a.window_size == (2047, 0) and a.olmo3_sliding_window == 2048
    for bad in ([4095, 0], "(4095, 0)", [2048, 0]):
        with pytest.raises(ValueError, match="disagrees"):
            _run_args_patch(**{**base, "olmo3_sliding_window": 4096, "window_size": bad})
    a = _run_args_patch(**{**base, "olmo3_sliding_window": 2048, "window_size": None})
    assert a.window_size == (2047, 0) and a.olmo3_sliding_window == 2048


def test_args_patch_window_size_forms():
    """The HF (W, 0) of the preset becomes Megatron (W - 1, 0) + olmo3_sliding_window W."""
    for ws in ([4096, 0], "(4096, 0)", (4096.0, 0)):
        a = _run_args_patch(window_size=ws, window_attn_skip_freq=4)
        assert a.window_size == (4095, 0) and a.olmo3_sliding_window == 4096
    a = _run_args_patch(window_size=(4096, 0), window_attn_skip_freq=[1, 1, 1, 0], num_layers=4)
    assert a.window_size == (4095, 0) and a.window_attn_skip_freq == [1, 1, 1, 0]
    # idempotent: a second run (or reloaded checkpoint args) keeps the converted value
    a2 = _run_args_patch(window_size=a.window_size, olmo3_sliding_window=4096, window_attn_skip_freq=4)
    assert a2.window_size == (4095, 0) and a2.olmo3_sliding_window == 4096
    # Megatron's --window-attn-skip-freq string forms (as moe_freq_type)
    assert _run_args_patch(window_size=[4096, 0], window_attn_skip_freq="4").window_attn_skip_freq == 4
    a = _run_args_patch(window_size=[4096, 0], window_attn_skip_freq="([1]*3+[0])*2", num_layers=8)
    assert a.window_attn_skip_freq == [1, 1, 1, 0] * 2


@pytest.mark.parametrize(
    "kw",
    [
        dict(window_size=None, window_attn_skip_freq=4),
        dict(window_size="", window_attn_skip_freq=4),
        dict(window_size="  ", window_attn_skip_freq=4),
        dict(window_size="()", window_attn_skip_freq=4),
        dict(window_size=[4096], window_attn_skip_freq=4),
        dict(window_size=[4096, 5], window_attn_skip_freq=4),
        dict(window_size=[0, 0], window_attn_skip_freq=4),
        dict(window_size=[-1, 0], window_attn_skip_freq=4),
        dict(window_size=[1, 0], window_attn_skip_freq=4),
        dict(window_size=[4096, 0]),
        dict(window_size=[4096, 0], window_attn_skip_freq=None),
        dict(window_size=[4096, 0], window_attn_skip_freq=1),
        dict(window_size=[4096, 0], window_attn_skip_freq=[0, 0, 0, 0], num_layers=4),
        dict(window_size=[4096, 0], window_attn_skip_freq=[1, 1, 0], num_layers=4),
        dict(window_size=[4096, 0], window_attn_skip_freq=4, context_parallel_size=2, cp_comm_type="p2p"),
        # non-integer windows are not truncated
        dict(window_size=[4096.7, 0], window_attn_skip_freq=4),
        dict(window_size="(4096.5, 0)", window_attn_skip_freq=4),
        dict(window_size=[True, 0], window_attn_skip_freq=4),
        # window_size disagrees with olmo3_sliding_window
        dict(window_size=[4000, 0], olmo3_sliding_window=4096, window_attn_skip_freq=4),
        # bad skip-freq strings
        dict(window_size=[4096, 0], window_attn_skip_freq="([1]*3+[0])*8", num_layers=4),
        dict(window_size=[4096, 0], window_attn_skip_freq="__import__('os')"),
        dict(window_size=[4096, 0], window_attn_skip_freq="[1,2,0,1]", num_layers=4),
    ],
)
def test_args_patch_refuses_bad_window(kw):
    with pytest.raises(ValueError, match="OLMo-3|olmo3|window"):
        _run_args_patch(**kw)


def test_args_patch_cp_comm_type():
    _run_args_patch(
        window_size=[4096, 0], window_attn_skip_freq=4, context_parallel_size=2, cp_comm_type="a2a"
    )
    _run_args_patch(
        window_size=[4096, 0], window_attn_skip_freq=4, context_parallel_size=2, cp_comm_type=["all_gather"]
    )
    _run_args_patch(
        window_size=[4096, 0], window_attn_skip_freq=4, context_parallel_size=1, cp_comm_type="p2p"
    )


def test_te_window_conversion():
    from primus.backends.megatron.core.transformer.olmo3.olmo3_attention import (
        olmo3_te_window,
    )

    assert olmo3_te_window(4096) == (4095, 0)
    assert olmo3_te_window(2) == (1, 0)
    # one rule with olmo3_window_from_settings: YAML / CLI forms of an int are the int
    assert olmo3_te_window("4096") == olmo3_te_window(4096.0) == (4095, 0)
    for bad in (None, (4096, 0), 0, -1, 1, 4096.5, True, "abc"):
        with pytest.raises(ValueError, match="window"):
            olmo3_te_window(bad)


def test_args_patch_turbo_attention_only():
    # Turbo GEMMs etc. are fine; only the Turbo attention kernel lacks sliding windows.
    w = dict(window_size=[4096, 0], window_attn_skip_freq=4)
    _run_args_patch(**w, enable_primus_turbo=True, use_turbo_attention=False)
    with pytest.raises(ValueError, match="use_turbo_attention"):
        _run_args_patch(**w, enable_primus_turbo=True, use_turbo_attention=True)
    with pytest.raises(ValueError, match="qk_layernorm"):
        _run_args_patch(**w, qk_layernorm=True)


def test_switch_settings_refuse_typos():
    """A switch setting is a bool, 0 / 1 or true / false / yes / no / on / off (any case); a typo is
    refused with the setting's message instead of being read as on (bool("ture")) or off."""
    from primus.backends.megatron.core.models.olmo3 import olmo3_validation as v

    for on in (True, 1, "true", " Yes ", "ON", "1"):
        assert v._truthy(on) is True
    for off in (False, 0, None, "", "null", "~", "false", "No", "off", "0"):
        assert v._truthy(off) is False
    for bad in ("ture", "flase", "enabled", 2, 0.5):
        with pytest.raises(ValueError):
            v._truthy(bad)
    w = dict(window_size=[4096, 0], window_attn_skip_freq=4)
    with pytest.raises(ValueError, match="qk_layernorm"):
        _run_args_patch(**w, qk_layernorm="flase")
    with pytest.raises(ValueError, match="swiglu"):
        _run_args_patch(**w, swiglu="ture")
    _run_args_patch(**w, qk_layernorm="off", swiglu="yes")


def test_args_patch_logs_conversion_only_when_it_happens(monkeypatch):
    from primus.backends.megatron.patches import olmo3_args_patches as ap

    logs = []
    monkeypatch.setattr(ap, "log_rank_0", logs.append)
    _run_args_patch(window_size=[4096, 0], window_attn_skip_freq=4)
    assert "window_size [4096, 0] -> (4095, 0)" in logs[-1]
    _run_args_patch(
        window_size=[4095, 0],
        olmo3_sliding_window=4096,
        olmo3_window_size_form="megatron",
        window_attn_skip_freq=4,
    )
    assert "->" not in logs[-1] and "already Megatron form" in logs[-1]


def test_args_patch_accepts_neutral_defaults():
    _run_args_patch(
        window_size=[4096, 0],
        window_attn_skip_freq=4,
        qk_layernorm=False,
        num_experts=None,
        rotary_percent=1.0,
        rotary_interleaved=False,
        rotary_seq_len_interpolation_factor=None,
        use_rope_scaling=False,
        fused_single_qkv_rope=False,
        attention_output_gate=False,
        flash_decode=False,
        transformer_impl="transformer_engine",
        olmo3_rope_layout="yarn_all",
    )


@pytest.mark.parametrize(
    "kw",
    [
        dict(fused_single_qkv_rope=True),
        dict(attention_output_gate=True),
        dict(rotary_interleaved=True),
        dict(window_size=(4000, 0)),  # disagrees with olmo3_sliding_window 4096
        dict(olmo3_sliding_window=1),
        dict(olmo3_sliding_window=None, window_size=None),
        dict(window_attn_skip_freq=None),
    ],
)
def test_config_refuses_unsupported(kw):
    with pytest.raises(ValueError, match="OLMo-3|olmo3"):
        _cfg(**kw)


def test_config_window_needs_marker_and_te_form():
    """The config reads window_size only in the form olmo3_window_size_form declares (unset =
    megatron) and needs olmo3_sliding_window W; it ends in the Megatron form."""
    assert _cfg(window_size=(4095, 0)).window_size == (4095, 0)
    assert _cfg(window_size=[4095, 0]).window_size == (4095, 0)
    assert _cfg().window_size == (4095, 0)  # derived from the marker
    cfg = _cfg(window_size=[4096, 0], olmo3_window_size_form="hf")
    assert (cfg.window_size, cfg.olmo3_window_size_form) == ((4095, 0), "megatron")
    # the config needs olmo3_sliding_window W (set by the args patch / converter)
    with pytest.raises(ValueError, match="olmo3_sliding_window"):
        _cfg(olmo3_sliding_window=None, window_size=[4095, 0])
    # HF form next to the marker without form: hf (the old W -> 4097 confusion): refused
    with pytest.raises(ValueError, match="disagrees"):
        _cfg(window_size=[4096, 0])
    # the Megatron form declared as hf: refused
    with pytest.raises(ValueError, match="window_size"):
        _cfg(window_size=(4095, 0), olmo3_window_size_form="hf")


def test_window_form_never_guessed():
    """A window_size without the marker is never read as HF by default: a lost marker on a
    converted (4095, 0) must not silently become a 4095-key window."""
    from primus.backends.megatron.core.models.olmo3.olmo3_validation import (
        olmo3_window_from_settings as f,
    )

    with pytest.raises(ValueError, match="ambiguous"):
        f((4095, 0), None)
    with pytest.raises(ValueError, match="ambiguous"):
        f("[4095, 0]", None, "  ")
    assert f((4096, 0), None, "hf") == ((4095, 0), 4096)
    assert f((4095, 0), None, "megatron") == ((4095, 0), 4096)
    assert f((4095, 0), 4096) == ((4095, 0), 4096)
    assert f((4095, 0), 4096, "megatron") == ((4095, 0), 4096)
    assert f((4096, 0), 4096, "hf") == ((4095, 0), 4096)
    assert f(None, 4096, "hf") == ((4095, 0), 4096)
    for ws, w, form in (((4096, 0), 4096, None), ((4096, 0), 4096, "megatron"), ((4095, 0), 4096, "hf")):
        with pytest.raises(ValueError, match="disagrees"):
            f(ws, w, form)
    with pytest.raises(ValueError, match="olmo3_window_size_form"):
        f((4096, 0), None, "te")
    # args patch: a converted window whose marker was lost (use_checkpoint_args, --olmo3_sliding_window null)
    with pytest.raises(ValueError, match="ambiguous"):
        _run_args_patch(window_size=[4095, 0], window_attn_skip_freq=4, olmo3_window_size_form=None)
    # the patch output declares its form, so it is idempotent even if the marker is then dropped
    a = _run_args_patch(window_size=[4096, 0], window_attn_skip_freq=4)
    assert (a.window_size, a.olmo3_sliding_window, a.olmo3_window_size_form) == ((4095, 0), 4096, "megatron")
    a.olmo3_sliding_window = None
    from primus.backends.megatron.patches.olmo3_args_patches import patch_olmo3_args
    from primus.core.patches import PatchContext

    patch_olmo3_args(
        PatchContext(
            backend="megatron", phase="before_train", extra={"module_config": SimpleNamespace(params=a)}
        )
    )
    assert (a.window_size, a.olmo3_sliding_window) == ((4095, 0), 4096)


# ---------------------------------------------------------------------------
# shared validation (olmo3_validation): window contract, skip_freq, one refusal table
# ---------------------------------------------------------------------------
def test_args_patch_hf_form_next_to_marker_raises():
    """(W, 0) with olmo3_sliding_window=W and no declared form is ambiguous (HF or a TE window of
    W + 1 keys): refused. With olmo3_window_size_form: hf it is the HF form of the same W."""
    with pytest.raises(ValueError, match="disagrees"):
        _run_args_patch(window_size=[4096, 0], olmo3_sliding_window=4096, window_attn_skip_freq=4)
    a = _run_args_patch(
        window_size=[4096, 0], olmo3_sliding_window=4096, window_attn_skip_freq=4, olmo3_window_size_form="hf"
    )
    assert a.window_size == (4095, 0)
    # the marker alone (window_size unset) is enough
    a = _run_args_patch(window_size=None, olmo3_sliding_window=4096, window_attn_skip_freq=4)
    assert a.window_size == (4095, 0)
    # CLI string marker
    a = _run_args_patch(window_size="(4095, 0)", olmo3_sliding_window="4096", window_attn_skip_freq="4")
    assert a.window_size == (4095, 0) and a.olmo3_sliding_window == 4096 and a.window_attn_skip_freq == 4


def test_args_patch_and_config_agree_on_window():
    """The args patch output builds a config with the same window (no 4097)."""
    a = _run_args_patch(window_size=[4096, 0], window_attn_skip_freq=4, num_layers=4)
    cfg = _cfg(
        window_size=a.window_size,
        olmo3_sliding_window=a.olmo3_sliding_window,
        window_attn_skip_freq=a.window_attn_skip_freq,
    )
    assert cfg.window_size == (4095, 0) and cfg.olmo3_sliding_window == 4096


@pytest.mark.parametrize(
    "skip, layers",
    [
        ([1, 1, 1, 1], 4),  # no full-attention layer
        ([0, 0, 0, 0], 4),  # no sliding layer
        (5, 4),  # int > num_layers: no full-attention layer
        (9, 8),
        ("abc", 4),
        ("**", 4),
        ("[1]*10**10", 4),
        ("[1]*100000000", 4),
        ("-" * 100 + "4", 4),  # nested too deeply
        ("[1,1,1,0]*", 4),
        ("[[1],0]", 4),
        ("[1]*[0]", 4),
        ("4.5", 4),
        ("x" * 5000, 4),
        (True, 4),
        (1, 4),
        (0, 4),
        (-4, 4),
        ("-4", 4),
        (4.0, 8),  # floats are refused, as in a list
        ("4.0", 8),
        ([1.0, 1.0, 1.0, 0.0], 4),
        ("[1,1,1,0]*-1", 4),
        ([1, 1, 2, 0], 4),
        ([True, True, True, False], 4),
    ],
)
def test_skip_freq_refuses(skip, layers):
    from primus.backends.megatron.core.models.olmo3.olmo3_validation import (
        parse_olmo3_skip_freq,
    )

    with pytest.raises(ValueError, match="window_attn_skip_freq"):
        parse_olmo3_skip_freq(skip, layers)
    with pytest.raises(ValueError, match="window_attn_skip_freq"):
        _run_args_patch(window_size=[4096, 0], window_attn_skip_freq=skip, num_layers=layers)
    with pytest.raises(ValueError, match="window_attn_skip_freq"):
        _cfg(num_layers=layers, window_attn_skip_freq=skip)


@pytest.mark.parametrize(
    "skip, layers, want",
    [
        (4, 32, 4),
        ("4", 8, 4),
        (" 4 ", 8, 4),
        ("[1,1,1,0,]*2", 8, [1, 1, 1, 0] * 2),  # trailing comma
        ("[1, 0]", 2, [1, 0]),
        (8, 8, 8),  # == num_layers: one full layer
        ("([1]*3+[0])*8", 32, [1, 1, 1, 0] * 8),
        ("[1,1,1,0]*2", 8, [1, 1, 1, 0] * 2),
        ("2*[1,0]", 4, [1, 0, 1, 0]),
        ([1, 1, 1, 0], 4, [1, 1, 1, 0]),
        ((1, 0), 2, [1, 0]),
        ("[1]*(1+2)+[0]", 4, [1, 1, 1, 0]),
    ],
)
def test_skip_freq_accepts(skip, layers, want):
    from primus.backends.megatron.core.models.olmo3.olmo3_validation import (
        parse_olmo3_skip_freq,
    )

    assert parse_olmo3_skip_freq(skip, layers) == want
    assert _cfg(num_layers=layers, window_attn_skip_freq=skip).window_attn_skip_freq == want


def test_skip_freq_proxy_depths():
    """The int skip_freq of the preset works at the reduced proxy depths the L4 gates use."""
    for layers in (4, 5, 8, 32):
        assert (
            _run_args_patch(
                window_size=[4096, 0], window_attn_skip_freq="4", num_layers=str(layers)
            ).window_attn_skip_freq
            == 4
        )
    with pytest.raises(ValueError, match="num_layers"):
        _run_args_patch(window_size=[4096, 0], window_attn_skip_freq=4, num_layers=2)


# Each entry: (setting, bad value). The same table refuses it on args (YAML or CLI string) and on
# the config (where the setting is a TransformerConfig field).
_SHARED_REFUSALS = [
    ("rotary_percent", 0.5),
    ("rotary_percent", "0.5"),
    ("use_rope_scaling", True),
    ("use_rope_scaling", "true"),
    ("rotary_seq_len_interpolation_factor", 2.0),
    ("use_turbo_attention", True),
    ("use_turbo_attention", "1"),
    ("no_rope_freq", 4),
    ("no_rope_freq", [0, 0, 0, 1]),
    ("qk_l2_norm", True),
    ("softmax_type", "off-by-one"),
    ("softmax_type", "learnable"),
    ("olmo3_rope_layout", "auto"),
    ("olmo3_rope_layout", "per-type"),
    ("olmo3_rope_layout", None),
    ("olmo3_rope_layout", "None"),
    ("untie_embeddings_and_output_weights", False),
    ("untie_embeddings_and_output_weights", "false"),
    ("swiglu", False),
    ("swiglu", "false"),
    ("normalization", "LayerNorm"),
    ("apply_residual_connection_post_layernorm", True),
    ("add_bias_linear", True),
    ("add_qkv_bias", True),
    ("layernorm_zero_centered_gamma", True),
    ("position_embedding_type", "learned_absolute"),
    ("position_embedding_type", "yarn"),
    ("apply_rope_fusion", True),
    ("apply_rope_fusion", "true"),
    ("qk_layernorm", True),
    ("multi_latent_attention", True),
    ("num_experts", 8),
    ("num_experts", "8"),
    ("rotary_interleaved", True),
    ("fused_single_qkv_rope", True),
    ("attention_output_gate", True),
    ("flash_decode", True),
    ("transformer_impl", "local"),
    ("gated_linear_unit", False),
    ("num_moe_experts", 8),
    ("use_te_activation_func", True),
    ("activation_func_clamp_value", 7.0),
    ("glu_linear_offset", 1.0),
    ("olmo3_window_size_form", "te"),
]


@pytest.mark.parametrize("name, value", _SHARED_REFUSALS)
def test_shared_refusal_table(name, value):
    from megatron.core.transformer.transformer_config import TransformerConfig

    from primus.backends.megatron.core.models.olmo3 import olmo3_builders
    from primus.backends.megatron.core.models.olmo3.olmo3_validation import (
        OLMO3_UNSUPPORTED,
        check_olmo3_unsupported,
    )

    assert name in {n for n, _, _ in OLMO3_UNSUPPORTED}
    # args patch (YAML values and CLI strings)
    with pytest.raises(ValueError, match=name):
        _run_args_patch(window_size=[4096, 0], window_attn_skip_freq=4, **{name: value})
    # builder (same table, before any config is built)
    args = SimpleNamespace(use_legacy_models=False, spec=None, mtp_num_layers=None, **{name: value})
    with pytest.raises(ValueError, match=name):
        olmo3_builders.olmo3_builder(args, True, True, config=_cfg())
    with pytest.raises(ValueError, match=name):
        check_olmo3_unsupported(args)
    # config, when it is a config field
    if name in TransformerConfig.__dataclass_fields__ or name.startswith("olmo3_"):
        with pytest.raises(ValueError, match="OLMo-3|olmo3"):
            _cfg(**{name: value})


def test_shared_refusal_table_accepts_olmo3_values():
    """The OLMo-3 preset / converter / CLI-string values pass the shared table."""
    from primus.backends.megatron.core.models.olmo3.olmo3_validation import (
        check_olmo3_unsupported,
    )

    good = SimpleNamespace(
        rotary_percent=1.0,
        use_rope_scaling=False,
        rotary_seq_len_interpolation_factor=None,
        use_turbo_attention=False,
        no_rope_freq=None,
        qk_l2_norm=False,
        softmax_type="vanilla",
        olmo3_rope_layout="per_type",
        untie_embeddings_and_output_weights=True,
        swiglu=True,
        normalization="RMSNorm",
        apply_residual_connection_post_layernorm=False,
        add_bias_linear=False,
        add_qkv_bias=False,
        layernorm_zero_centered_gamma=False,
        position_embedding_type="rope",
        apply_rope_fusion=False,
        qk_layernorm=False,
        multi_latent_attention=False,
        num_experts=None,
        rotary_interleaved=False,
        transformer_impl="transformer_engine",
        context_parallel_size=1,
    )
    check_olmo3_unsupported(good)
    cli = SimpleNamespace(
        **{k: (str(v).lower() if isinstance(v, bool) else v) for k, v in vars(good).items()}
    )
    cli.rotary_percent, cli.context_parallel_size = "1.0", "1"
    check_olmo3_unsupported(cli)
    check_olmo3_unsupported(_cfg())


def test_config_from_converter_margs_passes_table():
    """Config built by core_transformer_config_from_args from the converter's margs passes the table."""
    from megatron.training.arguments import core_transformer_config_from_args

    from primus.backends.megatron.checkpoint import native_loader_olmo3 as conv
    from primus.backends.megatron.core.models.olmo3.olmo3_transformer_config import (
        Olmo3TransformerConfig,
    )

    hf = dict(
        model_type="olmo3",
        hidden_size=256,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=4,
        intermediate_size=512,
        max_position_embeddings=65536,
        rms_norm_eps=1e-6,
        rope_theta=500000,
        tie_word_embeddings=False,
        attention_bias=False,
        hidden_act="silu",
        sliding_window=4096,
        vocab_size=128,
        layer_types=["sliding_attention"] * 3 + ["full_attention"],
        rope_scaling=dict(
            rope_type="yarn",
            factor=8.0,
            original_max_position_embeddings=8192,
            beta_fast=32,
            beta_slow=1,
            attention_factor=1.2079441541679836,
        ),
    )
    fields = conv.derive_megatron_fields(hf, rope_layout="per_type")
    args = SimpleNamespace(
        seq_length=64,
        make_vocab_size_divisible_by=128,
        hf_dir="x",
        dtype="fp32",
        save_dir="/tmp/x",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        expert_parallel_size=1,
    )
    margs = conv.build_margs(fields, args)  # build_margs output as is (it runs the table itself)
    from primus.backends.megatron.core.models.olmo3.olmo3_validation import (
        check_olmo3_unsupported,
    )

    check_olmo3_unsupported(margs)
    cfg = core_transformer_config_from_args(margs, config_class=Olmo3TransformerConfig)
    assert cfg.window_size == (4095, 0) and cfg.olmo3_sliding_window == 4096
    assert cfg.window_attn_skip_freq == 4
    assert margs.max_position_embeddings == 65536
    # a seq_length above the HF max_position_embeddings (the YaRN factor's base) is refused
    with pytest.raises(ValueError, match="seq_length 131072 > max_position_embeddings 65536"):
        conv.build_margs(fields, SimpleNamespace(**{**vars(args), "seq_length": 131072}))


def test_builder_validates_before_config_and_supplied_config(monkeypatch):
    from primus.backends.megatron.core.models.olmo3 import olmo3_builders

    built = []
    monkeypatch.setattr(olmo3_builders, "core_transformer_config_from_args", lambda *a, **k: built.append(1))
    base = dict(use_legacy_models=False, spec=None, mtp_num_layers=None)
    # refusals run before the config is built
    for bad in (
        dict(rotary_percent=0.5),
        dict(yaml_cfg="x.yaml"),
        dict(modelopt_enabled=True),
        dict(swiglu=False),
    ):
        with pytest.raises(ValueError, match="OLMo-3"):
            olmo3_builders.olmo3_builder(SimpleNamespace(**base, **bad), True, True)
    assert not built
    # a supplied config changed after __post_init__ is checked again
    for name, value in (
        ("window_size", (4096, 0)),
        ("window_attn_skip_freq", 1),
        ("add_bias_linear", True),
        ("apply_rope_fusion", True),
    ):
        cfg = _cfg()
        setattr(cfg, name, value)
        with pytest.raises(ValueError, match="OLMo-3|olmo3"):
            olmo3_builders.olmo3_builder(SimpleNamespace(**base), True, True, config=cfg)


def test_upstream_model_provider_import_errors(monkeypatch):
    """No model_provider module (older Megatron): None. A dependency of it missing: re-raised."""
    import builtins

    from primus.backends.megatron.core.models.olmo3 import olmo3_builders

    real_import = builtins.__import__

    def fake_import(missing):
        def imp(name, *a, **k):
            if name == "model_provider":
                raise ModuleNotFoundError(f"No module named {missing!r}", name=missing)
            return real_import(name, *a, **k)

        return imp

    monkeypatch.delitem(sys.modules, "model_provider", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import("model_provider"))
    assert olmo3_builders._upstream_model_provider() is None
    monkeypatch.setattr(builtins, "__import__", fake_import("modelopt"))
    with pytest.raises(ModuleNotFoundError, match="modelopt"):
        olmo3_builders._upstream_model_provider()


def test_built_layer_report_does_not_raise():
    """Unreadable values are "unknown"; a model without decoder layers gives []."""
    from primus.backends.megatron.core.models.olmo3 import olmo3_builders

    assert olmo3_builders.olmo3_built_layer_report(object()) == []
    layer = SimpleNamespace(layer_number="x", self_attention=SimpleNamespace(core_attention=None))
    model = SimpleNamespace(config=SimpleNamespace(), decoder=SimpleNamespace(layers=[layer]))
    (r,) = olmo3_builders.olmo3_built_layer_report(model)
    assert r["layer"] == "unknown" and r["rope"] == "unknown" and r["softmax_scale"] == "unknown"
    assert r["window"] == "unknown"


def test_model_provider_delegates_to_upstream(monkeypatch):
    """model_provider runs upstream's model_provider (record_memory_history, ...) with the OLMo-3 builder."""
    from primus.backends.megatron.core.models.olmo3 import olmo3_builders

    calls = []

    def upstream(builder, pre, post, vp, config=None, pg_collection=None):
        calls.append((builder, pre, post, vp, config, pg_collection))
        return "model"

    args = SimpleNamespace(
        use_legacy_models=False, spec=None, mtp_num_layers=None, record_memory_history=True
    )
    monkeypatch.setattr(olmo3_builders, "get_args", lambda: args)
    monkeypatch.setattr(olmo3_builders, "_upstream_model_provider", lambda: upstream)
    assert olmo3_builders.model_provider(None, True, False, 1) == "model"
    assert calls == [(olmo3_builders.olmo3_builder, True, False, 1, None, None)]


def test_model_provider_checks_before_upstream(monkeypatch):
    """model_provider checks the args before upstream (whose ModelOpt branch skips the builder);
    olmo3_builder checks again for direct callers. The config is never built first."""
    from primus.backends.megatron.core.models.olmo3 import olmo3_builders

    checks = []
    args = SimpleNamespace(yaml_cfg="x.yaml")
    monkeypatch.setattr(olmo3_builders, "get_args", lambda: args)
    monkeypatch.setattr(
        olmo3_builders,
        "_upstream_model_provider",
        lambda: lambda *a, **k: pytest.fail("upstream reached before the checks"),
    )
    real = olmo3_builders.check_olmo3_build_args
    monkeypatch.setattr(olmo3_builders, "check_olmo3_build_args", lambda a: (checks.append(a), real(a)))
    monkeypatch.setattr(
        olmo3_builders,
        "core_transformer_config_from_args",
        lambda *a, **k: pytest.fail("config built before the checks"),
    )
    with pytest.raises(ValueError, match="yaml_cfg"):
        olmo3_builders.model_provider(None)
    assert checks == [args]
    # ModelOpt: upstream would build its own model without olmo3_builder; refused up front
    args2 = SimpleNamespace(modelopt_enabled=True)
    monkeypatch.setattr(olmo3_builders, "get_args", lambda: args2)
    with pytest.raises(ValueError, match="ModelOpt"):
        olmo3_builders.model_provider(None)
    with pytest.raises(ValueError, match="yaml_cfg"):
        olmo3_builders.olmo3_builder(args, True, True)


def test_layer_spec_uses_the_te_spec_provider(monkeypatch):
    """Modules come from TESpecProvider at call time, so the Primus Turbo patch (which rebinds it)
    reaches OLMo-3's linears and core attention."""
    from megatron.core.extensions import transformer_engine_spec_provider as tsp

    from primus.backends.megatron.core.models.olmo3.olmo3_layer_specs import (
        get_olmo3_layer_spec,
    )

    class Provider:
        def column_parallel_linear(self):
            return "COL"

        def row_parallel_linear(self):
            return "ROW"

        def core_attention(self):
            return "CORE"

        def layer_norm(self, rms_norm=False, for_qk=False):
            return "NORM"

    monkeypatch.setattr(tsp, "TESpecProvider", Provider)
    s = get_olmo3_layer_spec().submodules
    att, mlp = s.self_attention, s.mlp
    assert (att.submodules.linear_qkv, att.submodules.core_attention, att.submodules.linear_proj) == (
        "COL",
        "CORE",
        "ROW",
    )
    assert (mlp.submodules.linear_fc1, mlp.submodules.linear_fc2) == ("COL", "ROW")
    assert att.params["post_norm"] == mlp.params["post_norm"] == "NORM"


def test_rotary_refuses_partial_rotary():
    from primus.backends.megatron.core.models.olmo3.olmo3_rotary import (
        Olmo3DualRotaryEmbedding,
    )

    with pytest.raises(ValueError, match="rotary_percent"):
        Olmo3DualRotaryEmbedding(_cfg(), rotary_base=500000.0, rotary_percent=0.5)


def test_flops_sliding_layer_count_uses_shared_parser():
    from primus.backends.megatron.patches.olmo3_flops_patches import (
        olmo3_sliding_layer_count,
    )

    assert olmo3_sliding_layer_count(8, (4095, 0), "([1]*3+[0])*2") == 6
    assert olmo3_sliding_layer_count(8, (4095, 0), "4") == 6
    for bad in (0, 1, "abc", 9, [1] * 8, None):
        with pytest.raises(ValueError, match="window_attn_skip_freq"):
            olmo3_sliding_layer_count(8, (4095, 0), bad)


def test_per_layer_te_window_cpu(monkeypatch):
    """The window and softmax scale each layer's core attention gets, without a per-layer config copy.

    CPU-only: SelfAttention.__init__ is stubbed to build core attention the way upstream does
    (``submodules.core_attention(config=..., softmax_scale=config.softmax_scale, ...)``).
    """
    from megatron.core.transformer.attention import (
        SelfAttention,
        SelfAttentionSubmodules,
    )
    from megatron.core.transformer.utils import is_layer_window_attention

    from primus.backends.megatron.core.transformer.olmo3 import olmo3_attention as oa

    seen = {}

    class FakeCore(torch.nn.Module):
        def __init__(self, config, layer_number, softmax_scale=None, **kw):
            super().__init__()
            # what TE / local DotProductAttention derive from the shared config
            win = (
                config.window_size
                if is_layer_window_attention(config.window_size, config.window_attn_skip_freq, layer_number)
                else None
            )
            seen[layer_number] = (config, win, softmax_scale)

    def fake_init(self, config, submodules, layer_number, *a, **k):
        torch.nn.Module.__init__(self)
        submodules.core_attention(
            config=config, layer_number=layer_number, softmax_scale=config.softmax_scale
        )
        raise StopIteration  # stop before the norms (they need the parallel state)

    monkeypatch.setattr(SelfAttention, "__init__", fake_init)
    cfg = _cfg(
        num_layers=8, olmo3_yarn_rotary_scaling_factor=8.0, olmo3_yarn_original_max_position_embeddings=8192
    )
    subs = SelfAttentionSubmodules(linear_qkv=None, core_attention=FakeCore)
    for n in range(1, 9):
        with pytest.raises(StopIteration):
            oa.Olmo3SelfAttention(cfg, subs, n)
    base = 1.0 / math.sqrt(cfg.kv_channels)
    for n, (c, win, scale) in seen.items():
        sliding = n % 4 != 0
        assert c is cfg, n  # no private config copy
        assert win == ((4095, 0) if sliding else None), n
        want = base if sliding else base * cfg.olmo3_yarn_mscale**2
        assert (scale or base) == pytest.approx(want), n
    assert cfg.window_size == (4095, 0) and cfg.softmax_scale is None  # the shared config is not modified
    assert subs.core_attention is FakeCore  # nor the shared spec


def test_softmax_scale_injection_survives_build_module():
    """The per-layer YaRN scale is a generated subclass, not a function: Megatron's build_module
    returns a function unchanged, so a closure would break if upstream built core attention through
    build_module. Class and ModuleSpec forms both build an instance with the fixed scale."""
    from megatron.core.transformer.spec_utils import ModuleSpec, build_module

    from primus.backends.megatron.core.transformer.olmo3 import olmo3_attention as oa

    class FakeCore(torch.nn.Module):
        def __init__(self, config=None, softmax_scale=None, extra=0):
            super().__init__()
            self.softmax_scale, self.extra = softmax_scale, extra

    cls = oa._with_softmax_scale(FakeCore, 0.5)
    assert isinstance(cls, type) and issubclass(cls, FakeCore) and cls.__name__ == "FakeCore"
    assert oa._with_softmax_scale(FakeCore, 0.5) is cls  # one class per (class, scale)
    built = build_module(cls, config=None, softmax_scale=0.125)
    assert isinstance(built, FakeCore) and built.softmax_scale == 0.5
    spec = oa._with_softmax_scale(ModuleSpec(module=FakeCore, params={"extra": 3}), 0.25)
    assert isinstance(spec, ModuleSpec) and spec.params == {"extra": 3}
    built = build_module(spec, config=None, softmax_scale=None)
    assert (built.softmax_scale, built.extra) == (0.25, 3)
    with pytest.raises(TypeError, match="class or a ModuleSpec"):
        oa._with_softmax_scale(lambda **k: None, 0.5)


# ---------------------------------------------------------------------------
# FullWidthRMSNorm: TP statistic and TP-sharded checkpoint
# ---------------------------------------------------------------------------
def test_full_width_rmsnorm_sharded_state_dict(monkeypatch):
    import megatron.core.utils as mcu

    from primus.backends.megatron.core.transformer.olmo3.olmo3_attention import (
        FullWidthRMSNorm,
    )

    tp, dp = object(), object()
    norm = FullWidthRMSNorm(8, 16, 1e-6)
    norm.tp_group = tp  # pretend to be TP rank 1 of 2 (no process group needed)
    monkeypatch.setattr(mcu, "get_pg_rank", lambda group=None: 1 if group is tp else 0)
    monkeypatch.setattr(mcu, "get_pg_size", lambda group=None: 2 if group is tp else 1)
    sd = norm.sharded_state_dict(
        prefix="decoder.layers.0.self_attention.q_norm.", metadata={"dp_cp_group": dp}
    )
    key = "decoder.layers.0.self_attention.q_norm.weight"
    assert list(sd) == [key]
    sh = sd[key]
    assert sh.key == key
    assert tuple(sh.global_shape) == (16,)
    assert tuple(sh.global_offset) == (8,)
    assert sh.data is norm.weight


def _eager_full_width_norm(x, weight, eps):
    """HF Olmo3RMSNorm on the un-split projection (fp32 statistic, weight * normed, cast back)."""
    h = x.reshape(*x.shape[:-2], -1).float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return (weight * h).to(x.dtype).reshape(x.shape)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_full_width_rmsnorm_matches_eager_forward_backward(dtype):
    """The single autograd op gives the eager HF formula's output and gradients (x and weight)."""
    from primus.backends.megatron.core.transformer.olmo3.olmo3_attention import (
        FullWidthRMSNorm,
    )

    torch.manual_seed(0)
    norm = FullWidthRMSNorm(32, 32, 1e-6).to(dtype)
    with torch.no_grad():
        norm.weight.uniform_(0.5, 1.5)
    x = torch.randn(3, 2, 4, 8, dtype=dtype, requires_grad=True)
    g = torch.randn(3, 2, 4, 8, dtype=dtype)
    out = norm(x)
    (out.float() * g.float()).sum().backward()
    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = norm.weight.detach().clone().requires_grad_(True)
    ref = _eager_full_width_norm(x_ref, w_ref, 1e-6)
    (ref.float() * g.float()).sum().backward()
    tol = dict(atol=2e-2, rtol=2e-2) if dtype == torch.bfloat16 else dict(atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out, ref, **tol)
    torch.testing.assert_close(x.grad, x_ref.grad, **tol)
    torch.testing.assert_close(norm.weight.grad, w_ref.grad, **tol)


def test_full_width_rmsnorm_fn_gradcheck():
    from primus.backends.megatron.core.transformer.olmo3.olmo3_attention import (
        _FullWidthRMSNormFn,
    )

    torch.manual_seed(0)
    x = torch.randn(3, 2, 4, 3, dtype=torch.float64, requires_grad=True)  # [s, b, heads, head_dim]
    w = torch.rand(4, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, b: _FullWidthRMSNormFn.apply(a, b, 1e-6, 12, None), (x, w))


def test_full_width_rmsnorm_does_not_copy_split_views():
    """q / k are non-contiguous split views of the fused qkv output: the norm reads them in place
    (no reshape copy saved for backward), and matches the norm of a contiguous copy."""
    from primus.backends.megatron.core.transformer.olmo3.olmo3_attention import (
        FullWidthRMSNorm,
    )

    torch.manual_seed(0)
    norm = FullWidthRMSNorm(4 * 8, 4 * 8, 1e-6)
    with torch.no_grad():
        norm.weight.uniform_(0.5, 1.5)
    qkv = torch.randn(5, 2, 4, 3 * 8, requires_grad=True)
    q = qkv.split(8, dim=-1)[0]
    assert not q.is_contiguous()
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda s: saved.append(s) or s, lambda s: s):
        out = norm(q)
    assert any(s.data_ptr() == qkv.data_ptr() for s in saved)  # the view itself, not a copy
    torch.testing.assert_close(out, norm(q.contiguous()))


def test_full_width_rmsnorm_saves_no_fp32_copies():
    """Autograd keeps the input (own dtype), the weight and one fp32 value per row, not fp32
    copies of the projection (the eager chain kept about 6x a bf16 input)."""
    from primus.backends.megatron.core.transformer.olmo3.olmo3_attention import (
        FullWidthRMSNorm,
    )

    norm = FullWidthRMSNorm(32 * 16, 32 * 16, 1e-6).to(torch.bfloat16)
    x = torch.randn(64, 2, 32, 16, dtype=torch.bfloat16, requires_grad=True)
    saved = []

    def pack(t):
        saved.append(t.numel() * t.element_size())
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        norm(x)
    in_bytes = x.numel() * x.element_size()
    assert sum(saved) <= in_bytes * 1.1, (sum(saved), in_bytes)


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "bad",
    [
        {"model_type": "olmo2"},
        {"hidden_act": "gelu"},
        {"attention_bias": True},
        {"max_position_embeddings": None},
        {"rope_scaling": {"rope_type": "llama3", "factor": 8.0}},
        {"rope_scaling": {"rope_type": "linear", "factor": 2.0}},
        {"rope_scaling": {"weird_layer": {"rope_type": "yarn"}}},
        {"rope_scaling": {"full_attention": "yarn", "sliding_attention": {"rope_type": "default"}}},
        {"rope_scaling": {"full_attention": {"rope_type": "yarn", "factor": 8.0}, "sliding_attention": 1}},
        {"rope_theta": None, "rope_scaling": {"rope_type": "yarn", "factor": 8.0}},
        # layer_types / sliding_window
        {"layer_types": None},
        {"layer_types": ["sliding_attention"] * 31 + ["full_attention"] * 2},
        {"layer_types": ["sliding_attention"] * 30 + ["full_attention"]},
        {"layer_types": ["sliding", "sliding", "sliding", "full"] * 8},
        {"layer_types": ["full_attention"] * 32},
        {"sliding_window": None},
        {"sliding_window": 0},
        {"sliding_window": 1},
        {"num_key_value_heads": 5},
        # top-level and nested rope_theta disagree
        {
            "rope_theta": 10000,
            "rope_scaling": None,
            "rope_parameters": {
                "full_attention": {"rope_type": "yarn", "factor": 8.0, "rope_theta": 500000},
                "sliding_attention": {"rope_type": "default", "rope_theta": 500000},
            },
        },
    ],
)
def test_converter_rejects_unsupported_configs(bad):
    from primus.backends.megatron.checkpoint.native_loader_olmo3 import (
        derive_megatron_fields,
    )

    derive_megatron_fields(dict(_HF_CFG), rope_layout="per_type")  # the baseline is valid
    cfg = {k: v for k, v in {**_HF_CFG, **bad}.items() if v is not None}
    with pytest.raises(ValueError, match="OLMo-3|olmo3|num_key_value_heads"):
        derive_megatron_fields(cfg, rope_layout="per_type")


def test_converter_refuses_all_sliding_and_tied_embeddings():
    """The converter refuses up front what the shared contract refuses later."""
    from primus.backends.megatron.checkpoint.native_loader_olmo3 import (
        derive_megatron_fields,
    )
    from primus.backends.megatron.core.models.olmo3.olmo3_validation import (
        check_olmo3_unsupported,
        parse_olmo3_skip_freq,
    )

    n = _HF_CFG["num_hidden_layers"]
    with pytest.raises(ValueError, match="no full_attention"):
        derive_megatron_fields({**_HF_CFG, "layer_types": ["sliding_attention"] * n}, rope_layout="per_type")
    with pytest.raises(ValueError, match="untied"):
        derive_megatron_fields({**_HF_CFG, "tie_word_embeddings": True}, rope_layout="per_type")
    # whatever the converter accepts passes the shared checks
    f = derive_megatron_fields(dict(_HF_CFG), rope_layout="per_type")
    parse_olmo3_skip_freq(f["window_attn_skip_freq"], f["num_layers"])
    assert f["untie_embeddings_and_output_weights"] is True
    check_olmo3_unsupported(
        SimpleNamespace(untie_embeddings_and_output_weights=True, olmo3_rope_layout=f["olmo3_rope_layout"])
    )


def test_converter_rope_layout_rules():
    """Flat config: explicit layout, else the default; yarn_all without YaRN is refused when
    explicit; a per-layer-type config's own layout cannot be overridden."""
    from primus.backends.megatron.checkpoint.native_loader_olmo3 import (
        derive_megatron_fields,
    )

    assert derive_megatron_fields(dict(_HF_CFG), default_layout="per_type")["olmo3_rope_layout"] == "per_type"
    assert derive_megatron_fields(dict(_HF_CFG), rope_layout="yarn_all")["olmo3_rope_layout"] == "yarn_all"
    no_yarn = {**_HF_CFG, "rope_scaling": None}
    with pytest.raises(ValueError, match="needs YaRN"):
        derive_megatron_fields(no_yarn, rope_layout="yarn_all")
    yarn = dict(_HF_CFG["rope_scaling"], rope_theta=500000)
    base = {k: v for k, v in _HF_CFG.items() if k not in ("rope_scaling", "rope_theta")}
    per_type = {
        **base,
        "rope_parameters": {
            "full_attention": yarn,
            "sliding_attention": {"rope_type": "default", "rope_theta": 500000},
        },
    }
    with pytest.raises(ValueError, match="declares"):
        derive_megatron_fields(per_type, rope_layout="yarn_all")
    assert derive_megatron_fields(per_type, rope_layout="auto")["olmo3_rope_layout"] == "per_type"


def test_converter_per_layer_type_rope_parameters():
    """transformers 5.x per-layer-type rope_parameters: YaRN kept, theta read, layout from the sliding entry."""
    from primus.backends.megatron.checkpoint.native_loader_olmo3 import (
        derive_megatron_fields,
    )

    yarn = dict(_HF_CFG["rope_scaling"], rope_theta=500000)
    base = {k: v for k, v in _HF_CFG.items() if k not in ("rope_scaling", "rope_theta")}
    for sliding, layout in (({"rope_type": "default", "rope_theta": 500000}, "per_type"), (yarn, "yarn_all")):
        f = derive_megatron_fields(
            {**base, "rope_parameters": {"full_attention": yarn, "sliding_attention": sliding}}
        )
        assert f["olmo3_yarn_rotary_scaling_factor"] == 8.0
        assert f["rotary_base"] == 500000.0
        assert f["olmo3_rope_layout"] == layout


def test_converter_yarn_original_max_defaults_to_max_position_embeddings():
    from primus.backends.megatron.checkpoint.native_loader_olmo3 import (
        derive_megatron_fields,
    )

    rope = {k: v for k, v in _HF_CFG["rope_scaling"].items() if k != "original_max_position_embeddings"}
    f = derive_megatron_fields({**_HF_CFG, "rope_scaling": rope}, rope_layout="per_type")
    assert f["olmo3_yarn_original_max_position_embeddings"] == 65536


def test_converter_window_pattern_list_and_null_top_level_theta():
    from primus.backends.megatron.checkpoint.native_loader_olmo3 import (
        derive_megatron_fields,
    )

    types = ["sliding_attention"] * 31 + ["full_attention"]  # period 32
    f = derive_megatron_fields({**_HF_CFG, "layer_types": types}, rope_layout="per_type")
    assert f["window_attn_skip_freq"] == 32
    pattern = [0 if i in (3, 7, 30) else 1 for i in range(32)]  # not periodic
    types = ["sliding_attention" if v else "full_attention" for v in pattern]
    f = derive_megatron_fields({**_HF_CFG, "layer_types": types}, rope_layout="per_type")
    assert f["window_attn_skip_freq"] == pattern
    assert f["window_size"] == (4095, 0)
    # transformers 5.x: rope_theta: null at the top level -> the nested value
    yarn = dict(_HF_CFG["rope_scaling"], rope_theta=500000)
    cfg = {k: v for k, v in _HF_CFG.items() if k != "rope_scaling"}
    cfg.update(
        rope_theta=None,
        rope_parameters={
            "full_attention": yarn,
            "sliding_attention": {"rope_type": "default", "rope_theta": 500000},
        },
    )
    assert derive_megatron_fields(cfg)["rotary_base"] == 500000.0


@pytest.mark.parametrize("sliding_yarn,layout", [(False, "per_type"), (True, "yarn_all")])
def test_convert_keeps_config_rope_layout(monkeypatch, tmp_path, sliding_yarn, layout):
    """convert() without rope_layout stores the layout the per-layer-type config declares."""
    import primus.backends.megatron.checkpoint.native_loader_olmo3 as conv

    yarn = dict(_HF_CFG["rope_scaling"], rope_theta=500000)
    cfg = {k: v for k, v in _HF_CFG.items() if k not in ("rope_scaling", "rope_theta")}
    cfg["rope_parameters"] = {
        "full_attention": yarn,
        "sliding_attention": yarn if sliding_yarn else {"rope_type": "default", "rope_theta": 500000},
    }
    seen = {}

    class _Stop(Exception):
        pass

    def fake_build_margs(fields, args):
        seen.update(fields)
        raise _Stop

    monkeypatch.setattr(conv.common, "ensure_megatron_on_path", lambda: None)
    monkeypatch.setattr(conv.common, "assert_bridge_free", lambda *a: None)
    monkeypatch.setattr(conv.common, "read_hf_config", lambda d: dict(cfg))
    monkeypatch.setattr(conv, "build_margs", fake_build_margs)
    with pytest.raises(_Stop):
        conv.convert(str(tmp_path), str(tmp_path / "out"), log=lambda *a: None)
    assert seen["olmo3_rope_layout"] == layout
    # flat (4.57-style) config: the trained per_type layout
    monkeypatch.setattr(conv.common, "read_hf_config", lambda d: dict(_HF_CFG))
    with pytest.raises(_Stop):
        conv.convert(str(tmp_path), str(tmp_path / "out"), log=lambda *a: None)
    assert seen["olmo3_rope_layout"] == "per_type"
    # flat config: an explicit concrete layout is used
    with pytest.raises(_Stop):
        conv.convert(str(tmp_path), str(tmp_path / "out"), rope_layout="yarn_all", log=lambda *a: None)
    assert seen["olmo3_rope_layout"] == "yarn_all"
    # auto never reaches a saved checkpoint
    with pytest.raises(ValueError, match="not auto"):
        conv.convert(str(tmp_path), str(tmp_path / "out"), rope_layout="auto", log=lambda *a: None)
    # per-layer-type config: a different explicit layout is refused, not applied silently
    monkeypatch.setattr(conv.common, "read_hf_config", lambda d: dict(cfg))
    other = "per_type" if layout == "yarn_all" else "yarn_all"
    with pytest.raises(ValueError, match="declares"):
        conv.convert(str(tmp_path), str(tmp_path / "out"), rope_layout=other, log=lambda *a: None)
    with pytest.raises(_Stop):
        conv.convert(str(tmp_path), str(tmp_path / "out"), rope_layout=layout, log=lambda *a: None)
    assert seen["olmo3_rope_layout"] == layout


def test_flops_patch_sliding_correction():
    from primus.backends.megatron.core.transformer.olmo3.olmo3_attention import (
        olmo3_layer_is_sliding,
    )
    from primus.backends.megatron.patches import olmo3_flops_patches as fpm
    from primus.backends.megatron.patches.olmo3_flops_patches import (
        olmo3_attention_flops_correction,
        olmo3_mean_visible_keys,
        olmo3_saved_keys,
        olmo3_sliding_layer_count,
    )

    assert olmo3_sliding_layer_count(32, (4095, 0), 4) == 24
    assert olmo3_sliding_layer_count(4, (4095, 0), [1, 1, 1, 0]) == 3
    assert olmo3_sliding_layer_count(32, None, 4) == 0
    with pytest.raises(ValueError, match="entries"):
        olmo3_sliding_layer_count(32, (4095, 0), [1, 1, 1, 0])
    # the count is the model's own per-layer rule
    cfg = SimpleNamespace(window_size=(4095, 0), window_attn_skip_freq=[1, 0, 1, 1, 0, 1])
    assert olmo3_sliding_layer_count(6, (4095, 0), cfg.window_attn_skip_freq) == sum(
        olmo3_layer_is_sliding(cfg, n) for n in range(1, 7)
    )
    for bad in (-1, 0, 1, 4096.5, "4k", True):
        with pytest.raises(ValueError, match="window"):
            olmo3_mean_visible_keys(10, bad)
        with pytest.raises(ValueError, match="window"):
            olmo3_saved_keys(10, bad)
    # brute force mean visible keys
    for s, w in ((16, 4), (8, 8), (8, 100), (10, 2)):
        ref = sum(min(i + 1, w) for i in range(s)) / s
        assert olmo3_mean_visible_keys(s, w) == pytest.approx(ref)
    # normalised args (after the OLMo-3 args patch)
    a = SimpleNamespace(
        num_layers=32,
        window_size=(4095, 0),
        olmo3_sliding_window=4096,
        olmo3_window_size_form="megatron",
        window_attn_skip_freq=4,
        seq_length=8192,
        kv_channels=128,
        num_attention_heads=32,
    )
    corr = olmo3_attention_flops_correction(a, 1)
    # upstream's convention: s / 2 minus the continuous windowed count W - W^2 / (2 s)
    saved = 8192 / 2 - (4096 - 4096**2 / (2 * 8192))
    assert saved == pytest.approx(1024.0)
    assert corr == pytest.approx(8192 * 24 * 3 * 2 * 4096 * saved * 2)
    assert olmo3_attention_flops_correction(a, 16) == pytest.approx(16 * corr)  # scaled per sample
    # the HF form the preset writes ([4096, 0], form hf) is W = 4096, not 4097
    hf = SimpleNamespace(**dict(vars(a), window_size=[4096, 0], olmo3_window_size_form="hf"))
    assert olmo3_attention_flops_correction(hf, 1) == pytest.approx(corr)
    del hf.olmo3_sliding_window
    assert olmo3_attention_flops_correction(hf, 1) == pytest.approx(corr)
    # memoised: one evaluation per setting
    fpm._correction_per_sample.cache_clear()
    for _ in range(5):
        olmo3_attention_flops_correction(a, 8)
    assert fpm._correction_per_sample.cache_info().misses == 1
    # window not active (seq <= window): exactly no correction
    for s in (1, 1024, 4095, 4096):
        a.seq_length = s
        assert olmo3_attention_flops_correction(a, 1) == 0.0
    # the windowed count a sliding layer keeps is within half a key of the exact (brute-force) mean,
    # as upstream's s / 2 is within half a key of the exact causal mean; never negative
    for s, w in ((10, 3), (8192, 4096), (4097, 4096), (64, 2), (100, 99)):
        kept = s / 2 - olmo3_saved_keys(s, w)
        exact = sum(min(i + 1, w) for i in range(s)) / s
        assert olmo3_saved_keys(s, w) >= 0
        assert 0 <= exact - kept <= 0.5
    a.seq_length, a.olmo3_sliding_window, a.window_size = 10, 3, (2, 0)
    a.num_layers, a.window_attn_skip_freq, a.kv_channels, a.num_attention_heads = 4, 4, 1, 1
    assert olmo3_attention_flops_correction(a, 1) == pytest.approx(10 * 3 * 3 * 2 * (7**2 / 20) * 2)
    a.num_layers, a.olmo3_sliding_window, a.window_size, a.seq_length = 32, 4096, (4095, 0), 8192
    a.window_attn_skip_freq, a.kv_channels, a.num_attention_heads = 4, 128, 32
    # string values (CLI / YAML overrides) as the args patch reads them; marker and window must agree
    a.olmo3_sliding_window = "4096"
    assert olmo3_attention_flops_correction(a, 1) == pytest.approx(corr)
    a.olmo3_sliding_window = 2048
    with pytest.raises(ValueError, match="olmo3_sliding_window"):
        olmo3_attention_flops_correction(a, 1)
    for bad in ("4k", 0, 1, True):
        a.olmo3_sliding_window = bad
        with pytest.raises(ValueError, match="window|OLMo-3"):
            olmo3_attention_flops_correction(a, 1)
    # no marker and no declared form: ambiguous, refused as by the args patch (never guessed)
    a.olmo3_sliding_window, a.olmo3_window_size_form = None, None
    with pytest.raises(ValueError, match="ambiguous"):
        olmo3_attention_flops_correction(a, 1)


def test_flops_patch_does_not_stack(monkeypatch):
    """A second patch run is a no-op even when another wrapper sits on top of ours."""
    import functools

    import megatron.training.training as training_module

    from primus.backends.megatron.patches.olmo3_flops_patches import patch_olmo3_flops

    monkeypatch.setattr(training_module, "num_floating_point_operations", lambda args, bs: 1.0e15)
    patch_olmo3_flops(None)
    ours = training_module.num_floating_point_operations
    assert ours._olmo3_flops_patched
    # a correction that fails at install time (bad window, a None field) never stops the run: it
    # is logged and the patch installs (the count turns the correction off at the first count)
    from primus.backends.megatron.patches import olmo3_flops_patches as fp

    warnings = []
    monkeypatch.setattr(fp, "warning_rank_0", warnings.append)
    base = dict(
        num_layers=4,
        window_size=(3, 0),
        olmo3_sliding_window=4,
        seq_length=16,
        window_attn_skip_freq=4,
        kv_channels=8,
        num_attention_heads=2,
    )
    for bad in (dict(olmo3_sliding_window=9), dict(kv_channels=None), dict(seq_length=None)):
        monkeypatch.setattr(training_module, "num_floating_point_operations", lambda args, bs: 1.0e15)
        warnings.clear()
        patch_olmo3_flops(SimpleNamespace(extra={"backend_args": SimpleNamespace(**{**base, **bad})}))
        assert training_module.num_floating_point_operations._olmo3_flops_patched
        assert any("at install time" in w for w in warnings), bad
    monkeypatch.setattr(training_module, "num_floating_point_operations", lambda args, bs: 1.0e15)
    patch_olmo3_flops(None)
    ours = training_module.num_floating_point_operations

    @functools.wraps(ours)
    def other(args, bs):
        return ours(args, bs)

    monkeypatch.setattr(training_module, "num_floating_point_operations", other)
    patch_olmo3_flops(None)
    assert training_module.num_floating_point_operations is other


def test_converter_stores_concrete_rope_layout(monkeypatch):
    """``auto`` is resolved before it reaches the (saved) args."""
    transformers = pytest.importorskip("transformers")
    from primus.backends.megatron.checkpoint.native_loader_olmo3 import (
        derive_megatron_fields,
    )

    monkeypatch.setattr(transformers, "__version__", "5.10.0")
    # default_layout auto (reference builds) follows transformers for a flat config
    assert derive_megatron_fields(dict(_HF_CFG))["olmo3_rope_layout"] == "yarn_all"
    assert derive_megatron_fields(dict(_HF_CFG), default_layout="per_type")["olmo3_rope_layout"] == "per_type"
    assert derive_megatron_fields(dict(_HF_CFG), rope_layout="auto")["olmo3_rope_layout"] == "yarn_all"
    assert derive_megatron_fields(dict(_HF_CFG), rope_layout="per_type")["olmo3_rope_layout"] == "per_type"


@pytest.mark.parametrize(
    "par", [dict(tensor_parallel_size=2), dict(pipeline_parallel_size=2), dict(expert_parallel_size=2)]
)
def test_converter_single_shard_only(par, tmp_path):
    from primus.backends.megatron.checkpoint import native_loader_olmo3 as conv

    with pytest.raises(ValueError, match="single-shard"):
        conv.convert(str(tmp_path), str(tmp_path / "out"), **par)


def test_converter_qkv_fusion_order_and_names():
    """GQA-style fusion: per query group [q heads of the group, k, v]; all HF tensors consumed."""
    from primus.backends.megatron.checkpoint.native_loader_olmo3 import (
        build_mcore_state_dict,
    )

    nh, ng, hd, hidden, ffn, vocab = 4, 2, 2, 8, 6, 16
    torch.manual_seed(0)
    hf = {
        "model.embed_tokens.weight": torch.randn(vocab, hidden),
        "model.norm.weight": torch.randn(hidden),
        "lm_head.weight": torch.randn(vocab, hidden),
    }
    h = "model.layers.0."
    hf.update(
        {
            h + "self_attn.q_proj.weight": torch.randn(nh * hd, hidden),
            h + "self_attn.k_proj.weight": torch.randn(ng * hd, hidden),
            h + "self_attn.v_proj.weight": torch.randn(ng * hd, hidden),
            h + "self_attn.o_proj.weight": torch.randn(hidden, nh * hd),
            h + "self_attn.q_norm.weight": torch.randn(nh * hd),
            h + "self_attn.k_norm.weight": torch.randn(ng * hd),
            h + "post_attention_layernorm.weight": torch.randn(hidden),
            h + "mlp.gate_proj.weight": torch.randn(ffn, hidden),
            h + "mlp.up_proj.weight": torch.randn(ffn, hidden),
            h + "mlp.down_proj.weight": torch.randn(hidden, ffn),
            h + "post_feedforward_layernorm.weight": torch.randn(hidden),
        }
    )

    class _Store:
        def get(self, name, dtype):
            return hf[name].to(dtype)

        def has(self, name):
            return name in hf

    class _Model:  # only the embedding parameter is read (state_dict() would touch TE extra state)
        embedding = SimpleNamespace(word_embeddings=SimpleNamespace(weight=torch.empty(vocab, hidden)))

        def state_dict(self):
            raise AssertionError("build_mcore_state_dict must not call model.state_dict()")

    margs = SimpleNamespace(
        params_dtype=torch.float32,
        num_attention_heads=nh,
        num_query_groups=ng,
        kv_channels=hd,
        hidden_size=hidden,
        untie_embeddings_and_output_weights=True,
        num_layers=1,
    )
    sd, consumed = build_mcore_state_dict(_Model(), _Store(), margs, {}, lambda *a: None)
    assert consumed == set(hf)
    q = hf[h + "self_attn.q_proj.weight"].reshape(ng, nh // ng, hd, hidden)
    k = hf[h + "self_attn.k_proj.weight"].reshape(ng, 1, hd, hidden)
    v = hf[h + "self_attn.v_proj.weight"].reshape(ng, 1, hd, hidden)
    want = torch.cat([q, k, v], dim=1).reshape(-1, hidden)
    assert torch.equal(sd["decoder.layers.0.self_attention.linear_qkv.weight"], want)
    fc1 = sd["decoder.layers.0.mlp.linear_fc1.weight"]
    assert torch.equal(fc1[:ffn], hf[h + "mlp.gate_proj.weight"])
    assert torch.equal(fc1[ffn:], hf[h + "mlp.up_proj.weight"])
    assert torch.equal(sd["decoder.layers.0.self_attention.q_norm.weight"], hf[h + "self_attn.q_norm.weight"])
    assert "decoder.layers.0.mlp.post_feedforward_layernorm.weight" in sd


def test_convert_hook_registers_olmo3(tmp_path, monkeypatch):
    import json

    mod = _load_convert_hook(monkeypatch)
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["Olmo3ForCausalLM"], "model_type": "olmo3"})
    )
    assert mod._detect_native_family(str(tmp_path)) == "olmo3"
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "olmo3"}))
    assert mod._detect_native_family(str(tmp_path)) == "olmo3"


def _load_convert_hook(monkeypatch):
    spec = importlib.util.spec_from_file_location("olmo3_convert_hook_run", _CONVERT_HOOK)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from primus.backends.megatron.checkpoint import native_convert_common as common

    monkeypatch.setattr(common, "ensure_megatron_on_path", lambda: None)
    monkeypatch.setattr(common, "apply_convert_patches", lambda **k: 0)
    return mod


def _stub_hook_config(mod, monkeypatch, hf_path, opts, checkpoint=None):
    """main() reads the config once: stub that reading with (hf_path, checkpoint) and ``opts``."""
    monkeypatch.setattr(mod, "_module_value_reader", lambda cfg, cli=None: None)
    monkeypatch.setattr(mod, "_checkpoint_paths", lambda value: (hf_path, checkpoint))
    monkeypatch.setattr(mod, "_native_opts", lambda value: {"enabled": None, **opts})


def test_convert_hook_routes_olmo3_to_its_converter(monkeypatch, tmp_path):
    """native_convert_checkpoint on an OLMo-3 config.json calls native_loader_olmo3.convert."""
    import json

    import primus.backends.megatron.checkpoint.native_loader_olmo3 as conv

    mod = _load_convert_hook(monkeypatch)
    calls = []
    monkeypatch.setattr(conv, "convert", lambda *a, **k: calls.append((a, k)))
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["Olmo3ForCausalLM"], "model_type": "olmo3", "torch_dtype": "bfloat16"})
    )
    mod.native_convert_checkpoint(str(tmp_path), str(tmp_path / "out"), {})
    assert calls == [
        (
            (str(tmp_path), str(tmp_path / "out")),
            dict(
                dtype="bf16",
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                expert_parallel_size=1,
                make_vocab_size_divisible_by=128,
            ),
        )
    ]


def test_convert_hook_refuses_unknown_family(monkeypatch, tmp_path):
    import json

    mod = _load_convert_hook(monkeypatch)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "gpt_neox"}))
    with pytest.raises(ValueError, match="no native converter"):
        mod.native_convert_checkpoint(str(tmp_path), str(tmp_path / "out"), {})


def test_convert_hook_refuses_olmo3_on_the_bridge_path(monkeypatch, tmp_path, capsys):
    """OLMo-3 never reaches the Megatron-Bridge path (it has no OLMo-3): native_ckpt_convert false, or
    an hf_path that is not a local HF directory (the preset's hub id) with model_type olmo3 that is
    not in the Hugging Face cache and cannot be downloaded, exits 1 with the reason before any lock
    or conversion. Other families keep the Bridge path."""
    import json

    mod = _load_convert_hook(monkeypatch)
    hf = tmp_path / "hf"
    hf.mkdir()
    (hf / "config.json").write_text(json.dumps({"model_type": "olmo3"}))
    monkeypatch.setattr(mod, "convert_checkpoint", lambda *a: pytest.fail("Bridge path for OLMo-3"))
    monkeypatch.setattr(mod, "_native_main", lambda *a: pytest.fail("native path"))
    monkeypatch.setattr(mod, "_resolve_hub_snapshot", lambda hf_path, opts: None)  # no cache, no network
    monkeypatch.setattr(sys, "argv", ["hook", "--config", str(tmp_path / "x.yaml")])
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "data"))
    cases = [
        (str(hf), dict(enabled=False), "native_ckpt_convert is"),
        ("allenai/Olmo-3-1025-7B", dict(enabled=None, model_type="olmo3"), "local HF directory"),
        ("allenai/Olmo-3-1025-7B", dict(enabled=True, model_type="olmo3"), "local HF directory"),
        ("allenai/Olmo-3-1025-7B", dict(enabled=False, model_type="olmo3"), "native_ckpt_convert is"),
    ]
    for hf_path, opts, msg in cases:
        _stub_hook_config(mod, monkeypatch, hf_path, opts)
        with pytest.raises(SystemExit):
            mod.main()
        assert msg in capsys.readouterr().err
    assert not (tmp_path / "data").exists()
    assert mod._olmo3_bridge_refusal("org/llama", None, dict(model_type="gpt")) is None
    # model_type olmo3 builds OLMo-3: a checkpoint of another family is refused, never converted
    refusal = mod._olmo3_bridge_refusal(str(hf), "llama", dict(model_type="olmo3"))
    assert "is llama" in refusal and "cannot load into OLMo-3" in refusal
    for family in ("llama", "qwen3", "deepseek"):
        assert mod.conversion_plan(str(hf), None, dict(model_type="olmo3", enabled=None), family) == "refuse"
        assert mod.conversion_plan(str(hf), None, dict(model_type="gpt", enabled=None), family) == "native"
    assert mod.conversion_plan(str(hf), None, dict(model_type="olmo3", enabled=None), "olmo3") == "native"
    # a hub id with model_type gpt (the SFT trainer default) is still recognised as OLMo-3 by the
    # HF id, whose name must start with olmo3 / olmo-3 / olmo_3 and a separator or the end
    for hf_id in ("allenai/Olmo-3-1025-7B", "/models/olmo3-7b", "org/OLMo_3", "org/olmo-3.1-7b"):
        assert "local HF directory" in mod._olmo3_bridge_refusal(hf_id, None, dict(model_type="gpt"))
    for hf_id in (
        "allenai/OLMo-2-1124-7B",
        "org/OLMo-30B",
        "org/olmo3x",
        "org/olmo_32b",
        "org/olmo-35",
        "org/llama-olmo3-distill",
    ):
        assert mod._olmo3_bridge_refusal(hf_id, None, dict(model_type="gpt")) is None, hf_id
    # the preset name is not an option: an explicit non-OLMo hf_path keeps its own family
    assert "model" not in mod._native_opts(None)
    assert mod._olmo3_bridge_refusal("Qwen/Qwen3-8B", None, dict(model_type="gpt")) is None
    # the message names the actual cause: a hub id that could not be fetched, or a local
    # directory without config.json (no download is tried for it)
    assert "could not be downloaded" in mod._olmo3_bridge_refusal(
        "allenai/Olmo-3-1025-7B", None, dict(model_type="olmo3")
    )
    empty = tmp_path / "empty"
    empty.mkdir()
    msg = mod._olmo3_bridge_refusal(str(empty), None, dict(model_type="olmo3"))
    assert "has no config.json" in msg and "downloaded" not in msg


def test_convert_hook_passes_olmo3_training_settings(monkeypatch, tmp_path):
    """make_vocab_size_divisible_by (the padded vocab rows) and the RoPE layout reach the OLMo-3
    converter; seq_length (no effect on the weights) does not."""
    import json

    import primus.backends.megatron.checkpoint.native_loader_olmo3 as conv

    mod = _load_convert_hook(monkeypatch)
    calls = []
    monkeypatch.setattr(conv, "convert", lambda *a, **k: calls.append(k))
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "olmo3", "torch_dtype": "bfloat16"}))
    opts = dict(seq_length="8192", make_vocab_size_divisible_by=64, olmo3_rope_layout="per_type")
    mod.native_convert_checkpoint(str(tmp_path), str(tmp_path / "out"), opts)
    assert "seq_length" not in calls[0] and calls[0]["make_vocab_size_divisible_by"] == 64
    assert calls[0]["rope_layout"] == "per_type"


def test_convert_hook_refuses_olmo3_single_shard_at_tp_pp(monkeypatch, tmp_path):
    """The OLMo-3 converter writes TP=PP=EP=1 only; training OLMo-3 at TP, PP or EP > 1 gets a clear
    message before any conversion (the hook checks OLMo-3 only; see the other-families test)."""
    from primus.backends.megatron.checkpoint import native_flow_olmo3 as flow

    mod = _load_convert_hook(monkeypatch)
    assert flow.single_shard_conflict({}) is None
    assert flow.single_shard_conflict(dict(tensor_model_parallel_size=1)) is None
    msg = flow.single_shard_conflict(dict(tensor_model_parallel_size=1, expert_model_parallel_size=8))
    assert msg and "expert_model_parallel_size=8" in msg
    for tp, pp in ((2, 1), (1, 2), ("8", "1")):
        msg = flow.single_shard_conflict(dict(tensor_model_parallel_size=tp, pipeline_model_parallel_size=pp))
        assert "native_ckpt_convert" not in msg and f"tensor_model_parallel_size={int(tp)}" in msg
        assert "pretrained_checkpoint" in msg and "UNVERIFIED" in msg
    # an unresolved ${VAR} (or any non-integer) is a clear error, not a bare int() traceback
    for bad in ("${TP:1}", "two", 1.5, True):
        with pytest.raises(ValueError, match="tensor_model_parallel_size must be an integer"):
            flow.single_shard_conflict(dict(tensor_model_parallel_size=bad))

    hf = tmp_path / "hf"
    hf.mkdir()
    (hf / "config.json").write_text('{"model_type": "olmo3"}')
    ran = []
    monkeypatch.setattr(mod, "_native_main", lambda *a: ran.append(a))
    _stub_hook_config(mod, monkeypatch, str(hf), dict(tensor_model_parallel_size=2))
    monkeypatch.setattr(sys, "argv", ["hook", "--config", str(tmp_path / "x.yaml")])
    with pytest.raises(SystemExit):
        mod.main()
    assert not ran
    _stub_hook_config(mod, monkeypatch, str(hf), dict(tensor_model_parallel_size="two"))
    with pytest.raises(SystemExit):
        mod.main()
    assert not ran
    _stub_hook_config(mod, monkeypatch, str(hf), dict(tensor_model_parallel_size=1))
    mod.main()
    assert len(ran) == 1
    # a hub id (the preset's tokenizer_model): the layout is refused before any snapshot download
    monkeypatch.setattr(mod, "_resolve_hub_snapshot", lambda *a, **k: pytest.fail("downloaded first"))
    _stub_hook_config(
        mod, monkeypatch, "allenai/Olmo-3-1025-7B", dict(model_type="olmo3", tensor_model_parallel_size=2)
    )
    with pytest.raises(SystemExit):
        mod.main()
    assert len(ran) == 1


def test_yarn_parameter_validation():
    """Only the YaRN keys Olmo3DualRotaryEmbedding implements are accepted, consistently."""
    from primus.backends.megatron.checkpoint.native_loader_olmo3 import (
        derive_megatron_fields,
    )

    rope = dict(_HF_CFG["rope_scaling"])

    def fields(**kw):
        return derive_megatron_fields({**_HF_CFG, "rope_scaling": {**rope, **kw}}, rope_layout="per_type")

    f = fields()
    assert (f["olmo3_yarn_beta_fast"], f["olmo3_yarn_beta_slow"]) == (32, 1)
    assert fields(truncate=True)["olmo3_yarn_rotary_scaling_factor"] == 8.0
    for bad, match in (
        (dict(mscale=1.0), "mscale"),
        (dict(mscale_all_dim=1.0), "mscale_all_dim"),
        (dict(truncate=False), "truncate"),
        (dict(factor=0.5), "factor"),
        (dict(factor="8"), "factor"),
        (dict(factor=4.0), "max_position_embeddings"),  # 65536 / 8192 = 8
        (dict(factor=1.0), "max_position_embeddings"),  # HF derives YaRN from the ratio: not "no YaRN"
        (dict(factor=1), "max_position_embeddings"),
        (dict(attention_factor=0.0), "attention_factor"),
        (dict(attention_factor=-1.2), "attention_factor"),
        (dict(beta_fast=1, beta_slow=32), "beta"),
    ):
        with pytest.raises(ValueError, match=match):
            fields(**bad)


def test_yarn_all_needs_equal_sliding_and_full_parameters():
    """Per-layer-type rope parameters: a sliding YaRN entry must equal the full-attention one
    (one YaRN table is built); any differing key is refused."""
    from primus.backends.megatron.checkpoint.native_loader_olmo3 import (
        derive_megatron_fields,
    )

    yarn = dict(_HF_CFG["rope_scaling"], rope_theta=500000)
    base = {k: v for k, v in _HF_CFG.items() if k not in ("rope_scaling", "rope_theta")}
    assert (
        derive_megatron_fields(
            {**base, "rope_parameters": {"full_attention": yarn, "sliding_attention": dict(yarn)}}
        )["olmo3_rope_layout"]
        == "yarn_all"
    )
    for key, value in (
        ("factor", 4.0),
        ("original_max_position_embeddings", 4096),
        ("beta_fast", 16),
        ("beta_slow", 2),
        ("attention_factor", 1.0),
        ("rope_theta", 10000),
    ):
        sliding = dict(yarn, **{key: value})
        with pytest.raises(ValueError, match="yarn_all|YaRN|rope"):
            derive_megatron_fields(
                {**base, "rope_parameters": {"full_attention": yarn, "sliding_attention": sliding}}
            )


def test_yarn_attention_factor_refusals():
    from primus.backends.megatron.core.models.olmo3.olmo3_transformer_config import (
        yarn_attention_factor,
    )

    for bad in ((0.5,), (8.0, 0.0), (8.0, -1.0), (1.0, 0.0)):
        with pytest.raises(ValueError, match="YaRN|yarn|attention_factor"):
            yarn_attention_factor(*bad)
    with pytest.raises(ValueError, match="attention_factor"):
        _cfg(
            olmo3_yarn_rotary_scaling_factor=8.0,
            olmo3_yarn_original_max_position_embeddings=8192,
            olmo3_yarn_attention_factor=0.0,
        )
    with pytest.raises(ValueError, match=">= 1"):
        _cfg(olmo3_yarn_rotary_scaling_factor=0.5)


def test_rope_parameters_and_rope_scaling_must_agree():
    """Both rope keys present: equal is fine, different is refused (neither is dropped); a null key
    counts as absent, and an empty dict is a value (no scaling), not a fall-through."""
    from primus.backends.megatron.checkpoint.native_loader_olmo3 import (
        _rope_config,
        _rope_source,
    )

    yarn = dict(_HF_CFG["rope_scaling"])
    base = {k: v for k, v in _HF_CFG.items() if k != "rope_scaling"}
    assert _rope_source({**base, "rope_scaling": yarn, "rope_parameters": dict(yarn)}) == yarn
    assert _rope_source({**base, "rope_scaling": yarn, "rope_parameters": None}) == yarn
    assert _rope_source({**base, "rope_parameters": yarn}) == yarn
    assert _rope_source(base) == {}
    with pytest.raises(ValueError, match="both rope_parameters and rope_scaling"):
        _rope_config(
            {
                **base,
                "rope_parameters": {"rope_type": "default", "rope_theta": 500000.0},
                "rope_scaling": yarn,
            }
        )
    with pytest.raises(ValueError, match="differ"):
        _rope_config({**base, "rope_parameters": {}, "rope_scaling": yarn})
    with pytest.raises(ValueError, match="must be a dict"):
        _rope_source({**base, "rope_scaling": "yarn"})


def _causal_attention_flops(a):
    """Upstream's core-attention FLOPs per sample at s / 2 keys per query (the count the OLMo-3
    correction subtracts from): 3 * 2 * q_size * s / 2 * 2 per token, s tokens, every layer."""
    return 6.0 * a.kv_channels * a.num_attention_heads * a.num_layers * a.seq_length**2


def test_flops_patch_backs_off_when_upstream_counts_the_window(monkeypatch):
    """If upstream's FLOPs already depend on window_size (a newer Megatron), the correction is not
    subtracted a second time; today's upstream does not, so it is applied."""
    import megatron.training.training as training_module

    from primus.backends.megatron.patches.olmo3_flops_patches import (
        olmo3_attention_flops_correction,
        patch_olmo3_flops,
        upstream_counts_window,
    )

    args = SimpleNamespace(
        num_layers=4,
        window_size=(3, 0),
        olmo3_sliding_window=4,
        olmo3_window_size_form="megatron",
        window_attn_skip_freq=4,
        seq_length=16,
        kv_channels=8,
        num_attention_heads=2,
    )
    corr = olmo3_attention_flops_correction(args, 1)
    assert corr > 0

    full = 1.0e9 + _causal_attention_flops(args)

    def window_blind(a, bs):
        return (1.0e9 + _causal_attention_flops(a)) * bs

    def window_aware(a, bs):
        return (1.0e9 + _causal_attention_flops(a) - (5.0 if a.window_size is not None else 0.0)) * bs

    assert not upstream_counts_window(window_blind, args)
    assert upstream_counts_window(window_aware, args)
    assert args.window_size == (3, 0) and args.window_attn_skip_freq == 4  # args untouched
    for fn, want in ((window_blind, full - corr), (window_aware, full - 5.0)):
        monkeypatch.setattr(training_module, "num_floating_point_operations", fn)
        patch_olmo3_flops(SimpleNamespace(extra={"backend_args": args}))
        assert training_module.num_floating_point_operations(args, 1) == pytest.approx(want)
        monkeypatch.setattr(training_module, "num_floating_point_operations", fn)
        patch_olmo3_flops(None)  # no built args: checked at the first call
        assert training_module.num_floating_point_operations(args, 1) == pytest.approx(want)
    # a later non-OLMo-3 run in the same process (args without the OLMo-3 form) is not corrected
    other = SimpleNamespace(**{k: v for k, v in vars(args).items() if k != "olmo3_window_size_form"})
    assert training_module.num_floating_point_operations(other, 1) == pytest.approx(full - 5.0)
    # an error in the deferred upstream check (inside training_log) does not stop training: it is
    # logged and the correction is turned off (upstream's count, never corrected unchecked)
    calls = []

    def flaky(a, bs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("not ready")
        return window_blind(a, bs)

    monkeypatch.setattr(training_module, "num_floating_point_operations", flaky)
    patch_olmo3_flops(None)
    assert training_module.num_floating_point_operations(args, 1) == pytest.approx(full)
    assert training_module.num_floating_point_operations(args, 1) == pytest.approx(full)
    # a second install in the same process with a window the correction cannot use: logged, the
    # installed wrapper is kept, the run goes on
    bad = SimpleNamespace(**{**vars(args), "window_attn_skip_freq": 1})
    installed = training_module.num_floating_point_operations
    patch_olmo3_flops(SimpleNamespace(extra={"backend_args": bad}))
    assert training_module.num_floating_point_operations is installed
    # an error at install time (args not final) is retried at the first count, not final
    calls.clear()
    monkeypatch.setattr(training_module, "num_floating_point_operations", flaky)
    patch_olmo3_flops(
        SimpleNamespace(extra={"backend_args": SimpleNamespace(**vars(args), padded_vocab_size=64)})
    )
    assert training_module.num_floating_point_operations(args, 1) == pytest.approx(full - corr)
    # an error in the correction itself (a window_size the correction cannot read) does not stop
    # training either: upstream's count is kept
    broken = SimpleNamespace(**{**vars(args), "window_size": "bogus"})
    assert training_module.num_floating_point_operations(broken, 1) == pytest.approx(full)
    # a second install (a new run in the process) resets the per-run state: checked afresh
    assert training_module.num_floating_point_operations(args, 1) == pytest.approx(full)
    patch_olmo3_flops(None)
    assert training_module.num_floating_point_operations(args, 1) == pytest.approx(full - corr)
    # ... and, with built args, the second install checks upstream at install time, as a first
    # install does (not only at the first count)
    from primus.backends.megatron.patches import olmo3_flops_patches as fp

    logs = []
    monkeypatch.setattr(fp, "log_rank_0", logs.append)
    installed = training_module.num_floating_point_operations
    patch_olmo3_flops(
        SimpleNamespace(extra={"backend_args": SimpleNamespace(**vars(args), padded_vocab_size=64)})
    )
    assert training_module.num_floating_point_operations is installed
    assert installed._olmo3_state == {"checked": True, "apply": True}
    assert any("count windowed attention FLOPs" in m for m in logs)


def test_flops_patch_backs_off_when_upstream_counts_attention_another_way(monkeypatch):
    """The correction assumes upstream counts causal attention as s / 2 keys per query. An upstream
    that counts another number (here s keys: no causal halving) is detected by its seq_length**2
    coefficient, and the correction is not applied."""
    import megatron.training.training as training_module

    from primus.backends.megatron.patches.olmo3_flops_patches import (
        patch_olmo3_flops,
        upstream_causal_keys_ok,
    )

    args = SimpleNamespace(
        num_layers=4,
        window_size=(3, 0),
        olmo3_sliding_window=4,
        olmo3_window_size_form="megatron",
        window_attn_skip_freq=4,
        seq_length=16,
        kv_channels=8,
        num_attention_heads=2,
        padded_vocab_size=64,
    )

    def half(a, bs):
        return (7.0 + 3.0 * a.seq_length + _causal_attention_flops(a)) * bs

    def no_halving(a, bs):
        return (7.0 + 2 * _causal_attention_flops(a)) * bs

    assert upstream_causal_keys_ok(half, args) and not upstream_causal_keys_ok(no_halving, args)
    monkeypatch.setattr(training_module, "num_floating_point_operations", no_halving)
    patch_olmo3_flops(SimpleNamespace(extra={"backend_args": args}))
    assert training_module.num_floating_point_operations(args, 1) == pytest.approx(no_halving(args, 1))


def test_flops_patch_waits_for_padded_vocab_size(monkeypatch):
    """At ``before_train`` the tokenizer has not set ``padded_vocab_size``: the patch must install
    and run the upstream check at the first count, not call upstream with None (the L4 runs logged
    "unsupported operand type(s) for *: 'int' and 'NoneType'")."""
    import megatron.training.training as training_module

    from primus.backends.megatron.patches.olmo3_flops_patches import (
        olmo3_attention_flops_correction,
        patch_olmo3_flops,
    )

    args = SimpleNamespace(
        num_layers=4,
        window_size=(3, 0),
        olmo3_sliding_window=4,
        olmo3_window_size_form="megatron",
        window_attn_skip_freq=4,
        seq_length=16,
        kv_channels=8,
        num_attention_heads=2,
        padded_vocab_size=None,
    )

    def upstream(a, bs):
        return (1.0e9 + _causal_attention_flops(a)) * bs + a.padded_vocab_size * 2

    from primus.backends.megatron.patches import olmo3_flops_patches as fp

    logs, seen = [], []
    monkeypatch.setattr(fp, "log_rank_0", logs.append)
    monkeypatch.setattr(fp, "warning_rank_0", logs.append)

    def spy(a, bs):
        seen.append(getattr(a, "padded_vocab_size", "absent"))
        return upstream(a, bs)

    monkeypatch.setattr(training_module, "num_floating_point_operations", spy)
    patch_olmo3_flops(SimpleNamespace(extra={"backend_args": args}))
    # upstream is not called at install time (padded_vocab_size None): no failed check logged
    assert seen == [] and not any("cannot check upstream" in m for m in logs)
    assert any("upstream check runs at the first FLOPs count" in m for m in logs)
    args.padded_vocab_size = 128
    want = 1.0e9 + _causal_attention_flops(args) + 256 - olmo3_attention_flops_correction(args, 1)
    assert training_module.num_floating_point_operations(args, 1) == pytest.approx(want)


def test_upstream_flops_ignore_the_window(monkeypatch):
    """Pins today's upstream by behaviour: the real num_floating_point_operations (unwrapped) gives
    the same count with and without the OLMo-3 window (upstream_counts_window, the probe the patch
    runs), so the OLMo-3 correction is needed and is applied once."""
    import sys

    import megatron.training.training as training_module
    from megatron.training.arguments import parse_args

    from primus.backends.megatron.patches.olmo3_flops_patches import (
        _installed_wrapper,
        upstream_causal_keys_ok,
        upstream_counts_window,
    )

    fn = training_module.num_floating_point_operations
    while _installed_wrapper(fn) is not None:
        fn = fn.__wrapped__
    monkeypatch.setattr(sys, "argv", ["test"])
    args = parse_args(ignore_unknown_args=True)  # every upstream default, not a hand-picked subset
    for k, v in dict(
        num_layers=8,
        hidden_size=256,
        ffn_hidden_size=512,
        num_attention_heads=4,
        num_query_groups=4,
        kv_channels=64,
        seq_length=1024,
        padded_vocab_size=1024,
        swiglu=True,
        group_query_attention=False,
        window_size=(127, 0),
        olmo3_sliding_window=128,
        olmo3_window_size_form="megatron",
        window_attn_skip_freq=4,
    ).items():
        setattr(args, k, v)
    assert fn(args, 1) > 0
    assert not upstream_counts_window(fn, args)
    assert upstream_causal_keys_ok(fn, args)  # s / 2 keys per query, the convention corrected


def test_convert_hook_switch_values(monkeypatch, tmp_path):
    """native_ckpt_convert: off / no / "false" mean False (not bool("off") == True). The switch
    parser itself is tested in tests/unit_tests/core/utils/test_arg_utils.py."""
    mod = _load_convert_hook(monkeypatch)
    cfg = tmp_path / "c.yaml"
    seen = {}
    for v in ("off", "no", '"false"'):
        monkeypatch.setattr(mod, "load_primus_config", lambda *a: None)
        params = SimpleNamespace(native_ckpt_convert=v.strip('"'))
        monkeypatch.setattr(
            mod,
            "get_module_config",
            lambda cfg_, name: SimpleNamespace(params=params) if name == "post_trainer" else None,
        )
        seen[v] = mod.read_native_opts(str(cfg), [])["enabled"]
        assert mod.read_native_opts(str(cfg))["enabled"] is False  # without a command line too
    assert seen == {"off": False, "no": False, '"false"': False}


_OLMO3_HF_CFG = {
    "model_type": "olmo3",
    "torch_dtype": "bfloat16",
    "rope_theta": 500000,
    "max_position_embeddings": 65536,
    "rope_scaling": {"rope_type": "yarn", "factor": 8.0, "original_max_position_embeddings": 8192},
}


def _olmo3_hf_dir(tmp_path, name="hf"):
    import json

    hf = tmp_path / name
    hf.mkdir(parents=True)
    (hf / "config.json").write_text(json.dumps(_OLMO3_HF_CFG))
    (hf / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "model-1.safetensors"}}))
    (hf / "model-1.safetensors").write_bytes(b"w" * 8)
    return hf


def test_convert_hook_olmo3_reuses_only_a_matching_checkpoint(monkeypatch, tmp_path, capsys):
    """OLMo-3: rank 0 writes a record of the conversion settings and source after the checkpoint; a
    checkpoint is reused only if its record matches. The record has the converter version, the
    resolved hf_path, a weights fingerprint, dtype, vocab padding and the saved RoPE layout. Settings
    that change nothing saved (seq_length) and side files of the HF directory do not block reuse.
    A refusal writes <out>.failed so the other nodes stop too."""
    import json

    from primus.backends.megatron.checkpoint import native_flow_olmo3 as flow

    mod = _load_convert_hook(monkeypatch)
    hf = _olmo3_hf_dir(tmp_path)
    out = tmp_path / "out"
    opts = dict(
        enabled=None,
        out=str(out),
        seq_length=8192,
        make_vocab_size_divisible_by=128,
        olmo3_rope_layout="per_type",
    )
    converted = []

    def fake_convert(hf_path, megatron_path, o):
        assert not (Path(megatron_path) / flow.CONVERT_RECORD).exists()  # written after the save
        converted.append(dict(o))
        Path(megatron_path).mkdir(parents=True, exist_ok=True)
        (Path(megatron_path) / "latest_checkpointed_iteration.txt").write_text("1")

    monkeypatch.setattr(mod, "native_convert_checkpoint", fake_convert)
    monkeypatch.setenv("NODE_RANK", "0")
    mod._native_main(str(hf), dict(opts))
    record = json.loads((out / flow.CONVERT_RECORD).read_text())
    assert record == {
        "family": "olmo3",
        "converter_version": flow.CONVERTER_VERSION,
        "hf_path": os.path.realpath(hf),
        "hf_config_sha256": record["hf_config_sha256"],
        "hf_weights_fingerprint": flow.weights_fingerprint(str(hf)),
        "dtype": "bf16",
        "make_vocab_size_divisible_by": 128,
        "olmo3_rope_layout": "per_type",
    }
    assert Path(f"{out}.done").exists() and not Path(f"{out}.converting.lock").exists()
    assert not flow.failed_marker(out).exists()
    (hf / "training_args.bin.txt").write_bytes(b"x")
    (hf / "optimizer.pt").write_bytes(b"y")
    for change in (
        {},
        dict(seq_length=4096),
        dict(olmo3_rope_layout=None),
    ):  # None: per_type for a flat config
        mod._native_main(str(hf), {**opts, **change})
        assert "skipping conversion" in capsys.readouterr().out
    assert len(converted) == 1
    for change in (
        dict(make_vocab_size_divisible_by=64),
        dict(dtype="fp32"),
        dict(olmo3_rope_layout="yarn_all"),
    ):
        with pytest.raises(SystemExit):
            mod._native_main(str(hf), {**opts, **change})
        assert "cannot be reused" in capsys.readouterr().err
        assert "cannot be reused" in json.loads(flow.failed_marker(out).read_text())["reason"]
    # another HF dir with the same basename (same config.json, other weights) is not reused
    other = _olmo3_hf_dir(tmp_path / "elsewhere")
    (other / "model-1.safetensors").write_bytes(b"w" * 9)
    with pytest.raises(SystemExit):
        mod._native_main(str(other), dict(opts))
    err = capsys.readouterr().err
    assert "hf_weights_fingerprint" in err and "hf_path:" not in err  # hf_path is not compared
    # an older converter's checkpoint is not reused
    record["converter_version"] = 0
    (out / flow.CONVERT_RECORD).write_text(json.dumps(record))
    with pytest.raises(SystemExit):
        mod._native_main(str(hf), dict(opts))
    assert "converter_version" in capsys.readouterr().err
    (out / flow.CONVERT_RECORD).unlink()  # a killed conversion (tracker, no record): not reused
    with pytest.raises(SystemExit):
        mod._native_main(str(hf), dict(opts))
    assert "has no primus_native_convert.json" in capsys.readouterr().err
    assert len(converted) == 1


def test_convert_hook_olmo3_conversion_failure_writes_marker(monkeypatch, tmp_path):
    """Rank 0: a conversion that raises leaves <out>.failed with the reason (and no lock); a new
    conversion removes an old marker first."""
    import json

    from primus.backends.megatron.checkpoint import native_flow_olmo3 as flow

    mod = _load_convert_hook(monkeypatch)
    hf = _olmo3_hf_dir(tmp_path)
    out = tmp_path / "out"

    def boom(*a):
        raise RuntimeError("shape mismatch in layer 3")

    monkeypatch.setattr(mod, "native_convert_checkpoint", boom)
    monkeypatch.setenv("NODE_RANK", "0")
    for key in flow.LAUNCH_ID_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "77")
    with pytest.raises(RuntimeError, match="shape mismatch"):
        mod._native_main(str(hf), dict(enabled=None, out=str(out)))
    data = json.loads(flow.failed_marker(out).read_text())
    assert data["reason"] == "RuntimeError: shape mismatch in layer 3" and data["launch"] == "SLURM_JOB_ID=77"
    assert not Path(f"{out}.converting.lock").exists()

    def ok(hf_path, megatron_path, o):
        assert not flow.failed_marker(out).exists()  # removed before the conversion
        Path(megatron_path).mkdir(parents=True, exist_ok=True)
        (Path(megatron_path) / "latest_checkpointed_iteration.txt").write_text("1")

    monkeypatch.setattr(mod, "native_convert_checkpoint", ok)
    mod._native_main(str(hf), dict(enabled=None, out=str(out)))
    assert (out / flow.CONVERT_RECORD).exists()


def test_convert_hook_olmo3_other_nodes_wait_for_a_ready_checkpoint(monkeypatch, tmp_path):
    """OLMo-3, node 1: it waits until the checkpoint has its tracker and a matching record; an old
    .done does not release it, and a .done is not needed (a ready checkpoint ends the wait). The
    timeout is native_ckpt_wait_timeout (default 3600 s), counted on a monotonic clock, and its
    message says what was seen."""
    import json

    from primus.backends.megatron.checkpoint import native_flow_olmo3 as flow

    mod = _load_convert_hook(monkeypatch)
    hf = _olmo3_hf_dir(tmp_path)
    out = tmp_path / "out"
    Path(f"{out}.done").touch()  # an earlier launch's marker
    want = flow.convert_record(str(hf), {}, "bf16")
    clock, steps = [0.0], []
    lock = Path(f"{out}.converting.lock")

    def sleep(s):
        steps.append(s)
        clock[0] += s
        if len(steps) == 1:
            lock.touch()  # rank 0 converting
        if len(steps) == 2:  # rank 0: weights + tracker, record not written yet
            out.mkdir()
            (out / "latest_checkpointed_iteration.txt").write_text("1")
        if len(steps) == 4:  # rank 0: record written last
            (out / flow.CONVERT_RECORD).write_text(json.dumps(want))
            lock.unlink()

    monkeypatch.setattr(flow.time, "sleep", sleep)
    monkeypatch.setattr(flow.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(mod, "native_convert_checkpoint", lambda *a: pytest.fail("node 1 converted"))
    monkeypatch.setenv("NODE_RANK", "1")
    mod._native_main(str(hf), dict(enabled=None, out=str(out)))
    assert len(steps) == 4
    steps.clear()
    mod._native_main(str(hf), dict(enabled=None, out=str(out)))  # ready: no wait
    assert steps == []

    out2 = tmp_path / "out2"
    monkeypatch.setattr(flow.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + s))
    clock[0] = 0.0
    with pytest.raises(
        TimeoutError, match="Timeout after 3600 s.*no checkpoint yet.*native_ckpt_wait_timeout"
    ):
        mod._native_main(str(hf), dict(enabled=None, out=str(out2)))
    assert 3600 <= clock[0] < 3610
    clock[0] = 0.0
    with pytest.raises(TimeoutError, match="Timeout after 120 s"):
        mod._native_main(str(hf), dict(enabled=None, out=str(out2), wait_timeout="120"))
    assert 120 <= clock[0] < 130


def test_convert_hook_olmo3_other_nodes_fail_fast(monkeypatch, tmp_path, capsys):
    """OLMo-3, node 1 stops (exit 1, with the reason) instead of waiting for the timeout when
    * rank 0 writes <out>.failed for this launch (the conversion raised); or
    * the checkpoint has a tracker, no lock and a record that does not match (rank 0 refuses it
      too; node 1 computes the reason itself).
    A marker of another launch (other SLURM_JOB_ID) does not stop it."""
    import json

    from primus.backends.megatron.checkpoint import native_flow_olmo3 as flow

    mod = _load_convert_hook(monkeypatch)
    hf = _olmo3_hf_dir(tmp_path)
    out = tmp_path / "out"
    clock = [0.0]
    monkeypatch.setattr(flow.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(mod, "native_convert_checkpoint", lambda *a: pytest.fail("node 1 converted"))
    monkeypatch.setenv("NODE_RANK", "1")
    for key in flow.LAUNCH_ID_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "5")

    flow.failed_marker(out).write_text(json.dumps({"reason": "old", "launch": "SLURM_JOB_ID=4", "time": 0}))

    def sleep(s):
        clock[0] += s
        if clock[0] == 10:  # rank 0 of this launch fails
            marker = flow.failed_marker(out)
            marker.write_text(
                json.dumps({"reason": "RuntimeError: bad shard", "launch": "SLURM_JOB_ID=5", "time": 1})
            )
            later = marker.stat().st_mtime + 1  # a coarse mtime could equal the wait's start
            os.utime(marker, (later, later))

    monkeypatch.setattr(flow.time, "sleep", sleep)
    with pytest.raises(SystemExit):
        mod._native_main(str(hf), dict(enabled=None, out=str(out)))
    assert "rank 0 failed to convert the checkpoint: RuntimeError: bad shard" in capsys.readouterr().err
    assert clock[0] == 10  # the old launch's marker did not stop it, this one did at once

    flow.failed_marker(out).unlink()
    out.mkdir()
    (out / "latest_checkpointed_iteration.txt").write_text("1")  # stale: no record, no lock
    monkeypatch.setattr(flow.time, "sleep", lambda s: pytest.fail("waited on a stale checkpoint"))
    with pytest.raises(SystemExit):
        mod._native_main(str(hf), dict(enabled=None, out=str(out)))
    assert "cannot be reused: it has no primus_native_convert.json" in capsys.readouterr().err


def test_convert_hook_olmo3_failure_marker_without_launch_id(monkeypatch, tmp_path):
    """Without a launch id, a marker counts only if the storage wrote it after the wait started
    (a storage time, no slack): an earlier launch's marker is never taken; no storage time, no marker."""
    import json

    from primus.backends.megatron.checkpoint import native_flow_olmo3 as flow

    for key in flow.LAUNCH_ID_KEYS:
        monkeypatch.delenv(key, raising=False)
    out = tmp_path / "out"
    flow.write_failure(out, "boom")
    assert json.loads(flow.failed_marker(out).read_text())["launch"] is None
    now = flow.failed_marker(out).stat().st_mtime
    assert flow.read_failure(out, since=now - 1) == "boom"
    assert flow.read_failure(out, since=now) is None  # written before (or as) the wait started
    assert flow.read_failure(out, since=None) is None
    since = flow._storage_now(out, 1)
    assert since is not None and not Path(f"{out}.waiting.node1").exists()


def test_olmo3_conversion_lock_has_an_owner(monkeypatch, tmp_path):
    """Rank 0 takes <out>.converting.lock with O_EXCL and its owner. A lock of a dead process on
    this host is replaced; a lock held by another host (or a live process) stops rank 0 at once with
    the owner and what to do, and writes <out>.failed so the other nodes stop too. The conversion
    goes to <out>.partial and is renamed to <out> with its record, so a killed conversion never
    leaves a partial <out>; its <out>.partial is deleted by the next owner."""
    import json
    import socket
    from types import SimpleNamespace

    from primus.backends.megatron.checkpoint import native_flow_olmo3 as flow

    monkeypatch.setattr(flow, "convert_record", lambda *a: {"family": "olmo3"})
    log = SimpleNamespace(info=lambda m: None, error=lambda m: None, success=lambda m: None)
    out = tmp_path / "out"
    seen = []

    def convert(hf, path, opts):
        seen.append(path)
        assert json.loads(flow.lock_path(out).read_text())["pid"] == os.getpid()
        Path(path).mkdir(parents=True)
        (Path(path) / "latest_checkpointed_iteration.txt").write_text("1")

    # a stale lock (dead pid on this host) and a partial directory of a killed conversion
    flow.lock_path(out).write_text(
        json.dumps({"host": socket.gethostname(), "pidns": flow._pid_namespace(), "pid": 2**22 + 12345})
    )
    flow.partial_path(out).mkdir()
    (flow.partial_path(out) / "junk").write_text("x")
    flow.run_native_flow("/hf", out, {}, convert, "bf16", 0, log)
    assert seen == [str(flow.partial_path(out))] and not flow.partial_path(out).exists()
    assert (out / flow.CONVERT_RECORD).exists() and not (out / "junk").exists()
    assert not flow.lock_path(out).exists()

    out2 = tmp_path / "out2"
    flow.lock_path(out2).write_text(json.dumps({"host": "other-node", "pid": 1}))
    with pytest.raises(flow.ConversionFailed, match="other-node.*delete the lock"):
        flow.run_native_flow("/hf", out2, {}, convert, "bf16", 0, log)
    assert "other-node" in json.loads(flow.failed_marker(out2).read_text())["reason"]
    assert flow.lock_path(out2).exists()  # not ours: left in place

    flow.lock_path(out2).unlink()

    def crash(hf, path, opts):
        Path(path).mkdir(parents=True)
        (Path(path) / "latest_checkpointed_iteration.txt").write_text("1")
        raise RuntimeError("killed")

    with pytest.raises(RuntimeError):
        flow.run_native_flow("/hf", out2, {}, crash, "bf16", 0, log)
    assert not out2.exists() and not flow.lock_path(out2).exists()  # <out> is never partial

    out3 = tmp_path / "out3"
    out3.mkdir()
    (out3 / "stray").write_text("x")  # not a checkpoint: refused, not overwritten
    with pytest.raises(flow.ConversionFailed, match="holds no checkpoint"):
        flow.run_native_flow("/hf", out3, {}, convert, "bf16", 0, log)


def test_olmo3_failure_marker_of_an_earlier_launch_with_the_same_id(monkeypatch, tmp_path):
    """A rerun outside a scheduler has the same MASTER_ADDR:MASTER_PORT id; an old marker (written
    before this node started waiting) is still not taken, a new one is."""
    from primus.backends.megatron.checkpoint import native_flow_olmo3 as flow

    for key in flow.LAUNCH_ID_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MASTER_ADDR", "node0")
    monkeypatch.setenv("MASTER_PORT", "29500")
    out = tmp_path / "out"
    flow.write_failure(out, "old launch failure")
    old = flow.failed_marker(out).stat().st_mtime
    assert flow.read_failure(out, since=old) is None
    assert flow.read_failure(out, since=None) is None
    assert flow.read_failure(out, since=old - 1) == "old launch failure"
    monkeypatch.setenv("MASTER_PORT", "29501")  # another launch's id: never taken
    assert flow.read_failure(out, since=old - 1) is None


def test_olmo3_weights_fingerprint_reads_the_weight_bytes(tmp_path):
    """Two checkpoints with the same index file and shard names / sizes but other weights (base vs.
    instruct) have different fingerprints; a copy has the same one."""
    from primus.backends.megatron.checkpoint import native_flow_olmo3 as flow

    dirs = []
    for name, fill in (("base", b"a"), ("instruct", b"b"), ("copy", b"a")):
        d = tmp_path / name
        d.mkdir()
        (d / "model.safetensors.index.json").write_text('{"weight_map": {"w": "model-1.safetensors"}}')
        (d / "model-1.safetensors").write_bytes(fill * (3 * 1024 * 1024 + 7))
        dirs.append(flow.weights_fingerprint(str(d)))
    assert dirs[0] != dirs[1] and dirs[0] == dirs[2]


def test_olmo3_failure_marker_launch_id_parts(monkeypatch):
    """The launch id joins every source, so another srun step or a requeue is another launch."""
    from primus.backends.megatron.checkpoint import native_flow_olmo3 as flow

    for key in flow.LAUNCH_ID_KEYS:
        monkeypatch.delenv(key, raising=False)
    assert flow._launch_id() is None
    monkeypatch.setenv("SLURM_JOB_ID", "5")
    monkeypatch.setenv("SLURM_STEP_ID", "0")
    first = flow._launch_id()
    monkeypatch.setenv("SLURM_STEP_ID", "1")
    assert flow._launch_id() != first
    monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
    monkeypatch.setenv("PRIMUS_LAUNCH_ID", "run-a")
    assert flow._launch_id().startswith("PRIMUS_LAUNCH_ID=run-a;SLURM_JOB_ID=5")


def test_olmo3_rope_layout_option_is_normalised_once(tmp_path):
    """converter_options and the record use the same normalised olmo3_rope_layout; a bad value is
    refused at option parsing."""
    from primus.backends.megatron.checkpoint import native_flow_olmo3 as flow

    assert flow.converter_options({"olmo3_rope_layout": " Per_Type "})["rope_layout"] == "per_type"
    assert "rope_layout" not in flow.converter_options({"olmo3_rope_layout": "null"})
    for bad in ("auto", "yarn"):
        with pytest.raises(ValueError, match="olmo3_rope_layout"):
            flow.converter_options({"olmo3_rope_layout": bad})


def test_olmo3_record_ignores_hf_path(tmp_path):
    """A node that sees the HF directory at another path reuses the checkpoint: hf_path is kept in
    the record for information, not compared."""
    import json

    from primus.backends.megatron.checkpoint import native_flow_olmo3 as flow

    out = tmp_path / "ckpt"
    out.mkdir()
    have = {"family": "olmo3", "hf_path": "/mnt/a/model", "hf_config_sha256": "x"}
    (out / flow.CONVERT_RECORD).write_text(json.dumps(have))
    assert flow.stale_checkpoint_reason(out, {**have, "hf_path": "/cache/model"}) is None
    assert "hf_config_sha256" in flow.stale_checkpoint_reason(out, {**have, "hf_config_sha256": "y"})


def test_olmo3_record_rope_layout(tmp_path):
    """The record's olmo3_rope_layout is what convert() saves: per_type for a flat config, the
    config's own layout for a per-layer-type config; auto and a conflict are refused."""
    import json

    from primus.backends.megatron.checkpoint import native_flow_olmo3 as flow

    hf = _olmo3_hf_dir(tmp_path)
    assert flow._record_rope_layout(str(hf), {}) == "per_type"
    assert flow._record_rope_layout(str(hf), dict(olmo3_rope_layout="yarn_all")) == "yarn_all"
    with pytest.raises(ValueError, match="not auto"):
        flow._record_rope_layout(str(hf), dict(olmo3_rope_layout="auto"))
    yarn = dict(_OLMO3_HF_CFG["rope_scaling"])
    cfg = dict(
        _OLMO3_HF_CFG, rope_scaling=None, rope_parameters={"full_attention": yarn, "sliding_attention": yarn}
    )
    (hf / "config.json").write_text(json.dumps(cfg))
    assert flow._record_rope_layout(str(hf), {}) == "yarn_all"
    with pytest.raises(ValueError, match="declares olmo3_rope_layout 'yarn_all'"):
        flow._record_rope_layout(str(hf), dict(olmo3_rope_layout="per_type"))


def test_convert_hook_layout_keys(monkeypatch, tmp_path):
    """Every family converts at TP = PP = EP = 1, as at the base hook (its opts had no layout
    keys); Megatron reshards at load. The training layout (``*_model_parallel_size``) is read only
    by the OLMo-3 flow's checks, never as the conversion layout of any family."""
    import json
    import sys
    import types

    mod = _load_convert_hook(monkeypatch)
    calls = []
    fake = types.ModuleType("primus.backends.megatron.checkpoint.native_loader_llama")
    fake.convert = lambda *a, **k: calls.append(k)
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    import primus.backends.megatron.checkpoint as pkg

    monkeypatch.setattr(pkg, "native_loader_llama", fake, raising=False)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama", "torch_dtype": "bfloat16"}))
    opts = dict(
        family="llama",
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size="2",
        expert_model_parallel_size=8,
        tensor_parallel_size=2,
    )
    mod.native_convert_checkpoint(str(tmp_path), str(tmp_path / "out"), opts)
    assert [
        (k["tensor_parallel_size"], k["pipeline_parallel_size"], k["expert_parallel_size"]) for k in calls
    ] == [(1, 1, 1)]
    # the other families keep TP = PP = EP = 1 and get a warning when the run trains otherwise
    msgs = []
    monkeypatch.setattr(mod, "log_info", msgs.append)
    mod._warn_single_shard_layout("deepseek", {"expert_model_parallel_size": 8})
    assert len(msgs) == 1 and "single-shard" in msgs[0] and "'expert_model_parallel_size': 8" in msgs[0]
    mod._warn_single_shard_layout("qwen3", {"tensor_model_parallel_size": "1"})
    mod._warn_single_shard_layout("qwen3", {})
    assert len(msgs) == 1
    mod._warn_single_shard_layout("llama", {"pipeline_model_parallel_size": "two"})
    assert len(msgs) == 2 and "unreadable" in msgs[1]


def test_convert_hook_other_families_keep_the_base_flow(monkeypatch, tmp_path):
    """Qwen3 / Llama / DeepSeek launches behave as before the port: TP / PP / EP > 1 (the shipped
    MoE SFT recipes use EP 8) still convert, in place, with no record, and an existing checkpoint is reused.
    """
    import json

    from primus.backends.megatron.checkpoint import native_flow_olmo3 as flow

    mod = _load_convert_hook(monkeypatch)
    hf = tmp_path / "hf"
    hf.mkdir()
    (hf / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe"})
    )
    out = tmp_path / "out"
    calls = []

    def fake_convert(hf_path, megatron_path, o):
        calls.append(megatron_path)
        Path(megatron_path).mkdir(parents=True, exist_ok=True)
        (Path(megatron_path) / "latest_checkpointed_iteration.txt").write_text("1")

    monkeypatch.setattr(mod, "native_convert_checkpoint", fake_convert)
    monkeypatch.setattr(flow, "convert_record", lambda *a: pytest.fail("record built for qwen3"))
    _stub_hook_config(
        mod,
        monkeypatch,
        str(hf),
        dict(out=str(out), tensor_model_parallel_size=1, expert_model_parallel_size=8),
    )
    monkeypatch.setattr(sys, "argv", ["hook", "--config", str(tmp_path / "x.yaml")])
    monkeypatch.setenv("NODE_RANK", "0")
    mod.main()
    assert calls == [str(out)] and (tmp_path / "out.done").exists()
    assert not (out / flow.CONVERT_RECORD).exists()
    mod.main()  # reused
    assert len(calls) == 1


def test_convert_hook_and_install_hook_decide_the_same_way(monkeypatch, tmp_path):
    """00_install_requirements.sh runs read_hook_settings + conversion_plan on the arguments
    execute_hooks.sh forwards, the same calls as 01's main(); e.g. --native_ckpt_convert false
    turns a Qwen3 run to the Bridge path (install) in both, and an OLMo-3 run is never the Bridge
    path. The install hook's Python block is run here as the shell runs it."""
    import json
    import re as re_
    import sys

    mod = _load_convert_hook(monkeypatch)
    qwen = tmp_path / "qwen"
    qwen.mkdir()
    (qwen / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    params = SimpleNamespace(hf_path=str(qwen), native_ckpt_convert=None, model_type="gpt")
    module = SimpleNamespace(name="post_trainer", params=params)
    monkeypatch.setattr(mod, "load_primus_config", lambda *a: SimpleNamespace(modules=[module]))
    plan = lambda cli: mod.conversion_plan(*mod.read_hook_settings("x.yaml", cli))  # noqa: E731
    assert plan(["train", "posttrain"]) == "native"
    assert plan(["train", "posttrain", "--native_ckpt_convert", "false"]) == "bridge"
    assert plan(["train", "posttrain", "--load", "/ck"]) == "skip"
    assert plan(["--hf_path", "allenai/Olmo-3-1025-7B", "--native_ckpt_convert", "false"]) == "refuse"
    assert plan(["--hf_path", "allenai/Olmo-3-1025-7B"]) == "native"
    assert plan(["--hf_path", "Qwen/Qwen3-8B"]) == "bridge"

    # the install hook's Python block, run with the hook module patched above: the same decisions
    script = _INSTALL_HOOK.read_text()
    block = re_.search(r"<<'PY'\n(.*?)\nPY\n", script, re_.S).group(1)
    import contextlib
    import importlib.util
    import io

    class _Loader:
        def exec_module(self, _module):
            pass

    monkeypatch.setattr(
        importlib.util, "spec_from_file_location", lambda *a: SimpleNamespace(loader=_Loader())
    )
    monkeypatch.setattr(importlib.util, "module_from_spec", lambda spec: mod)

    def install_decision(forwarded):
        monkeypatch.setattr(sys, "argv", ["-", "x.yaml", *forwarded])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(compile(block, "00_install_requirements.sh", "exec"), {"__name__": "__main__"})
        return out.getvalue().strip().splitlines()[-1]

    for cli, want in (
        (["train", "posttrain", "--config", "x.yaml"], "SKIP"),  # native
        (["train", "posttrain", "--config=x.yaml", "--native_ckpt_convert", "false"], "INSTALL"),
        (["--config", "x.yaml", "--load", "/ck", "--native_ckpt_convert", "false"], "SKIP"),
        (["--hf_path", "allenai/Olmo-3-1025-7B", "--native_ckpt_convert", "false"], "SKIP"),  # refused
        (["--native_ckpt_convert", "maybe"], "SETTINGS_ERROR"),
    ):
        assert install_decision(cli) == "__DECISION__:" + want, cli


@pytest.mark.parametrize(
    "fake_python, installs, message",
    [
        # an unknown failure (python3 without yaml): legacy fallback, install, reason logged
        ('echo "No module named yaml" >&2; exit 1', True, "no decision printed -> installing"),
        ('cat >/dev/null; echo "__REASON__:OSError: x"; echo "__DECISION__:INSTALL"', True, "OSError: x"),
        # a settings error 01 stops on: nothing installed
        (
            'cat >/dev/null; echo "__REASON__:ValueError: bad"; echo "__DECISION__:SETTINGS_ERROR"',
            False,
            "conversion settings error: ValueError: bad",
        ),
        ('cat >/dev/null; echo "__DECISION__:SKIP"', False, "Megatron-Bridge conversion will not run"),
    ],
)
def test_install_hook_falls_back_to_install_except_on_a_settings_error(
    tmp_path, fake_python, installs, message
):
    """00_install_requirements.sh: only a settings error (a ValueError from read_hook_settings, which
    01 stops on too) or a SKIP decision installs nothing; any other failure installs, and every
    failure logs its reason."""
    import subprocess

    script = _INSTALL_HOOK
    (tmp_path / "python3").write_text(f"#!/bin/bash\n{fake_python}\n")
    (tmp_path / "pip").write_text('#!/bin/bash\necho "FAKEPIP $*"\n')
    for exe in ("python3", "pip"):
        (tmp_path / exe).chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "DATA_PATH": str(tmp_path),
        "PRIMUS_SKIP_PIP": "",
    }
    out = subprocess.run(
        ["bash", str(script), "--config", "x.yaml"], env=env, capture_output=True, text=True, check=True
    )
    text = out.stdout + out.stderr
    assert message in text, text
    assert ("FAKEPIP" in text) is installs, text


@pytest.mark.parametrize(
    "argv, config",
    [
        (["train", "posttrain", "--config", "x.yaml"], "x.yaml"),
        (["train", "posttrain", "--config=x.yaml"], "x.yaml"),
        (["--config", "a.yaml", "--lr", "1", "--config=b.yaml"], "b.yaml"),  # the last one wins
        (["--config=a.yaml", "--config", "/abs/b.yaml"], "/abs/b.yaml"),
    ],
)
def test_install_hook_finds_the_config_in_both_forms(tmp_path, argv, config):
    """00_install_requirements.sh takes --config <file> and --config=<file> (the last one wins, as
    01's argparse): the decision runs on that file, so a native / skip run installs nothing."""
    import subprocess

    script = _INSTALL_HOOK
    seen = tmp_path / "argv.txt"
    (tmp_path / "python3").write_text(
        f'#!/bin/bash\nprintf "%s\\n" "$@" > {seen}\ncat >/dev/null\necho "__DECISION__:SKIP"\n'
    )
    (tmp_path / "pip").write_text('#!/bin/bash\necho "FAKEPIP $*"\n')
    for exe in ("python3", "pip"):
        (tmp_path / exe).chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "DATA_PATH": str(tmp_path)}
    env.pop("PRIMUS_SKIP_PIP", None)
    out = subprocess.run(["bash", str(script), *argv], env=env, capture_output=True, text=True, check=True)
    text = out.stdout + out.stderr
    assert "FAKEPIP" not in text and "Megatron-Bridge conversion will not run" in text, text
    want = config if config.startswith("/") else str(_REPO_ROOT / config)
    # python3 - <config file> <the forwarded args>
    assert seen.read_text().splitlines()[:2] == ["-", want]


def test_install_hook_settings_error_is_only_a_read_hook_settings_value_error(tmp_path):
    """The decision script in 00_install_requirements.sh: a ValueError raised by read_hook_settings
    is a settings error (01 stops on the same one); a ValueError from conversion_plan, such as a
    corrupt HF config.json (json.JSONDecodeError), is not one: the dependencies are installed."""
    import re
    import subprocess
    import sys

    script = _INSTALL_HOOK.read_text()
    decision = re.search(r"<<'PY'\n(.*?)\nPY\n", script, re.S).group(1)
    hook_dir = tmp_path / "runner/helpers/hooks/train/posttrain/megatron"
    hook_dir.mkdir(parents=True)
    cases = {
        "settings": ("raise ValueError('native_ckpt_convert must be true or false')", "return 'native'"),
        "corrupt_config": ("return 'hf', None, {}", "import json; json.loads('{')"),
        "bridge": ("return 'hf', None, {}", "return 'bridge'"),
        "native": ("return 'hf', None, {}", "return 'native'"),
    }
    seen = {}
    for name, (read, plan) in cases.items():
        (hook_dir / "01_convert_checkpoints.py").write_text(
            f"def read_hook_settings(cfg, overrides):\n    {read}\n\n"
            f"def conversion_plan(hf_path, pretrained, opts):\n    {plan}\n"
        )
        out = subprocess.run(
            [sys.executable, "-", "x.yaml", "--config", "x.yaml"],
            input=decision,
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        seen[name] = re.search(r"__DECISION__:(\w+)", out).group(1)
        if name == "corrupt_config":
            assert "__REASON__:JSONDecodeError" in out, out
    assert seen == {
        "settings": "SETTINGS_ERROR",
        "corrupt_config": "INSTALL",
        "bridge": "INSTALL",
        "native": "SKIP",
    }


def test_install_hook_forwards_the_overrides_01_reads(tmp_path):
    """The decision script passes read_hook_settings the arguments execute_hooks.sh forwards with
    ``--config <file>`` / ``--config=<file>`` removed and every other token kept in order, the list
    01's main() reads after argparse removes --config; the decision follows those overrides."""
    import json
    import re
    import subprocess
    import sys

    script = _INSTALL_HOOK.read_text()
    decision = re.search(r"<<'PY'\n(.*?)\nPY\n", script, re.S).group(1)
    hook_dir = tmp_path / "runner/helpers/hooks/train/posttrain/megatron"
    hook_dir.mkdir(parents=True)
    record = tmp_path / "seen.json"
    (hook_dir / "01_convert_checkpoints.py").write_text(
        "import json\n\n"
        "def read_hook_settings(cfg, overrides):\n"
        f"    json.dump([cfg, overrides], open({str(record)!r}, 'w'))\n"
        "    off = overrides[overrides.index('--native_ckpt_convert') + 1] == 'false' "
        "if '--native_ckpt_convert' in overrides else False\n"
        "    return 'hf', None, {'enabled': False if off else None}\n\n"
        "def conversion_plan(hf_path, pretrained, opts):\n"
        "    return 'bridge' if opts['enabled'] is False else 'native'\n"
    )

    def run(*forwarded):
        out = subprocess.run(
            [sys.executable, "-", "/abs/x.yaml", *forwarded],
            input=decision,
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return re.search(r"__DECISION__:(\w+)", out).group(1), json.loads(record.read_text())

    want = ["train", "posttrain", "--native_ckpt_convert", "false", "--load=/ck", "x=1"]
    for forwarded in (
        ["train", "posttrain", "--config", "x.yaml", "--native_ckpt_convert", "false", "--load=/ck", "x=1"],
        ["train", "posttrain", "--native_ckpt_convert", "false", "--config=x.yaml", "--load=/ck", "x=1"],
        ["train", "posttrain", "--native_ckpt_convert", "false", "--load=/ck", "x=1", "--config", "x.yaml"],
    ):
        assert run(*forwarded) == ("INSTALL", ["/abs/x.yaml", want]), forwarded
    assert run("--config", "x.yaml") == ("SKIP", ["/abs/x.yaml", []])
    assert run("--config", "x.yaml", "--native_ckpt_convert", "true")[0] == "SKIP"


def test_convert_hook_cli_values_drop_only_the_hook_target(monkeypatch):
    """At most the two leading <group> <name> tokens are dropped; the rest is parsed as the trainer
    parses overrides (a bare --flag is true, dashed keys are underscore keys)."""
    mod = _load_convert_hook(monkeypatch)
    assert mod._cli_values(["train", "posttrain", "--load", "/ck"]) == {"load": "/ck"}
    assert mod._cli_values(["--load", "/ck"]) == {"load": "/ck"}
    assert mod._cli_values(["train", "posttrain", "x=1", "--native-ckpt-out", "/o"]) == {
        "x": 1,
        "native_ckpt_out": "/o",
    }
    assert mod._cli_values(["train", "posttrain", "--flag"]) == {"flag": True}
    # a bare path flag names no path: dropped, so it is not a configured checkpoint
    for key in ("load", "pretrained_checkpoint", "hf_path"):
        assert mod._cli_values(["train", "posttrain", f"--{key}"]) == {}
        assert mod._cli_values(["train", "posttrain", f"--{key}", "--x", "1"]) == {"x": 1}
    assert "stray" not in mod._cli_values(["train", "posttrain", "stray", "--a", "1"])


def test_convert_hook_resolves_an_olmo3_hub_id(monkeypatch, tmp_path):
    """The preset's tokenizer_model is a hub id: an OLMo-3 run resolves it to a local snapshot
    (cache, else download) and converts it natively; the output is named after the model, not the
    snapshot hash. The other families keep the fixed wait (no native_ckpt_wait_timeout)."""
    import json

    mod = _load_convert_hook(monkeypatch)
    snap = tmp_path / "snapshots" / "a81bae42"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(json.dumps({"model_type": "olmo3"}))
    resolve = mod._resolve_hub_snapshot
    assert resolve(str(snap), {}) is None  # a local path is not a hub id
    monkeypatch.setattr(mod, "_resolve_hub_snapshot", lambda hf, opts: str(snap))
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "data"))
    seen = []
    monkeypatch.setattr(mod, "_native_main", lambda hf, opts, fam=None: seen.append((hf, fam, opts["out"])))
    opts = dict(enabled=None, model_type="olmo3", out=None, tensor_model_parallel_size=1)
    mod._main("allenai/Olmo-3-1025-7B", None, opts)
    assert seen == [(str(snap), "olmo3", str(tmp_path / "data/megatron_checkpoints/Olmo-3-1025-7B_native"))]
    import inspect

    assert not hasattr(mod, "_wait_timeout")
    assert inspect.signature(mod.wait_for_conversion).parameters["timeout"].default == 600


def test_olmo3_hub_snapshot_is_complete_and_only_rank0_downloads(monkeypatch, tmp_path):
    """A snapshot needs config.json and its safetensors weights (every shard of the index). Rank 0
    reads the cache, then downloads; a waiting node only reads the cache and stops when rank 0
    writes <out>.failed."""
    import json
    import sys
    import types

    from primus.backends.megatron.checkpoint import native_flow_olmo3 as flow

    snap = tmp_path / "snap"
    snap.mkdir()
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    assert "config.json" in flow.snapshot_incomplete_reason(snap)
    (snap / "config.json").write_text(json.dumps({"model_type": "olmo3"}))
    assert "safetensors" in flow.snapshot_incomplete_reason(snap)
    # a shard landed before the index (download in progress): incomplete, not "any safetensors"
    shard = snap / "model-00001-of-00003.safetensors"
    shard.write_bytes(b"x")
    assert "index" in flow.snapshot_incomplete_reason(snap)
    shard.unlink()
    single = tmp_path / "single"
    single.mkdir()
    (single / "config.json").write_text(json.dumps({"model_type": "olmo3"}))
    (single / "model.safetensors").write_bytes(b"x")
    assert flow.snapshot_incomplete_reason(single) is None  # one unsharded file needs no index
    (snap / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "m-1.safetensors", "b": "m-2.safetensors"}})
    )
    (snap / "m-1.safetensors").write_bytes(b"x")
    assert "m-2.safetensors" in flow.snapshot_incomplete_reason(snap)
    (snap / "m-2.safetensors").write_bytes(b"x")
    assert flow.snapshot_incomplete_reason(snap) is None
    assert flow.is_hub_id("allenai/Olmo-3-1025-7B") and not flow.is_hub_id(str(snap))

    calls, config_reads = [], []
    cached = {"ok": False}

    def snapshot_download(repo, allow_patterns, local_files_only):
        if allow_patterns == ["config.json"]:  # the model-type check reads config.json only
            config_reads.append(repo)
            return str(foreign if repo == "Qwen/Qwen3-8B" else snap)
        calls.append(local_files_only)
        if local_files_only and not cached["ok"]:
            raise FileNotFoundError("not cached")
        return str(snap)

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=snapshot_download)
    )
    monkeypatch.delenv("PRIMUS_LAUNCH_ID", raising=False)
    log = types.SimpleNamespace(info=lambda m: None, error=lambda m: None, success=lambda m: None)
    out = tmp_path / "out"
    assert flow.resolve_hub_snapshot("org/olmo-3", out, {}, 0, log) == str(snap)
    assert calls == [True, False]
    calls.clear()
    cached["ok"] = True
    assert flow.resolve_hub_snapshot("org/olmo-3", out, {}, 1, log, poll=0.01) == str(snap)
    assert calls == [True]  # a waiting node never downloads

    # rank 0 cannot get it: it writes <out>.failed; a waiting node stops with it
    cached["ok"] = False
    calls.clear()
    monkeypatch.setenv("PRIMUS_LAUNCH_ID", "launch-7")
    (snap / "m-2.safetensors").unlink()
    assert flow.resolve_hub_snapshot("org/olmo-3", out, {}, 0, log) is None
    assert "m-2.safetensors" in flow.FailureWatch(out, None, grace=0).check()
    calls.clear()
    monkeypatch.setattr(flow, "MARKER_GRACE", 0)  # the marker predates the wait (same process)
    assert flow.resolve_hub_snapshot("org/olmo-3", out, {"wait_timeout": 30}, 1, log, poll=0.01) is None
    assert set(calls) == {True}

    # a hub id of another family (model_type olmo3 runs it as OLMo-3): rank 0 reads its config.json
    # only and stops, before any weights are downloaded; the waiting nodes stop with it
    calls.clear()
    errors = []
    log = types.SimpleNamespace(info=lambda m: None, error=errors.append, success=lambda m: None)
    assert flow.resolve_hub_snapshot("Qwen/Qwen3-8B", out, {}, 0, log) is None
    assert calls == [] and config_reads[-1] == "Qwen/Qwen3-8B"
    assert "model_type 'qwen3'" in errors[-1] and "qwen3" in flow.FailureWatch(out, None, grace=0).check()


def test_olmo3_lock_owner_rules(monkeypatch, tmp_path):
    """A lock is stale only on this host and PID namespace with a dead pid or this process's pid (a
    relaunch with the same pid); another namespace or host is a running owner. release_lock removes
    only a lock this process wrote. Rank 0 checks the output again under the lock: a checkpoint
    written meanwhile is reused, not converted again and not marked failed."""
    import json
    import socket
    from types import SimpleNamespace as NS

    from primus.backends.megatron.checkpoint import native_flow_olmo3 as flow

    host, ns = socket.gethostname(), flow._pid_namespace()
    assert flow._lock_is_stale({"host": host, "pidns": ns, "pid": os.getpid()})
    assert flow._lock_is_stale({"host": host, "pidns": ns, "pid": 2**22 + 12345})
    assert not flow._lock_is_stale({"host": host, "pidns": "pid:[1]", "pid": 2**22 + 12345})
    assert not flow._lock_is_stale({"host": "other", "pidns": ns, "pid": 2**22 + 12345})
    out = tmp_path / "out"
    flow.lock_path(out).write_text(json.dumps({"host": "other", "pid": 1, "token": "t"}))
    flow.release_lock(out)
    assert flow.lock_path(out).exists()
    flow.lock_path(out).unlink()

    monkeypatch.setattr(flow, "convert_record", lambda *a: {"family": "olmo3"})
    original = flow.acquire_lock

    def acquire(path):
        original(path)  # another launch finished just before this one got the lock
        out.mkdir()
        (out / "latest_checkpointed_iteration.txt").write_text("1")
        (out / flow.CONVERT_RECORD).write_text(json.dumps({"family": "olmo3"}))

    monkeypatch.setattr(flow, "acquire_lock", acquire)
    log = NS(info=lambda m: None, error=lambda m: None, success=lambda m: None)
    flow.run_native_flow("/hf", out, {}, lambda *a: pytest.fail("converted again"), "bf16", 0, log)
    assert not flow.failed_marker(out).exists() and not flow.lock_path(out).exists()


def test_olmo3_waiting_node_stops_on_early_refusals(monkeypatch, tmp_path):
    """A waiting node computes rank 0's early refusals itself (a non-empty output without a
    tracker); a marker rank 0 wrote before the node looked counts once it stays in place for the
    grace (rank 0 removes stale markers first); a marker of another launch is ignored."""
    from primus.backends.megatron.checkpoint import native_flow_olmo3 as flow

    monkeypatch.setattr(flow.time, "sleep", lambda s: pytest.fail("waited"))
    out = tmp_path / "out"
    out.mkdir()
    (out / "junk").write_text("x")
    with pytest.raises(flow.ConversionFailed, match="holds no checkpoint"):
        flow.wait_for_checkpoint(out, {"v": 1}, 30, node_rank=1)

    for key in flow.LAUNCH_ID_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "7")
    out2 = tmp_path / "out2"
    flow.write_failure(out2, "lock held")
    later = flow.failed_marker(out2).stat().st_mtime + 100
    assert flow.read_failure(out2, since=later) is None  # older than the wait: FailureWatch decides
    clock = [0.0]
    monkeypatch.setattr(flow.time, "monotonic", lambda: clock[0])
    watch = flow.FailureWatch(out2, later)
    assert watch.check() is None  # may be a stale marker that rank 0 has not removed yet
    clock[0] = flow.MARKER_GRACE - 1
    assert watch.check() is None
    clock[0] = flow.MARKER_GRACE
    assert "lock held" in watch.check()  # still in place: rank 0 stopped early
    monkeypatch.setenv("SLURM_JOB_ID", "8")
    assert flow.FailureWatch(out2, None, grace=0).check() is None  # another launch: never


def test_olmo3_failure_watch_ignores_a_marker_that_rank0_clears(monkeypatch, tmp_path):
    """A marker older than the wait with the same launch id (a retry inside one allocation, a
    constant PRIMUS_LAUNCH_ID) does not stop the node when rank 0 removes it; a new marker that
    rank 0 then writes stops the node at once, also in the same mtime second (after the grace)."""
    from primus.backends.megatron.checkpoint import native_flow_olmo3 as flow

    for key in flow.LAUNCH_ID_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PRIMUS_LAUNCH_ID", "fixed")
    clock = [0.0]
    monkeypatch.setattr(flow.time, "monotonic", lambda: clock[0])
    out = tmp_path / "out"
    flow.write_failure(out, "previous attempt")
    mtime = flow.failed_marker(out).stat().st_mtime
    watch = flow.FailureWatch(out, mtime + 1)
    assert watch.check() is None
    clock[0] = 10.0
    flow.clear_failure(out)  # rank 0's first step
    assert watch.check() is None
    clock[0] = flow.MARKER_GRACE + 20
    assert watch.check() is None  # the stale marker is gone: no stop
    flow.write_failure(out, "this attempt")  # rank 0 stops; the storage time is coarse
    os.utime(flow.failed_marker(out), (mtime, mtime))
    assert watch.check() is None
    clock[0] += flow.MARKER_GRACE
    assert "this attempt" in watch.check()
    later = mtime + 5
    os.utime(flow.failed_marker(out), (later + 1, later + 1))
    assert flow.FailureWatch(out, later).check() == "this attempt"  # newer than the wait: at once


def test_convert_hook_keeps_the_install_hook_api(monkeypatch, tmp_path):
    """get_checkpoint_config(cfg) and read_native_opts(cfg) still accept the config path only (the
    pre-port API; 00_install_requirements.sh now calls read_hook_settings + conversion_plan)."""
    import json
    from types import SimpleNamespace

    from primus.backends.megatron.checkpoint import native_flow_olmo3 as flow

    mod = _load_convert_hook(monkeypatch)
    hf = tmp_path / "hf"
    hf.mkdir()
    (hf / "config.json").write_text(json.dumps({"model_type": "olmo3"}))
    params = SimpleNamespace(native_ckpt_convert=None, hf_path=str(hf), pretrained_checkpoint=None)
    monkeypatch.setattr(
        mod,
        "load_primus_config",
        lambda *a: SimpleNamespace(modules=[SimpleNamespace(name="post_trainer", params=params)]),
    )
    assert mod.get_checkpoint_config("/c.yaml") == (str(hf), None)
    assert mod.read_native_opts("/c.yaml")["enabled"] is None
    params.tensor_model_parallel_size = "${TP:1}"
    with pytest.raises(ValueError, match="tensor_model_parallel_size"):
        flow.single_shard_conflict(mod.read_native_opts("/c.yaml", []))


def test_convert_hook_reads_cli_overrides_and_preset_defaults(monkeypatch, tmp_path):
    """Layout and converter settings come from the resolved module params (preset + trainer
    defaults + YAML overrides) with the trainer's CLI overrides on top."""
    from types import SimpleNamespace

    from primus.backends.megatron.checkpoint import native_flow_olmo3 as flow

    mod = _load_convert_hook(monkeypatch)
    params = SimpleNamespace(
        tensor_model_parallel_size=1,
        seq_length=4096,
        expert_model_parallel_size=1,
        make_vocab_size_divisible_by=128,
        native_ckpt_convert=None,
        hf_path="/hf",
        tokenizer_model="/tok",
        pretrained_checkpoint=None,
        load=None,
    )
    module = SimpleNamespace(name="post_trainer", params=params)
    monkeypatch.setattr(mod, "load_primus_config", lambda *a: SimpleNamespace(modules=[module]))
    opts = mod.read_native_opts("x.yaml", [])
    assert (opts["tensor_model_parallel_size"], opts["make_vocab_size_divisible_by"]) == (1, 128)
    assert "seq_length" not in opts  # no consumer: not collected
    opts = mod.read_native_opts(
        "x.yaml",
        [
            "train",
            "posttrain",
            "--tensor_model_parallel_size",
            "2",
            "--seq_length=8192",
            "--native_ckpt_convert",
            "false",
        ],
    )
    assert (opts["tensor_model_parallel_size"], opts["enabled"]) == (2, False)
    assert flow.single_shard_conflict(opts) is not None
    opts = mod.read_native_opts("x.yaml", ["--expert_model_parallel_size", "8"])
    assert "expert_model_parallel_size=8" in flow.single_shard_conflict(opts)
    # dashed CLI keys are the same keys
    opts = mod.read_native_opts("x.yaml", ["--native-ckpt-out", "/o", "--tensor-model-parallel-size=4"])
    assert (opts["out"], opts["tensor_model_parallel_size"]) == ("/o", 4)
    # null on the command line clears the YAML value of that key, as in training (the converter
    # then uses its default)
    for null in ("null", "None", "~", ""):
        opts = mod.read_native_opts("x.yaml", [f"--make_vocab_size_divisible_by={null}"])
        assert opts["make_vocab_size_divisible_by"] is None
    params.native_ckpt_out = "null"
    assert mod.read_native_opts("x.yaml", [])["out"] is None
    # per key: a YAML hf_path wins over a CLI tokenizer_model (training reads hf_path first)
    assert mod.get_checkpoint_config("x.yaml", ["--tokenizer_model", "/tok2"]) == ("/hf", None)
    # a CLI null clears the YAML value: the next key is used (hf_path -> tokenizer_model)
    assert mod.get_checkpoint_config("x.yaml", ["--hf_path", "null"]) == ("/tok", None)
    # --load null / pretrained_checkpoint=null clear a YAML checkpoint, so the hook converts
    params.load = "/yaml_ckpt"
    assert mod.get_checkpoint_config("x.yaml", []) == ("/hf", "/yaml_ckpt")
    assert mod.get_checkpoint_config("x.yaml", ["--load", "null"]) == ("/hf", None)
    params.load, params.pretrained_checkpoint = None, "/yaml_pre"
    assert mod.get_checkpoint_config("x.yaml", ["pretrained_checkpoint=null"]) == ("/hf", None)
    assert mod.get_checkpoint_config("x.yaml", ["--pretrained_checkpoint", "~"]) == ("/hf", None)
    params.pretrained_checkpoint = None
    assert "load" in mod._cli_values(["--load", "null"]) and "load" in mod._cli_values(["load=null"])
    # the checkpoint paths read the same CLI overrides as main()
    assert mod.get_checkpoint_config("x.yaml", []) == ("/hf", None)
    assert mod.get_checkpoint_config("x.yaml", ["--hf_path", "/hf2", "--pretrained_checkpoint", "/ck"]) == (
        "/hf2",
        "/ck",
    )
    assert mod.get_checkpoint_config("x.yaml", ["train", "posttrain", "--load=/ld"]) == ("/hf", "/ld")
    params.hf_path = None
    assert mod.get_checkpoint_config("x.yaml", ["--tokenizer_model", "/tok2"]) == ("/tok2", None)
    params.hf_path = "null"
    assert mod.get_checkpoint_config("x.yaml", []) == ("/tok", None)
    # the order the hook always had: params (hf_path, then tokenizer_model) before the module itself
    params.hf_path, module.hf_path = None, "/module_hf"
    assert mod.get_checkpoint_config("x.yaml", []) == ("/tok", None)
    params.tokenizer_model = None
    assert mod.get_checkpoint_config("x.yaml", []) == ("/module_hf", None)


def test_convert_hook_reads_every_family_the_same_way(monkeypatch, tmp_path, capsys):
    """One reading for all families: the command line applies (hf_path, load, native_ckpt_*), null /
    ~ / none are unset, and native_ckpt_convert is a switch. So a CLI hf_path decides the family
    (no YAML family leaks through), and a CLI --load skips the conversion for Qwen3 too."""
    import json

    mod = _load_convert_hook(monkeypatch)
    qwen = tmp_path / "qwen"
    qwen.mkdir()
    (qwen / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    olmo = tmp_path / "olmo"
    olmo.mkdir()
    (olmo / "config.json").write_text(json.dumps({"model_type": "olmo3"}))
    params = SimpleNamespace(hf_path=str(olmo), load="null", native_ckpt_convert=None, model_type="gpt")
    module = SimpleNamespace(name="post_trainer", params=params)
    loads = []
    monkeypatch.setattr(
        mod, "load_primus_config", lambda *a: loads.append(a) or SimpleNamespace(modules=[module])
    )
    monkeypatch.setattr(sys, "argv", ["hook", "--config", str(tmp_path / "x.yaml"), "--hf_path", str(qwen)])
    native = []
    monkeypatch.setattr(mod, "_native_main", lambda hf, opts, fam=None: native.append((hf, fam)))
    mod.main()
    assert native == [(str(qwen), "qwen3")] and len(loads) == 1  # the config is read once

    params.hf_path = str(qwen)
    monkeypatch.setattr(sys, "argv", ["hook", "--config", "x.yaml", "--hf_path", str(olmo)])
    native.clear()
    mod.main()
    assert native == [(str(olmo), "olmo3")]

    monkeypatch.setattr(sys, "argv", ["hook", "--config", "x.yaml", "--load", "/ck"])
    native.clear()
    mod.main()
    assert native == [] and "already configured: /ck" in capsys.readouterr().out
    assert mod.get_checkpoint_config("x.yaml") == (str(qwen), None)  # "null" is unset

    # main(): a value no family can read exits 1 with the reason, before any conversion work
    params.native_ckpt_convert = "maybe"
    monkeypatch.setattr(sys, "argv", ["hook", "--config", "x.yaml"])
    with pytest.raises(SystemExit) as e:
        mod.main()
    assert e.value.code == 1 and native == []
    assert (
        "Checkpoint conversion settings: native_ckpt_convert must be true or false" in capsys.readouterr().err
    )


def test_convert_hook_conversion_errors_keep_their_traceback(monkeypatch, tmp_path):
    """Only the settings reading turns a ValueError into a one-line message; a ValueError raised
    during the conversion propagates with its traceback."""
    mod = _load_convert_hook(monkeypatch)
    hf = tmp_path / "hf"
    hf.mkdir()
    (hf / "config.json").write_text('{"model_type": "qwen3"}')
    _stub_hook_config(mod, monkeypatch, str(hf), {})

    def boom(*a):
        raise ValueError("tensor shape mismatch")

    monkeypatch.setattr(mod, "_native_main", boom)
    monkeypatch.setattr(sys, "argv", ["hook", "--config", "x.yaml"])
    with pytest.raises(ValueError, match="tensor shape mismatch"):
        mod.main()


def test_config_int_reads_conversion_settings():
    """native_flow_olmo3.config_int: unset values give the default; YAML / CLI strings are read as
    ints with the shared arg_utils.as_int (tested in tests/unit_tests/core/utils/test_arg_utils.py);
    a bad value names the key."""
    from primus.backends.megatron.checkpoint import native_flow_olmo3 as flow

    assert flow.config_int({}, "k", 7) == 7
    assert flow.config_int({"k": None}, "k", 7) == 7 and flow.config_int({"k": "null"}, "k", 7) == 7
    assert flow.config_int({"k": "3"}, "k", 7) == 3 and flow.config_int({"k": 3}, "k") == 3
    with pytest.raises(ValueError, match="wait_timeout must be an integer"):
        flow.config_int({"wait_timeout": "two"}, "wait_timeout", 7)


def _gpu_cfg(**kw):
    import torch.nn.functional as F

    base = dict(
        num_layers=4,
        params_dtype=torch.bfloat16,
        bf16=True,
        use_cpu_initialization=False,
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        bias_activation_fusion=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        olmo3_sliding_window=4,
        window_attn_skip_freq=4,
        olmo3_yarn_rotary_scaling_factor=8.0,
        olmo3_yarn_original_max_position_embeddings=8,
    )
    base.update(kw)
    return _cfg(**base)


def _hf_inv_freq(dim, base, factor=1.0, orig_max=None, beta_fast=32.0, beta_slow=1.0):
    """RoPE / YaRN inverse frequencies written from HF ``_compute_yarn_parameters`` (independent
    of the code under test)."""
    pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim)
    if factor <= 1.0:
        return 1.0 / pos_freqs

    def corr_dim(rot):
        return (dim * math.log(orig_max / (rot * 2 * math.pi))) / (2 * math.log(base))

    low = max(math.floor(corr_dim(beta_fast)), 0)
    high = min(math.ceil(corr_dim(beta_slow)), dim - 1)
    if low == high:
        high += 0.001
    ramp = ((torch.arange(dim // 2, dtype=torch.float64) - low) / (high - low)).clamp(0, 1)
    extra = 1.0 - ramp
    return (1.0 / (factor * pos_freqs)) * (1 - extra) + (1.0 / pos_freqs) * extra


def _rms(x, w, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def _reference_attention(att, cfg, x, *, yarn, window, attention_factor=None):
    """fp32 eager OLMo-3 attention from the raw weights only (HF formulas): q/k/v projection,
    full-width q/k RMSNorm, rotate-half RoPE (HF YaRN with cos/sin * attention_factor),
    causal (optionally windowed) softmax, o_proj, post-norm. It calls no method of ``att``."""
    s, b, _ = x.shape
    nh, ng, hd = cfg.num_attention_heads, cfg.num_query_groups, cfg.kv_channels
    xf = x.float()
    w = att.linear_qkv.weight.float().reshape(ng, (nh // ng + 2) * hd, -1)
    wq = w[:, : (nh // ng) * hd].reshape(nh * hd, -1)
    wk = w[:, (nh // ng) * hd : (nh // ng + 1) * hd].reshape(ng * hd, -1)
    wv = w[:, (nh // ng + 1) * hd :].reshape(ng * hd, -1)
    q = _rms(xf @ wq.T, att.q_norm.weight.float()).reshape(s, b, nh, hd)
    k = _rms(xf @ wk.T, att.k_norm.weight.float()).reshape(s, b, ng, hd)
    v = (xf @ wv.T).reshape(s, b, ng, hd)
    if yarn:
        inv = _hf_inv_freq(hd, cfg.rotary_base_for_test, 8.0, 8)
        m = attention_factor if attention_factor is not None else 0.1 * math.log(8.0) + 1.0
    else:
        inv, m = _hf_inv_freq(hd, cfg.rotary_base_for_test), 1.0
    ang = torch.arange(s, dtype=torch.float64)[:, None] * inv[None]
    emb = torch.cat([ang, ang], -1)
    cos = (emb.cos() * m).float().to(x.device)[:, None, None]
    sin = (emb.sin() * m).float().to(x.device)[:, None, None]

    def rot(t):
        t1, t2 = t[..., : hd // 2], t[..., hd // 2 :]
        return t * cos + torch.cat([-t2, t1], -1) * sin

    q, k = rot(q), rot(k)
    qf, kf, vf = (t.permute(1, 2, 0, 3) for t in (q, k, v))  # [b, h, s, d]
    rep = nh // ng
    kf, vf = kf.repeat_interleave(rep, 1), vf.repeat_interleave(rep, 1)
    i = torch.arange(s, device=x.device)
    allowed = i[None, :] <= i[:, None]
    if window is not None:
        allowed &= i[None, :] > i[:, None] - window  # HF: window keys incl. the current token
    scores = (qf @ kf.transpose(-1, -2)) / math.sqrt(hd)
    scores = scores.masked_fill(~allowed, float("-inf"))
    ctx = (scores.softmax(-1) @ vf).permute(2, 0, 1, 3).reshape(s, b, nh * hd)
    out = ctx @ att.linear_proj.weight.float().T
    return _rms(out, att.post_attention_layernorm.weight.float())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TE layers need a GPU")
def test_layer_forward_window_yarn_and_post_norms(init_parallel_state):
    """seq (32) > window (4): the sliding layer matches a windowed eager reference and not a
    full-causal one; the full layer uses YaRN with softmax_scale * mscale**2."""
    from megatron.core.transformer.spec_utils import build_module

    from primus.backends.megatron.core.models.olmo3.olmo3_layer_specs import (
        get_olmo3_layer_spec,
    )
    from primus.backends.megatron.core.models.olmo3.olmo3_rotary import (
        Olmo3DualRotaryEmbedding,
    )

    torch.manual_seed(0)
    cfg = _gpu_cfg()
    assert cfg.olmo3_yarn_mscale == pytest.approx(0.1 * math.log(8.0) + 1.0)
    spec = get_olmo3_layer_spec()
    s, bsz, w = 32, 2, 4
    rope = Olmo3DualRotaryEmbedding(cfg, rotary_base=500000.0)(s)
    assert rope.shape[0] == 2 and not torch.equal(rope[0], rope[1])
    x = torch.randn(s, bsz, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)
    cfg.rotary_base_for_test = 500000.0
    for layer_number, sliding in ((1, True), (4, False)):
        layer = build_module(spec, config=cfg, layer_number=layer_number).cuda()
        att = layer.self_attention
        assert att.olmo3_sliding == sliding
        assert att.olmo3_use_yarn == (not sliding)  # per_type
        assert att.q_norm.weight.shape == (cfg.num_attention_heads * cfg.kv_channels,)
        # swiglu MLP without biases
        assert layer.mlp.linear_fc1.weight.shape[0] == 2 * cfg.ffn_hidden_size
        assert layer.mlp.linear_fc1.bias is None or layer.mlp.linear_fc1.bias.numel() == 0
        with torch.no_grad():
            for prm in (att.q_norm.weight, att.k_norm.weight, att.post_attention_layernorm.weight):
                prm.uniform_(0.5, 1.5)  # non-trivial norm weights
            got, bias = att(x, None, rotary_pos_emb=rope)
            assert bias is None
            dyn = SimpleNamespace(is_static_batching=lambda: False)
            with pytest.raises(NotImplementedError, match="dynamic-batching"):
                att(x, None, rotary_pos_emb=rope, inference_context=dyn)
            with pytest.raises(NotImplementedError, match="dynamic-batching"):
                att(x, None, None, dyn, rotary_pos_emb=rope)
            ref = _reference_attention(att, cfg, x, yarn=not sliding, window=w if sliding else None)
            torch.testing.assert_close(got.float(), ref, atol=4e-2, rtol=4e-2)
            if sliding:
                full = _reference_attention(att, cfg, x, yarn=False, window=None)
                assert (got.float() - full).abs().max() > 0.2  # the window matters
                yarn = _reference_attention(att, cfg, x, yarn=True, window=w)
                assert (got.float() - yarn).abs().max() > 0.05  # sliding layers: plain RoPE
            else:
                # without the YaRN attention factor the output differs
                plain = _reference_attention(att, cfg, x, yarn=True, window=None, attention_factor=1.0)
                assert (got.float() - plain).abs().max() > 0.05
            out = layer(x, attention_mask=None, rotary_pos_emb=rope)
            out = out[0] if isinstance(out, tuple) else out
            assert out.shape == x.shape and torch.isfinite(out.float()).all()
            # residual layout: x + post_attn(attn(x)), then + post_ffn(mlp(.))
            h = x + got
            mlp_out, mlp_bias = layer.mlp(h)
            assert mlp_bias is None
            torch.testing.assert_close(out.float(), (h + mlp_out).float(), atol=4e-2, rtol=4e-2)
            rms = mlp_out.float().pow(2).mean(-1).sqrt()
            torch.testing.assert_close(rms, torch.ones_like(rms), atol=2e-2, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TE layers need a GPU")
def test_layer_sharded_state_dict_torch_dist_roundtrip(init_parallel_state, tmp_path):
    """Every OLMo-3 parameter (post-norms, full-width q/k norms) is in the sharded state dict
    and survives a torch_dist save / load."""
    from megatron.core import dist_checkpointing
    from megatron.core.dist_checkpointing.mapping import ShardedBase
    from megatron.core.transformer.spec_utils import build_module

    from primus.backends.megatron.core.models.olmo3.olmo3_layer_specs import (
        get_olmo3_layer_spec,
    )

    torch.manual_seed(0)
    cfg = _gpu_cfg()
    spec = get_olmo3_layer_spec()
    layer = build_module(spec, config=cfg, layer_number=1).cuda()
    with torch.no_grad():
        for prm in layer.parameters():
            prm.normal_()
    prefix = "decoder.layers.0."
    sd = layer.sharded_state_dict(prefix=prefix)
    params = {prefix + n for n, _ in layer.named_parameters()}
    for name in (
        "mlp.post_feedforward_layernorm.weight",
        "self_attention.post_attention_layernorm.weight",
        "self_attention.q_norm.weight",
        "self_attention.k_norm.weight",
        "mlp.linear_fc1.weight",
    ):
        assert prefix + name in params and prefix + name in sd, name
    assert params <= set(sd), params - set(sd)
    assert all(isinstance(v, ShardedBase) for k, v in sd.items() if k in params)

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    dist_checkpointing.save(sd, str(ckpt))
    before = {k: v.detach().clone() for k, v in layer.state_dict().items() if torch.is_tensor(v)}
    fresh = build_module(spec, config=cfg, layer_number=1).cuda()
    loaded = dist_checkpointing.load(fresh.sharded_state_dict(prefix=prefix), str(ckpt))
    fresh.load_state_dict({k[len(prefix) :]: v for k, v in loaded.items()}, strict=False)
    after = fresh.state_dict()
    for name in (
        "mlp.post_feedforward_layernorm.weight",
        "self_attention.q_norm.weight",
        "self_attention.post_attention_layernorm.weight",
        "mlp.linear_fc1.weight",
    ):
        torch.testing.assert_close(after[name], before[name], rtol=0, atol=0)


def test_builder_refuses_spec_and_mtp_and_dispatch():
    from primus.backends.megatron.core.models.olmo3 import olmo3_builders
    from primus.core.utils.import_utils import get_model_provider

    prov = get_model_provider("olmo3")
    assert prov.func is olmo3_builders.model_provider
    assert prov.args == (olmo3_builders.olmo3_builder,)
    cfg = _cfg()
    base = dict(use_legacy_models=False, spec=None, mtp_num_layers=None)
    with pytest.raises(ValueError, match="spec"):
        olmo3_builders.olmo3_builder(SimpleNamespace(**{**base, "spec": ["x", "y"]}), True, True, config=cfg)
    with pytest.raises(ValueError, match="MTP"):
        olmo3_builders.olmo3_builder(SimpleNamespace(**{**base, "mtp_num_layers": 1}), True, True, config=cfg)
    with pytest.raises(ValueError, match="legacy"):
        olmo3_builders.olmo3_builder(
            SimpleNamespace(**{**base, "use_legacy_models": True}), True, True, config=cfg
        )
    with pytest.raises(TypeError):
        from megatron.core.transformer.transformer_config import TransformerConfig

        plain = TransformerConfig(num_layers=2, hidden_size=64, num_attention_heads=4)
        olmo3_builders.olmo3_builder(SimpleNamespace(**base), True, True, config=plain)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TE layers need a GPU")
def test_builder_builds_olmo3_model_forward(init_parallel_state, monkeypatch):
    # LanguageModule (attention_backend auto) refuses preset NVTE_* backend variables
    for var in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
        monkeypatch.delenv(var, raising=False)
    from primus.backends.megatron.core.models.olmo3 import olmo3_builders
    from primus.backends.megatron.core.models.olmo3.olmo3_model import Olmo3Model
    from primus.backends.megatron.core.transformer.olmo3.olmo3_attention import (
        Olmo3SelfAttention,
    )

    cfg = _gpu_cfg()
    args = SimpleNamespace(
        use_legacy_models=False,
        spec=None,
        mtp_num_layers=None,
        padded_vocab_size=128,
        max_position_embeddings=64,
        fp16_lm_cross_entropy=False,
        untie_embeddings_and_output_weights=True,
        position_embedding_type="rope",
        rotary_base=500000.0,
    )
    model = olmo3_builders.olmo3_builder(args, True, True, config=cfg).cuda()
    assert isinstance(model, Olmo3Model)
    atts = [l.self_attention for l in model.decoder.layers]
    assert all(isinstance(a, Olmo3SelfAttention) for a in atts)
    assert [a.olmo3_sliding for a in atts] == [True, True, True, False]
    with pytest.raises(ValueError, match="rope"):
        Olmo3Model(
            cfg,
            transformer_layer_spec=None,
            vocab_size=128,
            max_sequence_length=64,
            position_embedding_type="learned_absolute",
        )
    ids = torch.randint(0, 128, (2, 32), device="cuda")
    pos = torch.arange(32, device="cuda")[None].expand(2, -1)
    with torch.no_grad():
        logits = model(ids, pos, attention_mask=None)
    assert logits.shape == (2, 32, 128) and torch.isfinite(logits.float()).all()


# ---------------------------------------------------------------------------
# In-repo HF parity: tiny random transformers Olmo3ForCausalLM -> converter mapping -> Olmo3Model
# ---------------------------------------------------------------------------
class _DictStore:
    """``SafetensorsStore`` stand-in over an in-memory HF state dict."""

    def __init__(self, sd):
        self.sd = sd

    def has(self, name):
        return name in self.sd

    def get(self, name, dtype):
        return self.sd[name].to(dtype)


_TINY_HF = dict(
    vocab_size=128,
    hidden_size=256,
    intermediate_size=512,
    num_hidden_layers=4,
    num_attention_heads=4,
    max_position_embeddings=64,
    rms_norm_eps=1e-6,
    sliding_window=4,
    tie_word_embeddings=False,
    attention_bias=False,
    rope_theta=500000.0,
    layer_types=["sliding_attention"] * 3 + ["full_attention"],
    model_type="olmo3",
)
_TINY_YARN = {
    "rope_type": "yarn",
    "factor": 8.0,
    "original_max_position_embeddings": 8,
    "beta_fast": 32.0,
    "beta_slow": 1.0,
}


def _tiny_hf_olmo3_weights(*, yarn, kv_heads):
    """HF-form config dict + random HF-named weights for the tiny OLMo-3, with no transformers."""
    c = dict(_TINY_HF, num_key_value_heads=kv_heads, rope_scaling=dict(_TINY_YARN) if yarn else None)
    H, F, nh, V = c["hidden_size"], c["intermediate_size"], c["num_attention_heads"], c["vocab_size"]
    hd = H // nh
    g = torch.Generator().manual_seed(0)

    def w(*shape):
        return torch.randn(*shape, generator=g) * 0.05

    def norm(n):
        return torch.rand(n, generator=g) + 0.5

    sd = {"model.embed_tokens.weight": w(V, H), "model.norm.weight": norm(H), "lm_head.weight": w(V, H)}
    for i in range(c["num_hidden_layers"]):
        p = f"model.layers.{i}."
        sd.update(
            {
                p + "self_attn.q_proj.weight": w(nh * hd, H),
                p + "self_attn.k_proj.weight": w(kv_heads * hd, H),
                p + "self_attn.v_proj.weight": w(kv_heads * hd, H),
                p + "self_attn.o_proj.weight": w(H, nh * hd),
                p + "self_attn.q_norm.weight": norm(nh * hd),
                p + "self_attn.k_norm.weight": norm(kv_heads * hd),
                p + "post_attention_layernorm.weight": norm(H),
                p + "post_feedforward_layernorm.weight": norm(H),
                p + "mlp.gate_proj.weight": w(F, H),
                p + "mlp.up_proj.weight": w(F, H),
                p + "mlp.down_proj.weight": w(H, F),
            }
        )
    return c, sd


def _ref_olmo3_logits(sd, c, ids, layout):
    """fp32 OLMo-3 forward written from the HF modeling formulas over HF-named weights (no
    transformers, no code under test): post-norm blocks, full-width q/k RMSNorm, rotate-half RoPE,
    window on the sliding layers. ``layout`` picks which layers use YaRN: ``per_type`` the full
    layers only, ``yarn_all`` every layer (sliding and full share one rope in HF 5.x)."""
    nh, ng = c["num_attention_heads"], c["num_key_value_heads"]
    hd = c["hidden_size"] // nh
    rs = c.get("rope_scaling") or c.get("rope_parameters") or {}
    # transformers 5.x keeps rope_theta inside rope_parameters only
    eps, theta, win = c["rms_norm_eps"], c.get("rope_theta", rs.get("rope_theta")), c["sliding_window"]
    yarn = rs.get("rope_type") == "yarn"
    b, s = ids.shape
    pos = torch.arange(s, dtype=torch.float64)

    def cos_sin(use_yarn):
        if use_yarn:
            inv = _hf_inv_freq(
                hd,
                theta,
                rs["factor"],
                rs["original_max_position_embeddings"],
                rs.get("beta_fast", 32.0),
                rs.get("beta_slow", 1.0),
            )
            m = rs.get("attention_factor") or 0.1 * math.log(rs["factor"]) + 1.0
        else:
            inv, m = _hf_inv_freq(hd, theta), 1.0
        emb = torch.cat([pos[:, None] * inv[None]] * 2, -1)
        return (emb.cos() * m).float(), (emb.sin() * m).float()

    def rot(t, cs):  # t [b, h, s, d]
        cos, sin = cs
        t1, t2 = t[..., : hd // 2], t[..., hd // 2 :]
        return t * cos + torch.cat([-t2, t1], -1) * sin

    i = torch.arange(s)
    causal = i[None, :] <= i[:, None]
    x = sd["model.embed_tokens.weight"].float()[ids]  # [b, s, H]
    for L, kind in enumerate(c["layer_types"]):
        p = f"model.layers.{L}."
        W = lambda n: sd[p + n].float()  # noqa: E731
        sliding = kind == "sliding_attention"
        use_yarn = yarn and (layout == "yarn_all" or not sliding)
        q = _rms(x @ W("self_attn.q_proj.weight").T, W("self_attn.q_norm.weight"), eps)
        k = _rms(x @ W("self_attn.k_proj.weight").T, W("self_attn.k_norm.weight"), eps)
        v = x @ W("self_attn.v_proj.weight").T
        q, k, v = (t.reshape(b, s, -1, hd).transpose(1, 2) for t in (q, k, v))
        cs = cos_sin(use_yarn)
        q, k = rot(q, cs), rot(k, cs)
        k, v = k.repeat_interleave(nh // ng, 1), v.repeat_interleave(nh // ng, 1)
        allowed = causal & (i[None, :] > i[:, None] - win) if sliding else causal
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(hd)
        ctx = scores.masked_fill(~allowed, float("-inf")).softmax(-1) @ v
        attn = ctx.transpose(1, 2).reshape(b, s, nh * hd) @ W("self_attn.o_proj.weight").T
        x = x + _rms(attn, W("post_attention_layernorm.weight"), eps)
        mlp = torch.nn.functional.silu(x @ W("mlp.gate_proj.weight").T) * (x @ W("mlp.up_proj.weight").T)
        x = x + _rms(mlp @ W("mlp.down_proj.weight").T, W("post_feedforward_layernorm.weight"), eps)
    return _rms(x, sd["model.norm.weight"].float(), eps) @ sd["lm_head.weight"].float().T


def _tiny_hf_olmo3(transformers, *, yarn, kv_heads):
    cfg_cls = transformers.Olmo3Config
    rope = (
        {
            "rope_type": "yarn",
            "factor": 8.0,
            "original_max_position_embeddings": 8,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
        }
        if yarn
        else None
    )
    kw = dict(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=kv_heads,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        sliding_window=4,
        tie_word_embeddings=False,
        attention_bias=False,
        layer_types=["sliding_attention"] * 3 + ["full_attention"],
    )
    if int(transformers.__version__.split(".")[0]) >= 5:
        kw["rope_parameters"] = {"rope_type": "default", "rope_theta": 500000.0, **(rope or {})}
    else:
        kw.update(rope_theta=500000.0, rope_scaling=rope)
    cfg = cfg_cls(**kw)
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    model = transformers.Olmo3ForCausalLM(cfg).float().eval()
    with torch.no_grad():
        for name, prm in model.named_parameters():
            if "norm" in name:
                prm.uniform_(0.5, 1.5)  # non-trivial norm weights
    return model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TE layers need a GPU")
@pytest.mark.parametrize("yarn,kv_heads", [(False, 4), (True, 2)])
def test_logits_match_tiny_hf_olmo3(init_parallel_state, monkeypatch, yarn, kv_heads):
    """Logits of a tiny random transformers ``Olmo3ForCausalLM`` (seq 32 > window 4, GQA, YaRN)
    and of the Primus ``Olmo3Model`` built from ``derive_megatron_fields`` + the converter's
    weight mapping. The RoPE layout is ``auto`` here, so it follows the installed transformers
    (this is a reference build); training uses an explicit layout."""
    try:
        import transformers
    except ImportError:
        transformers = None
    use_hf = transformers is not None and hasattr(transformers, "Olmo3ForCausalLM")
    for var in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
        monkeypatch.delenv(var, raising=False)
    import torch.nn.functional as F

    from primus.backends.megatron.checkpoint.native_loader_olmo3 import (
        build_mcore_state_dict,
        derive_megatron_fields,
    )
    from primus.backends.megatron.core.models.olmo3 import olmo3_builders

    if use_hf:
        hf = _tiny_hf_olmo3(transformers, yarn=yarn, kv_heads=kv_heads)
        hf_cfg, hf_sd = hf.config.to_dict(), hf.state_dict()
        f = derive_megatron_fields(hf_cfg, rope_layout="auto")
    else:  # CI's transformers has no Olmo3: HF-named weights + the HF-formula reference forward
        hf_cfg, hf_sd = _tiny_hf_olmo3_weights(yarn=yarn, kv_heads=kv_heads)
        f = derive_megatron_fields(hf_cfg, rope_layout="per_type")
    assert f["window_size"] == (3, 0) and f["olmo3_sliding_window"] == 4
    cfg = _cfg(
        hidden_size=f["hidden_size"],
        num_attention_heads=f["num_attention_heads"],
        num_query_groups=f["num_query_groups"],
        kv_channels=f["kv_channels"],
        ffn_hidden_size=f["ffn_hidden_size"],
        num_layers=f["num_layers"],
        layernorm_epsilon=f["norm_epsilon"],
        window_size=f["window_size"],
        olmo3_sliding_window=f["olmo3_sliding_window"],
        window_attn_skip_freq=f["window_attn_skip_freq"],
        olmo3_yarn_rotary_scaling_factor=f["olmo3_yarn_rotary_scaling_factor"],
        olmo3_yarn_original_max_position_embeddings=f["olmo3_yarn_original_max_position_embeddings"],
        olmo3_yarn_beta_fast=f["olmo3_yarn_beta_fast"],
        olmo3_yarn_beta_slow=f["olmo3_yarn_beta_slow"],
        olmo3_yarn_attention_factor=f["olmo3_yarn_attention_factor"],
        olmo3_rope_layout=f["olmo3_rope_layout"],
        params_dtype=torch.float32,
        use_cpu_initialization=False,
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        bias_activation_fusion=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    args = SimpleNamespace(
        use_legacy_models=False,
        spec=None,
        mtp_num_layers=None,
        padded_vocab_size=128,
        max_position_embeddings=64,
        fp16_lm_cross_entropy=False,
        untie_embeddings_and_output_weights=f["untie_embeddings_and_output_weights"],
        position_embedding_type="rope",
        rotary_base=f["rotary_base"],
    )
    model = olmo3_builders.olmo3_builder(args, True, True, config=cfg).cuda().eval()
    margs = SimpleNamespace(
        params_dtype=torch.float32,
        num_attention_heads=f["num_attention_heads"],
        num_query_groups=f["num_query_groups"],
        kv_channels=f["kv_channels"],
        hidden_size=f["hidden_size"],
        num_layers=f["num_layers"],
        untie_embeddings_and_output_weights=f["untie_embeddings_and_output_weights"],
    )
    sd, consumed = build_mcore_state_dict(model, _DictStore(hf_sd), margs, f, lambda *a: None)
    assert set(hf_sd) - consumed <= {k for k in hf_sd if "rotary_emb" in k}, set(hf_sd) - consumed
    missing, unexpected = model.load_state_dict({k: v.cuda() for k, v in sd.items()}, strict=False)
    assert not unexpected and all(k.endswith("_extra_state") for k in missing), missing

    torch.manual_seed(1)
    ids = torch.randint(0, 128, (2, 32))
    with torch.no_grad():
        want = (
            hf(ids).logits.float()
            if use_hf
            else _ref_olmo3_logits(hf_sd, hf_cfg, ids, f["olmo3_rope_layout"])
        )
        got = model(ids.cuda(), torch.arange(32, device="cuda")[None].expand(2, -1), attention_mask=None)
    got = got.float().cpu()
    torch.testing.assert_close(got, want, atol=2e-3, rtol=2e-3)


# ---------------------------------------------------------------------------
# CPU numerics (no GPU / TE): local linear + DotProductAttention + torch RMSNorm
# ---------------------------------------------------------------------------
def _cpu_numerics(monkeypatch):
    """Route the hard-coded CUDA spots of upstream RoPE / softmax masks to the CPU."""
    import megatron.core.fusions.fused_softmax as fs

    # The ROCm image bakes NVTE_FLASH_ATTN=0; upstream's attention_backend "auto" check in
    # LanguageModule refuses it (training clears it in the attention-backend patch). The CPU model
    # uses no TE attention, so clear the variables here.
    for name in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(
        fs, "get_default_causal_mask", lambda sq: torch.triu(torch.ones(sq, sq), diagonal=1).bool()
    )

    def window_mask(sq, skv, window_size):
        m = torch.ones(sq, skv, dtype=torch.bool)
        mu = torch.triu(m, diagonal=skv - sq - window_size[0])
        return ~torch.tril(mu, diagonal=skv - sq + window_size[1])

    monkeypatch.setattr(fs, "get_sliding_window_causal_mask", window_mask)
    # dropout is 0; the CUDA RNG tracker fork around it is a no-op on CPU
    import contextlib

    from megatron.core import tensor_parallel

    class _Tracker:
        def fork(self, *a, **k):
            return contextlib.nullcontext()

    monkeypatch.setattr(tensor_parallel, "get_cuda_rng_tracker", lambda *a, **k: _Tracker())
    # With TE installed, TransformerBlock asks TE for its CPU-offload context, which makes a
    # torch.cuda.Stream even when offloading is off; on a host without a GPU that fails. Offloading
    # is off here, so take upstream's no-TE branch (nullcontext).
    import megatron.core.transformer.transformer_block as tb

    monkeypatch.setattr(tb, "get_cpu_offload_context", None)
    # The block's final norm is TENorm when TE is installed (TransformerBlock.LayerNormImpl), and TE
    # modules synchronise the GPU in get_extra_state; use upstream's torch norm, as without TE.
    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    monkeypatch.setattr(tb, "LayerNormImpl", WrappedTorchNorm)


def _cpu_layer_spec():
    """The OLMo-3 layer spec with the TE modules swapped for upstream's local ones."""
    from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
    from megatron.core.transformer.attention import SelfAttentionSubmodules
    from megatron.core.transformer.dot_product_attention import DotProductAttention
    from megatron.core.transformer.mlp import MLPSubmodules
    from megatron.core.transformer.spec_utils import ModuleSpec
    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    from primus.backends.megatron.core.models.olmo3.olmo3_layer_specs import (
        get_olmo3_layer_spec,
    )

    spec = get_olmo3_layer_spec()
    att, mlp = spec.submodules.self_attention, spec.submodules.mlp
    att = ModuleSpec(
        module=att.module,
        params={**att.params, "post_norm": WrappedTorchNorm},
        submodules=SelfAttentionSubmodules(
            linear_qkv=ColumnParallelLinear, core_attention=DotProductAttention, linear_proj=RowParallelLinear
        ),
    )
    mlp = ModuleSpec(
        module=mlp.module,
        params={**mlp.params, "post_norm": WrappedTorchNorm},
        submodules=MLPSubmodules(linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear),
    )
    spec.submodules.self_attention, spec.submodules.mlp = att, mlp
    return spec


def _cpu_cfg(**kw):
    import torch.nn.functional as F

    base = dict(
        olmo3_sliding_window=4,
        window_size=(3, 0),
        olmo3_yarn_rotary_scaling_factor=8.0,
        olmo3_yarn_original_max_position_embeddings=8,
        params_dtype=torch.float32,
        activation_func=F.silu,
        bias_activation_fusion=False,
        masked_softmax_fusion=False,
        gradient_accumulation_fusion=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        persist_layer_norm=False,
    )
    base.update(kw)
    return _cfg(**base)


def test_layer_forward_window_yarn_and_post_norms_cpu(init_parallel_state, monkeypatch):
    """CPU twin of the GPU layer test: seq (32) > window (4) on the local attention path."""
    from megatron.core.transformer.spec_utils import build_module

    from primus.backends.megatron.core.models.olmo3.olmo3_rotary import (
        Olmo3DualRotaryEmbedding,
    )

    _cpu_numerics(monkeypatch)
    torch.manual_seed(0)
    cfg = _cpu_cfg()
    spec = _cpu_layer_spec()
    s, bsz, w = 32, 2, 4
    rope = Olmo3DualRotaryEmbedding(cfg, rotary_base=500000.0)(s)
    assert rope.device.type == "cpu" and not torch.equal(rope[0], rope[1])
    x = torch.randn(s, bsz, cfg.hidden_size)
    cfg.rotary_base_for_test = 500000.0
    for layer_number, sliding in ((1, True), (4, False)):
        layer = build_module(spec, config=cfg, layer_number=layer_number)
        att = layer.self_attention
        assert att.olmo3_sliding == sliding and att.config is cfg
        core = att.core_attention
        assert core.config is cfg
        base = 1.0 / math.sqrt(cfg.kv_channels)
        want_scale = base if sliding else base * cfg.olmo3_yarn_mscale**2
        assert core.softmax_scale == pytest.approx(want_scale)
        # upstream readers of the attention's own scale (dynamic batching) see the same m**2
        assert att.softmax_scale == pytest.approx(want_scale)
        assert core.scale_mask_softmax.window_size == ((3, 0) if sliding else None)
        with torch.no_grad():
            for prm in (att.q_norm.weight, att.k_norm.weight, att.post_attention_layernorm.weight):
                prm.uniform_(0.5, 1.5)
            got, bias = att(x, None, rotary_pos_emb=rope)
            assert bias is None
            ref = _reference_attention(att, cfg, x, yarn=not sliding, window=w if sliding else None)
            torch.testing.assert_close(got.float(), ref, atol=1e-4, rtol=1e-4)
            if sliding:
                full = _reference_attention(att, cfg, x, yarn=False, window=None)
                assert (got - full).abs().max() > 0.05  # the window matters
                off = _reference_attention(att, cfg, x, yarn=False, window=w + 1)
                assert (got - off).abs().max() > 1e-3  # and it is W, not W + 1
            else:
                plain = _reference_attention(att, cfg, x, yarn=True, window=None, attention_factor=1.0)
                assert (got - plain).abs().max() > 1e-3
            out = layer(x, attention_mask=None, rotary_pos_emb=rope)
            out = out[0] if isinstance(out, tuple) else out
            h = x + got
            mlp_out, _ = layer.mlp(h)
            torch.testing.assert_close(out, h + mlp_out, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "yarn,kv_heads,layout",
    [(False, 4, "per_type"), (True, 2, "per_type"), (True, 2, "yarn_all"), (True, 4, "per_type")],
)
@pytest.mark.parametrize("source", ["weights", "transformers"])
def test_logits_match_tiny_hf_olmo3_cpu(init_parallel_state, monkeypatch, yarn, kv_heads, layout, source):
    """CPU end-to-end parity: HF config + HF-named weights -> derive_megatron_fields +
    build_mcore_state_dict -> Olmo3Model logits, against an HF-formula reference forward. It needs
    no transformers in the ``weights`` case, so that case always runs (CI pins a transformers without
    Olmo3). The ``transformers`` case uses a real HF Olmo3ForCausalLM (config, weights) and first pins
    the reference to its logits; the other RoPE layout must not match HF."""
    from primus.backends.megatron.checkpoint.native_loader_olmo3 import (
        build_mcore_state_dict,
        derive_megatron_fields,
    )
    from primus.backends.megatron.core.models.olmo3 import olmo3_builders
    from primus.backends.megatron.core.models.olmo3.olmo3_model import Olmo3Model

    _cpu_numerics(monkeypatch)
    torch.manual_seed(1)
    ids = torch.randint(0, 128, (2, 32))
    try:
        import transformers
    except ImportError:
        transformers = None
    if source == "transformers":
        if transformers is None or not hasattr(transformers, "Olmo3ForCausalLM"):
            pytest.skip("installed transformers has no Olmo3 (the 'weights' case still runs)")
        from primus.backends.megatron.core.models.olmo3.olmo3_transformer_config import (
            resolve_olmo3_rope_layout,
        )

        hf = _tiny_hf_olmo3(transformers, yarn=yarn, kv_heads=kv_heads)
        hf_cfg, hf_sd = hf.config.to_dict(), hf.state_dict()
        hf_layout = resolve_olmo3_rope_layout("auto")  # the layout the installed transformers implements
        with torch.no_grad():
            want_hf = hf(ids).logits.float()
        torch.testing.assert_close(
            _ref_olmo3_logits(hf_sd, hf_cfg, ids, hf_layout), want_hf, atol=1e-4, rtol=1e-4
        )
        if yarn:
            other = "per_type" if hf_layout == "yarn_all" else "yarn_all"
            assert (_ref_olmo3_logits(hf_sd, hf_cfg, ids, other) - want_hf).abs().max() > 1e-2
    else:
        hf_cfg, hf_sd = _tiny_hf_olmo3_weights(yarn=yarn, kv_heads=kv_heads)
    f = derive_megatron_fields(hf_cfg, rope_layout=layout)
    assert f["olmo3_rope_layout"] == layout
    assert f["window_size"] == (3, 0) and f["olmo3_sliding_window"] == 4
    cfg = _cpu_cfg(
        hidden_size=f["hidden_size"],
        num_attention_heads=f["num_attention_heads"],
        num_query_groups=f["num_query_groups"],
        kv_channels=f["kv_channels"],
        ffn_hidden_size=f["ffn_hidden_size"],
        num_layers=f["num_layers"],
        layernorm_epsilon=f["norm_epsilon"],
        window_size=f["window_size"],
        olmo3_sliding_window=f["olmo3_sliding_window"],
        window_attn_skip_freq=f["window_attn_skip_freq"],
        olmo3_yarn_rotary_scaling_factor=f["olmo3_yarn_rotary_scaling_factor"],
        olmo3_yarn_original_max_position_embeddings=f["olmo3_yarn_original_max_position_embeddings"],
        olmo3_yarn_beta_fast=f["olmo3_yarn_beta_fast"],
        olmo3_yarn_beta_slow=f["olmo3_yarn_beta_slow"],
        olmo3_yarn_attention_factor=f["olmo3_yarn_attention_factor"],
        olmo3_rope_layout=layout,
    )
    import primus.backends.megatron.core.models.olmo3.olmo3_layer_specs as ls

    cpu_spec = _cpu_layer_spec()
    monkeypatch.setattr(ls, "get_olmo3_layer_spec", lambda *a, **k: cpu_spec)
    monkeypatch.setattr(olmo3_builders, "get_olmo3_layer_spec", lambda *a, **k: cpu_spec)
    args = SimpleNamespace(
        use_legacy_models=False,
        spec=None,
        mtp_num_layers=None,
        padded_vocab_size=128,
        max_position_embeddings=64,
        fp16_lm_cross_entropy=False,
        untie_embeddings_and_output_weights=f["untie_embeddings_and_output_weights"],
        position_embedding_type="rope",
        rotary_base=f["rotary_base"],
    )
    model = olmo3_builders.olmo3_builder(args, True, True, config=cfg).eval()
    assert isinstance(model, Olmo3Model)
    # No Transformer Engine module (they synchronise the GPU in get_extra_state / state_dict),
    # including the block's final norm, so this runs on a host with TE but no GPU.
    te_modules = [
        n
        for n, mod in model.named_modules()
        if type(mod).__module__.startswith("transformer_engine") or type(mod).__name__.startswith("TE")
    ]
    assert not te_modules, te_modules

    def no_gpu(*a, **k):
        raise RuntimeError("No HIP GPUs are available")

    monkeypatch.setattr(torch.cuda, "synchronize", no_gpu)
    plan = (
        "S S S F+yarn"
        if yarn and layout == "per_type"
        else ("S+yarn S+yarn S+yarn F+yarn" if yarn else "S S S F")
    )
    assert olmo3_builders.olmo3_layer_plan(cfg) == f"W=4 rope={layout} {plan}"
    # what the built modules apply (the trainer smoke test checks the same report on GPU / TE)
    rep = olmo3_builders.olmo3_built_layer_report(model)
    base = 1.0 / math.sqrt(f["kv_channels"])
    m2 = cfg.olmo3_yarn_mscale**2
    assert [r["window"] for r in rep] == [[3, 0]] * 3 + ["full"]
    assert [r["rope"] for r in rep] == ["yarn" if "yarn" in tok else "plain" for tok in plan.split()]
    for r in rep:
        want = base * m2 if r["rope"] == "yarn" and yarn else base
        assert r["softmax_scale"] == pytest.approx(want, rel=1e-6), r
    margs = SimpleNamespace(
        params_dtype=torch.float32,
        num_attention_heads=f["num_attention_heads"],
        num_query_groups=f["num_query_groups"],
        kv_channels=f["kv_channels"],
        hidden_size=f["hidden_size"],
        num_layers=f["num_layers"],
        untie_embeddings_and_output_weights=f["untie_embeddings_and_output_weights"],
    )
    sd, consumed = build_mcore_state_dict(model, _DictStore(hf_sd), margs, f, lambda *a: None)
    assert set(hf_sd) - consumed <= {k for k in hf_sd if "rotary_emb" in k}, set(hf_sd) - consumed
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected and all(k.endswith("_extra_state") for k in missing), missing

    with torch.no_grad():
        got = model(ids, torch.arange(32)[None].expand(2, -1), attention_mask=None).float()
    want = _ref_olmo3_logits(hf_sd, hf_cfg, ids, layout)
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)
    if yarn:  # the layout is not interchangeable
        other = "per_type" if layout == "yarn_all" else "yarn_all"
        assert (got - _ref_olmo3_logits(hf_sd, hf_cfg, ids, other)).abs().max() > 1e-2


class _SetAttr:
    """Minimal monkeypatch stand-in for the spawned TP worker processes."""

    def setattr(self, obj, name, value):
        setattr(obj, name, value)

    def delenv(self, name, raising=True):
        import os

        os.environ.pop(name, None)


def _tp2_worker(rank, port, out_dir):
    """One rank of the real TP=2 QK-norm check (gloo, CPU)."""
    import os

    from megatron.core import parallel_state as ps
    from megatron.core.transformer.spec_utils import build_module

    from primus.backends.megatron.core.models.olmo3.olmo3_rotary import (
        Olmo3DualRotaryEmbedding,
    )

    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    torch.distributed.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", world_size=2, rank=rank
    )
    _cpu_numerics(_SetAttr())
    ps.initialize_model_parallel(tensor_model_parallel_size=2)
    try:
        torch.manual_seed(0)
        cfg = _cpu_cfg(tensor_model_parallel_size=2, num_query_groups=2)
        spec = _cpu_layer_spec()
        s = 16
        rope = Olmo3DualRotaryEmbedding(cfg, rotary_base=500000.0)(s)
        full = torch.load(os.path.join(out_dir, "full.pt"))
        res = {}
        for layer_number in (1, 4):  # sliding (RoPE + window) and full (YaRN) layers
            att = build_module(spec.submodules.self_attention, config=cfg, layer_number=layer_number)
            w = full[layer_number]
            shard = {
                "linear_qkv.weight": w["linear_qkv.weight"].chunk(2, 0)[rank],
                "linear_proj.weight": w["linear_proj.weight"].chunk(2, 1)[rank],
                "q_norm.weight": w["q_norm.weight"].chunk(2, 0)[rank],
                "k_norm.weight": w["k_norm.weight"].chunk(2, 0)[rank],
                "post_attention_layernorm.weight": w["post_attention_layernorm.weight"],
            }
            missing, unexpected = att.load_state_dict(shard, strict=False)
            assert not unexpected, unexpected
            assert att.q_norm.weight.shape[0] * 2 == cfg.num_attention_heads * cfg.kv_channels
            x = full["x"].clone().requires_grad_(True)
            out, _ = att(x, None, rotary_pos_emb=rope)
            out.float().pow(2).sum().backward()
            res[layer_number] = dict(
                out=out.detach(),
                dx=x.grad.detach(),
                dq=att.q_norm.weight.grad.detach(),
                dk=att.k_norm.weight.grad.detach(),
            )
        torch.save(res, os.path.join(out_dir, f"rank{rank}.pt"))
    finally:
        ps.destroy_model_parallel()
        torch.distributed.destroy_process_group()


_TP2_SCRIPT = """
import importlib.util, sys

spec = importlib.util.spec_from_file_location("olmo3_tp_test", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod._tp2_worker(int(sys.argv[4]), int(sys.argv[2]), sys.argv[3])
"""


def test_full_width_qk_norm_tp2_real_allreduce(init_parallel_state, monkeypatch, tmp_path):
    """Real TP=2 (two gloo processes, no mocked all-reduce): the attention layer with full-width
    QK-norm matches the TP=1 layer on the same weights, forward and backward, on a sliding (RoPE,
    window) and a full (YaRN) layer."""
    import os
    import socket
    import subprocess
    import sys

    from megatron.core.transformer.spec_utils import build_module

    from primus.backends.megatron.core.models.olmo3.olmo3_rotary import (
        Olmo3DualRotaryEmbedding,
    )

    _cpu_numerics(monkeypatch)
    torch.manual_seed(0)
    cfg = _cpu_cfg(num_query_groups=2)
    spec = _cpu_layer_spec()
    s, bsz = 16, 2
    rope = Olmo3DualRotaryEmbedding(cfg, rotary_base=500000.0)(s)
    x = torch.randn(s, bsz, cfg.hidden_size)
    full, ref = {"x": x}, {}
    for layer_number in (1, 4):
        att = build_module(spec.submodules.self_attention, config=cfg, layer_number=layer_number)
        with torch.no_grad():
            for prm in (att.q_norm.weight, att.k_norm.weight, att.post_attention_layernorm.weight):
                prm.uniform_(0.5, 1.5)
        full[layer_number] = {
            k: v.detach().clone() for k, v in att.state_dict().items() if torch.is_tensor(v)
        }
        xr = x.clone().requires_grad_(True)
        out, _ = att(xr, None, rotary_pos_emb=rope)
        out.pow(2).sum().backward()
        ref[layer_number] = dict(
            out=out.detach(), dx=xr.grad, dq=att.q_norm.weight.grad, dk=att.k_norm.weight.grad
        )
    torch.save(full, tmp_path / "full.pt")
    (tmp_path / "run.py").write_text(_TP2_SCRIPT)
    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0))
        port = sk.getsockname()[1]
    # The workers re-import this module: give them the parent's import path (conftest.py adds
    # Megatron and the project root to sys.path in-process only, not through PYTHONPATH).
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    env.pop("MASTER_PORT", None)
    procs = [
        subprocess.Popen(
            [sys.executable, str(tmp_path / "run.py"), __file__, str(port), str(tmp_path), str(rank)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for rank in (0, 1)
    ]
    try:
        outs = [p.communicate(timeout=600) for p in procs]
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
    for p, (_, err) in zip(procs, outs):
        assert p.returncode == 0, f"worker exit {p.returncode}: {err[-4000:]}"
    r0, r1 = (torch.load(tmp_path / f"rank{r}.pt") for r in (0, 1))
    for n in (1, 4):
        torch.testing.assert_close(r0[n]["out"], ref[n]["out"], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(r1[n]["out"], ref[n]["out"], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(r0[n]["dx"], ref[n]["dx"], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(torch.cat([r0[n]["dq"], r1[n]["dq"]]), ref[n]["dq"], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(torch.cat([r0[n]["dk"], r1[n]["dk"]]), ref[n]["dk"], atol=1e-4, rtol=1e-4)
    # the two ranks hold different q/k norm shards, so the match needs the cross-rank statistic
    assert not torch.equal(r0[1]["dq"], r1[1]["dq"])


def test_converter_keeps_its_own_build_model(monkeypatch):
    """Tooling builds a family's model through ``<converter>.build_model`` (else plain GPT): the
    OLMo-3 converter must keep it, and it must ask for the olmo3 provider."""
    import primus.core.utils.import_utils as iu
    from primus.backends.megatron.checkpoint import native_loader_olmo3 as conv

    seen = []

    class _M:
        def to(self, dtype):
            return ("model", dtype)

    def provider(model_type="gpt"):
        seen.append(model_type)
        return lambda pre_process, post_process: _M()

    monkeypatch.setattr(iu, "get_model_provider", provider)
    assert conv.build_model(SimpleNamespace(params_dtype=torch.bfloat16)) == ("model", torch.bfloat16)
    assert seen == ["olmo3"]


def test_converter_runtime_margs_dtype_flags_are_exclusive():
    from primus.backends.megatron.checkpoint import (
        native_loader_olmo3 as common,  # the OLMo-3 converter's own setup
    )

    for dtype, want in (("bf16", torch.bfloat16), ("fp16", torch.float16), ("fp32", torch.float32)):
        m = SimpleNamespace(bf16=True, fp16=True)  # stale flags from a copied args object
        args = SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            expert_parallel_size=1,
            dtype=dtype,
            save_dir="/tmp/x",
        )
        common._apply_runtime_margs(m, args)
        assert m.params_dtype is want and m.bf16 == (dtype == "bf16") and m.fp16 == (dtype == "fp16")
    # a 1-rank, single-shard legacy torch save
    layout = (m.tensor_model_parallel_size, m.pipeline_model_parallel_size, m.expert_model_parallel_size)
    assert layout == (1, 1, 1) and m.expert_tensor_parallel_size == 1
    assert (m.world_size, m.rank, m.save, m.ckpt_format) == (1, 0, "/tmp/x", "torch")
    # the save iteration is set after validate_args (build_margs), not before it
    assert not hasattr(m, "iteration") and common.common.set_save_iteration(m).iteration == 1
    with pytest.raises(ValueError, match="dtype"):
        common._apply_runtime_margs(
            SimpleNamespace(),
            SimpleNamespace(
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                expert_parallel_size=1,
                dtype="bfloat16",
                save_dir="/tmp/x",
            ),
        )
    # the 1-rank world holds only TP=PP=EP=1; refused before m is changed
    for layout in ((2, 1, 1), (1, 2, 1), (1, 1, 2)):
        m = SimpleNamespace()
        with pytest.raises(ValueError, match="TP=PP=EP=1"):
            common._apply_runtime_margs(
                m,
                SimpleNamespace(
                    tensor_parallel_size=layout[0],
                    pipeline_parallel_size=layout[1],
                    expert_parallel_size=layout[2],
                    dtype="bf16",
                    save_dir="/tmp/x",
                ),
            )
        assert vars(m) == {}
    m = SimpleNamespace()
    common._apply_runtime_margs(
        m,
        SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            expert_parallel_size=1,
            dtype="bf16",
            save_dir="/x",
        ),
    )
    assert m.world_size == 1


def test_olmo3_patch_modules_import_without_megatron():
    """Patch modules are imported (patches/__init__) before Megatron is on sys.path."""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'megatron' or name.startswith('megatron.'):\n"
        "            raise ModuleNotFoundError(name)\n"
        "sys.meta_path.insert(0, Block())\n"
        "import importlib.util\n"
        "for m in ('primus.backends.megatron.patches.olmo3_flops_patches',\n"
        "          'primus.backends.megatron.patches.olmo3_args_patches'):\n"
        "    spec = importlib.util.spec_from_file_location(m, m.replace('.', '/') + '.py')\n"
        "    mod = importlib.util.module_from_spec(spec)\n"
        "    spec.loader.exec_module(mod)\n"
        "print('ok')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=_REPO_ROOT, capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-2000:]


def test_built_layer_report_reads_te_backend_scale():
    """TE's DotProductAttention keeps softmax_scale on its backends, not on itself. The report is a
    diagnostic: values it cannot read are reported, never raised (they must not stop a build)."""
    from primus.backends.megatron.core.models.olmo3.olmo3_builders import (
        _core_attention_scale,
        _core_attention_window,
    )

    class Backend(torch.nn.Module):
        def __init__(self, s):
            super().__init__()
            self.softmax_scale = s

    class TELike(torch.nn.Module):
        def __init__(self, a, b):
            super().__init__()
            self.flash_attention, self.unfused_attention = Backend(a), Backend(b)

    assert _core_attention_scale(TELike(0.25, 0.25)) == 0.25
    assert _core_attention_scale(torch.nn.Linear(1, 1)) is None
    assert _core_attention_scale(TELike(0.25, 0.5)) == "mixed"
    assert _core_attention_window(torch.nn.Linear(1, 1)) == "unknown"
    # "full" (no window) is distinct from "unknown" and from a missing value
    for no_window in (None, (-1, -1), (-1, 0)):
        assert _core_attention_window(SimpleNamespace(window_size=no_window)) == "full"
    assert _core_attention_window(SimpleNamespace(window_size=(4095, 0))) == [4095, 0]


def test_llama_loader_linear_qkv_is_the_grouped_q_k_v_layout():
    """Behaviour, not structure: the Llama converter's linear_qkv is, per query group, its query
    heads, then its key head, then its value head (checked row by row, not against fuse_qkv)."""
    from primus.backends.megatron.checkpoint import native_loader_llama

    nh, ng, hdim, hidden, inter, vocab = 4, 2, 2, 8, 6, 16
    g = torch.Generator().manual_seed(0)
    hf = {
        "model.embed_tokens.weight": torch.randn(vocab, hidden, generator=g),
        "model.norm.weight": torch.randn(hidden, generator=g),
        "lm_head.weight": torch.randn(vocab, hidden, generator=g),
    }
    h = "model.layers.0."
    shapes = {
        "input_layernorm.weight": (hidden,),
        "post_attention_layernorm.weight": (hidden,),
        "self_attn.q_proj.weight": (nh * hdim, hidden),
        "self_attn.k_proj.weight": (ng * hdim, hidden),
        "self_attn.v_proj.weight": (ng * hdim, hidden),
        "self_attn.o_proj.weight": (hidden, nh * hdim),
        "mlp.gate_proj.weight": (inter, hidden),
        "mlp.up_proj.weight": (inter, hidden),
        "mlp.down_proj.weight": (hidden, inter),
    }
    hf.update({h + k: torch.randn(*s, generator=g) for k, s in shapes.items()})
    model = SimpleNamespace(
        state_dict=lambda: {"embedding.word_embeddings.weight": torch.zeros(vocab, hidden)}
    )
    margs = SimpleNamespace(
        params_dtype=torch.float32,
        num_attention_heads=nh,
        num_query_groups=ng,
        kv_channels=hdim,
        hidden_size=hidden,
        num_layers=1,
        untie_embeddings_and_output_weights=True,
    )
    sd, _ = native_loader_llama.build_mcore_state_dict(model, _DictStore(hf), margs, None, lambda *a: None)
    qkv = sd["decoder.layers.0.self_attention.linear_qkv.weight"]
    q, k, v = (hf[h + f"self_attn.{n}_proj.weight"] for n in "qkv")
    per = nh // ng
    rows = []
    for grp in range(ng):
        rows += [q[(grp * per + j) * hdim : (grp * per + j + 1) * hdim] for j in range(per)]
        rows += [k[grp * hdim : (grp + 1) * hdim], v[grp * hdim : (grp + 1) * hdim]]
    assert torch.equal(qkv, torch.cat(rows))


# The runtime / save args every native converter had before set_converter_runtime_args shared them
# (the base Llama / Qwen3 / DeepSeek build_margs set exactly these values).
_BASE_CONVERTER_RUNTIME_ARGS = dict(
    use_cpu_initialization=True,
    perform_initialization=False,
    transformer_impl="transformer_engine",
    gradient_accumulation_fusion=False,
    async_tensor_model_parallel_allreduce=False,
    bias_gelu_fusion=False,
    bias_swiglu_fusion=False,
    masked_softmax_fusion=False,
    bias_dropout_fusion=False,
    apply_query_key_layer_scaling=False,
    save="/tmp/x",
    ckpt_format="torch",
    no_save_optim=True,
    no_save_rng=True,
    no_load_optim=True,
    no_load_rng=True,
    save_interval=1,
    async_save=False,
    ckpt_assume_constant_structure=False,
    iteration=1,
)


@pytest.mark.parametrize(
    "family, hf",
    [
        (
            "llama",
            dict(
                hidden_size=64,
                num_attention_heads=4,
                num_key_value_heads=2,
                num_hidden_layers=2,
                max_position_embeddings=128,
                intermediate_size=128,
                vocab_size=256,
            ),
        ),
        (
            "qwen3",
            dict(
                hidden_size=64,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=16,
                num_hidden_layers=2,
                max_position_embeddings=128,
                intermediate_size=128,
                rms_norm_eps=1e-6,
                vocab_size=256,
            ),
        ),
        (
            "deepseek",
            dict(
                hidden_size=64,
                intermediate_size=128,
                moe_intermediate_size=32,
                num_attention_heads=4,
                num_hidden_layers=2,
                first_k_dense_replace=1,
                n_shared_experts=1,
                n_routed_experts=4,
                num_experts_per_tok=2,
                kv_lora_rank=16,
                q_lora_rank=None,
                qk_nope_head_dim=16,
                qk_rope_head_dim=8,
                v_head_dim=16,
                vocab_size=256,
                max_position_embeddings=128,
            ),
        ),
    ],
)
def test_other_converters_keep_their_runtime_args(family, hf, monkeypatch):
    """Llama / Qwen3 / DeepSeek build_margs, through the shared set_converter_runtime_args and
    set_save_iteration, give the runtime / save args of their base build_margs (the values above),
    after validate_args. Runs without a GPU: DeepSeek turns on moe_grouped_gemm, for which
    Megatron's validate_args reads torch.cuda.get_device_capability(); on a CPU-only node the test
    answers that probe with a capability validate_args accepts (the args under test do not depend
    on it)."""
    import importlib

    if not torch.cuda.is_available():
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (9, 0))

    conv = importlib.import_module(f"primus.backends.megatron.checkpoint.native_loader_{family}")
    args = SimpleNamespace(
        seq_length=64,
        make_vocab_size_divisible_by=128,
        hf_dir="x",
        dtype="bf16",
        save_dir="/tmp/x",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        expert_parallel_size=1,
    )
    margs = conv.build_margs(conv.derive_megatron_fields(hf), args)
    got = {k: getattr(margs, k) for k in _BASE_CONVERTER_RUNTIME_ARGS}
    assert got == _BASE_CONVERTER_RUNTIME_ARGS
    assert margs.bf16 and not margs.fp16 and margs.params_dtype is torch.bfloat16


def test_qwen3_loader_linear_qkv_and_qk_norms():
    """Behaviour, not structure: the Qwen3 converter's linear_qkv is, per query group, its query
    heads, then its key head, then its value head, with an explicit head_dim != hidden / heads
    (checked row by row); the q / k norms map one to one. A transposed q_proj is refused."""
    from primus.backends.megatron.checkpoint import native_loader_qwen3

    nh, ng, hdim, hidden, inter, vocab = 4, 2, 3, 8, 6, 16
    g = torch.Generator().manual_seed(0)
    hf = {
        "model.embed_tokens.weight": torch.randn(vocab, hidden, generator=g),
        "model.norm.weight": torch.randn(hidden, generator=g),
        "lm_head.weight": torch.randn(vocab, hidden, generator=g),
    }
    h = "model.layers.0."
    shapes = {
        "input_layernorm.weight": (hidden,),
        "post_attention_layernorm.weight": (hidden,),
        "self_attn.q_proj.weight": (nh * hdim, hidden),
        "self_attn.k_proj.weight": (ng * hdim, hidden),
        "self_attn.v_proj.weight": (ng * hdim, hidden),
        "self_attn.q_norm.weight": (hdim,),
        "self_attn.k_norm.weight": (hdim,),
        "self_attn.o_proj.weight": (hidden, nh * hdim),
        "mlp.gate_proj.weight": (inter, hidden),
        "mlp.up_proj.weight": (inter, hidden),
        "mlp.down_proj.weight": (hidden, inter),
    }
    hf.update({h + k: torch.randn(*s, generator=g) for k, s in shapes.items()})
    model = SimpleNamespace(
        state_dict=lambda: {"embedding.word_embeddings.weight": torch.zeros(vocab, hidden)}
    )
    margs = SimpleNamespace(
        params_dtype=torch.float32,
        num_attention_heads=nh,
        num_query_groups=ng,
        kv_channels=hdim,
        hidden_size=hidden,
        num_layers=1,
        untie_embeddings_and_output_weights=True,
    )
    fields = dict(num_experts=None)
    sd, _ = native_loader_qwen3.build_mcore_state_dict(model, _DictStore(hf), margs, fields, lambda *a: None)
    p = "decoder.layers.0.self_attention."
    q, k, v = (hf[h + f"self_attn.{n}_proj.weight"] for n in "qkv")
    per = nh // ng
    rows = []
    for grp in range(ng):
        rows += [q[(grp * per + j) * hdim : (grp * per + j + 1) * hdim] for j in range(per)]
        rows += [k[grp * hdim : (grp + 1) * hdim], v[grp * hdim : (grp + 1) * hdim]]
    assert torch.equal(sd[p + "linear_qkv.weight"], torch.cat(rows))
    assert torch.equal(sd[p + "q_layernorm.weight"], hf[h + "self_attn.q_norm.weight"])
    assert torch.equal(sd[p + "k_layernorm.weight"], hf[h + "self_attn.k_norm.weight"])
    hf[h + "self_attn.q_proj.weight"] = q.t().contiguous()
    with pytest.raises(ValueError, match="q_proj weight has shape"):
        native_loader_qwen3.build_mcore_state_dict(model, _DictStore(hf), margs, fields, lambda *a: None)
