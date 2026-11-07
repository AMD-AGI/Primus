###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the OLMo-3 Megatron port (config, window, RoPE, QK-norm, converter)."""

import importlib.util
import json
import math
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


_NEEDS_GPU_FIXTURE = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the init_parallel_state fixture imports megatron.training (mamba_ssm / fla need a GPU)",
)


def _olmo3_builders():
    """``olmo3_builders`` (imports megatron.training, whose mamba_ssm / fla imports query GPU 0):
    skips the calling test on a host without a GPU; with a GPU an import error fails it."""
    if not torch.cuda.is_available():
        pytest.skip("megatron.training (mamba_ssm / fla) needs a GPU to import")
    from primus.backends.megatron.core.models.olmo3 import olmo3_builders

    return olmo3_builders


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
    # YaRN off (factor 1): unset or 1 gives 1.0; any other explicit value is refused (it would be
    # dropped, while HF applies it)
    assert yarn_attention_factor(1.0, 1.0) == 1.0
    with pytest.raises(ValueError, match="attention_factor"):
        yarn_attention_factor(1.0, 1.5)


def test_config_window_tuple_and_attention_factor():
    cfg = _cfg()
    # window_size is Megatron / TE (W - 1, 0); the HF W is olmo3_sliding_window
    assert cfg.window_size == (4095, 0) and cfg.olmo3_sliding_window == 4096
    # neutral defaults: YaRN off, attention factor 1
    assert cfg.olmo3_yarn_rotary_scaling_factor == 1.0
    assert cfg.olmo3_yarn_effective_attention_factor == 1.0
    # YaRN off with an explicit attention factor != 1 would drop it: refused
    with pytest.raises(ValueError, match="attention_factor"):
        _cfg(olmo3_yarn_attention_factor=1.2079441541679836)
    cfg = _cfg(olmo3_yarn_rotary_scaling_factor=8.0, olmo3_yarn_original_max_position_embeddings=8192)
    assert cfg.olmo3_yarn_effective_attention_factor == pytest.approx(0.1 * math.log(8.0) + 1.0)
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
    assert (
        cfg.olmo3_yarn_effective_attention_factor == pytest.approx(1.2079441541679836)
        and "differs" not in caplog.text
    )
    with caplog.at_level("WARNING"):
        cfg = _cfg(
            olmo3_yarn_rotary_scaling_factor=4.0, olmo3_yarn_attention_factor=1.2079441541679836, **yarn
        )
    assert cfg.olmo3_yarn_effective_attention_factor == pytest.approx(1.2079441541679836)
    assert "differs from 0.1*ln(factor)+1" in caplog.text


def test_layer_types_match_hf_pattern():
    from primus.backends.megatron.core.models.olmo3.olmo3_validation import (
        olmo3_te_window,
    )
    from primus.backends.megatron.core.transformer.olmo3.olmo3_attention import (
        olmo3_layer_is_sliding,
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


def test_dual_rope_matches_hf_formulas(monkeypatch):
    from primus.backends.megatron.core.models.olmo3.olmo3_rotary import (
        Olmo3DualRotaryEmbedding,
    )

    if not torch.cuda.is_available():
        _cpu_numerics(monkeypatch)  # the tables are built on torch.cuda.current_device()

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
    """auto maps only the measured minor versions (4.57 per_type, 5.10 yarn_all); any other version,
    and a pre-release, is refused instead of guessed."""
    transformers = pytest.importorskip("transformers")

    from primus.backends.megatron.core.models.olmo3.olmo3_transformer_config import (
        resolve_olmo3_rope_layout,
    )

    for ver, want in (
        ("4.57.0", "per_type"),
        ("4.57.3", "per_type"),
        ("5.10.0", "yarn_all"),
        ("5.10.2", "yarn_all"),
    ):
        monkeypatch.setattr(transformers, "__version__", ver)
        assert resolve_olmo3_rope_layout("auto") == want, ver
    for ver, why in (
        ("4.55.0", "not measured"),
        ("4.58.0", "not measured"),
        ("5.0.0", "not measured"),
        ("5.13.1", "not measured"),
        ("5.10.0.dev0", "not measured"),
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

    kw.setdefault("olmo3_sliding_window", 4096)  # the preset sets it
    params = SimpleNamespace(model_type="olmo3", **kw)
    ctx = PatchContext(
        backend="megatron", phase="before_train", extra={"module_config": SimpleNamespace(params=params)}
    )
    patch_olmo3_args(ctx)
    return params


def test_preset_window_has_one_source():
    """The preset states W as olmo3_sliding_window 4096 and window_size 4096 (an int: W itself, as
    config.json states sliding_window); the args patch sets window_size to Megatron/TE (W - 1, 0)."""
    import yaml

    presets = _REPO_ROOT / "primus/configs/models/megatron"
    preset = yaml.safe_load(open(presets / "olmo_base.yaml"))  # the family base
    size = yaml.safe_load(open(presets / "olmo_3_1025_7B.yaml"))
    assert size["extends"] == ["olmo_base.yaml"] and not (set(size) - {"extends"}) & set(preset)
    preset.update({k: v for k, v in size.items() if k != "extends"})
    assert preset["olmo3_sliding_window"] == 4096 and preset["window_size"] == 4096
    assert "olmo3_window_size_form" not in preset
    # derived from the factor (0.1 * ln(8) + 1 = the HF attention_factor), so a factor override moves it
    assert "olmo3_yarn_attention_factor" not in preset and preset["olmo3_yarn_rotary_scaling_factor"] == 8.0
    assert preset["norm_epsilon"] == preset["layernorm_epsilon"] == 1e-6
    base = dict(window_attn_skip_freq=4)
    # next to olmo3_sliding_window W, window_size is W (int), the TE pair (W - 1, 0), or unset
    for ws in (4096, "4096", [4095, 0], "(4095, 0)", "[4095,0]", None):
        a = _run_args_patch(**{**base, "window_size": ws})
        assert (a.window_size, a.olmo3_sliding_window) == ((4095, 0), 4096)
    for bad in ([2048, 0], [4094, 0], [4097, 0], [4096, 0], "(4096, 0)", 4095, 4097):
        with pytest.raises(ValueError, match="disagrees"):
            _run_args_patch(**{**base, "window_size": bad})
    # without olmo3_sliding_window, window_size alone is refused
    for ws in ([4095, 0], "[4095, 0]", "4095,0", 4096):
        with pytest.raises(ValueError, match="olmo3_sliding_window"):
            _run_args_patch(**{**base, "olmo3_sliding_window": None, "window_size": ws})
    # no window at all is refused
    with pytest.raises(ValueError, match="olmo3_sliding_window"):
        _run_args_patch(**{**base, "window_size": None, "olmo3_sliding_window": None})
    with pytest.raises(ValueError, match="disagrees"):
        _run_args_patch(**{**base, "olmo3_sliding_window": 4098, "window_size": 4096})


def test_args_patch_window_size_forms():
    """window_size is Megatron/TE (W - 1, 0) as for every model; next to olmo3_sliding_window W it
    may also be the int W. The result is (W - 1, 0) with no marker on the args, and a re-run on the
    output (or args rebuilt from a checkpoint) keeps it."""
    a = _run_args_patch(window_size=4096, window_attn_skip_freq=[1, 1, 1, 0], num_layers=4)
    assert a.window_size == (4095, 0) and a.window_attn_skip_freq == [1, 1, 1, 0]
    assert not hasattr(a, "_olmo3_window_size_is_te")
    a2 = _run_args_patch(window_size=a.window_size, olmo3_sliding_window=4096, window_attn_skip_freq=4)
    assert a2.window_size == (4095, 0) and a2.olmo3_sliding_window == 4096
    with pytest.raises(ValueError, match="olmo3_sliding_window"):
        _run_args_patch(window_size=a.window_size, olmo3_sliding_window=None, window_attn_skip_freq=4)
    # Megatron's --window-attn-skip-freq string forms, spaces allowed
    assert _run_args_patch(window_size=4096, window_attn_skip_freq="4").window_attn_skip_freq == 4
    for text in ("([1]*3+[0])*2", "([1]*3 + [0]) * 2", "[1, 1, 1, 0, 1, 1, 1, 0]"):
        a = _run_args_patch(window_size=4096, window_attn_skip_freq=text, num_layers=8)
        assert a.window_attn_skip_freq == [1, 1, 1, 0] * 2


@pytest.mark.parametrize(
    "kw",
    [
        # no window at all
        dict(window_size=None, olmo3_sliding_window=None, window_attn_skip_freq=4),
        dict(window_size=[4095, 0], olmo3_sliding_window=4097, window_attn_skip_freq=4),
        dict(window_size=[4097, 0], olmo3_sliding_window=4096, window_attn_skip_freq=4),
        dict(window_size=[4096], window_attn_skip_freq=4),
        dict(window_size=[4096, 5], window_attn_skip_freq=4),
        dict(window_size=[0, 0], window_attn_skip_freq=4),
        dict(window_size=[-1, 0], window_attn_skip_freq=4),
        dict(window_size=[1, 0], window_attn_skip_freq=4),
        dict(window_size=4096),
        dict(window_size=4096, window_attn_skip_freq=None),
        dict(window_size=4096, window_attn_skip_freq=1),
        dict(window_size=4096, window_attn_skip_freq=[0, 0, 0, 0], num_layers=4),
        dict(window_size=4096, window_attn_skip_freq=[1, 1, 0], num_layers=4),
        dict(window_size=4096, window_attn_skip_freq=4, context_parallel_size=2, cp_comm_type="p2p"),
        # window_size disagrees with olmo3_sliding_window
        dict(window_size=[4000, 0], olmo3_sliding_window=4096, window_attn_skip_freq=4),
        # bad skip-freq strings
        dict(window_size=4096, window_attn_skip_freq="([1]*3+[0])*8", num_layers=4),
        dict(window_size=4096, window_attn_skip_freq="__import__('os')"),
        dict(window_size=4096, window_attn_skip_freq="[1]*10**10", num_layers=4),
        dict(window_size=4096, window_attn_skip_freq="[1]*10000000000", num_layers=4),
        dict(window_size=4096, window_attn_skip_freq="[1,2,0,1]", num_layers=4),
    ],
)
def test_args_patch_refuses_bad_window(kw):
    with pytest.raises(ValueError, match="OLMo-3|olmo3|window"):
        _run_args_patch(**kw)


@pytest.mark.parametrize(
    "kw",
    [
        # each would agree with W after int() truncation: only the integer check refuses it
        dict(window_size=[4095.7, 0], window_attn_skip_freq=4),
        dict(window_size=[4095.0, 0], window_attn_skip_freq=4),
        dict(window_size="(4095.5, 0)", window_attn_skip_freq=4),
        dict(window_size=[True, 0], olmo3_sliding_window=2, window_attn_skip_freq=4),
        dict(window_size="4096.0", window_attn_skip_freq=4),
        dict(window_size=4096.0, window_attn_skip_freq=4),
    ],
)
def test_args_patch_does_not_truncate_non_integer_windows(kw):
    """Non-integer window entries are refused as such, not truncated to a value that agrees."""
    with pytest.raises(ValueError, match="must be an integer"):
        _run_args_patch(**kw)


def test_window_size_messages_name_the_pair_form():
    """A one-element window_size points at olmo3_sliding_window; a bad pair names (W - 1, 0)."""
    with pytest.raises(ValueError, match=r"olmo3_sliding_window") as one:
        _run_args_patch(window_size=(4096,), window_attn_skip_freq=4)
    assert "(W - 1, 0)" in str(one.value) and "1 entries" in str(one.value)
    with pytest.raises(ValueError, match=r"\(W - 1, 0\)"):
        _run_args_patch(window_size=[4095, 5], window_attn_skip_freq=4)


def test_args_patch_cp_comm_type():
    _run_args_patch(window_size=4096, window_attn_skip_freq=4, context_parallel_size=2, cp_comm_type="a2a")
    _run_args_patch(
        window_size=4096, window_attn_skip_freq=4, context_parallel_size=2, cp_comm_type=["all_gather"]
    )
    _run_args_patch(window_size=4096, window_attn_skip_freq=4, context_parallel_size=1, cp_comm_type="p2p")


def test_te_window_conversion():
    from primus.backends.megatron.core.models.olmo3.olmo3_validation import (
        olmo3_te_window,
    )

    assert olmo3_te_window(4096) == (4095, 0)
    assert olmo3_te_window(2) == (1, 0)
    # one rule with window_size: an int or a bare-int string; floats are refused (Megatron's type=int)
    assert olmo3_te_window("4096") == (4095, 0)
    for bad in (None, (4096, 0), 0, -1, 1, 4096.0, "4096.0", 4096.5, True, "abc"):
        with pytest.raises(ValueError, match="window"):
            olmo3_te_window(bad)


def test_args_patch_turbo_attention_only():
    # Turbo GEMMs etc. are fine; only the Turbo attention kernel lacks sliding windows.
    w = dict(window_size=4096, window_attn_skip_freq=4)
    _run_args_patch(**w, enable_primus_turbo=True, use_turbo_attention=False)
    with pytest.raises(ValueError, match="use_turbo_attention"):
        _run_args_patch(**w, enable_primus_turbo=True, use_turbo_attention=True)
    with pytest.raises(ValueError, match="qk_layernorm"):
        _run_args_patch(**w, qk_layernorm=True)


def test_switch_settings_refuse_typos():
    """A switch setting is read as Primus reads a CLI override (arg_utils.parse_cli_overrides):
    a bool, 0 / 1, or true / false (any case). A typo is refused with the setting's message instead
    of being read as on (bool("ture")) or off."""
    from primus.backends.megatron.core.models.olmo3 import olmo3_validation as v

    for on in (True, 1, "true", " True ", "TRUE", "1"):
        assert v._truthy(on) is True
    for off in (False, 0, None, "", "None", "false", "False", "0"):
        assert v._truthy(off) is False
    # the shared reader does not read these as switches, so neither does OLMo-3
    for bad in ("ture", "flase", "enabled", 2, 0.5, "yes", "off", "null", "~"):
        with pytest.raises(ValueError):
            v._truthy(bad)
    w = dict(window_size=4096, window_attn_skip_freq=4)
    with pytest.raises(ValueError, match="qk_layernorm"):
        _run_args_patch(**w, qk_layernorm="flase")
    with pytest.raises(ValueError, match="swiglu"):
        _run_args_patch(**w, swiglu="ture")
    _run_args_patch(**w, qk_layernorm="false", swiglu="true")


def test_args_patch_logs_the_window(monkeypatch):
    from primus.backends.megatron.patches import olmo3_args_patches as ap

    logs = []
    monkeypatch.setattr(ap, "log_rank_0", logs.append)
    _run_args_patch(window_size=4096, window_attn_skip_freq=4)
    assert "sliding_window=4096: window_size 4096 -> (4095, 0)" in logs[-1]


def test_args_patch_accepts_neutral_defaults():
    _run_args_patch(
        window_size=4096,
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


def test_config_window_reads_like_the_args_patch():
    """The config reads window_size with the args patch's rule and always holds (W - 1, 0)."""
    assert _cfg(window_size=(4095, 0)).window_size == (4095, 0)
    assert _cfg(window_size=[4095, 0]).window_size == (4095, 0)
    assert _cfg(window_size=4096).window_size == (4095, 0)  # W (int) next to W
    with pytest.raises(ValueError, match="disagrees"):
        _cfg(window_size=[4096, 0])  # a pair is the TE form: 4097 keys
    assert _cfg().window_size == (4095, 0)  # derived from olmo3_sliding_window
    with pytest.raises(ValueError, match="needs olmo3_sliding_window"):  # built after the args patch
        _cfg(olmo3_sliding_window=None, window_size=[4095, 0])
    for bad in ([4097, 0], [4094, 0]):
        with pytest.raises(ValueError, match="disagrees"):
            _cfg(window_size=bad)


def test_window_rule():
    """W is olmo3_sliding_window; window_size next to it is W (int), the TE pair (W - 1, 0) or
    unset. Without it, window_size alone is refused."""
    from primus.backends.megatron.core.models.olmo3.olmo3_validation import (
        olmo3_window_from_settings as f,
    )

    assert f(4096, 4096) == ((4095, 0), 4096)
    assert f("4096", "4096") == ((4095, 0), 4096)
    assert f((4095, 0), 4096) == ((4095, 0), 4096)
    assert f("[4095, 0]", "4096") == ((4095, 0), 4096)
    assert f(None, 4096) == ((4095, 0), 4096)
    for ws, w in (((4097, 0), 4096), ((4096, 0), 4096), ((4094, 0), 4096), (4096, "4098"), (4096, 4097)):
        with pytest.raises(ValueError, match="disagrees"):
            f(ws, w)
    for ws in ((2047, 0), 4096, None):
        with pytest.raises(ValueError, match="olmo3_sliding_window"):
            f(ws, None)
    with pytest.raises(ValueError, match="> 1"):
        f((1, 0), 1)
    # the patch output carries W, so a second run keeps the window
    a = _run_args_patch(window_size=4096, window_attn_skip_freq=4)
    assert (a.window_size, a.olmo3_sliding_window) == ((4095, 0), 4096)
    from primus.backends.megatron.patches.olmo3_args_patches import patch_olmo3_args
    from primus.core.patches import PatchContext

    patch_olmo3_args(
        PatchContext(
            backend="megatron", phase="before_train", extra={"module_config": SimpleNamespace(params=a)}
        )
    )
    assert (a.window_size, a.olmo3_sliding_window) == ((4095, 0), 4096)


def test_olmo3_preset_with_another_model_type_is_refused():
    """The OLMo-3 preset (olmo3_sliding_window set, window_size an int W) with model_type
    overridden away from olmo3 is refused at before_train, so no other model reads the int W
    window_size; without the OLMo-3 keys the patch does not apply."""
    from primus.backends.megatron.patches import olmo3_args_patches as ap
    from primus.core.patches import PatchContext

    def ctx(**kw):
        return PatchContext(
            backend="megatron",
            phase="before_train",
            extra={"module_config": SimpleNamespace(params=SimpleNamespace(**kw))},
        )

    bad = ctx(model_type="gpt", olmo3_sliding_window=4096, window_size=4096)
    assert ap._olmo3_keys_without_olmo3_model(bad)
    with pytest.raises(ValueError, match="needs model_type: olmo3"):
        ap.patch_olmo3_preset_needs_olmo3_model(bad)
    assert not ap._olmo3_keys_without_olmo3_model(ctx(model_type="olmo3", olmo3_sliding_window=4096))
    assert not ap._olmo3_keys_without_olmo3_model(ctx(model_type="gpt", window_size=None))


def test_olmo3_patch_condition_selects_olmo3_only():
    """is_olmo3_model_type (the condition of every OLMo-3 patch) is true for model_type olmo3 only."""
    from primus.backends.megatron.patches.olmo3_args_patches import is_olmo3_model_type
    from primus.core.patches import PatchContext

    def ctx(model_type):
        params = SimpleNamespace(model_type=model_type)
        return PatchContext(
            backend="megatron", phase="before_train", extra={"module_config": SimpleNamespace(params=params)}
        )

    assert is_olmo3_model_type(ctx("olmo3"))
    assert not is_olmo3_model_type(ctx("gpt"))
    assert not is_olmo3_model_type(ctx(None))


def test_checkpoint_layout_refuses_single_shard_legacy_at_tp2(tmp_path):
    """The native converter's single-shard legacy checkpoint loaded with TP or PP > 1 is refused
    before the model is built; TP = PP = 1, or a sharded checkpoint, passes."""
    from primus.backends.megatron.core.models.olmo3.olmo3_validation import (
        check_olmo3_checkpoint_layout as check,
    )

    (tmp_path / "iter_0000001" / "mp_rank_00").mkdir(parents=True)
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("1")
    ck = str(tmp_path)
    check(
        SimpleNamespace(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1, pretrained_checkpoint=ck
        )
    )
    for tp_, pp_ in ((2, 1), (1, 2)):
        with pytest.raises(ValueError, match="single-shard legacy torch checkpoint"):
            check(
                SimpleNamespace(
                    tensor_model_parallel_size=tp_, pipeline_model_parallel_size=pp_, pretrained_checkpoint=ck
                )
            )
    with pytest.raises(ValueError, match="load="):
        check(SimpleNamespace(tensor_model_parallel_size=2, pipeline_model_parallel_size=1, load=ck))
    (tmp_path / "iter_0000001" / "mp_rank_01").mkdir()
    check(
        SimpleNamespace(
            tensor_model_parallel_size=2, pipeline_model_parallel_size=1, pretrained_checkpoint=ck
        )
    )
    check(
        SimpleNamespace(
            tensor_model_parallel_size=2, pipeline_model_parallel_size=1, pretrained_checkpoint=None
        )
    )


# ---------------------------------------------------------------------------
# shared validation (olmo3_validation): window contract, skip_freq, one refusal table
# ---------------------------------------------------------------------------


def test_checkpoint_layout_reads_sizes_strictly(tmp_path):
    """TP / PP 0 is refused (not read as 1); unset is 1. A lone mp_rank_00_000 (the first shard of a
    PP > 1 save: Megatron adds _{pp:03d} only for PP > 1) is not read as single-shard."""
    from primus.backends.megatron.core.models.olmo3.olmo3_validation import (
        check_olmo3_checkpoint_layout as check,
    )

    (tmp_path / "iter_0000001" / "mp_rank_00").mkdir(parents=True)
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("1")
    ck = str(tmp_path)
    with pytest.raises(ValueError, match=">= 1"):
        check(SimpleNamespace(tensor_model_parallel_size=0, pretrained_checkpoint=ck))
    check(SimpleNamespace(tensor_model_parallel_size=None, pretrained_checkpoint=ck))
    with pytest.raises(ValueError, match="single-shard legacy torch checkpoint"):
        check(SimpleNamespace(tensor_model_parallel_size=2, pretrained_checkpoint=ck))
    pp_ck = tmp_path / "pp"
    (pp_ck / "iter_0000001" / "mp_rank_00_000").mkdir(parents=True)
    (pp_ck / "latest_checkpointed_iteration.txt").write_text("1")
    check(SimpleNamespace(tensor_model_parallel_size=2, pretrained_checkpoint=str(pp_ck)))


def test_checkpoint_layout_checks_only_what_megatron_loads(tmp_path):
    """Megatron reads pretrained_checkpoint only when load holds no checkpoint (no tracker file), so
    a single-shard pretrained_checkpoint next to a resharded load is not refused; ckpt_step picks
    the iteration that is inspected."""
    from primus.backends.megatron.core.models.olmo3.olmo3_validation import (
        check_olmo3_checkpoint_layout as check,
    )

    pre = tmp_path / "converted"
    (pre / "iter_0000001" / "mp_rank_00").mkdir(parents=True)
    (pre / "latest_checkpointed_iteration.txt").write_text("1")
    load = tmp_path / "resumed"
    (load / "iter_0000005").mkdir(parents=True)  # torch_dist: no mp_rank dirs
    (load / "latest_checkpointed_iteration.txt").write_text("5")
    args = dict(tensor_model_parallel_size=2, pipeline_model_parallel_size=1, pretrained_checkpoint=str(pre))
    check(SimpleNamespace(**args, load=str(load)))  # the reviewer's resume workflow
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="pretrained_checkpoint="):
        check(SimpleNamespace(**args, load=str(empty)))  # load has no checkpoint: pretrained is read
    # ckpt_step selects a legacy single-shard iteration inside the resumed directory
    (load / "iter_0000002" / "mp_rank_00").mkdir(parents=True)
    with pytest.raises(ValueError, match="load="):
        check(SimpleNamespace(**args, load=str(load), ckpt_step=2))
    check(SimpleNamespace(**args, load=str(load), ckpt_step=5))


def test_args_patch_hf_form_next_to_marker():
    """W (int, the preset's form) next to olmo3_sliding_window=W is the same window: it becomes
    (W - 1, 0). A window_size that is neither W nor the pair (W - 1, 0) is refused."""
    a = _run_args_patch(window_size=4096, olmo3_sliding_window=4096, window_attn_skip_freq=4)
    assert (a.window_size, a.olmo3_sliding_window) == ((4095, 0), 4096)
    for bad in ([4097, 0], [4094, 0], [4096, 0], 4097):
        with pytest.raises(ValueError, match="disagrees"):
            _run_args_patch(window_size=bad, olmo3_sliding_window=4096, window_attn_skip_freq=4)
    # the marker alone (window_size unset) is enough
    a = _run_args_patch(window_size=None, olmo3_sliding_window=4096, window_attn_skip_freq=4)
    assert a.window_size == (4095, 0)
    # CLI string marker
    a = _run_args_patch(window_size="(4095, 0)", olmo3_sliding_window="4096", window_attn_skip_freq="4")
    assert a.window_size == (4095, 0) and a.olmo3_sliding_window == 4096 and a.window_attn_skip_freq == 4


def test_args_patch_and_config_agree_on_window():
    """The args patch output builds a config with the same window (no 4097)."""
    a = _run_args_patch(window_size=4096, window_attn_skip_freq=4, num_layers=4)
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
        ("[1]*10**10", 4),  # ** is refused before anything is built
        ("[1]*100000000", 4),  # longer than num_layers: refused before it is built
        ("([1]*1000)*1000*1000", 4),
        ("[1]*4096*4096", 4),
        ("[1,1,1,0]*2+[0]", 8),  # 9 entries
        ("int('4')", 4),
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
        _run_args_patch(window_size=4096, window_attn_skip_freq=skip, num_layers=layers)
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
        # spaces, as YAML / CLI users write them (Megatron's own parser refuses them)
        ("([1]*3 + [0])*8", 32, [1, 1, 1, 0] * 8),
        (" ( [1] * 3 + [0] ) * 2 ", 8, [1, 1, 1, 0] * 2),
    ],
)
def test_skip_freq_accepts(skip, layers, want):
    from primus.backends.megatron.core.models.olmo3.olmo3_validation import (
        parse_olmo3_skip_freq,
    )

    assert parse_olmo3_skip_freq(skip, layers) == want
    assert _cfg(num_layers=layers, window_attn_skip_freq=skip).window_attn_skip_freq == want


def test_skip_freq_string_is_bounded_without_num_layers():
    """With num_layers unknown the list bound is _MAX_PATTERN; no input builds a huge list."""
    from primus.backends.megatron.core.models.olmo3 import olmo3_validation as v

    assert v._skip_freq_value("([1]*3 + [0])*8") == [1, 1, 1, 0] * 8
    for bad in ("[1]*10**10", "[1]*4097", "[1]*64*64*64", "9**9**9"):
        with pytest.raises(ValueError, match="window_attn_skip_freq"):
            v._skip_freq_value(bad)
    # an int is bounded by the caller, with one bound for both paths
    with pytest.raises(ValueError, match="window_attn_skip_freq"):
        v.parse_olmo3_skip_freq("4097", num_layers=None)


def test_skip_freq_proxy_depths():
    """The int skip_freq of the preset works at the reduced proxy depths the L4 gates use."""
    for layers in (4, 5, 8, 32):
        assert (
            _run_args_patch(
                window_size=4096, window_attn_skip_freq="4", num_layers=str(layers)
            ).window_attn_skip_freq
            == 4
        )
    with pytest.raises(ValueError, match="num_layers"):
        _run_args_patch(window_size=4096, window_attn_skip_freq=4, num_layers=2)


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
]


@pytest.mark.parametrize("name, value", _SHARED_REFUSALS)
def test_shared_refusal_table(name, value):
    """The args patch, check_olmo3_unsupported and the config refuse every table entry (no GPU
    needed; the builder path is test_shared_refusal_table_builder)."""
    from megatron.core.transformer.transformer_config import TransformerConfig

    from primus.backends.megatron.core.models.olmo3.olmo3_validation import (
        OLMO3_UNSUPPORTED,
        check_olmo3_unsupported,
    )

    assert name in {n for n, _, _ in OLMO3_UNSUPPORTED}
    # args patch (YAML values and CLI strings)
    with pytest.raises(ValueError, match=name):
        _run_args_patch(window_size=4096, window_attn_skip_freq=4, **{name: value})
    args = SimpleNamespace(use_legacy_models=False, spec=None, mtp_num_layers=None, **{name: value})
    with pytest.raises(ValueError, match=name):
        check_olmo3_unsupported(args)
    # config, when it is a config field
    if name in TransformerConfig.__dataclass_fields__ or name.startswith("olmo3_"):
        with pytest.raises(ValueError, match="OLMo-3|olmo3"):
            _cfg(**{name: value})


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="megatron.training imports mamba_ssm, which needs a GPU"
)
@pytest.mark.parametrize("name, value", _SHARED_REFUSALS)
def test_shared_refusal_table_builder(name, value):
    """The builder refuses the same table before any config is built."""
    olmo3_builders = _olmo3_builders()

    args = SimpleNamespace(use_legacy_models=False, spec=None, mtp_num_layers=None, **{name: value})
    with pytest.raises(ValueError, match=name):
        olmo3_builders.olmo3_builder(args, True, True, config=_cfg())


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


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="megatron.training imports mamba_ssm / fla, which need a GPU"
)
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
    olmo3_builders = _olmo3_builders()

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

    olmo3_builders = _olmo3_builders()

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


def test_model_provider_delegates_to_upstream(monkeypatch):
    """model_provider runs upstream's model_provider (record_memory_history, ...) with the OLMo-3 builder."""
    olmo3_builders = _olmo3_builders()

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

    # no upstream model_provider: record_memory_history is refused, else the builder runs directly
    monkeypatch.setattr(olmo3_builders, "_upstream_model_provider", lambda: None)
    with pytest.raises(ValueError, match="record_memory_history"):
        olmo3_builders.model_provider(None, True, False, 1)
    args.record_memory_history = False
    direct = []

    def builder(a, pre, post, vp, config=None, pg_collection=None):
        direct.append((a, pre, post, vp, config, pg_collection))
        return "direct"

    assert (
        olmo3_builders.model_provider(builder, True, False, 1, config="cfg", pg_collection="pg") == "direct"
    )
    assert direct == [(args, True, False, 1, "cfg", "pg")]


def test_model_provider_checks_before_upstream(monkeypatch):
    """model_provider checks the args before upstream (whose ModelOpt branch skips the builder);
    olmo3_builder checks again for direct callers. The config is never built first."""
    olmo3_builders = _olmo3_builders()

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
    subs = SelfAttentionSubmodules(linear_qkv=None, core_attention=FakeCore, linear_proj=None)
    for n in range(1, 9):
        with pytest.raises(StopIteration):
            oa.Olmo3SelfAttention(cfg, subs, n, post_norm=object)
    with pytest.raises(ValueError, match="post_norm"):
        oa.Olmo3SelfAttention(cfg, subs, 1)
    base = 1.0 / math.sqrt(cfg.kv_channels)
    for n, (c, win, scale) in seen.items():
        sliding = n % 4 != 0
        assert c is cfg, n  # no private config copy
        assert win == ((4095, 0) if sliding else None), n
        want = base if sliding else base * cfg.olmo3_yarn_effective_attention_factor**2
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
    assert isinstance(cls, type) and issubclass(cls, FakeCore) and cls.__name__ == "FakeCoreOlmo3YarnScaled"
    assert cls.olmo3_softmax_scale == 0.5 and cls.__qualname__ != FakeCore.__qualname__
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


@pytest.mark.parametrize(
    "bad, match",
    [
        ({"num_key_value_heads": 0}, "num_key_value_heads"),
        ({"num_key_value_heads": None}, "num_key_value_heads"),
        ({"num_key_value_heads": -2}, "num_key_value_heads"),
        ({"head_dim": 0}, "head_dim"),
        ({"head_dim": 2.5}, "head_dim"),
        ({"hidden_size": 258}, "divisible by num_attention_heads"),
        ({"num_attention_heads": 0}, "num_attention_heads"),
    ],
)
def test_converter_refuses_invalid_head_settings(bad, match):
    """Explicit invalid head settings are refused, not replaced by a default or floored."""
    from primus.backends.megatron.checkpoint.native_loader_olmo3 import (
        derive_megatron_fields,
    )

    cfg = {**_HF_CFG, **bad}
    if "hidden_size" in bad:
        cfg.pop("head_dim", None)
    with pytest.raises(ValueError, match=match):
        derive_megatron_fields(cfg, rope_layout="per_type")


def test_converter_rope_type_and_legacy_type_key():
    """rope_type and the legacy type key must agree; a null rope_type falls back to type."""
    from primus.backends.megatron.checkpoint.native_loader_olmo3 import _rope_type

    assert _rope_type({"rope_type": None, "type": "yarn"}) == "yarn"
    assert _rope_type({"type": "yarn"}) == "yarn" and _rope_type({}) == "default"
    assert _rope_type({"rope_type": "yarn", "type": "yarn"}) == "yarn"
    with pytest.raises(ValueError, match="must agree"):
        _rope_type({"rope_type": "yarn", "type": "default"})


def test_converter_head_defaults():
    """No num_key_value_heads: MHA; no head_dim: hidden_size / num_attention_heads."""
    from primus.backends.megatron.checkpoint.native_loader_olmo3 import (
        derive_megatron_fields,
    )

    cfg = {k: v for k, v in _HF_CFG.items() if k not in ("num_key_value_heads", "head_dim")}
    f = derive_megatron_fields(cfg, rope_layout="per_type")
    assert f["num_query_groups"] == cfg["num_attention_heads"]
    assert f["kv_channels"] == cfg["hidden_size"] // cfg["num_attention_heads"]


def test_converter_checks_vocab_rows_against_config_and_tokenizer():
    """The HF embedding / lm_head rows must equal config.json vocab_size and fit the padded
    vocabulary from the tokenizer; a mismatch is refused instead of truncated or padded."""
    from primus.backends.megatron.checkpoint.native_loader_olmo3 import (
        check_vocab_rows,
        derive_megatron_fields,
    )

    assert (
        derive_megatron_fields({**_HF_CFG, "vocab_size": 100278}, rope_layout="per_type")["vocab_size"]
        == 100278
    )
    check_vocab_rows("lm_head.weight", 100278, 100278, 100352)  # the released 7B
    check_vocab_rows("lm_head.weight", 100278, None, 100352)  # no vocab_size stated: fit only
    with pytest.raises(ValueError, match="config.json vocab_size"):
        check_vocab_rows("lm_head.weight", 100277, 100278, 100352)
    with pytest.raises(ValueError, match="padded vocabulary"):
        check_vocab_rows("model.embed_tokens.weight", 100278, 100278, 100224)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the init_parallel_state fixture imports megatron.training (needs a GPU)",
)
def test_qk_norm_modules_run_their_hooks(init_parallel_state, monkeypatch):
    """Running the OLMo-3 attention fires forward pre-hooks registered on q_norm and k_norm
    (Megatron DDP's overlap_param_gather wait is such a hook)."""
    from megatron.core.transformer.spec_utils import build_module

    from primus.backends.megatron.core.models.olmo3.olmo3_rotary import (
        Olmo3DualRotaryEmbedding,
    )

    _cpu_numerics(monkeypatch)
    cfg = _cpu_cfg()
    att = build_module(_cpu_layer_spec().submodules.self_attention, config=cfg, layer_number=1)
    fired = []
    att.q_norm.register_forward_pre_hook(lambda m, a: fired.append("q"))
    att.k_norm.register_forward_pre_hook(lambda m, a: fired.append("k"))
    s_, b = 8, 1
    rope = Olmo3DualRotaryEmbedding(cfg, rotary_base=500000.0)(s_)
    att(torch.randn(s_, b, cfg.hidden_size), None, rotary_pos_emb=rope)
    assert sorted(fired) == ["k", "q"]


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


def test_converter_per_type_without_sliding_entry():
    """A per-layer-type rope_parameters with only full_attention means default RoPE on the sliding
    layers (transformers 5.x): the layout is per_type, and yarn_all is refused."""
    from primus.backends.megatron.checkpoint.native_loader_olmo3 import (
        derive_megatron_fields,
    )

    yarn = dict(_HF_CFG["rope_scaling"], rope_theta=500000)
    base = {k: v for k, v in _HF_CFG.items() if k not in ("rope_scaling", "rope_theta")}
    cfg = {**base, "rope_parameters": {"full_attention": yarn}}
    for asked in (None, "auto", "per_type"):
        f = derive_megatron_fields(cfg, rope_layout=asked)
        assert f["olmo3_rope_layout"] == "per_type"
        assert f["olmo3_yarn_rotary_scaling_factor"] == 8.0 and f["rotary_base"] == 500000.0
    with pytest.raises(ValueError, match="declares"):
        derive_megatron_fields(cfg, rope_layout="yarn_all")


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
    """Refused by convert() (check_single_shard_layout) before Megatron is imported or initialised."""
    import json

    from primus.backends.megatron.checkpoint import native_loader_olmo3 as conv

    (tmp_path / "config.json").write_text(json.dumps(_HF_CFG))
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
    # independent row-by-row oracle: for each query group g, the rows of its q heads, then its k
    # head, then its v head (not the reshape / cat the converter uses)
    qw, kw, vw = (hf[h + f"self_attn.{n}_proj.weight"] for n in "qkv")
    rows = []
    for g in range(ng):
        for j in range(nh // ng):
            head = g * (nh // ng) + j
            rows += [qw[head * hd + r] for r in range(hd)]
        rows += [kw[g * hd + r] for r in range(hd)]
        rows += [vw[g * hd + r] for r in range(hd)]
    want = torch.stack(rows)
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
                ckpt_format="torch_dist",
            ),
        )
    ]


def test_per_type_yarn_without_scaling_is_per_type():
    """Per-layer-type YaRN entries with factor 1.0 on both layer types are plain RoPE: the layout
    is per_type, as the flat form gives, and an explicit yarn_all is refused."""
    from primus.backends.megatron.checkpoint.native_loader_olmo3 import (
        derive_megatron_fields,
    )

    base = {k: v for k, v in _HF_CFG.items() if k not in ("rope_scaling", "rope_theta")}
    y1 = {"rope_type": "yarn", "factor": 1.0, "rope_theta": 500000}
    cfg = {**base, "rope_parameters": {"full_attention": y1, "sliding_attention": dict(y1)}}
    assert derive_megatron_fields(cfg, rope_layout="per_type")["olmo3_rope_layout"] == "per_type"
    assert derive_megatron_fields(cfg)["olmo3_rope_layout"] == "per_type"
    with pytest.raises(ValueError, match="declares"):
        derive_megatron_fields(cfg, rope_layout="yarn_all")


def _olmo3_prepare(hf_config, params, tp=1):
    """The family's prepare step for an experiment whose sft_trainer params are ``params``."""
    from primus.backends.megatron.checkpoint import native_registry

    fam = native_registry.get_family("olmo3")
    ctx = native_registry.ConvertContext(
        hf_path="/hf",
        hf_config=hf_config,
        family=fam,
        experiment={"tp": tp, "pp": 1, "ep": 1, "ckpt_format": "torch_dist"},
        options={},
        config_value=lambda name: getattr(params, name, None),
    )
    return fam.hook("prepare")(ctx)


def test_olmo3_prepare_forwards_the_experiments_settings():
    """The family's prepare step reads the fine-tune's seq_length and olmo3_rope_layout and passes
    them to native_loader_olmo3.convert, with the vocabulary divisor times the run's TP (training
    pads to make_vocab_size_divisible_by * TP; the converter runs at TP = 1) and the padded size."""
    params = SimpleNamespace(seq_length=8192, make_vocab_size_divisible_by=64, olmo3_rope_layout="yarn_all")
    plan = _olmo3_prepare({"vocab_size": 100278}, params, tp=2)
    assert plan.convert_kwargs == {
        "seq_length": 8192,
        "make_vocab_size_divisible_by": 128,
        "rope_layout": "yarn_all",
        "expected_padded_vocab_size": 100352,
    }
    assert plan.path_suffix == "_vocab100352"
    # unset keys: converter defaults, divisor 128 * TP (TP 32 -> 4096: 102400 rows, as training)
    plan = _olmo3_prepare({"vocab_size": 100278}, SimpleNamespace(), tp=32)
    assert plan.convert_kwargs == {"make_vocab_size_divisible_by": 4096, "expected_padded_vocab_size": 102400}
    # no vocab_size: refused with a reason
    from primus.backends.megatron.checkpoint import native_registry

    with pytest.raises(native_registry.NativeConvertRefused, match="vocab_size"):
        _olmo3_prepare({}, SimpleNamespace())


def test_olmo3_family_dtype_from_the_dtype_key():
    """The released config.json has ``dtype`` (transformers >= 4.56), not ``torch_dtype``."""
    from primus.backends.megatron.checkpoint import native_registry

    fam = native_registry.get_family("olmo3")
    assert fam.hook("dtype_from_config")({"dtype": "float32"}) == "float32"


def _olmo3_hook_world(mod, monkeypatch, tmp_path, params):
    """An OLMo-3 HF dir (vocab_size 100278), an experiment with ``params`` and a fake converter that
    writes a torch_dist checkpoint; returns (hf_dir, convert calls)."""
    import json

    from primus.backends.megatron.checkpoint import native_convert_common as common

    hf = tmp_path / "hf" / "Olmo-3"
    hf.mkdir(parents=True)
    (hf / "config.json").write_text(
        json.dumps({"model_type": "olmo3", "torch_dtype": "bfloat16", "vocab_size": 100278})
    )
    module = SimpleNamespace(params=params)
    monkeypatch.setattr(mod, "load_primus_config", lambda path, cli: SimpleNamespace())
    monkeypatch.setattr(mod, "get_module_config", lambda cfg, name: module if name == "sft_trainer" else None)
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "data"))
    monkeypatch.setenv("NODE_RANK", "0")
    calls = []

    def fake_native_convert(hf_path, megatron_path, opts):
        calls.append(dict(opts))
        out = Path(megatron_path) / "release"
        out.mkdir(parents=True)
        (out / "metadata.json").write_text("{}")
        (Path(megatron_path) / common.TRACKER_FILENAME).write_text("release")

    monkeypatch.setattr(mod, "native_convert_checkpoint", fake_native_convert)
    return hf, calls


def test_convert_hook_main_passes_olmo3_settings(monkeypatch, tmp_path, capsys):
    """main() for an OLMo-3 hf_path runs the family's prepare step and hands its settings to the
    converter, with the padded vocabulary size the converter must reproduce."""
    mod = _load_convert_hook(monkeypatch)
    params = SimpleNamespace(seq_length=8192, olmo3_rope_layout="per_type", tensor_model_parallel_size=2)
    hf, calls = _olmo3_hook_world(mod, monkeypatch, tmp_path, params)
    params.hf_path = str(hf)
    monkeypatch.setattr(sys, "argv", ["hook", "--config", str(tmp_path / "exp.yaml")])
    mod.main()
    want = {
        "seq_length": 8192,
        "rope_layout": "per_type",
        "make_vocab_size_divisible_by": 256,
        "expected_padded_vocab_size": 100352,
    }
    assert len(calls) == 1 and calls[0]["convert_kwargs"] == want and calls[0]["family"] == "olmo3"
    out = capsys.readouterr().out
    assert "extra.pretrained_checkpoint=" in out and "_vocab100352" in out


def test_olmo3_checkpoint_cache_is_keyed_on_the_padded_vocabulary(monkeypatch, tmp_path, capsys):
    """Runs whose settings pad the vocabulary alike (any seq_length, TP 1..16 for 100278 tokens)
    share one converted checkpoint; a run that pads differently (TP 32: 102400) gets its own, and a
    checkpoint of another padding at an explicit native_ckpt_out is refused instead of reused."""
    from primus.backends.megatron.checkpoint import olmo3_convert_hooks as hooks

    mod = _load_convert_hook(monkeypatch)
    params = SimpleNamespace()
    hf, calls = _olmo3_hook_world(mod, monkeypatch, tmp_path, params)

    def run(out=None, tp=1, **settings):
        for k in ("seq_length", "olmo3_rope_layout"):
            setattr(params, k, settings.get(k))
        mod._native_main(
            str(hf),
            {"out": out, "ckpt_format": None},
            {"tp": tp, "pp": 1, "ep": 1, "ckpt_format": None},
            config_file="exp.yaml",
        )

    run(seq_length=8192)
    run(seq_length=4096, tp=2)  # TP 2, other seq_length: same padding
    run(seq_length=65536, tp=16, olmo3_rope_layout="per_type")
    assert len(calls) == 1
    run(seq_length=8192, tp=32)  # TP 32 pads to 102400
    assert len(calls) == 2 and calls[1]["convert_kwargs"]["expected_padded_vocab_size"] == 102400
    dirs = sorted(p.name for p in (tmp_path / "data" / "megatron_checkpoints").iterdir() if p.is_dir())
    assert dirs == ["Olmo-3_native_torch_dist_vocab100352", "Olmo-3_native_torch_dist_vocab102400"]

    out = tmp_path / "explicit"
    run(out=str(out))
    assert hooks.read_settings(out) == {"padded_vocab_size": 100352}
    with pytest.raises(SystemExit):
        run(out=str(out), tp=32)
    assert "was converted with settings" in capsys.readouterr().err
    assert len(calls) == 3
    # a checkpoint without the key file (an older Primus) is refused for an OLMo-3 run
    (out / hooks.CONVERT_SETTINGS_FILE).unlink()
    with pytest.raises(SystemExit):
        run(out=str(out))
    assert "no converter settings file" in capsys.readouterr().err


def test_convert_hook_writes_olmo3_settings_before_converting(monkeypatch, tmp_path):
    """The key file exists before the converter writes the tracker file, so a late node or a
    concurrent job that sees the tracker always sees the key it was converted with."""
    from primus.backends.megatron.checkpoint import olmo3_convert_hooks as hooks

    mod = _load_convert_hook(monkeypatch)
    hf, calls = _olmo3_hook_world(mod, monkeypatch, tmp_path, SimpleNamespace(seq_length=4096))
    seen = []
    inner = mod.native_convert_checkpoint

    def convert_and_look(hf_path, megatron_path, opts):
        seen.append(hooks.read_settings(Path(megatron_path)))
        inner(hf_path, megatron_path, opts)

    monkeypatch.setattr(mod, "native_convert_checkpoint", convert_and_look)
    mod._native_main(str(hf), {"out": None, "ckpt_format": None}, {"tp": 1}, config_file="exp.yaml")
    assert seen == [{"padded_vocab_size": 100352}] and len(calls) == 1


def test_convert_hook_refuses_unknown_family(monkeypatch, tmp_path):
    import json

    mod = _load_convert_hook(monkeypatch)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "gpt_neox"}))
    with pytest.raises(ValueError, match="no native converter"):
        mod.native_convert_checkpoint(str(tmp_path), str(tmp_path / "out"), {})


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

    for bad in ((0.5,), (8.0, 0.0), (8.0, -1.0), (1.0, 0.0), (1.0, 1.5)):
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


def test_rope_theta_is_compared_across_top_level_and_layer_types():
    """Every rope_theta given (top level, full_attention, sliding_attention) must agree; a missing
    one falls back to the others."""
    from primus.backends.megatron.checkpoint.native_loader_olmo3 import _rope_config

    base = {k: v for k, v in _HF_CFG.items() if k not in ("rope_scaling", "rope_theta")}
    full = {"rope_type": "yarn", "factor": 8.0, "original_max_position_embeddings": 8192}
    # the reviewer's config: top level and sliding 500000, full_attention without rope_theta
    cfg = {
        **base,
        "rope_theta": 500000,
        "rope_parameters": {
            "full_attention": dict(full),
            "sliding_attention": {"rope_type": "default", "rope_theta": 500000},
        },
    }
    assert _rope_config(cfg)[1:] == ("per_type", 500000.0)
    # sliding differs from the top level (full_attention silent): refused
    cfg["rope_parameters"]["sliding_attention"]["rope_theta"] = 10000
    with pytest.raises(ValueError, match="one rope_theta for all layer types"):
        _rope_config(cfg)
    # only the sliding layers give it: the full layers' theta is unknown, refused
    sliding = {"rope_type": "default", "rope_theta": 500000}
    cfg = {**base, "rope_parameters": {"full_attention": dict(full), "sliding_attention": sliding}}
    with pytest.raises(ValueError, match="no rope_theta"):
        _rope_config(cfg)


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
    assert cfg.olmo3_yarn_effective_attention_factor == pytest.approx(0.1 * math.log(8.0) + 1.0)
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


def test_checkpoint_tracker_name_matches_the_converter():
    from primus.backends.megatron.checkpoint import native_convert_common as common
    from primus.backends.megatron.core.models.olmo3 import olmo3_validation

    assert olmo3_validation.CHECKPOINT_TRACKER_FILENAME == common.TRACKER_FILENAME


def test_builder_refuses_spec_and_mtp_and_dispatch():
    olmo3_builders = _olmo3_builders()
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
    from megatron.core.transformer.transformer_config import TransformerConfig

    plain = TransformerConfig(num_layers=2, hidden_size=64, num_attention_heads=4)
    with pytest.raises(TypeError, match="Olmo3TransformerConfig"):
        olmo3_builders.olmo3_builder(SimpleNamespace(**base), True, True, config=plain)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TE layers need a GPU")
def test_builder_builds_olmo3_model_forward(init_parallel_state, monkeypatch):
    # LanguageModule (attention_backend auto) refuses preset NVTE_* backend variables
    for var in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
        monkeypatch.delenv(var, raising=False)
    olmo3_builders = _olmo3_builders()
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

    olmo3_builders = _olmo3_builders()

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


@_NEEDS_GPU_FIXTURE
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
        want_scale = base if sliding else base * cfg.olmo3_yarn_effective_attention_factor**2
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
@_NEEDS_GPU_FIXTURE
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

    olmo3_builders = _olmo3_builders()
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
    # what the built attention layers apply
    atts = [layer.self_attention for layer in model.decoder.layers]
    base = 1.0 / math.sqrt(f["kv_channels"])
    m2 = cfg.olmo3_yarn_effective_attention_factor**2
    assert [att.olmo3_sliding for att in atts] == [True] * 3 + [False]
    assert ["yarn" if att.olmo3_use_yarn else "plain" for att in atts] == [
        "yarn" if "yarn" in tok else "plain" for tok in plan.split()
    ]
    for att in atts:
        want = base * m2 if att.olmo3_use_yarn and yarn else base
        assert att.softmax_scale == pytest.approx(want, rel=1e-6)
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


@_NEEDS_GPU_FIXTURE
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
    # a 1-rank save; torch_dist unless args asks for another format
    layout = (m.tensor_model_parallel_size, m.pipeline_model_parallel_size, m.expert_model_parallel_size)
    assert layout == (1, 1, 1) and m.expert_tensor_parallel_size == 1
    assert (m.world_size, m.rank, m.save, m.ckpt_format) == (1, 0, "/tmp/x", "torch_dist")
    common._apply_runtime_margs(m, SimpleNamespace(**dict(vars(args), ckpt_format="torch")))
    assert m.ckpt_format == "torch"
    with pytest.raises(ValueError, match="native checkpoint format"):
        common._apply_runtime_margs(m, SimpleNamespace(**dict(vars(args), ckpt_format="zarr")))
    # the save iteration is set after validate_args (build_margs), not before it
    assert not hasattr(m, "iteration") and common.set_save_iteration(m).iteration == 1
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
        "for m in ('primus.backends.megatron.patches.olmo3_args_patches',):\n"
        "    spec = importlib.util.spec_from_file_location(m, m.replace('.', '/') + '.py')\n"
        "    mod = importlib.util.module_from_spec(spec)\n"
        "    spec.loader.exec_module(mod)\n"
        "print('ok')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=_REPO_ROOT, capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-2000:]


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


# The runtime / save args every native converter sets (Llama / Qwen3 / DeepSeek inline; OLMo-3 through set_converter_runtime_args)
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


def test_olmo3_converter_runtime_args_match_the_table():
    """native_loader_olmo3.set_converter_runtime_args + set_save_iteration set the same runtime /
    save args as the Llama / Qwen3 / DeepSeek converters (the table above, which those converters
    are checked against below), including the fusion and initialisation flags."""
    from primus.backends.megatron.checkpoint import native_loader_olmo3 as conv

    m = SimpleNamespace()
    conv.set_converter_runtime_args(m, "/tmp/x", "torch")
    conv.set_save_iteration(m)
    assert {k: getattr(m, k) for k in _BASE_CONVERTER_RUNTIME_ARGS} == _BASE_CONVERTER_RUNTIME_ARGS
    assert set(vars(m)) == set(_BASE_CONVERTER_RUNTIME_ARGS)  # nothing else


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
@_NEEDS_GPU_FIXTURE
def test_other_converters_keep_their_runtime_args(family, hf, monkeypatch):
    """Llama / Qwen3 / DeepSeek build_margs (their own inline runtime blocks; they do not call
    native_loader_olmo3.set_converter_runtime_args / set_save_iteration) give the runtime / save
    args above after validate_args; test_olmo3_converter_runtime_args_match_the_table checks that
    the OLMo-3 helpers set the same table. Needs a GPU to import megatron.training (mamba_ssm /
    fla); the capability probe below covers a GPU that reports no compute capability."""
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
        ckpt_format="torch",
    )
    margs = conv.build_margs(conv.derive_megatron_fields(hf), args)
    got = {k: getattr(margs, k) for k in _BASE_CONVERTER_RUNTIME_ARGS}
    assert got == _BASE_CONVERTER_RUNTIME_ARGS
    assert margs.bf16 and not margs.fp16 and margs.params_dtype is torch.bfloat16


def test_converter_refuses_attention_factor_with_factor_one():
    """A YaRN rope config with factor 1.0 and attention_factor != 1 would convert but lose the
    logit scale (OLMo-3 builds no YaRN table for factor 1): it is refused."""
    from primus.backends.megatron.checkpoint.native_loader_olmo3 import (
        derive_megatron_fields,
    )

    cfg = json.loads(json.dumps(_HF_CFG))
    cfg["max_position_embeddings"] = 8192
    cfg["rope_scaling"] = dict(cfg["rope_scaling"], factor=1.0, attention_factor=1.5)
    with pytest.raises(ValueError, match="attention_factor"):
        derive_megatron_fields(cfg, rope_layout="per_type")
    cfg["rope_scaling"]["attention_factor"] = 1.0
    derive_megatron_fields(cfg, rope_layout="per_type")


def test_convert_hook_dispatches_every_family_and_refuses_unknown_ones(monkeypatch, tmp_path):
    """Run the hook for every registered family: each reaches its own converter's convert();
    an unregistered family raises."""
    from primus.backends.megatron.checkpoint import native_registry

    mod = _load_convert_hook(monkeypatch)
    (tmp_path / "config.json").write_text("{}")
    families = [f.name for f in native_registry.families()]
    assert "olmo3" in families
    for fam in families:
        conv = native_registry.get_family(fam).converter_module()
        calls = []
        monkeypatch.setattr(conv, "convert", lambda *a, _c=calls, **k: _c.append(a))
        mod.native_convert_checkpoint(str(tmp_path), str(tmp_path / "out"), {"family": fam, "dtype": "bf16"})
        assert calls == [(str(tmp_path), str(tmp_path / "out"))], fam
    with pytest.raises(ValueError, match="no native converter family 'mistral'"):
        mod.native_convert_checkpoint(
            str(tmp_path), str(tmp_path / "out"), {"family": "mistral", "dtype": "bf16"}
        )


@pytest.mark.parametrize(
    "key, bad",
    [
        ("max_position_embeddings", "8k"),
        ("max_position_embeddings", 0),
        ("max_position_embeddings", -5),
        ("max_position_embeddings", 65536.0),
        ("rms_norm_eps", "x"),
        ("rms_norm_eps", -1e-6),
        ("rms_norm_eps", 0),
        ("rms_norm_eps", True),
        ("rope_theta", True),
        ("rope_theta", "500k"),
        ("rope_theta", 0),
        ("vocab_size", True),
        ("vocab_size", 0),
        ("vocab_size", "100278"),
    ],
)
def test_converter_refuses_malformed_required_numbers(key, bad):
    """max_position_embeddings / vocab_size must be positive ints, rms_norm_eps / rope_theta
    positive numbers (no bools, no strings); a bad value fails with a message naming the key."""
    from primus.backends.megatron.checkpoint.native_loader_olmo3 import (
        derive_megatron_fields,
    )

    with pytest.raises(ValueError, match=key):
        derive_megatron_fields({**_HF_CFG, key: bad}, rope_layout="per_type")


def test_converter_accepts_the_released_numbers_and_no_vocab_size():
    from primus.backends.megatron.checkpoint.native_loader_olmo3 import (
        derive_megatron_fields,
    )

    f = derive_megatron_fields({**_HF_CFG, "vocab_size": 100278}, rope_layout="per_type")
    assert f["norm_epsilon"] == 1e-6 and f["rotary_base"] == 500000.0 and f["vocab_size"] == 100278
    cfg = {k: v for k, v in _HF_CFG.items() if k != "vocab_size"}
    assert derive_megatron_fields(cfg, rope_layout="per_type")["vocab_size"] is None


@pytest.mark.parametrize(
    "key, bad",
    [
        ("beta_fast", float("nan")),
        ("beta_fast", "32"),
        ("beta_fast", True),
        ("beta_fast", -1.0),
        ("beta_fast", float("inf")),
        ("beta_slow", "1"),
        ("beta_slow", float("nan")),
        ("beta_slow", 0),
        ("original_max_position_embeddings", float("inf")),
        ("original_max_position_embeddings", 8192.5),
        ("original_max_position_embeddings", "8192"),
        ("factor", float("nan")),
        ("factor", float("inf")),
        ("attention_factor", float("nan")),
    ],
)
def test_yarn_numbers_are_finite_typed_and_in_range(key, bad):
    """Every YaRN number is an int/float (no string, no bool), finite and in range; a bad value
    raises ValueError naming the key (inf no longer escapes as OverflowError, NaN no longer slips
    through the beta_fast >= beta_slow check)."""
    from primus.backends.megatron.checkpoint.native_loader_olmo3 import (
        derive_megatron_fields,
    )

    rope = {**_HF_CFG["rope_scaling"], key: bad}
    with pytest.raises(ValueError, match=key):
        derive_megatron_fields({**_HF_CFG, "rope_scaling": rope}, rope_layout="per_type")


def test_yarn_integral_float_original_max_position_embeddings_is_accepted():
    from primus.backends.megatron.checkpoint.native_loader_olmo3 import (
        derive_megatron_fields,
    )

    rope = {**_HF_CFG["rope_scaling"], "original_max_position_embeddings": 8192.0}
    f = derive_megatron_fields({**_HF_CFG, "rope_scaling": rope}, rope_layout="per_type")
    assert f["olmo3_yarn_original_max_position_embeddings"] == 8192
    assert isinstance(f["olmo3_yarn_original_max_position_embeddings"], int)
