"""Tests for `pilot.tools._axis_translator`.

The translator is a single-source-of-truth for axis -> override mapping.
Coverage targets:

1. Every axis that the trace-driven diagnose engine can emit must map to
   a known channel (else replan would silently drop it). This guards
   against drift between diagnose.py and the translator.
2. Channel selection (trainer_override / structural / env) is correct.
3. Boolean rendering for env values is normalized to "1" / "0".
4. Unknown axes return None instead of raising.
"""

from __future__ import annotations

from pilot.tools import _axis_translator as axt
from pilot.tools import diagnose as diag


# ---------------------------------------------------------------------------
# Channel selection
# ---------------------------------------------------------------------------


def test_trainer_override_axis() -> None:
    a = axt.translate("turbo_deepep_use_comm_stream", True)
    assert a is not None
    assert a.channel == "trainer_override"
    assert a.key == "turbo_deepep_use_comm_stream"
    assert a.rendered_value is True


def test_structural_axis() -> None:
    a = axt.translate("micro_batch_size", 16)
    assert a is not None
    assert a.channel == "structural"
    assert a.key == "micro_batch_size"
    assert a.rendered_value == 16


def test_env_axis_string_passthrough() -> None:
    a = axt.translate("NCCL_BUFFSIZE", "16M")
    assert a is not None
    assert a.channel == "env"
    assert a.key == "NCCL_BUFFSIZE"
    assert a.rendered_value == "16M"


def test_env_axis_bool_normalized() -> None:
    a = axt.translate("RCCL_MSCCL_ENABLE", True)
    assert a is not None and a.channel == "env"
    assert a.rendered_value == "1"
    a_off = axt.translate("RCCL_MSCCL_ENABLE", False)
    assert a_off is not None and a_off.rendered_value == "0"


def test_attention_kernel_yaml_key_mapping() -> None:
    """`attention_kernel` axis name maps to `attention_backend` YAML key."""
    a = axt.translate("attention_kernel", "flash")
    assert a is not None and a.channel == "trainer_override"
    assert a.key == "attention_backend"


def test_unknown_axis_returns_none() -> None:
    assert axt.translate("definitely_not_an_axis", 42) is None


def test_is_known_and_channel_of_match_translate() -> None:
    for axis in (
        "turbo_deepep_use_comm_stream",
        "micro_batch_size",
        "NCCL_BUFFSIZE",
    ):
        assert axt.is_known(axis)
        ch = axt.channel_of(axis)
        assert ch is not None
        translated = axt.translate(axis, 1)
        assert translated is not None and translated.channel == ch


# ---------------------------------------------------------------------------
# Coverage: every axis the engine can emit MUST translate.
# ---------------------------------------------------------------------------


# Axes that `_build_axes()` and `_build_evidence()` in pilot.tools.diagnose
# can produce today (kept in sync with skills/workflow/diagnose.md §9.1).
_DIAGNOSE_EMITTABLE_AXES = {
    "turbo_deepep_use_comm_stream",
    "turbo_deepep_num_cu",
    "overlap_grad_reduce",
    "overlap_param_gather",
    "MOE_PERMUTE_FUSION",
    "attention_kernel",
    "gradient_accumulation_fusion",
    "micro_batch_size",
    "expert_model_parallel_size",
    "virtual_pipeline_model_parallel_size",
    "recompute_granularity",
}

# env_suspect flags `_build_axes()` can emit; replan turns these into
# env-channel actions.
_DIAGNOSE_EMITTABLE_ENV_FLAGS = {
    "turbo_deepep_use_comm_stream",  # also a trainer override
    "RCCL_MSCCL_ENABLE",
    "PYTORCH_HIP_ALLOC_CONF",
}


def test_every_axis_emittable_by_diagnose_is_known() -> None:
    missing = [a for a in _DIAGNOSE_EMITTABLE_AXES if not axt.is_known(a)]
    assert not missing, f"axes emitted by diagnose but unknown to translator: {missing}"


def test_every_env_flag_emittable_by_diagnose_is_known() -> None:
    missing = [a for a in _DIAGNOSE_EMITTABLE_ENV_FLAGS if not axt.is_known(a)]
    assert not missing, f"env flags emitted by diagnose but unknown to translator: {missing}"


def test_diagnose_module_imports() -> None:
    """Sanity: diagnose import does not crash, so the engine -> translator
    contract above can actually be exercised at runtime."""
    assert hasattr(diag, "run")
