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


# ---------------------------------------------------------------------------
# Catalog coverage for axes added after session 20260513T024603Z
# (skills/workflow/axis_taxonomy.md §§2.6-2.13). These are the axes that
# moved the champion in the first runnable session — they MUST be in the
# catalog so REPLAN doesn't depend on orchestrator hand-injection.
# ---------------------------------------------------------------------------


def test_session_winning_axes_are_in_catalog() -> None:
    """The four axes that promoted a champion in session 20260513T024603Z
    must all be translatable. See IMPL_VS_DESIGN.md §2."""
    winners = {
        "turbo_deepep_num_cu": ("trainer_override", "turbo_deepep_num_cu"),
        "fp8_recipe": ("trainer_override", "fp8_recipe"),
        "OMP_NUM_THREADS": ("env", "OMP_NUM_THREADS"),
        "apply_rope_fusion": ("trainer_override", "apply_rope_fusion"),
    }
    for axis, (channel, key) in winners.items():
        action = axt.translate(
            axis, "delayed" if axis == "fp8_recipe" else 4 if axis == "OMP_NUM_THREADS" else True
        )
        assert action is not None, f"session winner axis missing from catalog: {axis}"
        assert action.channel == channel, f"{axis} on wrong channel: {action.channel}"
        assert action.key == key, f"{axis} key mismatch: {action.key}"


def test_fp8_recipe_renders_string() -> None:
    a = axt.translate("fp8_recipe", "delayed")
    assert a is not None and a.channel == "trainer_override"
    assert a.rendered_value == "delayed"


def test_omp_num_threads_renders_int_as_string_passthrough() -> None:
    a = axt.translate("OMP_NUM_THREADS", 4)
    assert a is not None and a.channel == "env"
    assert a.rendered_value == "4"


def test_megatron_fusion_axes_are_trainer_override() -> None:
    for axis in (
        "apply_rope_fusion",
        "bias_activation_fusion",
        "bias_dropout_fusion",
        "masked_softmax_fusion",
    ):
        a = axt.translate(axis, True)
        assert a is not None and a.channel == "trainer_override"
        assert a.key == axis


def test_pp_only_axes_translate_but_constraint_check_will_gate_them() -> None:
    """defer_embedding_wgrad_compute / overlap_p2p_communication are valid
    trainer overrides but require pp>=2 (axis_taxonomy.md §2.9). The
    translator returns them; constraint.check rejects when pp<2."""
    for axis in (
        "defer_embedding_wgrad_compute",
        "overlap_p2p_communication",
        "overlap_param_gather_with_optimizer_step",
    ):
        a = axt.translate(axis, True)
        assert a is not None and a.channel == "trainer_override"


def test_cuda_graph_family_in_catalog_for_diagnose_visibility() -> None:
    """CUDA graphs are stack-blocked on MI355X (axis_taxonomy.md §2.8) but
    the translator still maps them so DIAGNOSE can name the axis and
    REPLAN can attach axis_meta.known_blocker."""
    for axis in ("enable_cuda_graph", "external_cuda_graph", "cuda_graph_impl", "cuda_graph_scope"):
        a = axt.translate(axis, "local" if axis == "cuda_graph_impl" else True)
        assert a is not None and a.channel == "trainer_override"


def test_rccl_extras_and_hsa_in_env_channel() -> None:
    for axis in (
        "RCCL_PROTO",
        "RCCL_ALGO",
        "RCCL_NTHREADS",
        "TORCH_NCCL_HIGH_PRIORITY",
        "HSA_NO_SCRATCH_RECLAIM",
        "HSA_ENABLE_INTERRUPT",
        "GPU_MAX_HW_QUEUES",
        "MIOPEN_FIND_MODE",
        "PRIMUS_HIPBLASLT_TUNING",
    ):
        a = axt.translate(axis, "1")
        assert a is not None and a.channel == "env"


def test_moe_router_dtype_in_catalog() -> None:
    """moe_router_dtype is registered (axis_taxonomy.md §2.13) so
    constraint.check can enforce the DeepEP fp32 mutex."""
    a = axt.translate("moe_router_dtype", "fp32")
    assert a is not None and a.channel == "trainer_override"
