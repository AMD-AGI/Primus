###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the TE hipBLASLt GEMM fallback patch.

The patch recovers from ``HIPBLASLT Error: 6`` in TE's ROCm GEMM by
recomputing the call with torch. These tests drive the wrapper with a stub
``general_gemm`` that raises the same error, so they check the two things
that must hold on a real ROCm node: the fallback reproduces hipBLASLt's
result for every layout TE uses, and it refuses calls it cannot reproduce.
"""

from types import SimpleNamespace

import pytest
import torch

from primus.backends.megatron.patches.te_patches import (
    hipblaslt_gemm_fallback_patches as patch_mod,
)

HIPBLASLT_ERROR = RuntimeError(
    "/staging/TransformerEngine/transformer_engine/common/gemm/rocm_gemm.hip:1934 "
    "in function hipblaslt_gemm: HIPBLASLT Error: 6"
)


def _stub_general_gemm(
    A,
    B,
    out_dtype=None,
    quantization_params=None,
    gelu=False,
    gelu_in=None,
    alpha=1.0,
    beta=None,
    accumulate=False,
    layout="TN",
    out=None,
    bias=None,
    use_split_accumulator=False,
    grad=False,
    ub=None,
    ub_type=None,
    extra_output=None,
    bulk_overlap=False,
):
    """Stand-in for TE's ``general_gemm`` that always fails like gfx950 does."""
    raise HIPBLASLT_ERROR


@pytest.fixture(autouse=True)
def _reset_patch_state(monkeypatch):
    """Keep the module-level failure memo from leaking across tests."""
    monkeypatch.setattr(patch_mod, "_broken_gemms", {})
    monkeypatch.setattr(patch_mod, "_workspace_grown", True)  # skip the retry step
    yield


@pytest.mark.parametrize(
    "layout,shape_a,shape_b",
    [
        # fprop: A=weight[out, hidden], B=inp[tokens, hidden] -> inp @ weight.T
        ("TN", (48, 16), (32, 16)),
        # dgrad: A=weight[out, hidden], B=grad_out[tokens, out] -> grad_out @ weight
        ("NN", (48, 16), (32, 48)),
        # wgrad: A=x[tokens, hidden], B=dy[tokens, out] -> dy.T @ x
        ("NT", (32, 16), (32, 48)),
    ],
)
def test_fallback_matches_reference_for_every_te_layout(layout, shape_a, shape_b):
    general_gemm = patch_mod._make_fallback_general_gemm(_stub_general_gemm)
    A = torch.randn(shape_a, dtype=torch.float32)
    B = torch.randn(shape_b, dtype=torch.float32)

    out, bias_grad, gelu_input, extra_output = general_gemm(A, B, layout=layout, grad=layout == "NT")

    reference = {
        "TN": lambda: B @ A.T,
        "NN": lambda: B @ A,
        "NT": lambda: B.T @ A,
    }[layout]()
    torch.testing.assert_close(out, reference)
    assert bias_grad is None and gelu_input is None and extra_output is None


def test_failed_shape_is_remembered_and_skips_te():
    calls = {"n": 0}

    def counting_gemm(A, B, layout="TN", grad=False, **kwargs):
        calls["n"] += 1
        raise HIPBLASLT_ERROR

    general_gemm = patch_mod._make_fallback_general_gemm(counting_gemm)
    A = torch.randn(32, 16)
    B = torch.randn(32, 48)

    first, *_ = general_gemm(A, B, layout="NT", grad=True)
    second, *_ = general_gemm(A, B, layout="NT", grad=True)

    torch.testing.assert_close(first, second)
    assert calls["n"] == 1, "the known-broken shape should not be handed to TE again"


def test_known_broken_shape_still_defers_to_te_for_unsupported_calls():
    """The memo is keyed on shapes only, so other kwargs must be re-checked."""
    general_gemm = patch_mod._make_fallback_general_gemm(_stub_general_gemm)
    A = torch.randn(32, 16)
    B = torch.randn(32, 48)

    general_gemm(A, B, layout="NT", grad=True)  # memoises the shape
    with pytest.raises(RuntimeError, match="HIPBLASLT Error: 6"):
        general_gemm(A, B, layout="NT", grad=True, ub=object())


def test_wgrad_bias_grad_and_accumulation():
    general_gemm = patch_mod._make_fallback_general_gemm(_stub_general_gemm)
    x = torch.randn(32, 16)
    dy = torch.randn(32, 48)
    bias = torch.zeros(48)
    main_grad = torch.randn(48, 16)
    expected = main_grad + dy.T @ x

    out, bias_grad, *_ = general_gemm(
        x,
        dy,
        layout="NT",
        grad=True,
        bias=bias,
        out=main_grad,
        accumulate=True,
    )

    assert out is main_grad
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(bias_grad, dy.sum(dim=0))


def test_fprop_bias_epilogue_is_applied():
    general_gemm = patch_mod._make_fallback_general_gemm(_stub_general_gemm)
    weight = torch.randn(48, 16)
    inp = torch.randn(32, 16)
    bias = torch.randn(48)

    out, bias_grad, *_ = general_gemm(weight, inp, layout="TN", bias=bias)

    torch.testing.assert_close(out, inp @ weight.T + bias)
    assert bias_grad is None


def test_out_dtype_is_honoured():
    general_gemm = patch_mod._make_fallback_general_gemm(_stub_general_gemm)
    A = torch.randn(32, 16, dtype=torch.bfloat16)
    B = torch.randn(32, 48, dtype=torch.bfloat16)

    out, *_ = general_gemm(A, B, layout="NT", grad=True, out_dtype=torch.float32)

    assert out.dtype == torch.float32


def test_quantized_gemm_reraises_te_error():
    """FP8/MXFP4 operands are never rerouted; the original error must surface."""

    class FakeQuantizedTensor(torch.Tensor):
        pass

    general_gemm = patch_mod._make_fallback_general_gemm(_stub_general_gemm)
    A = FakeQuantizedTensor(torch.randn(32, 16))
    B = torch.randn(32, 48)

    with pytest.raises(RuntimeError, match="HIPBLASLT Error: 6"):
        general_gemm(A, B, layout="NT", grad=True)


def test_comm_overlap_gemm_reraises_te_error():
    general_gemm = patch_mod._make_fallback_general_gemm(_stub_general_gemm)
    A = torch.randn(32, 16)
    B = torch.randn(32, 48)

    with pytest.raises(RuntimeError, match="HIPBLASLT Error: 6"):
        general_gemm(A, B, layout="NT", grad=True, ub=object(), ub_type=object())


def test_non_hipblaslt_errors_are_not_swallowed():
    def oom_gemm(A, B, **kwargs):
        raise RuntimeError("HIP out of memory")

    general_gemm = patch_mod._make_fallback_general_gemm(oom_gemm)

    with pytest.raises(RuntimeError, match="HIP out of memory"):
        general_gemm(torch.randn(32, 16), torch.randn(32, 48), layout="NT")


def test_successful_gemm_is_passed_through_untouched():
    sentinel = (object(), None, None, None)

    def happy_gemm(A, B, **kwargs):
        return sentinel

    general_gemm = patch_mod._make_fallback_general_gemm(happy_gemm)

    assert general_gemm(torch.randn(4, 4), torch.randn(4, 4), layout="TN") is sentinel


def _ctx(**params):
    """Minimal stand-in for a PatchContext carrying YAML overrides."""
    return SimpleNamespace(extra={"module_config": SimpleNamespace(params=SimpleNamespace(**params))})


def test_patch_is_disabled_off_rocm(monkeypatch):
    """Registration is global, so the condition must exclude CUDA runs."""
    monkeypatch.setattr(torch.version, "hip", None, raising=False)

    assert patch_mod._fallback_enabled(_ctx()) is False


def test_enabled_by_default_on_rocm(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "7.0.0", raising=False)
    monkeypatch.delenv("PRIMUS_TE_HIPBLASLT_GEMM_FALLBACK", raising=False)

    assert patch_mod._fallback_enabled(_ctx()) is True


def test_yaml_knob_overrides_env(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "7.0.0", raising=False)
    monkeypatch.setenv("PRIMUS_TE_HIPBLASLT_GEMM_FALLBACK", "1")

    assert patch_mod._fallback_enabled(_ctx(te_hipblaslt_gemm_fallback=False)) is False


def test_env_knob_disables(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "7.0.0", raising=False)
    monkeypatch.setenv("PRIMUS_TE_HIPBLASLT_GEMM_FALLBACK", "0")

    assert patch_mod._fallback_enabled(_ctx()) is False
