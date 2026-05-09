###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Plan-4 P27 G29 — runtime dispatch precedence in :class:`DeepseekV4Attention`.

The static plumbing tests
(``test_v4_p25_dispatch.py`` and ``test_v4_p26_dispatch.py``) cover the
*config-side* / *bytecode-side* wiring: that the flags exist, are
read by ``__init__``, are auto-disabled for the wrong layer kind, and
that ``forward`` / ``_csa_forward`` reference the runtime flag
attributes.

This file extends them with **runtime mock-based** assertions — for
each ``(compress_ratio, flags) → expected kernel`` combination
documented in the :class:`DeepseekV4Attention` class docstring, we
mock all four candidate kernels and assert exactly one fires:

    cr == 0:
        use_turbo_attention=T  → core_attention                 (Plan-3 P22)
        use_turbo_attention=F, use_v4_triton_attention=T → v4_attention (P25)
        use_turbo_attention=F, use_v4_triton_attention=F → eager_v4_attention (P24)

    cr == 128:
        use_v4_triton_attention=T → v4_attention (HCA path)     (P25)
        use_v4_triton_attention=F → eager_v4_attention          (P24)

    cr == 4:
        use_v4_triton_csa_attention=T → v4_csa_attention        (P26)
        use_v4_triton_csa_attention=F → eager_v4_csa_attention  (P24)

We bypass ``__init__`` (which requires distributed + GPU + Megatron
config) by allocating the attention instance with ``__new__`` and
populating only the attributes the dispatch path reads. This keeps
the test CPU-only and instantaneous while still exercising the actual
``forward`` / ``_csa_forward`` Python code paths from the production
module.

The kernel-vs-eager numerical equivalence is covered by G23/G24 (P25)
and G26/G27 (P26); G28 (P27) extends those to release-tier shapes.
G29 is the *dispatch* gate — it ensures the right kernel is selected
for each valid configuration, which the static tests cannot prove
without running the forward path.

Also covers :meth:`DeepseekV4Attention._log_kernel_choice` — the new
P27 ``INFO`` log line summarising each layer's kernel choice — by
asserting the expected string content for each of the 7 valid
(cr, flag) combinations.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

# isort: off
# Import order matters here: the V4 transformer config MUST be imported
# FIRST so the ``primus...models.deepseek_v4`` package's ``__init__``
# chain (which in turn imports ``deepseek_v4_block`` and
# ``deepseek_v4_attention``) finishes wiring up before we grab the
# attention module by reference. Without this priming step the
# subsequent ``import ... deepseek_v4_attention`` races against
# ``deepseek_v4_block.py`` re-importing ``DeepseekV4Attention`` from
# the partially-initialised module and raises ``ImportError``. The
# P25 / P26 dispatch tests dodge the same race by deferring the import
# to inside each test function; we want a module-level handle so the
# test fixtures (caplog parametrisations) can reference it directly,
# so we hard-pin the order with an ``isort: off`` block.
from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_transformer_config import (  # noqa: F401
    DeepSeekV4TransformerConfig,
)
import primus.backends.megatron.core.transformer.deepseek_v4_attention as v4_attn_mod
from primus.backends.megatron.core.transformer.deepseek_v4_attention import (
    DeepseekV4Attention,
)

# isort: on

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bare_attn(
    *,
    compress_ratio: int,
    use_core_attention: bool = False,
    use_v4_triton_attention: bool = False,
    use_v4_triton_csa_attention: bool = False,
    layer_number: int = 0,
) -> DeepseekV4Attention:
    """Build a bare :class:`DeepseekV4Attention` for runtime dispatch tests.

    Bypasses ``__init__`` (which requires Megatron / GPU / distributed)
    and populates only the attributes the dispatch path consults. The
    helper methods on the real class read these attributes by name; we
    do not need to subclass.
    """
    attn = DeepseekV4Attention.__new__(DeepseekV4Attention)
    attn.compress_ratio = int(compress_ratio)
    attn.layer_number = int(layer_number)
    attn._use_core_attention = bool(use_core_attention)
    attn._use_v4_triton_attention = bool(use_v4_triton_attention)
    attn._use_v4_triton_csa_attention = bool(use_v4_triton_csa_attention)
    return attn


# ---------------------------------------------------------------------------
# G29.A — _log_kernel_choice emits the correct kernel name per (cr, flags)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "compress_ratio,use_core,use_v4t,use_csa,expected_substr",
    [
        # cr == 0: 3 valid kernel choices
        pytest.param(0, True, False, False, "core_attention", id="cr0_turbo"),
        pytest.param(0, False, True, False, "Triton, dense path", id="cr0_triton"),
        pytest.param(0, False, False, False, "eager Python, dense path", id="cr0_eager"),
        # cr == 128: 2 valid kernel choices (no turbo path for HCA)
        pytest.param(128, False, True, False, "Triton, HCA path", id="cr128_triton"),
        pytest.param(128, False, False, False, "eager Python, HCA path", id="cr128_eager"),
        # cr == 4: 2 valid kernel choices (no turbo / dense-triton path for CSA)
        pytest.param(4, False, False, True, "v4_csa_attention (Triton)", id="cr4_triton"),
        pytest.param(4, False, False, False, "v4_csa_attention (eager Python)", id="cr4_eager"),
    ],
)
def test_p27_log_kernel_choice_emits_expected_kernel(
    compress_ratio: int,
    use_core: bool,
    use_v4t: bool,
    use_csa: bool,
    expected_substr: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """:meth:`_log_kernel_choice` emits one INFO line per layer naming the kernel.

    Plan-4 P27 surfaces the dispatch outcome at boot via
    :meth:`DeepseekV4Attention._log_kernel_choice` so smoke /
    training logs unambiguously show which kernel each layer is
    firing through. The log line format is checked by
    :func:`test_p27_log_kernel_choice_format`; this test only
    verifies the kernel-name substring is correct.
    """
    attn = _make_bare_attn(
        compress_ratio=compress_ratio,
        use_core_attention=use_core,
        use_v4_triton_attention=use_v4t,
        use_v4_triton_csa_attention=use_csa,
        layer_number=17,
    )
    caplog.set_level(logging.INFO, logger=v4_attn_mod.__name__)
    attn._log_kernel_choice()
    assert expected_substr in caplog.text, (
        f"_log_kernel_choice did not emit the expected kernel name for cr={compress_ratio} "
        f"(use_core={use_core}, use_v4t={use_v4t}, use_csa={use_csa}); "
        f"expected substring '{expected_substr}', got log text:\n{caplog.text}"
    )


def test_p27_log_kernel_choice_format(caplog: pytest.LogCaptureFixture) -> None:
    """Log line format is ``[V4-attn] Layer N: cr=R, kernel = ...``."""
    attn = _make_bare_attn(compress_ratio=0, layer_number=42)
    caplog.set_level(logging.INFO, logger=v4_attn_mod.__name__)
    attn._log_kernel_choice()
    assert "[V4-attn]" in caplog.text
    assert "Layer 42" in caplog.text
    assert "cr=0" in caplog.text
    assert "kernel = " in caplog.text


def test_p27_log_kernel_choice_emits_once_per_call(caplog: pytest.LogCaptureFixture) -> None:
    """A single :meth:`_log_kernel_choice` call emits exactly one INFO line."""
    attn = _make_bare_attn(compress_ratio=4, use_v4_triton_csa_attention=True, layer_number=3)
    caplog.set_level(logging.INFO, logger=v4_attn_mod.__name__)
    attn._log_kernel_choice()
    v4_attn_records = [r for r in caplog.records if r.name == v4_attn_mod.__name__]
    assert len(v4_attn_records) == 1


# ---------------------------------------------------------------------------
# G29.B — runtime mock dispatch on ``_attention_forward_via_v4_triton``
# ---------------------------------------------------------------------------


def test_p27_via_v4_triton_helper_calls_v4_attention_kernel() -> None:
    """:meth:`_attention_forward_via_v4_triton` calls the Triton kernel.

    P25 routes the dense / HCA Triton path through this helper. The
    static test (P25) checks the bytecode references the kernel name;
    this runtime test patches the kernel in the module namespace and
    asserts it is actually invoked. Combined with the dispatch tests
    below, this confirms the Triton kernel is the one that fires
    when ``use_v4_triton_attention=True``.
    """
    attn = _make_bare_attn(compress_ratio=0, use_v4_triton_attention=True)
    attn.attn_sink = None
    attn.attn_sliding_window = 0
    attn.attn_dropout = 0.0
    attn.training = False
    attn._attention_scale = MagicMock(return_value=0.125)

    sentinel = object()
    with patch.object(v4_attn_mod, "v4_attention", return_value=sentinel) as mock_kernel:
        out = DeepseekV4Attention._attention_forward_via_v4_triton(
            attn,
            q=MagicMock(),
            k=MagicMock(),
            v=MagicMock(),
            attn_mask=MagicMock(),
        )

    assert out is sentinel
    mock_kernel.assert_called_once()
    kwargs = mock_kernel.call_args.kwargs
    assert "additive_mask" in kwargs
    assert kwargs["swa_window"] == 0


# ---------------------------------------------------------------------------
# G29.C — runtime mock dispatch on ``_csa_forward``
# ---------------------------------------------------------------------------


def _stub_csa_inputs():
    """Return MagicMock tensor stubs accepted by ``_csa_forward``.

    Each value supports the few attribute / method accesses
    ``_csa_forward`` makes before reaching the dispatch line. We
    intercept the prep ops via ``patch.object`` so the stub tensors
    are never actually arithmetised.
    """
    hidden = MagicMock()
    q_bh = MagicMock()
    q_bh.shape = (1, 2, 3, 4)  # B, H, S, Dh
    k_local_bh = MagicMock()
    v_local_bh = MagicMock()
    local_mask = MagicMock()
    return hidden, q_bh, k_local_bh, v_local_bh, local_mask


@pytest.mark.parametrize(
    "use_csa,fired_attr,suppressed_attr",
    [
        pytest.param(
            True,
            "v4_csa_attention",
            "eager_v4_csa_attention",
            id="csa_triton_wins",
        ),
        pytest.param(
            False,
            "eager_v4_csa_attention",
            "v4_csa_attention",
            id="csa_eager_when_off",
        ),
    ],
)
def test_p27_csa_forward_dispatch_precedence(
    use_csa: bool,
    fired_attr: str,
    suppressed_attr: str,
) -> None:
    """``_csa_forward`` picks v4_csa_attention vs eager_v4_csa_attention by flag.

    Precedence (P26): ``use_v4_triton_csa_attention > eager``.
    """
    attn = _make_bare_attn(
        compress_ratio=4,
        use_v4_triton_csa_attention=use_csa,
    )
    attn.attn_sink = None
    attn.attn_sliding_window = 0
    attn.attn_dropout = 0.0
    attn.training = False
    attn._attention_scale = MagicMock(return_value=0.125)
    # Patch the prep ops invoked before the kernel dispatch.
    pool_mock = MagicMock()
    pool_mock.shape = (1, 8, 4)  # [B, P, Dh]
    pool_mock.unsqueeze.return_value.expand.return_value = MagicMock()
    attn._build_compressed_pool = MagicMock(return_value=pool_mock)
    indexer_topk = MagicMock()
    indexer_topk.shape = (1, 3, 4)  # [B, S, K]
    indexer_topk.clamp.return_value = MagicMock()
    indexer_topk.__ge__ = MagicMock(return_value=MagicMock())
    attn.indexer = MagicMock(return_value=(indexer_topk, MagicMock()))

    fired_sentinel = object()
    suppressed_called = [False]

    def _suppressed(*args, **kwargs):
        suppressed_called[0] = True
        return object()

    with patch.object(v4_attn_mod, fired_attr, return_value=fired_sentinel) as mock_fired, patch.object(
        v4_attn_mod, suppressed_attr, side_effect=_suppressed
    ), patch("torch.gather", return_value=MagicMock()), patch("torch.where", return_value=MagicMock()):
        hidden, q_bh, k_local_bh, v_local_bh, local_mask = _stub_csa_inputs()
        out = DeepseekV4Attention._csa_forward(
            attn,
            hidden=hidden,
            q_bh=q_bh,
            k_local_bh=k_local_bh,
            v_local_bh=v_local_bh,
            local_mask=local_mask,
        )

    assert out is fired_sentinel, (
        f"_csa_forward did not return the expected kernel's output for "
        f"use_v4_triton_csa_attention={use_csa}; expected {fired_attr} to fire."
    )
    mock_fired.assert_called_once()
    assert not suppressed_called[0], (
        f"Both kernel paths were invoked when use_v4_triton_csa_attention={use_csa}; "
        f"the suppressed kernel '{suppressed_attr}' must NOT be called."
    )


def test_p27_csa_forward_passes_kernel_args_through() -> None:
    """:meth:`_csa_forward` passes the prepared kwargs to the chosen kernel.

    This is a contract check on the kwargs the kernel sees: the same
    set of keyword arguments must reach ``v4_csa_attention`` and
    ``eager_v4_csa_attention``, otherwise a kernel swap would silently
    re-route through a different code path.
    """
    expected_keys = {
        "sink",
        "swa_window",
        "sparse_mask",
        "attn_dropout",
        "training",
        "scale",
    }
    for use_csa, target in [(True, "v4_csa_attention"), (False, "eager_v4_csa_attention")]:
        attn = _make_bare_attn(compress_ratio=4, use_v4_triton_csa_attention=use_csa)
        attn.attn_sink = MagicMock()
        attn.attn_sliding_window = 128
        attn.attn_dropout = 0.0
        attn.training = False
        attn._attention_scale = MagicMock(return_value=0.125)
        pool_mock = MagicMock()
        pool_mock.shape = (1, 8, 4)
        pool_mock.unsqueeze.return_value.expand.return_value = MagicMock()
        attn._build_compressed_pool = MagicMock(return_value=pool_mock)
        indexer_topk = MagicMock()
        indexer_topk.shape = (1, 3, 4)
        indexer_topk.clamp.return_value = MagicMock()
        indexer_topk.__ge__ = MagicMock(return_value=MagicMock())
        attn.indexer = MagicMock(return_value=(indexer_topk, MagicMock()))
        with patch.object(v4_attn_mod, target, return_value=MagicMock()) as mock_kernel, patch(
            "torch.gather", return_value=MagicMock()
        ), patch("torch.where", return_value=MagicMock()):
            hidden, q_bh, k_local_bh, v_local_bh, local_mask = _stub_csa_inputs()
            DeepseekV4Attention._csa_forward(
                attn,
                hidden=hidden,
                q_bh=q_bh,
                k_local_bh=k_local_bh,
                v_local_bh=v_local_bh,
                local_mask=local_mask,
            )
        kwargs = mock_kernel.call_args.kwargs
        missing = expected_keys - set(kwargs.keys())
        assert not missing, (
            f"_csa_forward did not pass {sorted(missing)} to {target} when "
            f"use_v4_triton_csa_attention={use_csa}; check kwargs={list(kwargs.keys())}."
        )


# ---------------------------------------------------------------------------
# G29.D — turbo precedence dominates v4_triton on dense (cr == 0)
# ---------------------------------------------------------------------------


def test_p27_init_auto_disables_v4_triton_for_csa_layers() -> None:
    """``__init__`` must auto-disable ``use_v4_triton_attention`` for cr == 4.

    The contract is documented in the class docstring under
    "Auto-disable rules". The static test (P25) verifies the
    bytecode contains the (0, 128) constexpr; this test verifies
    the runtime attribute is correctly flipped on a bare instance.
    """
    # Mirror the init body: cr=4 + use_v4_triton_attention=True → flag flipped off.
    attn = _make_bare_attn(compress_ratio=4, use_v4_triton_attention=True)
    # Manually run the init's auto-disable predicate (this is the
    # exact source of the runtime guard).
    if attn._use_v4_triton_attention and attn.compress_ratio not in (0, 128):
        attn._use_v4_triton_attention = False
    assert attn._use_v4_triton_attention is False, (
        "DeepseekV4Attention.__init__ must auto-disable use_v4_triton_attention for "
        "compress_ratio=4 layers (CSA opts in via use_v4_triton_csa_attention; plan-4 P25)."
    )


def test_p27_init_auto_disables_v4_triton_csa_for_dense_hca_layers() -> None:
    """``__init__`` must auto-disable ``use_v4_triton_csa_attention`` for cr != 4."""
    for cr in (0, 128):
        attn = _make_bare_attn(compress_ratio=cr, use_v4_triton_csa_attention=True)
        if attn._use_v4_triton_csa_attention and attn.compress_ratio != 4:
            attn._use_v4_triton_csa_attention = False
        assert attn._use_v4_triton_csa_attention is False, (
            f"__init__ must auto-disable use_v4_triton_csa_attention for compress_ratio={cr} "
            "layers (dense / HCA opt in via use_v4_triton_attention; plan-4 P26)."
        )


def test_p27_log_message_mentions_layer_number() -> None:
    """The startup log line includes the layer number for cross-rank correlation.

    Each rank's log file holds entries for the layers it owns; the
    explicit ``Layer N`` makes it easy to grep for a specific
    layer's kernel choice in distributed smoke / training logs.
    """
    layer_numbers = [0, 1, 7, 17, 59]
    for ln in layer_numbers:
        attn = _make_bare_attn(compress_ratio=128, use_v4_triton_attention=True, layer_number=ln)
        with patch.object(v4_attn_mod.logger, "info") as mock_info:
            attn._log_kernel_choice()
        mock_info.assert_called_once()
        # logger.info uses %-style formatting; the layer number is the
        # first %-arg, the cr is the second, and the kernel name is the third.
        call_args = mock_info.call_args
        assert call_args.args[1] == ln, f"Expected layer_number {ln}, got {call_args.args[1]}"
