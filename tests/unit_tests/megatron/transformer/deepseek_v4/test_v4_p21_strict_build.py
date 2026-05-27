###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Plan-3 P21 — strict spec build for DeepSeek-V4.

Coverage:

* **G15 (AST audit)** — no source file under
  ``primus/backends/megatron/core/`` may pattern-match
  ``try: build_module(...) / except ...: ... return nn.Linear(...)``
  or ``try: ... / except ...: ... return <local class>(...)`` for
  the V4 ``Compressor`` / ``Indexer`` / ``AttentionSink`` slots.
  These patterns silently produced an unsharded model at TP=1 that
  would diverge at TP>1 — the canonical strict-build path raises.
* **G15a (provider helpers)** — the V4 spec provider exposes
  :meth:`column_parallel_linear_with_gather_output` and
  :meth:`row_parallel_linear_with_scatter_input` returning the
  upstream non-TE :class:`ColumnParallelLinear` /
  :class:`RowParallelLinear`.  These are the only build paths that
  accept ``gather_output=True`` / ``input_is_parallel=False``
  respectively (TE wrappers explicitly raise).
* **G15b (TP=1 vs TP=2 forward-equivalence)** — placeholder gate.
  Real TP=2 forward-equivalence requires a multi-process
  ``init_distributed`` and is exercised by the P24 smoke matrix on
  ``mi355-gpu-12``.  We keep a CPU-only TP=1 build smoke here so a
  regression in the strict path (e.g. a re-introduced fallback)
  fails locally before hitting the cluster.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable, List, Tuple

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[5]
_V4_BACKEND_ROOT = _REPO_ROOT / "primus" / "backends" / "megatron" / "core"


def _python_sources(root: Path) -> Iterable[Path]:
    """Yield every ``.py`` file under ``root`` (recursive, sorted, no dunder)."""
    for path in sorted(root.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        yield path


def _is_nn_linear_call(node: ast.AST) -> bool:
    """``True`` if ``node`` is ``nn.Linear(...)`` / ``torch.nn.Linear(...)`` /
    bare ``Linear(...)`` (i.e. a class call that produces an unsharded torch
    ``nn.Linear``)."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    # nn.Linear(...)
    if isinstance(func, ast.Attribute) and func.attr == "Linear":
        owner = func.value
        if isinstance(owner, ast.Name) and owner.id in {"nn", "torch"}:
            return True
        if (
            isinstance(owner, ast.Attribute)
            and owner.attr == "nn"
            and isinstance(owner.value, ast.Name)
            and owner.value.id == "torch"
        ):
            return True
    # bare Linear(...) where Linear was imported as `from torch.nn import Linear`
    if isinstance(func, ast.Name) and func.id == "Linear":
        return True
    return False


def _excepthandler_body_returns(handler: ast.ExceptHandler, predicate) -> bool:
    """``True`` if any ``return <call>`` in ``handler.body`` matches the
    predicate (ignoring nested defs / classes)."""
    for node in ast.walk(ast.Module(body=handler.body, type_ignores=[])):
        if isinstance(node, ast.Return) and node.value is not None and predicate(node.value):
            return True
        if isinstance(node, ast.Assign):
            for target in node.targets:
                # `self.foo = nn.Linear(...)` inside the except body still counts.
                if isinstance(target, ast.Attribute) and predicate(node.value):
                    return True
    return False


def _scan_try_returns_nn_linear(tree: ast.AST) -> List[Tuple[int, str]]:
    """Find ``try:/except: return nn.Linear(...)`` patterns and return
    their (line, snippet) for diagnostic reporting."""
    findings: List[Tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for handler in node.handlers:
                if _excepthandler_body_returns(handler, _is_nn_linear_call):
                    findings.append((handler.lineno, "except → returns nn.Linear(...)"))
    return findings


# ---------------------------------------------------------------------------
# G15 — AST audit gate
# ---------------------------------------------------------------------------


class TestG15NoNNLinearFallback:
    """No try/except/return nn.Linear() in V4 backend code."""

    def test_no_try_except_returns_nn_linear(self):
        offenders: List[str] = []
        for path in _python_sources(_V4_BACKEND_ROOT):
            try:
                tree = ast.parse(path.read_text())
            except SyntaxError:
                continue
            for line, why in _scan_try_returns_nn_linear(tree):
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{line}  {why}")
        assert not offenders, (
            "Plan-3 P21 forbids try/except → nn.Linear fallbacks under "
            "primus/backends/megatron/core/. Either fix the spec so the "
            "provider-built module instantiates cleanly (see "
            "DeepSeekV4SpecProvider.column_parallel_linear_with_gather_output "
            "/ row_parallel_linear_with_scatter_input for the canonical "
            "non-TE path), or let build_module raise.\n\nOffenders:\n  "
            + "\n  ".join(offenders)
        )

    @pytest.mark.parametrize(
        "marker",
        [
            "submodule init failed",
            "fallback to nn.Linear",
            "fallback to local Compressor",
            "using local Compressor",
            "fallback to local Indexer",
            "using local Indexer",
            "attn_sink submodule init failed",
        ],
    )
    def test_no_known_fallback_warning_strings(self, marker: str):
        """Belt-and-braces grep for the warning strings the runtime used
        to emit before P21.  These strings must not reappear in any V4
        backend source file because they only ever appeared inside the
        retired try/except fallbacks.
        """
        offenders: List[str] = []
        for path in _python_sources(_V4_BACKEND_ROOT):
            text = path.read_text()
            if marker in text:
                offenders.append(str(path.relative_to(_REPO_ROOT)))
        assert not offenders, (
            f"Plan-3 P21 retired the {marker!r} warning string; reappearing "
            f"in {offenders} means a fallback was reintroduced."
        )


# ---------------------------------------------------------------------------
# G15a — provider helper contract
# ---------------------------------------------------------------------------


class TestG15aProviderHelpers:
    """``DeepSeekV4SpecProvider`` must expose non-TE helpers for the gather /
    scatter cases that TE wrappers reject."""

    def test_column_parallel_with_gather_output_returns_non_te(self):
        from megatron.core.tensor_parallel.layers import ColumnParallelLinear

        from primus.backends.megatron.core.extensions.transformer_engine_spec_provider import (
            DeepSeekV4SpecProvider,
        )

        provider = DeepSeekV4SpecProvider()
        assert provider.column_parallel_linear_with_gather_output() is ColumnParallelLinear, (
            "provider must return upstream Megatron ColumnParallelLinear "
            "(non-TE) so callers can pass gather_output=True; the TE "
            "wrapper rejects this flag."
        )

    def test_row_parallel_with_scatter_input_returns_non_te(self):
        from megatron.core.tensor_parallel.layers import RowParallelLinear

        from primus.backends.megatron.core.extensions.transformer_engine_spec_provider import (
            DeepSeekV4SpecProvider,
        )

        provider = DeepSeekV4SpecProvider()
        assert provider.row_parallel_linear_with_scatter_input() is RowParallelLinear, (
            "provider must return upstream Megatron RowParallelLinear "
            "(non-TE) so callers can pass input_is_parallel=False; the "
            "TE wrapper rejects this flag."
        )

    def test_v4_attention_sink_provider_method_is_gone(self):
        """Plan-3 P21 retired ``provider.v4_attention_sink()`` along with
        the dead ``attn_sink_module`` build branch.  Confirm the surface
        no longer carries the method (so any caller would fail loudly
        at attribute access rather than silently rebuild a never-used
        module)."""
        from primus.backends.megatron.core.extensions.transformer_engine_spec_provider import (
            DeepSeekV4SpecProvider,
        )

        assert not hasattr(DeepSeekV4SpecProvider, "v4_attention_sink"), (
            "v4_attention_sink() must stay deleted; the canonical sink "
            "lives as nn.Parameter on DeepseekV4Attention.attn_sink."
        )


# ---------------------------------------------------------------------------
# G15a' — submodules dataclass surface
# ---------------------------------------------------------------------------


class TestSubmodulesDataclassSurface:
    """The ``DeepseekV4AttentionSubmodules`` dataclass must not expose the
    retired ``attn_sink`` slot — any spec helper that still emits one
    would break the dataclass init."""

    def test_attn_sink_field_is_gone(self):
        from dataclasses import fields

        # Import the V4 transformer config first so the package
        # ``__init__`` resolves the deepseek_v4_block ↔ deepseek_v4_attention
        # cycle before we ask for the dataclass.
        from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_transformer_config import (  # noqa: F401
            DeepSeekV4TransformerConfig,
        )
        from primus.backends.megatron.core.transformer.deepseek_v4_attention import (
            DeepseekV4AttentionSubmodules,
        )

        names = {f.name for f in fields(DeepseekV4AttentionSubmodules)}
        assert "attn_sink" not in names, (
            "DeepseekV4AttentionSubmodules.attn_sink was retired by "
            f"Plan-3 P21 (current fields: {sorted(names)})."
        )
        # Sanity: the live slots are still there.
        for required in (
            "linear_q_down_proj",
            "linear_q_up_proj",
            "linear_kv",
            "linear_o_a",
            "linear_o_b",
            "compressor",
            "indexer",
        ):
            assert required in names, f"missing required slot {required!r}"


# ---------------------------------------------------------------------------
# G15b — TP=1 build smoke (forward-equivalence vs TP=2 lives on the cluster)
# ---------------------------------------------------------------------------


class TestG15bTPOneBuildSmoke:
    """1L V4 CPU build smoke that exercises the strict spec path.

    The full TP=1 vs TP=2 forward-equivalence smoke needs a
    multi-process ``init_distributed``; that runs under the P24 smoke
    matrix on ``mi355-gpu-12``.  Here we only verify that the strict
    path *builds* a single attention layer end-to-end on CPU with TP=1,
    i.e. that none of the retired ``try/except/nn.Linear`` paths gets
    hit anymore (since the test would raise instead of silently
    falling back).
    """

    @pytest.fixture
    def _tp1_distributed(self):
        """Initialize a 1-rank torch.distributed group on gloo (CPU)."""
        import os

        import torch
        import torch.distributed as dist
        from megatron.core import parallel_state

        if dist.is_initialized():
            yield
            return

        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29509")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")

        dist.init_process_group(backend="gloo", world_size=1, rank=0)
        try:
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
            )
            yield
        finally:
            parallel_state.destroy_model_parallel()
            dist.destroy_process_group()

    def test_v4_attention_strict_build_at_tp1(self, _tp1_distributed):
        import torch

        from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_transformer_config import (
            DeepSeekV4TransformerConfig,
        )
        # Order matters: importing the V4 model package above (via the
        # config import) makes ``deepseek_v4_block`` finish loading,
        # which then unblocks the ``deepseek_v4_attention`` import below.
        from primus.backends.megatron.core.extensions.transformer_engine_spec_provider import (
            DeepSeekV4SpecProvider,
        )
        from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_layer_specs import (
            _build_v4_attention_submodules,
        )
        from primus.backends.megatron.core.transformer.deepseek_v4_attention import (
            DeepseekV4Attention,
        )
        from primus.backends.megatron.core.transformer.dual_rope import DualRoPE

        cfg = DeepSeekV4TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            ffn_hidden_size=256,
            kv_channels=32,
            q_lora_rank=64,
            o_groups=2,
            o_lora_rank=32,
            attn_sink=True,
            qk_pos_emb_head_dim=16,
            attn_sliding_window=8,
            num_query_groups=1,
            multi_latent_attention=False,
            params_dtype=torch.float32,
            init_method=lambda w: torch.nn.init.zeros_(w),
            output_layer_init_method=lambda w: torch.nn.init.zeros_(w),
            use_cpu_initialization=True,
            perform_initialization=True,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

        provider = DeepSeekV4SpecProvider(config=cfg)
        submods = _build_v4_attention_submodules(
            config=cfg,
            provider=provider,
            compress_ratio=0,
        )
        # The strict-build contract: this constructor must not raise and
        # must not drop into any nn.Linear fallback.
        attn = DeepseekV4Attention(
            config=cfg,
            rope=DualRoPE(
                rotary_dim=cfg.qk_pos_emb_head_dim,
                rope_theta=10000.0,
                compress_rope_theta=10000.0,
            ),
            compress_ratio=0,
            submodules=submods,
            layer_number=0,
        )

        # Sanity: every linear is a Megatron parallel module, not a bare nn.Linear.
        from megatron.core.tensor_parallel.layers import (
            ColumnParallelLinear,
            RowParallelLinear,
        )

        from primus.backends.megatron.core.extensions.primus_turbo import PrimusTurboLinear
        from megatron.core.extensions.transformer_engine import TELinear

        sharded_or_te = (
            ColumnParallelLinear,
            RowParallelLinear,
            TELinear,
            PrimusTurboLinear,
        )
        for name in (
            "linear_q_down_proj",
            "linear_q_up_proj",
            "linear_kv",
            "linear_o_a",
            "linear_o_b",
        ):
            mod = getattr(attn, name)
            assert isinstance(mod, sharded_or_te), (
                f"{name} resolved to {type(mod).__name__}; expected one of "
                f"{[c.__name__ for c in sharded_or_te]}.  A bare nn.Linear "
                f"means the fallback was reintroduced."
            )
        assert attn.attn_sink is not None and attn.attn_sink.shape == (cfg.num_attention_heads,)
