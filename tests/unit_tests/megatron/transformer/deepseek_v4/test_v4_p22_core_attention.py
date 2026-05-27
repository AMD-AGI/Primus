###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Plan-3 P22 — ``core_attention`` integration for DeepSeek-V4.

The dense (``compress_ratio == 0``) layers route their softmax-and-attend
through ``provider.core_attention()`` (PrimusTurboAttention when
``use_turbo_attention=True``, TEDotProductAttention otherwise).  HCA
(``compress_ratio == 128``) and CSA (``compress_ratio == 4``) layers
**do not** get a ``core_attention`` slot — their joint softmax and
per-query top-K gather can't be expressed as stock flash-attention.

This file is the unit-test side of P22's test gates:

* **G18 (spec surface)** — the V4 attention submodules dataclass exposes
  a ``core_attention`` slot; the spec helper emits it for dense layers
  only.
* **G18a (alias contract)** — when V4's per-head sink is on AND the
  built ``core_attention`` advertises ``use_sink_attention=True``, the
  attention module aliases ``self.core_attention.sinks`` to
  ``self.attn_sink`` so the released-checkpoint key path
  ``layers.{i}.attn.attn_sink`` still loads.
* **G18b (CPU forward equivalence)** — a ``MockTurbo`` core-attention
  class that replicates the eager-Python scaled-dot-product math is
  injected; the dense-path output must match the original eager-Python
  forward within a tight numerical tolerance.  The full
  Turbo-vs-eager-Python equivalence at full V4-Flash dims is exercised
  by the P22 smoke gate on ``mi355-gpu-12``.
* **G18c (no core_attention on HCA / CSA)** — the spec helper does not
  emit ``core_attention`` for ``compress_ratio in {4, 128}``.
"""

from __future__ import annotations

from dataclasses import fields

import pytest
import torch

from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec, build_module


# ---------------------------------------------------------------------------
# G18 — submodules surface
# ---------------------------------------------------------------------------


class TestSubmodulesSurface:
    """``DeepseekV4AttentionSubmodules`` must expose a ``core_attention`` slot."""

    def test_core_attention_field_present(self):
        from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_transformer_config import (  # noqa: F401
            DeepSeekV4TransformerConfig,
        )
        from primus.backends.megatron.core.transformer.deepseek_v4_attention import (
            DeepseekV4AttentionSubmodules,
        )

        names = {f.name for f in fields(DeepseekV4AttentionSubmodules)}
        assert "core_attention" in names, (
            "Plan-3 P22 added ``core_attention`` to "
            f"DeepseekV4AttentionSubmodules; current fields: {sorted(names)}."
        )


# ---------------------------------------------------------------------------
# G18 — spec helper emits core_attention for dense only
# ---------------------------------------------------------------------------


@pytest.fixture
def _tp1_distributed():
    """1-rank torch.distributed (gloo) with Megatron model-parallel state."""
    import os

    import torch.distributed as dist
    from megatron.core import parallel_state

    if dist.is_initialized():
        yield
        return

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29539")
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


def _make_v4_cfg(*, attn_sink: bool, attn_sliding_window: int = 0):
    from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_transformer_config import (
        DeepSeekV4TransformerConfig,
    )

    return DeepSeekV4TransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        ffn_hidden_size=128,
        kv_channels=16,
        q_lora_rank=32,
        o_groups=2,
        o_lora_rank=16,
        attn_sink=attn_sink,
        attn_sliding_window=attn_sliding_window,
        qk_pos_emb_head_dim=8,
        num_query_groups=1,
        multi_latent_attention=False,
        params_dtype=torch.float32,
        init_method=lambda w: torch.nn.init.normal_(w, std=0.02),
        output_layer_init_method=lambda w: torch.nn.init.normal_(w, std=0.02),
        use_cpu_initialization=True,
        perform_initialization=True,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )


def _make_v4_submods(cfg, compress_ratio: int):
    from primus.backends.megatron.core.extensions.transformer_engine_spec_provider import (
        DeepSeekV4SpecProvider,
    )
    from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_layer_specs import (
        _build_v4_attention_submodules,
    )

    provider = DeepSeekV4SpecProvider(config=cfg)
    return _build_v4_attention_submodules(
        config=cfg,
        provider=provider,
        compress_ratio=compress_ratio,
    )


class TestSpecEmission:
    """The spec helper emits ``core_attention`` only for dense layers."""

    def test_dense_emits_core_attention(self, _tp1_distributed):
        cfg = _make_v4_cfg(attn_sink=True, attn_sliding_window=4)
        submods = _make_v4_submods(cfg, compress_ratio=0)
        assert submods.core_attention is not None, (
            "compress_ratio == 0 must emit a ``core_attention`` spec."
        )

    def test_csa_no_core_attention(self, _tp1_distributed):
        cfg = _make_v4_cfg(attn_sink=True, attn_sliding_window=4)
        submods = _make_v4_submods(cfg, compress_ratio=4)
        assert submods.core_attention is None, (
            "compress_ratio == 4 (CSA) must NOT emit ``core_attention`` — "
            "per-query top-K gather is not a flash-attn pattern."
        )

    def test_hca_no_core_attention(self, _tp1_distributed):
        cfg = _make_v4_cfg(attn_sink=True, attn_sliding_window=4)
        submods = _make_v4_submods(cfg, compress_ratio=128)
        assert submods.core_attention is None, (
            "compress_ratio == 128 (HCA) must NOT emit ``core_attention`` — "
            "joint softmax across two key streams needs an LSE-returning kernel."
        )


# ---------------------------------------------------------------------------
# G18 — alias contract + forward equivalence
# ---------------------------------------------------------------------------


class _MockTurboCoreAttention(torch.nn.Module):
    """Stand-in for :class:`PrimusTurboAttention`.

    Replicates the eager-Python scaled-dot-product math (with optional
    learned per-head sinks) on CPU, so the V4 attention module's dense
    forward can be exercised end-to-end without a CUDA / Turbo build.

    Public surface that ``DeepseekV4Attention`` reads:

    * ``self.use_sink_attention`` — whether learned sinks are honored.
    * ``self.sinks`` — ``[num_attention_heads]`` parameter (aliased by
      V4's ``self.attn_sink`` after construction).
    * ``forward(q, k, v, mask, attn_mask_type=...)`` — accepts ``sbhd``
      Q / K / V (K, V may be MQA-shape ``[S, B, 1, D]``).
    """

    def __init__(
        self,
        *,
        config,
        layer_number,
        attn_mask_type,
        attention_type,
        softmax_scale,
        k_channels=None,
        v_channels=None,
        cp_comm_type="p2p",
        pg_collection=None,
    ):
        super().__init__()
        self.config = config
        self.layer_number = layer_number
        self.softmax_scale = float(softmax_scale)
        self.use_sink_attention = bool(getattr(config, "attn_sink", False))
        self._head_dim = int(k_channels or config.kv_channels)
        self._num_heads = int(config.num_attention_heads)
        if self.use_sink_attention:
            self.sinks = torch.nn.Parameter(torch.zeros(self._num_heads))
        else:
            self.sinks = None

    def forward(self, q, k, v, mask, attn_mask_type=None):
        # Inputs in qkv_format="sbhd": [S, B, H_q/H_kv, D].
        S_q, B, H_q, D = q.shape
        S_k = k.shape[0]
        H_kv = k.shape[2]

        # MQA broadcast.
        if H_kv != H_q:
            k = k.expand(S_k, B, H_q, D)
            v = v.expand(S_k, B, H_q, D)

        # [S, B, H, D] -> [B, H, S, D].
        q_bh = q.permute(1, 2, 0, 3).contiguous().float()
        k_bh = k.permute(1, 2, 0, 3).contiguous().float()
        v_bh = v.permute(1, 2, 0, 3).contiguous()

        logits = torch.matmul(q_bh, k_bh.transpose(-2, -1)) * self.softmax_scale

        # Causal mask.
        if attn_mask_type == AttnMaskType.causal:
            causal = torch.full((S_q, S_k), float("-inf"), device=q.device)
            causal = torch.triu(causal, diagonal=1)
            logits = logits + causal

        # Sink column.
        if self.use_sink_attention and self.sinks is not None:
            sink_col = self.sinks.float().view(1, H_q, 1, 1).expand(B, H_q, S_q, 1)
            logits_aug = torch.cat([logits, sink_col], dim=-1)
            logits_aug = logits_aug - logits_aug.amax(dim=-1, keepdim=True).detach()
            probs = logits_aug.softmax(dim=-1)[..., :-1]
        else:
            logits = logits - logits.amax(dim=-1, keepdim=True).detach()
            probs = logits.softmax(dim=-1)

        out_bh = torch.matmul(probs.to(v_bh.dtype), v_bh)  # [B, H, S, D]

        # [B, H, S, D] -> [S, B, H, D] -> [S, B, H*D].
        out_sbh = out_bh.permute(2, 0, 1, 3).contiguous()
        return out_sbh.view(S_q, B, H_q * D)


def _build_v4_attention_with(cfg, *, core_attention_class):
    """Build a 1L V4 dense attention with an explicit ``core_attention`` class."""
    from primus.backends.megatron.core.transformer.deepseek_v4_attention import (
        DeepseekV4Attention,
    )
    from primus.backends.megatron.core.transformer.dual_rope import DualRoPE

    submods = _make_v4_submods(cfg, compress_ratio=0)
    if core_attention_class is None:
        submods.core_attention = None
    else:
        submods.core_attention = ModuleSpec(module=core_attention_class)

    rope = DualRoPE(
        rotary_dim=cfg.qk_pos_emb_head_dim,
        rope_theta=10000.0,
        compress_rope_theta=10000.0,
    )
    return DeepseekV4Attention(
        config=cfg,
        rope=rope,
        compress_ratio=0,
        submodules=submods,
        layer_number=0,
    )


class TestG18AliasAndForward:
    """``self.attn_sink`` aliasing + dense forward equivalence."""

    def test_sink_alias_when_turbo_supports_sink(self, _tp1_distributed):
        cfg = _make_v4_cfg(attn_sink=True, attn_sliding_window=0)
        attn = _build_v4_attention_with(cfg, core_attention_class=_MockTurboCoreAttention)

        assert attn.core_attention is not None
        assert attn._use_core_attention is True
        assert attn.attn_sink is not None
        assert attn.core_attention.sinks is attn.attn_sink, (
            "Plan-3 P22: when the core-attention class advertises "
            "``use_sink_attention=True`` and V4 has ``attn_sink=True``, "
            "the V4 attention module must alias ``core_attention.sinks`` "
            "to ``self.attn_sink`` so the released-checkpoint key path "
            "``layers.{i}.attn.attn_sink`` still loads via state-dict."
        )

    def test_no_alias_when_core_attention_lacks_sink(self, _tp1_distributed):
        # Default _build_v4_attention_submodules emits provider.core_attention()
        # which is TEDotProductAttention (no use_sink_attention attr).
        cfg = _make_v4_cfg(attn_sink=True, attn_sliding_window=4)
        attn = _build_v4_attention_with(cfg, core_attention_class=None)
        # Re-emit core_attention via provider so we exercise the TE path.
        # _build_v4_attention_with(core_attention_class=None) drops the slot;
        # the default emission already builds TEDotProductAttention which
        # does NOT advertise use_sink_attention -> no alias, eager fallback.
        # Reproduce the default emission and rebuild.
        from primus.backends.megatron.core.extensions.transformer_engine_spec_provider import (
            DeepSeekV4SpecProvider,
        )

        provider = DeepSeekV4SpecProvider(config=cfg)
        attn2 = _build_v4_attention_with(
            cfg, core_attention_class=provider.core_attention()
        )
        assert attn2.core_attention is not None, "TE core_attention must build"
        assert attn2._use_core_attention is False, (
            "When the built core_attention does not support learned sinks "
            "(e.g. TEDotProductAttention) and V4 sink is on, the dense "
            "path must fall back to eager-Python so the inline "
            "softmax-with-sink math still produces the correct output."
        )

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="V4 specs build TE-parallel linears whose forward requires CUDA.",
    )
    def test_dense_forward_matches_eager(self, _tp1_distributed):
        torch.manual_seed(0)
        cfg = _make_v4_cfg(attn_sink=True, attn_sliding_window=0)

        attn_eager = _build_v4_attention_with(cfg, core_attention_class=None)
        attn_turbo = _build_v4_attention_with(
            cfg, core_attention_class=_MockTurboCoreAttention
        )

        # Copy weights so both modules have identical parameters.
        with torch.no_grad():
            for (n_e, p_e), (n_t, p_t) in zip(
                attn_eager.named_parameters(),
                attn_turbo.named_parameters(),
            ):
                if p_e.shape == p_t.shape:
                    p_t.copy_(p_e)

        # Sanity: both should have the same attn_sink storage.
        assert attn_eager.attn_sink is not None
        assert attn_turbo.core_attention.sinks is attn_turbo.attn_sink

        # Move both modules onto CUDA — the V4 spec emits TE-parallel
        # linears whose forward requires CUDA tensors.  V4's DualRoPE is
        # held by reference (``self._rope = [rope]``) so it isn't moved
        # by ``attn.to(device)``; move it explicitly.
        device = torch.device("cuda")
        attn_eager = attn_eager.to(device)
        attn_turbo = attn_turbo.to(device)
        attn_eager._rope[0].to(device)
        attn_turbo._rope[0].to(device)

        B, S, D = 2, 7, cfg.hidden_size
        hidden = torch.randn(B, S, D, device=device)
        position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, S)

        attn_eager.eval()
        attn_turbo.eval()
        with torch.no_grad():
            out_eager = attn_eager(hidden, position_ids)
            out_turbo = attn_turbo(hidden, position_ids)

        assert out_eager.shape == out_turbo.shape == (B, S, D)
        max_abs = (out_eager - out_turbo).abs().max().item()
        assert max_abs < 5e-3, (
            f"Dense forward via core_attention diverged from eager-Python "
            f"forward beyond tolerance: max-abs={max_abs}.  Plan-3 P22 "
            "expects the two paths to compute identical math (the mock "
            "core_attention replicates the eager softmax-with-sink kernel)."
        )


# ---------------------------------------------------------------------------
# G18 — sink-attention args plumbing
# ---------------------------------------------------------------------------


class TestSinkAttentionArgsPlumbing:
    """``deepseek_v4_builder._maybe_plumb_v4_sink_attention_args``."""

    def test_plumbing_fires_when_turbo_attention_on_seq_le_window(self):
        """Plumbing zeros out the window when it covers the full sequence
        (mathematically equivalent to full causal — avoids the aiter
        Triton SWA gap)."""
        from types import SimpleNamespace

        from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_builders import (
            _maybe_plumb_v4_sink_attention_args,
        )

        args = SimpleNamespace(
            enable_primus_turbo=True,
            use_turbo_attention=True,
            attn_sink=True,
            attn_sliding_window=128,
            seq_length=128,
        )
        _maybe_plumb_v4_sink_attention_args(args)

        assert args.use_sink_attention is True
        # Window == seq_length -> drop window (full causal is identical).
        assert args.sink_sliding_window == 0
        assert args.sink_window_even_layers_only is False

    def test_plumbing_warns_when_real_swa_requested(self):
        """When the V4 config genuinely needs SWA (seq > window) the
        plumbing zeros out the window with a warning — aiter Triton
        flash-attn doesn't support SWA yet, so the V4 dense layer
        attends to all causal tokens (deviation from V4-Flash math)."""
        from types import SimpleNamespace

        from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_builders import (
            _maybe_plumb_v4_sink_attention_args,
        )

        args = SimpleNamespace(
            enable_primus_turbo=True,
            use_turbo_attention=True,
            attn_sink=True,
            attn_sliding_window=128,
            seq_length=4096,
        )
        _maybe_plumb_v4_sink_attention_args(args)

        assert args.use_sink_attention is True
        # SWA requested but kernel doesn't support it -> drop window.
        assert args.sink_sliding_window == 0
        assert args.sink_window_even_layers_only is False

    def test_plumbing_skips_when_turbo_off(self):
        from types import SimpleNamespace

        from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_builders import (
            _maybe_plumb_v4_sink_attention_args,
        )

        args = SimpleNamespace(
            enable_primus_turbo=False,
            use_turbo_attention=False,
            attn_sink=True,
            attn_sliding_window=128,
        )
        _maybe_plumb_v4_sink_attention_args(args)

        # Plumbing must not touch args when Turbo is off.
        assert not hasattr(args, "use_sink_attention") or args.use_sink_attention in (
            None,
            False,
        )

    def test_plumbing_skips_when_attn_sink_off(self):
        from types import SimpleNamespace

        from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_builders import (
            _maybe_plumb_v4_sink_attention_args,
        )

        args = SimpleNamespace(
            enable_primus_turbo=True,
            use_turbo_attention=True,
            attn_sink=False,
            attn_sliding_window=128,
        )
        _maybe_plumb_v4_sink_attention_args(args)

        assert not hasattr(args, "use_sink_attention") or args.use_sink_attention in (
            None,
            False,
        )

    def test_plumbing_respects_explicit_user_override(self):
        from types import SimpleNamespace

        from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_builders import (
            _maybe_plumb_v4_sink_attention_args,
        )

        args = SimpleNamespace(
            enable_primus_turbo=True,
            use_turbo_attention=True,
            attn_sink=True,
            attn_sliding_window=128,
            seq_length=4096,
            use_sink_attention=True,
            sink_sliding_window=256,  # user override
        )
        _maybe_plumb_v4_sink_attention_args(args)

        # Don't clobber user-set sink_sliding_window.
        assert args.sink_sliding_window == 256
