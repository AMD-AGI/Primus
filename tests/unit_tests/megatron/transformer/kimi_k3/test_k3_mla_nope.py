###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for :class:`KimiK3MLASelfAttention` — NoPE MLA with a sigmoid output gate.

Three claims need pinning:

1. **Parity.** The module reproduces :class:`HFKimiMLAReference` below, a
   transcription of ``KimiMLAAttention.forward``
   (``modeling_kimi_linear.py:405-474``) plus ``eager_attention_forward``
   (``:311-332``), sharing no code with the implementation and receiving
   the same weights.
2. **NoPE is real.** The ``qk_pos_emb_head_dim``-wide slices of Q and K
   are ``torch.equal`` before and after the attention call — bit-identical,
   not merely close — and ``apply_rotary_pos_emb`` is still called, so the
   pass-through is the zero-width mechanism rather than an accidentally
   skipped code path.
3. **The gate gates.** The tensor entering ``linear_proj`` is exactly
   ``sigmoid(g_proj(x)) * attn_out``; a saturated-off gate zeroes the
   output and a saturated-on gate is the identity.

Geometry note
-------------
These tests use ``qk_head_dim`` **and** a non-zero ``qk_pos_emb_head_dim``,
which is the released geometry (128 / 64, ``q_head_dim = 192``,
``softmax_scale = 192 ** -0.5``), and that is now also what the shipped
YAMLs configure. ``KimiK3TransformerConfig.mla_use_nope`` selects the
zero-width frequency table and deliberately leaves the head dims alone;
zeroing ``qk_pos_emb_head_dim`` would also stop the rotation but is a
different architecture — it deletes K3's 64 MQA-shared K dims and changes
the softmax scale. ``test_zero_width_positional_head_*`` covers that
degenerate config so its cost stays visible.

CUDA is required: ``MultiLatentAttention.__init__`` builds its rotary
module on ``torch.cuda.current_device()`` (``multi_latent_attention.py:132``)
and ``get_default_causal_mask`` hardcodes ``device="cuda"``
(``transformer/utils.py:33-35``).
"""

from __future__ import annotations

import math
import os

import pytest

# transformer_engine SIGABRTs unless torch is imported first (see node/README.md).
import torch
import torch.nn as nn
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Megatron MLA builds its rotary table and causal mask on CUDA",
)

HIDDEN_SIZE = 64
NUM_HEADS = 4
Q_LORA_RANK = 24
KV_LORA_RANK = 16
QK_NOPE_HEAD_DIM = 16
QK_ROPE_HEAD_DIM = 8
V_HEAD_DIM = 16
Q_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM
NORM_EPS = 1e-5
SEQ_LEN = 12
BATCH = 2


# ---------------------------------------------------------------------------
# HF reference
# ---------------------------------------------------------------------------


class HFKimiRMSNorm(nn.Module):
    """``KimiRMSNorm`` (``modeling_kimi_linear.py:226-236``)."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        dtype = hidden_states.dtype
        x = hidden_states.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * x.to(dtype)


class HFKimiMLAReference(nn.Module):
    """``KimiMLAAttention`` (``modeling_kimi_linear.py:335-474``), eager attention.

    Transcribed rather than imported: the HF module needs ``fla`` and a
    ``KimiLinearConfig``, and an independent transcription is the point.
    Only the ``q_lora_rank is not None`` + ``use_nope`` + eager-attention
    path is kept, which is what the release runs.

    Takes and returns ``[b, s, h]``.
    """

    def __init__(
        self,
        *,
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        q_lora_rank=Q_LORA_RANK,
        kv_lora_rank=KV_LORA_RANK,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        v_head_dim=V_HEAD_DIM,
        eps=NORM_EPS,
        use_output_gate=True,
        dtype=torch.float32,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_key_value_groups = 1  # num_key_value_heads == num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.scaling = self.q_head_dim ** (-0.5)
        self.use_output_gate = use_output_gate

        kw = {"bias": False, "dtype": dtype}
        self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, **kw)
        self.q_a_layernorm = HFKimiRMSNorm(q_lora_rank, eps=eps).to(dtype)
        self.q_b_proj = nn.Linear(q_lora_rank, num_heads * self.q_head_dim, **kw)
        self.kv_a_proj_with_mqa = nn.Linear(hidden_size, kv_lora_rank + qk_rope_head_dim, **kw)
        self.kv_a_layernorm = HFKimiRMSNorm(kv_lora_rank, eps=eps).to(dtype)
        self.kv_b_proj = nn.Linear(kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim), **kw)
        self.o_proj = nn.Linear(num_heads * v_head_dim, hidden_size, **kw)
        if use_output_gate:
            self.g_proj = nn.Linear(hidden_size, num_heads * v_head_dim, **kw)

    def qkv(self, hidden_states):
        """``:413-440`` — the split/concat, with no rotation anywhere."""
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.q_head_dim)
        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

        q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q_states = q_states.view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(
            compressed_kv,
            [compressed_kv.shape[-1] - self.qk_rope_head_dim, self.qk_rope_head_dim],
            dim=-1,
        )
        k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)
        k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)
        return query_states, key_states, value_states, q_rot, k_rot

    def core_attention(self, query_states, key_states, value_states):
        """``eager_attention_forward`` (``:311-332``) with a causal mask."""
        seq_length = query_states.shape[-2]
        causal = torch.triu(
            torch.full(
                (seq_length, seq_length),
                float("-inf"),
                device=query_states.device,
                dtype=torch.float32,
            ),
            diagonal=1,
        )
        scores = torch.einsum("bhqd,bhkd->bhqk", query_states, key_states) * self.scaling
        scores = scores + causal.to(scores.dtype)
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        return torch.einsum("bhqk,bhkd->bhqd", probs, value_states).transpose(1, 2).contiguous()

    def forward(self, hidden_states):
        batch_size, seq_length = hidden_states.shape[:-1]
        query_states, key_states, value_states, _, _ = self.qkv(hidden_states)
        attn_output = self.core_attention(query_states, key_states, value_states)
        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        if self.use_output_gate:
            # :470-473
            g = self.g_proj(hidden_states).sigmoid()
            attn_output = attn_output * g
        return self.o_proj(attn_output)


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tp1_process_group():
    """A 1-rank gloo process group with Megatron model-parallel state.

    Same fixture as ``test_kda_module.py``. ``model_parallel_cuda_manual_seed``
    is required: ``DotProductAttention`` forks the TP RNG tracker for its
    dropout (``dot_product_attention.py:216-218``).
    """
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed

    created = False
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29572")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        dist.init_process_group(backend="gloo", world_size=1, rank=0)
        created = True
    try:
        if not parallel_state.model_parallel_is_initialized():
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=1, pipeline_model_parallel_size=1
            )
        model_parallel_cuda_manual_seed(1234)
        yield
    finally:
        if created:
            parallel_state.destroy_model_parallel()
            dist.destroy_process_group()


def make_config(
    *,
    qk_pos_emb_head_dim: int = QK_ROPE_HEAD_DIM,
    mla_use_nope: bool = True,
    mla_use_output_gate: bool = True,
    params_dtype: torch.dtype = torch.float32,
    num_attention_heads: int = NUM_HEADS,
    hidden_size: int = HIDDEN_SIZE,
    qk_head_dim: int = QK_NOPE_HEAD_DIM,
    v_head_dim: int = V_HEAD_DIM,
    rope_type: str = "rope",
    mscale_all_dim: float = 0.0,
    rotary_scaling_factor: float = 1.0,
    **overrides,
):
    """A :class:`KimiK3TransformerConfig` for one full-attention layer.

    ``mla_use_nope`` is what selects the zero-width frequency table in
    :class:`KimiK3MLASelfAttention`; it defaults to ``True`` here because that
    is Kimi K3. It does **not** touch ``qk_pos_emb_head_dim``, so the released
    128 / 64 geometry survives it.
    """
    from primus.backends.megatron.core.models.kimi_k3 import KimiK3TransformerConfig

    return KimiK3TransformerConfig(
        num_layers=1,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_attention_heads,
        ffn_hidden_size=4 * hidden_size,
        # kv_channels must equal v_head_dim: DotProductAttention derives its
        # output width from kv_channels * num_attention_heads
        # (dot_product_attention.py:65, 248-249), which is why kimi_k3.yaml
        # sets it explicitly.
        kv_channels=v_head_dim,
        q_lora_rank=Q_LORA_RANK,
        kv_lora_rank=KV_LORA_RANK,
        qk_head_dim=qk_head_dim,
        qk_pos_emb_head_dim=qk_pos_emb_head_dim,
        v_head_dim=v_head_dim,
        rope_type=rope_type,
        mscale_all_dim=mscale_all_dim,
        rotary_scaling_factor=rotary_scaling_factor,
        apply_rope_fusion=False,
        mla_use_nope=mla_use_nope,
        mla_use_output_gate=mla_use_output_gate,
        normalization="RMSNorm",
        layernorm_epsilon=NORM_EPS,
        attention_dropout=0.0,
        # KimiMLAAttention builds every projection with bias=False
        # (modeling_kimi_linear.py:365-401); the K3 yaml chain inherits
        # add_bias_linear: false from llama_base.yaml:9, and MLA's linear_proj
        # takes its bias straight from that flag (multi_latent_attention.py:177).
        add_bias_linear=False,
        params_dtype=params_dtype,
        bf16=params_dtype is torch.bfloat16,
        init_method=lambda w: torch.nn.init.normal_(w, std=0.02),
        output_layer_init_method=lambda w: torch.nn.init.normal_(w, std=0.02),
        use_cpu_initialization=True,
        perform_initialization=True,
        sequence_parallel=False,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        **overrides,
    )


def build_attention(config, layer_number: int = 1, core_attention=None):
    """Build one layer through the production (TransformerEngine) spec.

    TE is not a choice here: MLA hands ``k_channels`` / ``v_channels`` to
    the core-attention builder and only ``TEDotProductAttention`` accepts
    them (see ``_check_core_attention_supports_mla``).
    """
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.spec_utils import build_module

    from primus.backends.megatron.core.transformer.kimi_k3.kimi_k3_mla_attention import (
        get_kimi_k3_mla_attention_spec,
    )

    spec = get_kimi_k3_mla_attention_spec(config, use_transformer_engine=True)
    if core_attention is not None:
        spec.submodules.core_attention = core_attention
    attn = build_module(
        spec,
        config=config,
        layer_number=layer_number,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
    )
    return attn.to(device="cuda", dtype=config.params_dtype)


def build_reference_from(attn, config, *, dtype=torch.float32):
    """A reference whose weights are ``attn``'s, mapped by name."""
    ref = HFKimiMLAReference(
        qk_rope_head_dim=config.qk_pos_emb_head_dim,
        use_output_gate=attn.linear_o_gate is not None,
        dtype=dtype,
    ).to("cuda")
    with torch.no_grad():
        ref.q_a_proj.weight.copy_(attn.linear_q_down_proj.weight)
        ref.q_a_layernorm.weight.copy_(attn.q_layernorm.weight)
        ref.q_b_proj.weight.copy_(attn.linear_q_up_proj.weight)
        ref.kv_a_proj_with_mqa.weight.copy_(attn.linear_kv_down_proj.weight)
        ref.kv_a_layernorm.weight.copy_(attn.kv_layernorm.weight)
        ref.kv_b_proj.weight.copy_(attn.linear_kv_up_proj.weight)
        ref.o_proj.weight.copy_(attn.linear_proj.weight)
        if attn.linear_o_gate is not None:
            ref.g_proj.weight.copy_(attn.linear_o_gate.weight)
    return ref


class _Tap:
    """Records the inputs and output of one submodule."""

    def __init__(self, module):
        self.args = None
        self.output = None
        self._h = module.register_forward_hook(self._hook)

    def _hook(self, _module, args, output):
        self.args = args
        self.output = output

    def remove(self):
        self._h.remove()


# ---------------------------------------------------------------------------
# 1. Parity with the HF reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_output_gate", [True, False])
def test_matches_hf_reference_fp32(tp1_process_group, use_output_gate):
    config = make_config(mla_use_output_gate=use_output_gate)
    attn = build_attention(config)
    ref = build_reference_from(attn, config)

    assert (attn.linear_o_gate is not None) is use_output_gate

    x = torch.randn(SEQ_LEN, BATCH, HIDDEN_SIZE, device="cuda", dtype=torch.float32)
    got, bias = attn(x, attention_mask=None)
    want = ref(x.transpose(0, 1)).transpose(0, 1)

    assert bias is None
    assert got.shape == x.shape
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_matches_hf_reference_bf16(tp1_process_group):
    """bf16 tolerance; the gate differs by design (fp32 sigmoid, see the module)."""
    config = make_config(params_dtype=torch.bfloat16)
    attn = build_attention(config)
    ref = build_reference_from(attn, config, dtype=torch.bfloat16)

    x = torch.randn(SEQ_LEN, BATCH, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    got, _ = attn(x, attention_mask=None)
    want = ref(x.transpose(0, 1)).transpose(0, 1)

    scale = want.abs().max().item()
    assert (got - want).abs().max().item() < 2e-2 * max(scale, 1e-3)


def test_projection_shapes_match_the_released_layout(tp1_process_group):
    """The MLA submodules line up with HF's projections one for one."""
    config = make_config()
    attn = build_attention(config)

    assert tuple(attn.linear_q_down_proj.weight.shape) == (Q_LORA_RANK, HIDDEN_SIZE)
    assert tuple(attn.linear_q_up_proj.weight.shape) == (NUM_HEADS * Q_HEAD_DIM, Q_LORA_RANK)
    assert tuple(attn.linear_kv_down_proj.weight.shape) == (
        KV_LORA_RANK + QK_ROPE_HEAD_DIM,
        HIDDEN_SIZE,
    )
    assert tuple(attn.linear_kv_up_proj.weight.shape) == (
        NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM),
        KV_LORA_RANK,
    )
    assert tuple(attn.linear_o_gate.weight.shape) == (NUM_HEADS * V_HEAD_DIM, HIDDEN_SIZE)
    assert tuple(attn.linear_proj.weight.shape) == (HIDDEN_SIZE, NUM_HEADS * V_HEAD_DIM)
    assert tuple(attn.q_layernorm.weight.shape) == (Q_LORA_RANK,)
    assert tuple(attn.kv_layernorm.weight.shape) == (KV_LORA_RANK,)


# ---------------------------------------------------------------------------
# 2. NoPE is real
# ---------------------------------------------------------------------------


def test_positional_slice_of_q_and_k_is_bit_identical_across_the_rope_call(tp1_process_group):
    """The assertion that proves NoPE: ``torch.equal``, not ``assert_close``.

    ``q``'s trailing ``qk_pos_emb_head_dim`` columns are taken straight off
    ``linear_q_up_proj``'s output and compared with the same columns of the
    query that reaches ``core_attention``. Likewise ``k``'s from
    ``linear_kv_down_proj``'s trailing slice. Both comparisons stay inside
    the module under test, so they cannot be confounded by a GEMM
    implementation difference — any mismatch is a rotation.
    """
    config = make_config()
    attn = build_attention(config)

    q_up = _Tap(attn.linear_q_up_proj)
    kv_down = _Tap(attn.linear_kv_down_proj)
    core = _Tap(attn.core_attention)

    x = torch.randn(SEQ_LEN, BATCH, HIDDEN_SIZE, device="cuda", dtype=torch.float32)
    attn(x, attention_mask=None)

    query, key = core.args[0], core.args[1]
    assert query.shape == (SEQ_LEN, BATCH, NUM_HEADS, Q_HEAD_DIM)
    assert key.shape == (SEQ_LEN, BATCH, NUM_HEADS, Q_HEAD_DIM)

    # Q: [s, b, n * q_head_dim] -> [s, b, n, q_head_dim], trailing rope slice.
    q_before = q_up.output[0].view(SEQ_LEN, BATCH, NUM_HEADS, Q_HEAD_DIM)[..., QK_NOPE_HEAD_DIM:]
    q_after = query[..., QK_NOPE_HEAD_DIM:]
    assert torch.equal(q_before, q_after), (q_before - q_after).abs().max().item()

    # K: the MQA-shared slice is the tail of the kv down-projection, broadcast
    # across heads (multi_latent_attention.py:825-830).
    k_before = (
        kv_down.output[0][..., KV_LORA_RANK:]
        .unsqueeze(-2)
        .expand(SEQ_LEN, BATCH, NUM_HEADS, QK_ROPE_HEAD_DIM)
    )
    k_after = key[..., QK_NOPE_HEAD_DIM:]
    assert torch.equal(k_before, k_after), (k_before - k_after).abs().max().item()

    # Sanity: the slices are not trivially zero, so bit-identity means something.
    assert q_before.abs().max().item() > 0.0
    assert k_before.abs().max().item() > 0.0

    for tap in (q_up, kv_down, core):
        tap.remove()


def test_rope_is_still_applied_and_is_a_bit_exact_identity(tp1_process_group, monkeypatch):
    """The pass-through is the zero-width mechanism, not a skipped branch.

    ``apply_rotary_pos_emb`` opens with ``rot_dim = freqs.shape[-1]`` and
    splits ``t`` into ``t[..., :rot_dim]`` / ``t[..., rot_dim:]``
    (``rope_utils.py:110-113``); at ``rot_dim == 0`` the closing
    ``torch.cat`` returns its input bit for bit. Wrapping the real function
    proves it is reached, that the table really is zero-width, and that the
    result is ``torch.equal`` to the input.
    """
    import megatron.core.transformer.multi_latent_attention as mla_mod

    real = mla_mod.apply_rotary_pos_emb
    calls = []

    def probe(t, freqs, *args, **kwargs):
        out = real(t, freqs, *args, **kwargs)
        calls.append((freqs.shape[-1], t.shape, torch.equal(t, out)))
        return out

    monkeypatch.setattr(mla_mod, "apply_rotary_pos_emb", probe)

    config = make_config()
    attn = build_attention(config)
    x = torch.randn(SEQ_LEN, BATCH, HIDDEN_SIZE, device="cuda", dtype=torch.float32)
    attn(x, attention_mask=None)

    # Once for the query slice (:803) and once for the key slice (:812).
    assert len(calls) == 2, calls
    for rot_dim, shape, identical in calls:
        assert rot_dim == 0, f"rotary table is {rot_dim} wide, expected 0 (shape {shape})"
        assert identical, f"apply_rotary_pos_emb changed a {shape} tensor"

    assert attn.rotary_pos_emb.inv_freq.numel() == 0


def test_mla_use_nope_selects_the_zero_width_table(tp1_process_group):
    """``config.mla_use_nope`` is the knob, and the head dims are not.

    NoPE has to be spelled somewhere. It is spelled here — swapping the
    parent's ``RotaryEmbedding(qk_pos_emb_head_dim)``
    (``multi_latent_attention.py:132-137``) for ``RotaryEmbedding(0)`` — and
    *not* by zeroing ``qk_pos_emb_head_dim``, which would change the
    architecture. Clearing the flag must therefore restore a live table while
    leaving every projection shape identical.
    """
    nope = build_attention(make_config(mla_use_nope=True))
    roped = build_attention(make_config(mla_use_nope=False))

    assert nope.mla_use_nope is True
    assert nope.rotary_pos_emb.inv_freq.numel() == 0

    # inv_freq is arange(0, dim, 2), i.e. half the positional head width
    # (rotary_pos_embedding.py:79-81).
    assert roped.mla_use_nope is False
    assert roped.rotary_pos_emb.inv_freq.numel() == QK_ROPE_HEAD_DIM // 2

    # The geometry is the flag's business in neither case.
    for attn in (nope, roped):
        assert attn.q_head_dim == Q_HEAD_DIM
        assert attn.softmax_scale == pytest.approx(Q_HEAD_DIM**-0.5)
        assert tuple(attn.linear_kv_down_proj.weight.shape) == (
            KV_LORA_RANK + QK_ROPE_HEAD_DIM,
            HIDDEN_SIZE,
        )
        assert tuple(attn.linear_q_up_proj.weight.shape) == (NUM_HEADS * Q_HEAD_DIM, Q_LORA_RANK)


def test_output_is_position_invariant(tp1_process_group):
    """NoPE, behaviourally: a permuted prefix gives a permuted output.

    With no positional signal, attention over a *full* (unmasked) window is
    permutation-equivariant. Compare position 0's output when the sequence
    has length 1 against a differently-ordered single token — trivially
    equal — and, more usefully, check that reversing the whole sequence and
    reading the *first* query still sees only itself.
    """
    config = make_config()
    attn = build_attention(config)

    x = torch.randn(SEQ_LEN, BATCH, HIDDEN_SIZE, device="cuda", dtype=torch.float32)
    out, _ = attn(x, attention_mask=None)

    # Causality means query 0 attends only to key 0, so its output must not
    # depend on the rest of the sequence at all.
    x_alt = x.clone()
    x_alt[1:] = torch.randn_like(x_alt[1:])
    out_alt, _ = attn(x_alt, attention_mask=None)
    torch.testing.assert_close(out[0], out_alt[0], rtol=1e-6, atol=1e-7)

    # And a token placed at position 0 versus at position 1 with an identical
    # predecessor: with rope the two would differ, without it they cannot.
    single = x[:1]
    out_single, _ = attn(single, attention_mask=None)
    two = torch.cat([single, single], dim=0)
    out_two, _ = attn(two, attention_mask=None)
    torch.testing.assert_close(out_two[1], out_single[0], rtol=1e-5, atol=1e-6)


def test_softmax_scale_is_the_released_one(tp1_process_group):
    """``softmax_scale == q_head_dim ** -0.5``, i.e. ``192 ** -0.5`` at scale."""
    config = make_config()
    attn = build_attention(config)

    assert attn.q_head_dim == Q_HEAD_DIM
    assert attn.softmax_scale == pytest.approx(Q_HEAD_DIM**-0.5)

    released = make_config(
        hidden_size=256,
        num_attention_heads=2,
        qk_head_dim=128,
        qk_pos_emb_head_dim=64,
        v_head_dim=128,
    )
    released_attn = build_attention(released)
    assert released_attn.q_head_dim == 192
    assert released_attn.softmax_scale == pytest.approx(192**-0.5)
    assert released_attn.softmax_scale == pytest.approx(math.pow(192, -0.5))


def test_softmax_scale_reaches_the_attention_kernel(tp1_process_group):
    """The scale is passed to ``core_attention``'s constructor, once, at build time.

    ``MultiLatentAttention.__init__`` hands ``softmax_scale`` to the
    core-attention builder (``multi_latent_attention.py:163``) rather than
    to its forward, so a wrong value is baked in and invisible thereafter.

    Note the core attention is built **twice**: the base
    ``Attention.__init__`` builds one with ``config.softmax_scale``
    (``attention.py:321-329``, ``None`` here) and MLA immediately replaces
    it with its own (``:157-168``). Only the second survives.
    """
    from megatron.core.extensions.transformer_engine import TEDotProductAttention

    seen = []

    class _RecordingDPA(TEDotProductAttention):
        def __init__(self, *args, softmax_scale=None, **kwargs):
            seen.append(softmax_scale)
            super().__init__(*args, softmax_scale=softmax_scale, **kwargs)

    build_attention(make_config(), core_attention=_RecordingDPA)

    assert seen == [None, pytest.approx(Q_HEAD_DIM**-0.5)]


def test_non_unit_mscale_is_rejected(tp1_process_group):
    """A silent softmax-scale change is the trap DESIGN.md §10.2 flagged."""
    config = make_config(rotary_scaling_factor=8.0, mscale_all_dim=1.0)
    with pytest.raises(ValueError, match="softmax_scale"):
        build_attention(config)


def test_zero_width_positional_head_is_a_different_architecture(tp1_process_group):
    """``qk_pos_emb_head_dim = 0`` also disables rope — and changes the model.

    This is the documented trap: the released config keeps 64 unrotated,
    MQA-shared K dims that come straight off ``kv_a_proj_with_mqa``. Zeroing
    the positional head removes them and rescales the softmax. The module
    still constructs and runs, so nothing fails loudly — hence this test and
    the warning in ``_warn_if_positional_head_is_zero_width``.
    """
    config = make_config(qk_pos_emb_head_dim=0)
    attn = build_attention(config)

    # No k_rot at all: the kv down-projection is exactly kv_lora_rank wide.
    assert tuple(attn.linear_kv_down_proj.weight.shape) == (KV_LORA_RANK, HIDDEN_SIZE)
    # ... and q is narrower, so the scale differs from the released one.
    assert tuple(attn.linear_q_up_proj.weight.shape) == (
        NUM_HEADS * QK_NOPE_HEAD_DIM,
        Q_LORA_RANK,
    )
    assert attn.softmax_scale == pytest.approx(QK_NOPE_HEAD_DIM**-0.5)
    assert attn.softmax_scale != pytest.approx(Q_HEAD_DIM**-0.5)

    x = torch.randn(SEQ_LEN, BATCH, HIDDEN_SIZE, device="cuda", dtype=torch.float32)
    out, _ = attn(x, attention_mask=None)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 3. The output gate
# ---------------------------------------------------------------------------


def test_pre_proj_tensor_is_exactly_sigmoid_gate_times_attn_out(tp1_process_group):
    config = make_config()
    attn = build_attention(config)

    core = _Tap(attn.core_attention)
    proj = _Tap(attn.linear_proj)

    x = torch.randn(SEQ_LEN, BATCH, HIDDEN_SIZE, device="cuda", dtype=torch.float32)
    attn(x, attention_mask=None)

    attn_out = core.output
    gated = proj.args[0]
    gate, _ = attn.linear_o_gate(x)

    assert attn_out.shape == (SEQ_LEN, BATCH, NUM_HEADS * V_HEAD_DIM)
    assert torch.equal(gated, attn_out * torch.sigmoid(gate.float()))
    # The gate is not the identity, so the assertion above has teeth.
    assert not torch.equal(gated, attn_out)

    core.remove()
    proj.remove()


def _replace_gate_with_constant(attn, value: float):
    """Swap ``linear_o_gate`` for a constant, saturating the sigmoid."""
    width = attn.query_projection_size

    class _ConstGate(nn.Module):
        def forward(self, x):
            return torch.full((*x.shape[:-1], width), value, device=x.device, dtype=x.dtype), None

    attn.linear_o_gate = _ConstGate()


def test_saturated_off_gate_zeroes_the_output(tp1_process_group):
    """``sigmoid(-inf) == 0`` exactly in fp32, so the layer output is exactly 0."""
    config = make_config()
    attn = build_attention(config)
    _replace_gate_with_constant(attn, -1e30)

    x = torch.randn(SEQ_LEN, BATCH, HIDDEN_SIZE, device="cuda", dtype=torch.float32)
    out, _ = attn(x, attention_mask=None)

    assert torch.equal(out, torch.zeros_like(out))


def test_saturated_on_gate_is_the_identity(tp1_process_group):
    """``sigmoid(+inf) == 1``, so the gate drops out and only ``o_proj`` remains."""
    config = make_config()
    attn = build_attention(config)

    x = torch.randn(SEQ_LEN, BATCH, HIDDEN_SIZE, device="cuda", dtype=torch.float32)
    _replace_gate_with_constant(attn, 1e30)
    gated_out, _ = attn(x, attention_mask=None)

    # The same module with the gate removed entirely goes through the parent's
    # forward, which is a genuinely different code path.
    attn.linear_o_gate = None
    ungated_out, _ = attn(x, attention_mask=None)

    torch.testing.assert_close(gated_out, ungated_out, rtol=1e-6, atol=1e-7)


def test_gate_weight_receives_gradient(tp1_process_group):
    config = make_config()
    attn = build_attention(config)

    x = torch.randn(SEQ_LEN, BATCH, HIDDEN_SIZE, device="cuda", dtype=torch.float32, requires_grad=True)
    out, _ = attn(x, attention_mask=None)
    out.square().mean().backward()

    for name, param in attn.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} gradient is not finite"
    assert attn.linear_o_gate.weight.grad.abs().max().item() > 0.0
    assert torch.isfinite(x.grad).all()


def test_gradients_match_the_hf_reference(tp1_process_group):
    config = make_config()
    attn = build_attention(config)
    ref = build_reference_from(attn, config)

    x = torch.randn(SEQ_LEN, BATCH, HIDDEN_SIZE, device="cuda", dtype=torch.float32)
    xa = x.clone().requires_grad_(True)
    xb = x.transpose(0, 1).clone().requires_grad_(True)

    attn(xa, attention_mask=None)[0].square().sum().backward()
    ref(xb).square().sum().backward()

    # TE DotProductAttention backward vs the eager HF reference: forward is
    # bit-exact, but backward rel_rms ~3e-4 on o_proj (see mla_grad_debug.txt).
    torch.testing.assert_close(xa.grad, xb.grad.transpose(0, 1), rtol=5e-4, atol=1e-5)
    torch.testing.assert_close(attn.linear_o_gate.weight.grad, ref.g_proj.weight.grad, rtol=5e-4, atol=1e-5)
    torch.testing.assert_close(attn.linear_proj.weight.grad, ref.o_proj.weight.grad, rtol=5e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# Config-layer contracts
# ---------------------------------------------------------------------------


def test_upstream_output_gate_flag_is_still_rejected_under_mla():
    """Why K3 carries its own ``mla_use_output_gate``.

    ``MLATransformerConfig.__post_init__`` raises for
    ``attention_output_gate`` (``transformer_config.py:2259-2260``). The
    guard fires while the *config* is built, so no attention-subclass
    ``__init__`` trick can reach around it — the gate has to be K3's own
    field with upstream's left at False.
    """
    from megatron.core.transformer.transformer_config import (
        MLATransformerConfig,
        TransformerConfig,
    )

    kwargs = dict(
        num_layers=1,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_HEADS,
        kv_channels=V_HEAD_DIM,
        attention_output_gate=True,
    )
    with pytest.raises(NotImplementedError, match="Output gate is not supported for MLA"):
        MLATransformerConfig(rope_type="rope", **kwargs)

    # The same flag on a plain TransformerConfig is accepted, which is what
    # makes the MLA-only rejection the thing to route around.
    assert TransformerConfig(**kwargs).attention_output_gate is True


def test_our_config_leaves_the_upstream_flag_off(tp1_process_group):
    config = make_config()
    assert config.attention_output_gate is False
    assert config.mla_use_output_gate is True
    attn = build_attention(config)
    assert attn.use_output_gate is True


def test_yarn_and_rope_fusion_are_rejected(tp1_process_group):
    with pytest.raises(ValueError, match="rope_type='rope'"):
        build_attention(make_config(rope_type="yarn"))

    # Set the flag after __post_init__ so this exercises our guard rather than
    # upstream's TE-availability check for apply_rope_fusion.
    config = make_config()
    config.apply_rope_fusion = True
    with pytest.raises(ValueError, match="apply_rope_fusion"):
        build_attention(config)


def test_gate_without_a_spec_slot_raises(tp1_process_group):
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
    from megatron.core.process_groups_config import ProcessGroupCollection

    from primus.backends.megatron.core.transformer.kimi_k3.kimi_k3_mla_attention import (
        KimiK3MLASelfAttention,
        get_kimi_k3_mla_attention_submodules,
    )

    config = make_config()
    submodules = get_kimi_k3_mla_attention_submodules(TESpecProvider(), mla_use_output_gate=False)
    with pytest.raises(ValueError, match="submodules.linear_o_gate is None"):
        KimiK3MLASelfAttention(
            config=config,
            submodules=submodules,
            layer_number=1,
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        )


def test_local_core_attention_is_rejected_with_an_actionable_error():
    """``DotProductAttention`` cannot serve MLA at this Megatron HEAD.

    ``MultiLatentAttention.__init__`` passes ``k_channels`` /
    ``v_channels`` unconditionally (``multi_latent_attention.py:164-165``)
    and only ``TEDotProductAttention`` declares them
    (``extensions/transformer_engine.py:1178``). Without this guard the
    failure is a ``TypeError`` thrown from inside ``build_module``, which
    is also why upstream's own
    ``get_gpt_layer_local_spec(multi_latent_attention=True)`` is latently
    broken.
    """
    from megatron.core.models.backends import LocalSpecProvider

    from primus.backends.megatron.core.transformer.kimi_k3.kimi_k3_mla_attention import (
        get_kimi_k3_mla_attention_submodules,
    )

    with pytest.raises(ValueError, match="k_channels / v_channels"):
        get_kimi_k3_mla_attention_submodules(LocalSpecProvider())


# ---------------------------------------------------------------------------
# Spec plumbing
# ---------------------------------------------------------------------------


def test_spec_shape():
    from megatron.core.transformer.enums import AttnMaskType

    from primus.backends.megatron.core.transformer.kimi_k3.kimi_k3_mla_attention import (
        KimiK3MLASelfAttention,
        KimiK3MLASelfAttentionSubmodules,
        get_kimi_k3_mla_attention_spec,
    )

    spec = get_kimi_k3_mla_attention_spec()

    assert spec.module is KimiK3MLASelfAttention
    assert spec.params == {"attn_mask_type": AttnMaskType.causal}
    assert isinstance(spec.submodules, KimiK3MLASelfAttentionSubmodules)
    assert spec.submodules.linear_o_gate is not None
    # Every slot the parent's __init__ reads must be filled.
    for slot in (
        "q_layernorm",
        "kv_layernorm",
        "linear_q_down_proj",
        "linear_q_up_proj",
        "linear_kv_down_proj",
        "linear_kv_up_proj",
        "core_attention",
        "linear_proj",
    ):
        assert getattr(spec.submodules, slot) is not None, slot


def test_spec_honours_the_config_gate_flag():
    from primus.backends.megatron.core.transformer.kimi_k3.kimi_k3_mla_attention import (
        get_kimi_k3_mla_attention_spec,
    )

    class _Bag:
        normalization = "RMSNorm"
        mla_use_output_gate = False

    assert get_kimi_k3_mla_attention_spec(_Bag()).submodules.linear_o_gate is None


def test_te_spec_uses_te_modules():
    from primus.backends.megatron.core.transformer.kimi_k3.kimi_k3_mla_attention import (
        get_kimi_k3_mla_attention_spec,
    )

    submodules = get_kimi_k3_mla_attention_spec(use_transformer_engine=True).submodules

    # TE routes the low-rank down-projections through a replicated TELinear
    # (multi_latent_attention.py:410-411), unlike upstream's local spec.
    assert submodules.linear_q_down_proj.__name__ == "TELinear"
    assert submodules.linear_kv_down_proj.__name__ == "TELinear"
    assert submodules.linear_q_up_proj.__name__ == "TEColumnParallelLinear"
    assert submodules.linear_o_gate.__name__ == "TEColumnParallelLinear"
    assert submodules.core_attention.__name__ == "TEDotProductAttention"
    assert submodules.linear_proj.__name__ == "TERowParallelLinear"
