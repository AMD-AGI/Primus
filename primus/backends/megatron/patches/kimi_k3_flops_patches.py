###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Kimi K3 FLOPs reporting patch.

Megatron's :func:`megatron.training.training.num_floating_point_operations`
gets Kimi K3 wrong on **five** independent axes. All five were read off the
smoke run's own argument dump rather than assumed:

* **Dispatch.** ``is_hybrid_model(args)`` is ``args.hybrid_layer_pattern is
  not None`` (``training/utils.py:434-436``) and K3 leaves that ``None``
  (log ``:1576``), so K3 lands in ``transformer_flops()``.
* **Attention branch.** ``args.multi_latent_attention`` must stay ``False``
  for K3 (``arguments.py:1589-1590`` would otherwise replace the config
  class), so
  ``transformer_flops`` takes the **MHA/GQA** branch (``training.py:429``)
  and models every layer as a dense ``h -> 3h`` QKV projection.
* **KDA is charged as quadratic attention.** ``args.experimental_attention_variant``
  is ``None`` (log ``:1508``) — correctly, because K3 builds its own spec
  tree — so ``num_linear_attention_layers = 0`` (``training.py:521``) and all
  ``num_layers`` layers pay ``query_projection_size * seq_length / 2 * 2``.
  KDA's cost is **linear in T**. Upstream's ``gated_delta_net`` branch
  (``:488-514``) is both unreachable here and shaped wrong for KDA: it has
  no ``g_proj`` term and models the recurrence as
  ``4 * num_v_heads * v_head_dim**2`` with no chunk-size dependence.
* **The latent MoE bottleneck is invisible.** ``training.py:361`` reads
  ``args.moe_latent_size``, which is ``None`` (log ``:1760``) while
  ``routed_expert_hidden_size`` is set (log ``:1996``).
  :func:`patch_k3_args_moe_latent_size` in this module closes that at the
  args layer; the closed form below does not depend on it.
* **Two K3 modules are not modelled at all**: the MLA sigmoid output gate
  (upstream's ``args.attention_output_gate`` stays ``False`` because
  ``MLATransformerConfig.__post_init__`` raises on it) and the
  attention-residual mixers.

Honest calibration note. At the 8-layer debug shape the five errors very
nearly cancel: the untied 163 968-row vocabulary head is ~78 % of all FLOPs
against ``hidden_size = 1024``, so upstream over-reports by only ~0.7 %.
The correction matters at production-ish shapes, where the vocab head stops
dominating and the quadratic KDA charge grows with ``seq_length``.

Convention follows upstream Megatron and
``deepseek_v4_flops_patches.py``: every helper returns pure **FMAC**
(multiply) counts, and a single ``forward_backward_factor (3) *
fma_factor (2) = 6`` multiplier is applied once at the end.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

# ---------------------------------------------------------------------------
# Shared constants (mirror Megatron's expansion factors)
# ---------------------------------------------------------------------------

_FORWARD_BACKWARD_FACTOR: int = 3  # 1 forward + 2 backward.
_FMA_FACTOR: int = 2  # multiply + add per matmul element.
_SWIGLU_FFN_EXPANSION_FACTOR: int = 3  # gate + up + down.


# ---------------------------------------------------------------------------
# Per-layer pattern parsing
# ---------------------------------------------------------------------------


def _parse_int_sequence(raw: Any) -> Optional[List[int]]:
    """Parse a per-layer pattern into ``list[int]`` or return ``None``.

    K3's YAMLs carry ``linear_attention_freq`` and ``moe_layer_freq`` as
    strings (``"[1, 1, 1, 0, ...]"`` / ``"([0]*1+[1]*7)"``) which Primus's
    ``megatron.args.moe_layer_freq`` patch normalises for ``moe_layer_freq``
    only, so ``linear_attention_freq`` can still be a string here. Accepts a
    list/tuple, a literal string, or a python expression string.
    """
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        return [int(x) for x in raw]
    if isinstance(raw, int):
        return None  # an int is a *ratio*, handled by the caller
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        # `([0]*1+[1]*7)` is a BinOp, not a literal, so literal_eval alone is
        # not enough; fall back to a restricted eval of a parsed expression.
        try:
            value = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            value = eval(  # noqa: S307 - config-authored expression, same as upstream's moe_layer_freq
                compile(ast.parse(text, mode="eval"), "<linear_attention_freq>", "eval"), {}, {}
            )
        if isinstance(value, int):
            return None
        return [int(x) for x in value]
    raise TypeError(f"Unsupported per-layer pattern type: {type(raw).__name__}")


def linear_attention_pattern(raw: Any, num_layers: int) -> List[int]:
    """``1`` = KDA (linear attention), ``0`` = full attention, per layer.

    Mirrors upstream's ratio semantics at ``training.py:460-465`` — an int
    ``N`` means every ``N``-th layer is *full* attention — and requires an
    explicit list otherwise, because K3's released tail is irregular (both
    0-indexed 91 and 92 are full attention).
    """
    if isinstance(raw, int) and not isinstance(raw, bool):
        if raw <= 0:
            return [0] * num_layers
        return [0 if ((i + 1) % raw == 0) else 1 for i in range(num_layers)]
    pattern = _parse_int_sequence(raw)
    if pattern is None:
        # No pattern at all: every layer is full attention. Never true for a
        # real K3 config (kimi_k3_transformer_config validates it), but keeps
        # the closed form total rather than raising inside a logging path.
        return [0] * num_layers
    if len(pattern) != num_layers:
        raise ValueError(
            f"linear_attention_freq has {len(pattern)} entries, expected num_layers={num_layers}"
        )
    return pattern


def moe_layer_pattern(raw: Any, num_layers: int) -> List[int]:
    """``1`` = MoE MLP, ``0`` = dense MLP, per layer.

    Same semantics as ``training.py:329-341``: an int ``N`` means MoE every
    ``N``-th layer starting at 0.
    """
    if raw is None:
        return [0] * num_layers
    if isinstance(raw, int) and not isinstance(raw, bool):
        if raw <= 0:
            return [0] * num_layers
        return [1 if (i % raw == 0) else 0 for i in range(num_layers)]
    pattern = _parse_int_sequence(raw)
    if pattern is None:
        return [1] * num_layers
    if len(pattern) != num_layers:
        raise ValueError(f"moe_layer_freq has {len(pattern)} entries, expected num_layers={num_layers}")
    return pattern


def attn_res_num_blocks_before(layer_idx: int, block_size: Optional[int]) -> int:
    """Checkpoints in flight on **entry** to ``layer_idx``.

    A checkpoint is appended whenever ``layer_idx % block_size == 0``
    (``modeling_kimi_linear.py:995``), so on entry to layer ``L`` the count
    is ``ceil(L / block_size)``. Duplicated from
    ``kimi_k3_block.attn_res_num_blocks_before`` on purpose: this patch
    installs at ``before_train``, before the model modules are importable
    in some configurations, exactly as ``deepseek_v4_flops_patches`` inlines
    ``_parse_int_sequence``.
    """
    if not block_size or block_size <= 0 or layer_idx <= 0:
        return 0
    return -(-int(layer_idx) // int(block_size))  # ceil division


# ---------------------------------------------------------------------------
# Per-component closed forms. All return FMAC (multiply-only) PER TOKEN.
# ---------------------------------------------------------------------------


def kda_proj_fmac_per_token(
    *,
    hidden_size: int,
    num_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    conv_kernel_dim: int,
) -> int:
    """The nine KDA projections plus the three depthwise causal convolutions.

    Read off ``kimi_delta_attention.py:303-386``:

    ==================  =====================  ====================================
    module              shape                  line
    ==================  =====================  ====================================
    ``q_proj``          ``H -> qk_dim``        ``:303``
    ``k_proj``          ``H -> qk_dim``        ``:304``
    ``v_proj``          ``H -> v_dim``         ``:305``
    ``q/k/v_conv1d``    depthwise, kernel 4    ``:308-310``
    ``f_a_proj``        ``H -> K`` (duplicated) ``:315``
    ``f_b_proj``        ``K -> qk_dim``        ``:328``
    ``b_proj``          ``H -> num_heads``     ``:355``
    ``g_proj``          ``H -> v_dim``         ``:366``
    ``o_proj``          ``v_dim -> H``         ``:374``
    ==================  =====================  ====================================

    ``g_proj`` is the full-rank output gate (``kda_use_full_rank_gate:
    true``, the only variant implemented — ``:360-365`` raises otherwise).
    It is the term upstream's ``gated_delta_net`` branch has no slot for.
    """
    qk_dim = key_head_dim * num_heads
    v_dim = value_head_dim * num_heads
    proj = (
        hidden_size * qk_dim  # q_proj
        + hidden_size * qk_dim  # k_proj
        + hidden_size * v_dim  # v_proj
        + hidden_size * key_head_dim  # f_a_proj
        + key_head_dim * qk_dim  # f_b_proj
        + hidden_size * num_heads  # b_proj
        + hidden_size * v_dim  # g_proj
        + v_dim * hidden_size  # o_proj
    )
    # Three separate depthwise convs over q / k / v channels.
    conv = conv_kernel_dim * (2 * qk_dim + v_dim)
    return proj + conv


def kda_core_fmac_per_token(
    *,
    num_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    chunk_size: int,
) -> int:
    """The chunkwise delta-rule recurrence — **linear in the sequence length**.

    Matmul inventory of ``kda_kernels/_eager/reference.py::eager_chunk_kda``
    (``:301-402``), which is the numerical definition every KDA backend
    (``fla``'s ``chunk_kda`` today) is validated against. Per chunk of ``C``
    steps, per head, with ``K`` the key/state dim and ``V`` the value dim:

    =========================================  ============  ==========
    term                                       reference     FMAC
    =========================================  ============  ==========
    ``L = -beta * tril(<k.Gamma, k>, -1)``      ``:370``      ``C^2 K``
    ``(I - L)^-1`` by forward substitution      ``:372-375``  ``~C^3/3``
    ``W = M (Gamma * K)``                       ``:380``      ``C^2 K``
    ``U = M V``                                 ``:381``      ``C^2 V``
    ``V~ = U - W S``                            ``:389``      ``C K V``
    ``A = tril(<q.Gamma, k>)``                  ``:391``      ``C^2 K``
    ``(Q * Gamma) S``                           ``:394``      ``C K V``
    ``A V~``                                    ``:394``      ``C^2 V``
    ``(K * Gamma)^T V~``                        ``:398``      ``C K V``
    =========================================  ============  ==========

    Dividing by ``C`` gives the per-token cost
    ``3 C K + 2 C V + 3 K V + C^2/3``, with **no ``seq_length`` term at
    all**. That is the single biggest correction in this module: upstream
    charges these layers ``2 * n * d * S / 2`` per token, which grows
    linearly per token and therefore quadratically per sequence.

    The ``C^3/3`` triangular-inverse term is the one approximation here.
    ``fla``'s ``chunk_kda`` factors the same inverse over 16-wide
    sub-blocks, so its constant differs; at ``C = 64, K = V = 128`` the term
    is 2.7 % of the per-token core cost and the core is itself a minority of
    the layer, so the choice is not material. It is kept because dropping it
    would make the model silently optimistic.
    """
    per_head = (
        3 * chunk_size * key_head_dim  # three C^2 K matmuls
        + 2 * chunk_size * value_head_dim  # two C^2 V matmuls
        + 3 * key_head_dim * value_head_dim  # three C K V matmuls
        + (chunk_size * chunk_size) // 3  # triangular inverse
    )
    return num_heads * per_head


def mla_fmac_per_token(
    *,
    seq_len: int,
    hidden_size: int,
    num_heads: int,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_head_dim: int,
    qk_pos_emb_head_dim: int,
    v_head_dim: int,
    use_output_gate: bool,
) -> int:
    """NoPE MLA with the sigmoid output gate.

    Projection layout is upstream ``MLASelfAttention``'s verbatim — K3
    keeps the parent ``__init__`` (``kimi_k3_mla_attention.py:19-26``) —
    plus one new module:

    * ``linear_q_down_proj``   ``H -> q_lora_rank``
    * ``linear_q_up_proj``     ``q_lora_rank -> n * (qk_head_dim + qk_pos_emb_head_dim)``
    * ``linear_kv_down_proj``  ``H -> kv_lora_rank + qk_pos_emb_head_dim``
      (the trailing ``qk_pos_emb_head_dim`` dims are MQA-shared and bypass
      the latent)
    * ``linear_kv_up_proj``    ``kv_lora_rank -> n * (qk_head_dim + v_head_dim)``
    * ``linear_o_gate``        ``H -> n * v_head_dim``   (``:306-318``, new)
    * ``linear_proj``          ``n * v_head_dim -> H``

    NoPE costs nothing and saves nothing: the rotary table is zero-width, so
    the tensors fall through ``t_pass`` unchanged (``rope_utils.py:110-126``).
    The 64 positional dims are still projected, still concatenated into Q/K
    and still attended over, which is why ``q_head_dim`` below is
    ``qk_head_dim + qk_pos_emb_head_dim`` and not ``qk_head_dim``.

    Core attention is causal, so only half the ``(q, k)`` pairs are visible:
    ``n * q_head_dim * S/2`` for ``QK^T`` plus ``n * v_head_dim * S/2`` for
    ``PV``, matching upstream's convention at ``training.py:422-425``.
    """
    q_head_dim = qk_head_dim + qk_pos_emb_head_dim
    n_qk = num_heads * q_head_dim
    n_v = num_heads * v_head_dim

    if q_lora_rank and q_lora_rank > 0:
        q_term = hidden_size * q_lora_rank + q_lora_rank * n_qk
    else:
        q_term = hidden_size * n_qk

    kv_term = hidden_size * (kv_lora_rank + qk_pos_emb_head_dim) + kv_lora_rank * (
        num_heads * (qk_head_dim + v_head_dim)
    )

    gate = hidden_size * n_v if use_output_gate else 0
    o_proj = n_v * hidden_size
    core = (n_qk * seq_len) // 2 + (n_v * seq_len) // 2

    return q_term + kv_term + gate + o_proj + core


def dense_mlp_fmac_per_token(*, hidden_size: int, ffn_hidden_size: int, swiglu: bool) -> int:
    """Dense (``situ``-SwiGLU) MLP: gate + up + down, all ``H x ffn``."""
    expansion = _SWIGLU_FFN_EXPANSION_FACTOR if swiglu else 2
    return expansion * hidden_size * ffn_hidden_size


def latent_moe_fmac_per_token(
    *,
    hidden_size: int,
    latent_size: Optional[int],
    moe_ffn_hidden_size: int,
    moe_router_topk: int,
    num_experts: int,
    shared_expert_ffn_hidden_size: int,
    swiglu: bool,
) -> int:
    """Stable Latent MoE: router + latent down/up + routed experts + shared expert.

    The latent bottleneck is upstream's ``config.moe_latent_size`` feature
    (``moe_layer.py:198-221``), and the two projections run **once over every
    token, outside the dispatch** — ``fc1_latent_proj`` in ``preprocess``
    (``:359-363``) and ``fc2_latent_proj`` in ``postprocess`` (``:448-449``).
    So they are ``H*latent + latent*H`` per token, *not* per routed expert.

    Routed experts then run at ``latent`` width
    (``mlp.py:208-213``, ``experts.py:185, 206-207``), and each token visits
    ``moe_router_topk`` of them.

    The shared expert runs on the **pre-down-projection** hidden state, i.e.
    at ``H`` — ``shared_experts_compute`` (``:379-402``) is called with the
    original ``hidden_states``, not the latent-projected copy.

    ``StableLatentMoE``'s only addition over the stock latent ``MoELayer`` is
    an RMSNorm on the combined routed output, which is elementwise and carries
    no matmul.
    """
    expansion = _SWIGLU_FFN_EXPANSION_FACTOR if swiglu else 2
    router = hidden_size * num_experts
    expert_width = latent_size if latent_size else hidden_size
    latent = (hidden_size * latent_size + latent_size * hidden_size) if latent_size else 0
    routed = moe_router_topk * expansion * expert_width * moe_ffn_hidden_size
    shared = expansion * hidden_size * shared_expert_ffn_hidden_size if shared_expert_ffn_hidden_size else 0
    return router + latent + routed + shared


def attn_res_fmac_per_token(
    *,
    hidden_size: int,
    num_layers: int,
    attn_res_block_size: Optional[int],
) -> int:
    """The attention-residual mixers and the post-stack head.

    Each mixer scores ``num_blocks + 1`` candidates with a rank-1 vector and
    then takes a convex combination of them
    (``attention_residual.py:209-213``), i.e. two ``(num_blocks+1) x hidden``
    reductions per token. The RMSNorm at ``:206-207`` is elementwise.

    The schedule is read off ``kimi_k3_block.py``:

    * ``mlp_res_mixer`` is built on **every** layer (``:353-358``) and runs
      *after* the checkpoint append (``:499`` then ``:515``), so it sees the
      post-append count.
    * ``attn_res_mixer`` is built only where ``num_blocks_in > 0``
      (``:368-376``) and runs *before* the append (``:494-495``), so it sees
      the pre-append count.
    * exactly one ``attn_res_head`` exists, on the ``post_process`` stage
      (``:594-601``), and sees the final count.

    Two orders of magnitude below the MoE term at every shape tried; counted
    because leaving a real module out of a "correct" closed form is how the
    number under repair got wrong in the first place.
    """
    if not attn_res_block_size or attn_res_block_size <= 0:
        return 0

    total_candidates = 0
    for layer_idx in range(num_layers):
        before = attn_res_num_blocks_before(layer_idx, attn_res_block_size)
        if before > 0:  # attn_res_mixer exists
            total_candidates += before + 1
        after = before + (1 if layer_idx % attn_res_block_size == 0 else 0)
        total_candidates += after + 1  # mlp_res_mixer, always present
    # The head, on the final checkpoint count.
    total_candidates += attn_res_num_blocks_before(num_layers, attn_res_block_size) + 1

    return 2 * total_candidates * hidden_size


def logits_fmac_per_token(*, hidden_size: int, padded_vocab_size: int, mtp_num_layers: int) -> int:
    """LM head, one per main path plus one per MTP depth."""
    return (mtp_num_layers + 1) * hidden_size * padded_vocab_size


def mtp_eh_proj_fmac_per_token(*, hidden_size: int, mtp_num_layers: int) -> int:
    """The per-depth ``eh_proj``, ``[2h] -> [h]``.

    ``MultiTokenPredictionLayer`` concatenates the normalised previous hidden
    state with the normalised embedding of the token one position to the right
    and projects the pair back to model width
    (``multi_token_prediction.py:799-810``, ``:905-906``). The two RMSNorms and
    the final one are elementwise and are charged nowhere, consistent with how
    this closed form treats every other norm.
    """
    return mtp_num_layers * 2 * hidden_size * hidden_size


def mtp_body_fmac_per_token(
    *,
    mtp_num_layers: int,
    mtp_layer_is_kda: bool,
    mtp_layer_is_moe: bool,
    kda_proj_pt: int,
    kda_core_pt: int,
    mla_pt: int,
    dense_pt: int,
    moe_pt: int,
) -> int:
    """One full K3 decoder layer per MTP depth.

    The MTP layer is a real :class:`KimiK3Layer` -- same attention module,
    same Stable Latent MoE -- so it costs exactly what the corresponding
    backbone layer costs. It runs **no** attention-residual mixers (see
    ``kimi_k3_mtp_specs.py``), so nothing is added to the ``attn_res`` term.

    Charging it matters for the measurement rather than for the model: without
    this term an MTP run's reported TFLOP/s is understated by roughly
    ``1 / num_layers`` of the backbone, which is precisely the quantity an
    MTP-on vs MTP-off throughput comparison is trying to see.
    """
    if mtp_num_layers <= 0:
        return 0
    attention_pt = (kda_proj_pt + kda_core_pt) if mtp_layer_is_kda else mla_pt
    ffn_pt = moe_pt if mtp_layer_is_moe else dense_pt
    return mtp_num_layers * (attention_pt + ffn_pt)


# ---------------------------------------------------------------------------
# Public closed form
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KimiK3FlopsBreakdown:
    """Per-component FMAC totals for one global batch (pre fwd+bwd / FMA scaling)."""

    kda_proj: int
    kda_core: int
    mla: int
    dense_mlp: int
    moe: int
    attn_res: int
    logits: int
    mtp: int = 0

    num_kda_layers: int = 0
    num_full_attn_layers: int = 0
    num_dense_layers: int = 0
    num_moe_layers: int = 0
    num_mtp_layers: int = 0

    def total_fmac(self) -> int:
        return (
            self.kda_proj
            + self.kda_core
            + self.mla
            + self.dense_mlp
            + self.moe
            + self.attn_res
            + self.logits
            + self.mtp
        )

    def to_total_flops(self) -> int:
        """Apply the fwd+bwd (3) x FMA (2) = 6 expansion."""
        return _FORWARD_BACKWARD_FACTOR * _FMA_FACTOR * self.total_fmac()


def compute_kimi_k3_flops(args: Any, batch_size: int) -> Tuple[int, KimiK3FlopsBreakdown]:
    """Closed-form Kimi K3 FLOPs for one global batch.

    Returns ``(total_flops, breakdown)``; ``total_flops`` is in Megatron's
    reporting convention and can be substituted directly for upstream's
    return value.
    """
    seq_len = int(args.seq_length)
    hidden_size = int(args.hidden_size)
    num_layers = int(args.num_layers)
    num_heads = int(args.num_attention_heads)
    swiglu = bool(getattr(args, "swiglu", False))

    la_pattern = linear_attention_pattern(getattr(args, "linear_attention_freq", None), num_layers)
    moe_pattern = moe_layer_pattern(getattr(args, "moe_layer_freq", None), num_layers)

    # --- KDA geometry ---
    kda_num_heads = int(getattr(args, "linear_num_value_heads", None) or num_heads)
    key_head_dim = int(getattr(args, "linear_key_head_dim", None) or getattr(args, "kv_channels", 0) or 0)
    value_head_dim = int(getattr(args, "linear_value_head_dim", None) or key_head_dim)
    conv_kernel_dim = int(getattr(args, "linear_conv_kernel_dim", 0) or 0)
    chunk_size = int(getattr(args, "kda_chunk_size", 64) or 64)

    # --- MLA geometry ---
    q_lora_rank = int(getattr(args, "q_lora_rank", 0) or 0)
    kv_lora_rank = int(getattr(args, "kv_lora_rank", 0) or 0)
    qk_head_dim = int(getattr(args, "qk_head_dim", 0) or 0)
    qk_pos_emb_head_dim = int(getattr(args, "qk_pos_emb_head_dim", 0) or 0)
    v_head_dim = int(getattr(args, "v_head_dim", None) or getattr(args, "kv_channels", 0) or 0)
    use_output_gate = bool(getattr(args, "mla_use_output_gate", False))

    # --- MLP / MoE geometry ---
    ffn_hidden_size = int(args.ffn_hidden_size)
    moe_ffn_hidden_size = int(getattr(args, "moe_ffn_hidden_size", None) or ffn_hidden_size)
    moe_router_topk = int(getattr(args, "moe_router_topk", 1) or 1)
    num_experts = int(getattr(args, "num_experts", 0) or 0)
    shared_ffn = int(getattr(args, "moe_shared_expert_intermediate_size", 0) or 0)
    # Prefer K3's own field: args.moe_latent_size is None unless
    # patch_k3_args_moe_latent_size ran, and this closed form must be
    # independent of patch ordering.
    latent_size = getattr(args, "routed_expert_hidden_size", None)
    if latent_size is None:
        latent_size = getattr(args, "moe_latent_size", None)
    latent_size = int(latent_size) if latent_size else None

    attn_res_block_size = int(getattr(args, "attn_res_block_size", 0) or 0)
    mtp_num_layers = int(getattr(args, "mtp_num_layers", 0) or 0)
    padded_vocab_size = int(getattr(args, "padded_vocab_size", None) or args.vocab_size)

    tokens = batch_size * seq_len

    kda_proj_pt = kda_proj_fmac_per_token(
        hidden_size=hidden_size,
        num_heads=kda_num_heads,
        key_head_dim=key_head_dim,
        value_head_dim=value_head_dim,
        conv_kernel_dim=conv_kernel_dim,
    )
    kda_core_pt = kda_core_fmac_per_token(
        num_heads=kda_num_heads,
        key_head_dim=key_head_dim,
        value_head_dim=value_head_dim,
        chunk_size=chunk_size,
    )
    mla_pt = mla_fmac_per_token(
        seq_len=seq_len,
        hidden_size=hidden_size,
        num_heads=num_heads,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_head_dim=qk_head_dim,
        qk_pos_emb_head_dim=qk_pos_emb_head_dim,
        v_head_dim=v_head_dim,
        use_output_gate=use_output_gate,
    )
    dense_pt = dense_mlp_fmac_per_token(
        hidden_size=hidden_size, ffn_hidden_size=ffn_hidden_size, swiglu=swiglu
    )
    moe_pt = latent_moe_fmac_per_token(
        hidden_size=hidden_size,
        latent_size=latent_size,
        moe_ffn_hidden_size=moe_ffn_hidden_size,
        moe_router_topk=moe_router_topk,
        num_experts=num_experts,
        shared_expert_ffn_hidden_size=shared_ffn,
        swiglu=swiglu,
    )

    num_kda = sum(1 for flag in la_pattern if flag)
    num_full = num_layers - num_kda
    num_moe = sum(1 for flag in moe_pattern if flag)
    num_dense = num_layers - num_moe

    # The MTP layer mirrors the *final* backbone layer unless mtp_layer_type
    # forces a variant. Resolved off the args-layer patterns rather than off a
    # config, because this whole module runs against ``args``.
    mtp_layer_type = str(getattr(args, "mtp_layer_type", "mirror_last") or "mirror_last")
    if mtp_layer_type == "kda":
        mtp_is_kda = True
    elif mtp_layer_type == "mla":
        mtp_is_kda = False
    else:
        mtp_is_kda = bool(la_pattern[-1]) if la_pattern else False
    mtp_is_moe = bool(moe_pattern[-1]) if moe_pattern else False

    breakdown = KimiK3FlopsBreakdown(
        kda_proj=tokens * num_kda * kda_proj_pt,
        kda_core=tokens * num_kda * kda_core_pt,
        mla=tokens * num_full * mla_pt,
        dense_mlp=tokens * num_dense * dense_pt,
        moe=tokens * num_moe * moe_pt,
        attn_res=tokens
        * attn_res_fmac_per_token(
            hidden_size=hidden_size,
            num_layers=num_layers,
            attn_res_block_size=attn_res_block_size,
        ),
        logits=tokens
        * logits_fmac_per_token(
            hidden_size=hidden_size,
            padded_vocab_size=padded_vocab_size,
            mtp_num_layers=mtp_num_layers,
        ),
        mtp=tokens
        * (
            mtp_body_fmac_per_token(
                mtp_num_layers=mtp_num_layers,
                mtp_layer_is_kda=mtp_is_kda,
                mtp_layer_is_moe=mtp_is_moe,
                kda_proj_pt=kda_proj_pt,
                kda_core_pt=kda_core_pt,
                mla_pt=mla_pt,
                dense_pt=dense_pt,
                moe_pt=moe_pt,
            )
            + mtp_eh_proj_fmac_per_token(hidden_size=hidden_size, mtp_num_layers=mtp_num_layers)
        ),
        num_kda_layers=num_kda,
        num_full_attn_layers=num_full,
        num_dense_layers=num_dense,
        num_moe_layers=num_moe,
        num_mtp_layers=mtp_num_layers,
    )
    return breakdown.to_total_flops(), breakdown


# ---------------------------------------------------------------------------
# Args-layer fix: make `moe_latent_size` reach `training.py:361`
# ---------------------------------------------------------------------------


@register_patch(
    "megatron.args.kimi_k3_moe_latent_size",
    backend="megatron",
    phase="build_args",
    description=(
        "Kimi K3: mirror routed_expert_hidden_size onto the args-layer "
        "moe_latent_size, which training.py:361 reads for FLOPs/params "
        "reporting and which the config-layer __post_init__ mapping does "
        "not reach."
    ),
    # Gated tightly: `routed_expert_hidden_size` is a K3-only field, so this
    # would be a no-op elsewhere anyway, but `patches/__init__.py` asks for
    # conditions precise enough that a patch cannot alter an unrelated job.
    condition=lambda ctx: getattr(get_args(ctx), "model_type", None) == "kimi_k3",
)
def patch_k3_args_moe_latent_size(ctx: PatchContext):
    """Set ``args.moe_latent_size`` from ``routed_expert_hidden_size``.

    ``KimiK3TransformerConfig.__post_init__`` already mirrors the two fields on
    the **config**, which is what every consumer that shapes the model reads:
    ``moe_layer.py:198`` (the latent projections), ``experts.py:185``/``:206``
    (the routed-expert widths), ``mlp.py:210`` and
    ``transformer_layer.py:440``. The *args* copy is a separate object and is
    read only by ``num_floating_point_operations``
    (``training.py:236-309``, ``:361``, ``:545-554``, ``:616``).

    **The model was never affected**; this patch only fixes reporting. Verified
    on a live launcher run, not inferred: the builder's own
    ``[Primus:Kimi-K3] latent MoE width`` line shows
    ``config.moe_latent_size=512, args.moe_latent_size=None`` with
    ``built.fc1_latent_proj=(…, (512, 1024))`` — i.e. the weights are the
    configured width even while the args copy is ``None``.

    **Read the source key off ``module_config.params``, not off
    ``backend_args``.** This is the phase-ordering trap that made the original
    version of this patch a no-op for its entire life:

    * ``train_runtime.py:404`` builds ``backend_args`` from
      ``adapter.convert_config``, which only knows the keys Megatron declares.
      ``routed_expert_hidden_size`` is Kimi-K3-only, so at this point it lives
      on ``module_config.params`` and **not** on ``backend_args``.
    * ``train_runtime.py:412`` runs the ``build_args`` phase — here.
    * ``train_runtime.py:442`` merges the two, *after* the patches.

    So ``getattr(backend_args, "routed_expert_hidden_size")`` returns ``None``
    here and the patch silently did nothing, while still logging
    "✓ Applied". Writing to ``backend_args`` *is* correct, because
    ``merge_namespace(backend_args, module_config.params,
    allow_override=False)`` keeps the destination's value for any key it
    already has (``yaml_utils.py:121-122``), so the write survives the merge
    and becomes the live ``get_args()`` value.

    Legal because ``arguments.py:1534-1535`` only asserts ``> 0`` on a
    non-``None`` value, ``:1656`` forwards it into the config kwargs, and
    ``_resolve_latent_fields`` raises only when the two **disagree**.
    """
    args = ctx.extra.get("backend_args", None)
    if args is None:
        return

    latent = None
    try:
        latent = getattr(get_args(ctx), "routed_expert_hidden_size", None)
    except AssertionError:
        # No module_config in the context (unit tests build a bare one); fall
        # back to backend_args, which is where the key lands post-merge.
        pass
    if not latent:
        latent = getattr(args, "routed_expert_hidden_size", None)
    if not latent:
        return

    existing = getattr(args, "moe_latent_size", None)
    if existing is not None and int(existing) != int(latent):
        raise ValueError(
            f"[Patch:megatron.args.kimi_k3_moe_latent_size] moe_latent_size={existing} "
            f"disagrees with routed_expert_hidden_size={latent}; they name the same "
            "latent width and the config layer would raise on the mismatch."
        )

    args.moe_latent_size = int(latent)
    log_rank_0(
        "[Patch:megatron.args.kimi_k3_moe_latent_size] args.moe_latent_size "
        f"= {args.moe_latent_size} (from routed_expert_hidden_size); "
        "training.py:361 FLOPs/params reporting now sees the latent bottleneck"
    )


# ---------------------------------------------------------------------------
# Args-layer fix: make `num_nextn_predict_layers` drive MTP
# ---------------------------------------------------------------------------


@register_patch(
    "megatron.args.kimi_k3_mtp_num_layers",
    backend="megatron",
    phase="build_args",
    description=(
        "Kimi K3: mirror num_nextn_predict_layers onto the args-layer "
        "mtp_num_layers, which arguments.py's MTP validation, training.py's "
        "FLOPs/params reporting and MTPLossLoggingHelper all read and which "
        "the config-layer __post_init__ mapping does not reach."
    ),
    condition=lambda ctx: getattr(get_args(ctx), "model_type", None) == "kimi_k3",
)
def patch_k3_args_mtp_num_layers(ctx: PatchContext):
    """Set ``args.mtp_num_layers`` from ``num_nextn_predict_layers``.

    ``num_nextn_predict_layers`` is the name Kimi K3's own ``config.json`` and
    ``configuration_kimi_k3.py`` use; ``mtp_num_layers`` is Megatron's. The
    config dataclass reconciles them
    (``kimi_k3_transformer_config.py::_resolve_mtp_fields``), but that object is
    built long after ``args`` and is not what the reporting path reads. Three
    consumers live on the args side only:

    * ``arguments.py:1492-1500`` -- the MTP position-embedding validation.
    * ``training.py:347-351`` and ``:594-623`` -- FLOPs and parameter counts,
      which add ``mtp_num_layers`` layers to both.
    * ``training.py:2062-2065`` -- ``MTPLossLoggingHelper.track_mtp_metrics``,
      i.e. whether the per-depth MTP loss appears in the training log at all.

    **Read the source key off ``module_config.params``, not off
    ``backend_args``.** At this phase ``backend_args`` is only what
    ``adapter.convert_config`` produced from the keys Megatron declares
    (``train_runtime.py:404``), and every Kimi-K3-only key is still on the
    Primus side; the two are merged at ``train_runtime.py:442``, *after* the
    ``build_args`` phase has run. A patch that reads
    ``backend_args.num_nextn_predict_layers`` here therefore sees ``None`` and
    silently does nothing. (Verified in a real launcher run: the "Final backend
    args (after patches)" dump lists ``mtp_num_layers ... None`` while the
    "Primus-specific parameters" dump 350 lines later lists
    ``num_nextn_predict_layers ... 1``. The sibling
    ``patch_k3_args_moe_latent_size`` above reads ``backend_args`` and is a
    no-op for exactly this reason.)

    Writing to ``backend_args`` *is* correct: ``merge_namespace(backend_args,
    module_config.params, allow_override=False)`` keeps the destination's value
    for any key it already has (``yaml_utils.py:121-122``), so a value set here
    survives the merge and becomes the live ``get_args()`` value.

    ``0`` is normalised to ``None`` for the reason on the config field:
    ``mtp_on_this_rank`` tests ``is not None``, so a literal 0 turns MTP
    half-on and then asserts.
    """
    args = ctx.extra.get("backend_args", None)
    if args is None:
        return

    nextn = None
    try:
        nextn = getattr(get_args(ctx), "num_nextn_predict_layers", None)
    except AssertionError:
        # No module_config in the context (unit tests construct a bare one);
        # fall back to backend_args, which is where a directly-set yaml key
        # would land once the merge has happened.
        pass
    if nextn is None:
        nextn = getattr(args, "num_nextn_predict_layers", None)
    existing = getattr(args, "mtp_num_layers", None)

    if nextn is None:
        # Nothing K3-native to mirror. Still normalise a literal 0.
        if existing is not None and int(existing) == 0:
            args.mtp_num_layers = None
            log_rank_0(
                "[Patch:megatron.args.kimi_k3_mtp_num_layers] args.mtp_num_layers 0 -> None; "
                "0 is not 'MTP off' upstream (mtp_on_this_rank tests `is not None`)"
            )
        return

    if existing is not None and int(existing) != int(nextn):
        raise ValueError(
            f"[Patch:megatron.args.kimi_k3_mtp_num_layers] mtp_num_layers={existing} "
            f"disagrees with num_nextn_predict_layers={nextn}; they name the same "
            "number of MTP depths and the config layer would raise on the mismatch."
        )

    args.mtp_num_layers = int(nextn) or None
    log_rank_0(
        "[Patch:megatron.args.kimi_k3_mtp_num_layers] args.mtp_num_layers "
        f"= {args.mtp_num_layers} (from num_nextn_predict_layers={nextn}); "
        "arguments.py MTP validation, training.py FLOPs/params and "
        "MTPLossLoggingHelper now see it"
    )


# ---------------------------------------------------------------------------
# Wrapper installation
# ---------------------------------------------------------------------------

# Module-level latch so the breakdown is logged exactly once, even though
# `num_floating_point_operations` is called on every `training_log` and every
# `train_step`.
_BREAKDOWN_LOGGED = False


def _emit_breakdown(
    *,
    args: Any,
    batch_size: int,
    breakdown: KimiK3FlopsBreakdown,
    total_flops: int,
    upstream_flops: Optional[int],
) -> None:
    """Log the per-component breakdown, one ``log_rank_0`` call per row.

    Also logs the number upstream *would* have reported, because the whole
    point of this patch is that the two differ and the difference is
    shape-dependent. A reviewer reading a log should not have to re-derive
    it.
    """

    def _tflops(fmac: int) -> float:
        return fmac * _FORWARD_BACKWARD_FACTOR * _FMA_FACTOR / 1.0e12

    rows: List[Tuple[str, float]] = [
        ("kda_proj", _tflops(breakdown.kda_proj)),
        ("kda_core", _tflops(breakdown.kda_core)),
        ("mla", _tflops(breakdown.mla)),
        ("dense_mlp", _tflops(breakdown.dense_mlp)),
        ("moe", _tflops(breakdown.moe)),
        ("attn_res", _tflops(breakdown.attn_res)),
        ("logits", _tflops(breakdown.logits)),
        ("mtp", _tflops(breakdown.mtp)),
    ]

    log_rank_0(
        "[Patch:megatron.kimi_k3.flops_reporting] K3 closed-form FLOPs breakdown -- "
        f"batch_size={batch_size}, seq_length={int(args.seq_length)}, "
        f"num_layers={int(args.num_layers)} "
        f"(kda={breakdown.num_kda_layers}, full_attn={breakdown.num_full_attn_layers}, "
        f"dense_mlp={breakdown.num_dense_layers}, moe={breakdown.num_moe_layers}, "
        f"mtp={breakdown.num_mtp_layers}), "
        f"padded_vocab={int(getattr(args, 'padded_vocab_size', None) or args.vocab_size)}"
    )
    for name, tflops in rows:
        log_rank_0(f"[Patch:megatron.kimi_k3.flops_reporting]   {name:<10s} = {tflops:10.4f} TFLOP")
    log_rank_0(
        f"[Patch:megatron.kimi_k3.flops_reporting]   {'TOTAL':<10s} = "
        f"{total_flops / 1.0e12:10.4f} TFLOP / global-batch"
    )
    if upstream_flops is not None:
        delta = (upstream_flops - total_flops) / total_flops * 100.0 if total_flops else 0.0
        log_rank_0(
            f"[Patch:megatron.kimi_k3.flops_reporting]   {'upstream':<10s} = "
            f"{upstream_flops / 1.0e12:10.4f} TFLOP / global-batch "
            f"({delta:+.2f} % vs the K3 closed form)"
        )


def _make_k3_num_floating_point_operations(original_fn, *, dispatch_k3: bool):
    """Return a wrapper that dispatches K3 vs upstream model types.

    ``dispatch_k3`` is captured at install time from ``args.model_type``
    rather than re-checked per call: Megatron's ``pretrain()`` overwrites
    ``args.model_type`` with the ``ModelType`` enum at ``training.py:1210``
    *before* ``train()`` ever calls ``num_floating_point_operations``, so a
    runtime string check would silently fall through to the upstream
    formula. Same reasoning as ``deepseek_v4_flops_patches.py``.
    """

    def wrapped(args, batch_size):
        if not dispatch_k3:
            return original_fn(args, batch_size)

        total_flops, breakdown = compute_kimi_k3_flops(args, batch_size)

        global _BREAKDOWN_LOGGED
        if not _BREAKDOWN_LOGGED:
            _BREAKDOWN_LOGGED = True
            try:
                upstream_flops = original_fn(args, batch_size)
            except Exception:  # noqa: BLE001 - a logging nicety must never break training
                upstream_flops = None
            _emit_breakdown(
                args=args,
                batch_size=batch_size,
                breakdown=breakdown,
                total_flops=total_flops,
                upstream_flops=upstream_flops,
            )

        return total_flops

    wrapped.__wrapped__ = original_fn
    wrapped._k3_flops_patched = True
    return wrapped


_TRAINER_REBIND_TARGETS: Sequence[str] = (
    # Primus's Megatron trainer imports `num_floating_point_operations` at
    # module load time and resolves the bare name from its OWN globals at the
    # call site, so patching only `megatron.training.training` never reaches
    # the name that actually drives per-iter TFLOPs reporting. Same target
    # list as `deepseek_v4_flops_patches._TRAINER_REBIND_TARGETS`.
    "primus.modules.trainer.megatron.trainer",
)


def _rebind_trainer_imports(wrapped_fn) -> List[str]:
    """Rebind ``num_floating_point_operations`` wherever it was captured at import."""
    import sys

    rebound: List[str] = []
    for module_name in _TRAINER_REBIND_TARGETS:
        mod = sys.modules.get(module_name)
        if mod is None:
            continue
        if getattr(mod, "num_floating_point_operations", None) is wrapped_fn:
            continue
        if not hasattr(mod, "num_floating_point_operations"):
            continue
        mod.num_floating_point_operations = wrapped_fn
        rebound.append(module_name)
    return rebound


@register_patch(
    "megatron.kimi_k3.flops_reporting",
    backend="megatron",
    phase="before_train",
    description=(
        "Kimi K3: replace Megatron's GPT/MLA-shaped "
        "num_floating_point_operations with a K3 closed form (KDA linear in "
        "T, NoPE MLA with the sigmoid output gate, the latent MoE "
        "bottleneck, and the attention-residual mixers). Falls through "
        "byte-for-byte for non-K3 model types."
    ),
    condition=lambda ctx: getattr(get_args(ctx), "model_type", None) == "kimi_k3",
)
def patch_k3_flops_reporting(ctx: PatchContext):
    """Install the K3 FLOPs wrapper on ``training.num_floating_point_operations``."""
    import megatron.training.training as training_module

    original_fn = training_module.num_floating_point_operations
    if getattr(original_fn, "_k3_flops_patched", False):
        log_rank_0(
            "[Patch:megatron.kimi_k3.flops_reporting] " "num_floating_point_operations already patched, skip"
        )
        return

    wrapped = _make_k3_num_floating_point_operations(original_fn, dispatch_k3=True)
    training_module.num_floating_point_operations = wrapped
    rebound = _rebind_trainer_imports(wrapped)

    log_rank_0(
        "[Patch:megatron.kimi_k3.flops_reporting] wrapped "
        "num_floating_point_operations; per-iter TFLOPs now reported with the "
        "K3-aware closed form"
    )
    if rebound:
        log_rank_0(f"[Patch:megatron.kimi_k3.flops_reporting] rebound trainer import bindings: {rebound}")
    else:
        log_rank_0(
            "[Patch:megatron.kimi_k3.flops_reporting] no trainer modules needed "
            f"rebinding (none of {list(_TRAINER_REBIND_TARGETS)} were imported yet)."
        )


__all__: Sequence[str] = (
    "KimiK3FlopsBreakdown",
    "attn_res_fmac_per_token",
    "attn_res_num_blocks_before",
    "compute_kimi_k3_flops",
    "dense_mlp_fmac_per_token",
    "kda_core_fmac_per_token",
    "kda_proj_fmac_per_token",
    "latent_moe_fmac_per_token",
    "linear_attention_pattern",
    "logits_fmac_per_token",
    "mla_fmac_per_token",
    "moe_layer_pattern",
    "patch_k3_args_moe_latent_size",
    "patch_k3_flops_reporting",
)
