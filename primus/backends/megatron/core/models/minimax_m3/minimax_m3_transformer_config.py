###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MiniMax-M3 transformer config (MSA = MiniMax Sparse Attention).

M3's text tower is a plain GQA + MoE decoder, so it reuses upstream
``GPTModel`` unchanged. The one thing upstream ``TransformerConfig`` cannot
express is MSA: an index branch scores each 128-token KV block, the top
``sparse_topk_blocks`` blocks are kept per GQA group, and the main branch then
runs exact attention over that subset only.

``MSATransformerConfig`` therefore extends ``TransformerConfig`` -- *not*
``MLATransformerConfig``: M3 has no latent KV, and Megatron's own sparse
attention path (``experimental_attention_variant: dsa``) is MLA-only.
``minimax_sparse_attention`` is the family switch, mirroring how
``multi_latent_attention`` selects ``MLATransformerConfig``; the selection
itself lives in ``primus/backends/megatron/patches/minimax_m3_config_patches.py``.

M3's other two deviations from a stock GQA+MoE decoder -- the ``swigluoai``
activation and ``use_gemma_norm`` -- need no new machinery: Megatron already
computes both, so this config carries them under their ``config.json`` names
and derives the upstream fields in ``__post_init__``.

Values and semantics come from https://huggingface.co/MiniMaxAI/MiniMax-M3
(config.json) and from the official implementation in transformers,
``models/minimax_m3_vl/modeling_minimax_m3_vl.py`` (``MiniMaxM3VLRMSNorm``,
``MiniMaxM3VLDenseMLP``, ``MiniMaxM3VLExperts``). Field names are kept verbatim
so the preset reads like the reference config.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from megatron.core.transformer.transformer_config import TransformerConfig

_LayerPattern = Optional[Union[int, str, List[int], Tuple[int, ...]]]

# Block-score reductions MSA defines. Only `max` is in the released config, and
# anything else would change what the index branch means, so the rest are
# rejected rather than silently treated as `max`.
_SUPPORTED_SCORE_TYPES = frozenset({"max"})

# Kernels that can compute the block-sparse attention: the plain-PyTorch
# reference and the gfx950 FlyDSL kernels.
_SUPPORTED_MSA_BACKENDS = frozenset({"eager", "flydsl"})

# Indexer distillation losses; see indexer_loss.py.
_SUPPORTED_INDEXER_LOSS_TYPES = frozenset({"sparse", "dense"})


def normalize_sparse_layer_pattern(
    value: _LayerPattern, num_layers: int, *, field_name: str
) -> Tuple[int, ...]:
    """Normalize a per-layer 0/1 pattern to a tuple of length ``num_layers``.

    Accepts the same spellings as ``moe_layer_freq`` -- an int N (every Nth
    layer), a list, or a Python list expression such as ``"([0]*3+[1]*57)"`` --
    and reuses upstream's ``moe_freq_type`` to evaluate the expression form.
    These keys are not in Megatron's argparse, so Primus merges them onto
    ``args`` as the raw YAML value and the string never gets parsed on the way
    in; this is where that happens.

    ``None`` means "every layer", which is what a config that turns MSA on
    without narrowing it to a subset of layers should get.
    """
    if value is None:
        return (1,) * num_layers

    # Imported lazily: megatron.training pulls in the whole training stack,
    # which the config dataclass itself does not need.
    from megatron.training.arguments import moe_freq_type

    pattern = moe_freq_type(value) if isinstance(value, (int, str)) else value

    if isinstance(pattern, int):
        return tuple(1 if (i % pattern == 0) else 0 for i in range(num_layers))

    if not isinstance(pattern, (list, tuple)):
        raise TypeError(f"{field_name} must be an int, a list, or a list expression; got {value!r}")

    if len(pattern) != num_layers:
        raise ValueError(
            f"Invalid length of {field_name}: {len(pattern)}, expected num_layers={num_layers}; "
            f"current pattern: {value!r}"
        )

    invalid = sorted({entry for entry in pattern if entry not in (0, 1)})
    if invalid:
        raise ValueError(f"{field_name} entries must be 0 or 1; got {invalid}")

    return tuple(int(entry) for entry in pattern)


@dataclass
class MSATransformerConfig(TransformerConfig):
    """Configuration object for MiniMax Sparse Attention (MSA) transformers.

    Field names mirror ``config.json``'s ``sparse_attention_config``; the
    defaults are M3's released values, so a preset only has to set
    ``minimax_sparse_attention: true`` plus whatever it wants to change.
    """

    minimax_sparse_attention: bool = True
    """Whether to use MiniMax Sparse Attention. Selects this config class."""

    sparse_attention_freq: _LayerPattern = None
    """Per-layer 0/1 pattern: 1 = MSA layer, 0 = dense attention. None = all layers."""

    sparse_num_index_heads: int = 4
    """Number of heads in the index branch that scores KV blocks."""

    sparse_index_dim: int = 128
    """Dimension per index-branch head."""

    sparse_block_size: int = 128
    """Number of KV tokens per scored block."""

    sparse_topk_blocks: int = 16
    """Blocks kept per GQA group, i.e. sparse_topk_blocks * sparse_block_size KV tokens."""

    sparse_score_type: str = "max"
    """How a block's token scores reduce to one block score."""

    sparse_init_block: int = 0
    """Leading blocks always attended, on top of the top-k selection."""

    sparse_local_block: int = 1
    """Trailing (most recent) blocks always attended, on top of the top-k selection."""

    sparse_disable_index_value: _LayerPattern = None
    """Per-layer 0/1 pattern for the index branch; M3 ships the same list as the freq."""

    # ---- Activation: config.json's `hidden_act: swigluoai` ----

    swiglu_alpha: float = 1.702
    """Gate steepness. Megatron's `quick_gelu` hardcodes 1.702, so only that value is expressible."""

    swiglu_limit: float = 7.0
    """Clamp bound on both GLU halves; mirrored into `activation_func_clamp_value`."""

    # ---- Normalization: config.json's `use_gemma_norm` ----

    use_gemma_norm: bool = True
    """RMSNorm weighted by (1 + w); mirrored into `layernorm_zero_centered_gamma`."""

    # ---- MSA runtime (Primus-side, not from config.json) ----

    msa_backend: str = "eager"
    """Which kernel computes the block-sparse attention: 'eager' or 'flydsl'."""

    msa_fused_qk_norm_rope: bool = False
    """The attention's q/k split, q/k RMSNorm and partial RoPE, and the indexer's RMSNorm + RoPE,
    as one Triton pass each way (qk_norm_rope.py). Bit-identical to the eager ops, checked on
    first use; off for shapes or modules it does not cover."""

    msa_fused_indexer_loss: bool = False
    """The sparse indexer loss in a few Triton launches (indexer_loss_fused.py). Bit-identical to
    compute_sparse_indexer_loss, checked on first use."""

    msa_fused_indexer_loss_scatter: bool = False
    """With msa_fused_indexer_loss: the loss backward writes the gradient of block_scores row by
    row from its last kernel instead of zeros + clamp + scatter_add_ (three launches fewer per
    call). Bit-identical, checked on first use; rows wider than 64 blocks keep scatter_add_."""

    msa_fused_zero_centered_gamma: bool = False
    """With msa_fused_qk_norm_rope: the q / k and indexer norm + RoPE kernels form a zero-centered
    norm's BF16 ``1 + weight`` in-register instead of reading the result of a separate
    ``weight + 1`` launch (four launches fewer per MSA layer call). Bit-identical, checked on
    first use; norms that are not zero-centered or whose weight is not BF16 are unaffected."""

    msa_fused_indexer_input_norm: bool = False
    """The MSA indexer's input norm (``rms_norm`` of the FP32 copy of the residual stream, cast
    back) as one Triton launch that follows ATen's reduction order (indexer_input_norm.py)
    instead of five launches. Bit-identical, checked on first use per row width; other shapes
    keep the eager ops."""

    msa_shared_rope_tables: bool = False
    """The cos / sin RoPE tables of the MSA layers are kept per rotary ``freqs`` tensor (Megatron
    caches that tensor) instead of being recomputed in every layer call (rope.py). Same ops, so
    the same bits; the first reuse is compared with recomputed tables."""

    msa_fused_select_plan: bool = False
    """flydsl backend: the indexer's block selection and the backward's BlockPlan in a few Triton
    launches (flydsl/select_plan.py) instead of the eager ops and build_block_plan. Integer
    bookkeeping, identical by construction; shapes past select_plan.supported() keep the eager
    path."""

    msa_shared_tile_staging: bool = False
    """flydsl backend on gfx1250: the MSA forward and dQ kernels run 4 / 2 adjacent tokens of a KV
    group per workgroup, in lockstep, and stage a K/V tile that several of them select at the same
    step once (flydsl/msa_token_fwd_w32.py, msa_token_bwd_w32.py). Each token keeps its own step
    order and arithmetic; the first call for a shape compares every output bit with the
    one-token-per-workgroup kernels and keeps those on a mismatch. Needs S and the block size to be
    multiples of 4 / 2, else unchanged."""

    msa_lean_masks: bool = False
    """flydsl backend on gfx1250: the MSA kernels make their masking decisions per step or per lane
    where a per-element select cannot change the result. The forward, dQ and dK/dV kernels and the
    indexer's scoring kernel skip the per-key causal compare and select on steps / key blocks
    whose keys are all visible to the wave's token(s) -- every step outside a token's own block
    -- behind a wave-uniform scalar branch; the indexer's dK kernel loads the zeros of the WMMA's
    unused K half from an LDS pad instead of selecting them per element (flydsl/
    msa_token_fwd_w32.py, msa_token_bwd_w32.py, index_block_max.py). The skipped selects would
    return their input and the pad holds the same zeros, so the bits are the same; the first call
    for a shape compares every output bit of each kernel with the kernel without it and keeps
    that one on a mismatch."""

    msa_aligned_block_walk: bool = False
    """flydsl backend on gfx1250, with msa_shared_tile_staging: the forward and dQ kernels walk each
    token's selected blocks in an order aligned with its neighbours' (flydsl/block_walk.py), so
    the tokens of a workgroup meet a common block at the same step and stage its K/V tiles once.
    The selection and the terms summed are unchanged; the order of the sums is not (relaxed
    numerics: o, lse, the slot mass and dq change in the last bits; deterministic).
    msa_exact_numerics turns it off."""

    msa_dq_from_ds: bool = False
    """flydsl backend on gfx1250: the MSA backward takes dQ from the dK/dV kernel's dS. That kernel
    already forms dS = P * (dO V^T - delta) for every (token, selected block) as a bf16 WMMA
    operand; it also writes it out (4 KB per pair, a transient buffer), and a slim dQ kernel forms
    dQ = scale * dS K from it: no V staging and no second S / dS computation in the dQ kernel
    (flydsl/msa_token_bwd_w32.py, build_dq_ds_w32). delta = rowsum(dO * O) comes from one Triton
    launch. Relaxed numerics: the same terms, with the dK/dV kernel's dS instead of the dQ kernel's
    own (last-bit differences) and delta summed in another order; deterministic. The first call for
    a shape compares dq / dk / dv with the reference kernels (tolerance 1e-3) and keeps those
    otherwise; dS buffers above 16 GiB keep them too. msa_exact_numerics turns it off."""

    msa_owner_carry: bool = False
    """flydsl backend on gfx1250, with msa_shared_tile_staging: the MSA forward keeps each step's
    tile owner and block id in its loop state, read once when the step's tiles are issued one
    iteration earlier, instead of re-reading the workgroup's table entries in the step (flydsl/
    msa_token_fwd_w32.py, carry_owner). Same operands and arithmetic: the outputs are
    bit-identical, and the first call for a shape compares every output bit with the kernel
    without it and keeps that one on a mismatch. It changes an MSA operator kernel, so it is
    grouped with the relaxed-numerics options: msa_exact_numerics turns it off."""

    msa_index_dk_scatter: bool = False
    """flydsl backend on gfx1250, relaxed numerics: the indexer backward's dK chunk kernel adds
    each plan entry's ``g * q_row`` (fp32) into the one winning key row of an fp32 LDS
    accumulator, entry by entry in plan order, instead of a one-hot masked WMMA product over
    all the block's keys (flydsl/index_block_max.py, _build_index_dk_scatter_w32; one wave per
    chunk, no barrier). The chunk partials go through the same fixed-order reduce. The terms
    are the same; g enters in full fp32 instead of 16 mantissa bits and a key row's terms are
    added in entry order, so d index k differs in its last bits (deterministic). The first
    call for a shape runs the kernel twice and the reference kernel once on that call's
    inputs: identical bits between the two runs and partials within 1e-4 of the reference's
    largest partial, else the reference kernel stays. Used only for sparse_index_dim 128 (4
    elements per lane) and a power-of-two top-k, the one form that was compiled and run on the
    device (index_dk_scatter_supported and the dispatcher in index_dk); any other shape keeps the
    reference kernel. msa_exact_numerics turns it off."""

    msa_ds_chunk_store: bool = False
    """flydsl backend on gfx1250, with msa_dq_from_ds: the dK/dV kernel writes its dS records
    through one buffer descriptor per workgroup (the (batch, KV head) slice of the dS buffer, with
    a per-lane record offset) instead of one descriptor per record, with the entry ids carried in
    the loop state next to their tokens (flydsl/msa_token_bwd_w32.py, ds_chunk_store). The same
    bytes go to the same records, so dq / dk / dv are bit-identical to msa_dq_from_ds without it.
    Used while a slice (S * topk records of 4 KB) is at most 2 GiB (32-bit offsets); larger shapes
    keep the per-record stores. Part of the msa_dq_from_ds path: its first-call comparison runs
    with it, and msa_exact_numerics turns it off."""

    msa_dq_early_issue: bool = False
    """flydsl backend on gfx1250, with msa_dq_from_ds: the slim dQ kernel issues the Tensor Data
    Mover copies of step j+1 before it waits for those of step j, so two steps' copies are in
    flight at a time in the same 2-stage LDS ring (flydsl/msa_token_bwd_w32.py, early_issue; two
    workgroup barriers per step instead of one, a count-based wait). Same tiles, arithmetic and
    order: dq is bit-identical, and the first call for a shape compares every bit of dq with the
    kernel without it and keeps that one on a mismatch. Part of the msa_dq_from_ds path, so
    msa_exact_numerics turns it off."""

    msa_two_tile_slots: bool = False
    """flydsl backend on gfx1250, with msa_owner_carry (and msa_shared_tile_staging): the MSA
    forward keeps two LDS tile slots per wave, one for K and one for V, instead of a 2-stage ring
    of both. A step reads its K tile first and its V tile last, so the step's V is copied while
    the step works on K and the next step's K while it works on V, each after a workgroup barrier
    (flydsl/msa_token_fwd_w32.py, two_tile_slots). A workgroup then takes 68 KB of LDS instead of
    140 KB, and four workgroups instead of two fit a compute unit's 320 KB. Same tiles, arithmetic
    and order: the outputs are bit-identical, and the first call for a shape compares every output
    bit with the kernel without it and keeps that one on a mismatch. An MSA operator kernel
    option: msa_exact_numerics turns it off."""

    msa_walk_one_wave: bool = False
    """flydsl backend on gfx1250, with msa_aligned_block_walk: the walk's Triton kernel is launched
    with one wave per program (2 token groups each) instead of four waves (16 groups), so its
    K x K compares and ranks are not exchanged between waves through LDS and workgroup barriers
    (flydsl/block_walk.py). The same integer kernel: the walk table and slots are identical (the
    first table of a shape is compared with the PyTorch reference; a mismatch keeps the four-wave
    launch), so every output keeps its bits. An option of the aligned walk, which
    msa_exact_numerics turns off."""

    msa_indexer_loss_one_pass: bool = False
    """flydsl backend, with msa_fused_indexer_loss: the slot mass and the sparse indexer KL of an
    MSA layer come from one Triton kernel (it reads the forward's slot_lse / lse, the block table
    and the block scores and writes each row's KL), and their backward from one more
    (indexer_loss_one_pass.py), instead of three ATen launches for the slot mass, the walk-order
    scatter and the bit-exact fused loss's kernels and ATen reductions. The loss is the same
    block-level KL over the selected blocks; relaxed numerics: a row's sums, exp and log are
    Triton's, and with msa_aligned_block_walk a row is summed in walk order (deterministic). The
    first call for a shape runs it twice and compares it with the path it replaces (loss within
    5e-7, block-score gradient within 2e-5, identical bits between the two runs) and keeps that
    path otherwise. msa_exact_numerics turns it off."""

    msa_exact_numerics: bool = False
    """Keep the options of the MSA operator kernels off, whatever their own switches say, so a run
    reproduces the reference flydsl kernels' losses bit for bit (for regression checks). The group
    is attention._MSA_RELAXED: the four relaxed-numerics options, whose arithmetic is not
    bit-identical to the reference kernels (msa_aligned_block_walk, msa_dq_from_ds,
    msa_index_dk_scatter, msa_indexer_loss_one_pass), and the bit-preserving options that are
    built on them or change the same kernels (msa_owner_carry, msa_ds_chunk_store,
    msa_dq_early_issue, msa_two_tile_slots, msa_walk_one_wave). Each of them, or the option it
    is a part of, tests this flag before any work; the Triton fusions and msa_shared_tile_staging
    / msa_lean_masks, which are bit-identical and part of the reference state, ignore it."""

    sparse_indexer_loss_coeff: float = 1.0e-2
    """Weight of the indexer distillation loss. 0.0 leaves the indexer frozen: top-k is not
    differentiable, so this loss is the only gradient its projections ever receive."""

    sparse_indexer_loss_type: str = "sparse"
    """'sparse': KL over each query's selected blocks, against the attention the block-sparse
    layer paid them (DSA's sparse loss). 'dense': KL over every visible block, against dense
    attention; that needs the O(S^2) dense scores, so it is eager-only."""

    def __post_init__(self):
        # Derive the upstream fields BEFORE super(), so TransformerConfig's own
        # validation (bias_activation_fusion vs glu_linear_offset, etc.) sees
        # the values this model actually runs with.
        if self.minimax_sparse_attention:
            self._derive_activation()
            self._derive_normalization()

        super().__post_init__()

        if not self.minimax_sparse_attention:
            return

        if self.multi_latent_attention:
            raise ValueError(
                "minimax_sparse_attention and multi_latent_attention are mutually exclusive: "
                "MSA runs on GQA, and core_transformer_config_from_args replaces the config "
                "class with MLATransformerConfig whenever multi_latent_attention is set."
            )

        if self.msa_backend not in _SUPPORTED_MSA_BACKENDS:
            raise ValueError(
                f"msa_backend={self.msa_backend!r} is not a known backend; "
                f"supported: {sorted(_SUPPORTED_MSA_BACKENDS)}"
            )
        if self.sparse_indexer_loss_type not in _SUPPORTED_INDEXER_LOSS_TYPES:
            raise ValueError(
                f"sparse_indexer_loss_type={self.sparse_indexer_loss_type!r}; "
                f"supported: {sorted(_SUPPORTED_INDEXER_LOSS_TYPES)}"
            )
        if self.msa_backend == "flydsl" and self.sparse_indexer_loss_type == "dense":
            raise ValueError(
                "msa_backend='flydsl' supports sparse_indexer_loss_type='sparse' only: the dense "
                "loss needs the full [b, heads, sq, sk] scores the flydsl kernels never form."
            )

        # The eager backend builds a dense [b, h, sq, sk] mask over the whole key
        # range on every rank, and reads `hidden_states` for the indexer -- which
        # sequence parallel leaves sharded as [sq/tp, b, h] while q/k/v are full
        # length. Neither is handled yet, so refuse loudly instead of silently
        # attending the wrong keys.
        if self.context_parallel_size != 1:
            raise NotImplementedError(
                "MiniMax Sparse Attention does not support context parallelism yet; "
                f"got context_parallel_size={self.context_parallel_size}."
            )
        if self.sequence_parallel:
            raise NotImplementedError(
                "MiniMax Sparse Attention does not support sequence parallelism yet; "
                "set `sequence_parallel: false`."
            )
        # The indexer is replicated and selects blocks for every KV head, while a
        # TP rank holds only num_query_groups / TP of them, so the selection and
        # the local heads would not line up (on either backend).
        if self.tensor_model_parallel_size != 1:
            raise NotImplementedError(
                "MiniMax Sparse Attention does not support tensor parallelism yet; "
                f"got tensor_model_parallel_size={self.tensor_model_parallel_size}. "
                "Shard the model with expert and pipeline parallelism instead."
            )
        # The indexer emits one block selection per GQA group, which is what the
        # reference implementation means by index_n_heads == num_key_value_heads.
        num_query_groups = self.num_query_groups or self.num_attention_heads
        if self.sparse_num_index_heads != num_query_groups:
            raise ValueError(
                f"sparse_num_index_heads ({self.sparse_num_index_heads}) must equal the number of "
                f"GQA groups ({num_query_groups}): the indexer produces one block selection per "
                "group."
            )

        if self.sparse_score_type not in _SUPPORTED_SCORE_TYPES:
            raise NotImplementedError(
                f"sparse_score_type={self.sparse_score_type!r} is not implemented; "
                f"supported: {sorted(_SUPPORTED_SCORE_TYPES)}"
            )

        for name in (
            "sparse_num_index_heads",
            "sparse_index_dim",
            "sparse_block_size",
            "sparse_topk_blocks",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive; got {getattr(self, name)}")

        for name in ("sparse_init_block", "sparse_local_block"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative; got {getattr(self, name)}")

        self.sparse_attention_freq = normalize_sparse_layer_pattern(
            self.sparse_attention_freq, self.num_layers, field_name="sparse_attention_freq"
        )
        self.sparse_disable_index_value = normalize_sparse_layer_pattern(
            self.sparse_disable_index_value,
            self.num_layers,
            field_name="sparse_disable_index_value",
        )

    def _derive_activation(self) -> None:
        """Map `swigluoai` onto Megatron's quick-GEGLU fields.

        The official implementation (``MiniMaxM3VLDenseMLP.forward`` and
        ``MiniMaxM3VLExperts._apply_gate``) is::

            gate, up = gate_up.chunk(2, dim=-1)
            gate = gate.clamp(max=swiglu_limit)
            up   = up.clamp(-swiglu_limit, swiglu_limit)
            out  = (up + 1.0) * gate * sigmoid(gate * swiglu_alpha)

        and Megatron's GLU path (``mlp.py`` / ``moe/experts.py``) is::

            out = activation_func(x_glu) * (x_linear + glu_linear_offset)

        with the same two clamps and the same ``chunk`` split (M3 does *not*
        interleave the halves the way GPT-OSS does -- see that class's own
        comment). With ``activation_func = quick_gelu`` these are the same
        function, so the ``+1`` is ``glu_linear_offset`` and the limit is
        ``activation_func_clamp_value``.

        The one value Megatron cannot carry is a non-default alpha: `quick_gelu`
        hardcodes 1.702. That happens to be M3's `swiglu_alpha`, but a preset
        that changes it would otherwise train a different activation than it
        asked for, so it is rejected.

        Note the import: Megatron defines `quick_gelu` twice, in
        ``core/activations.py`` and in ``core/fusions/fused_bias_geglu.py``.
        The identity checks in ``arguments.py``, ``transformer_config.py``,
        ``mlp.py`` and ``moe/experts.py`` all use the fusions one, so that is
        the object a config actually carries and the one to compare against --
        the two are the same maths but different objects, and `is` tells them
        apart.
        """
        from megatron.core.fusions.fused_bias_geglu import quick_gelu

        if self.activation_func is not quick_gelu:
            raise ValueError(
                "MiniMax-M3 uses the `swigluoai` activation, which Megatron spells "
                "`quick_geglu`. Set `quick_geglu: true` (and `swiglu: false`) in the preset; "
                f"got activation_func={getattr(self.activation_func, '__name__', self.activation_func)}."
            )

        if self.swiglu_alpha != 1.702:
            raise NotImplementedError(
                f"swiglu_alpha={self.swiglu_alpha} is not expressible: Megatron's quick_gelu "
                "hardcodes 1.702. Only M3's released value is supported."
            )

        self.gated_linear_unit = True
        self.glu_linear_offset = 1.0
        self.activation_func_clamp_value = self.swiglu_limit

        # The MoE path fuses quick_geglu (TEGroupedMLP -> weighted_bias_quick_geglu_impl,
        # which applies both clamps), but the dense MLP's fused branch only covers
        # gelu and swiglu: quick_gelu without a per-token scale falls through to
        # `raise ValueError("Only support fusion of gelu and swiglu")` in mlp.py,
        # at forward time. M3 has dense layers, so catch it here instead.
        if self.bias_activation_fusion:
            raise ValueError(
                "MiniMax-M3's swigluoai activation cannot use bias_activation_fusion: the dense "
                "MLP's fused branch handles only gelu and swiglu, and would raise at the first "
                "forward. Set `bias_gelu_fusion: false` in the preset (that is the flag "
                "core_transformer_config_from_args reads when swiglu is off)."
            )

    def _derive_normalization(self) -> None:
        """Map `use_gemma_norm` onto `layernorm_zero_centered_gamma`.

        ``MiniMaxM3VLRMSNorm`` normalises in fp32, scales by ``1.0 + weight``
        and initialises ``weight`` at zeros -- exactly TE's
        ``RMSNorm(zero_centered_gamma=True)``
        (``megatron/core/extensions/transformer_engine.py``). Primus's turbo and
        fused-residual RMSNorm paths read the same flag, so the fused kernels
        stay correct.

        The reference uses that one class for every norm, q/k norms included;
        Megatron's qk norm is ``TENorm`` reading the same flag, so setting it
        once covers them all.
        """
        if not self.use_gemma_norm:
            return

        if self.normalization != "RMSNorm":
            raise ValueError(f"use_gemma_norm requires normalization=RMSNorm; got {self.normalization}.")

        self.layernorm_zero_centered_gamma = True

    @property
    def sparse_layer_pattern(self) -> Tuple[int, ...]:
        """Per-layer 0/1 tuple: 1 where the layer runs MSA, 0 where it stays dense."""
        if not self.minimax_sparse_attention:
            return (0,) * self.num_layers
        return normalize_sparse_layer_pattern(
            self.sparse_attention_freq, self.num_layers, field_name="sparse_attention_freq"
        )
