###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for MiniMax Sparse Attention's eager backend.

The parity test carries a transcription of transformers'
``MiniMaxM3VLIndexer.forward`` / ``build_block_mask`` as its reference, so the
eager backend is pinned against the official implementation rather than against
itself. That matters twice over: it is what a future flydsl backend gets
aligned to, and a parity test written against the *upstream* class rather than
the one the runtime builds is exactly what let the turbo-RMSNorm bug through.
"""

import pytest
import torch

pytest.importorskip("megatron")

from primus.backends.megatron.core.models.minimax_m3.minimax_m3_transformer_config import (
    MSATransformerConfig,
)
from primus.backends.megatron.core.transformer.minimax_m3.eager import (
    build_block_keep,
    eager_block_sparse_attention,
)
from primus.backends.megatron.core.transformer.minimax_m3.indexer import (
    MinimaxM3Indexer,
)
from primus.backends.megatron.core.transformer.minimax_m3.indexer_loss import (
    compute_indexer_loss,
)

BLOCK = 8
TOPK = 2
N_INDEX = 2
SQ = 40


def _config(**overrides):
    """A small MSA config; the shapes are scaled down, the semantics are not."""
    base = dict(
        num_layers=4,
        hidden_size=32,
        num_attention_heads=4,
        num_query_groups=N_INDEX,
        normalization="RMSNorm",
        sparse_num_index_heads=N_INDEX,
        sparse_index_dim=16,
        sparse_block_size=BLOCK,
        sparse_topk_blocks=TOPK,
        sparse_local_block=1,
        sparse_init_block=0,
        use_gemma_norm=False,
        minimax_sparse_attention=False,  # skip the swigluoai/quick_geglu requirement
    )
    return MSATransformerConfig(**{**base, **overrides})


# ---------------------------------------------------------------------------
# Reference: transformers models/minimax_m3_vl/modeling_minimax_m3_vl.py
# ---------------------------------------------------------------------------


def reference_select_blocks(scores, block_size, topk_blocks, local_blocks, position_ids):
    """`MiniMaxM3VLIndexer.forward`, from the per-key scores onwards."""
    batch, num_heads, q_len, k_len = scores.shape
    num_key_blocks = -(-k_len // block_size)
    pad = num_key_blocks * block_size - k_len

    k_positions = torch.arange(k_len, device=scores.device)
    token_future = k_positions[None, None, None, :] > position_ids[:, None, :, None]
    scores = scores.masked_fill(token_future, float("-inf"))
    if pad:
        scores = torch.nn.functional.pad(scores, (0, pad), value=float("-inf"))
    scores = scores.view(batch, num_heads, q_len, num_key_blocks, block_size)
    block_scores = scores.amax(dim=-1)

    q_block = position_ids // block_size
    if local_blocks > 0:
        local = torch.arange(local_blocks, device=scores.device)
        local_idx = (q_block[..., None] - local.view(1, 1, -1)).clamp(min=0)
        local_idx = local_idx.unsqueeze(1).expand(-1, num_heads, -1, -1)
        block_scores.scatter_(-1, local_idx, float("inf"))

    topk = min(topk_blocks, num_key_blocks)
    topk_scores, topk_indices = block_scores.topk(topk, dim=-1)
    return topk_indices.masked_fill(topk_scores == float("-inf"), -1), block_scores


def reference_block_keep(block_indices, key_length, block_size):
    """The keep half of `MiniMaxM3VLIndexer.build_block_mask`."""
    batch, n_idx_heads, q_len, _ = block_indices.shape
    num_key_blocks = -(-key_length // block_size)
    safe = block_indices.masked_fill(block_indices < 0, num_key_blocks)
    bias = block_indices.new_full(
        (batch, n_idx_heads, q_len, num_key_blocks + 1), float("-inf"), dtype=torch.float32
    )
    bias.scatter_(-1, safe, 0.0)
    bias = bias[..., :num_key_blocks]
    return (bias == 0.0).repeat_interleave(block_size, dim=-1)[..., :key_length]


# ---------------------------------------------------------------------------
# Block selection
# ---------------------------------------------------------------------------


def _scores(seed=0):
    torch.manual_seed(seed)
    return torch.randn(1, N_INDEX, SQ, SQ)


def _select(config, scores):
    """Run just the selection half of the indexer, without building the module."""
    return MinimaxM3Indexer.select_blocks(_FakeIndexer(config), scores)


class _FakeIndexer:
    """The selection logic reads only these attributes."""

    def __init__(self, config):
        self.block_size = config.sparse_block_size
        self.topk_blocks = config.sparse_topk_blocks
        self.init_blocks = config.sparse_init_block
        self.local_blocks = config.sparse_local_block

    _boost_always_visible = MinimaxM3Indexer._boost_always_visible


def test_block_selection_matches_the_reference():
    config = _config()
    scores = _scores()
    position_ids = torch.arange(SQ).unsqueeze(0)

    indices, _ = _select(config, scores)
    ref_indices, _ = reference_select_blocks(
        scores.clone(), BLOCK, TOPK, config.sparse_local_block, position_ids
    )

    assert torch.equal(indices.sort(dim=-1).values, ref_indices.sort(dim=-1).values)


def test_block_keep_matches_the_reference():
    config = _config()
    indices, _ = _select(config, _scores())

    keep = build_block_keep(indices, SQ, BLOCK)
    ref_keep = reference_block_keep(indices, SQ, BLOCK)

    assert torch.equal(keep, ref_keep)


def test_every_query_can_see_its_own_block():
    """The local block is force-selected, so a query is never left with nothing."""
    keep, _ = _select(_config(), _scores())
    keep = build_block_keep(keep, SQ, BLOCK)

    for q in range(SQ):
        assert keep[0, :, q, q].all(), f"query {q} cannot attend itself"


def test_selection_is_causal_and_deduplicated():
    indices, _ = _select(_config(), _scores())

    for q in range(SQ):
        for head in range(N_INDEX):
            picked = [int(i) for i in indices[0, head, q] if i >= 0]
            assert len(picked) == len(set(picked)), "a block was selected twice"
            assert max(picked) <= q // BLOCK, "a future block was selected"


def test_init_blocks_are_always_selected():
    config = _config(sparse_init_block=1, sparse_topk_blocks=2)
    indices, _ = _select(config, _scores())

    # Every query past the first block must keep block 0 alive.
    for q in range(BLOCK, SQ):
        assert 0 in indices[0, :, q].tolist()[0] or 0 in indices[0, :, q].tolist()[1]


# ---------------------------------------------------------------------------
# Eager attention
# ---------------------------------------------------------------------------


def _qkv(n_q=4, n_kv=N_INDEX, dim=8, seed=1):
    torch.manual_seed(seed)
    q = torch.randn(1, n_q, SQ, dim, dtype=torch.float64)
    k = torch.randn(1, n_kv, SQ, dim, dtype=torch.float64)
    v = torch.randn(1, n_kv, SQ, dim, dtype=torch.float64)
    return q, k, v


def test_full_topk_reproduces_dense_causal_attention():
    """With every block selectable, MSA must collapse onto dense causal attention."""
    n_blocks = -(-SQ // BLOCK)
    config = _config(sparse_topk_blocks=n_blocks)
    indices, _ = _select(config, _scores())
    keep = build_block_keep(indices, SQ, BLOCK)

    q, k, v = _qkv()
    out, dense_scores = eager_block_sparse_attention(q, k, v, keep, softmax_scale=0.5)

    reference = torch.matmul(torch.softmax(dense_scores, dim=-1).to(v.dtype), _repeat(v, 2))
    torch.testing.assert_close(out, reference)


def _repeat(x, n_rep):
    b, n, s, d = x.shape
    return x[:, :, None].expand(b, n, n_rep, s, d).reshape(b, n * n_rep, s, d)


def test_attention_is_confined_to_selected_blocks():
    config = _config()
    indices, _ = _select(config, _scores())
    keep = build_block_keep(indices, SQ, BLOCK)

    q, k, v = _qkv()
    # Corrupt the values outside the selection; the output must not move.
    out_a, _ = eager_block_sparse_attention(q, k, v, keep, softmax_scale=0.5)
    v_b = v.clone()
    dropped = ~keep[:, :, :, :]
    # Any key dropped by *every* query is safe to perturb wholesale.
    never_read = dropped.all(dim=2)[0]
    for head in range(v.shape[1]):
        v_b[0, head, never_read[head]] += 100.0
    out_b, _ = eager_block_sparse_attention(q, k, v_b, keep, softmax_scale=0.5)

    torch.testing.assert_close(out_a, out_b)


# ---------------------------------------------------------------------------
# Indexer loss -- the thing that makes MSA trainable
# ---------------------------------------------------------------------------


def test_indexer_loss_is_a_finite_scalar_and_carries_gradient():
    torch.manual_seed(2)
    dense_scores = torch.randn(1, 4, SQ, SQ, dtype=torch.float64)
    block_scores = torch.randn(1, N_INDEX, SQ, -(-SQ // BLOCK), dtype=torch.float64)
    block_scores.requires_grad_(True)

    loss = compute_indexer_loss(dense_scores, block_scores, BLOCK, loss_coeff=1.0)

    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert block_scores.grad is not None
    assert block_scores.grad.abs().sum() > 0, "the indexer would get no training signal"


def test_indexer_loss_survives_the_always_visible_infinities():
    """`+inf` boosts would make a plain softmax NaN."""
    torch.manual_seed(3)
    dense_scores = torch.randn(1, 4, SQ, SQ, dtype=torch.float64)
    block_scores = torch.randn(1, N_INDEX, SQ, -(-SQ // BLOCK), dtype=torch.float64)
    block_scores[..., 0] = float("inf")
    block_scores[..., -1] = float("-inf")

    loss = compute_indexer_loss(dense_scores, block_scores, BLOCK, loss_coeff=1.0)

    assert torch.isfinite(loss)


def _causal_dense_scores(seed=5):
    torch.manual_seed(seed)
    scores = torch.randn(1, 4, SQ, SQ, dtype=torch.float64)
    future = torch.ones(SQ, SQ, dtype=torch.bool).triu(1)
    return scores.masked_fill(future, float("-inf"))


def test_indexer_loss_is_finite_on_real_causal_selections():
    """Queries in the first block see only their forced local block: no finite score.

    Random block scores never produce such a row; the indexer's own causal
    selection does, for every query in block 0.
    """
    _, block_scores = _select(_config(), _scores())
    assert not torch.isfinite(block_scores[..., :BLOCK, :]).any(), "premise: block 0 rows are all forced"
    block_scores = block_scores.double().requires_grad_(True)

    loss = compute_indexer_loss(_causal_dense_scores(), block_scores, BLOCK, loss_coeff=1.0)

    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(block_scores.grad).all()
    assert block_scores.grad.abs().sum() > 0


def test_indexer_loss_ignores_fully_masked_padding_rows():
    _, block_scores = _select(_config(), _scores())
    dense_scores = _causal_dense_scores()
    dense_scores[:, :, -3:, :] = float("-inf")  # three padding queries

    loss = compute_indexer_loss(dense_scores, block_scores.double(), BLOCK, loss_coeff=1.0)

    assert torch.isfinite(loss)


def test_zero_coefficient_scales_the_loss_away():
    torch.manual_seed(4)
    dense_scores = torch.randn(1, 4, SQ, SQ, dtype=torch.float64)
    block_scores = torch.randn(1, N_INDEX, SQ, -(-SQ // BLOCK), dtype=torch.float64)

    assert compute_indexer_loss(dense_scores, block_scores, BLOCK, loss_coeff=0.0).item() == 0.0


# ---------------------------------------------------------------------------
# Config guards
# ---------------------------------------------------------------------------


def _msa_config(**overrides):
    """A config with MSA actually enabled, so the guards run."""
    from megatron.core.fusions.fused_bias_geglu import quick_gelu

    base = dict(
        num_layers=4,
        hidden_size=32,
        num_attention_heads=4,
        num_query_groups=N_INDEX,
        normalization="RMSNorm",
        activation_func=quick_gelu,
        gated_linear_unit=True,
        sparse_num_index_heads=N_INDEX,
    )
    return MSATransformerConfig(**{**base, **overrides})


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="msa_backend"):
        _msa_config(msa_backend="cutlass")


def test_context_parallel_is_rejected():
    with pytest.raises(NotImplementedError, match="context parallelism"):
        _msa_config(context_parallel_size=2)


def test_sequence_parallel_is_rejected():
    with pytest.raises(NotImplementedError, match="sequence parallelism"):
        _msa_config(tensor_model_parallel_size=2, sequence_parallel=True)


def test_index_heads_must_match_the_gqa_groups():
    with pytest.raises(ValueError, match="one block selection per group"):
        _msa_config(sparse_num_index_heads=N_INDEX + 1)


def test_flydsl_is_declared_but_not_implemented():
    """The config accepts it; constructing the module is what refuses."""
    config = _msa_config(msa_backend="flydsl")
    assert config.msa_backend == "flydsl"

    from primus.backends.megatron.core.transformer.minimax_m3 import (
        MSA_BACKENDS,
        MinimaxSparseAttention,
    )

    assert "flydsl" in MSA_BACKENDS
    with pytest.raises(NotImplementedError, match="flydsl"):
        MinimaxSparseAttention(config, None, layer_number=1)


# ---------------------------------------------------------------------------
# Building the real module
#
# Everything above exercises the maths through helpers. This section builds
# MinimaxSparseAttention the way the runtime does -- real TE specs, real
# parallel state -- and runs a forward and a backward through it. The
# helper-level tests cannot catch a wrong `build_module` call (a missing
# `skip_weight_param_allocation` got through them once); this can.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TE specs need a GPU")
class TestMinimaxSparseAttentionModule:
    @pytest.fixture(autouse=True)
    def _parallel(self, init_parallel_state):
        pass

    @staticmethod
    def _build(**overrides):
        from megatron.core.models.gpt.gpt_layer_specs import (
            get_gpt_layer_with_transformer_engine_spec,
        )

        from primus.backends.megatron.core.transformer.minimax_m3 import (
            MinimaxSparseAttention,
            MinimaxSparseAttentionSubmodules,
        )
        from primus.backends.megatron.patches.minimax_m3_patches import (
            _widen_submodules,
        )

        config = _msa_config(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            kv_channels=16,
            sparse_index_dim=16,
            sparse_block_size=BLOCK,
            sparse_topk_blocks=TOPK,
            params_dtype=torch.float32,
            bf16=False,
            **overrides,
        )
        upstream = get_gpt_layer_with_transformer_engine_spec(qk_layernorm=True)
        attn_spec = upstream.submodules.self_attention
        submodules = _widen_submodules(attn_spec.submodules)
        assert isinstance(submodules, MinimaxSparseAttentionSubmodules)

        return config, MinimaxSparseAttention(config, submodules, layer_number=1).cuda()

    @staticmethod
    def _rope(config, rotary_percent=0.5):
        """The freqs GPTModel hands the decoder: [sq, 1, 1, rotary_dim]."""
        from megatron.core.models.common.embeddings.rotary_pos_embedding import (
            RotaryEmbedding,
        )

        rope = RotaryEmbedding(config.kv_channels, rotary_percent=rotary_percent)
        return rope(SQ).cuda()

    @pytest.mark.parametrize("with_rope", [False, True])
    def test_builds_and_runs_forward(self, with_rope):
        config, attention = self._build()
        hidden = torch.randn(SQ, 2, config.hidden_size, device="cuda")
        rotary_pos_emb = self._rope(config) if with_rope else None

        output, bias = attention(hidden, rotary_pos_emb=rotary_pos_emb)

        assert output.shape == (SQ, 2, config.hidden_size)
        assert torch.isfinite(output).all()

    def test_partial_rope_rotates_only_the_leading_dims(self):
        """M3 is rotary_percent 0.5, so half of each head must pass through."""
        config, attention = self._build()
        rotary_pos_emb = self._rope(config)

        assert rotary_pos_emb.shape[-1] == config.kv_channels // 2
        hidden = torch.randn(SQ, 2, config.hidden_size, device="cuda")
        output, _ = attention(hidden, rotary_pos_emb=rotary_pos_emb)
        assert torch.isfinite(output).all()

    def test_indexer_gets_gradients_from_the_distillation_loss(self):
        """Without the loss the indexer never moves; this is that contract."""
        config, attention = self._build(sparse_indexer_loss_coeff=1.0e-2)
        attention.train()
        hidden = torch.randn(SQ, 2, config.hidden_size, device="cuda", requires_grad=True)

        output, _ = attention(hidden, rotary_pos_emb=self._rope(config))
        output.sum().backward()

        for name in ("linear_index_q", "linear_index_k"):
            grad = getattr(attention.indexer, name).weight.grad
            assert grad is not None, f"{name} received no gradient"
            assert grad.abs().sum() > 0, f"{name}'s gradient is all zero"

    def test_zero_coefficient_leaves_the_indexer_without_gradients(self):
        config, attention = self._build(sparse_indexer_loss_coeff=0.0)
        attention.train()
        hidden = torch.randn(SQ, 2, config.hidden_size, device="cuda")

        output, _ = attention(hidden)
        output.sum().backward()

        # top-k is not differentiable, so with the loss off nothing reaches the
        # indexer -- which is exactly why the loss has to exist.
        assert attention.indexer.linear_index_q.weight.grad is None

    def test_indexer_reads_the_normalised_attention_input(self):
        """The TE spec fuses input_layernorm into linear_qkv, so the module gets the raw
        residual stream -- but the reference indexer reads input_layernorm's output."""
        config, attention = self._build()
        assert config.layernorm_zero_centered_gamma, "premise: M3's gemma norm, weight is (1 + w)"
        with torch.no_grad():
            attention.linear_qkv.layer_norm_weight.normal_(0.0, 0.2)

        seen = {}

        def record_input(_module, args):
            seen["x"] = args[0]

        attention.indexer.register_forward_pre_hook(record_input)
        hidden = torch.randn(SQ, 2, config.hidden_size, device="cuda")
        attention(hidden)

        weight = 1.0 + attention.linear_qkv.layer_norm_weight.float()
        rms = torch.rsqrt(hidden.pow(2).mean(-1, keepdim=True) + config.layernorm_epsilon)
        torch.testing.assert_close(seen["x"], hidden * rms * weight)
        assert not seen["x"].requires_grad

    def test_distillation_loss_does_not_reach_the_layers_below(self):
        """The KL trains the indexer only: the input gradient must not change with it."""
        config, attention = self._build(sparse_indexer_loss_coeff=1.0)
        attention.train()
        hidden = torch.randn(SQ, 2, config.hidden_size, device="cuda")
        rotary_pos_emb = self._rope(config)

        def input_grad(coeff):
            attention.indexer_loss_coeff = coeff
            attention.zero_grad(set_to_none=True)
            x = hidden.clone().requires_grad_(True)
            output, _ = attention(x, rotary_pos_emb=rotary_pos_emb)
            output.sum().backward()
            return x.grad

        torch.testing.assert_close(input_grad(1.0), input_grad(0.0))

    def test_no_loss_is_computed_without_grad(self):
        """Full recompute's first forward runs under no_grad; it must not record a loss."""
        from megatron.core.transformer.moe.moe_utils import (
            get_moe_layer_wise_logging_tracker,
        )

        from primus.backends.megatron.core.transformer.minimax_m3 import (
            MSA_INDEXER_LOSS_NAME,
        )

        config, attention = self._build(sparse_indexer_loss_coeff=1.0e-2)
        attention.train()
        tracker = get_moe_layer_wise_logging_tracker()
        tracker.pop(MSA_INDEXER_LOSS_NAME, None)

        with torch.no_grad():
            attention(torch.randn(SQ, 2, config.hidden_size, device="cuda"))
        assert MSA_INDEXER_LOSS_NAME not in tracker

    def test_loss_is_recorded_in_the_moe_aux_loss_tracker(self):
        from megatron.core.transformer.moe.moe_utils import (
            get_moe_layer_wise_logging_tracker,
        )

        from primus.backends.megatron.core.transformer.minimax_m3 import (
            MSA_INDEXER_LOSS_NAME,
        )

        config, attention = self._build(sparse_indexer_loss_coeff=1.0e-2)
        attention.train()
        tracker = get_moe_layer_wise_logging_tracker()
        tracker.pop(MSA_INDEXER_LOSS_NAME, None)

        attention(torch.randn(SQ, 2, config.hidden_size, device="cuda"))

        values = tracker.pop(MSA_INDEXER_LOSS_NAME)["values"]
        assert values.shape == (config.num_layers,)
        # layer_number=1 writes slot 0 and nothing else.
        assert values[0] > 0
        assert (values[1:] == 0).all()


class TestIndexerLossScale:
    """The seeded gradient must follow the MoE aux-loss scale the schedule installs."""

    @pytest.fixture(autouse=True)
    def _restore_scales(self):
        from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler

        from primus.backends.megatron.core.transformer.minimax_m3 import (
            MSAIndexerLossAutoScaler,
        )

        saved_moe = MoEAuxLossAutoScaler.main_loss_backward_scale
        saved_msa = MSAIndexerLossAutoScaler.main_loss_backward_scale
        yield
        MoEAuxLossAutoScaler.main_loss_backward_scale = saved_moe
        MSAIndexerLossAutoScaler.main_loss_backward_scale = saved_msa

    @staticmethod
    def _seeded_grad():
        from primus.backends.megatron.core.transformer.minimax_m3 import (
            MSAIndexerLossAutoScaler,
        )

        weight = torch.ones(3, requires_grad=True)
        activation = torch.zeros(2, requires_grad=True)
        out = MSAIndexerLossAutoScaler.apply(activation, (weight * 2.0).sum())
        out.sum().backward()
        return weight.grad

    def test_follows_the_moe_aux_loss_scale(self):
        from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler

        from primus.backends.megatron.core.transformer.minimax_m3 import (
            MSAIndexerLossAutoScaler,
        )

        MSAIndexerLossAutoScaler.set_loss_scale(None)
        # What the schedule installs with 4 microbatches and no grad scaler.
        MoEAuxLossAutoScaler.main_loss_backward_scale = torch.tensor(0.25)

        torch.testing.assert_close(self._seeded_grad(), torch.full((3,), 0.5))

    def test_explicit_override_wins(self):
        from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler

        from primus.backends.megatron.core.transformer.minimax_m3 import (
            MSAIndexerLossAutoScaler,
        )

        MoEAuxLossAutoScaler.main_loss_backward_scale = torch.tensor(0.25)
        MSAIndexerLossAutoScaler.set_loss_scale(torch.tensor(1.0))

        torch.testing.assert_close(self._seeded_grad(), torch.full((3,), 2.0))


def test_external_attention_mask_is_composed():
    """Megatron's dataloader mask (True = do not attend) must be honoured."""
    config = _config()
    indices, _ = _select(config, _scores())
    keep = build_block_keep(indices, SQ, BLOCK)
    q, k, v = _qkv()

    # Forbid the second half of the keys outright.
    external = torch.zeros(1, 1, SQ, SQ, dtype=torch.bool)
    external[..., SQ // 2 :] = True

    _, dense_scores = eager_block_sparse_attention(q, k, v, keep, softmax_scale=0.5, attention_mask=external)

    assert torch.isneginf(dense_scores[..., SQ // 2 :]).all()


def test_fully_masked_query_yields_zeros_not_nan():
    """A padding row can be masked everywhere; softmax would return NaN."""
    config = _config()
    indices, _ = _select(config, _scores())
    keep = build_block_keep(indices, SQ, BLOCK)
    q, k, v = _qkv()

    external = torch.zeros(1, 1, SQ, SQ, dtype=torch.bool)
    external[:, :, 0, :] = True  # query 0 may attend nothing

    out, _ = eager_block_sparse_attention(q, k, v, keep, softmax_scale=0.5, attention_mask=external)

    assert torch.isfinite(out).all()
    assert (out[:, :, 0] == 0).all()
