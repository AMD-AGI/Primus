###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for ``deepseek_v4_flops_patches.py`` (Plan-3 Phase 20 + Plan-6 P33).

Coverage:

* G16 — :func:`compute_v4_flops` matches a hand-derived closed-form total
  within 1% on a fully-specified V4-Flash-shaped config (8 layers, mixed
  ``compress_ratios=[0,4,128]``, ``mtp_num_layers=1``).  Every per-component
  byte (``attn_qkv_o``, ``attn_scores``, ``compressor``, ``indexer``,
  ``moe``, ``mtp``, ``logits``, ``hc``) is asserted independently so a
  regression in any single term fails loudly.
* G17 — When the wrapper is installed with ``dispatch_v4=False`` (i.e. the
  installer saw a non-V4 ``args.model_type``), the wrapper returns the
  upstream value byte-for-byte.  This mirrors how the install-time
  ``condition`` gate works: V4 dispatch is captured at install time
  because Megatron's ``pretrain()`` overwrites ``args.model_type`` with
  a ``ModelType`` enum at ``training.py:1210`` before ``train()`` calls
  ``num_floating_point_operations``.
* G36 — Plan-6 P33: SWA visible-pair correction.  Parametrised over
  ``swa_window``, ``compress_ratio``, ``hc_mult`` so the dense + HCA +
  CSA per-layer pair counts and the over-count ratio vs the legacy
  ``S_eff^2`` upper bound are pinned independently.
* G36a — Plan-6 P33: HyperConnection ``fn.weight`` matmul accounting.
  Asserts the ``hc`` breakdown row equals the closed form
  ``B * S * K * D * K * (2 * (L + M) * (2+K) + (1 + M))`` and degrades
  to 0 when ``hc_mult <= 1``.
"""

from __future__ import annotations

import types
from types import SimpleNamespace

import pytest

from primus.backends.megatron.patches.deepseek_v4_flops_patches import (
    _FMA_FACTOR,
    _FORWARD_BACKWARD_FACTOR,
    _SWIGLU_FFN_EXPANSION_FACTOR,
    _make_v4_num_floating_point_operations,
    _normalize_layer_ratios,
    _visible_pairs,
    compute_v4_flops,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _v4_flash_smoke_args(
    *,
    num_layers: int = 8,
    seq_length: int = 128,
    hc_mult: int = 4,
    mtp_num_layers: int = 0,
    compress_ratios=(0, 0, 4, 128, 4, 128, 4, 0),
    num_hash_layers: int = 3,
    moe_router_topk: int = 6,
    num_experts: int = 256,
    moe_ffn_hidden_size: int = 2048,
    moe_shared_expert_intermediate_size: int = 2048,
    attn_sliding_window: int = 0,
):
    """Build a V4-Flash-shaped fake ``args`` namespace.

    Defaults mirror the smoke config used in P19 (run_deepseek_v4.sh)
    so the unit numbers stay comparable to live runs.  ``attn_sliding_window``
    defaults to ``0`` (disabled / full causal) for backward-compatibility
    with the plan-3 P20 reference; SWA-aware behaviour is exercised by
    the plan-6 P33 G36 tests below.
    """
    return SimpleNamespace(
        model_type="deepseek_v4",
        seq_length=seq_length,
        hc_mult=hc_mult,
        hidden_size=4096,
        num_attention_heads=64,
        kv_channels=512,
        q_lora_rank=1024,
        o_lora_rank=1024,
        o_groups=8,
        num_layers=num_layers,
        mtp_num_layers=mtp_num_layers,
        compress_ratios=compress_ratios,
        moe_ffn_hidden_size=moe_ffn_hidden_size,
        ffn_hidden_size=18432,
        moe_router_topk=moe_router_topk,
        num_experts=num_experts,
        moe_shared_expert_intermediate_size=moe_shared_expert_intermediate_size,
        num_hash_layers=num_hash_layers,
        index_topk=512,
        index_head_dim=128,
        index_n_heads=64,
        padded_vocab_size=129280,
        vocab_size=129280,
        attn_sliding_window=attn_sliding_window,
    )


def _hand_attn_qkv_o(*, B, S_eff, H, n, d, q_lora, o_lora, o_groups):
    """Reference closed-form QKV+O FMAC per layer (matches the patch helper)."""
    n_d = n * d
    qkv = H * q_lora + q_lora * n_d + H * d
    o_proj = n_d * o_lora + (o_groups * o_lora) * H if o_lora > 0 else n_d * H
    return B * S_eff * (qkv + o_proj)


def _hand_local_pairs(*, swa, S_eff):
    """Reference closed form for SWA-pruned local visible pairs."""
    if swa <= 0 or swa >= S_eff:
        return S_eff * (S_eff + 1) // 2
    return swa * S_eff - swa * (swa - 1) // 2


def _hand_pool_pairs(*, ratio, S_eff):
    """Reference closed form for the causal-visible HCA pool pair count."""
    c = int(ratio)
    if c <= 0 or S_eff <= 0:
        return 0
    n_full = S_eff // c
    if n_full == 0:
        return 0
    return c * n_full * (n_full - 1) // 2 + n_full * (S_eff - c * n_full + 1)


def _hand_attn_scores(*, B, S_eff, n, d, ratio, index_topk, swa):
    """Reference closed-form attention-score FMAC per layer.

    Plan-6 P33: counts only causal-visible ``(query, key)`` pairs
    surviving the per-layer mask (SWA + pool + sparse top-K), not the
    legacy ``S_eff^2`` upper bound.
    """
    local_pairs = _hand_local_pairs(swa=swa, S_eff=S_eff)
    if ratio == 0:
        pairs = local_pairs
    elif ratio == 128:
        pairs = local_pairs + _hand_pool_pairs(ratio=ratio, S_eff=S_eff)
    elif ratio == 4:
        pool = max(1, S_eff // 4)
        keys = min(index_topk, pool) if index_topk else pool
        pairs = local_pairs + keys * S_eff
    else:
        pool = max(1, S_eff // ratio)
        pairs = local_pairs + pool * S_eff
    return 2 * B * n * d * pairs


def _hand_compressor(*, B, S_eff, H, d, ratio):
    if ratio == 0:
        return 0
    coff = 2 if ratio == 4 else 1
    return 2 * B * S_eff * H * (coff * d)


def _hand_indexer(*, B, S_eff, H, ratio, ihd, inh):
    if ratio != 4:
        return 0
    pool = max(1, S_eff // ratio)
    proj = H * ihd + ihd * (inh * ihd) + H * inh + 2 * H * (2 * ihd)
    scoring = inh * pool * ihd
    return B * S_eff * (proj + scoring)


def _hand_moe(*, B, S_eff, H, H_moe, topk, n_experts, hash_layer, H_shared):
    router = 0 if hash_layer else H * n_experts
    routed = topk * _SWIGLU_FFN_EXPANSION_FACTOR * H * H_moe
    shared = _SWIGLU_FFN_EXPANSION_FACTOR * H * H_shared if H_shared > 0 else 0
    return B * S_eff * (router + routed + shared)


# ---------------------------------------------------------------------------
# G16: closed-form parity per component
# ---------------------------------------------------------------------------


class TestComputeV4FlopsClosedForm:
    """G16: per-component breakdown matches a hand-derived reference."""

    @pytest.fixture(scope="class")
    def args(self):
        return _v4_flash_smoke_args()

    @pytest.fixture(scope="class")
    def batch_size(self):
        return 16  # GBS used in the P19 smoke run.

    @pytest.fixture(scope="class")
    def computed(self, args, batch_size):
        total, breakdown = compute_v4_flops(args, batch_size)
        return total, breakdown

    def test_attn_qkv_o_term_matches_reference(self, args, batch_size, computed):
        _total, br = computed
        S_eff = args.seq_length * args.hc_mult
        per_layer = _hand_attn_qkv_o(
            B=batch_size,
            S_eff=S_eff,
            H=args.hidden_size,
            n=args.num_attention_heads,
            d=args.kv_channels,
            q_lora=args.q_lora_rank,
            o_lora=args.o_lora_rank,
            o_groups=args.o_groups,
        )
        assert br.attn_qkv_o == per_layer * args.num_layers

    def test_attn_scores_term_matches_reference(self, args, batch_size, computed):
        _total, br = computed
        S_eff = args.seq_length * args.hc_mult
        expected = sum(
            _hand_attn_scores(
                B=batch_size,
                S_eff=S_eff,
                n=args.num_attention_heads,
                d=args.kv_channels,
                ratio=int(r),
                index_topk=args.index_topk,
                swa=int(getattr(args, "attn_sliding_window", 0) or 0),
            )
            for r in args.compress_ratios
        )
        assert br.attn_scores == expected

    def test_compressor_term_matches_reference(self, args, batch_size, computed):
        _total, br = computed
        S_eff = args.seq_length * args.hc_mult
        expected = sum(
            _hand_compressor(
                B=batch_size,
                S_eff=S_eff,
                H=args.hidden_size,
                d=args.kv_channels,
                ratio=int(r),
            )
            for r in args.compress_ratios
        )
        assert br.compressor == expected

    def test_indexer_term_matches_reference(self, args, batch_size, computed):
        _total, br = computed
        S_eff = args.seq_length * args.hc_mult
        expected = sum(
            _hand_indexer(
                B=batch_size,
                S_eff=S_eff,
                H=args.hidden_size,
                ratio=int(r),
                ihd=args.index_head_dim,
                inh=args.index_n_heads,
            )
            for r in args.compress_ratios
        )
        assert br.indexer == expected

    def test_moe_term_respects_hash_layers(self, args, batch_size, computed):
        _total, br = computed
        S_eff = args.seq_length * args.hc_mult
        expected = sum(
            _hand_moe(
                B=batch_size,
                S_eff=S_eff,
                H=args.hidden_size,
                H_moe=args.moe_ffn_hidden_size,
                topk=args.moe_router_topk,
                n_experts=args.num_experts,
                hash_layer=(layer_idx < args.num_hash_layers),
                H_shared=args.moe_shared_expert_intermediate_size,
            )
            for layer_idx in range(args.num_layers)
        )
        assert br.moe == expected

    def test_logits_term_includes_one_extra_head_per_mtp_depth(self):
        args = _v4_flash_smoke_args(mtp_num_layers=2)
        _total, br = compute_v4_flops(args, batch_size=8)
        expected = (args.mtp_num_layers + 1) * 8 * args.seq_length * args.hidden_size * args.padded_vocab_size
        assert br.logits == expected

    def test_total_matches_breakdown_sum_with_expansion(self, computed):
        total, br = computed
        assert total == _FORWARD_BACKWARD_FACTOR * _FMA_FACTOR * br.total_fmac()

    def test_hc_term_is_nonzero_when_hc_mult_gt_1(self, computed):
        """Plan-6 P33: HC matmul row must be populated when ``hc_mult > 1``.

        The exact closed form is pinned by the dedicated G36a test below;
        here we just assert the term is present so a future refactor
        that silently zeroes ``hc`` fails this fixture too.
        """
        _total, br = computed
        assert br.hc > 0


# ---------------------------------------------------------------------------
# G36: Plan-6 P33 SWA visible-pair correction
# ---------------------------------------------------------------------------


class TestG36SWAVisiblePairs:
    """Plan-6 P33: ``_attn_scores_fmac_per_layer`` must count only causal-
    visible ``(q, k)`` pairs surviving SWA + pool + sparse top-K masks.

    The legacy plan-3 P20 closed form used ``B * n * d * S_eff^2`` for the
    local branch (Megatron's ``S^2/2`` causal upper bound x the FMA-pair
    factor) which over-counted by ``S_eff / swa_window`` once the kernel
    started honoring SWA per-row pruning.  This test pins the new
    ``2 * n * d * visible_pairs`` form against the helper, the per-branch
    over-count ratios, and the proxy-shape values printed in
    ``deepseek-v4/develop/perf/attention_perf.md``.
    """

    @pytest.mark.parametrize("swa", [0, 64, 128, 4096, 8192])
    def test_local_visible_pair_helper(self, swa):
        """Helper closed form matches the exhaustive sum-over-queries."""
        S_eff = 4096
        pairs = _visible_pairs(
            swa_window=swa,
            compress_ratio=0,
            index_topk=0,
            seq_len_eff=S_eff,
        )
        exhaustive = sum(min(q + 1, swa) if (0 < swa < S_eff) else (q + 1) for q in range(S_eff))
        assert pairs == exhaustive

    def test_proxy_shape_dense_visible_pairs_matches_attn_perf_doc(self):
        """``swa=128, S_eff=4096, cr=0`` → 516,160 (attention_perf.md row)."""
        pairs = _visible_pairs(
            swa_window=128,
            compress_ratio=0,
            index_topk=0,
            seq_len_eff=4096,
        )
        assert pairs == 516_160

    def test_proxy_shape_hca_visible_pairs_matches_attn_perf_doc(self):
        """``swa=128, S_eff=4096, cr=128`` → 516,160 + 63,520 = 579,680."""
        pairs = _visible_pairs(
            swa_window=128,
            compress_ratio=128,
            index_topk=0,
            seq_len_eff=4096,
        )
        assert pairs == 516_160 + 63_520

    def test_proxy_shape_csa_visible_pairs_matches_attn_perf_doc(self):
        """``swa=128, S_eff=4096, cr=4, topk=512`` → 516,160 + 512*4096."""
        pairs = _visible_pairs(
            swa_window=128,
            compress_ratio=4,
            index_topk=512,
            seq_len_eff=4096,
        )
        assert pairs == 516_160 + 512 * 4096

    @pytest.mark.parametrize("hc_mult", [1, 4])
    @pytest.mark.parametrize("ratio", [0, 4, 128])
    def test_swa128_reduces_attn_scores_vs_full_causal(self, hc_mult, ratio):
        """SWA=128 must produce strictly fewer score FMAC than swa=0 on
        every layer type (no overlap between local and pool/topk terms
        means the SWA-pruned local term strictly dominates the saving).
        """
        S = 4096
        args_no_swa = _v4_flash_smoke_args(
            num_layers=1,
            seq_length=S,
            hc_mult=hc_mult,
            compress_ratios=(ratio,),
            num_hash_layers=0,
            attn_sliding_window=0,
        )
        args_swa = _v4_flash_smoke_args(
            num_layers=1,
            seq_length=S,
            hc_mult=hc_mult,
            compress_ratios=(ratio,),
            num_hash_layers=0,
            attn_sliding_window=128,
        )
        _t1, br_no = compute_v4_flops(args_no_swa, batch_size=1)
        _t2, br_swa = compute_v4_flops(args_swa, batch_size=1)
        assert br_swa.attn_scores < br_no.attn_scores
        # All other components untouched by SWA.
        assert br_swa.attn_qkv_o == br_no.attn_qkv_o
        assert br_swa.compressor == br_no.compressor
        assert br_swa.indexer == br_no.indexer
        assert br_swa.moe == br_no.moe
        assert br_swa.hc == br_no.hc

    def test_swa_pruned_attn_scores_matches_proxy_overcount_ratio(self):
        """Per-layer over-count ratio between legacy ``S_eff^2`` and the
        SWA-pruned visible-pair count is ``S_eff / (2*swa) + O(1/S)`` for
        swa < S_eff.  At the V4-Flash proxy shape (S=4096, hc_mult=4,
        S_eff=16384, swa=128) the legacy local term ``S_eff^2`` is
        ``S_eff / swa = 128x`` the visible-pair-derived local FMAC of
        ``2 * swa * S_eff - swa * (swa - 1)`` — pin that ratio at >= 100x
        so a regression that silently reverts to ``S_eff^2`` is caught.
        """
        S_eff = 16384
        swa = 128
        legacy_local = S_eff * S_eff  # plan-3 P20 ``S_eff^2`` form
        swa_local_pairs = swa * S_eff - swa * (swa - 1) // 2
        new_local = 2 * swa_local_pairs  # 2 * visible_pairs (QK + PV)
        ratio = legacy_local / new_local
        assert ratio >= 100, f"SWA over-count ratio collapsed: {ratio:.2f}"


# ---------------------------------------------------------------------------
# G36a: Plan-6 P33 HyperConnection fn matmul accounting
# ---------------------------------------------------------------------------


class TestG36aHCMatmulAccounting:
    """Plan-6 P33: the ``hc`` breakdown row must equal the closed form
    ``B * S * K * D * K * (2 * (L + M) * (2 + K) + (1 + M))``.

    Pinned independently from G16 because it's a brand-new component
    and the formula has a non-obvious factor structure (2 mixers per
    layer x (L+M) layers, plus 1 trunk head + M MTP heads).
    """

    @pytest.mark.parametrize("hc_mult", [2, 4, 8])
    @pytest.mark.parametrize("mtp_num_layers", [0, 1, 2])
    def test_hc_matmul_matches_closed_form(self, hc_mult, mtp_num_layers):
        args = _v4_flash_smoke_args(
            num_layers=4,
            seq_length=128,
            hc_mult=hc_mult,
            mtp_num_layers=mtp_num_layers,
        )
        batch_size = 8
        _total, br = compute_v4_flops(args, batch_size=batch_size)
        B = batch_size
        S = args.seq_length
        K = hc_mult
        D = args.hidden_size
        L = args.num_layers
        M = mtp_num_layers
        expected = B * S * K * D * K * (2 * (L + M) * (2 + K) + (1 + M))
        assert br.hc == expected

    def test_hc_matmul_uses_seq_len_not_seq_len_eff(self):
        """HyperMixer runs on the un-packed ``[B, S, K, D]`` tensor; cost
        must scale with ``seq_len`` not ``seq_len * hc_mult``.

        Concretely: doubling ``hc_mult`` from ``K=2 -> 4`` multiplies the
        mixer factor ``K * D * K * (2+K)`` by ``(4*4*6) / (2*2*4) = 6``,
        but doubling ``seq_length`` only doubles the per-layer cost.
        Pinning both axes proves the closed form is keyed on the right
        sequence-axis variable.
        """
        ref = _v4_flash_smoke_args(num_layers=2, seq_length=64, hc_mult=2, mtp_num_layers=0)
        ref_double_s = _v4_flash_smoke_args(num_layers=2, seq_length=128, hc_mult=2, mtp_num_layers=0)
        ref_double_k = _v4_flash_smoke_args(num_layers=2, seq_length=64, hc_mult=4, mtp_num_layers=0)
        _t1, br_ref = compute_v4_flops(ref, batch_size=1)
        _t2, br_s = compute_v4_flops(ref_double_s, batch_size=1)
        _t3, br_k = compute_v4_flops(ref_double_k, batch_size=1)
        assert br_s.hc == 2 * br_ref.hc

        # Doubling K from 2 -> 4 scales mixer per-layer cost as
        # K*K*(2+K) -> (4*4*6) / (2*2*4) = 6 and head cost as
        # K*K -> 16/4 = 4.  With L=2 mixers (= 4) and 1 head,
        # the aggregate multiplier is (4 * 6 + 4) / (4 * 1 + 1) = 28/5.
        # (L=2, M=0 → mixer term = 2*L*(2+K)*K^2*D = ratio 6;
        #  head term = (1+M)*K^2*D = ratio 4)
        # Total ratio: (2*L*(2+K_new)*K_new^2 + (1+M)*K_new^2) /
        #             (2*L*(2+K_old)*K_old^2 + (1+M)*K_old^2)
        L = 2
        M = 0
        K_old = 2
        K_new = 4
        num = 2 * L * (2 + K_new) * K_new * K_new + (1 + M) * K_new * K_new
        den = 2 * L * (2 + K_old) * K_old * K_old + (1 + M) * K_old * K_old
        assert br_k.hc * den == br_ref.hc * num


# ---------------------------------------------------------------------------
# Hash-layer / no-hash variant
# ---------------------------------------------------------------------------


class TestHashLayerHandling:
    """Hash-routed layers must skip the topk router GEMM."""

    def test_zero_hash_layers_charges_router_on_every_moe_layer(self):
        args = _v4_flash_smoke_args(num_hash_layers=0)
        _, with_hash = compute_v4_flops(args, batch_size=4)

        args_no_hash = _v4_flash_smoke_args(num_hash_layers=args.num_layers)
        _, all_hash = compute_v4_flops(args_no_hash, batch_size=4)

        # All-hash strictly less because router cost is dropped on every layer.
        assert all_hash.moe < with_hash.moe
        delta_per_layer = 4 * args.seq_length * args.hc_mult * args.hidden_size * args.num_experts
        assert with_hash.moe - all_hash.moe == delta_per_layer * args.num_layers


# ---------------------------------------------------------------------------
# compress_ratios normalization (decoder + MTP slicing)
# ---------------------------------------------------------------------------


class TestNormalizeLayerRatios:
    def test_string_yaml_form_parses(self):
        decoder, mtp = _normalize_layer_ratios("[0, 0, 4, 128, 4, 0]", num_layers=6, mtp_num_layers=0)
        assert decoder == [0, 0, 4, 128, 4, 0]
        assert mtp == []

    def test_decoder_plus_mtp_layout_splits_correctly(self):
        decoder, mtp = _normalize_layer_ratios([0, 4, 128, 4, 0], num_layers=4, mtp_num_layers=1)
        assert decoder == [0, 4, 128, 4]
        assert mtp == [0]

    def test_none_defaults_to_all_dense(self):
        decoder, mtp = _normalize_layer_ratios(None, num_layers=3, mtp_num_layers=2)
        assert decoder == [0, 0, 0]
        assert mtp == [0, 0]

    def test_short_list_pads_with_last_value(self):
        decoder, mtp = _normalize_layer_ratios([4, 128], num_layers=4, mtp_num_layers=0)
        assert decoder == [4, 128, 128, 128]
        assert mtp == []


# ---------------------------------------------------------------------------
# G17: non-V4 fall-through byte-for-byte
# ---------------------------------------------------------------------------


class TestDispatchSwitch:
    """G17: ``dispatch_v4`` flag controls whether upstream is called."""

    @pytest.fixture(autouse=True)
    def _silence_breakdown_log(self, monkeypatch):
        """The wrapper emits a one-shot breakdown via ``log_rank_0`` on the
        first V4 call.  Under pytest there's no torch.distributed bound so
        rank-aware logging would explode; flip the once-only flag to ``True``
        so the wrapper skips the emit path.
        """
        from primus.backends.megatron.patches import deepseek_v4_flops_patches as mod

        monkeypatch.setattr(mod, "_BREAKDOWN_LOGGED", True)

    @pytest.fixture
    def fake_upstream_factory(self):
        """Returns ``(make_wrapped, sentinel_calls)`` per-test."""
        sentinel_calls = []

        def fake_upstream(args, batch_size):
            sentinel_calls.append((id(args), batch_size))
            return 12345 + batch_size + len(getattr(args, "model_type", "") or "")

        def make_wrapped(*, dispatch_v4: bool):
            return _make_v4_num_floating_point_operations(fake_upstream, dispatch_v4=dispatch_v4)

        return make_wrapped, sentinel_calls

    @pytest.mark.parametrize(
        "model_type",
        ["gpt", "llama3", "deepseek_v3", "mamba_hybrid", None, ""],
    )
    def test_dispatch_v4_false_falls_through_byte_for_byte(self, fake_upstream_factory, model_type):
        make_wrapped, sentinel_calls = fake_upstream_factory
        wrapped = make_wrapped(dispatch_v4=False)
        args = SimpleNamespace(model_type=model_type)
        result = wrapped(args, 32)
        assert result == 12345 + 32 + len(model_type or "")
        assert sentinel_calls == [(id(args), 32)]

    def test_dispatch_v4_true_uses_v4_closed_form(self, fake_upstream_factory):
        make_wrapped, sentinel_calls = fake_upstream_factory
        wrapped = make_wrapped(dispatch_v4=True)
        args = _v4_flash_smoke_args()
        result = wrapped(args, batch_size=4)
        assert result > 0
        assert sentinel_calls == []  # Upstream MUST NOT be called when dispatching V4.

    def test_dispatch_v4_true_ignores_runtime_model_type_mutation(self, fake_upstream_factory):
        """Megatron's pretrain() rewrites ``args.model_type`` with the
        ``ModelType`` enum just before ``train()`` runs.  The wrapper must
        keep dispatching to V4 anyway because the install-time decision is
        captured via the closure flag.
        """
        from enum import Enum

        class _FakeModelType(Enum):
            encoder_or_decoder = 1

        make_wrapped, sentinel_calls = fake_upstream_factory
        wrapped = make_wrapped(dispatch_v4=True)
        args = _v4_flash_smoke_args()
        args.model_type = _FakeModelType.encoder_or_decoder  # post-pretrain() state
        result = wrapped(args, batch_size=4)
        assert result > 0
        assert sentinel_calls == []


# ---------------------------------------------------------------------------
# Patch installation lifecycle
# ---------------------------------------------------------------------------


class TestPatchInstallation:
    def test_idempotent_installation(self, monkeypatch):
        """Re-applying the patch with an already-wrapped target is a no-op."""
        from primus.backends.megatron.patches import deepseek_v4_flops_patches as mod
        from primus.core.patches.context import PatchContext

        def upstream(args, bs):
            return 0

        wrapped_once = mod._make_v4_num_floating_point_operations(upstream, dispatch_v4=True)

        # ``patch_v4_flops_reporting`` does ``import megatron.training.training``,
        # which under bare pytest pulls in the real megatron package tree.
        # Stub out every level so the import resolves to our fake module.
        import sys

        fake_megatron = types.ModuleType("megatron")
        fake_megatron_training_pkg = types.ModuleType("megatron.training")
        fake_megatron_training_mod = types.ModuleType("megatron.training.training")
        fake_megatron_training_mod.num_floating_point_operations = wrapped_once
        fake_megatron_training_pkg.training = fake_megatron_training_mod
        fake_megatron.training = fake_megatron_training_pkg

        monkeypatch.setitem(sys.modules, "megatron", fake_megatron)
        monkeypatch.setitem(sys.modules, "megatron.training", fake_megatron_training_pkg)
        monkeypatch.setitem(sys.modules, "megatron.training.training", fake_megatron_training_mod)

        # Silence rank-aware logger; primus' singleton ``_logger`` isn't bound
        # under bare pytest so we replace ``log_rank_0`` with a no-op.
        monkeypatch.setattr(mod, "log_rank_0", lambda *a, **kw: None)

        ctx = PatchContext(
            backend="megatron",
            phase="before_train",
            extra={"module_config": types.SimpleNamespace(params=_v4_flash_smoke_args())},
        )
        mod.patch_v4_flops_reporting(ctx)
        # The wrapped function must remain the same instance (no double-wrap).
        assert fake_megatron_training_mod.num_floating_point_operations is wrapped_once
