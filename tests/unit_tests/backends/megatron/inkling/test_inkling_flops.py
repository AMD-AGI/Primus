###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Inkling FLOPs patch: the closed form by hand, and the patch lifecycle against a fake
upstream ``num_floating_point_operations`` (wrapper result, upstream called once for the
log, ``__wrapped__``, no double wrap on re-apply, the condition)."""

import sys
import types
from types import SimpleNamespace

import pytest

from primus.backends.megatron.patches import inkling_flops_patches as fp
from primus.core.patches.context import PatchContext
from primus.core.patches.patch_registry import PatchRegistry


def _args(**kw):
    a = dict(
        num_layers=2,
        hidden_size=8,
        num_attention_heads=2,
        num_query_groups=1,
        group_query_attention=True,
        kv_channels=4,
        inkling_swa_num_attention_heads=2,
        inkling_swa_num_query_groups=2,
        inkling_swa_kv_channels=4,
        inkling_sliding_window=2,
        inkling_rel_extent=6,
        inkling_local_layer_ids=[0],
        inkling_d_rel=3,
        seq_length=4,
        moe_ffn_hidden_size=5,
        moe_router_topk=2,
        inkling_num_shared_experts=1,
        num_experts=4,
        padded_vocab_size=10,
    )
    a.update(kw)
    return SimpleNamespace(**a)


def test_mean_keys():
    assert fp._mean_keys(4, 0) == 2.5  # 1 + 2 + 3 + 4 over 4
    assert fp._mean_keys(4, 2) == (1 + 2 + 2 + 2) / 4
    assert fp._mean_keys(4, 8) == 2.5


def test_closed_form_by_hand():
    a = _args()
    # projections (q, k, v, r; o) + q.k and p.v over the mean keys + rel logits (nh * E * d_rel)
    glob = 8 * (8 + 8 + 6) + 8 * 8 + 2 * 2.5 * 8 + 2 * 6 * 3  # E = inkling_rel_extent
    loc = 8 * (8 + 16 + 6) + 8 * 8 + 2 * 1.75 * 8 + 2 * 2 * 3  # E = inkling_sliding_window
    moe = 8 * (4 + 1) + 3 * 3 * 8 * 5  # router over routed + shared experts; top-2 + 1 shared
    total, br = fp.compute_inkling_flops(a, batch_size=3)
    assert br["attn_global"] == glob * 12 * 6 and br["attn_local"] == loc * 12 * 6
    assert br["moe"] == 2 * moe * 12 * 6 and br["logits"] == 80 * 12 * 6
    assert total == int(sum(br.values()))


def test_without_gqa_every_global_head_has_its_own_kv():
    """group_query_attention off: num_query_groups (Megatron's default 1) is ignored, as upstream."""
    on = fp.attention_fmac_per_token(_args(num_query_groups=2), local=False)
    off = fp.attention_fmac_per_token(_args(group_query_attention=False, num_query_groups=1), local=False)
    assert off == on  # nh = 2 K/V heads either way


def test_the_upstream_log_call_does_not_change_the_args(monkeypatch):
    monkeypatch.setattr(fp, "log_rank_0", lambda *a, **k: None)

    def upstream(args, batch_size):
        args.num_query_groups = 99  # as Megatron does with group_query_attention off
        return 1

    a = _args()
    fp._make_wrapper(upstream)(a, 1)
    assert a.num_query_groups == 1


@pytest.fixture
def fake_training(monkeypatch):
    """A fake ``megatron.training.training`` whose upstream records its calls."""
    calls = []

    def upstream(args, batch_size):
        calls.append(batch_size)
        return 777

    mod = types.ModuleType("megatron.training.training")
    mod.num_floating_point_operations = upstream
    pkg = types.ModuleType("megatron.training")
    pkg.training = mod
    root = types.ModuleType("megatron")
    root.training = pkg
    monkeypatch.setitem(sys.modules, "megatron", root)
    monkeypatch.setitem(sys.modules, "megatron.training", pkg)
    monkeypatch.setitem(sys.modules, "megatron.training.training", mod)
    logs = []
    monkeypatch.setattr(fp, "log_rank_0", lambda msg, *a, **k: logs.append(msg))
    return SimpleNamespace(mod=mod, upstream=upstream, calls=calls, logs=logs)


def test_wrapper_returns_closed_form_and_calls_upstream_once(fake_training):
    wrapped = fp._make_wrapper(fake_training.upstream)
    a = _args()
    assert wrapped.__wrapped__ is fake_training.upstream
    assert wrapped(a, 3) == fp.compute_inkling_flops(a, 3)[0]
    assert wrapped(a, 5) == fp.compute_inkling_flops(a, 5)[0]
    assert fake_training.calls == [3]  # upstream only runs for the one-time log line
    assert sum("TFLOP" in m for m in fake_training.logs) == 1


def test_each_wrapper_logs_once(fake_training):
    a = _args()
    fp._make_wrapper(fake_training.upstream)(a, 1)
    fp._make_wrapper(fake_training.upstream)(a, 1)
    assert sum("TFLOP" in m for m in fake_training.logs) == 2  # per wrapper, not process-global


def _ctx(args):
    return PatchContext(
        backend="megatron",
        phase="before_train",
        extra={"module_config": SimpleNamespace(params=args)},
    )


def test_patch_installs_once(fake_training):
    ctx = _ctx(_args())
    fp.patch_inkling_flops_reporting(ctx)
    first = fake_training.mod.num_floating_point_operations
    assert first.__wrapped__ is fake_training.upstream
    fp.patch_inkling_flops_reporting(ctx)  # re-apply: no double wrap
    assert fake_training.mod.num_floating_point_operations is first
    assert first(_args(), 2) == fp.compute_inkling_flops(_args(), 2)[0]
    assert fake_training.calls == [2]


@pytest.mark.parametrize(
    "bad",
    [
        {"seq_length": 0},
        {"inkling_swa_kv_channels": None},
        {"inkling_num_shared_experts": 0},
        {"mtp_num_layers": -1},
    ],
)
def test_patch_refuses_bad_args_at_start(fake_training, bad):
    with pytest.raises(ValueError, match="cannot be computed"):
        fp.patch_inkling_flops_reporting(_ctx(_args(**bad)))
    assert fake_training.mod.num_floating_point_operations is fake_training.upstream


@pytest.mark.parametrize(
    "model_type,applies", [("inkling", True), ("gpt", False), ("deepseek_v3", False), (None, False)]
)
def test_condition_only_for_inkling(model_type, applies):
    patch = PatchRegistry.get("megatron.inkling.flops_reporting")
    ctx = PatchContext(
        backend="megatron",
        phase="before_train",
        extra={"module_config": SimpleNamespace(params=SimpleNamespace(model_type=model_type))},
    )
    assert bool(patch.condition(ctx)) is applies


def test_mtp_term_by_hand():
    """MTP depths: global / sliding attention (the MTP window), dense MLP + input projection, and
    one more output-layer pass each; mtp_num_layers unset adds nothing."""
    base, br0 = fp.compute_inkling_flops(_args(), batch_size=1)
    a = _args(mtp_num_layers=2, inkling_mtp_local_layer_ids=[1], inkling_dense_intermediate_size=7)
    total, br = fp.compute_inkling_flops(a, batch_size=1)
    glob = fp.attention_fmac_per_token(a, local=False)
    loc = fp.attention_fmac_per_token(a, local=True)
    assert br["mtp_attn"] == (glob + loc) * 6 * 4  # x 6 (fwd + bwd, FMA) x 4 tokens
    assert br["mtp_mlp"] == 2 * (3 * 8 * 7 + 2 * 8 * 8) * 6 * 4
    assert br["mtp_logits"] == 2 * 80 * 6 * 4
    assert total == base + int(br["mtp_attn"] + br["mtp_mlp"] + br["mtp_logits"])
    # the MTP window / heads override the trunk's sliding ones
    wide = _args(mtp_num_layers=2, inkling_mtp_local_layer_ids=[1], inkling_mtp_local_extent=4)
    assert fp.attention_fmac_per_token(wide, local=True, mtp=True) > loc
