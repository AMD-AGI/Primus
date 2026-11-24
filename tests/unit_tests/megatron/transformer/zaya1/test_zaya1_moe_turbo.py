###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Primus Turbo grouped-GEMM experts and zaya_router_defer_ar against the default paths.

The experts run as Turbo grouped GEMMs when Primus Turbo's enable_primus_turbo
and use_turbo_grouped_gemm are set. The CPU tests swap primus_turbo's grouped
GEMM for a reference with the same contract (rows past group_offs[-1] come
back zero, and are never read), its moe_permute and moe_unpermute for a stable
sort and its inverse, and its swiglu_with_probs for silu(gate) * up * p, so
they check the dispatch, the MoD skip slot, empty experts, gradients and
checkpoint keys of the grouped path against the per-expert split path.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from test_zaya1_math import _cfg  # noqa: E402  (pytest puts this dir on sys.path)

from primus.backends.megatron.core.models.zaya1 import zaya1_modules as zm


def _reference_grouped_gemm(a, b, group_lens, group_offs):
    lens = group_lens.tolist()
    offs = group_offs.tolist()
    pieces = [a[offs[g] : offs[g] + n] @ b[g].t() for g, n in enumerate(lens)]
    pieces.append(a.new_zeros(a.shape[0] - offs[-1], b.shape[1]))
    return torch.cat(pieces, dim=0)


def _reference_permute(flat, flat_idx, num_groups, probs=None):
    """moe_permute's contract: rows grouped by slot in token order, a map back, each slot's count, the probs."""
    perm = torch.sort(flat_idx, stable=True)[1]
    row_probs = None if probs is None else probs.view(-1)[perm]
    return flat[perm], perm, torch.bincount(flat_idx, minlength=num_groups), row_probs


def _reference_unpermute(rows, row_map, num_groups):
    return rows.new_zeros(rows.shape).index_copy(0, row_map, rows)


def _reference_swiglu_with_probs(gate_up, probs, group_lens):
    gate, up = gate_up.float().chunk(2, dim=-1)
    return (F.silu(gate) * up * probs.unsqueeze(-1)).to(gate_up.dtype)


@pytest.fixture
def turbo_cpu(monkeypatch):
    monkeypatch.setattr(zm, "_turbo_grouped_gemm", _reference_grouped_gemm)
    monkeypatch.setattr(zm, "_turbo_grouped_gemm_fp8", _reference_grouped_gemm)
    monkeypatch.setattr(zm, "_turbo_permute", _reference_permute)
    monkeypatch.setattr(zm, "_turbo_unpermute", _reference_unpermute)
    monkeypatch.setattr(zm, "_turbo_swiglu_with_probs", _reference_swiglu_with_probs)


class _TeLinearStandIn(torch.nn.Linear):
    """``nn.Linear`` with Transformer Engine's extra state, in place of TELinear."""

    def get_extra_state(self):
        return torch.empty(0, dtype=torch.uint8)

    def set_extra_state(self, state):
        self.loaded_extra_state = state


class _Fp8LinearStandIn(torch.nn.Module):
    """``_Fp8Linear``'s layout, the TE linear under ``.linear``, without Transformer Engine."""

    def __init__(self, in_features, out_features, bias):
        super().__init__()
        self.linear = _TeLinearStandIn(in_features, out_features, bias=bias)

    def forward(self, x):
        return self.linear(x)


_linear = zm._linear


def _linear_stand_in(in_features, out_features, bias, config, fp8=True):
    """``_linear``, with ``_Fp8Linear``'s layout under FP8 but no Transformer Engine."""
    if not (fp8 and zm._fp8_requested(config)):
        return _linear(in_features, out_features, bias, config, fp8=False)
    stand_in = _Fp8LinearStandIn(in_features, out_features, bias)
    zm._init_weight(stand_in.linear.weight, config)
    if bias:
        torch.nn.init.zeros_(stand_in.linear.bias)
    return stand_in


_FP8 = dict(fp8="hybrid", fp8_recipe="tensorwise")


def _turbo_settings(monkeypatch, on, gate_in_act=False):
    """Megatron arguments with Turbo's grouped GEMM on or off, and its default backend left as is."""
    args = SimpleNamespace(
        enable_primus_turbo=on,
        use_turbo_grouped_gemm=on,
        use_turbo_fused_act_with_probs=gate_in_act,
        turbo_grouped_gemm_backend="default",
    )
    monkeypatch.setattr(zm, "_megatron_args", lambda: args)
    monkeypatch.setattr(zm, "_publish_turbo_grouped_gemm_backend", lambda: None)


def _build_moe(mode, monkeypatch, seed=0, gate_in_act=False, **cfg_overrides):
    _turbo_settings(monkeypatch, mode == "turbo", gate_in_act)
    torch.manual_seed(seed)
    return zm.ZayaMoE(_cfg(**cfg_overrides))


def _skew_routes(moe, empty=(1,), mod_bias=0.3):
    # At init the router's scores barely depend on the token, so every token
    # takes the slot with the largest bias. Scale its projections so the
    # routes spread, push some experts out of every topk and give the MoD slot
    # a share.
    with torch.no_grad():
        for module in (moe.router.down_proj, *moe.router.router_mlp):
            if isinstance(module, torch.nn.Linear):
                module.weight.mul_(50.0)
        for e in empty:
            moe.router.balancing_biases[e] = -100.0
        moe.router.balancing_biases[-1] = mod_bias


def _expert_grads(moe, fc):
    if moe.use_turbo_gemm:
        g = getattr(moe, f"{fc}_weight").grad
        return [g[i] for i in range(moe.num_real_experts)]
    out = []
    for expert in moe.local_experts:
        w = zm._expert_weight(getattr(expert, f"linear_{fc}"))
        out.append(torch.zeros_like(w) if w.grad is None else w.grad)
    return out


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("gate_in_act", [False, True])
@pytest.mark.parametrize("fuse_moe", [True, False])
@pytest.mark.parametrize("use_mod", [True, False])
def test_turbo_matches_split_forward_and_backward(
    turbo_cpu, monkeypatch, use_mod, fuse_moe, gate_in_act, fp8
):
    if fp8:
        monkeypatch.setattr(zm, "_linear", _linear_stand_in)
    cfg = dict(zaya_use_mod=use_mod, zaya_fuse_moe=fuse_moe, **(_FP8 if fp8 else {}))
    split = _build_moe("split", monkeypatch, **cfg)
    turbo = _build_moe("turbo", monkeypatch, gate_in_act=gate_in_act, **cfg)
    assert split.local_experts is not None and turbo.use_turbo_gemm
    assert turbo.gate_in_act is gate_in_act and not split.gate_in_act
    assert turbo.turbo_fp8 is fp8
    # Same init stream, same weights.
    for i, expert in enumerate(split.local_experts):
        torch.testing.assert_close(turbo.fc1_weight[i], zm._expert_weight(expert.linear_fc1), rtol=0, atol=0)
        torch.testing.assert_close(turbo.fc2_weight[i], zm._expert_weight(expert.linear_fc2), rtol=0, atol=0)
    for moe in (split, turbo):
        moe.eval()  # keep the balancing bias fixed between the two runs
        _skew_routes(moe, empty=(1, 2) if use_mod else (1,))

    torch.manual_seed(1)
    hidden = torch.randn(16, 3, 32)
    prev = torch.randn(16, 3, 8)
    outs = {}
    for name, moe in (("split", split), ("turbo", turbo)):
        h = hidden.clone().requires_grad_(True)
        p = prev.clone().requires_grad_(True)
        out, nxt = moe(h, p)
        (out * torch.linspace(-1, 1, out.numel()).view_as(out)).sum().add(nxt.square().sum()).backward()
        outs[name] = (out.detach(), h.grad, p.grad)

    _, idx, _ = split.router(hidden, prev)
    routed = set(idx.flatten().tolist())
    assert 1 not in routed  # an empty expert is exercised
    assert len(routed - {split.num_real_experts}) >= 2  # so are the offsets between groups
    if use_mod:
        assert split.num_real_experts in routed  # the skip slot is exercised

    for a, b in zip(outs["split"], outs["turbo"]):
        torch.testing.assert_close(b, a)
    for fc in ("fc1", "fc2"):
        for gs, gt in zip(_expert_grads(split, fc), _expert_grads(turbo, fc)):
            torch.testing.assert_close(gt, gs)
    assert torch.count_nonzero(turbo.fc1_weight.grad[1]) == 0
    for (n_s, p_s), (n_t, p_t) in zip(split.router.named_parameters(), turbo.router.named_parameters()):
        assert n_s == n_t
        if p_s.grad is None:
            # Only the first-layer EDA scale (prev_router given here, so none).
            assert torch.count_nonzero(p_t.grad) == 0
        else:
            torch.testing.assert_close(p_t.grad, p_s.grad)


class _NanGradRows(torch.autograd.Function):
    """The identity, whose gradient comes back NaN in ``rows``."""

    @staticmethod
    def forward(ctx, x, rows):
        ctx.save_for_backward(rows)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        (rows,) = ctx.saved_tensors
        return grad.masked_fill(rows.view(-1, *[1] * (grad.dim() - 1)), float("nan")), None


def _swiglu_with_probs_nan_masked(gate_up, probs, group_lens):
    """The reference, with NaN wherever swiglu_with_probs leaves the masked rows unwritten or unspecified."""
    rows = torch.arange(gate_up.shape[0]) >= group_lens.sum()
    out = _reference_swiglu_with_probs(
        _NanGradRows.apply(gate_up, rows), _NanGradRows.apply(probs, rows), group_lens
    )
    return out.masked_fill(rows.unsqueeze(-1), float("nan"))


class _ScaledByEveryElement(torch.autograd.Function):
    """``out``, but NaN everywhere if any element of ``seen`` is not finite, and the same for its gradient."""

    @staticmethod
    def forward(ctx, out, seen):
        return out * (seen.abs().amax() * 0 + 1)

    @staticmethod
    def backward(ctx, grad):
        return grad * (grad.abs().amax() * 0 + 1), None


def _grouped_gemm_reading_every_row(a, b, group_lens, group_offs):
    """The reference grouped GEMM, reading every row of its input and output gradient as tensorwise FP8 does."""
    return _ScaledByEveryElement.apply(_reference_grouped_gemm(a, b, group_lens, group_offs), a)


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("fuse_moe", [True, False])
def test_turbo_gate_in_act_leaves_no_row_unwritten(turbo_cpu, monkeypatch, fuse_moe, fp8):
    """No row the kernel leaves unwritten reaches a grouped GEMM, and the skip rows still come out as ``x * p``."""
    monkeypatch.setattr(zm, "_turbo_swiglu_with_probs", _swiglu_with_probs_nan_masked)
    monkeypatch.setattr(zm, "_turbo_grouped_gemm", _grouped_gemm_reading_every_row)
    monkeypatch.setattr(zm, "_turbo_grouped_gemm_fp8", _grouped_gemm_reading_every_row)
    if fp8:
        monkeypatch.setattr(zm, "_linear", _linear_stand_in)
    cfg = dict(zaya_fuse_moe=fuse_moe, **(_FP8 if fp8 else {}))
    split = _build_moe("split", monkeypatch, **cfg)
    turbo = _build_moe("turbo", monkeypatch, gate_in_act=True, **cfg)
    assert turbo.turbo_fp8 is fp8
    for moe in (split, turbo):
        moe.eval()
        _skew_routes(moe, empty=(1, 2))
    torch.manual_seed(1)
    hidden = torch.randn(16, 3, 32)
    prev = torch.randn(16, 3, 8)
    _, idx, _ = split.router(hidden, prev)
    assert split.num_real_experts in set(idx.flatten().tolist())
    runs = []
    for moe in (split, turbo):
        h = hidden.clone().requires_grad_(True)
        p = prev.clone().requires_grad_(True)
        out, nxt = moe(h, p)
        (out * torch.linspace(-1, 1, out.numel()).view_as(out)).sum().add(nxt.square().sum()).backward()
        runs.append([out.detach(), h.grad, p.grad] + [q.grad for q in moe.router.parameters()])
    for a, b in zip(*runs):
        if a is None:
            assert torch.count_nonzero(b) == 0
        else:
            assert torch.isfinite(b).all()
            torch.testing.assert_close(b, a)
    for fc in ("fc1", "fc2"):
        assert torch.isfinite(getattr(turbo, f"{fc}_weight").grad).all()


def test_turbo_keeps_per_expert_checkpoint_keys(turbo_cpu, monkeypatch):
    split = _build_moe("split", monkeypatch, seed=3)
    turbo = _build_moe("turbo", monkeypatch, seed=4)
    sd_split = split.state_dict()
    sd_turbo = turbo.state_dict()
    assert sorted(sd_split) == sorted(sd_turbo)
    assert "local_experts.0.linear_fc1.weight" in sd_turbo
    assert not any(k.startswith("fc1_weight") or k.startswith("fc2_weight") for k in sd_turbo)
    for k, v in sd_split.items():
        assert sd_turbo[k].shape == v.shape

    # split -> turbo
    turbo.load_state_dict(sd_split, strict=True)
    for i, expert in enumerate(split.local_experts):
        torch.testing.assert_close(turbo.fc1_weight[i], expert.linear_fc1.weight, rtol=0, atol=0)
        torch.testing.assert_close(turbo.fc2_weight[i], expert.linear_fc2.weight, rtol=0, atol=0)
    # turbo -> split, through a prefixed parent module
    with torch.no_grad():
        turbo.fc2_weight.add_(1.0)
    parent_t = torch.nn.ModuleDict({"moe": turbo})
    parent_s = torch.nn.ModuleDict({"moe": split})
    parent_s.load_state_dict(parent_t.state_dict(), strict=True)
    for i, expert in enumerate(split.local_experts):
        torch.testing.assert_close(expert.linear_fc2.weight, turbo.fc2_weight[i], rtol=0, atol=0)
    names = [n for n, _ in turbo.named_parameters()]
    assert "fc1_weight" in names and "fc2_weight" in names


def test_turbo_needs_both_settings_and_tensorwise_fp8(monkeypatch):
    published = []
    monkeypatch.setattr(zm, "_publish_turbo_grouped_gemm_backend", lambda: published.append(True))
    for enable, grouped in ((True, True), (True, False), (False, True), (False, False)):
        args = SimpleNamespace(enable_primus_turbo=enable, use_turbo_grouped_gemm=grouped)
        monkeypatch.setattr(zm, "_megatron_args", lambda: args)
        assert zm.ZayaMoE(_cfg()).use_turbo_gemm == (enable and grouped)
    assert len(published) == 1
    # Megatron not initialized: no Turbo settings, so split.
    monkeypatch.setattr(zm, "_megatron_args", lambda: None)
    assert not zm.ZayaMoE(_cfg()).use_turbo_gemm
    # Under FP8 only the tensorwise recipe runs Turbo's FP8 grouped GEMM; the others keep the split experts.
    _turbo_settings(monkeypatch, True)
    monkeypatch.setattr(zm, "_publish_turbo_grouped_gemm_backend", lambda: published.append(True))
    monkeypatch.setattr(zm, "_linear", _linear_stand_in)
    for recipe in ("delayed", "blockwise", "mxfp8", None):
        moe = zm.ZayaMoE(_cfg(fp8="hybrid", fp8_recipe=recipe))
        assert not moe.use_turbo_gemm and not moe.turbo_fp8
    assert len(published) == 1
    moe = zm.ZayaMoE(_cfg(**_FP8))
    assert moe.use_turbo_gemm and moe.turbo_fp8
    assert len(published) == 2


def test_turbo_gate_in_act_needs_the_turbo_experts(monkeypatch):
    """use_turbo_fused_act_with_probs moves the gate only when the experts run as Turbo grouped GEMMs."""
    monkeypatch.setattr(zm, "_publish_turbo_grouped_gemm_backend", lambda: None)
    for enable, grouped, fused in (
        (True, True, True),
        (True, True, False),
        (True, False, True),
        (False, True, True),
    ):
        args = SimpleNamespace(
            enable_primus_turbo=enable, use_turbo_grouped_gemm=grouped, use_turbo_fused_act_with_probs=fused
        )
        monkeypatch.setattr(zm, "_megatron_args", lambda: args)
        assert zm.ZayaMoE(_cfg()).gate_in_act == (enable and grouped and fused)
    _turbo_settings(monkeypatch, True, gate_in_act=True)
    monkeypatch.setattr(zm, "_linear", _linear_stand_in)
    assert not zm.ZayaMoE(_cfg(fp8="hybrid", fp8_recipe="delayed")).gate_in_act
    assert zm.ZayaMoE(_cfg(**_FP8)).gate_in_act


def _build_stack(defer):
    torch.manual_seed(7)
    stack = zm.ZayaStack(_cfg(num_layers=6, zaya_router_defer_ar=defer))
    stack.train()
    return stack


def _biases(stack):
    return [r.balancing_biases.clone() for r in stack._routers()]


def test_deferred_router_updates_are_bit_exact(monkeypatch):
    eager = _build_stack(False)
    deferred = _build_stack(True)
    assert all(r.defer_bias_update for r in deferred._routers())
    reduces = []
    monkeypatch.setattr(zm, "_allreduce_dp", lambda t: reduces.append(t.numel()))
    n_routers = len(eager._routers())
    for step in range(3):
        torch.manual_seed(100 + step)
        hidden = torch.randn(8, 2, 32)
        outs = []
        for stack in (eager, deferred):
            h = hidden.clone().requires_grad_(True)
            out = stack(h, position_ids=None, attention_mask=None)
            out.square().sum().backward()
            outs.append(out.detach())
        # Outputs see identical biases, and after backward the biases match.
        torch.testing.assert_close(outs[1], outs[0], rtol=0, atol=0)
        for a, b in zip(_biases(eager), _biases(deferred)):
            torch.testing.assert_close(b, a, rtol=0, atol=0)
        assert all(r._pending_counts is None for r in deferred._routers())
        for a, b in zip(eager._routers(), deferred._routers()):
            assert int(a.balance_step) == int(b.balance_step) == step + 1
    # eager: one reduce per router per step; deferred: one per step.
    assert reduces.count(eager._routers()[0].num_scored) == 3 * n_routers
    assert reduces.count(n_routers * eager._routers()[0].num_scored) == 3


def test_deferred_router_flushes_on_next_forward_without_backward():
    eager = _build_stack(False)
    deferred = _build_stack(True)
    hidden = torch.randn(8, 2, 32)
    with torch.no_grad():
        eager(hidden, None, None)
        deferred(hidden, None, None)
    assert all(r._pending_counts is not None for r in deferred._routers())
    eager.eval()
    deferred.eval()
    with torch.no_grad():
        a = eager(hidden, None, None)
        b = deferred(hidden, None, None)
    torch.testing.assert_close(b, a, rtol=0, atol=0)
    for x, y in zip(_biases(eager), _biases(deferred)):
        torch.testing.assert_close(y, x, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU and primus_turbo")
@pytest.mark.parametrize("gate_in_act", [False, True])
def test_turbo_gpu_matches_split_without_host_sync(monkeypatch, gate_in_act):
    pytest.importorskip("primus_turbo.pytorch.ops")
    split = (
        _build_moe(
            "split",
            monkeypatch,
            hidden_size=256,
            moe_ffn_hidden_size=128,
            ffn_hidden_size=128,
            num_moe_experts=16,
        )
        .cuda()
        .bfloat16()
    )
    turbo = (
        _build_moe(
            "turbo",
            monkeypatch,
            gate_in_act=gate_in_act,
            hidden_size=256,
            moe_ffn_hidden_size=128,
            ffn_hidden_size=128,
            num_moe_experts=16,
        )
        .cuda()
        .bfloat16()
    )
    for moe in (split, turbo):
        moe.eval()
        _skew_routes(moe, empty=(3,))
    hidden = torch.randn(64, 8, 256, device="cuda", dtype=torch.bfloat16)
    ref, _ = split(hidden, None)
    turbo(hidden, None)  # warm up / autotune
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        out, _ = turbo(hidden, None)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU and primus_turbo")
@pytest.mark.parametrize("gate_in_act", [False, True])
def test_turbo_permute_moves_rows_like_a_stable_sort(monkeypatch, gate_in_act):
    """Turbo's moe_permute and moe_unpermute against a stable sort and its inverse, bit for bit."""
    pytest.importorskip("primus_turbo.pytorch.ops")
    shape = dict(hidden_size=256, moe_ffn_hidden_size=128, ffn_hidden_size=128, num_moe_experts=16)
    moe = _build_moe("turbo", monkeypatch, gate_in_act=gate_in_act, **shape).cuda().bfloat16().eval()
    _skew_routes(moe, empty=(3,))
    torch.manual_seed(5)
    hidden = torch.randn(64, 8, 256, device="cuda", dtype=torch.bfloat16)
    prev = torch.randn(64, 8, 8, device="cuda", dtype=torch.bfloat16)
    weights = torch.linspace(-1, 1, hidden.numel(), device="cuda").view_as(hidden)
    _, idx, _ = moe.router(hidden, prev)
    routed = set(idx.flatten().tolist())
    assert 3 not in routed and moe.num_real_experts in routed

    def run():
        moe.zero_grad(set_to_none=True)
        h = hidden.clone().requires_grad_(True)
        p = prev.clone().requires_grad_(True)
        out, nxt = moe(h, p)
        ((out.float() * weights).sum() + nxt.float().square().sum()).backward()
        grads = {f"{n} grad": q.grad for n, q in moe.named_parameters()}
        return {"output": out, "input grad": h.grad, "router-state grad": p.grad, **grads}

    turbo = run()
    monkeypatch.setattr(zm, "_turbo_permute", _reference_permute)
    monkeypatch.setattr(zm, "_turbo_unpermute", _reference_unpermute)
    reference = run()
    assert turbo.keys() == reference.keys()
    for name, ref in reference.items():
        assert ref is not None and torch.equal(turbo[name], ref), name


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU and primus_turbo")
@pytest.mark.parametrize("gate_in_act", [False, True])
@pytest.mark.parametrize("fuse_moe", [True, False])
def test_turbo_gpu_backward_matches_split(monkeypatch, fuse_moe, gate_in_act):
    """primus_turbo's grouped GEMM backward, with an empty expert and MoD skip rows, against split."""
    pytest.importorskip("primus_turbo.pytorch.ops")
    shape = dict(
        hidden_size=256,
        moe_ffn_hidden_size=128,
        ffn_hidden_size=128,
        num_moe_experts=16,
        zaya_fuse_moe=fuse_moe,
    )
    split = _build_moe("split", monkeypatch, **shape).cuda().bfloat16()
    turbo = _build_moe("turbo", monkeypatch, gate_in_act=gate_in_act, **shape).cuda().bfloat16()
    for moe in (split, turbo):
        moe.eval()  # keep the balancing bias fixed between the two runs
        _skew_routes(moe, empty=(3,))
    torch.manual_seed(5)
    hidden = torch.randn(64, 8, 256, device="cuda", dtype=torch.bfloat16)
    prev = torch.randn(64, 8, 8, device="cuda", dtype=torch.bfloat16)
    weights = torch.linspace(-1, 1, hidden.numel(), device="cuda").view_as(hidden)
    runs = {}
    for name, moe in (("split", split), ("turbo", turbo)):
        h = hidden.clone().requires_grad_(True)
        p = prev.clone().requires_grad_(True)
        out, nxt = moe(h, p)
        ((out.float() * weights).sum() + nxt.float().square().sum()).backward()
        runs[name] = (out.detach(), h.grad, p.grad)

    _, idx, _ = split.router(hidden, prev)
    routed = set(idx.flatten().tolist())
    assert 3 not in routed and split.num_real_experts in routed
    for what, a, b in zip(("output", "input grad", "router-state grad"), runs["split"], runs["turbo"]):
        assert _rel(b, a) < 2e-2, f"{what}: {_rel(b, a):.3e}"
    for fc in ("fc1", "fc2"):
        for expert_id, (gs, gt) in enumerate(zip(_expert_grads(split, fc), _expert_grads(turbo, fc))):
            if expert_id == 3:
                assert torch.count_nonzero(gt) == 0
            else:
                assert _rel(gt, gs) < 2e-2, f"{fc} expert {expert_id}: {_rel(gt, gs):.3e}"
    for (n_s, p_s), (n_t, p_t) in zip(split.router.named_parameters(), turbo.router.named_parameters()):
        assert n_s == n_t
        assert _rel(p_t.grad, p_s.grad) < 2e-2, f"{n_s}: {_rel(p_t.grad, p_s.grad):.3e}"


@pytest.mark.parametrize("fuse_moe", [True, False])
def test_turbo_every_param_gets_a_grad(turbo_cpu, monkeypatch, fuse_moe):
    """--overlap_grad_reduce needs a grad hook on every param every step."""
    _turbo_settings(monkeypatch, False)
    torch.manual_seed(11)
    split = zm.ZayaStack(_cfg(num_layers=6, zaya_fuse_moe=fuse_moe))
    _turbo_settings(monkeypatch, True)
    torch.manual_seed(11)
    turbo = zm.ZayaStack(_cfg(num_layers=6, zaya_fuse_moe=fuse_moe))
    turbo.load_state_dict(split.state_dict(), strict=True)
    for stack in (split, turbo):
        stack.eval()  # fixed balancing biases
        for r in stack._routers():
            with torch.no_grad():
                r.balancing_biases[1] = -100.0  # expert 1 gets no tokens
    torch.manual_seed(12)
    hidden = torch.randn(8, 2, 32)
    outs = []
    for stack in (split, turbo):
        h = hidden.clone().requires_grad_(True)
        out = stack(h, position_ids=None, attention_mask=None)
        out.square().sum().backward()
        outs.append((out.detach(), h.grad))
    for a, b in zip(*outs):
        torch.testing.assert_close(b, a)
    split_missing = [n for n, p in split.named_parameters() if p.grad is None]
    assert any("router_states_scale" in n for n in split_missing)  # the bug being fixed
    assert any("local_experts.1." in n for n in split_missing)
    assert [n for n, p in turbo.named_parameters() if p.grad is None] == []
    sp = dict(split.named_parameters())
    for n, p in turbo.named_parameters():
        if n in sp:
            ref = sp[n].grad if sp[n].grad is not None else torch.zeros_like(p)
            torch.testing.assert_close(p.grad, ref)
