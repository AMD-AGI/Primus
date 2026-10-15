###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the fused clip + AdamW + bf16 write optimizer step and the global grad NaN check.

CPU part (always): the hooks install once on Megatron's real classes / module globals; the patch gates
(uncovered optimizer setups, clip_grad, check_for_nan_in_loss_and_grad, rerun_mode, sft_sanitize_nan_grads)
and the PRIMUS_FUSED_ADAM_CLIP exclusion (SystemExit through run_patches); the clip deferral, the decoupled-grad
passthrough and the global-norm NaN check; the FP32Optimizer flush of a deferred clip; the main -> model param
map against Megatron's own copy; and the step host logic on a DistributedOptimizer shell with CPU stand-ins for
the GPU leaves (Triton kernel, TE FusedAdam, TE multi_tensor_scale): every eligibility gate, first-step state
init, step counters, per-group scalars, the deferred clip coefficient reaching the kernel, bf16 and fp32 model
param writes, fallbacks (before and inside the fused step) with an exact clip flush and nothing updated.
GPU part (skipped without a device): the Triton kernel against TE's FusedAdam + multi_tensor_scale + copy,
bitwise, over several steps with and without clipping.
"""

import collections
import importlib
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron")
pytest.importorskip("triton")

import primus.backends.megatron.patches.fused_adamw_clip_patches as FP  # noqa: E402
from primus.backends.megatron.core.optimizer import fused_adamw_clip as F  # noqa: E402
from primus.backends.megatron.core.optimizer import (  # noqa: E402
    triton_fused_adamw as FA,
)
from primus.core.patches.context import PatchContext  # noqa: E402
from primus.core.patches.patch_registry import PatchRegistry  # noqa: E402
from primus.core.patches.patch_runner import run_patches  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _ctx(**args):
    return PatchContext(
        backend="megatron",
        phase="before_train",
        extra={"module_config": SimpleNamespace(params=SimpleNamespace(**args))},
    )


@pytest.fixture
def restore_hooks():
    """Undo the hooks and the switches after a test."""
    import megatron.core.distributed.param_and_grad_buffer as pb
    import megatron.core.optimizer.optimizer as om

    saved = (
        om.clip_grad_by_total_norm_fp32,
        om.MixedPrecisionOptimizer.step_with_ready_grads,
        om.FP32Optimizer.step_with_ready_grads,
        pb._ParamAndGradBucketGroup.check_grads,
    )
    F._STATE["last_fallback"] = None
    yield
    F.set_fused(False)
    F.set_nancheck(False)
    F._PENDING.clear()
    F._STATE["last_fallback"] = None
    om.clip_grad_by_total_norm_fp32 = saved[0]
    om.MixedPrecisionOptimizer.step_with_ready_grads = saved[1]
    om.FP32Optimizer.step_with_ready_grads = saved[2]
    pb._ParamAndGradBucketGroup.check_grads = saved[3]


def _hooked():
    import megatron.core.distributed.param_and_grad_buffer as pb
    import megatron.core.optimizer.optimizer as om

    return (
        om.clip_grad_by_total_norm_fp32,
        om.MixedPrecisionOptimizer.step_with_ready_grads,
        om.FP32Optimizer.step_with_ready_grads,
        pb._ParamAndGradBucketGroup.check_grads,
    )


def test_hooks_install_once(restore_hooks):
    before = _hooked()
    FP.patch_fused_adamw_clip(_ctx(use_primus_fused_adamw_clip=True, use_distributed_optimizer=True))
    assert F.fused_enabled() and not F.nancheck_enabled()
    hooked = _hooked()
    assert all(getattr(h, F.HOOKED_ATTR) is b for h, b in zip(hooked, before))
    FP.patch_global_grad_nan_check(
        _ctx(use_primus_global_grad_nan_check=True, check_for_nan_in_loss_and_grad=True, clip_grad=1.0)
    )
    assert F.nancheck_enabled()
    assert _hooked() == hooked  # one set of hooks shared by both patches


@pytest.mark.parametrize(
    "args,reason",
    [
        (dict(use_distributed_optimizer=False), "use_distributed_optimizer is off"),
        (
            dict(use_distributed_optimizer=True, use_precision_aware_optimizer=True),
            "use_precision_aware_optimizer",
        ),
        (dict(use_distributed_optimizer=True, use_megatron_fsdp=True), "FSDP"),
        (dict(use_distributed_optimizer=True, use_torch_fsdp2=True), "FSDP"),
    ],
)
def test_fused_patch_not_enabled_for_uncovered_setups(restore_hooks, monkeypatch, args, reason):
    warned = []
    monkeypatch.setattr(FP, "warning_rank_0", warned.append)
    before = _hooked()
    FP.patch_fused_adamw_clip(_ctx(use_primus_fused_adamw_clip=True, **args))
    assert not F.fused_enabled()
    assert _hooked() == before  # no hook installed
    assert len(warned) == 1 and reason in warned[0] and "not enabled" in warned[0]


@pytest.mark.parametrize(
    "args,reason",
    [
        (dict(check_for_nan_in_loss_and_grad=True, clip_grad=0.0), "clip_grad=0.0"),
        (dict(check_for_nan_in_loss_and_grad=False, clip_grad=1.0), "check_for_nan_in_loss_and_grad is off"),
        (
            dict(check_for_nan_in_loss_and_grad=True, clip_grad=1.0, sft_sanitize_nan_grads=True),
            "sft_sanitize_nan_grads",
        ),
        (
            dict(check_for_nan_in_loss_and_grad=True, clip_grad=1.0, rerun_mode="validate_results"),
            "rerun_mode=validate_results",
        ),
        (
            dict(check_for_nan_in_loss_and_grad=True, clip_grad=1.0, rerun_mode="report_stats"),
            "rerun_mode=report_stats",
        ),
    ],
)
def test_nan_check_patch_gates(restore_hooks, monkeypatch, args, reason):
    warned = []
    monkeypatch.setattr(FP, "warning_rank_0", warned.append)
    FP.patch_global_grad_nan_check(_ctx(use_primus_global_grad_nan_check=True, **args))
    assert not F.nancheck_enabled()
    assert len(warned) == 1 and reason in warned[0]


def test_nan_check_patch_enabled_with_rerun_mode_disabled(restore_hooks):
    FP.patch_global_grad_nan_check(
        _ctx(
            use_primus_global_grad_nan_check=True,
            check_for_nan_in_loss_and_grad=True,
            clip_grad=1.0,
            rerun_mode="disabled",
        )
    )
    assert F.nancheck_enabled()


@pytest.mark.parametrize(
    "patch,arg",
    [
        ("patch_fused_adamw_clip", "use_primus_fused_adamw_clip"),
        ("patch_global_grad_nan_check", "use_primus_global_grad_nan_check"),
    ],
)
def test_exclusive_with_te_fused_adam_clip(restore_hooks, monkeypatch, patch, arg):
    monkeypatch.setenv("PRIMUS_FUSED_ADAM_CLIP", "1")
    ctx = _ctx(
        **{arg: True}, check_for_nan_in_loss_and_grad=True, clip_grad=1.0, use_distributed_optimizer=True
    )
    with pytest.raises(SystemExit, match="PRIMUS_FUSED_ADAM_CLIP"):
        getattr(FP, patch)(ctx)
    assert not F.fused_enabled() and not F.nancheck_enabled()


@pytest.mark.parametrize(
    "key,arg",
    [
        ("megatron.optimizer.fused_adamw_clip", "use_primus_fused_adamw_clip"),
        ("megatron.ddp.global_grad_nan_check", "use_primus_global_grad_nan_check"),
    ],
)
def test_exclusive_aborts_through_run_patches(restore_hooks, monkeypatch, key, arg):
    # run_patches swallows Exception: the conflict must surface as SystemExit.
    if PatchRegistry.get(key) is None:  # some suites clear the global registry
        importlib.reload(FP)
    monkeypatch.setenv("PRIMUS_FUSED_ADAM_CLIP", "1")
    params = SimpleNamespace(
        **{arg: True}, check_for_nan_in_loss_and_grad=True, clip_grad=1.0, use_distributed_optimizer=True
    )
    before = _hooked()
    with pytest.raises(SystemExit, match="PRIMUS_FUSED_ADAM_CLIP"):
        run_patches(
            backend="megatron",
            phase="before_train",
            enabled_ids=[key],
            extra={"module_config": SimpleNamespace(params=params)},
        )
    assert not F.fused_enabled() and not F.nancheck_enabled() and _hooked() == before


class _CudaLike(torch.Tensor):
    @property
    def is_cuda(self):
        return True

    def type(self, *a, **k):
        if a or k:
            return super().type(*a, **k)
        return {torch.float32: "torch.cuda.FloatTensor"}.get(self.dtype, "torch.cuda.Tensor")


def _cl(t):
    return t.as_subclass(_CudaLike)


def test_clip_defers_and_global_nan_check_raises(restore_hooks):
    calls = []
    clip = F.make_clip_grad_by_total_norm_fp32(lambda *a: calls.append(a))
    p = _cl(torch.zeros(8))
    p.grad = _cl(torch.ones(8))
    # off: original clip
    clip([p], 1.0, 2.0)
    assert len(calls) == 1 and not F._PENDING
    # fused: deferred coefficient, grads untouched
    F.set_fused(True)
    clip([p], 1.0, 2.0)
    assert F._PENDING[id(p)][1] == 1.0 / (2.0 + 1.0e-6) and torch.equal(p.grad, torch.ones(8))
    clip([p], 1.0, 0.5)  # no clip needed: the stale deferral is dropped
    assert not F._PENDING and len(calls) == 1
    # NaN check on the global norm, before anything is deferred
    F.set_nancheck(True)
    for bad in (float("nan"), float("inf")):
        with pytest.raises(RuntimeError, match="non-finite global grad norm"):
            clip([p], 1.0, bad)
    assert not F._PENDING
    check = F.make_check_grads(lambda self, a, b: calls.append(("bucket", a, b)))
    n = len(calls)
    check(None, True, False)  # skipped
    assert len(calls) == n
    check(None, True, True)  # check_for_large is never skipped
    assert calls[-1] == ("bucket", True, True)
    F.set_nancheck(False)
    check(None, True, False)  # switch off: the original bucket check runs
    assert calls[-1] == ("bucket", True, False)


def test_clip_passthrough_with_decoupled_grads(restore_hooks):
    calls = []
    clip = F.make_clip_grad_by_total_norm_fp32(lambda *a: calls.append(a))
    p = _cl(torch.zeros(8))
    p.grad = None
    p.decoupled_grad = _cl(torch.ones(8))
    F.set_fused(True)
    F._PENDING[id(p)] = (p, 0.5)  # a stale deferral is dropped
    clip([p], 1.0, 2.0, True)
    assert not F._PENDING
    clip([p], max_norm=1.0, total_norm=2.0, use_decoupled_grad=True)  # ChainedOptimizer's call form
    assert len(calls) == 2 and not F._PENDING
    for a in calls:
        assert a[0][0] is p and len(a[0]) == 1 and a[1:] == (1.0, 2.0, True)


@pytest.mark.parametrize("use_decoupled_grad", [False, True])
def test_global_nan_check_without_fused_switch(restore_hooks, use_decoupled_grad):
    calls = []
    clip = F.make_clip_grad_by_total_norm_fp32(lambda *a: calls.append(a))
    p = _cl(torch.zeros(8))
    p.grad = _cl(torch.ones(8))
    p.decoupled_grad = _cl(torch.ones(8))
    F.set_fused(False)
    F.set_nancheck(True)
    for bad in (float("nan"), float("inf")):
        with pytest.raises(RuntimeError, match="non-finite global grad norm"):
            clip([p], 1.0, bad, use_decoupled_grad)
    assert calls == []
    clip([p], 1.0, 2.0, use_decoupled_grad)
    assert len(calls) == 1 and calls[0][0][0] is p and calls[0][1:] == (1.0, 2.0, use_decoupled_grad)


def _cpu_multi_tensor_scale(monkeypatch):
    """CPU stand-in for TE multi_tensor_scale (and the "cuda" overflow buffer). Returns the list of coefficients."""
    import megatron.core.optimizer.clip_grads as cg

    scaled = []

    def applier(fn, buf, lists, coeff):
        scaled.append(coeff)
        for src, dst in zip(*lists):
            dst.mul_(coeff)

    real_zeros = torch.zeros

    def zeros(*a, **k):
        if k.get("device") == "cuda":
            k["device"] = "cpu"
        return real_zeros(*a, **k)

    monkeypatch.setattr(cg, "multi_tensor_applier", applier)
    monkeypatch.setattr(torch, "zeros", zeros)
    return scaled


def test_fp32_optimizer_step_flushes_deferred_clip(restore_hooks, monkeypatch):
    import megatron.core.optimizer.optimizer as om

    scaled = _cpu_multi_tensor_scale(monkeypatch)
    gen = torch.Generator().manual_seed(2)
    params = [_cl(torch.randn(n, generator=gen)) for n in (16, 5)]
    for p in params:
        p.grad = _cl(torch.randn(p.numel(), generator=gen))
    grads0 = [p.grad.clone() for p in params]
    steps = []
    opt = object.__new__(om.FP32Optimizer)
    opt.__dict__.update(
        is_stub_optimizer=False,
        optimizer=SimpleNamespace(param_groups=[{"params": params}], step=lambda: steps.append(1)),
        config=SimpleNamespace(timers=None, barrier_with_L1_time=False),
    )
    F.install_hooks()
    F.set_fused(True)
    om.clip_grad_by_total_norm_fp32(params, 1.0, 20.0)  # deferred
    assert len(F._PENDING) == 2 and torch.equal(params[0].grad, grads0[0])
    assert om.FP32Optimizer.step_with_ready_grads(opt) is True
    coeff = 1.0 / (20.0 + 1.0e-6)
    assert scaled == [coeff] and steps == [1] and not F._PENDING
    for p, g0 in zip(params, grads0):
        assert torch.equal(p.grad, g0 * coeff)


# ---------------------------------------------------------------- step host logic on CPU stand-ins


def _te_like_fused_adam(groups):
    class FusedAdam:
        capturable = False
        master_weights = False
        adam_w_mode = 1
        exp_avg_dtype = torch.float32
        exp_avg_sq_dtype = torch.float32
        use_decoupled_grad = False

        def __init__(self):
            self.param_groups = groups
            self.state = collections.defaultdict(dict)
            self.steps = 0

        def initialize_state(self, p, store_param_remainders):  # in place, like TE's set_scaled_state
            self.state[p].update(exp_avg=_cl(torch.zeros_like(p)), exp_avg_sq=_cl(torch.zeros_like(p)))

        def step(self):
            self.steps += 1

    FusedAdam.__module__ = "transformer_engine.pytorch.optimizers.fused_adam"
    return FusedAdam()


def _reference_adamw(p, g, m, v, clip, sc):
    """The kernel's math without fma (CPU stand-in for the Triton kernel)."""
    g = g * clip
    m.copy_(sc["beta1"] * m + sc["omb1"] * g)
    v.copy_(sc["beta2"] * v + sc["omb2"] * g * g)
    denom = torch.sqrt(v * sc["bc2_inv"]) + sc["eps"]
    p.copy_(p - sc["step_size"] * (m / denom) - sc["lr_decay"] * p)


def _dist_opt_shell(inner):
    from megatron.core.optimizer import distrib_optimizer as dm

    calls = {"orig": 0}
    opt = object.__new__(dm.DistributedOptimizer)
    opt.__dict__.update(
        is_stub_optimizer=False,
        optimizer=inner,
        ddp_config=SimpleNamespace(use_megatron_fsdp=False),
        config=SimpleNamespace(
            use_precision_aware_optimizer=False,
            timers=None,
            reuse_grad_buf_for_mxfp8_param_ag=False,
            barrier_with_L1_time=False,
        ),
    )

    def orig(self):
        calls["orig"] += 1
        self.optimizer.step()
        return True

    return opt, orig, calls, dm


def test_fused_step_host_logic_with_cpu_stand_ins(restore_hooks, monkeypatch):
    gen = torch.Generator().manual_seed(0)
    params = [_cl(torch.randn(n, generator=gen)) for n in (300, 5000, 64, 7)]
    for p in params:
        p.grad = _cl(torch.randn(p.numel(), generator=gen))
    groups = [
        {"params": params[:2], "lr": 1e-3, "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.1},
        {"params": params[2:], "lr": 1e-3, "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.0},
    ]
    inner = _te_like_fused_adam(groups)
    model = {id(p): torch.full((p.numel(),), float("nan"), dtype=torch.bfloat16) for p in params}
    opt, orig, calls, dm = _dist_opt_shell(inner)
    launched = []

    def kernel(p, g, m, v, out, *, clip, scalars, check=True, **kw):
        launched.append((p.numel(), clip, scalars["step_size"], scalars["lr_decay"]))
        _reference_adamw(p, g, m, v, clip, scalars)
        out.copy_(p)

    monkeypatch.setattr(F, "dst_map", lambda self: model)
    monkeypatch.setattr(FA, "adamw_clip_", kernel)
    clip = F.make_clip_grad_by_total_norm_fp32(lambda *a: pytest.fail("original clip taken"))
    step = F.make_step_with_ready_grads(orig, dm)
    F.set_fused(True)

    ref = [p.detach().clone().as_subclass(torch.Tensor) for p in params]
    ref_m = [torch.zeros_like(r) for r in ref]
    ref_v = [torch.zeros_like(r) for r in ref]
    for k, total_norm in enumerate((2.0, 0.5)):
        clip(params, 1.0, total_norm)
        assert step(opt) is True
        coeff = 1.0 / (2.0 + 1.0e-6) if total_norm > 1 else 1.0
        for gi, grp in enumerate(groups):
            assert grp["step"] == k + 1
            sc = FA.te_scalars(grp["lr"], 0.9, 0.95, grp["eps"], grp["weight_decay"], k + 1)
            for p in grp["params"]:
                i = next(j for j, q in enumerate(params) if q is p)
                _reference_adamw(ref[i], p.grad.as_subclass(torch.Tensor), ref_m[i], ref_v[i], coeff, sc)
        for i, p in enumerate(params):
            assert torch.equal(p.as_subclass(torch.Tensor), ref[i])
            st = inner.state[p]
            assert torch.equal(st["exp_avg"].as_subclass(torch.Tensor), ref_m[i])
            assert torch.equal(st["exp_avg_sq"].as_subclass(torch.Tensor), ref_v[i])
            assert torch.equal(model[id(p)], ref[i].bfloat16())
        # largest first, the deferred coefficient reached every launch, per-group weight decay
        sizes = [x[0] for x in launched[-4:]]
        assert sizes == sorted(sizes, reverse=True)
        assert {x[1] for x in launched[-4:]} == {coeff}
        assert {x[3] for x in launched[-4:]} == {
            FA.te_scalars(1e-3, 0.9, 0.95, 1e-8, 0.1, k + 1)["lr_decay"],
            0.0,
        }
    assert calls["orig"] == 0 and inner.steps == 0 and not F._PENDING


def test_fallback_flushes_deferred_clip_exactly(restore_hooks, monkeypatch):
    gen = torch.Generator().manual_seed(1)
    params = [_cl(torch.randn(n, generator=gen)) for n in (100, 33)]
    for p in params:
        p.grad = _cl(torch.randn(p.numel(), generator=gen))
    grads0 = [p.grad.clone() for p in params]
    groups = [{"params": params, "lr": 1e-3, "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.1}]
    inner = _te_like_fused_adam(groups)
    opt, orig, calls, dm = _dist_opt_shell(inner)
    scaled = _cpu_multi_tensor_scale(monkeypatch)
    clip = F.make_clip_grad_by_total_norm_fp32(lambda *a: pytest.fail("original clip taken"))
    step = F.make_step_with_ready_grads(orig, dm)
    F.set_fused(True)
    clip(params, 1.0, 4.0)
    F.set_fused(False)  # switched off between the clip and the update
    assert step(opt) is True
    coeff = 1.0 / (4.0 + 1.0e-6)
    assert scaled == [coeff] and calls["orig"] == 1 and inner.steps == 1
    for p, g0 in zip(params, grads0):
        assert torch.equal(p.grad, g0 * coeff)
    # ineligible inner optimizer -> fallback reason, original step
    F.set_fused(True)
    opt.optimizer = SimpleNamespace(param_groups=groups, step=lambda: None)
    assert F.ineligible_reason(opt, dm).startswith("inner optimizer")
    assert step(opt) is True and calls["orig"] == 2


def _set(obj, **kw):
    def mutate(opt, inner):
        target = opt if obj == "opt" else inner if obj == "inner" else getattr(opt, obj)
        for k, v in kw.items():
            setattr(target, k, v)
        return opt

    return mutate


def _not_distributed(opt, inner):
    import megatron.core.optimizer.optimizer as om

    other = object.__new__(om.Float16OptimizerWithFloat16Params)
    other.__dict__.update(opt.__dict__)
    return other


@pytest.mark.parametrize(
    "mutate,reason",
    [
        (_set("config", use_precision_aware_optimizer=True), "precision-aware optimizer"),
        (_set("ddp_config", use_megatron_fsdp=True), "megatron fsdp"),
        (_set("opt", is_stub_optimizer=True), "stub optimizer"),
        (_set("config", reuse_grad_buf_for_mxfp8_param_ag=True), "mxfp8 param all-gather"),
        (_set("inner", master_weights=True), "capturable / master-weight FusedAdam"),
        (_set("inner", capturable=True), "capturable / master-weight FusedAdam"),
        (_set("inner", adam_w_mode=0), "not AdamW mode"),
        (_set("inner", exp_avg_dtype=torch.bfloat16), "non-fp32 moments"),
        (_set("inner", exp_avg_sq_dtype=torch.bfloat16), "non-fp32 moments"),
        (_set("inner", use_decoupled_grad=True), "decoupled grads"),
        (_not_distributed, "not a DistributedOptimizer"),
    ],
)
def test_ineligible_setups_run_original_step(restore_hooks, monkeypatch, mutate, reason):
    p = _cl(torch.zeros(8))
    p.grad = _cl(torch.ones(8))
    groups = [{"params": [p], "lr": 1e-3, "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.1}]
    inner = _te_like_fused_adam(groups)
    opt, orig, calls, dm = _dist_opt_shell(inner)
    opt = mutate(opt, inner)
    monkeypatch.setattr(FA, "adamw_clip_", lambda *a, **k: pytest.fail("fused kernel launched"))
    monkeypatch.setattr(F, "dst_map", lambda self: pytest.fail("fused step entered"))
    step = F.make_step_with_ready_grads(orig, dm)
    F.set_fused(True)
    assert F.ineligible_reason(opt, dm) == reason
    assert step(opt) is True
    assert calls["orig"] == 1 and inner.steps == 1
    assert F._STATE["last_fallback"] == reason


def test_fallback_inside_fused_step_updates_nothing(restore_hooks, monkeypatch):
    scaled = _cpu_multi_tensor_scale(monkeypatch)
    gen = torch.Generator().manual_seed(4)
    params = [_cl(torch.randn(n, generator=gen)) for n in (40, 24, 9)]
    for p in params:
        p.grad = _cl(torch.randn(p.numel(), generator=gen))
    groups = [
        {"params": params[:2], "lr": 1e-3, "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.1},
        {"params": params[2:], "lr": 1e-3, "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.0},
    ]
    inner = _te_like_fused_adam(groups)
    for p in params[:2]:  # eligible, already-initialised state of the first group
        inner.state[p].update(
            exp_avg=_cl(torch.randn(p.numel(), generator=gen)),
            exp_avg_sq=_cl(torch.rand(p.numel(), generator=gen)),
        )
    # second group: bf16 exp_avg_sq, found only after the first group was checked
    inner.state[params[2]].update(
        exp_avg=_cl(torch.zeros(9)), exp_avg_sq=_cl(torch.zeros(9, dtype=torch.bfloat16))
    )
    before = {
        id(p): [t.clone() for t in (p, inner.state[p]["exp_avg"], inner.state[p]["exp_avg_sq"])]
        for p in params
    }
    grads0 = [p.grad.clone() for p in params]
    model = {id(p): torch.zeros(p.numel(), dtype=torch.bfloat16) for p in params}
    opt, orig, calls, dm = _dist_opt_shell(inner)
    monkeypatch.setattr(F, "dst_map", lambda self: model)
    monkeypatch.setattr(FA, "adamw_clip_", lambda *a, **k: pytest.fail("fused kernel launched"))
    clip = F.make_clip_grad_by_total_norm_fp32(lambda *a: pytest.fail("original clip taken"))
    step = F.make_step_with_ready_grads(orig, dm)
    F.set_fused(True)
    clip(params, 1.0, 3.0)
    assert step(opt) is True
    coeff = 1.0 / (3.0 + 1.0e-6)
    assert calls["orig"] == 1 and inner.steps == 1
    assert scaled == [coeff] and not F._PENDING  # deferred clip applied once, exactly
    for p, g0 in zip(params, grads0):
        assert torch.equal(p.grad, g0 * coeff)
        for t, t0 in zip((p, inner.state[p]["exp_avg"], inner.state[p]["exp_avg_sq"]), before[id(p)]):
            assert torch.equal(t, t0)
    assert all("step" not in g for g in groups)
    assert all(torch.equal(m, torch.zeros_like(m)) for m in model.values())
    assert F._STATE["last_fallback"].startswith("fallback:")


def _dist_opt_with_buffers():
    """DistributedOptimizer shell with real param buffers / range maps (bf16 buffer with 2 buckets, fp32 buffer)."""
    from megatron.core.optimizer import distrib_optimizer as dm

    bf16_buckets = [torch.zeros(40, dtype=torch.bfloat16), torch.zeros(30, dtype=torch.bfloat16)]
    fp32_buckets = [torch.zeros(25)]
    buffers = [
        SimpleNamespace(params=[], buckets=[SimpleNamespace(param_data=b) for b in bf16_buckets]),
        SimpleNamespace(params=[], buckets=[SimpleNamespace(param_data=b) for b in fp32_buckets]),
    ]
    # model param -> (gbuf index, dtype, bucket, world range in the bucket)
    layout = [
        (torch.zeros(12, dtype=torch.bfloat16), 0, torch.bfloat16, 0, dm.Range(5, 17)),
        (torch.zeros(16, dtype=torch.bfloat16), 0, torch.bfloat16, 0, dm.Range(20, 36)),
        (torch.zeros(10, dtype=torch.bfloat16), 0, torch.bfloat16, 1, dm.Range(3, 13)),
        (torch.zeros(15), 1, torch.float32, 0, dm.Range(4, 19)),
    ]
    gbuf_ranges = [
        {torch.bfloat16: [{"param_map": {}}, {"param_map": {}}]},
        {torch.float32: [{"param_map": {}}]},
    ]
    gbuf_map = {}
    for mp, gi, dt, bi, rng in layout:
        gbuf_map[mp] = (gi, dt, bi)
        gbuf_ranges[gi][dt][bi]["param_map"][mp] = {"gbuf_world_in_bucket": rng}
    mains = [
        _cl(torch.arange(rng.size, dtype=torch.float32) + 100.0 * (i + 1) + 0.25)
        for i, (*_, rng) in enumerate(layout)
    ]
    mps = [x[0] for x in layout]
    opt = object.__new__(dm.DistributedOptimizer)
    opt.__dict__.update(
        is_stub_optimizer=False,
        ddp_config=SimpleNamespace(use_megatron_fsdp=False),
        config=SimpleNamespace(
            use_precision_aware_optimizer=False,
            use_precision_aware_optimizer_no_fp8_or_ds_fp8=False,
            timers=None,
            reuse_grad_buf_for_mxfp8_param_ag=False,
            barrier_with_L1_time=False,
        ),
        data_parallel_group=None,
        buffers=buffers,
        gbuf_ranges=gbuf_ranges,
        model_param_gbuf_map=gbuf_map,
        shard_fp32_from_float16_groups=[[mains[0], mains[1]], [mains[2]]],
        model_float16_groups=[[mps[0], mps[1]], [mps[2]]],
        shard_fp32_groups=[[mains[3]], []],
        model_fp32_groups=[[mps[3]], []],
    )
    return opt, mains, dm


def test_dst_map_matches_megatron_copy_main_params(restore_hooks):
    opt, mains, dm = _dist_opt_with_buffers()
    bufs = [b.param_data for buf in opt.buffers for b in buf.buckets]
    dm.DistributedOptimizer._copy_main_params_to_model_params(opt)  # Megatron's own main -> model copy
    ref = [b.clone() for b in bufs]
    assert all(b.abs().sum() > 0 for b in ref)
    for b in bufs:
        b.zero_()
    dst = F.dst_map(opt)
    assert sorted(dst) == sorted(id(m) for m in mains)
    for m in mains:
        dst[id(m)].copy_(m)
        assert any(dst[id(m)].untyped_storage().data_ptr() == b.untyped_storage().data_ptr() for b in bufs)
    for b, r in zip(bufs, ref):
        assert torch.equal(b, r)
    assert F.dst_map(opt) is dst  # cached while the buffers stay put
    # A new param buffer invalidates the cache.
    bucket = opt.buffers[0].buckets[1]
    bucket.param_data = torch.zeros(30, dtype=torch.bfloat16)
    dst2 = F.dst_map(opt)
    assert dst2 is not dst
    assert dst2[id(mains[2])].untyped_storage().data_ptr() == bucket.param_data.untyped_storage().data_ptr()
    # A range that does not match its main param is refused.
    mp = opt.model_float16_groups[1][0]
    opt.gbuf_ranges[0][torch.bfloat16][1]["param_map"][mp]["gbuf_world_in_bucket"] = dm.Range(3, 12)
    bucket.param_data = torch.zeros(30, dtype=torch.bfloat16)
    with pytest.raises(F._Fallback, match="param range mismatch"):
        F.dst_map(opt)


def test_fused_step_writes_bf16_and_fp32_model_params(restore_hooks, monkeypatch):
    opt, mains, dm = _dist_opt_with_buffers()
    gen = torch.Generator().manual_seed(5)
    for m in mains:
        m.grad = _cl(torch.randn(m.numel(), generator=gen))
    groups = [
        {
            "params": [mains[0], mains[1], mains[3]],
            "lr": 1e-3,
            "betas": (0.9, 0.95),
            "eps": 1e-8,
            "weight_decay": 0.1,
        },
        {"params": [mains[2]], "lr": 1e-3, "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.0},
    ]
    opt.optimizer = _te_like_fused_adam(groups)

    def kernel(p, g, m, v, out, *, clip, scalars, check=True, **kw):
        _reference_adamw(p, g, m, v, clip, scalars)
        if out is not None:
            out.copy_(p)

    monkeypatch.setattr(FA, "adamw_clip_", kernel)
    F.set_fused(True)
    assert F.ineligible_reason(opt, dm) is None
    assert F.fused_step(opt) is True
    dst = F.dst_map(opt)
    for m in mains:
        d = dst[id(m)]
        want = m.as_subclass(torch.Tensor)
        assert torch.equal(d, want.to(d.dtype))
    assert dst[id(mains[3])].dtype == torch.float32  # the fp32 model param took the copy path
    assert all(g["step"] == 1 for g in groups)


def test_te_scalars_follow_te_host_math():
    import numpy as np

    sc = FA.te_scalars(1e-4, 0.9, 0.95, 1e-8, 0.1, 3)
    f = np.float32
    bc1 = f(1.0 - float(f(0.9)) ** 3)
    bc2 = f(1.0 - float(f(0.95)) ** 3)
    assert sc["step_size"] == float(f(1e-4) / bc1)
    assert sc["bc2_inv"] == float(f(1.0) / bc2)
    assert sc["lr_decay"] == float(f(1e-4) * f(0.1))
    assert sc["omb1"] == float(f(1.0) - f(0.9))


# ---------------------------------------------------------------- GPU


@cuda
@pytest.mark.parametrize("clip_coeff", [1.0, 0.37])
def test_fused_kernel_bitwise_vs_te_fused_adam(clip_coeff):
    pytest.importorskip("transformer_engine.pytorch")
    te_opt = pytest.importorskip("transformer_engine.pytorch.optimizers")
    import transformer_engine_torch as tex

    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(3)
    sizes = (1000, 4097, (1 << 16) + 3, 7168)
    p_ref = [torch.randn(n, generator=gen, device=dev) for n in sizes]
    p_fus = [p.clone() for p in p_ref]
    m_fus = [torch.zeros_like(p) for p in p_ref]
    v_fus = [torch.zeros_like(p) for p in p_ref]
    out = [torch.full((n,), float("nan"), device=dev, dtype=torch.bfloat16) for n in sizes]
    groups = [
        {"params": p_ref[:2], "weight_decay": 0.1},
        {"params": p_ref[2:], "weight_decay": 0.0},
    ]
    lr, b1, b2, eps = 3e-4, 0.9, 0.95, 1e-8
    opt = te_opt.FusedAdam(groups, lr=lr, betas=(b1, b2), eps=eps, adam_w_mode=True)
    for step in range(1, 4):
        grads = [torch.randn(n, generator=gen, device=dev) for n in sizes]
        for p, g in zip(p_ref, grads):
            p.grad = g.clone()
        if clip_coeff != 1.0:
            dummy = torch.zeros(1, dtype=torch.int, device=dev)
            te_opt.multi_tensor_applier(
                tex.multi_tensor_scale, dummy, [[p.grad for p in p_ref]] * 2, clip_coeff
            )
        opt.step()
        for i, n in enumerate(sizes):
            wd = 0.1 if i < 2 else 0.0
            sc = FA.te_scalars(lr, b1, b2, eps, wd, step)
            FA.adamw_clip_(p_fus[i], grads[i], m_fus[i], v_fus[i], out[i], clip=clip_coeff, scalars=sc)
        torch.cuda.synchronize()
        for i, p in enumerate(p_ref):
            st = opt.state[p]
            assert torch.equal(p_fus[i].view(torch.int32), p.detach().view(torch.int32)), (step, i)
            assert torch.equal(m_fus[i].view(torch.int32), st["exp_avg"].view(torch.int32)), (step, i)
            assert torch.equal(v_fus[i].view(torch.int32), st["exp_avg_sq"].view(torch.int32)), (step, i)
            assert torch.equal(out[i], p.detach().bfloat16()), (step, i)
