###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Triton AdamW step and global grad NaN check patches: hooks, gates and the step's host logic on CPU
(stand-ins for the GPU leaves), the kernel against TE's FusedAdam on a GPU."""

import collections
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron")
pytest.importorskip("triton")

import primus.backends.megatron.patches.triton_adamw_step_patches as FP  # noqa: E402
from primus.backends.megatron.core.optimizer import (  # noqa: E402
    triton_adamw_kernel as K,
)
from primus.backends.megatron.core.optimizer import triton_adamw_step as F  # noqa: E402
from primus.core.patches.patch_runner import run_patches  # noqa: E402
from tests.unit_tests.backends.megatron.conftest import (  # noqa: E402
    cuda_like,
    patch_ctx,
    registered_patch,
    requires_gpu,
)


def _hook_owners():
    import megatron.core.distributed.param_and_grad_buffer as pb
    import megatron.core.optimizer.optimizer as om

    return (
        (om, "clip_grad_by_total_norm_fp32"),
        (om.MixedPrecisionOptimizer, "step_with_ready_grads"),
        (om.FP32Optimizer, "step_with_ready_grads"),
        (pb._ParamAndGradBucketGroup, "check_grads"),
    )


def _hooked():
    return tuple(getattr(owner, attr) for owner, attr in _hook_owners())


@pytest.fixture
def restore_hooks(restore_patched):
    """Undo the hooks and the switches after a test."""
    for owner, attr in _hook_owners():
        restore_patched(owner, attr)
    yield
    F.set_fused(False)
    F.set_nancheck(False)
    F._PENDING.clear()


def _record_fallbacks(monkeypatch):
    warned = []
    monkeypatch.setattr(F, "warn_fallback_once", lambda key, why: warned.append(why))
    return warned


def test_hooks_install_once(restore_hooks):
    before = _hooked()
    FP.patch_triton_adamw_step(patch_ctx(use_primus_triton_adamw_step=True, use_distributed_optimizer=True))
    assert F.fused_enabled() and not F.nancheck_enabled()
    hooked = _hooked()
    assert all(h.__wrapped__ is b for h, b in zip(hooked, before))
    FP.patch_global_grad_nan_check(
        patch_ctx(use_primus_global_grad_nan_check=True, check_for_nan_in_loss_and_grad=True, clip_grad=1.0)
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
def test_step_patch_not_enabled_for_uncovered_setups(restore_hooks, monkeypatch, args, reason):
    warned = []
    monkeypatch.setattr(FP, "warning_rank_0", warned.append)
    before = _hooked()
    FP.patch_triton_adamw_step(patch_ctx(use_primus_triton_adamw_step=True, **args))
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
    FP.patch_global_grad_nan_check(patch_ctx(use_primus_global_grad_nan_check=True, **args))
    assert not F.nancheck_enabled()
    assert len(warned) == 1 and reason in warned[0]


def test_nan_check_patch_enabled_with_rerun_mode_disabled(restore_hooks):
    FP.patch_global_grad_nan_check(
        patch_ctx(
            use_primus_global_grad_nan_check=True,
            check_for_nan_in_loss_and_grad=True,
            clip_grad=1.0,
            rerun_mode="disabled",
        )
    )
    assert F.nancheck_enabled()


@pytest.mark.parametrize(
    "key,arg",
    [
        ("megatron.optimizer.triton_adamw_step", "use_primus_triton_adamw_step"),
        ("megatron.ddp.global_grad_nan_check", "use_primus_global_grad_nan_check"),
    ],
)
def test_exclusive_with_te_fused_adam_clip_aborts_the_run(restore_hooks, monkeypatch, key, arg):
    # run_patches swallows Exception: the conflict must surface as SystemExit.
    registered_patch(key, FP)
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


def test_clip_defers_and_global_nan_check_raises(restore_hooks):
    calls = []
    clip = F.make_clip_grad_by_total_norm_fp32(lambda *a: calls.append(a))
    p = cuda_like(torch.zeros(8))
    p.grad = cuda_like(torch.ones(8))
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
    p = cuda_like(torch.zeros(8))
    p.grad = None
    p.decoupled_grad = cuda_like(torch.ones(8))
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
    p = cuda_like(torch.zeros(8))
    p.grad = cuda_like(torch.ones(8))
    p.decoupled_grad = cuda_like(torch.ones(8))
    F.set_fused(False)
    F.set_nancheck(True)
    for bad in (float("nan"), float("inf")):
        with pytest.raises(RuntimeError, match="non-finite global grad norm"):
            clip([p], 1.0, bad, use_decoupled_grad)
    assert calls == []
    clip([p], 1.0, 2.0, use_decoupled_grad)
    assert len(calls) == 1 and calls[0][0][0] is p and calls[0][1:] == (1.0, 2.0, use_decoupled_grad)


def _cpu_multi_tensor_scale(monkeypatch):
    """CPU stand-in for TE multi_tensor_scale. Returns the list of coefficients it applied."""
    import megatron.core.optimizer.clip_grads as cg

    scaled = []

    def applier(fn, buf, lists, coeff):
        scaled.append(coeff)
        for src, dst in zip(*lists):
            dst.mul_(coeff)

    monkeypatch.setattr(cg, "multi_tensor_applier", applier)
    return scaled


def _params(gen, *sizes):
    params = [cuda_like(torch.randn(n, generator=gen)) for n in sizes]
    for p in params:
        p.grad = cuda_like(torch.randn(p.numel(), generator=gen))
    return params


def _group(params, wd=0.1):
    return {"params": params, "lr": 1e-3, "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": wd}


def test_fp32_optimizer_step_flushes_deferred_clip(restore_hooks, monkeypatch):
    import megatron.core.optimizer.optimizer as om

    scaled = _cpu_multi_tensor_scale(monkeypatch)
    params = _params(torch.Generator().manual_seed(2), 16, 5)
    grads0 = [p.grad.clone() for p in params]
    steps = []
    opt = object.__new__(om.FP32Optimizer)
    opt.__dict__.update(
        is_stub_optimizer=False,
        optimizer=SimpleNamespace(param_groups=[{"params": params}], step=lambda: steps.append(1)),
        config=SimpleNamespace(timers=None, barrier_with_L1_time=False),
    )
    FP._install_hooks()
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
            self.state[p].update(
                exp_avg=cuda_like(torch.zeros_like(p)), exp_avg_sq=cuda_like(torch.zeros_like(p))
            )

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
    params = _params(torch.Generator().manual_seed(0), 300, 5000, 64, 7)
    groups = [_group(params[:2]), _group(params[2:], wd=0.0)]
    inner = _te_like_fused_adam(groups)
    model = {id(p): torch.full((p.numel(),), float("nan"), dtype=torch.bfloat16) for p in params}
    opt, orig, calls, dm = _dist_opt_shell(inner)
    launched = []

    def kernel(p, g, m, v, out, *, clip, scalars, check=True, **kw):
        launched.append((p.numel(), clip, scalars["step_size"], scalars["lr_decay"]))
        _reference_adamw(p, g, m, v, clip, scalars)
        out.copy_(p)

    monkeypatch.setattr(F, "dst_map", lambda self: model)
    monkeypatch.setattr(K, "adamw_clip_", kernel)
    clip = F.make_clip_grad_by_total_norm_fp32(lambda *a: pytest.fail("original clip taken"))
    step = F.make_step_with_ready_grads(orig, dm, "key")
    F.set_fused(True)

    ref = [p.detach().clone().as_subclass(torch.Tensor) for p in params]
    ref_m = [torch.zeros_like(r) for r in ref]
    ref_v = [torch.zeros_like(r) for r in ref]
    for k, total_norm in enumerate((2.0, 0.5)):
        clip(params, 1.0, total_norm)
        assert step(opt) is True
        coeff = 1.0 / (2.0 + 1.0e-6) if total_norm > 1 else 1.0
        for grp in groups:
            assert grp["step"] == k + 1
            sc = K.te_scalars(grp["lr"], 0.9, 0.95, grp["eps"], grp["weight_decay"], k + 1)
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
            K.te_scalars(1e-3, 0.9, 0.95, 1e-8, 0.1, k + 1)["lr_decay"],
            0.0,
        }
    assert calls["orig"] == 0 and inner.steps == 0 and not F._PENDING


@pytest.mark.parametrize("where", ["switched off", "ineligible optimizer", "inside the fused step"])
def test_fallback_flushes_the_deferred_clip_and_updates_nothing(restore_hooks, monkeypatch, where):
    scaled = _cpu_multi_tensor_scale(monkeypatch)
    warned = _record_fallbacks(monkeypatch)
    gen = torch.Generator().manual_seed(4)
    params = _params(gen, 40, 24, 9)
    groups = [_group(params[:2]), _group(params[2:], wd=0.0)]
    inner = _te_like_fused_adam(groups)
    for p in params[:2]:  # eligible, already-initialised state of the first group
        inner.state[p].update(
            exp_avg=cuda_like(torch.randn(p.numel(), generator=gen)),
            exp_avg_sq=cuda_like(torch.rand(p.numel(), generator=gen)),
        )
    # "inside the fused step": a bf16 exp_avg_sq in the second group, found after the first group was checked
    v_dtype = torch.bfloat16 if where == "inside the fused step" else torch.float32
    inner.state[params[2]].update(
        exp_avg=cuda_like(torch.zeros(9)), exp_avg_sq=cuda_like(torch.zeros(9, dtype=v_dtype))
    )
    before = {
        id(p): [t.clone() for t in (p, inner.state[p]["exp_avg"], inner.state[p]["exp_avg_sq"])]
        for p in params
    }
    grads0 = [p.grad.clone() for p in params]
    model = {id(p): torch.zeros(p.numel(), dtype=torch.bfloat16) for p in params}
    opt, orig, calls, dm = _dist_opt_shell(inner)
    monkeypatch.setattr(F, "dst_map", lambda self: model)
    monkeypatch.setattr(K, "adamw_clip_", lambda *a, **k: pytest.fail("fused kernel launched"))
    clip = F.make_clip_grad_by_total_norm_fp32(lambda *a: pytest.fail("original clip taken"))
    step = F.make_step_with_ready_grads(orig, dm, "key")
    F.set_fused(True)
    clip(params, 1.0, 3.0)
    if where == "switched off":  # between the clip and the update
        F.set_fused(False)
    elif where == "ineligible optimizer":
        opt.optimizer = SimpleNamespace(param_groups=groups, step=inner.step)
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
    assert (
        warned
        == {
            "switched off": [],
            "ineligible optimizer": ["inner optimizer types.SimpleNamespace"],
            "inside the fused step": ["fallback: tensor layout"],
        }[where]
    )


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
    warned = _record_fallbacks(monkeypatch)
    p = cuda_like(torch.zeros(8))
    p.grad = cuda_like(torch.ones(8))
    inner = _te_like_fused_adam([_group([p])])
    opt, orig, calls, dm = _dist_opt_shell(inner)
    opt = mutate(opt, inner)
    monkeypatch.setattr(K, "adamw_clip_", lambda *a, **k: pytest.fail("fused kernel launched"))
    monkeypatch.setattr(F, "dst_map", lambda self: pytest.fail("fused step entered"))
    step = F.make_step_with_ready_grads(orig, dm, "key")
    F.set_fused(True)
    assert F.ineligible_reason(opt, dm) == reason
    assert step(opt) is True
    assert calls["orig"] == 1 and inner.steps == 1 and warned == [reason]


def _dist_opt_with_buffers():
    """DistributedOptimizer shell with real param buffers and range maps (bf16: 2 buckets, fp32: 1 bucket)."""
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
        cuda_like(torch.arange(rng.size, dtype=torch.float32) + 100.0 * (i + 1) + 0.25)
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
        m.grad = cuda_like(torch.randn(m.numel(), generator=gen))
    groups = [_group([mains[0], mains[1], mains[3]]), _group([mains[2]], wd=0.0)]
    opt.optimizer = _te_like_fused_adam(groups)

    def kernel(p, g, m, v, out, *, clip, scalars, check=True, **kw):
        _reference_adamw(p, g, m, v, clip, scalars)
        if out is not None:
            out.copy_(p)

    monkeypatch.setattr(K, "adamw_clip_", kernel)
    F.set_fused(True)
    assert F.ineligible_reason(opt, dm) is None
    F.fused_step(opt)
    dst = F.dst_map(opt)
    for m in mains:
        d = dst[id(m)]
        assert torch.equal(d, m.as_subclass(torch.Tensor).to(d.dtype))
    assert dst[id(mains[3])].dtype == torch.float32  # the fp32 model param took the copy path
    assert all(g["step"] == 1 for g in groups)


def test_te_scalars_follow_te_host_math():
    import numpy as np

    sc = K.te_scalars(1e-4, 0.9, 0.95, 1e-8, 0.1, 3)
    f = np.float32
    bc1 = f(1.0 - float(f(0.9)) ** 3)
    bc2 = f(1.0 - float(f(0.95)) ** 3)
    assert sc["step_size"] == float(f(1e-4) / bc1)
    assert sc["bc2_inv"] == float(f(1.0) / bc2)
    assert sc["lr_decay"] == float(f(1e-4) * f(0.1))
    assert sc["omb1"] == float(f(1.0) - f(0.9))


# ---------------------------------------------------------------- GPU


@requires_gpu
@pytest.mark.parametrize("clip_coeff", [1.0, 0.37])
def test_fused_kernel_bitwise_vs_te_fused_adam(clip_coeff):
    pytest.importorskip("transformer_engine.pytorch")
    te_opt = pytest.importorskip("transformer_engine.pytorch.optimizers")
    import transformer_engine_torch as tex

    dev = torch.device("cuda")
    bitwise = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1250"
    gen = torch.Generator(device=dev).manual_seed(3)
    sizes = (1000, 4097, (1 << 16) + 3, 7168)  # ragged: not multiples of BLOCK, one above 2^16
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
            sc = K.te_scalars(lr, b1, b2, eps, wd, step)
            K.adamw_clip_(p_fus[i], grads[i], m_fus[i], v_fus[i], out[i], clip=clip_coeff, scalars=sc)
        torch.cuda.synchronize()
        for i, p in enumerate(p_ref):
            st = opt.state[p]
            pairs = ((p_fus[i], p.detach()), (m_fus[i], st["exp_avg"]), (v_fus[i], st["exp_avg_sq"]))
            if bitwise:
                for got, want in pairs:
                    assert torch.equal(got.view(torch.int32), want.view(torch.int32)), (step, i)
                assert torch.equal(out[i], p.detach().bfloat16()), (step, i)
            else:  # the fma placement reproduces TE's kernel as compiled for gfx1250; elsewhere last-bit ties
                for got, want in pairs:
                    torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-7)
