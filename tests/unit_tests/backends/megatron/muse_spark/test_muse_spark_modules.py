###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark module behaviour on CPU: fp32 islands through casts, the bda closure,
and the build_args patches (selective recompute, CP comm type)."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron.core")

from primus.backends.megatron.core.transformer.muse_spark.moe import (  # noqa: E402
    MuseSparkTopKRouter,
)
from primus.backends.megatron.core.transformer.muse_spark.norms import (  # noqa: E402
    MuseSparkCenteredRMSNorm,
    MuseSparkGatedResidual,
    MuseSparkInputScaledRMSNorm,
    MuseSparkRMSNorm,
    casts_to_half,
)
from primus.backends.megatron.patches.args import (  # noqa: E402
    muse_spark_recompute_cp_patches as rc,
)


def _bare_router(weight):
    r = MuseSparkTopKRouter.__new__(MuseSparkTopKRouter)
    torch.nn.Module.__init__(r)
    r.weight = torch.nn.Parameter(weight)
    return r


def test_router_weight_survives_bf16_cast_exactly():
    w = torch.full((4, 8), 1.0 + 1e-3)  # not representable in bf16
    r = _bare_router(w.clone())
    r.bfloat16()
    assert r.weight.dtype == torch.float32
    assert torch.equal(r.weight.detach(), w)
    r.half()
    assert torch.equal(r.weight.detach(), w)


def test_router_weight_built_in_bf16_is_promoted():
    r = _bare_router(torch.ones(2, 3, dtype=torch.bfloat16))
    r.bfloat16()
    assert r.weight.dtype == torch.float32


def _gated_residual(kind="attn", init=0.0):
    cfg = SimpleNamespace(
        hidden_size=8,
        post_norm_eps=1e-8,
        residual_gate_temperature=0.3,
        residual_gate_init_value=init,
        sequence_parallel=False,
        normalization="RMSNorm",
        layernorm_epsilon=1e-5,
        memory_efficient_layer_norm=False,
        layernorm_zero_centered_gamma=False,
    )
    return MuseSparkGatedResidual(cfg, kind=kind)


def test_gate_init_value_starts_near_identity_residual():
    # 0.0: zero (HF-like) init
    assert torch.equal(_gated_residual().gate.detach(), torch.zeros(8))
    from primus.backends.megatron.core.models.muse_spark.muse_spark_transformer_config import (
        MuseSparkTransformerConfig,
    )

    # the config's default init (not a literal) must start near an identity residual
    init = MuseSparkTransformerConfig.__dataclass_fields__["residual_gate_init_value"].default
    cfg = SimpleNamespace(
        hidden_size=8,
        post_norm_eps=1e-8,
        residual_gate_temperature=0.3,
        residual_gate_init_value=init,
        sequence_parallel=False,
    )
    m = MuseSparkGatedResidual(cfg, kind="attn")
    assert torch.equal(m.gate.detach(), torch.full((8,), float(init)))
    beta = torch.sigmoid(m.gate / 0.3)
    alpha = torch.sqrt(torch.sigmoid(-m.gate / 0.3) * (1 + beta))
    assert float(beta.max()) < 0.05 and abs(float(alpha.mean()) - 1.0) < 0.01
    with torch.no_grad():
        m.gate.zero_()
    m.reset_parameters()
    assert torch.equal(m.gate.detach(), torch.full((8,), -0.9))


@pytest.mark.parametrize("kind", ["attn", "mlp"])
def test_gate_stays_fp32_and_bda_closure_reenters_module(kind):
    m = _gated_residual(kind)
    with torch.no_grad():
        m.gate.fill_(0.5 + 1e-4)  # not representable in bf16
    before = m.gate.detach().clone()
    m.bfloat16()
    assert m.gate.dtype == torch.float32 and torch.equal(m.gate.detach(), before)
    calls = []
    m.register_forward_pre_hook(lambda mod, args: calls.append(len(args)))
    bda = m(True, False)
    x, res = torch.randn(3, 8, dtype=torch.bfloat16), torch.randn(3, 8, dtype=torch.bfloat16)
    out = bda((x, None), res, 0.0)
    # hooks fire once per bda use: only around the tensor call that gates (FSDP2 unshards
    # the gate there); the bool-only factory call does not go through nn.Module.__call__
    assert calls == [4]
    assert callable(m.forward(True, False))
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.double(), _gated_reference(m, kind, x, res), **BF16_TOL)


# two bf16 ulps (2 * 2**-8) relative, plus an absolute floor for values near 0
BF16_TOL = dict(rtol=2 * 2**-8, atol=2 * 2**-8)


def _gated_reference(m, kind, branch, residual):
    """Independent fp64 reference written from the HF formulas, not from the module (kept
    in fp64: callers compare the module output as ``out.double()``):
    ``alpha * residual + beta * post_norm(branch)`` with ``w = gate / T``,
    ``beta = sigmoid(w)``, ``alpha = sqrt(max(sigmoid(-w) * (1 + beta), 1e-3))`` and
    post_norm the weightless RMSNorm (attn) or ``rms(x * (1 + scale))`` (mlp)."""
    b = branch.double()
    if kind == "mlp":
        b = b * (1.0 + m.post_norm.weight.detach().double())
    b = b / torch.sqrt(b.pow(2).mean(-1, keepdim=True) + 1e-8)
    w = m.gate.detach().double() / 0.3
    beta = 1.0 / (1.0 + torch.exp(-w))
    alpha = torch.sqrt(torch.clamp((1.0 / (1.0 + torch.exp(w))) * (1.0 + beta), min=1e-3))
    return alpha * residual.double() + beta * b


@pytest.mark.parametrize("kind", ["attn", "mlp"])
def test_gated_residual_matches_an_independent_fp64_reference(kind):
    torch.manual_seed(0)
    m = _gated_residual(kind)
    with torch.no_grad():
        m.gate.copy_(torch.linspace(-2.0, 2.0, 8))  # alphas / betas far from 0.5
        if kind == "mlp":
            m.post_norm.weight.copy_(torch.linspace(-0.5, 0.5, 8))
    m.bfloat16()
    x = torch.randn(5, 8, dtype=torch.bfloat16) * 3
    res = torch.randn(5, 8, dtype=torch.bfloat16)
    out = m(True, False)((x, None), res, 0.0)
    ref = _gated_reference(m, kind, x, res)
    # bf16 output vs the full-precision reference: within the bf16 rounding of the output
    # and of the bf16 post-norm (the module's only rounding points)
    torch.testing.assert_close(out.double(), ref, **BF16_TOL)
    # the reference is not trivially close: a wrong gate (residual * beta only) fails it
    wrong = torch.sigmoid(m.gate.detach().double() / 0.3) * res.double()
    assert not torch.allclose(wrong, ref, **BF16_TOL)


def _ctx(**args):
    return SimpleNamespace(extra={"backend_args": SimpleNamespace(**args)})


def test_selective_recompute_drops_leftovers_and_warns(monkeypatch):
    msgs = []
    monkeypatch.setattr(rc, "warning_rank_0", msgs.append)
    ctx = _ctx(recompute_granularity="selective", recompute_method="uniform", recompute_num_layers=1)
    rc.patch_muse_spark_selective_recompute(ctx)
    a = ctx.extra["backend_args"]
    assert a.recompute_method is None and a.recompute_num_layers is None
    assert (
        len(msgs) == 1
        and "'recompute_method': 'uniform'" in msgs[0]
        and "'recompute_num_layers': 1" in msgs[0]
    )


def test_full_recompute_untouched(monkeypatch):
    monkeypatch.setattr(rc, "warning_rank_0", lambda m: pytest.fail(m))
    ctx = _ctx(recompute_granularity="full", recompute_method="uniform", recompute_num_layers=1)
    rc.patch_muse_spark_selective_recompute(ctx)
    assert ctx.extra["backend_args"].recompute_method == "uniform"


def test_missing_backend_args_raises():
    with pytest.raises(RuntimeError, match="backend_args"):
        rc.patch_muse_spark_selective_recompute(SimpleNamespace(extra={}))


@pytest.mark.parametrize(
    "comm, expected",
    [
        ("p2p", "all_gather"),  # language_model.yaml default
        (["p2p"], ["all_gather"]),  # Megatron argparse default
        ("allgather", "all_gather"),
        ("all_gather", "all_gather"),
        ("a2a", "a2a"),
        (["all_gather", "p2p", "a2a"], ["all_gather", "all_gather", "a2a"]),
        (None, "all_gather"),
        ([None], ["all_gather"]),
    ],
)
def test_cp_comm_resolution(comm, expected):
    assert rc.resolve_cp_comm_type(comm) == expected


@pytest.mark.parametrize(
    "comm, reason",
    [
        ("a2a+p2p", "no sliding-window backend"),
        (["all_gather", "a2a+p2p"], "no sliding-window backend"),
        ("ring", "unknown cp_comm_type entry 'ring'.*accepted"),
        ("allgather2", "unknown cp_comm_type entry 'allgather2'.*accepted"),
    ],
)
def test_cp_comm_rejected(comm, reason):
    with pytest.raises(ValueError, match=reason):
        rc.resolve_cp_comm_type(comm)


def test_cp_comm_patch_only_with_window_and_cp(monkeypatch):
    msgs = []
    monkeypatch.setattr(rc, "warning_rank_0", msgs.append)
    ctx = _ctx(context_parallel_size=1, window_size=(2048, 0), cp_comm_type=["p2p"])
    rc.patch_muse_spark_cp_comm(ctx)  # CP 1: nothing to check
    assert ctx.extra["backend_args"].cp_comm_type == ["p2p"] and msgs == []
    ctx = _ctx(context_parallel_size=2, window_size=None, cp_comm_type=["p2p"])
    rc.patch_muse_spark_cp_comm(ctx)  # no sliding window: p2p is fine
    assert ctx.extra["backend_args"].cp_comm_type == ["p2p"] and msgs == []
    ctx = _ctx(context_parallel_size=2, window_size=(2048, 0), cp_comm_type=["p2p"])
    rc.patch_muse_spark_cp_comm(ctx)
    assert ctx.extra["backend_args"].cp_comm_type == ["all_gather"] and len(msgs) == 1
    ctx = _ctx(context_parallel_size=2, window_size=(2048, 0), cp_comm_type=["all_gather"])
    rc.patch_muse_spark_cp_comm(ctx)
    assert len(msgs) == 1  # unchanged: no warning
    assert "no sliding-window attention backend" in msgs[0]
    # the TE-spelling rewrite says so, not that p2p lacks a backend
    rc.patch_muse_spark_cp_comm(
        _ctx(context_parallel_size=2, window_size=(2048, 0), cp_comm_type="allgather")
    )
    assert "spelled 'all_gather'" in msgs[1] and "p2p" not in msgs[1]
    with pytest.raises(ValueError):
        rc.patch_muse_spark_cp_comm(
            _ctx(context_parallel_size=2, window_size=(2048, 0), cp_comm_type="a2a+p2p")
        )


def test_fused_linear_ce_is_allowed():
    # the chunked linear+CE calls model._scale_logits (soft-cap), so nothing rejects it
    assert not hasattr(rc, "patch_muse_spark_no_fused_linear_ce")


def test_merged_args_reads_backend_args_first_then_params():
    from primus.backends.megatron.core.models.muse_spark.model_types import (
        merged_args_of,
    )

    params = SimpleNamespace(use_turbo_deepep=True, expert_model_parallel_size=8)
    ctx = _ctx(expert_model_parallel_size=1)
    ctx.extra["module_config"] = SimpleNamespace(params=params)
    m = merged_args_of(ctx)
    assert m.expert_model_parallel_size == 1 and m.use_turbo_deepep is True
    assert getattr(m, "missing", "dflt") == "dflt"
    m.use_turbo_deepep = False  # written where the key lives
    assert params.use_turbo_deepep is False and not hasattr(ctx.extra["backend_args"], "use_turbo_deepep")


def test_deepep_condition_sees_the_backend_args_ep_override(monkeypatch):
    # the yaml (params) says EP 8, the CLI (backend_args) says EP 1: DeepEP must go off
    monkeypatch.setattr(rc, "warning_rank_0", lambda m: None)
    params = SimpleNamespace(
        use_turbo_deepep=True,
        enable_primus_turbo=True,
        expert_model_parallel_size=8,
        expert_tensor_parallel_size=1,
        tensor_model_parallel_size=1,
    )
    ctx = _ctx(expert_model_parallel_size=1, expert_tensor_parallel_size=1, tensor_model_parallel_size=1)
    ctx.extra["module_config"] = SimpleNamespace(params=params)
    rc.patch_muse_spark_deepep_group_size(ctx)
    assert params.use_turbo_deepep is False


@pytest.mark.parametrize(
    "params, backend, found",
    [
        (dict(turbo_sync_free_moe_stage=1), {}, ["turbo_sync_free_moe_stage=1"]),
        (dict(turbo_sync_free_moe_stage="3"), {}, ["turbo_sync_free_moe_stage=3"]),
        (
            dict(enable_primus_turbo=True, moe_use_fused_router_with_aux_score=True),
            {},
            ["moe_use_fused_router_with_aux_score=True"],
        ),
        ({}, dict(turbo_sync_free_moe_stage=2), ["turbo_sync_free_moe_stage=2"]),
    ],
)
def test_fused_router_is_rejected(params, backend, found):
    ctx = _ctx(**backend)
    ctx.extra["module_config"] = SimpleNamespace(params=SimpleNamespace(**params))
    with pytest.raises(ValueError, match="e_score_correction_bias") as err:
        rc.patch_muse_spark_no_fused_router(ctx)
    for f in found:
        assert f in str(err.value)


@pytest.mark.parametrize(
    "params",
    [
        dict(
            turbo_sync_free_moe_stage=0, enable_primus_turbo=True, moe_use_fused_router_with_aux_score=False
        ),
        dict(turbo_sync_free_moe_stage=None),
        dict(enable_primus_turbo=False, moe_use_fused_router_with_aux_score=True),  # no Turbo router then
        {},
    ],
)
def test_stage_0_and_the_unfused_router_pass(params):
    ctx = _ctx()
    ctx.extra["module_config"] = SimpleNamespace(params=SimpleNamespace(**params))
    rc.patch_muse_spark_no_fused_router(ctx)
    rc.patch_muse_spark_no_fused_router(_ctx())  # no module_config: backend_args only


def test_backend_args_helper_is_public_in_model_types():
    from primus.backends.megatron.core.models.muse_spark import model_types
    from primus.backends.megatron.patches.args import (
        muse_spark_layer_patterns_patches as lpp,
    )

    assert rc.backend_args_of is model_types.backend_args_of is lpp.backend_args_of
    assert not hasattr(lpp, "_backend_args")


def test_island_opt_in_attribute():
    # Primus's FSDP2 wrapper shards fp32 params as their own group only when opted in
    assert MuseSparkTopKRouter.fsdp2_dtype_island is True
    assert MuseSparkGatedResidual.fsdp2_dtype_island is True


def test_casts_to_half_probe():
    assert casts_to_half(lambda t: t.bfloat16())
    assert casts_to_half(lambda t: t.half())
    assert not casts_to_half(lambda t: t.float())
    assert not casts_to_half(lambda t: t.double())
    assert not casts_to_half(lambda t: t.to("cpu"))


def test_router_weight_follows_device_and_fp64_casts():
    # R5-11: a pure device move or a non-half cast is not undone.
    r = _bare_router(torch.ones(2, 3))
    r.double()
    assert r.weight.dtype == torch.float64
    r.to("cpu")
    assert r.weight.dtype == torch.float64
    r.float()
    r.bfloat16()
    assert r.weight.dtype == torch.float32


@pytest.mark.parametrize("cls", [MuseSparkRMSNorm, MuseSparkCenteredRMSNorm, MuseSparkInputScaledRMSNorm])
def test_norms_build_without_hidden_size(cls):
    # R5-12: hidden_size=None falls back to config.hidden_size, then 1.
    assert cls(config=SimpleNamespace(hidden_size=6, sequence_parallel=False)).weight.shape == (6,)
    assert cls(config=None).weight.shape == (1,)


def test_expert_class_follows_the_provider_base():
    from megatron.core.transformer.moe.experts import SequentialMLP, TEGroupedMLP

    from primus.backends.megatron.core.transformer.muse_spark import moe

    assert moe.muse_spark_expert_cls(TEGroupedMLP) is moe.MuseSparkGroupedMLP
    assert moe.muse_spark_expert_cls(SequentialMLP) is moe.MuseSparkSequentialMLP
    # needs no Turbo import: runs (and fails) whether or not Primus-Turbo is installed
    with pytest.raises(TypeError, match="no expert variant"):
        moe.muse_spark_expert_cls(torch.nn.Linear)


def test_expert_class_follows_the_turbo_provider_base():
    pytest.importorskip("primus_turbo")
    from primus.backends.megatron.core.transformer.experts import PrimusGroupedMLP
    from primus.backends.megatron.core.transformer.muse_spark import moe

    turbo = moe.muse_spark_expert_cls(PrimusGroupedMLP)
    assert issubclass(turbo, PrimusGroupedMLP) and issubclass(turbo, moe._PostExpertNormMixin)
    assert moe.muse_spark_expert_cls(PrimusGroupedMLP) is turbo


def test_layer_spec_looks_up_the_patched_provider_at_call_time(monkeypatch):
    from megatron.core.extensions import transformer_engine_spec_provider as tsp
    from megatron.core.transformer.moe.experts import SequentialMLP

    from primus.backends.megatron.core.models.muse_spark import (
        muse_spark_layer_specs as specs,
    )
    from primus.backends.megatron.core.transformer.muse_spark import moe

    class _Provider:
        def grouped_mlp_modules(self, grouped):
            return SequentialMLP, None

        def activation_func(self):
            return None

    monkeypatch.setattr(tsp, "TESpecProvider", _Provider)  # what the Turbo patch does
    spec = specs.get_muse_spark_moe_spec(grouped_gemm=True)
    assert spec.submodules.experts.module is moe.MuseSparkSequentialMLP


def test_latent_dispatch_config_is_a_live_latent_width_view():
    from types import SimpleNamespace

    from primus.backends.megatron.core.transformer.muse_spark.moe import (
        latent_dispatch_config,
    )

    cfg = SimpleNamespace(hidden_size=8192, moe_latent_size=2048, num_moe_experts=16, moe_router_topk=4)
    out = latent_dispatch_config(cfg)
    assert out.hidden_size == 2048 and out.num_moe_experts == 16 and cfg.hidden_size == 8192
    # a later change to the shared config is seen (a copy would go stale)
    cfg.moe_router_topk = 8
    cfg.moe_latent_size = 1024
    assert out.moe_router_topk == 8 and out.hidden_size == 1024
    with pytest.raises(AttributeError):
        out.hidden_size = 8192


def test_latent_dispatch_config_copies_and_pickles():
    import copy
    import pickle
    from types import SimpleNamespace

    from primus.backends.megatron.core.transformer.muse_spark.moe import (
        _LatentWidthConfig,
        latent_dispatch_config,
    )

    cfg = SimpleNamespace(hidden_size=8192, moe_latent_size=2048)
    out = latent_dispatch_config(cfg)
    deep = copy.deepcopy(out)
    assert deep.hidden_size == 2048 and deep._base is not cfg
    assert copy.copy(out)._base is cfg
    assert pickle.loads(pickle.dumps(out)).hidden_size == 2048
    bare = object.__new__(_LatentWidthConfig)  # no __init__: no recursion, a plain AttributeError
    with pytest.raises(AttributeError):
        bare.anything


def test_deepep_dispatcher_constructor_does_not_size_from_hidden_size():
    """The latent view is set after ``MoELayer.__init__`` builds the dispatcher. That is
    only right while the constructor sizes nothing from ``hidden_size``; this fails if
    it starts to."""
    import ast
    import pathlib

    import primus

    src = (
        pathlib.Path(primus.__file__).parent / "backends/megatron/core/extensions/primus_turbo.py"
    ).read_text()
    tree = ast.parse(src)
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PrimusTurboDeepEPTokenDispatcher"
    )
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    used = {n.attr for n in ast.walk(init) if isinstance(n, ast.Attribute)}
    assert "hidden_size" not in used


def test_deepep_is_turned_off_only_at_tp_ep_1():
    from types import SimpleNamespace

    def a(tp, ep, deepep=True, turbo=True, etp=1):
        return SimpleNamespace(
            tensor_model_parallel_size=tp,
            expert_model_parallel_size=ep,
            expert_tensor_parallel_size=etp,
            use_turbo_deepep=deepep,
            enable_primus_turbo=turbo,
        )

    assert _deepep_off(a(1, 1))
    # the dispatcher's TP is the expert TP (Muse Spark: 1), so TP 2 with EP 1 is still 1
    assert _deepep_off(a(2, 1))
    assert _deepep_on(a(1, 8)) and _deepep_on(a(2, 1, etp=None))
    assert _deepep_on(a(1, 1, deepep=False)) and _deepep_on(a(1, 1, turbo=False))


def test_deepep_is_turned_off_when_the_expert_group_spans_nodes(monkeypatch):
    from primus.backends.megatron.patches.args.muse_spark_recompute_cp_patches import (
        deepep_disable_reason,
    )

    def a(ep, etp=1):
        return SimpleNamespace(
            tensor_model_parallel_size=1,
            expert_model_parallel_size=ep,
            expert_tensor_parallel_size=etp,
            use_turbo_deepep=True,
            enable_primus_turbo=True,
        )

    monkeypatch.delenv("PRIMUS_TURBO_DEEPEP_INTERNODE", raising=False)
    assert _deepep_on(a(8), local_gpus=8)
    assert "num_rdma_ranks" in deepep_disable_reason(a(64), local_gpus=8)
    assert "PRIMUS_TURBO_DEEPEP_INTERNODE=1" in deepep_disable_reason(a(64), local_gpus=8)
    # a Primus-Turbo build with rocSHMEM: internode dispatch, keep DeepEP across nodes
    monkeypatch.setenv("PRIMUS_TURBO_DEEPEP_INTERNODE", "1")
    assert _deepep_on(a(64), local_gpus=8)
    assert _deepep_off(a(1), local_gpus=8)  # the group-1 rule still holds
    monkeypatch.delenv("PRIMUS_TURBO_DEEPEP_INTERNODE")
    assert _deepep_off(a(8, etp=2), local_gpus=8)
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    monkeypatch.setenv("GPUS_PER_NODE", "4")
    assert _deepep_off(a(8)) and _deepep_on(a(4))


def test_turbo_grouped_gemm_is_turned_off_only_with_megatron_fsdp():
    from types import SimpleNamespace

    from primus.backends.megatron.patches.args.muse_spark_recompute_cp_patches import (
        turbo_grouped_gemm_needs_disabling as f,
    )

    def a(mfsdp, gg=True, turbo=True):
        return SimpleNamespace(use_megatron_fsdp=mfsdp, use_turbo_grouped_gemm=gg, enable_primus_turbo=turbo)

    assert f(a(True))
    assert not f(a(False)) and not f(a(True, gg=False)) and not f(a(True, turbo=False))


def test_qk_norm_is_the_weightless_rms_norm():
    from primus.backends.megatron.core.transformer.muse_spark.attention import (
        MuseSparkQKNorm,
    )
    from primus.backends.megatron.core.transformer.muse_spark.norms import (
        MuseSparkRMSNorm,
    )

    q = MuseSparkQKNorm(config=SimpleNamespace(hidden_size=64), hidden_size=16, eps=1e-5)
    assert isinstance(q, MuseSparkRMSNorm) and q.weight is None
    assert not list(q.parameters())
    x = torch.randn(2, 3, 16, dtype=torch.bfloat16)
    assert torch.equal(q(x), MuseSparkRMSNorm(None, 16, eps=1e-5, with_scale=False)(x))


def test_rms_helper_equals_torch_rms_norm():
    # the eager fp32 helper (all Muse Spark norms and the embedding norm) is the same
    # function as F.rms_norm: a later kernel swap only needs the GPU parity check
    from primus.backends.megatron.core.transformer.muse_spark.norms import _rms

    torch.manual_seed(0)
    x = (torch.randn(64, 256) * 3).bfloat16().float()
    ref = torch.nn.functional.rms_norm(x, (256,), eps=1e-8)
    torch.testing.assert_close(_rms(x, 1e-8), ref, rtol=1e-6, atol=1e-6)


def test_final_logit_softcapping_must_be_null_zero_or_positive():
    from primus.backends.megatron.patches.args.muse_spark_layer_patterns_patches import (
        check_softcap,
    )

    for ok in (None, 0, 20.0, "20"):
        check_softcap(ok)
    for bad in (-1.0, float("nan"), float("inf"), "abc", True):
        with pytest.raises(ValueError, match="final_logit_softcapping"):
            check_softcap(bad)


def _deepep_off(args, local_gpus=None):
    from primus.backends.megatron.patches.args.muse_spark_recompute_cp_patches import (
        deepep_disable_reason,
    )

    return deepep_disable_reason(args, local_gpus) is not None


def _deepep_on(args, local_gpus=None):
    return not _deepep_off(args, local_gpus)


def test_stacked_expert_factory_names_the_unsupported_optimizer_format():
    # the stacked Turbo expert weights: one factory that builds the per-expert entries; a
    # flattened range (only 'fully_sharded_model_space', a legacy load format, makes one)
    # fails with a message naming the formats that work
    from megatron.core.dist_checkpointing import ShardedTensor

    from primus.backends.megatron.core.transformer.muse_spark.moe import (
        stack_expert_sharded_state_dict,
    )

    weights = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    sd = {
        f"mlp.weight{i}": ShardedTensor.from_rank_offsets(
            "mlp.weight", weights[i], (0, i, 2), replica_id=(0, 0, 0)
        )
        for i in range(2)
    }
    linear = SimpleNamespace(num_gemms=2, weights=weights)
    stack_expert_sharded_state_dict(sd, "mlp.", linear)
    factory = sd.pop("mlp.weights")
    assert not sd
    built = factory.build_fn("opt.mlp.weight", weights, (0, 0, 0), None)
    assert [built[f"expert{i}"].key for i in range(2)] == ["opt.mlp.weight"] * 2
    assert torch.equal(factory.merge_fn({f"expert{i}": weights[i] for i in range(2)}), weights)
    with pytest.raises(NotImplementedError, match="fully_sharded_model_space.*dp_reshardable"):
        factory.build_fn("opt.mlp.weight", weights, (0, 0, 0), slice(0, 4))
