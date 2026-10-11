###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Forward healing (``mxfp6_fwd_fp4_heal_*``): schedule, plan and switch mechanics, config validation, and the Dynamo
guard the switch relies on. No GPU needed; the numerics of a healed layer are in test_mxfp6_fused_paths_gpu.py.

The invariant everything rests on: the healed set is a pure function of the iteration, so every rank heals the same
layers at the same step, the MLPerf warmup's re-entry into train_step at a fixed iteration is harmless, and a resumed
run lands in the right state.
"""

import types

import pytest
import torch

from primus.backends.megatron.patches import mxfp6_fwd_fp4_heal_patches as H

# -- the schedule -----------------------------------------------------------------------------------------------------


class TestTargetLayerCount:
    def test_nothing_before_the_heal_iteration(self):
        assert [H.target_layer_count(i, 10, 0, 19) for i in (0, 9)] == [0, 0]

    def test_all_at_once_from_the_heal_iteration(self):
        assert [H.target_layer_count(i, 10, 0, 19) for i in (10, 11, 500)] == [19, 19, 19]

    def test_missing_iteration_or_off(self):
        assert H.target_layer_count(None, 10, 0, 19) == 0
        assert H.target_layer_count(100, 0, 0, 19) == 0

    def test_ramp_grows_by_rate_and_clamps(self):
        assert [H.target_layer_count(i, 10, 2, 19) for i in (9, 10, 11, 18, 19, 40)] == [0, 2, 4, 18, 19, 19]

    def test_rate_at_or_above_the_plan_is_all_at_once(self):
        assert H.target_layer_count(10, 10, 19, 19) == H.target_layer_count(10, 10, 50, 19) == 19

    def test_pure_and_order_independent(self):
        its = list(range(0, 40))
        a = [H.target_layer_count(i, 10, 3, 19) for i in its]
        b = {i: H.target_layer_count(i, 10, 3, 19) for i in reversed(its)}
        assert a == [b[i] for i in its]


# -- the plan ---------------------------------------------------------------------------------------------------------


def _flagged(**attrs):
    m = torch.nn.Module()
    m._mxfp6_fwd_fp4_healed = False
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


def _toy_model(n_joint=3, n_single=2):
    """transformer.layers: joint blocks (img + txt MLP, text out-projection), then single blocks (an out-projection
    that carries the flag but is not a joint target)."""
    layers = torch.nn.ModuleList()
    for _ in range(n_joint):
        blk = torch.nn.Module()
        blk.mlp = _flagged(_mxfp6_joint=True, _mxfp6_joint_img=True)
        blk.context_mlp = _flagged(_mxfp6_joint=True)
        blk.self_attention = torch.nn.Module()
        blk.self_attention.linear_proj = _flagged()
        blk.self_attention.added_linear_proj = _flagged()
        layers.append(blk)
    for _ in range(n_single):
        blk = torch.nn.Module()
        blk.mlp = _flagged()  # single blocks' MLPs are not tagged joint
        blk.self_attention = torch.nn.Module()
        blk.self_attention.linear_proj = _flagged()
        layers.append(blk)
    model = torch.nn.Module()
    model.transformer = torch.nn.Module()
    model.transformer.layers = layers
    return model


class TestPlan:
    def test_joint_targets_only_deep_first(self):
        model = _toy_model()
        plan = H.build_heal_plan([model], "deep_to_shallow")
        assert [i for i, _ in plan] == [2, 1, 0]
        blk = model.transformer.layers[0]
        assert plan[-1][1] == [blk.mlp, blk.context_mlp, blk.self_attention.added_linear_proj]

    def test_shallow_first(self):
        assert [i for i, _ in H.build_heal_plan([_toy_model()], "shallow_to_deep")] == [0, 1, 2]

    def test_empty_for_a_model_without_targets(self):
        assert H.build_heal_plan([_toy_model(n_joint=0)]) == []

    def test_set_heal_is_absolute(self):
        model = _toy_model()
        plan = H.build_heal_plan([model])
        H.set_heal(plan, 2)
        H.set_heal(plan, 2)
        healed = [all(m._mxfp6_fwd_fp4_healed for m in mods) for _, mods in plan]
        assert healed == [True, True, False]
        H.set_heal(plan, 0)
        assert not any(m._mxfp6_fwd_fp4_healed for _, mods in plan for m in mods)
        # the image projection and the single blocks are never touched
        assert not model.transformer.layers[0].self_attention.linear_proj._mxfp6_fwd_fp4_healed


# -- the train_step wrapper -------------------------------------------------------------------------------------------


class TestWrapper:
    @pytest.fixture
    def wrapped(self, monkeypatch):
        import megatron.training.training as mt

        from primus.backends.megatron.patches import _patch_guard

        calls = []
        model = _toy_model()

        def fake_train_step(*a, **kw):
            plan = H.build_heal_plan([model])
            calls.append(sum(all(m._mxfp6_fwd_fp4_healed for m in mods) for _, mods in plan))

        monkeypatch.setattr(mt, "train_step", fake_train_step)
        monkeypatch.setattr(_patch_guard, "is_patched", lambda *a: False)
        monkeypatch.setattr(_patch_guard, "mark_patched", lambda *a: None)
        monkeypatch.setattr(H, "_report_heal", lambda *a: None)

        def make(heal_iter, rate=0):
            args = types.SimpleNamespace(mxfp6_fwd_fp4_heal_iter=heal_iter, mxfp6_fwd_fp4_heal_layers_per_iter=rate,
                                         mxfp6_fwd_fp4_heal_order="deep_to_shallow", train_iters=1000)
            monkeypatch.setattr(H, "get_args", lambda ctx: args)
            H.patch_mxfp6_fwd_fp4_heal(None)
            step = mt.train_step
            return lambda it: step(None, None, [model], iteration=it)

        return make, calls, model

    def test_switches_at_the_boundary(self, wrapped):
        make, calls, _ = wrapped
        step = make(5)
        for it in (3, 4, 5, 6):
            step(it)
        assert calls == [0, 0, 3, 3]

    def test_ramp(self, wrapped):
        make, calls, _ = wrapped
        step = make(5, rate=1)
        for it in (4, 5, 6, 7, 8):
            step(it)
        assert calls == [0, 1, 2, 3, 3]

    def test_warmup_reentry_keeps_a_forced_prewarm(self, wrapped):
        make, calls, model = wrapped
        step = make(5)
        step(0)  # warmup step at iteration 0
        plan = H.build_heal_plan([model])
        H.set_heal(plan, len(plan))  # what prewarm_healed_graphs does around its step
        step(0)
        H.set_heal(plan, 0)
        assert calls == [0, 3]

    def test_resume_past_the_heal_iteration_starts_healed(self, wrapped):
        make, calls, _ = wrapped
        step = make(5)
        step(40)
        assert calls == [3]

    def test_empty_plan_raises(self, wrapped, monkeypatch):
        make, _, _ = wrapped
        step = make(5)
        monkeypatch.setattr(H, "build_heal_plan", lambda *a: [])
        with pytest.raises(RuntimeError, match="no-op"):
            step(1)


# -- config validation ------------------------------------------------------------------------------------------------


class TestConfigValidation:
    @staticmethod
    def _config(**overrides):
        from primus.backends.megatron.core.models.diffusion.common.config import (
            BaseDiffusionConfig,
        )

        d = dict(hidden_size=256, num_attention_heads=8, num_layers=1, fp6="mxfp6", mxfp6_gemm_layout="tilescale",
                 mxfp6_bwd_fp4_dgrad=True, mxfp6_bwd_fp4_wgrad=True, mxfp6_joint_proj=True,
                 mxfp6_fwd_fp4_joint_mlp=True, mxfp6_fwd_fp4_joint_mlp_parts="txt_fc1,txt_fc2",
                 mxfp6_fwd_fp4_joint_txt_proj=True)
        d.update(overrides)
        return BaseDiffusionConfig(**d)

    def test_off_by_default_and_accepted_on_the_fp4_recipe(self):
        assert self._config().mxfp6_fwd_fp4_heal_iter == 0
        assert self._config(mxfp6_fwd_fp4_heal_iter=100, mxfp6_fwd_fp4_heal_layers_per_iter=2)

    @pytest.mark.parametrize(
        "overrides,match",
        [
            (dict(mxfp6_fwd_fp4_heal_iter=-1), ">= 0"),
            (dict(mxfp6_fwd_fp4_heal_layers_per_iter=-1), ">= 0"),
            (dict(mxfp6_fwd_fp4_heal_order="random"), "deep_to_shallow"),
            (dict(mxfp6_fwd_fp4_heal_iter=100, mxfp6_fwd_fp4_joint_mlp=False, mxfp6_fwd_fp4_joint_txt_proj=False),
             "change nothing"),
            (dict(mxfp6_fwd_fp4_heal_iter=100, mxfp6_fwd_fp4_single_fc1=True), "joint blocks only"),
            (dict(mxfp6_fwd_fp4_heal_iter=100, mxfp6_packed_param_gather=True), "packed_param_gather"),
        ],
    )
    def test_rejections(self, overrides, match):
        with pytest.raises(ValueError, match=match):
            self._config(**overrides)


# -- target fp8: every forward linear of the healed layers on tensorwise FP8 with delayed scales --------------------


def _L():
    return pytest.importorskip("primus.backends.megatron.core.extensions.primus_turbo_mxfp6_local")


class TestFp8Plan:
    def test_plan_is_every_layer(self):
        model = _toy_model()
        plan = H.build_heal_plan([model], "deep_to_shallow", "fp8")
        assert [i for i, _ in plan] == [4, 3, 2, 1, 0]
        assert [mods for _, mods in plan] == [[model.transformer.layers[i]] for i in (4, 3, 2, 1, 0)]

    def test_set_heal_seeds_only_the_newly_healed(self):
        plan = H.build_heal_plan([_toy_model()], "shallow_to_deep", "fp8")
        H.set_heal(plan, 3, "fp8", seed_from=1)
        assert [m._mxfp6_fwd_fp8_healed for _, (m,) in plan] == [True, True, True, False, False]
        assert [m._mxfp6_fwd_fp8_seed for _, (m,) in plan] == [False, True, True, False, False]
        H.set_heal(plan, 3, "fp8")  # default: no seed step
        assert not any(m._mxfp6_fwd_fp8_seed for _, (m,) in plan)
        # the MXFP6 target's flags are not touched
        assert not any(m._mxfp6_fwd_fp4_healed for m in plan[0][1][0].modules() if hasattr(m, "_mxfp6_fwd_fp4_healed"))

    def test_hooks_set_and_restore_the_gates(self):
        from primus.backends.megatron.core.models.diffusion.common.mxfp6_gates import Mxfp6Gates, gates, reset

        seen = []
        inner, outer = torch.nn.Module(), torch.nn.Module()
        inner.forward = lambda: seen.append((gates().fwd_fp8, gates().fwd_fp8_seed))
        outer.inner = inner
        outer.forward = lambda: (seen.append((gates().fwd_fp8, gates().fwd_fp8_seed)), inner())
        for m, healed, seed in ((outer, False, False), (inner, True, True)):
            m._mxfp6_fwd_fp8_healed, m._mxfp6_fwd_fp8_seed = healed, seed
            m.register_forward_pre_hook(H._f8_pre)
            m.register_forward_hook(H._f8_post)
        reset(Mxfp6Gates())
        outer()
        assert seen == [(False, False), (True, True)]
        assert (gates().fwd_fp8, gates().fwd_fp8_seed) == (False, False) and not H._F8_SAVED


class TestF8Step:
    def test_rolls_the_multiplier_from_the_history(self):
        L = _L()
        a, b = L.F8State("cpu", history=4), L.F8State("cpu", history=4)
        for slot, (va, vw) in enumerate([(2.0, 0.5), (8.0, 0.25), (1.0, 0.125)]):
            a.amax["a"].fill_(va)
            a.amax["w"].fill_(vw)
            b.amax["a"].fill_(va / 2)
            L.f8_step([a, b], slot)
            assert a.amax["a"].item() == a.amax["w"].item() == b.amax["a"].item() == 0
        assert a.q["a"].item() == 448.0 / 8.0 and a.q["w"].item() == 448.0 / 0.5 and b.q["a"].item() == 448.0 / 4.0
        assert 1e30 < b.q["w"].item() < float("inf")  # never seen: finite (a seed step precedes any real use)
        for slot in range(3, 7):  # the ring forgets the old maxima
            a.amax["a"].fill_(1.0)
            L.f8_step([a], slot)
        assert a.q["a"].item() == 448.0

    def test_reset(self):
        L = _L()
        st = L.F8State("cpu")
        st.amax["a"].fill_(3.0)
        L.f8_step([st], 0)
        st.reset()
        assert st.q["a"].item() == 1.0 and st.hist["a"].abs().sum() == 0 and st.amax["a"].item() == 0


class TestFp8Wrapper:
    @pytest.fixture
    def wrapped(self, monkeypatch):
        import megatron.training.training as mt

        from primus.backends.megatron.patches import _patch_guard

        L = _L()
        calls, rolled = [], []
        model = _toy_model()

        def fake_train_step(*a, **kw):
            layers = list(model.transformer.layers)
            calls.append((sum(getattr(m, "_mxfp6_fwd_fp8_healed", False) for m in layers),
                          sum(getattr(m, "_mxfp6_fwd_fp8_seed", False) for m in layers)))

        monkeypatch.setattr(mt, "train_step", fake_train_step)
        monkeypatch.setattr(_patch_guard, "is_patched", lambda *a: False)
        monkeypatch.setattr(_patch_guard, "mark_patched", lambda *a: None)
        monkeypatch.setattr(H, "_report_heal", lambda *a: None)
        monkeypatch.setattr(H, "f8_setup", lambda plan: {idx: [idx] for idx, _ in plan})
        monkeypatch.setattr(L, "f8_step", lambda states, slot: rolled.append((sorted(states), slot)))

        def make(heal_iter, rate=0):
            args = types.SimpleNamespace(mxfp6_fwd_fp4_heal_iter=heal_iter, mxfp6_fwd_fp4_heal_layers_per_iter=rate,
                                         mxfp6_fwd_fp4_heal_order="deep_to_shallow", train_iters=1000,
                                         mxfp6_fwd_fp4_heal_target="fp8")
            monkeypatch.setattr(H, "get_args", lambda ctx: args)
            H.patch_mxfp6_fwd_fp4_heal(None)
            step = mt.train_step
            return lambda it: step(None, None, [model], iteration=it)

        return make, calls, rolled, model

    def test_seed_step_then_delayed(self, wrapped):
        make, calls, rolled, _ = wrapped
        step = make(5)
        for it in (4, 5, 6, 7):
            step(it)
        assert calls == [(0, 0), (5, 5), (5, 0), (5, 0)]
        assert rolled == [([0, 1, 2, 3, 4], 6), ([0, 1, 2, 3, 4], 7)]

    def test_ramp_seeds_each_new_layer_once(self, wrapped):
        make, calls, rolled, _ = wrapped
        step = make(5, rate=2)
        for it in (5, 6, 7, 8):
            step(it)
        assert calls == [(2, 2), (4, 2), (5, 1), (5, 0)]
        assert rolled == [([3, 4], 6), ([1, 2, 3, 4], 7), ([0, 1, 2, 3, 4], 8)]  # deep first: layers 4, 3 lead

    def test_warmup_reentry_keeps_a_forced_prewarm(self, wrapped):
        make, calls, rolled, model = wrapped
        step = make(5)
        step(0)
        plan = H.build_heal_plan([model], target="fp8")
        H.set_heal(plan, len(plan), "fp8", 0)  # prewarm_healed_graphs' seed variant
        step(0)
        H.set_heal(plan, 0, "fp8")
        assert calls == [(0, 0), (5, 5)] and rolled == []

    def test_resume_past_the_heal_iteration_seeds_everything(self, wrapped):
        make, calls, _, _ = wrapped
        step = make(5)
        step(40)
        step(41)
        assert calls == [(5, 5), (5, 0)]


class TestFp8ConfigValidation:
    @staticmethod
    def _config(**overrides):
        d = dict(mxfp6_fwd_fp4_heal_iter=100, mxfp6_fwd_fp4_heal_target="fp8", mxfp6_fwd_fp4_joint_mlp=False,
                 mxfp6_fwd_fp4_joint_txt_proj=False)
        d.update(overrides)
        return TestConfigValidation._config(**d)

    def test_accepted_with_and_without_an_mxfp4_forward(self):
        assert self._config().mxfp6_fwd_fp4_heal_target == "fp8"
        assert self._config(mxfp6_fwd_fp4_joint_mlp=True, mxfp6_fwd_fp4_joint_mlp_parts="txt_fc1,txt_fc2",
                            mxfp6_fwd_fp4_joint_txt_proj=True)

    @pytest.mark.parametrize(
        "overrides,match",
        [
            (dict(mxfp6_fwd_fp4_heal_target="fp16"), "fp8"),
            (dict(mxfp6_gemm_layout="blob"), "tilescale"),
            (dict(mxfp6_bwd_fp4_wgrad=False), "bwd_fp4"),
            (dict(mxfp6_fwd_a6w4=True), "fwd_a6w4"),
            (dict(mxfp6_weight_format="mxfp4"), "weight_format"),
            (dict(mxfp6_packed_param_gather=True), "packed_param_gather"),
        ],
    )
    def test_rejections(self, overrides, match):
        with pytest.raises(ValueError, match=match):
            self._config(**overrides)


# -- the Dynamo guard the switch relies on ----------------------------------------------------------------------------


class _Block(torch.nn.Module):
    """Several instances, one forward code object; the flag picks the arithmetic, as the format choice does."""

    def __init__(self):
        super().__init__()
        self._mxfp6_fwd_fp4_healed = False

    def forward(self, x):
        return x * 2 if self._mxfp6_fwd_fp4_healed else x + 1


def test_flag_is_guarded_and_shared_across_instances():
    from torch._dynamo.utils import counters

    blocks = [_Block() for _ in range(4)]
    compiled = [torch.compile(b, backend="eager") for b in blocks]
    torch._dynamo.reset()
    counters.clear()
    x = torch.ones(4)
    assert all(torch.equal(c(x), x + 1) for c in compiled)
    unhealed = counters["stats"]["unique_graphs"]
    for b in blocks:
        b._mxfp6_fwd_fp4_healed = True
    assert all(torch.equal(c(x), x * 2) for c in compiled)
    healed = counters["stats"]["unique_graphs"]
    for b in blocks:
        b._mxfp6_fwd_fp4_healed = False
    assert all(torch.equal(c(x), x + 1) for c in compiled)
    assert unhealed < len(blocks) and 0 < healed - unhealed < len(blocks)  # shared, not per instance
    assert counters["stats"]["unique_graphs"] == healed  # un-healing is a cache hit
