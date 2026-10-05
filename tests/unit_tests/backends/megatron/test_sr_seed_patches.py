###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""sr_seed_patches: every train_step sets the Primus-Turbo SR base seed to sr_step_seed(run seed, global rank,
iteration) before running the step, and the patch applies only to MXFP-quantized jobs."""

import sys
import types
from types import SimpleNamespace

import pytest

from primus.backends.megatron.patches import sr_seed_patches as P


def _ctx(**params):
    return SimpleNamespace(extra={"module_config": SimpleNamespace(params=SimpleNamespace(**params))})


@pytest.fixture
def fake_megatron(monkeypatch):
    """A fake megatron.training.{training, global_vars} and recording SR-seed ops."""
    calls, steps = [], []
    args = SimpleNamespace(seed=1234, curr_iteration=0)
    training = types.ModuleType("megatron.training.training")

    def train_step(*a, **k):
        steps.append(args.curr_iteration)
        return "stepped"

    training.train_step = train_step
    global_vars = types.ModuleType("megatron.training.global_vars")
    global_vars.get_args = lambda: args
    pkg = types.ModuleType("megatron.training")
    pkg.training, pkg.global_vars = training, global_vars
    root = types.ModuleType("megatron")
    root.training = pkg
    for name, mod in (
        ("megatron", root),
        ("megatron.training", pkg),
        ("megatron.training.training", training),
        ("megatron.training.global_vars", global_vars),
    ):
        monkeypatch.setitem(sys.modules, name, mod)

    def sr_step_seed(seed, rank, iteration):
        return (seed, rank, iteration)

    monkeypatch.setattr(P, "_sr_seed_ops", lambda: (lambda s: calls.append((s, list(steps))), sr_step_seed))
    return training, args, calls, steps


def test_every_step_seeds_from_seed_rank_iteration_before_running(fake_megatron):
    training, args, calls, steps = fake_megatron
    P.patch_train_step_with_sr_seed(_ctx())
    for it in (0, 1, 7):
        args.curr_iteration = it
        assert training.train_step() == "stepped"
    # one set per step, with (run seed, rank 0 without torch.distributed, iteration), each before its step ran
    assert [c[0] for c in calls] == [(1234, 0, 0), (1234, 0, 1), (1234, 0, 7)]
    assert [c[1] for c in calls] == [[], [0], [0, 1]]


def test_wrapping_twice_is_a_no_op(fake_megatron):
    training, _, calls, _ = fake_megatron
    P.patch_train_step_with_sr_seed(_ctx())
    once = training.train_step
    P.patch_train_step_with_sr_seed(_ctx())
    assert training.train_step is once
    training.train_step()
    assert len(calls) == 1


def test_applies_only_to_mxfp_jobs(monkeypatch):
    monkeypatch.setattr(P, "_sr_seed_ops", lambda: (None, None))
    assert P._is_sr_seed_patch_needed(_ctx(fp6="mxfp6", fp4=None))
    assert P._is_sr_seed_patch_needed(_ctx(fp6=None, fp4="mxfp4"))
    assert not P._is_sr_seed_patch_needed(_ctx(fp6=None, fp4=None))
    monkeypatch.setattr(P, "_sr_seed_ops", lambda: None)  # a Primus-Turbo without set_sr_seed
    assert not P._is_sr_seed_patch_needed(_ctx(fp6="mxfp6", fp4=None))
