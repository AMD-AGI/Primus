###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the diffusion recipe's step metrics computed locally.

WHY THIS TEST EXISTS:
  The patch swaps three methods of AutoModel's recipe for versions that log the
  same numbers without a device sync or a collective. The upstream methods are
  read from the pinned AutoModel and run next to the replacements on a stand-in
  recipe, so a replacement that logs something else, or an upstream method that
  drifts away from it, fails here. On a GPU the replacements also run with
  synchronizing CUDA calls turned into errors.
"""

import importlib.util
import inspect
import logging
import sys
import time
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist  # noqa: E402

from primus.backends.automodel.training import local_log_metrics  # noqa: E402

UPSTREAM = ("_elapsed_seconds_since", "_count_global_samples", "_get_memory_metrics")
DP_GROUP = object()
DP_SIZE = 4
GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _loader(**settings):
    return {"data": {"dataloader": settings}}


class _RecipeBase:
    """The rest of TrainDiffusionRecipe, as far as the three upstream methods use it."""

    def __init__(self, cfg=None):
        self.cfg = _loader(drop_last=True) if cfg is None else cfg
        # As the recipe's constructor picks it.
        self.device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
        self.syncs = 0

    def setup(self):
        pass

    def _sync_device(self):
        self.syncs += 1

    def _get_collective_device(self):
        return self.device

    def _get_dp_group(self):
        return DP_GROUP


@pytest.fixture
def recipe_cls(tmp_path, monkeypatch, pinned_automodel):
    """A recipe class with the pinned upstream methods, imported as AutoModel's module."""
    path = tmp_path / "train.py"
    path.write_text(
        "from __future__ import annotations\n\nimport time\n\nimport torch\nimport torch.distributed as dist\n\n\n"
        "class TrainDiffusionRecipe(_RecipeBase):\n"
        + pinned_automodel("recipes/diffusion/train.py", "TrainDiffusionRecipe", *UPSTREAM)
    )
    name = "nemo_automodel.recipes.diffusion.train"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    module._RecipeBase = _RecipeBase
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module.TrainDiffusionRecipe


def _upstream(recipe_cls):
    return {name: recipe_cls.__dict__[name] for name in UPSTREAM}


@pytest.fixture
def four_ranks(monkeypatch):
    """torch.distributed as one of four ranks with the same counters; lists each reduction."""
    reductions = []

    def all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
        reductions.append((op, group))
        if op == dist.ReduceOp.SUM:
            tensor.mul_(DP_SIZE)

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: DP_SIZE)
    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    return reductions


def test_one_rank_logs_the_upstream_numbers(recipe_cls):
    upstream = _upstream(recipe_cls)
    assert local_log_metrics.install()
    recipe = recipe_cls()
    recipe.setup()
    held = torch.empty(1 << 20, device=recipe.device)

    # The replacement first: the upstream method's own tensor would raise the peak.
    memory = recipe._get_memory_metrics()
    assert memory == upstream["_get_memory_metrics"](recipe)
    if recipe.device.type == "cuda":
        assert memory["memory_allocated_gb"] >= held.numel() * held.element_size() / 1024**3
    assert recipe._count_global_samples(7) == upstream["_count_global_samples"](recipe, 7) == 7

    start = time.perf_counter()
    elapsed, end = recipe._elapsed_seconds_since(start)
    assert elapsed == max(end - start, 1e-12)
    assert recipe.syncs == 0


def test_four_ranks_log_the_upstream_numbers_without_a_sync_or_a_reduction(recipe_cls, four_ranks):
    upstream = _upstream(recipe_cls)
    assert local_log_metrics.install()
    recipe = recipe_cls()
    recipe.setup()

    got = (recipe._count_global_samples(16), recipe._get_memory_metrics())
    recipe._elapsed_seconds_since(time.perf_counter())
    assert (recipe.syncs, four_ranks) == (0, [])

    want = (upstream["_count_global_samples"](recipe, 16), upstream["_get_memory_metrics"](recipe))
    upstream["_elapsed_seconds_since"](recipe, time.perf_counter())
    assert got == want
    assert got[0] == 16 * DP_SIZE
    assert recipe.syncs == 1
    # Without a GPU the upstream memory metrics are zeros and reduce nothing.
    memory = [(dist.ReduceOp.MAX, None)] if torch.cuda.is_available() else []
    assert four_ranks == [(dist.ReduceOp.SUM, DP_GROUP), *memory, (dist.ReduceOp.MAX, None)]


@GPU
def test_on_a_gpu_the_replacements_never_synchronize(recipe_cls, four_ranks):
    upstream = _upstream(recipe_cls)
    assert local_log_metrics.install()
    recipe = recipe_cls()
    recipe.setup()
    calls = {"_elapsed_seconds_since": (0.0,), "_count_global_samples": (16,), "_get_memory_metrics": ()}

    torch.cuda.set_sync_debug_mode("error")
    try:
        for name, args in calls.items():
            getattr(recipe, name)(*args)
        for name, args in calls.items():
            with pytest.raises(RuntimeError, match="synchroniz"):
                upstream[name](recipe, *args)
    finally:
        torch.cuda.set_sync_debug_mode("default")


@pytest.mark.parametrize(
    "cfg,uniform",
    [
        (_loader(drop_last=True), True),
        (_loader(drop_last=True, dynamic_batch_size=False), True),
        ({"data": {"dataloader": SimpleNamespace(drop_last=True)}}, True),
        (_loader(drop_last=True, dynamic_batch_size=True), False),
        (_loader(drop_last=False), False),
        (_loader(), False),
        (SimpleNamespace(), False),
    ],
    ids=["drop_last", "fixed_batch", "attribute_config", "dynamic_batch", "keep_last", "unset", "unreadable"],
)
def test_the_sample_count_is_reduced_unless_every_rank_has_the_same_batch(
    recipe_cls, four_ranks, caplog, cfg, uniform
):
    assert local_log_metrics.install()
    recipe = recipe_cls(cfg)
    with caplog.at_level(logging.WARNING, logger=local_log_metrics.__name__):
        recipe.setup()

    assert recipe._count_global_samples(16) == 16 * DP_SIZE
    assert len(four_ranks) == (0 if uniform else 1)
    assert ("keeping the sample-count all-reduce" in caplog.text) == (not uniform)


def test_install_is_idempotent_and_keeps_the_upstream_signatures(recipe_cls):
    upstream = _upstream(recipe_cls)
    assert local_log_metrics.install()
    installed = {name: recipe_cls.__dict__[name] for name in (*UPSTREAM, "setup")}
    assert local_log_metrics.install()

    assert {name: recipe_cls.__dict__[name] for name in installed} == installed
    for name in UPSTREAM:
        assert installed[name] is not upstream[name]
        parameters = list(inspect.signature(installed[name]).parameters)
        assert parameters == list(inspect.signature(upstream[name]).parameters), name


def test_the_patch_is_registered_and_gated(set_option):
    from primus.core.patches import PatchContext
    from primus.core.patches.patch_registry import PatchRegistry

    patch = PatchRegistry.get("automodel.training.local_log_metrics")
    assert patch is not None and (patch.backend, patch.phase) == ("automodel", "before_train")
    ctx = PatchContext(backend="automodel", phase="before_train")
    assert not patch.condition(ctx)
    set_option("primus_train.local_log_metrics", False)
    assert not patch.condition(ctx)
    set_option("primus_train.local_log_metrics", True)
    assert patch.condition(ctx)
