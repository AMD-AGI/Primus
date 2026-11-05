###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the deferred host-sync train loop.

WHY THIS TEST EXISTS:
  The train-loop change is a source rewrite of an upstream method, so two things
  can go wrong without an error: AutoModel's loop drifts and the rewrite no longer
  matches it, or the rewrite changes what is logged. A miniature of the loop with
  every anchored line at its upstream indentation pins the second; the real
  loop, read from the pinned AutoModel's source, pins the first.
"""

import ast
import importlib.util
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from primus.backends.automodel.profiling.torch_profiler import (  # noqa: E402
    LOOP_WRAPPED_FLAG,
)
from primus.backends.automodel.training import deferred_host_sync  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[4]

# A miniature of TrainDiffusionRecipe.run_train_validation_loop. Every line the
# rewrite anchors on is reproduced at the upstream indentation; the rest is the
# minimum needed to observe what the loop logs and when it holds Python floats.
STUB_SOURCE = textwrap.dedent(
    """
    import logging

    import torch


    def clip_grad_norm(grads):
        return torch.stack(grads).norm()


    class Recipe:
        def __init__(self, steps, log_every, ckpt_every=0):
            self.steps = steps
            self.log_every = log_every
            self.ckpt_every = ckpt_every
            self.types = []
            self.logged = []
            self.checkpoints = []
            self.epoch_avgs = []

        def _elapsed_seconds_since(self, start):
            return 1.0, start + 1.0

        def save_checkpoint(self, epoch, step, train_loss):
            self.checkpoints.append(train_loss)

        def run_train_validation_loop(self):
            perf_window_start_time = 0.0
            for epoch in range(1):
                epoch_loss = 0.0
                num_steps = 0
                global_step = 0
                for micro_losses, grads in self.steps:
                    grad_norm = clip_grad_norm(grads)
                    grad_norm = float(grad_norm) if torch.is_tensor(grad_norm) else grad_norm

                    group_loss_mean = float(torch.stack(micro_losses).mean().item())
                    epoch_loss += group_loss_mean
                    num_steps += 1
                    global_step += 1

                    should_log = global_step % self.log_every == 0
                    if should_log:
                        elapsed_seconds, perf_window_end_time = self._elapsed_seconds_since(perf_window_start_time)
                        self.logged.append((group_loss_mean, epoch_loss / num_steps, grad_norm))
                    self.types.append((type(group_loss_mean), type(epoch_loss), type(grad_norm)))

                    if self.ckpt_every and global_step % self.ckpt_every == 0:
                        self.save_checkpoint(epoch, global_step, epoch_loss / num_steps)

                avg_loss = epoch_loss / num_steps
                logging.info(f"epoch avg_loss={avg_loss:.6f}")
                self.epoch_avgs.append(avg_loss)
    """
)


def _load_stub(tmp_path, monkeypatch, source):
    """Import ``source`` from a real file, which ``inspect.getsource`` needs."""
    path = tmp_path / "deferred_sync_stub.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("deferred_sync_stub", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "deferred_sync_stub", module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def stub_module(tmp_path, monkeypatch):
    return _load_stub(tmp_path, monkeypatch, STUB_SOURCE)


def _steps(n=6, micro=3):
    gen = torch.Generator().manual_seed(0)
    return [
        (
            [torch.rand((), generator=gen) for _ in range(micro)],
            [torch.rand((), generator=gen) for _ in range(4)],
        )
        for _ in range(n)
    ]


def _run(recipe_cls, steps, log_every=2, ckpt_every=3):
    recipe = recipe_cls(steps, log_every=log_every, ckpt_every=ckpt_every)
    recipe.run_train_validation_loop()
    return recipe


def test_logged_values_are_unchanged_to_the_bit(stub_module, set_option):
    """The deferred sum is float64, like the Python float it replaces."""
    set_option("primus_train.defer_host_sync", True)
    steps = _steps(n=24)
    reference = _run(stub_module.Recipe, steps, log_every=4, ckpt_every=6)

    deferred_host_sync.patch_recipe(stub_module.Recipe)
    deferred = _run(stub_module.Recipe, steps, log_every=4, ckpt_every=6)

    assert len(deferred.logged) == len(reference.logged) == 6
    for got, want in zip(deferred.logged, reference.logged):
        assert all(isinstance(value, float) for value in got)
        assert got == want
    assert all(isinstance(value, float) for value in deferred.checkpoints)
    assert deferred.checkpoints == reference.checkpoints
    assert isinstance(deferred.epoch_avgs[0], float)
    assert deferred.epoch_avgs == reference.epoch_avgs


def test_only_logging_steps_hold_python_floats(stub_module, set_option):
    set_option("primus_train.defer_host_sync", True)
    deferred_host_sync.patch_recipe(stub_module.Recipe)
    recipe = _run(stub_module.Recipe, _steps(), log_every=2, ckpt_every=0)

    for step, types_ in enumerate(recipe.types, start=1):
        expected = float if step % 2 == 0 else torch.Tensor
        assert types_ == (expected, expected, expected), f"step {step}"


def test_a_later_run_with_the_setting_off_gets_the_upstream_loop(stub_module, set_option):
    set_option("primus_train.defer_host_sync", True)
    deferred_host_sync.patch_recipe(stub_module.Recipe)
    deferred = _run(stub_module.Recipe, _steps(), log_every=2, ckpt_every=0)

    set_option("primus_train.defer_host_sync", False)
    upstream = _run(stub_module.Recipe, _steps(), log_every=2, ckpt_every=0)

    assert deferred.types[0] == (torch.Tensor, torch.Tensor, torch.Tensor)
    assert all(types_ == (float, float, float) for types_ in upstream.types)


def test_the_rewrite_is_idempotent(stub_module):
    deferred_host_sync.patch_recipe(stub_module.Recipe)
    patched = stub_module.Recipe.run_train_validation_loop

    deferred_host_sync.patch_recipe(stub_module.Recipe)
    assert stub_module.Recipe.run_train_validation_loop is patched


def test_a_subclass_that_overrides_the_loop_gets_its_own_rewrite(tmp_path, monkeypatch, set_option):
    loop = STUB_SOURCE[STUB_SOURCE.index("    def run_train_validation_loop") :]
    module = _load_stub(tmp_path, monkeypatch, STUB_SOURCE + "\n\nclass SubRecipe(Recipe):\n" + loop)
    set_option("primus_train.defer_host_sync", True)

    deferred_host_sync.patch_recipe(module.Recipe)
    deferred_host_sync.patch_recipe(module.SubRecipe)

    recipe = _run(module.SubRecipe, _steps(), log_every=2, ckpt_every=0)
    assert recipe.types[0] == (torch.Tensor, torch.Tensor, torch.Tensor)


def test_a_profiler_wrapped_loop_is_refused(stub_module):
    def wrapped(self):
        pass

    setattr(wrapped, LOOP_WRAPPED_FLAG, True)
    stub_module.Recipe.run_train_validation_loop = wrapped

    with pytest.raises(RuntimeError, match="profiler"):
        deferred_host_sync.patch_recipe(stub_module.Recipe)


def test_a_rewrite_that_fails_names_the_setting_and_leaves_the_loop(tmp_path, monkeypatch):
    drifted = STUB_SOURCE.replace(
        "epoch_loss += group_loss_mean", "epoch_loss = epoch_loss + group_loss_mean"
    )
    module = _load_stub(tmp_path, monkeypatch, drifted)
    upstream = module.Recipe.__dict__["run_train_validation_loop"]
    monkeypatch.setitem(
        sys.modules,
        "nemo_automodel.recipes.diffusion.train",
        SimpleNamespace(TrainDiffusionRecipe=module.Recipe),
    )

    with pytest.raises(
        RuntimeError, match="primus_train.defer_host_sync is on, but the train loop could not"
    ):
        deferred_host_sync.install()
    assert module.Recipe.__dict__["run_train_validation_loop"] is upstream


def _pinned_train_loop() -> Path:
    """AutoModel's diffusion train loop: the submodule checkout, else the installed package."""
    packages = [REPO_ROOT / "third_party" / "Automodel" / "nemo_automodel"]
    spec = importlib.util.find_spec("nemo_automodel")
    if spec is not None and spec.submodule_search_locations:
        packages.append(Path(list(spec.submodule_search_locations)[0]))
    candidates = [package / "recipes" / "diffusion" / "train.py" for package in packages]
    train_py = next((path for path in candidates if path.is_file()), None)
    if train_py is None:
        pytest.skip("neither the AutoModel submodule nor nemo_automodel is available")
    return train_py


def test_the_anchors_match_the_pinned_train_loop(tmp_path, monkeypatch):
    """Fails when AutoModel's train loop drifts away from the rewrite's anchors.

    The method is read from the file rather than imported, so this test never
    rewrites the real recipe class for the rest of the session, and it runs
    where AutoModel is checked out but not installed, as in CI.
    """
    lines = _pinned_train_loop().read_text().splitlines(keepends=True)

    tree = ast.parse("".join(lines))
    recipe = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TrainDiffusionRecipe")
    loop = next(n for n in recipe.body if getattr(n, "name", None) == "run_train_validation_loop")
    method_source = "".join(lines[loop.lineno - 1 : loop.end_lineno])

    path = tmp_path / "upstream_loop_stub.py"
    path.write_text("class TrainDiffusionRecipe:\n" + method_source)
    module_spec = importlib.util.spec_from_file_location("upstream_loop_stub", path)
    module = importlib.util.module_from_spec(module_spec)
    monkeypatch.setitem(sys.modules, "upstream_loop_stub", module)
    module_spec.loader.exec_module(module)

    deferred_host_sync.patch_recipe(module.TrainDiffusionRecipe)


def _ctx(transformer_cls=None):
    from primus.core.patches import PatchContext

    params = SimpleNamespace(
        model=SimpleNamespace(pipeline_spec=SimpleNamespace(transformer_cls=transformer_cls))
    )
    return PatchContext(
        backend="automodel",
        phase="before_train",
        extra={"module_config": SimpleNamespace(params=params), "backend_args": params},
    )


def test_the_patch_is_registered_gated_and_runs_before_the_profiler(set_option):
    from primus.core.patches.patch_registry import PatchRegistry

    profiler = PatchRegistry.get("automodel.profiling.torch_profiler")
    loop = PatchRegistry.get("automodel.training.deferred_host_sync")
    assert loop is not None and profiler is not None
    assert loop.priority < profiler.priority

    contexts = [_ctx(), _ctx("FluxTransformer2DModel"), _ctx("WanTransformer3DModel")]
    assert not any(loop.condition(ctx) for ctx in contexts)
    set_option("primus_train.defer_host_sync", False)
    assert not any(loop.condition(ctx) for ctx in contexts)
    set_option("primus_train.defer_host_sync", True)
    assert all(loop.condition(ctx) for ctx in contexts)
