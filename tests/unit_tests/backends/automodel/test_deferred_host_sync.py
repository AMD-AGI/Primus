###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the deferred host-sync train loop and the on-device FLUX latent ids.

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
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from primus.backends.automodel.models.flux import latent_ids  # noqa: E402
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


@pytest.fixture
def stub_module(tmp_path, monkeypatch):
    path = tmp_path / "deferred_sync_stub.py"
    path.write_text(STUB_SOURCE)
    spec = importlib.util.spec_from_file_location("deferred_sync_stub", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "deferred_sync_stub", module)
    spec.loader.exec_module(module)
    return module


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


def test_logged_values_are_unchanged_to_the_bit(stub_module):
    """The deferred sum is float64, like the Python float it replaces."""
    steps = _steps(n=24)
    reference = _run(stub_module.Recipe, steps, log_every=4, ckpt_every=6)

    assert deferred_host_sync.patch_recipe(stub_module.Recipe) is True
    deferred = _run(stub_module.Recipe, steps, log_every=4, ckpt_every=6)

    assert len(deferred.logged) == len(reference.logged) == 6
    for got, want in zip(deferred.logged, reference.logged):
        assert all(isinstance(value, float) for value in got)
        assert got == want
    assert all(isinstance(value, float) for value in deferred.checkpoints)
    assert deferred.checkpoints == reference.checkpoints
    assert isinstance(deferred.epoch_avgs[0], float)
    assert deferred.epoch_avgs == reference.epoch_avgs


def test_only_logging_steps_hold_python_floats(stub_module):
    deferred_host_sync.patch_recipe(stub_module.Recipe)
    recipe = _run(stub_module.Recipe, _steps(), log_every=2, ckpt_every=0)

    for step, types_ in enumerate(recipe.types, start=1):
        expected = float if step % 2 == 0 else torch.Tensor
        assert types_ == (expected, expected, expected), f"step {step}"


def test_the_rewrite_is_idempotent(stub_module):
    deferred_host_sync.patch_recipe(stub_module.Recipe)
    patched = stub_module.Recipe.run_train_validation_loop

    assert deferred_host_sync.patch_recipe(stub_module.Recipe) is True
    assert stub_module.Recipe.run_train_validation_loop is patched


def test_a_profiler_wrapped_loop_is_refused(stub_module):
    def wrapped(self):
        pass

    wrapped._primus_profile_wrapped = True
    stub_module.Recipe.run_train_validation_loop = wrapped

    with pytest.raises(RuntimeError, match="profiler"):
        deferred_host_sync.patch_recipe(stub_module.Recipe)


def _pinned_train_loop() -> Path:
    """AutoModel's diffusion train loop: the submodule checkout, else the installed package."""
    packages = [REPO_ROOT / "third_party" / "Automodel" / "nemo_automodel"]
    spec = importlib.util.find_spec("nemo_automodel")
    if spec is not None and spec.submodule_search_locations:
        packages.append(Path(list(spec.submodule_search_locations)[0]))
    for package in packages:
        train_py = package / "recipes" / "diffusion" / "train.py"
        if train_py.is_file():
            return train_py
    pytest.skip("neither the AutoModel submodule nor nemo_automodel is available")


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

    assert deferred_host_sync.patch_recipe(module.TrainDiffusionRecipe) is True


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


def test_the_patches_are_registered_gated_and_run_before_the_profiler(set_option):
    from primus.core.patches.patch_registry import PatchRegistry

    profiler = PatchRegistry.get("automodel.profiling.torch_profiler")
    loop = PatchRegistry.get("automodel.training.deferred_host_sync")
    ids = PatchRegistry.get("automodel.models.flux.latent_ids")
    assert loop is not None and ids is not None and profiler is not None
    assert loop.priority < profiler.priority

    assert loop.condition(_ctx()) is False and ids.condition(_ctx("FluxTransformer2DModel")) is False
    set_option("primus_train.defer_host_sync", True)
    assert loop.condition(_ctx("WanTransformer3DModel")) is True
    assert ids.condition(_ctx("FluxTransformer2DModel")) is True
    assert ids.condition(_ctx("WanTransformer3DModel")) is False, "the latent ids are FLUX's"


# --------------------------------------------------------------------------- #
# FLUX latent ids                                                             #
# --------------------------------------------------------------------------- #
MODULE_PATH = "nemo_automodel.components.flow_matching.adapters.flux"


def _upstream_latent_image_ids(self, batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height // 2, width // 2, 3)
    latent_image_ids[..., 1] = torch.arange(height // 2)[:, None]
    latent_image_ids[..., 2] = torch.arange(width // 2)[None, :]

    latent_image_ids = latent_image_ids.reshape(-1, 3)
    return latent_image_ids.to(device=device, dtype=dtype)


@pytest.fixture
def stub_flux_adapter(monkeypatch):
    class FluxAdapter:
        _prepare_latent_image_ids = _upstream_latent_image_ids

    module = types.ModuleType(MODULE_PATH)
    module.FluxAdapter = FluxAdapter
    parts = MODULE_PATH.split(".")
    for index in range(1, len(parts)):
        name = ".".join(parts[:index])
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = []
            monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(sys.modules, MODULE_PATH, module)
    parent_name, _, leaf = MODULE_PATH.rpartition(".")
    monkeypatch.setattr(sys.modules[parent_name], leaf, module, raising=False)
    return FluxAdapter


_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("height,width", [(32, 32), (32, 48)])
def test_the_ids_match_upstream(stub_flux_adapter, device, dtype, height, width):
    adapter = stub_flux_adapter()
    want = adapter._prepare_latent_image_ids(2, height, width, torch.device(device), dtype)

    assert latent_ids.install() is True
    got = adapter._prepare_latent_image_ids(2, height, width, torch.device(device), dtype)

    assert got.shape == want.shape == ((height // 2) * (width // 2), 3)
    assert got.dtype == dtype
    assert got.device.type == device
    assert torch.equal(got, want)


def test_the_replacement_is_idempotent(stub_flux_adapter):
    assert latent_ids.install() is True
    patched = stub_flux_adapter._prepare_latent_image_ids
    assert patched is not _upstream_latent_image_ids

    assert latent_ids.install() is True
    assert stub_flux_adapter._prepare_latent_image_ids is patched
