###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared fixtures for the Primus parallelization sidecar tests.

Two kinds of support:

* ``install_stub_automodel`` replaces the few AutoModel names the sidecars touch
  with recording stand-ins, so the sidecars' own logic is tested without
  AutoModel or a GPU. A sidecar that starts using something new fails here with
  AttributeError, which is the intended signal.
* ``requires_automodel`` and the ``tiny_*`` builders back the contract tests,
  which run the sidecars against the real pinned AutoModel on CPU and are
  skipped when it is not installed.
"""

import dataclasses
import importlib.util
import sys
import types

import pytest

DIST_PATH = "nemo_automodel.components.distributed"
AC_PATH = "nemo_automodel.components.distributed.activation_checkpointing"
REGISTRY_PATH = "nemo_automodel._diffusers.parallelization"

requires_automodel = pytest.mark.skipif(
    importlib.util.find_spec("nemo_automodel") is None, reason="nemo_automodel is not installed"
)


@dataclasses.dataclass
class FakeConfig:
    activation_checkpointing: object = False
    enable_compile: bool = False


@dataclasses.dataclass
class FakeMeshContext:
    strategy_config: object = dataclasses.field(default_factory=FakeConfig)
    activation_checkpointing: object = False


def _ensure_parent_packages(monkeypatch, dotted_path):
    parts = dotted_path.split(".")
    for i in range(1, len(parts)):
        name = ".".join(parts[:i])
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or types.ModuleType(name))


def install_stub_module(monkeypatch, dotted_path, **attributes):
    """Install a stub module, creating parents and binding it on its parent."""
    _ensure_parent_packages(monkeypatch, dotted_path)
    module = types.ModuleType(dotted_path)
    for key, value in attributes.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, dotted_path, module)
    parent, _, leaf = dotted_path.rpartition(".")
    monkeypatch.setattr(sys.modules[parent], leaf, module, raising=False)
    return module


def install_stub_automodel(monkeypatch):
    """Stub AutoModel's sidecar contract and AC helpers; return the recorders.

    The returned namespace has ``registry`` (the stub ``_PARALLELIZERS``), the
    ``ModelParallelizer`` and ``WanModelParallelizer`` stand-ins, and lists
    recording each call: ``parallelize`` (mesh contexts), ``apply`` (kwargs),
    ``sharding`` ((args, kwargs)), ``selective`` and ``full`` (layer lists).
    """
    rec = types.SimpleNamespace(parallelize=[], apply=[], sharding=[], selective=[], full=[])

    class ModelParallelizer:
        def parallelize(self, model, mesh_context, /):
            rec.parallelize.append(mesh_context)
            return model

        def _apply(self, model, *args, **kwargs):
            rec.apply.append(kwargs)
            return model

    install_stub_module(monkeypatch, DIST_PATH, ModelParallelizer=ModelParallelizer)
    install_stub_module(
        monkeypatch,
        AC_PATH,
        is_selective_activation_checkpointing=lambda value: value == "selective",
        apply_selective_checkpointing_to_layers=lambda model, layers, kv, **kw: rec.selective.append(
            list(layers)
        ),
        apply_full_layer_checkpointing_to_layers=lambda model, layers: rec.full.append(list(layers)),
    )
    registry_module = install_stub_module(monkeypatch, REGISTRY_PATH, _PARALLELIZERS={})

    def apply_fsdp2_sharding_recursively(*args, **kwargs):
        rec.sharding.append((args, kwargs))

    registry_module.apply_fsdp2_sharding_recursively = apply_fsdp2_sharding_recursively

    class WanModelParallelizer(ModelParallelizer):
        """Mimics upstream: truthiness AC test, seven positional sharding args."""

        def _apply(self, model, device_mesh=None, activation_checkpointing=False, **kwargs):
            rec.apply.append(dict(activation_checkpointing=activation_checkpointing, **kwargs))
            if activation_checkpointing:
                rec.full.append(list(model.blocks))
            registry_module.apply_fsdp2_sharding_recursively(model, device_mesh, None, None, True, 2, 1)
            return model

    WanModelParallelizer._apply.__module__ = REGISTRY_PATH

    rec.registry = registry_module._PARALLELIZERS
    rec.registry_module = registry_module
    rec.ModelParallelizer = ModelParallelizer
    rec.WanModelParallelizer = WanModelParallelizer
    return rec


# --- real-AutoModel builders -------------------------------------------------


def tiny_flux():
    """A CPU FLUX transformer with 2 dual-stream and 3 single-stream blocks."""
    from diffusers import FluxTransformer2DModel

    return FluxTransformer2DModel(
        patch_size=1,
        in_channels=4,
        num_layers=2,
        num_single_layers=3,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=32,
        pooled_projection_dim=16,
        guidance_embeds=False,
        axes_dims_rope=(4, 6, 6),
    )


def tiny_wan():
    """A CPU Wan transformer with 2 blocks."""
    from diffusers import WanTransformer3DModel

    return WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=16,
        in_channels=4,
        out_channels=4,
        text_dim=32,
        freq_dim=32,
        ffn_dim=64,
        num_layers=2,
    )


def is_checkpoint_wrapped(module):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointWrapper,
    )

    return isinstance(module, CheckpointWrapper)


def record_strategy_dispatch(monkeypatch):
    """Record what AutoModel's FSDP2 and DDP dispatch would receive.

    Returns a list of ``(strategy, mesh_context)`` tuples. Everything up to the
    dispatch -- the sidecar lookup and the sidecar's own ``parallelize`` -- is
    real; only the final sharding, which needs a process group, is recorded.
    """
    from nemo_automodel.components.distributed import model_parallelizer as mp

    seen = []
    monkeypatch.setattr(
        mp, "_parallelize_fsdp2", lambda model, ctx, *, parallelizer: seen.append(("fsdp2", ctx)) or model
    )
    monkeypatch.setattr(mp, "_parallelize_ddp", lambda model, ctx: seen.append(("ddp", ctx)) or model)
    return seen


def real_mesh_context(strategy, activation_checkpointing):
    from nemo_automodel.components.distributed.config import DDPConfig, FSDP2Config
    from nemo_automodel.components.distributed.mesh import MeshContext

    config = DDPConfig() if strategy == "ddp" else FSDP2Config()
    return MeshContext(strategy_config=config, activation_checkpointing=activation_checkpointing)


def attach_and_parallelize(model, mesh_context):
    """Run the pipeline's own attach-then-parallelize sequence on ``model``."""
    from nemo_automodel._diffusers.parallelization import attach_parallelizer
    from nemo_automodel.components.distributed.model_parallelizer import (
        parallelize_model,
    )

    attach_parallelizer(model)
    return parallelize_model(model, mesh_context)
