###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the on-device FLUX latent ids (``models/flux/latent_ids``).

WHY THIS TEST EXISTS:
  The patch applies to every FLUX run with no setting, on the promise that the
  ids are the same values AutoModel builds. So the tests pin that equality across
  devices, dtypes and non-square shapes, and that a changed upstream method is
  left alone rather than replaced.
"""

import ast
import inspect
import logging
import sys
import textwrap
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from primus.backends.automodel.models.flux import latent_ids  # noqa: E402

MODULE_PATH = "nemo_automodel.components.flow_matching.adapters.flux"
LOGGER = "primus.backends.automodel.models.flux.latent_ids"
PINNED_ADAPTER = (
    Path(__file__).resolve().parents[4]
    / "third_party/Automodel/nemo_automodel/components/flow_matching/adapters/flux.py"
)


def _upstream_latent_image_ids(self, batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height // 2, width // 2, 3)
    latent_image_ids[..., 1] = torch.arange(height // 2)[:, None]
    latent_image_ids[..., 2] = torch.arange(width // 2)[None, :]

    latent_image_ids = latent_image_ids.reshape(-1, 3)
    return latent_image_ids.to(device=device, dtype=dtype)


def _body_without_docstring(function):
    body = function.body
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
        body = body[1:]
    return [ast.dump(statement) for statement in body]


def test_the_copied_upstream_method_is_the_pinned_one():
    """Fails when AutoModel's method changes, so the tests here keep comparing against what it builds.

    Read from the submodule's source, which CI checks out without installing AutoModel.
    """
    if not PINNED_ADAPTER.is_file():
        pytest.skip("the AutoModel submodule is not checked out")
    adapter = next(
        node
        for node in ast.parse(PINNED_ADAPTER.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "FluxAdapter"
    )
    pinned = next(node for node in adapter.body if getattr(node, "name", None) == "_prepare_latent_image_ids")
    copied = ast.parse(textwrap.dedent(inspect.getsource(_upstream_latent_image_ids))).body[0]

    assert [a.arg for a in pinned.args.args] == [a.arg for a in copied.args.args]
    assert _body_without_docstring(pinned) == _body_without_docstring(copied)


def _install_stub_adapter(monkeypatch, method):
    class FluxAdapter:
        _prepare_latent_image_ids = method

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


@pytest.fixture
def stub_flux_adapter(monkeypatch):
    return _install_stub_adapter(monkeypatch, _upstream_latent_image_ids)


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


def test_a_method_with_a_new_signature_is_left_alone(monkeypatch, caplog):
    def changed(self, batch_size, height, width, device, dtype, patch_size=2):
        return None

    adapter = _install_stub_adapter(monkeypatch, changed)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert latent_ids.install() is False

    assert adapter._prepare_latent_image_ids is changed
    assert "has a new signature; not installed" in caplog.text


def test_a_missing_adapter_is_left_alone(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, MODULE_PATH, None)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert latent_ids.install() is False

    assert "FluxAdapter is not importable" in caplog.text


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


def test_the_patch_applies_to_flux_with_no_setting():
    from primus.core.patches.patch_registry import PatchRegistry

    patch = PatchRegistry.get("automodel.models.flux.latent_ids")

    assert patch is not None
    assert patch.condition(_ctx("FluxTransformer2DModel")) is True
    assert patch.condition(_ctx("WanTransformer3DModel")) is False, "the latent ids are FLUX's"
