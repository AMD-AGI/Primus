###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the on-device FLUX latent ids (``models/flux/latent_ids``).

WHY THIS TEST EXISTS:
  The patch applies to every FLUX run with no setting, on the promise that the
  ids are the same values AutoModel builds, built on the device. So the tests pin
  that equality across devices, dtypes and non-square shapes, that no tensor is
  built on the host, and that an upstream method other than the pinned one is
  left alone rather than replaced.
"""

import ast
import importlib.util
import inspect
import logging
import sys
import textwrap
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402
from torch.utils._pytree import tree_leaves  # noqa: E402

from primus.backends.automodel.models.flux import latent_ids  # noqa: E402

MODULE_PATH = "nemo_automodel.components.flow_matching.adapters.flux"
LOGGER = "primus.backends.automodel.models.flux.latent_ids"
REPO_ROOT = Path(__file__).resolve().parents[4]
ADAPTER_FILE = Path("components") / "flow_matching" / "adapters" / "flux.py"


def _pinned_adapter_file() -> Path:
    """AutoModel's FLUX adapter: the submodule checkout, else the installed package."""
    packages = [REPO_ROOT / "third_party" / "Automodel" / "nemo_automodel"]
    spec = importlib.util.find_spec("nemo_automodel")
    if spec is not None and spec.submodule_search_locations:
        packages.append(Path(list(spec.submodule_search_locations)[0]))
    path = next((package / ADAPTER_FILE for package in packages if (package / ADAPTER_FILE).is_file()), None)
    if path is None:
        pytest.skip("neither the AutoModel submodule nor nemo_automodel is available")
    return path


def _body_without_docstring(function):
    body = function.body
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
        body = body[1:]
    return [ast.dump(statement) for statement in body]


def test_the_copied_upstream_method_is_the_pinned_one():
    """Fails when AutoModel's method changes, so ``install`` and the tests here keep matching what it builds.

    Read from the source rather than imported, so it runs where AutoModel is
    checked out but not installed, as in CI.
    """
    path = _pinned_adapter_file()
    adapter = next(
        (
            node
            for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.ClassDef) and node.name == "FluxAdapter"
        ),
        None,
    )
    assert adapter is not None, f"no FluxAdapter class in {path}"
    pinned = next(
        (node for node in adapter.body if getattr(node, "name", None) == "_prepare_latent_image_ids"), None
    )
    assert pinned is not None, f"FluxAdapter in {path} has no _prepare_latent_image_ids"
    copied = ast.parse(textwrap.dedent(inspect.getsource(latent_ids._pinned_upstream))).body[0]

    assert [a.arg for a in pinned.args.args] == [a.arg for a in copied.args.args]
    assert _body_without_docstring(pinned) == _body_without_docstring(copied)


def _install_stub_adapter(monkeypatch, method=None):
    class FluxAdapter:
        pass

    if method is not None:
        FluxAdapter._prepare_latent_image_ids = method
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
    return _install_stub_adapter(monkeypatch, latent_ids._pinned_upstream)


_DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")),
]


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


class _RecordDevices(TorchDispatchMode):
    """Records the device of every tensor an op returns."""

    def __init__(self):
        super().__init__()
        self.devices = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        self.devices.update(t.device.type for t in tree_leaves(out) if isinstance(t, torch.Tensor))
        return out


def test_the_ids_are_built_on_the_target_device(stub_flux_adapter):
    """On the meta device, so it runs without a GPU: a tensor on the CPU means a build on the host."""
    adapter = stub_flux_adapter()
    meta = torch.device("meta")
    with _RecordDevices() as upstream:
        adapter._prepare_latent_image_ids(2, 32, 48, meta, torch.bfloat16)

    assert latent_ids.install() is True
    with _RecordDevices() as replacement:
        ids = adapter._prepare_latent_image_ids(2, 32, 48, meta, torch.bfloat16)

    assert "cpu" in upstream.devices
    assert replacement.devices == {"meta"}
    assert ids.shape == (16 * 24, 3)


def test_the_replacement_is_idempotent(stub_flux_adapter):
    assert latent_ids.install() is True
    assert stub_flux_adapter._prepare_latent_image_ids is latent_ids._prepare_latent_image_ids

    def rebound(self, batch_size, height, width, device, dtype):
        return None

    stub_flux_adapter._prepare_latent_image_ids = rebound
    assert latent_ids.install() is True
    assert stub_flux_adapter._prepare_latent_image_ids is rebound, "a second install() changes nothing"


def test_a_method_with_a_new_signature_is_left_alone(monkeypatch, caplog):
    def changed(self, batch_size, height, width, device, dtype, patch_size=2):
        return None

    adapter = _install_stub_adapter(monkeypatch, changed)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert latent_ids.install() is False

    assert adapter._prepare_latent_image_ids is changed
    assert "not the pinned AutoModel's; not installed" in caplog.text


def test_a_method_with_a_new_body_is_left_alone(monkeypatch, caplog):
    def changed(self, batch_size, height, width, device, dtype):
        return torch.zeros((height // 2) * (width // 2), 3, device=device, dtype=dtype)

    adapter = _install_stub_adapter(monkeypatch, changed)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert latent_ids.install() is False

    assert adapter._prepare_latent_image_ids is changed
    assert "not the pinned AutoModel's; not installed" in caplog.text


def test_a_missing_method_is_left_alone(monkeypatch, caplog):
    adapter = _install_stub_adapter(monkeypatch)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert latent_ids.install() is False

    assert not hasattr(adapter, "_prepare_latent_image_ids")
    assert "is missing or not the pinned AutoModel's" in caplog.text


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


def test_the_patch_applies_to_flux_with_no_setting(stub_flux_adapter):
    from primus.backends.automodel.patches.flux_latent_ids_patches import apply
    from primus.core.patches.patch_registry import PatchRegistry

    patch = PatchRegistry.get("automodel.models.flux.latent_ids")

    assert patch is not None
    assert patch.condition(_ctx("FluxTransformer2DModel")) is True
    assert patch.condition(_ctx()) is True, "a config that names no transformer, as with from_pretrained"
    assert patch.condition(_ctx("WanTransformer3DModel")) is False, "the latent ids are FLUX's"

    apply(_ctx("FluxTransformer2DModel"))
    assert stub_flux_adapter._prepare_latent_image_ids is latent_ids._prepare_latent_image_ids
