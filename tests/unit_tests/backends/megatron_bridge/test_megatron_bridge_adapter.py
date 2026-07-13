###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for MegatronBridgeAdapter's ROCm-compat dependency-stub machinery.

These shims let ``import megatron.bridge`` succeed without the NV-only optional
stack (modelopt[torch], megatron-energon, VLM transformers classes). They are pure
sys.modules/attribute/AST logic with real regression risk (the dunder-passthrough
and sys.modules-before-sentinel rules are load-bearing), so they're worth a CPU test.

Import is deferred into a fixture (like test_config_utils.py) so the package's
torch-pulling __init__ doesn't run at collection time under the coverage C tracer.
"""

import sys

import pytest


@pytest.fixture
def adapter():
    import primus.backends.megatron_bridge.megatron_bridge_adapter as m

    return m


@pytest.fixture
def sys_modules_guard(adapter):
    """Isolate stub-eligible modules for the duration of a test.

    Drops any relevant modules BOTH before and after the test. The
    before-drop matters because some entries (e.g. ``megatron.energon``) are
    real, top-level packages that may already be installed and cached in
    ``sys.modules`` from elsewhere in the session. If left in place, the stub
    installer's ``if pkg_name in sys.modules: continue`` fast-path would skip
    them, bypassing a test's forced-missing ``import_module`` mock and breaking
    assertions that expect every package to be stubbed.
    """
    prefixes = ("modelopt",) + adapter._BRIDGE_OPTIONAL_PACKAGES

    def _drop():
        for key in list(sys.modules):
            if any(key == p or key.startswith(p + ".") for p in prefixes):
                sys.modules.pop(key, None)

    _drop()
    yield
    _drop()


# ---------------------------------------------------------------------------
# _BridgeOptionalStub / _BridgeOptionalSentinel (module-level, env-independent)
# ---------------------------------------------------------------------------
def test_bridge_optional_stub_looks_like_namespace_package(adapter):
    import os

    stub = adapter._BridgeOptionalStub("megatron.bridge.models.qwen_vl")
    assert stub.__file__ == os.devnull  # never "" (would read as builtin)
    assert stub.__path__ == []
    assert stub.__all__ == []  # `from stub import *` exports nothing
    assert stub.__spec__ is None


def test_bridge_optional_stub_dunder_access_raises_attributeerror(adapter):
    # Must NOT mint a sentinel for dunders: Primus patch infra walks sys.modules
    # and calls module.__file__.endswith(...); intercepting dunders would crash it.
    stub = adapter._BridgeOptionalStub("pkg")
    with pytest.raises(AttributeError):
        stub.__some_missing_dunder__


def test_bridge_optional_stub_mints_sentinel_for_normal_attr(adapter):
    stub = adapter._BridgeOptionalStub("pkg")
    cls = stub.SomeClass
    # isinstance against a real object must be False (bridge relies on this)
    assert isinstance(object(), cls) is False
    # instantiating the sentinel is a loud, helpful error
    with pytest.raises(RuntimeError, match="stubbed out by Primus"):
        cls()


def test_bridge_optional_stub_prefers_existing_sys_module(adapter):
    # `import a.b.c as X` is `X = getattr(a.b, 'c')`; if a child stub already
    # lives in sys.modules it must be returned instead of a fresh sentinel.
    parent = adapter._BridgeOptionalStub("pkg")
    child = adapter._BridgeOptionalStub("pkg.child")
    sys.modules["pkg.child"] = child
    try:
        assert parent.child is child
    finally:
        sys.modules.pop("pkg.child", None)


# ---------------------------------------------------------------------------
# _install_bridge_optional_stubs (force the "missing" path via importlib mock)
# ---------------------------------------------------------------------------
def test_install_bridge_optional_stubs_injects_and_wires(adapter, monkeypatch, sys_modules_guard):
    def _always_missing(name):
        raise ImportError(f"forced-missing: {name}")

    monkeypatch.setattr(adapter.importlib, "import_module", _always_missing)

    stubbed = adapter._install_bridge_optional_stubs()

    assert set(stubbed) == set(adapter._BRIDGE_OPTIONAL_PACKAGES)
    for pkg in adapter._BRIDGE_OPTIONAL_PACKAGES:
        assert isinstance(sys.modules.get(pkg), adapter._BridgeOptionalStub)
    # parent -> child attribute wiring for dotted entries
    for pkg in adapter._BRIDGE_OPTIONAL_PACKAGES:
        if "." in pkg:
            parent_name, child_name = pkg.rsplit(".", 1)
            parent = sys.modules.get(parent_name)
            if isinstance(parent, adapter._BridgeOptionalStub):
                assert getattr(parent, child_name) is sys.modules[pkg]


def test_install_bridge_optional_stubs_is_idempotent(adapter, monkeypatch, sys_modules_guard):
    monkeypatch.setattr(
        adapter.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError(name)),
    )
    adapter._install_bridge_optional_stubs()
    # second call: everything already in sys.modules -> nothing new stubbed
    assert adapter._install_bridge_optional_stubs() == []


# ---------------------------------------------------------------------------
# _install_transformers_stub (transformers must be importable)
# ---------------------------------------------------------------------------
def test_install_transformers_stub_adds_placeholders_for_missing(adapter):
    transformers = pytest.importorskip("transformers")

    name, _used_by = adapter._TRANSFORMERS_PLACEHOLDER_CLASSES[0]
    had_attr = hasattr(transformers, name)
    original = getattr(transformers, name, None)
    if had_attr:
        delattr(transformers, name)
    try:
        stubbed = adapter._install_transformers_stub()
        assert name in stubbed
        placeholder = getattr(transformers, name)
        assert getattr(placeholder, "_primus_placeholder", False) is True
        with pytest.raises(RuntimeError, match="Primus placeholder"):
            placeholder()
    finally:
        # restore: remove every placeholder we (or the call) injected
        for cname, _ in adapter._TRANSFORMERS_PLACEHOLDER_CLASSES:
            obj = getattr(transformers, cname, None)
            if getattr(obj, "_primus_placeholder", False):
                delattr(transformers, cname)
        if had_attr:
            setattr(transformers, name, original)


def test_install_transformers_stub_skips_existing_classes(adapter):
    transformers = pytest.importorskip("transformers")

    name, _used_by = adapter._TRANSFORMERS_PLACEHOLDER_CLASSES[0]
    sentinel = type("RealClass", (), {})
    had_attr = hasattr(transformers, name)
    original = getattr(transformers, name, None)
    setattr(transformers, name, sentinel)
    try:
        stubbed = adapter._install_transformers_stub()
        assert name not in stubbed  # already present -> not overwritten
        assert getattr(transformers, name) is sentinel
    finally:
        for cname, _ in adapter._TRANSFORMERS_PLACEHOLDER_CLASSES:
            obj = getattr(transformers, cname, None)
            if getattr(obj, "_primus_placeholder", False):
                delattr(transformers, cname)
        if had_attr:
            setattr(transformers, name, original)
        elif hasattr(transformers, name):
            delattr(transformers, name)


# ---------------------------------------------------------------------------
# _install_modelopt_stub: force the "missing" branch (env-independent).
# The real modelopt[torch] is often half-installed on ROCm, so we drop any
# cached modelopt modules and make every `import modelopt...` fail, which is
# exactly the situation the stub exists to handle.
# ---------------------------------------------------------------------------
def test_install_modelopt_stub_injects_when_missing(adapter, monkeypatch, sys_modules_guard):
    import builtins

    real_import = builtins.__import__
    for key in [k for k in sys.modules if k == "modelopt" or k.startswith("modelopt.")]:
        monkeypatch.delitem(sys.modules, key, raising=False)

    def fake_import(name, *args, **kwargs):
        if name.startswith("modelopt"):
            raise ImportError("forced-missing modelopt for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert adapter._install_modelopt_stub() is True
    for path in ("modelopt", "modelopt.torch", "modelopt.torch.distill"):
        assert path in sys.modules

    stub = sys.modules["modelopt.torch.distill"]
    # dunder passthrough (must not mint sentinel; patch infra relies on this)
    with pytest.raises(AttributeError):
        stub.__some_missing_dunder__
    # normal attr mints a sentinel: isinstance False, call raises a guiding error
    sentinel_cls = stub.DistillationModel
    assert isinstance(object(), sentinel_cls) is False
    with pytest.raises(RuntimeError, match="nvidia-modelopt"):
        sentinel_cls()


# ---------------------------------------------------------------------------
# detect_backend_version (AST parse of package_info.py; env-independent)
# ---------------------------------------------------------------------------
def test_detect_backend_version_parses_package_info(adapter, tmp_path, monkeypatch):
    pkg_info = tmp_path / "megatron" / "bridge" / "package_info.py"
    pkg_info.parent.mkdir(parents=True)
    pkg_info.write_text('__version__ = "9.9.9-test"\nfoo = 1\n')
    monkeypatch.syspath_prepend(str(tmp_path))

    a = adapter.MegatronBridgeAdapter()
    assert a.detect_backend_version() == "9.9.9-test"


# ---------------------------------------------------------------------------
# load_trainer_class (registry path + invalid stage; no trainer import)
# ---------------------------------------------------------------------------
def test_load_trainer_class_uses_registry(adapter, monkeypatch):
    sentinel = type("DummyTrainer", (), {})
    monkeypatch.setattr(
        adapter.BackendRegistry,
        "get_trainer_class",
        staticmethod(lambda framework, stage: sentinel),
    )
    a = adapter.MegatronBridgeAdapter()
    assert a.load_trainer_class("sft") is sentinel


def test_load_trainer_class_invalid_stage_raises(adapter, monkeypatch):
    def _miss(framework, stage):
        raise ValueError("not registered")

    monkeypatch.setattr(adapter.BackendRegistry, "get_trainer_class", staticmethod(_miss))
    a = adapter.MegatronBridgeAdapter()
    with pytest.raises(ValueError, match="Invalid stage"):
        a.load_trainer_class("not-a-real-stage")
