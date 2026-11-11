###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import ast
import importlib
import importlib.util
import pkgutil
from collections import defaultdict
from pathlib import Path

import pytest

from primus.backends.automodel import options
from primus.core.patches.patch_registry import PatchRegistry

REPO_ROOT = Path(__file__).resolve().parents[4]


def _registry_state():
    return PatchRegistry._patches_by_backend_phase, PatchRegistry._all_patches


def _set_registry_state(state):
    by_phase, all_patches = state
    PatchRegistry._patches_by_backend_phase = by_phase
    PatchRegistry._all_patches = all_patches


def _copy_registry_state(state):
    by_phase, all_patches = state
    copied = defaultdict(lambda: defaultdict(list))
    for backend, phases in by_phase.items():
        for phase, patches in phases.items():
            copied[backend][phase] = list(patches)
    return copied, list(all_patches)


def production_registry():
    """The registry as this backend's patch modules fill it, built from scratch.

    The patch modules register at import, and Python imports a module once, so
    a test elsewhere that empties the global registry (``main``'s
    ``tests/unit_tests/core/patches`` do, without restoring it) would leave
    every later test here without them. Each module is re-executed against an
    empty registry instead of trusting whatever the session left behind.
    """
    import primus.backends.automodel.patches as package

    saved = _registry_state()
    _set_registry_state((defaultdict(lambda: defaultdict(list)), []))
    try:
        for info in pkgutil.walk_packages(package.__path__, prefix=package.__name__ + "."):
            if info.name.endswith(("_patches", "_patch")):
                importlib.reload(importlib.import_module(info.name))
        return _copy_registry_state(_registry_state())
    finally:
        _set_registry_state(saved)


@pytest.fixture(scope="session")
def _production_registry():
    return production_registry()


@pytest.fixture
def build_production_registry():
    return production_registry


@pytest.fixture(autouse=True)
def _patch_registry(_production_registry):
    """Every test sees this backend's patches registered, whatever ran before it."""
    saved = _registry_state()
    _set_registry_state(_copy_registry_state(_production_registry))
    yield
    _set_registry_state(saved)


@pytest.fixture(autouse=True)
def _default_options():
    """Every test starts from the defaults: settings are process-global."""
    options.load({})
    yield
    options.load({})


@pytest.fixture
def set_option():
    """``set_option("primus_turbo.fp8_linear", True)``; ``None`` unsets."""
    sections = {}

    def _set(path, value):
        section, key = path.split(".", 1)
        sections.setdefault(section, {})[key] = value
        options.load(sections)

    return _set


@pytest.fixture
def pinned_automodel():
    """``pinned_automodel("components/x.py", "Class", "method", ...)``: upstream source.

    Returns the named methods of the class, or the named top-level functions
    when the class is None, at their original indentation. The pinned AutoModel
    is read from the submodule checkout, else the installed package, rather
    than imported, so a test runs where AutoModel is checked out but not
    installed, as in CI, and never patches the real classes.
    """

    def _definitions(relative, class_name, *names):
        packages = [REPO_ROOT / "third_party" / "Automodel" / "nemo_automodel"]
        spec = importlib.util.find_spec("nemo_automodel")
        if spec is not None and spec.submodule_search_locations:
            packages.append(Path(list(spec.submodule_search_locations)[0]))
        path = next((package / relative for package in packages if (package / relative).is_file()), None)
        if path is None:
            pytest.skip("neither the AutoModel submodule nor nemo_automodel is available")

        lines = path.read_text().splitlines(keepends=True)
        body = ast.parse("".join(lines)).body
        if class_name is not None:
            classes = [n for n in body if isinstance(n, ast.ClassDef) and n.name == class_name]
            assert classes, f"{relative} no longer defines {class_name}"
            body = classes[0].body
        found = {n.name: n for n in body if isinstance(n, ast.FunctionDef)}
        missing = [name for name in names if name not in found]
        assert not missing, f"{relative}: {class_name or 'module'} no longer defines {missing}"
        chunks = []
        for name in names:
            node = found[name]
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            chunks.append("".join(lines[start - 1 : node.end_lineno]))
        return "\n".join(chunks)

    return _definitions
