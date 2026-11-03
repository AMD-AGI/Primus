###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for primus.core.utils.import_utils.lazy_import.

lazy_import tries a list of module paths and returns the first one that exposes
the requested symbol, else raises ImportError. Tested with real stdlib modules
so no mocking of importlib is required.
"""

import pytest

from primus.core.utils.import_utils import lazy_import


@pytest.fixture(autouse=True)
def _silence_logger(monkeypatch):
    """lazy_import logs via log_rank_0, which needs a configured global logger.
    Stub it so these tests exercise import logic without logger setup."""
    monkeypatch.setattr("primus.core.utils.import_utils.log_rank_0", lambda *a, **k: None)


def test_lazy_import_returns_symbol_from_first_valid_path():
    # `os.getcwd` exists -> returned directly
    fn = lazy_import(["os"], "getcwd")
    assert fn() == __import__("os").getcwd()


def test_lazy_import_falls_back_to_later_path():
    # first module exists but lacks the symbol path resolution still finds it in
    # a later candidate module that does provide it.
    sep = lazy_import(["nonexistent_module_xyz", "os"], "sep")
    assert sep == __import__("os").sep


def test_lazy_import_raises_when_symbol_missing_everywhere():
    with pytest.raises(ImportError):
        lazy_import(["nonexistent_a", "nonexistent_b"], "whatever")


# The Primus-owned model types this checkout registers. A literal on purpose: dropping a type
# from _PRIMUS_MODEL_PROVIDERS (or adding one without a test) fails here.
_EXPECTED_PRIMUS_MODEL_TYPES = {"deepseek_v4", "glm5_next", "inkling", "kimi_k3"}


def test_primus_model_provider_table_has_the_expected_types():
    from primus.core.utils.import_utils import _PRIMUS_MODEL_PROVIDERS

    assert set(_PRIMUS_MODEL_PROVIDERS) == _EXPECTED_PRIMUS_MODEL_TYPES


@pytest.mark.parametrize("model_type", sorted(_EXPECTED_PRIMUS_MODEL_TYPES))
def test_primus_model_provider_table_names_existing_builders(model_type):
    """Each entry's builders module is a file in this checkout that defines model_provider and
    the named builder. Read from the source, not imported: the builders need Megatron and a
    GPU at import, which this CPU test does not have. A stale or typo'd path fails, never skips."""
    import ast
    from pathlib import Path

    import primus
    from primus.core.utils.import_utils import _PRIMUS_MODEL_PROVIDERS

    module_path, builder_name, display_name = _PRIMUS_MODEL_PROVIDERS[model_type]
    assert module_path.startswith("primus.") and display_name
    root = Path(primus.__file__).resolve().parent.parent
    path = root.joinpath(*module_path.split(".")).with_suffix(".py")
    assert path.is_file(), f"{model_type}: {module_path} is not a module in this checkout ({path})"
    names = {
        n.name for n in ast.parse(path.read_text()).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
    } | {
        t.id
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Name)
    }
    assert "model_provider" in names, f"{path} defines no model_provider"
    assert builder_name in names, f"{path} defines no {builder_name}"


@pytest.mark.parametrize("model_type", sorted(_EXPECTED_PRIMUS_MODEL_TYPES))
def test_get_model_provider_binds_each_primus_builder(model_type, monkeypatch):
    """Each Primus-owned model type imports exactly its builders module and returns its
    model_provider bound to its builder. The import is stubbed: the builders' own dependencies
    (Primus-Turbo, which needs a GPU at import) are not this CPU test's subject."""
    import types

    from primus.core.utils import import_utils

    module_path, builder_name, _ = import_utils._PRIMUS_MODEL_PROVIDERS[model_type]
    fake = types.ModuleType(module_path)
    fake.model_provider = lambda *a, **k: None
    builder = object()
    setattr(fake, builder_name, builder)
    imported = []

    def import_module(name, *a, **k):
        imported.append(name)
        if name != module_path:
            raise AssertionError(f"get_model_provider({model_type!r}) imported {name}, not {module_path}")
        return fake

    monkeypatch.setattr(import_utils.importlib, "import_module", import_module)
    provider = import_utils.get_model_provider(model_type=model_type)
    assert imported == [module_path]
    assert provider.func is fake.model_provider
    assert provider.args == (builder,)


def test_a_type_registered_only_in_the_table_resolves_from_it(monkeypatch):
    """get_model_provider routes on the table itself (as the pretrain trainer does): a new entry
    binds its builder, never the upstream GPT provider."""
    import types

    from primus.core.utils import import_utils

    fake = types.ModuleType("fake_newm_builders")
    fake.model_provider = lambda *a, **k: None
    fake.newm_builder = object()
    monkeypatch.setitem(
        import_utils._PRIMUS_MODEL_PROVIDERS,
        "newm",
        import_utils.PrimusModelProvider("fake_newm_builders", "newm_builder", "NewM"),
    )
    monkeypatch.setattr(import_utils.importlib, "import_module", lambda name, *a, **k: fake)

    def no_gpt(*_a, **_k):
        raise AssertionError("the upstream GPT provider was resolved")

    monkeypatch.setattr(import_utils, "lazy_import", no_gpt)
    provider = import_utils.get_model_provider(model_type="newm")
    assert provider.func is fake.model_provider and provider.args == (fake.newm_builder,)


def test_public_accessors_read_the_table():
    from primus.core.utils import import_utils

    assert import_utils.is_primus_model_type("inkling")
    assert not import_utils.is_primus_model_type("gpt")
    assert import_utils.primus_model_display_name("inkling") == "Inkling"
    for entry in import_utils._PRIMUS_MODEL_PROVIDERS.values():
        assert isinstance(entry, import_utils.PrimusModelProvider)


def test_a_missing_builder_names_the_model_type_and_the_table(monkeypatch):
    import types

    from primus.core.utils import import_utils

    fake = types.ModuleType("fake_bad_builders")
    fake.model_provider = lambda *a, **k: None
    monkeypatch.setitem(
        import_utils._PRIMUS_MODEL_PROVIDERS,
        "badm",
        import_utils.PrimusModelProvider("fake_bad_builders", "badm_builder", "BadM"),
    )
    monkeypatch.setattr(import_utils.importlib, "import_module", lambda name, *a, **k: fake)
    with pytest.raises(AttributeError, match="model_type='badm'.*_PRIMUS_MODEL_PROVIDERS.*badm_builder"):
        import_utils.get_model_provider(model_type="badm")
