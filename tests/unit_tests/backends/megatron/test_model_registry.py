###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for the Megatron model-family registry and the shared Primus builder."""

import ast
import inspect
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest

from primus.backends.megatron.core.models import registry
from primus.backends.megatron.core.models.registry import (
    DEFAULT_MODEL_TYPE,
    MODEL_FAMILIES,
    ModelFamily,
    get_model_family,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
PRIMUS_FAMILIES = sorted(name for name, family in MODEL_FAMILIES.items() if not family.builder)
HOOK_FIELDS = ("decoder_spec", "prepare_args", "model_kwargs", "check_model")


def test_every_entry_is_a_literal_so_schema_check_can_read_it():
    tree = ast.parse(Path(registry.__file__).read_text())
    (table,) = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "MODEL_FAMILIES"
    ]
    assert isinstance(table, ast.Dict)
    for key, value in zip(table.keys, table.values):
        assert isinstance(key, ast.Constant) and isinstance(key.value, str)
        assert isinstance(value, ast.Call) and getattr(value.func, "id", None) == "ModelFamily"
        assert not value.args, f"{key.value}: pass ModelFamily fields by keyword"
        for keyword in value.keywords:
            assert isinstance(keyword.value, ast.Constant), f"{key.value}.{keyword.arg} must be a literal"


def test_schema_check_reads_the_registered_config_classes():
    from primus.core.config.schema_check import _registered_config_classes

    default, config_classes, source = _registered_config_classes(REPO_ROOT, "megatron")

    assert default == DEFAULT_MODEL_TYPE
    assert source == "primus/backends/megatron/core/models/registry.py"
    assert config_classes == {
        name: MODEL_FAMILIES[name].config_class.rpartition(":")[2] for name in PRIMUS_FAMILIES
    }


@pytest.mark.parametrize("name", sorted(MODEL_FAMILIES))
def test_family_is_either_upstream_or_fully_primus(name):
    family = MODEL_FAMILIES[name]
    primus_fields = ("package", "config_class", "model_class") + HOOK_FIELDS
    if family.builder:
        assert not any(getattr(family, field) for field in primus_fields)
    else:
        assert family.package and family.config_class and family.model_class


@pytest.mark.parametrize("name", PRIMUS_FAMILIES)
def test_primus_family_references_resolve(name):
    from megatron.core.transformer.transformer_config import TransformerConfig

    family = MODEL_FAMILIES[name]
    assert issubclass(family.resolve(family.config_class), TransformerConfig)
    assert inspect.isclass(family.resolve(family.model_class))
    for field in HOOK_FIELDS:
        if getattr(family, field):
            assert callable(family.resolve(getattr(family, field))), field


@pytest.mark.parametrize("name", PRIMUS_FAMILIES)
def test_family_package_keeps_its_builder_names(name):
    import importlib

    from primus.backends.megatron.core.models.builder import build_model

    builders = importlib.import_module(f"{MODEL_FAMILIES[name].package}.{name}_builders")
    builder = getattr(builders, f"{name}_builder")
    assert (builder.func, builder.args) == (build_model, (name,))
    assert builders.model_provider.default_builder is builder


@pytest.mark.parametrize("name", PRIMUS_FAMILIES)
@pytest.mark.parametrize(
    ("call", "expected_builder"),
    [
        (lambda provider: provider(pre_process=True, post_process=False), "default"),
        (lambda provider: provider(None, True, False), "default"),
        (lambda provider: provider("CUSTOM", True, False), "CUSTOM"),
        (lambda provider: provider(model_builder="CUSTOM", pre_process=True, post_process=False), "CUSTOM"),
    ],
)
def test_family_model_provider_defaults_to_its_own_builder(monkeypatch, name, call, expected_builder):
    import importlib

    from primus.backends.megatron.core.models import builder

    builders = importlib.import_module(f"{MODEL_FAMILIES[name].package}.{name}_builders")
    calls = []

    def fake_model_provider(*a, **k):
        calls.append((a, k))
        return "MODEL"

    monkeypatch.setattr(builder, "model_provider", fake_model_provider)

    assert call(builders.model_provider) == "MODEL"

    family_builder = getattr(builders, f"{name}_builder")
    expected = family_builder if expected_builder == "default" else expected_builder
    assert calls == [((expected, True, False, None), {"config": None, "pg_collection": None})]


@pytest.mark.parametrize("model_type", [None, "", "diffusion_model"])
def test_unregistered_model_type_trains_as_the_default(model_type):
    assert get_model_family(model_type) is MODEL_FAMILIES[DEFAULT_MODEL_TYPE]


# Values the diffusion trainers set, which supply their own model_provider.
UNREGISTERED_MODEL_TYPES = {"diffusion_model"}


def _reads_model_type(node):
    if isinstance(node, ast.Name):
        return node.id == "model_type"
    if isinstance(node, ast.Attribute):
        return node.attr == "model_type"
    # getattr(args, "model_type", None), get_param(ctx, "model_type")
    return (
        isinstance(node, ast.Call)
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == "model_type"
    )


def test_model_type_gates_in_the_megatron_backend_name_registered_families():
    """A patch gated on a misspelt or renamed family silently never applies."""
    unknown = []
    for py in sorted((REPO_ROOT / "primus" / "backends" / "megatron").rglob("*.py")):
        for node in ast.walk(ast.parse(py.read_text())):
            if not (isinstance(node, ast.Compare) and len(node.ops) == 1):
                continue
            if not isinstance(node.ops[0], (ast.Eq, ast.NotEq)):
                continue
            left, right = node.left, node.comparators[0]
            for read, value in ((left, right), (right, left)):
                if not (_reads_model_type(read) and isinstance(value, ast.Constant)):
                    continue
                if not isinstance(value.value, str):
                    continue
                if value.value not in MODEL_FAMILIES and value.value not in UNREGISTERED_MODEL_TYPES:
                    unknown.append(f"{py.relative_to(REPO_ROOT)}:{node.lineno}: {value.value!r}")
    assert not unknown, "model_type compared against unregistered values:\n" + "\n".join(unknown)


# ---------------------------------------------------------------------------
# get_model_provider
# ---------------------------------------------------------------------------


@pytest.fixture
def quiet(monkeypatch):
    from primus.core.utils import import_utils

    monkeypatch.setattr(import_utils, "log_rank_0", lambda *a, **k: None)


@pytest.mark.parametrize("name", PRIMUS_FAMILIES)
def test_primus_family_gets_the_shared_provider(quiet, name):
    from primus.backends.megatron.core.models.builder import build_model, model_provider
    from primus.core.utils.import_utils import get_model_provider

    provider = get_model_provider(model_type=name)

    assert provider.func is model_provider
    (builder,) = provider.args
    assert (builder.func, builder.args) == (build_model, (name,))


@pytest.mark.parametrize("model_type", ["not_a_family", "kimi-k3", "diffusion_model"])
def test_get_model_provider_rejects_an_unregistered_model_type(quiet, model_type):
    from primus.core.utils.import_utils import get_model_provider

    with pytest.raises(ValueError, match=rf"{model_type!r} is not registered.*registered: deepseek_v4, "):
        get_model_provider(model_type=model_type)


@pytest.mark.parametrize(
    ("model_type", "entrypoint", "builder_module", "builder_name"),
    [
        ("gpt", "pretrain_gpt", "gpt_builders", "gpt_builder"),
        ("mamba", "pretrain_mamba", "mamba_builders", "mamba_builder"),
        (None, "pretrain_gpt", "gpt_builders", "gpt_builder"),
    ],
)
def test_upstream_family_resolves_across_megatron_versions(
    quiet, monkeypatch, model_type, entrypoint, builder_module, builder_name
):
    from primus.core.utils import import_utils

    calls = []
    resolved = {}

    def fake_lazy_import(paths, symbol, log_prefix=""):
        calls.append((tuple(paths), symbol))
        return resolved.setdefault(symbol, lambda *a, **k: None)

    monkeypatch.setattr(import_utils, "lazy_import", fake_lazy_import)

    provider = import_utils.get_model_provider(model_type=model_type)

    assert calls == [(("model_provider", entrypoint), "model_provider"), ((builder_module,), builder_name)]
    assert provider.func is resolved["model_provider"]
    assert provider.args == (resolved[builder_name],)


def test_upstream_family_without_a_builder_module_returns_the_bare_provider(quiet, monkeypatch):
    from primus.core.utils import import_utils

    def bare_provider(*a, **k):
        return None

    def fake_lazy_import(paths, symbol, log_prefix=""):
        if symbol == "model_provider":
            return bare_provider
        raise ImportError(symbol)

    monkeypatch.setattr(import_utils, "lazy_import", fake_lazy_import)

    assert import_utils.get_model_provider(model_type="gpt") is bare_provider


# ---------------------------------------------------------------------------
# build_model
# ---------------------------------------------------------------------------


@dataclass
class FakeConfig:
    tag: str = "fake"


class FakeModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


_events = []


def fake_prepare_args(args):
    _events.append("prepare_args")
    args.prepared = True


def fake_decoder_spec(config, vp_stage=None):
    _events.append(("decoder_spec", config.tag, vp_stage))
    return "FAMILY_SPEC"


def fake_model_kwargs(args):
    return {"position_embedding_type": args.position_embedding_type}


def fake_check_model(args, config, model):
    _events.append(("check_model", type(model).__name__))


def _fake_family(**overrides):
    fields = dict(
        config_class=f"{__name__}:FakeConfig",
        model_class=f"{__name__}:FakeModel",
        decoder_spec=f"{__name__}:fake_decoder_spec",
        prepare_args=f"{__name__}:fake_prepare_args",
        model_kwargs=f"{__name__}:fake_model_kwargs",
        check_model=f"{__name__}:fake_check_model",
    )
    fields.update(overrides)
    return ModelFamily(**fields)


def _args(**overrides):
    args = SimpleNamespace(
        use_legacy_models=False,
        spec=None,
        padded_vocab_size=128,
        max_position_embeddings=64,
        fp16_lm_cross_entropy=False,
        untie_embeddings_and_output_weights=True,
        position_embedding_type="rope",
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


@pytest.fixture
def builder(monkeypatch):
    from primus.backends.megatron.core.models import builder

    _events.clear()
    monkeypatch.setattr(builder, "print_rank_0", lambda *a, **k: None)

    def fake_config_from_args(args, config_class):
        _events.append(("config", getattr(args, "prepared", False)))
        return config_class()

    monkeypatch.setattr(builder, "core_transformer_config_from_args", fake_config_from_args)
    monkeypatch.setitem(MODEL_FAMILIES, "fake", _fake_family())
    return builder


def test_build_model_runs_the_family_hooks_in_order(builder):
    model = builder.build_model("fake", _args(), True, False, vp_stage=1, pg_collection="PG")

    assert _events == [
        "prepare_args",
        ("config", True),
        ("decoder_spec", "fake", 1),
        ("check_model", "FakeModel"),
    ]
    config = model.kwargs.pop("config")
    assert isinstance(config, FakeConfig)
    assert model.kwargs == dict(
        transformer_layer_spec="FAMILY_SPEC",
        vocab_size=128,
        max_sequence_length=64,
        pre_process=True,
        post_process=False,
        fp16_lm_cross_entropy=False,
        parallel_output=True,
        share_embeddings_and_output_weights=False,
        pg_collection="PG",
        vp_stage=1,
        position_embedding_type="rope",
    )


def test_build_model_honours_the_spec_override(builder, monkeypatch):
    monkeypatch.setattr(builder, "import_module", lambda spec: f"imported:{spec}")

    model = builder.build_model("fake", _args(spec="pkg.mod.spec"), True, True)

    assert model.kwargs["transformer_layer_spec"] == "imported:pkg.mod.spec"
    assert not any(isinstance(event, tuple) and event[0] == "decoder_spec" for event in _events)


def test_build_model_without_hooks_passes_only_the_common_arguments(builder, monkeypatch):
    monkeypatch.setitem(
        MODEL_FAMILIES,
        "fake",
        _fake_family(decoder_spec="", prepare_args="", model_kwargs="", check_model=""),
    )

    model = builder.build_model("fake", _args(spec="ignored.without.a.decoder_spec"), True, True)

    assert "transformer_layer_spec" not in model.kwargs
    assert "position_embedding_type" not in model.kwargs
    assert _events == [("config", False)]


def test_build_model_uses_a_given_config(builder):
    config = FakeConfig(tag="given")

    model = builder.build_model("fake", _args(), True, True, config=config)

    assert model.kwargs["config"] is config
    assert _events == ["prepare_args", ("decoder_spec", "given", None), ("check_model", "FakeModel")]


def test_build_model_rejects_a_swapped_config_class(builder):
    with pytest.raises(AssertionError, match="Expected a FakeConfig.*multi_latent_attention"):
        builder.build_model("fake", _args(), True, True, config=object())


def test_build_model_rejects_legacy_models(builder):
    with pytest.raises(AssertionError, match="use_legacy_models=False"):
        builder.build_model("fake", _args(use_legacy_models=True), True, True)


def test_build_model_refuses_upstream_families(builder):
    with pytest.raises(AssertionError, match="gpt_builders:gpt_builder"):
        builder.build_model("gpt", _args(), True, True)


def test_model_provider_forwards_to_the_bound_builder(builder, monkeypatch):
    args = _args(record_memory_history=False)
    monkeypatch.setattr(builder, "get_args", lambda: args)
    calls = []

    def fake_builder(*a, **k):
        calls.append((a, k))
        return "MODEL"

    provider = partial(builder.model_provider, fake_builder)

    assert provider(pre_process=False, post_process=True, vp_stage=2, config="CFG") == "MODEL"
    assert calls == [((args, False, True, 2), {"config": "CFG", "pg_collection": None})]


def test_model_provider_records_memory_history_and_snapshots_on_oom(builder, monkeypatch):
    import torch

    args = _args(record_memory_history=True, memory_snapshot_path="snapshot.pickle")
    monkeypatch.setattr(builder, "get_args", lambda: args)
    recorded, observers, dumped = [], [], []
    monkeypatch.setattr(torch.cuda.memory, "_record_memory_history", lambda *a, **k: recorded.append(k))
    monkeypatch.setattr(torch.cuda.memory, "_dump_snapshot", dumped.append)
    monkeypatch.setattr(torch._C, "_cuda_attach_out_of_memory_observer", observers.append, raising=False)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 3)

    builder.model_provider(lambda *a, **k: "MODEL")

    assert recorded == [dict(trace_alloc_max_entries=100000, trace_alloc_record_context=True)]
    (observer,) = observers
    observer(0, 1, 2, 3)
    assert dumped == ["oom_rank-3_snapshot.pickle"]
