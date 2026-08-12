###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the backend config schema drift scanner.

Most tests run against a synthetic repo built in ``tmp_path`` so they neither
need the ``third_party/`` submodules nor break when upstream moves a field.
A few tests do touch the real tree, because they guard the failure modes that
motivated the scanner:

* ``training.mock_data`` must be derived automatically from the patch system
  (it is a Primus-only field; a previous audit wrongly called it dead code);
* ``primus_turbo`` must be derived from ``experimental.custom_args_module``;
* ``train_iters`` and friends must come out of Megatron's generated arguments,
  because reading its literal ``add_argument`` calls alone would report roughly
  half of every valid Megatron config as drift;
* ``norm_epsilon`` must come out as scoped to DeepSeek-V4, and no DeepSeek-V4
  config may be reported for setting it.
"""

from __future__ import annotations

import importlib.util
import textwrap
from pathlib import Path

import pytest

from primus.core.config.schema_check import (
    Allowlist,
    AllowRule,
    Finding,
    KeyFinding,
    ScopedFinding,
    build_allowlist,
    build_model_scopes,
    build_schema,
    check_manual_allowlist_consumers,
    dedupe,
    dedupe_scoped,
    extract_args_read_rules,
    extract_backend_config_extensions,
    extract_custom_args_schema,
    extract_get_param_rules,
    load_manual_allowlist,
    load_maxtext_schema,
    load_megatron_schema,
    load_torchtitan_schema,
    scan_backend,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]

_JOB_CONFIG_SRC = """
# Upstream imports torch at module scope, which is why the scanner must use AST.
import torch
from dataclasses import dataclass, field


@dataclass
class Training:
    seq_len: int = 2048
    steps: int = 10000


@dataclass
class Debug:
    seed: int | None = None
    moe_force_load_balance: bool = False


@dataclass
class Float8Linear:
    filter_fqns: list[str] = field(default_factory=list)


@dataclass
class QuantizedLinear:
    float8: Float8Linear = field(default_factory=Float8Linear)


@dataclass
class Quantize:
    linear: QuantizedLinear = field(default_factory=QuantizedLinear)


class NotADataclass:
    ignored: int = 0


@dataclass
class Experimental:
    custom_args_module: str = ""


@dataclass
class JobConfig:
    training: Training = field(default_factory=Training)
    debug: Debug = field(default_factory=Debug)
    quantize: Quantize = field(default_factory=Quantize)
    experimental: Experimental = field(default_factory=Experimental)
"""

_CONFIG_EXTENSION_SRC = """
from dataclasses import dataclass, field

from torchtitan.config.job_config import JobConfig as TTJobConfig


@dataclass
class PrimusTurboConfig:
    enable_primus_turbo: bool = False
    use_turbo_attention: bool = False


@dataclass
class JobConfig(TTJobConfig):
    primus_turbo: PrimusTurboConfig = field(default_factory=PrimusTurboConfig)
"""

_PATCH_SRC = """
from primus.core.patches import get_param, register_patch


@register_patch(patch_id="fake", backend="torchtitan", phase="setup")
def fake_patch(ctx):
    if get_param(ctx, "training.mock_data", False):
        return get_param(ctx, "primus_turbo.enable_primus_turbo", False)
    return None
"""

_ALLOWLIST_SRC = """
common:
  - key: work_group
    reason: Primus envelope.
    consumer: primus/core/launcher/parser.py:211
  - key: user_name
    reason: Primus envelope.
    consumer: primus/core/launcher/parser.py:212
  - key: exp_name
    reason: Primus envelope.
    consumer: primus/core/launcher/parser.py:213
torchtitan:
  - key: model.*
    reason: Model-arg override, validated at runtime.
    consumer: primus/backends/torchtitan/patches/model_override_patches.py:273
"""


_MEGATRON_ARGUMENTS_SRC = """
# Upstream imports torch at module scope, which is why the scanner must use AST.
import torch

from megatron.core.transformer import MLATransformerConfig, TransformerConfig
from megatron.training.argument_utils import ArgumentGroupFactory


def add_megatron_arguments(parser):
    group = parser.add_argument_group(title="core")
    group.add_argument("--seq-length", type=int, default=None)
    group.add_argument("--num-experts", type=int, default=None)
    # An explicit dest is the only thing that keeps this off `no_persist_layer_norm`.
    group.add_argument("--no-persist-layer-norm", action="store_false", dest="persist_layer_norm")

    # Upstream builds this list in code rather than inline.
    exclude = [
        "moe_token_dropping",
        "gradient_accumulation_fusion",
    ]
    ArgumentGroupFactory(TransformerConfig, exclude=exclude).build_group(parser, "transformer")

    from megatron.training.config import LoggerConfig, TrainingConfig

    ArgumentGroupFactory(TrainingConfig).build_group(parser, "training")
    ArgumentGroupFactory(LoggerConfig, exclude=["memory_keys"]).build_group(parser, "logging")
    return parser


def core_transformer_config_from_args(args, config_class=None):
    config_class = config_class or TransformerConfig
    if args.multi_latent_attention:
        config_class = MLATransformerConfig
    return config_class(**{f.name: getattr(args, f.name) for f in dataclasses.fields(config_class)})
"""

_MEGATRON_TRAINING_CONFIG_SRC = """
from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    train_iters: int | None = None
    global_batch_size: int | None = None
    micro_batch_size: int = 1
    # Derived after parsing, so ArgumentGroupFactory skips it.
    computed_batches: int | None = field(init=False, default=None)


@dataclass
class LoggerConfig:
    log_interval: int = 100
    memory_keys: dict | None = None
    use_nsys_profiler: bool = field(
        default=False, metadata={"argparse_meta": {"arg_names": ["--profile"], "dest": "profile"}}
    )
"""

_MEGATRON_TRANSFORMER_CONFIG_SRC = """
from dataclasses import dataclass

from ..model_parallel_config import ModelParallelConfig


@dataclass
class TransformerConfig(ModelParallelConfig):
    num_layers: int = 0
    moe_token_dropping: bool = False
    gradient_accumulation_fusion: bool = True


@dataclass
class MLATransformerConfig(TransformerConfig):
    q_lora_rank: int | None = None
"""

_MEGATRON_PATCH_SRC = """
import os

from primus.core.patches import get_args, register_patch

_PARAM_NAME = "dataloader_mp_context"


@register_patch(patch_id="fake", backend="megatron", phase="setup")
def fake_patch(ctx, tensor):
    # An attribute of something that is not an args object must stay out.
    _ = tensor.shape
    os.environ["HSA_NO_SCRATCH_RECLAIM"] = "1"
    if getattr(get_args(ctx), "use_turbo_attention", False):
        return get_args(ctx).log_avg_skip_iterations
    return getattr(get_args(ctx), _PARAM_NAME, None)
"""

_MEGATRON_CONFIG_EXTENSION_SRC = """
from dataclasses import dataclass

from megatron.core.transformer.transformer_config import MLATransformerConfig


@dataclass
class PrimusBaseConfig(MLATransformerConfig):
    norm_epsilon: float | None = None


@dataclass
class PrimusChildConfig(PrimusBaseConfig):
    conv_bias: bool = False
"""

# The config class of one model, holding one field of each kind the scope
# derivation has to tell apart.
_MEGATRON_SCOPED_CONFIG_SRC = """
from dataclasses import dataclass

from megatron.core.transformer.transformer_config import MLATransformerConfig


@dataclass
class FakeV4TransformerConfig(MLATransformerConfig):
    hash_routing_seed: int = 0
    attn_sink: bool = False
    # Upstream defines this one, so every model gets it from `args`.
    num_layers: int = 0
    # A config class no model selects declares this one too.
    conv_bias: bool = False
    # A patch reads this one off `args`, which every model has.
    index_topk: int = 0
    # The hand-written allowlist vouches for this one everywhere.
    swiglu_limit: float = 0.0
"""

_MEGATRON_SCOPED_BUILDER_SRC = """
from megatron.training.arguments import core_transformer_config_from_args

from .fake_v4_config import FakeV4TransformerConfig


def fake_v4_builder(args, pre_process, post_process, config=None):
    if config is None:
        config = core_transformer_config_from_args(args, config_class=FakeV4TransformerConfig)
    return config
"""

# Mirrors the real dispatch: the branch that names a Primus module outright is
# the one that leads to a Primus config class; the others reach upstream.
_MODEL_PROVIDER_SRC = """
import importlib
from functools import partial


def get_model_provider(model_type="gpt"):
    if model_type == "fake_v4":
        module = importlib.import_module(
            "primus.backends.megatron.core.models.fake_v4.fake_v4_builders"
        )
        return partial(module.model_provider, module.fake_v4_builder)
    if model_type == "mamba":
        return lazy_import(["model_provider", "pretrain_mamba"], "model_provider")
    return lazy_import(["model_provider", "pretrain_gpt"], "model_provider")
"""

_MEGATRON_SCOPE_PATCH_SRC = """
from primus.core.patches import get_args, register_patch


@register_patch(patch_id="fake_flops", backend="megatron", phase="setup")
def fake_flops_patch(ctx):
    if get_args(ctx).model_type == "fake_v4":
        # A read off the args namespace, which every model has: not model-scoped.
        return get_args(ctx).index_topk
    return None
"""


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).lstrip())
    return path


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    """A minimal repo with both submodules, a patch, and an extension dataclass."""
    _write(tmp_path / "third_party/torchtitan/torchtitan/config/job_config.py", _JOB_CONFIG_SRC)
    _write(
        tmp_path / "third_party/maxtext/src/maxtext/configs/base.yml",
        "run_name: ''\nsteps: 10\nper_device_batch_size: 1\n",
    )
    _write(
        tmp_path / "primus/backends/torchtitan/primus_turbo_extensions/config_extension.py",
        _CONFIG_EXTENSION_SRC,
    )
    _write(tmp_path / "primus/backends/torchtitan/patches/fake_patches.py", _PATCH_SRC)
    _write(
        tmp_path / "primus/configs/modules/torchtitan/pre_trainer.yaml",
        """
        experimental:
          custom_args_module: "primus.backends.torchtitan.primus_turbo_extensions.config_extension"
        training:
          mock_data: true
        """,
    )
    _write(tmp_path / "tools/ci/config_schema_allowlist.yaml", _ALLOWLIST_SRC)
    return tmp_path


@pytest.fixture
def megatron_repo(tmp_path: Path) -> Path:
    """A minimal repo whose Megatron mirrors the parts the scanner has to read.

    Namely: config dataclasses reached through a re-export, a base class in
    another package, an ``exclude`` list built in code, an ``init=False`` field
    and an ``argparse_meta`` dest override.
    """
    megatron = tmp_path / "third_party/Megatron-LM/megatron"
    _write(megatron / "training/arguments.py", _MEGATRON_ARGUMENTS_SRC)
    _write(
        megatron / "training/config/__init__.py",
        "from megatron.training.config.training_config import LoggerConfig, TrainingConfig\n",
    )
    _write(megatron / "training/config/training_config.py", _MEGATRON_TRAINING_CONFIG_SRC)
    _write(
        megatron / "core/transformer/__init__.py",
        "from .transformer_config import MLATransformerConfig, TransformerConfig\n",
    )
    _write(megatron / "core/transformer/transformer_config.py", _MEGATRON_TRANSFORMER_CONFIG_SRC)
    _write(
        megatron / "core/model_parallel_config.py",
        "from dataclasses import dataclass\n\n\n"
        "@dataclass\nclass ModelParallelConfig:\n"
        "    tensor_model_parallel_size: int = 1\n    timers: object = None\n",
    )
    _write(tmp_path / "primus/backends/megatron/patches/fake_patches.py", _MEGATRON_PATCH_SRC)
    _write(tmp_path / "primus/backends/megatron/patches/fake_flops_patches.py", _MEGATRON_SCOPE_PATCH_SRC)
    _write(
        tmp_path / "primus/backends/megatron/core/models/fake_config.py",
        _MEGATRON_CONFIG_EXTENSION_SRC,
    )
    _write(
        tmp_path / "primus/backends/megatron/core/models/fake_v4/fake_v4_config.py",
        _MEGATRON_SCOPED_CONFIG_SRC,
    )
    _write(
        tmp_path / "primus/backends/megatron/core/models/fake_v4/fake_v4_builders.py",
        _MEGATRON_SCOPED_BUILDER_SRC,
    )
    _write(tmp_path / "primus/core/utils/import_utils.py", _MODEL_PROVIDER_SRC)
    _write(tmp_path / "primus/configs/models/megatron/fake_v4.yaml", "model_type: fake_v4\nnum_layers: 2\n")
    _write(tmp_path / "primus/configs/models/megatron/plain.yaml", "num_layers: 2\n")
    _write(
        tmp_path / "tools/ci/config_schema_allowlist.yaml",
        "common:\n"
        "  - key: work_group\n"
        "    reason: Primus envelope.\n"
        "    consumer: primus/core/launcher/parser.py\n"
        "megatron:\n"
        "  - key: swiglu_limit\n"
        "    reason: Read by every MoE model, not just this one.\n"
        "    consumer: primus/core/launcher/parser.py\n",
    )
    _write(tmp_path / "primus/core/launcher/parser.py", "cfg.work_group, cfg.swiglu_limit\n")
    return tmp_path


# ---------------------------------------------------------------------------
# Schema extraction
# ---------------------------------------------------------------------------


def test_torchtitan_schema_extracted_from_ast(fake_repo: Path):
    schema = load_torchtitan_schema(fake_repo)

    assert schema is not None
    assert schema.sections == {
        "training",
        "debug",
        "quantize",
        "quantize.linear",
        "quantize.linear.float8",
        "experimental",
    }
    assert schema.keys == {
        "training.seq_len",
        "training.steps",
        "debug.seed",
        "debug.moe_force_load_balance",
        "quantize.linear.float8.filter_fqns",
        "experimental.custom_args_module",
    }
    # A plain class must not leak into the schema.
    assert not any(k.startswith("ignored") for k in schema.keys)


def test_maxtext_schema_is_flat(fake_repo: Path):
    schema = load_maxtext_schema(fake_repo)

    assert schema is not None
    assert schema.keys == {"run_name", "steps", "per_device_batch_size"}
    assert schema.sections == frozenset()


def test_schema_suggests_where_a_moved_key_went(fake_repo: Path):
    schema = load_torchtitan_schema(fake_repo)

    assert schema.suggest("training.seed") == "debug.seed"
    # Upstream drops the section-name prefix when it moves a field into a section.
    assert schema.suggest("training.debug_moe_force_load_balance") == "debug.moe_force_load_balance"
    assert schema.suggest("training.no_such_thing_anywhere") is None


def test_missing_submodule_yields_no_schema(tmp_path: Path):
    assert load_torchtitan_schema(tmp_path) is None
    assert load_maxtext_schema(tmp_path) is None
    assert load_megatron_schema(tmp_path) is None


# ---------------------------------------------------------------------------
# Megatron schema extraction
# ---------------------------------------------------------------------------


def test_megatron_schema_joins_literal_and_generated_arguments(megatron_repo: Path):
    schema = load_megatron_schema(megatron_repo)

    assert schema is not None
    assert schema.sections == frozenset()  # Megatron's namespace is flat.
    # The literal `add_argument` half.
    assert {"seq_length", "num_experts"} <= schema.keys
    # The ArgumentGroupFactory half, which is where the most-used names live.
    assert {"train_iters", "global_batch_size", "micro_batch_size", "log_interval"} <= schema.keys
    # A field's base class contributes too: TransformerConfig(ModelParallelConfig).
    assert {"tensor_model_parallel_size", "timers"} <= schema.keys


def test_megatron_factory_exclude_list_is_honoured(megatron_repo: Path):
    """`memory_keys` is excluded from LoggerConfig, so no argument carries it."""
    schema = load_megatron_schema(megatron_repo)

    assert "log_interval" in schema.keys
    assert "memory_keys" not in schema.keys


def test_megatron_transformer_config_keeps_its_excluded_fields(megatron_repo: Path):
    """`core_transformer_config_from_args` copies every field off `args`.

    So a TransformerConfig field stays a legal key even where the factory
    excludes it from argparse -- unlike LoggerConfig, which nothing builds from
    `args`.
    """
    schema = load_megatron_schema(megatron_repo)

    assert {"moe_token_dropping", "gradient_accumulation_fusion"} <= schema.keys
    # MLATransformerConfig is the other class that function can pick.
    assert "q_lora_rank" in schema.keys


def test_megatron_derived_fields_are_not_arguments(megatron_repo: Path):
    """`field(init=False)` is computed after parsing; `build_group` skips it."""
    assert "computed_batches" not in load_megatron_schema(megatron_repo).keys


def test_megatron_argparse_meta_dest_wins_over_the_field_name(megatron_repo: Path):
    """`use_nsys_profiler` reaches `args` as `profile`, so only `profile` is legal."""
    schema = load_megatron_schema(megatron_repo)

    assert "profile" in schema.keys
    assert "use_nsys_profiler" not in schema.keys


def test_megatron_explicit_dest_wins_over_the_option_name(megatron_repo: Path):
    """`--no-persist-layer-norm dest=persist_layer_norm` must not yield both."""
    schema = load_megatron_schema(megatron_repo)

    assert "persist_layer_norm" in schema.keys
    assert "no_persist_layer_norm" not in schema.keys


def test_megatron_suggests_a_renamed_argument(megatron_repo: Path):
    """A flat namespace renames in place, by adding a prefix or a suffix."""
    schema = load_megatron_schema(megatron_repo)

    assert schema.suggest("layers") == "num_layers"
    assert schema.suggest("nothing_like_this") is None


# ---------------------------------------------------------------------------
# Allowlist derivation
# ---------------------------------------------------------------------------


def test_get_param_paths_are_derived_from_patches(fake_repo: Path):
    rules = extract_get_param_rules(fake_repo, "torchtitan")

    patterns = {r.pattern for r in rules}
    assert patterns == {"training.mock_data", "primus_turbo.enable_primus_turbo"}
    assert all(r.consumer.startswith("primus/backends/torchtitan/patches/") for r in rules)


def test_custom_args_module_is_followed(fake_repo: Path):
    keys, sections, module_path = extract_custom_args_schema(fake_repo)

    assert module_path.endswith("config_extension")
    assert sections == {"primus_turbo"}
    assert keys == {"primus_turbo.enable_primus_turbo", "primus_turbo.use_turbo_attention"}


def test_extension_section_stays_checked(fake_repo: Path):
    """`primus_turbo` is a section, not an opaque wildcard: bad children still fail."""
    schema = build_schema(fake_repo, "torchtitan")

    assert "primus_turbo" in schema.sections
    assert "primus_turbo.use_turbo_attention" in schema.keys
    assert "primus_turbo.nonexistent" not in schema.keys


def test_megatron_args_reads_are_derived_from_the_backend_package(megatron_repo: Path):
    rules = extract_args_read_rules(megatron_repo, "megatron")
    patterns = {r.pattern for r in rules}

    assert "use_turbo_attention" in patterns  # getattr(args, "<name>", default)
    assert "log_avg_skip_iterations" in patterns  # get_args(ctx).<name>
    assert "dataloader_mp_context" in patterns  # read through a module constant
    # An attribute of something that is not an args object is not a config key.
    assert "shape" not in patterns
    # Neither is an environment variable, so a config that sets one still fails.
    assert "HSA_NO_SCRATCH_RECLAIM" not in patterns
    assert all(r.consumer.startswith("primus/backends/megatron/") for r in rules)


def test_megatron_config_extension_fields_are_schema(megatron_repo: Path):
    """A Primus TransformerConfig subclass widens the key set, transitively."""
    keys, source = extract_backend_config_extensions(megatron_repo, "megatron")

    assert source == "primus/backends/megatron"
    assert {"norm_epsilon", "conv_bias"} <= keys
    schema = build_schema(megatron_repo, "megatron")
    assert {"norm_epsilon", "conv_bias"} <= schema.keys


# ---------------------------------------------------------------------------
# Model scopes
# ---------------------------------------------------------------------------


def test_model_scope_is_read_off_the_model_dispatch(megatron_repo: Path):
    """`get_model_provider` is what decides the model, so it is what we follow."""
    scopes = build_model_scopes(megatron_repo, "megatron")

    assert len(scopes) == 1
    scope = scopes[0]
    assert (scope.key, scope.value, scope.default) == ("model_type", "fake_v4", "gpt")
    assert scope.config_class == "FakeV4TransformerConfig"
    assert scope.source == "primus/core/utils/import_utils.py"


def test_only_a_field_nothing_else_explains_is_model_scoped(megatron_repo: Path):
    """Four kinds of field share the scoped class and none of them is scoped."""
    scope = build_model_scopes(megatron_repo, "megatron")[0]

    assert scope.keys == {"hash_routing_seed", "attn_sink"}
    assert "num_layers" not in scope.keys  # upstream defines it for every model
    assert "conv_bias" not in scope.keys  # a config class no model selects has it too
    assert "index_topk" not in scope.keys  # a patch reads it off `args`
    assert "swiglu_limit" not in scope.keys  # the hand-written allowlist vouches for it


def test_no_model_dispatch_means_no_scopes(fake_repo: Path):
    """TorchTitan has no such dispatch; the check must simply not look for one."""
    assert build_model_scopes(fake_repo, "torchtitan") == ()


def test_a_scoped_key_on_another_model_is_reported(megatron_repo: Path):
    _write(
        megatron_repo / "examples/megatron/configs/other_model.yaml",
        """
        work_group: amd
        modules:
          pre_trainer:
            framework: megatron
            config: pre_trainer.yaml
            model: plain.yaml
            overrides:
              train_iters: 50
              hash_routing_seed: 7
        """,
    )

    result = scan_backend(megatron_repo, "megatron")

    assert result.findings == ()  # the key is not unknown, only misplaced
    misplaced = [f for f in result.scoped if f.file.endswith("other_model.yaml")]
    assert [(f.key, f.actual) for f in misplaced] == [("hash_routing_seed", "gpt")]
    assert misplaced[0].scope.describe() == "model_type: fake_v4"


def test_a_scoped_key_on_its_own_model_is_not_reported(megatron_repo: Path):
    """The `model_type` lives in the model preset, not in the experiment YAML."""
    _write(
        megatron_repo / "examples/megatron/configs/right_model.yaml",
        """
        work_group: amd
        modules:
          pre_trainer:
            framework: megatron
            config: pre_trainer.yaml
            model: fake_v4.yaml
            overrides:
              hash_routing_seed: 7
        """,
    )

    result = scan_backend(megatron_repo, "megatron")

    assert result.scoped == ()


def test_an_experiment_may_override_the_model_type_itself(megatron_repo: Path):
    _write(
        megatron_repo / "examples/megatron/configs/override.yaml",
        """
        work_group: amd
        modules:
          pre_trainer:
            framework: megatron
            config: pre_trainer.yaml
            model: plain.yaml
            overrides:
              model_type: fake_v4
              hash_routing_seed: 7
        """,
    )

    assert scan_backend(megatron_repo, "megatron").scoped == ()


def test_an_unreadable_model_preset_reports_nothing(megatron_repo: Path):
    """Guessing here would report the one config where the key belongs."""
    _write(
        megatron_repo / "examples/megatron/configs/unknown_model.yaml",
        """
        work_group: amd
        modules:
          pre_trainer:
            framework: megatron
            config: pre_trainer.yaml
            model: not_a_preset.yaml
            overrides:
              hash_routing_seed: 7
        """,
    )

    assert scan_backend(megatron_repo, "megatron").scoped == ()


def test_a_model_preset_of_its_own_is_judged_on_what_it_declares(megatron_repo: Path):
    """A preset that omits `model_type` gets the dispatch default, which is `gpt`."""
    _write(megatron_repo / "primus/configs/models/megatron/stray.yaml", "attn_sink: true\n")
    _write(
        megatron_repo / "primus/configs/models/megatron/scoped.yaml", "model_type: fake_v4\nattn_sink: true\n"
    )

    result = scan_backend(megatron_repo, "megatron")

    assert [(f.file, f.key, f.actual) for f in result.scoped] == [
        ("primus/configs/models/megatron/stray.yaml", "attn_sink", "gpt")
    ]


def test_dedupe_scoped_collapses_one_key_across_files(megatron_repo: Path):
    scope = build_model_scopes(megatron_repo, "megatron")[0]
    findings = [ScopedFinding("megatron", "attn_sink", f"cfg{i}.yaml", scope, "gpt") for i in range(4)]
    findings.append(ScopedFinding("megatron", "attn_sink", "cfg4.yaml", scope, "mamba"))

    rows = dedupe_scoped(findings)

    assert len(rows) == 1
    assert (rows[0].key, rows[0].count, rows[0].models) == ("attn_sink", 5, ("gpt", "mamba"))
    assert rows[0].sample(2) == "`cfg0.yaml`, `cfg1.yaml` (+3 more)"


def test_manual_allowlist_requires_reason_and_consumer(tmp_path: Path):
    path = _write(tmp_path / "allow.yaml", "common:\n  - key: foo\n    reason: because\n")

    with pytest.raises(ValueError, match="consumer"):
        load_manual_allowlist(path, "torchtitan")


def test_manual_allowlist_scopes_are_merged(fake_repo: Path):
    allow = build_allowlist(fake_repo, "torchtitan")

    assert allow.match("work_group") is not None
    assert allow.match("model.n_layers").pattern == "model.*"
    assert allow.match("training.mock_data").origin == "patch:get_param"
    assert allow.match("nope") is None


def test_allowlist_entry_is_flagged_once_its_consumer_stops_reading_it(tmp_path: Path):
    _write(tmp_path / "primus/core/launcher/parser.py", "cfg.work_group = x\n")
    path = _write(
        tmp_path / "allow.yaml",
        "common:\n"
        "  - key: work_group\n"
        "    reason: still read\n"
        "    consumer: primus/core/launcher/parser.py\n"
        "  - key: removed_key\n"
        "    reason: stale\n"
        "    consumer: primus/core/launcher/parser.py\n",
    )

    problems = check_manual_allowlist_consumers(tmp_path, path, "torchtitan")

    assert len(problems) == 1
    assert "removed_key" in problems[0]


def test_allowlist_entry_is_flagged_once_its_consumer_file_is_gone(tmp_path: Path):
    path = _write(
        tmp_path / "allow.yaml",
        "common:\n  - key: foo\n    reason: r\n    consumer: primus/core/deleted.py\n",
    )

    problems = check_manual_allowlist_consumers(tmp_path, path, "torchtitan")

    assert len(problems) == 1 and "does not exist" in problems[0]


def test_a_line_number_left_on_a_consumer_does_not_break_the_check(tmp_path: Path):
    """The path is what gets verified, so a legacy `path:line` must still pass."""
    _write(tmp_path / "primus/core/launcher/parser.py", "cfg.work_group = x\n")
    path = _write(
        tmp_path / "allow.yaml",
        "common:\n"
        "  - key: work_group\n"
        "    reason: r\n"
        "    consumer: primus/core/launcher/parser.py:211\n",
    )

    assert check_manual_allowlist_consumers(tmp_path, path, "torchtitan") == []


def test_an_unchecked_out_submodule_is_not_reported_as_a_stale_consumer(tmp_path: Path):
    """Absence of a submodule proves nothing about the entry, so stay quiet."""
    path = _write(
        tmp_path / "allow.yaml",
        "maxtext:\n  - key: base_config\n    reason: r\n    consumer: third_party/maxtext/x.py\n",
    )

    assert check_manual_allowlist_consumers(tmp_path, path, "maxtext") == []


def test_the_shipped_allowlist_entries_are_all_still_backed():
    for backend in ("torchtitan", "maxtext", "megatron"):
        problems = check_manual_allowlist_consumers(
            _REPO_ROOT, _REPO_ROOT / "tools/ci/config_schema_allowlist.yaml", backend
        )
        assert problems == []


def test_real_repo_derives_primus_only_fields():
    """Regression guard: these must never be reported as drift."""
    allow = build_allowlist(_REPO_ROOT, "torchtitan")
    schema = build_schema(_REPO_ROOT, "torchtitan")
    if schema is None:
        pytest.skip("third_party/torchtitan is not checked out")

    mock_data = allow.match("training.mock_data")
    assert mock_data is not None and mock_data.origin == "patch:get_param"
    assert "primus_turbo" in schema.sections
    assert "primus_turbo.enable_primus_turbo" in schema.keys


def test_real_repo_megatron_schema_holds_the_generated_arguments():
    """Guard the half of Megatron's namespace that no `add_argument` call names."""
    schema = build_schema(_REPO_ROOT, "megatron")
    if schema is None:
        pytest.skip("third_party/Megatron-LM is not checked out")

    assert {"train_iters", "global_batch_size", "micro_batch_size", "log_interval"} <= schema.keys
    assert {"seq_length", "lr", "num_experts"} <= schema.keys  # the literal half
    assert len(schema.keys) > 600


def test_real_repo_derives_primus_only_megatron_fields():
    """Regression guard: these must never be reported as drift."""
    allow = build_allowlist(_REPO_ROOT, "megatron")

    for key in ("use_turbo_attention", "log_avg_skip_iterations", "enable_zero_bubble", "odc_gda_pipe"):
        rule = allow.match(key)
        assert rule is not None and rule.origin.startswith("backend:"), key


def test_real_repo_scopes_norm_epsilon_to_deepseek_v4():
    """The case this feature exists for.

    `norm_epsilon` is not a Megatron argparse dest -- `layernorm_epsilon` is, and
    it merely declares `--norm-epsilon` as its flag. Only
    `DeepSeekV4TransformerConfig` gives the yaml key a field to land in.
    """
    scopes = build_model_scopes(_REPO_ROOT, "megatron")
    if not scopes:
        pytest.skip("third_party/Megatron-LM is not checked out")

    by_value = {scope.value: scope for scope in scopes}
    assert "deepseek_v4" in by_value
    scope = by_value["deepseek_v4"]
    assert (scope.key, scope.default) == ("model_type", "gpt")
    assert scope.config_class == "DeepSeekV4TransformerConfig"
    assert "norm_epsilon" in scope.keys
    # A field read off `args` by code every model runs is not this model's.
    assert "moe_use_legacy_grouped_gemm" not in scope.keys


def test_real_repo_never_scopes_the_hand_assembled_flux_fields():
    """`FluxConfig` is assembled by `FluxPretrainTrainer`, not by the dispatch.

    It never reaches `core_transformer_config_from_args`, and its trainer class
    appears in no model preset, so no `model_type` value owns its fields and
    they stay legal on the backend as a whole. Should scoping ever start
    attributing an unowned config class to some model, every diffusion preset in
    the tree gets reported for keys it does read.
    """
    result = scan_backend(_REPO_ROOT, "megatron")
    if not result.available:
        pytest.skip("third_party/Megatron-LM is not checked out")

    assert result.scopes, "the model dispatch no longer resolves"
    # Declared by `FluxConfig` and `BaseDiffusionConfig`, by nothing upstream.
    diffusion_fields = {
        "num_joint_layers",
        "num_single_layers",
        "context_dim",
        "vec_in_dim",
        "guidance_embed",
        "single_block_bias",
        "sensitive_layers_enabled",
        "fp8_scaling_strategy",
    }
    for scope in result.scopes:
        assert diffusion_fields.isdisjoint(scope.keys), scope.config_class
    assert [f.key for f in result.scoped if f.key in diffusion_fields] == []
    assert [f.file for f in result.scoped if "diffusion" in f.file] == []


def test_real_repo_never_reports_a_scoped_key_on_the_model_it_belongs_to():
    """The false-positive guard: DeepSeek-V4's own configs set every V4 key there is."""
    result = scan_backend(_REPO_ROOT, "megatron")
    if not result.available:
        pytest.skip("third_party/Megatron-LM is not checked out")

    assert result.scopes, "the model dispatch no longer resolves"
    for finding in result.scoped:
        assert finding.actual != finding.scope.value
    assert [f.file for f in result.scoped if "deepseek_v4" in f.file] == []


# ---------------------------------------------------------------------------
# Scanning and reporting
# ---------------------------------------------------------------------------


def _exp_yaml(overrides: str) -> str:
    return (
        textwrap.dedent(
            """
            work_group: ${PRIMUS_TEAM:amd}
            user_name: root
            exp_name: probe
            modules:
              pre_trainer:
                framework: torchtitan
                config: pre_trainer.yaml
                model: fake.yaml
                overrides:
            """
        ).lstrip()
        + textwrap.indent(textwrap.dedent(overrides).strip() + "\n", " " * 6)
    )


def test_scan_reports_only_unknown_keys(fake_repo: Path):
    _write(
        fake_repo / "examples/torchtitan/configs/probe.yaml",
        _exp_yaml(
            """
            training:
              seq_len: 4096
              seed: 42
            model:
              n_layers: 4
            primus_turbo:
              enable_primus_turbo: true
              bogus: 1
            """
        ),
    )

    result = scan_backend(fake_repo, "torchtitan")

    assert result.available
    assert {f.key for f in result.findings} == {"training.seed", "primus_turbo.bogus"}
    assert not result.errors


def test_scan_skips_modules_of_other_backends(fake_repo: Path):
    _write(
        fake_repo / "examples/torchtitan/configs/other.yaml",
        """
        work_group: amd
        modules:
          pre_trainer:
            framework: megatron
            config: pre_trainer.yaml
            model: fake.yaml
            overrides:
              anything_goes: 1
        """,
    )

    assert scan_backend(fake_repo, "torchtitan").findings == ()


def test_unloadable_config_is_reported_not_raised(fake_repo: Path):
    _write(fake_repo / "examples/torchtitan/configs/broken.yaml", "extends:\n  - missing.yaml\n")

    result = scan_backend(fake_repo, "torchtitan")

    assert [e.file for e in result.errors] == ["examples/torchtitan/configs/broken.yaml"]
    # The reader wrote `missing.yaml`, so name that, not the raw exception path.
    assert result.errors[0].message.startswith("cannot resolve `examples/torchtitan/configs/missing.yaml`")


def test_unloadable_config_does_not_count_as_checked(fake_repo: Path):
    """The whole point of the tool is that nothing passes unexamined."""
    _write(fake_repo / "examples/torchtitan/configs/ok.yaml", _exp_yaml("training:\n  seq_len: 8\n"))
    _write(fake_repo / "examples/torchtitan/configs/broken.yaml", "extends:\n  - missing.yaml\n")

    result = scan_backend(fake_repo, "torchtitan")

    # pre_trainer.yaml + ok.yaml + broken.yaml
    assert result.scanned == 3
    assert result.checked == 2
    assert result.findings == ()


def test_megatron_scan_reports_only_unknown_keys(megatron_repo: Path):
    """Megatron configs are flat, so every finding is a top-level key."""
    _write(
        megatron_repo / "examples/megatron/configs/probe.yaml",
        """
        work_group: amd
        modules:
          pre_trainer:
            framework: megatron
            config: pre_trainer.yaml
            model: fake.yaml
            overrides:
              train_iters: 50
              norm_epsilon: 1.0e-6
              use_turbo_attention: true
              renamed_away: 1
        """,
    )

    result = scan_backend(megatron_repo, "megatron")

    assert result.available
    assert {f.key for f in result.findings} == {"renamed_away"}
    assert not result.errors


def test_scan_without_submodule_is_unavailable(tmp_path: Path):
    result = scan_backend(tmp_path, "torchtitan")

    assert not result.available
    assert result.findings == () and result.scanned == 0


def test_dedupe_collapses_one_key_across_files(fake_repo: Path):
    findings = [Finding("torchtitan", "training.seed", f"cfg{i}.yaml") for i in range(5)]
    findings.append(Finding("torchtitan", "training.steps", "cfg0.yaml"))

    rows = dedupe(findings, load_torchtitan_schema(fake_repo))

    assert [(r.key, r.count) for r in rows] == [("training.seed", 5), ("training.steps", 1)]
    assert rows[0].suggestion == "debug.seed"
    assert rows[0].files == tuple(f"cfg{i}.yaml" for i in range(5))
    assert rows[1].files == ("cfg0.yaml",)


def test_dedupe_drops_duplicate_files_for_one_key():
    rows = dedupe([Finding("maxtext", "gone", "a.yaml")] * 3)

    assert rows[0].count == 3 and rows[0].files == ("a.yaml",)


def test_dedupe_orders_by_blast_radius_whatever_order_it_is_handed():
    """`(backend, key)` is unique per row, so the order is total: no tie for the input order to
    break, and CI can diff one run's summary against the next."""
    findings = [Finding("megatron", "wide", f"cfg{i}.yaml") for i in range(3)]
    findings += [Finding("megatron", "narrow", "cfg0.yaml"), Finding("maxtext", "other", "cfg0.yaml")]

    assert [(r.backend, r.key) for r in dedupe(findings)] == [
        ("maxtext", "other"),
        ("megatron", "wide"),
        ("megatron", "narrow"),
    ]
    assert dedupe(findings) == dedupe(findings[::-1])


def test_allowlist_matches_exact_before_glob():
    allow = Allowlist((AllowRule("a.b", "r", "c", "manual"), AllowRule("a.*", "r", "c", "manual")))

    assert allow.match("a.b").pattern == "a.b"
    assert allow.match("a.c").pattern == "a.*"
    assert allow.match("b.c") is None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_cli():
    path = _REPO_ROOT / "tools/ci/check_config_schema.py"
    spec = importlib.util.spec_from_file_location("primus_check_config_schema", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(key: str, files: tuple[str, ...], suggestion: str | None = None) -> KeyFinding:
    return KeyFinding("megatron", key, len(files), files, suggestion)


def test_group_rows_merges_the_keys_that_share_their_configs():
    """The whole point: one preset's worth of rot is one row, not twenty."""
    cli = _load_cli()
    rows = [
        _row("zeta", ("a.yaml", "b.yaml")),
        _row("alpha", ("a.yaml", "b.yaml")),
        _row("solo", ("c.yaml",)),
    ]

    renamed, merged = cli.group_rows(rows)

    assert renamed == []
    assert [(g.keys, g.count) for g in merged] == [(("alpha", "zeta"), 2), (("solo",), 1)]
    # Merging must not lose a config: the row's files are every member's files.
    assert merged[0].files == ("a.yaml", "b.yaml")


def test_group_rows_keeps_a_renamed_key_out_of_the_merged_table():
    """A rename is a per-key fix; hiding it inside a list of twenty-five undoes the report."""
    cli = _load_cli()
    rows = [
        _row("bulk_one", ("a.yaml", "b.yaml", "c.yaml")),
        _row("bulk_two", ("a.yaml", "b.yaml", "c.yaml")),
        _row("moved", ("a.yaml", "b.yaml", "c.yaml"), suggestion="moved_elsewhere"),
    ]

    renamed, merged = cli.group_rows(rows)

    assert [g.keys for g in renamed] == [("moved",)]
    assert renamed[0].suggestion == "moved_elsewhere"
    assert [g.keys for g in merged] == [("bulk_one", "bulk_two")]


def test_group_rows_does_not_merge_across_a_differing_config_set():
    cli = _load_cli()
    rows = [_row("here", ("a.yaml",)), _row("there", ("b.yaml",))]

    assert [g.keys for g in cli.group_rows(rows)[1]] == [("here",), ("there",)]


def test_group_rows_ignores_the_order_it_is_given():
    """CI diffs the summary, so the same findings must render the same way every run."""
    cli = _load_cli()
    rows = [_row(k, ("a.yaml", "b.yaml")) for k in ("m", "b", "z")] + [_row("x", ("c.yaml",))]

    def keys(found):
        return [[g.keys for g in half] for half in found]

    assert keys(cli.group_rows(rows)) == keys(cli.group_rows(rows[::-1]))


def test_shorten_paths_drops_the_shared_prefix_but_keeps_what_disambiguates():
    cli = _load_cli()
    short = cli.shorten_paths(
        [
            "primus/configs/models/megatron/deepseek_v2.yaml",
            "primus/configs/modules/megatron/pre_trainer.yaml",
            "examples/megatron/configs/MI300X/shared.yaml",
            "examples/megatron/configs/MI355X/shared.yaml",
        ]
    )

    assert short["primus/configs/models/megatron/deepseek_v2.yaml"] == "deepseek_v2.yaml"
    assert short["primus/configs/modules/megatron/pre_trainer.yaml"] == "pre_trainer.yaml"
    # `shared.yaml` alone would name two different configs, so the tail grows.
    assert short["examples/megatron/configs/MI300X/shared.yaml"] == "MI300X/shared.yaml"
    assert short["examples/megatron/configs/MI355X/shared.yaml"] == "MI355X/shared.yaml"


def test_shorten_paths_falls_back_to_the_whole_path_when_a_tail_never_disambiguates():
    """A path that is another one's suffix runs out of segments before it runs out of rivals."""
    cli = _load_cli()

    short = cli.shorten_paths(["a/b.yaml", "x/a/b.yaml"])

    assert short == {"a/b.yaml": "a/b.yaml", "x/a/b.yaml": "x/a/b.yaml"}


_RENAMED_TABLE = "### Keys that look renamed rather than dropped"
_MERGED_TABLE = "### Keys with no obvious replacement"


def _drift_out(fake_repo: Path, monkeypatch, capsys, overrides: str, files: int = 1) -> str:
    for i in range(files):
        _write(fake_repo / f"examples/torchtitan/configs/probe{i}.yaml", _exp_yaml(overrides))
    cli = _load_cli()
    monkeypatch.setattr(cli, "ROOT", fake_repo)
    monkeypatch.setattr("sys.argv", ["check_config_schema.py", "--backend", "torchtitan", "--warn-only"])
    assert cli.main() == 0
    return capsys.readouterr().out


def test_cli_prints_every_grouped_key_as_plain_text(fake_repo: Path, monkeypatch, capsys):
    """CI logs get grepped for a key name, so merging rows must not hide one."""
    out = _drift_out(fake_repo, monkeypatch, capsys, "training:\n  seed: 1\n  gone_a: 1\n  gone_b: 2\n", 3)

    assert _RENAMED_TABLE in out and _MERGED_TABLE in out
    assert "`training.gone_a`, `training.gone_b`" in out  # merged onto one row, both still named
    assert "| `training.seed` | 3 | `debug.seed` |" in out  # the rename keeps its own row and column
    assert "2 distinct set(s) of configs" not in out  # the two share one set
    assert "The other 2 drifted out of only 1 distinct set(s) of configs" in out


def test_cli_omits_the_rename_table_when_nothing_looks_renamed(fake_repo: Path, monkeypatch, capsys):
    """An empty table is a heading and a header row saying nothing; print neither."""
    out = _drift_out(fake_repo, monkeypatch, capsys, "training:\n  gone_a: 1\n")

    assert _RENAMED_TABLE not in out
    assert "Likely moved to" not in out  # the column only ever holds a suggestion
    assert _MERGED_TABLE in out
    assert "All 1 drifted out of only 1 distinct set(s) of configs" in out  # not "The other 1"
    assert "`training.gone_a`" in out


def test_cli_omits_the_merged_table_when_everything_looks_renamed(fake_repo: Path, monkeypatch, capsys):
    out = _drift_out(fake_repo, monkeypatch, capsys, "training:\n  seed: 1\n")

    assert _MERGED_TABLE not in out
    assert "distinct set(s) of configs" not in out
    assert _RENAMED_TABLE in out
    assert "| `training.seed` | 1 | `debug.seed` |" in out


def test_cli_fails_when_no_submodule_is_checked_out(tmp_path: Path, monkeypatch, capsys):
    cli = _load_cli()
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr("sys.argv", ["check_config_schema.py"])

    assert cli.main() == 1
    out = capsys.readouterr().out
    assert "backend(s) not checked" in out and "git submodule update --init" in out


def test_cli_fails_on_a_skipped_backend_even_when_warning_only(fake_repo: Path, monkeypatch, capsys):
    """The bug this exists for: a partial checkout used to pass in silence.

    ``fake_repo`` has TorchTitan and MaxText but no Megatron, which is exactly
    what a CI job that fetches two of the three submodules produces.
    """
    cli = _load_cli()
    monkeypatch.setattr(cli, "ROOT", fake_repo)
    monkeypatch.setattr("sys.argv", ["check_config_schema.py", "--warn-only"])

    assert cli.main() == 1
    out = capsys.readouterr().out
    assert "FAILED -- 1 backend(s) not checked" in out
    assert "**megatron**" in out
    assert "No drift" in out  # the two backends that were checked are clean

    # ... while the same two backends on their own still pass.
    monkeypatch.setattr(
        "sys.argv",
        ["check_config_schema.py", "--warn-only", "--backend", "torchtitan", "--backend", "maxtext"],
    )
    assert cli.main() == 0


def test_cli_warn_only_suppresses_the_failure(fake_repo: Path, monkeypatch, capsys):
    _write(
        fake_repo / "examples/torchtitan/configs/probe.yaml",
        _exp_yaml("training:\n  seed: 42\n"),
    )
    cli = _load_cli()
    monkeypatch.setattr(cli, "ROOT", fake_repo)

    monkeypatch.setattr("sys.argv", ["check_config_schema.py", "--backend", "torchtitan"])
    assert cli.main() == 1
    assert "`training.seed`" in capsys.readouterr().out

    monkeypatch.setattr("sys.argv", ["check_config_schema.py", "--backend", "torchtitan", "--warn-only"])
    assert cli.main() == 0


def test_cli_reports_a_misplaced_key_apart_from_drift(megatron_repo: Path, monkeypatch, capsys):
    """The two need different fixes, so they must not share a table."""
    _write(
        megatron_repo / "examples/megatron/configs/stray.yaml",
        """
        work_group: amd
        modules:
          pre_trainer:
            framework: megatron
            config: pre_trainer.yaml
            model: plain.yaml
            overrides:
              hash_routing_seed: 7
        """,
    )
    cli = _load_cli()
    monkeypatch.setattr(cli, "ROOT", megatron_repo)
    monkeypatch.setattr("sys.argv", ["check_config_schema.py", "--backend", "megatron"])

    assert cli.main() == 1
    out = capsys.readouterr().out
    assert "No drift: every key in" in out  # the key exists; it is only misplaced
    assert "Keys set on a model that cannot read them" in out
    assert "`hash_routing_seed`" in out and "model_type: fake_v4" in out

    monkeypatch.setattr("sys.argv", ["check_config_schema.py", "--backend", "megatron", "--warn-only"])
    assert cli.main() == 0


def test_cli_fails_when_a_config_could_not_be_loaded(fake_repo: Path, monkeypatch, capsys):
    """An unchecked config must never be reported as a clean one."""
    _write(fake_repo / "examples/torchtitan/configs/broken.yaml", "extends:\n  - missing.yaml\n")
    cli = _load_cli()
    monkeypatch.setattr(cli, "ROOT", fake_repo)

    monkeypatch.setattr("sys.argv", ["check_config_schema.py", "--backend", "torchtitan"])
    assert cli.main() == 1
    out = capsys.readouterr().out
    assert "could not be loaded and were NOT checked" in out
    assert "broken.yaml" in out
    assert "No drift: every key in" not in out
    assert "Checked 1 of 2 config(s)" in out

    monkeypatch.setattr("sys.argv", ["check_config_schema.py", "--backend", "torchtitan", "--warn-only"])
    assert cli.main() == 0
