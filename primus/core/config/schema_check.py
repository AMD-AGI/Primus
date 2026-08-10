###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Detect Primus YAML keys that no longer exist in the upstream backend schema.

Every adapter takes unknown keys silently -- TorchTitan attaches them as dynamic attributes,
MaxText lets pydantic drop them, Megatron gets them merged onto ``args`` by ``train_runtime`` --
so a renamed field keeps parsing while doing nothing: ``training.debug_moe_force_load_balance``
moved to ``debug.moe_force_load_balance``, and every DeepSeek config went on training with MoE
load balancing off. Schemas are read statically (AST, ``yaml.safe_load``), so the check imports
neither torch nor jax and runs on a plain CI runner.

Legal keys are upstream's plus the Primus extensions, and the extensions are derived rather than
listed: ``get_param(ctx, "<path>", ...)`` calls in the patch packages, reads off Megatron's
``args`` namespace, Primus dataclasses extending a backend config class, and a hand-written
allowlist. ``MegatronArgBuilder`` stripping a key is not drift: ``train_runtime`` merges the
original params back over the result, which is how roughly 130 Primus-only keys reach ``args``.

A legal key can still be dead where it is written: ``core_transformer_config_from_args`` copies a
Primus config class's fields off ``args`` only on the model naming that class. ``norm_epsilon``
is DeepSeek-V4's and not an upstream dest -- upstream's ``layernorm_epsilon`` merely spells its
flag ``--norm-epsilon`` -- so it is live there and inert on a Llama config. Those are reported as
misplaced rather than unknown, and a config whose model cannot be resolved is left alone.
"""

from __future__ import annotations

import ast
import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import yaml

from primus.core.config.yaml_loader import parse_yaml

BACKENDS = ("torchtitan", "maxtext", "megatron")

TORCHTITAN_JOB_CONFIG = Path("third_party/torchtitan/torchtitan/config/job_config.py")
MAXTEXT_BASE_YML = Path("third_party/maxtext/src/maxtext/configs/base.yml")
MEGATRON_ROOT = Path("third_party/Megatron-LM")
MEGATRON_ARGUMENTS = Path("megatron/training/arguments.py")

# `core_transformer_config_from_args` copies every field of these classes off
# `args`, so their fields stay legal even where ArgumentGroupFactory excludes
# them from argparse (`moe_token_dropping`, `fp8_multi_head_attention`, ...).
MEGATRON_CONFIG_CLASSES = ("TransformerConfig", "MLATransformerConfig")

# Module preset whose `experimental.custom_args_module` points at the dataclass
# that extends TorchTitan's JobConfig.
TORCHTITAN_MODULE_PRESET = Path("primus/configs/modules/torchtitan/pre_trainer.yaml")

PATCH_DIRS = {
    "torchtitan": Path("primus/backends/torchtitan/patches"),
    "maxtext": Path("primus/backends/maxtext/patches"),
}

# Packages whose reads off the backend's argument namespace define the
# Primus-only half of its key set, and which hold the Primus subclasses of the
# backend's config dataclass.
BACKEND_PACKAGES = {"megatron": Path("primus/backends/megatron")}

# `get_model_provider(model_type=...)` is where a Megatron run decides which
# model it builds, and the branches that name a Primus module are the ones that
# lead to a Primus config class. See `extract_model_scopes`.
MODEL_PROVIDERS = {
    "megatron": (Path("primus/core/utils/import_utils.py"), "get_model_provider", "model_type"),
}

# `module.model` names a preset here, and that preset is where a model declares
# its `model_type`; the experiment YAML that sets a model-scoped key usually
# does not repeat it.
MODEL_PRESETS = Path("primus/configs/models")

# Every Python file Primus owns. A name read off an argument namespace anywhere
# in here is model-agnostic evidence, which is what keeps a shared key such as
# `moe_use_legacy_grouped_gemm` out of the model-scoped key sets.
PRIMUS_PACKAGE = Path("primus")

SCAN_ROOTS = {
    "torchtitan": (
        Path("examples/torchtitan/configs"),
        Path("primus/configs/modules/torchtitan"),
        Path("primus/configs/models/torchtitan"),
    ),
    "maxtext": (
        Path("examples/maxtext/configs"),
        Path("primus/configs/modules/maxtext"),
        Path("primus/configs/models/maxtext"),
    ),
    "megatron": (
        Path("examples/megatron/configs"),
        Path("primus/configs/modules/megatron"),
        Path("primus/configs/models/megatron"),
    ),
}

DEFAULT_ALLOWLIST = Path("tools/ci/config_schema_allowlist.yaml")

# Keys PrimusParser.parse_trainer_module consumes on the module node itself;
# they never reach the backend.
MODULE_RESERVED_KEYS = frozenset({"name", "framework", "config", "model", "overrides", "params"})


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BackendSchema:
    """Upstream key universe for one backend."""

    backend: str
    keys: frozenset[str]
    sections: frozenset[str]
    source: str

    def extended(self, keys: Iterable[str], sections: Iterable[str], source: str) -> "BackendSchema":
        return BackendSchema(
            backend=self.backend,
            keys=self.keys | frozenset(keys),
            sections=self.sections | frozenset(sections),
            source=f"{self.source} + {source}",
        )

    def suggest(self, path: str) -> str | None:
        """Best guess for where a dropped key moved to: the leaf name, the leaf minus a leading
        ``<section>_`` (moved into a new section), or the leaf plus a prefix or suffix (renamed
        in place). Ambiguity yields ``None``."""
        leaf = path.rsplit(".", 1)[-1]
        candidates = {leaf}
        head = leaf.split("_", 1)
        if len(head) == 2 and head[0] in {s.split(".")[0] for s in self.sections | self.keys}:
            candidates.add(head[1])
        matches = sorted(k for k in self.keys if k != path and k.rsplit(".", 1)[-1] in candidates)
        if not matches:
            matches = sorted(
                k
                for k in self.keys
                if (tail := k.rsplit(".", 1)[-1]) != leaf
                and (tail.startswith(f"{leaf}_") or tail.endswith(f"_{leaf}"))
            )
        return matches[0] if len(matches) == 1 else None


def _dataclass_defs(tree: ast.Module) -> dict[str, list[tuple[str, str]]]:
    """Map ``@dataclass`` class name -> ``[(field name, annotation source)]``."""
    defs: dict[str, list[tuple[str, str]]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        decorated = any("dataclass" in ast.unparse(d) for d in node.decorator_list)
        if not decorated:
            continue
        fields = [
            (stmt.target.id, ast.unparse(stmt.annotation).strip("\"'"))
            for stmt in node.body
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
        ]
        defs[node.name] = fields
    return defs


def _expand_dataclass(
    defs: dict[str, list[tuple[str, str]]],
    cls: str,
    prefix: str,
    keys: set[str],
    sections: set[str],
    seen: frozenset[str] = frozenset(),
) -> None:
    for name, annotation in defs.get(cls, ()):
        path = f"{prefix}.{name}" if prefix else name
        if annotation in defs and annotation not in seen:
            sections.add(path)
            _expand_dataclass(defs, annotation, path, keys, sections, seen | {annotation})
        else:
            keys.add(path)


def load_torchtitan_schema(repo_root: Path) -> BackendSchema | None:
    """Extract ``JobConfig`` from TorchTitan's ``job_config.py`` via AST; the module imports
    torch, so it cannot simply be imported here."""
    src = repo_root / TORCHTITAN_JOB_CONFIG
    if not src.is_file():
        return None
    defs = _dataclass_defs(ast.parse(src.read_text()))
    if "JobConfig" not in defs:
        raise ValueError(f"no JobConfig dataclass found in {TORCHTITAN_JOB_CONFIG}")
    keys: set[str] = set()
    sections: set[str] = set()
    _expand_dataclass(defs, "JobConfig", "", keys, sections)
    return BackendSchema(
        backend="torchtitan",
        keys=frozenset(keys),
        sections=frozenset(sections),
        source=str(TORCHTITAN_JOB_CONFIG),
    )


def load_maxtext_schema(repo_root: Path) -> BackendSchema | None:
    """MaxText's ``base.yml`` is a flat mapping and is itself the schema."""
    src = repo_root / MAXTEXT_BASE_YML
    if not src.is_file():
        return None
    data = yaml.safe_load(src.read_text()) or {}
    return BackendSchema(
        backend="maxtext",
        keys=frozenset(data),
        sections=frozenset(),
        source=str(MAXTEXT_BASE_YML),
    )


def _calls_named(tree: ast.AST, name: str) -> Iterator[ast.Call]:
    """Every call whose callee is spelled ``name`` or ``<something>.name``."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if called == name:
            yield node


def _string_elements(node: ast.expr | None) -> set[str] | None:
    """String constants of a literal list/tuple/set; ``None`` if not one."""
    if not isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return None
    return {e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)}


def _argparse_dest(call: ast.Call) -> str | None:
    """The attribute name argparse derives from one ``add_argument`` call: ``dest=`` if given, else
    the first long option, so ``--no-persist-layer-norm`` gives ``no_persist_layer_norm``."""
    for keyword in call.keywords:
        if keyword.arg == "dest" and isinstance(keyword.value, ast.Constant):
            return str(keyword.value.value)
    options = [a.value for a in call.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
    chosen = next((o for o in options if o.startswith("--")), None) or next(iter(options), None)
    return chosen.lstrip("-").replace("-", "_") if chosen else None


class _MegatronSource:
    """Just enough of an import resolver to walk the checked-out Megatron tree: config dataclasses
    arrive through re-exports and inherit across packages, so a flat class index is not faithful."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._trees: dict[Path, ast.Module] = {}

    def tree(self, path: Path) -> ast.Module:
        if path not in self._trees:
            self._trees[path] = ast.parse(path.read_text())
        return self._trees[path]

    def module_file(self, module: str) -> Path | None:
        rel = module.replace(".", "/")
        for candidate in (self.root / f"{rel}.py", self.root / rel / "__init__.py"):
            if candidate.is_file():
                return candidate
        return None

    def imports(self, path: Path) -> dict[str, str]:
        """``local name -> absolute module`` for every ``from X import name``."""
        package = ".".join(path.relative_to(self.root).parts[:-1])
        found: dict[str, str] = {}
        for node in ast.walk(self.tree(path)):
            if not isinstance(node, ast.ImportFrom):
                continue
            module = node.module or ""
            if node.level:  # `from ..model_parallel_config import ModelParallelConfig`
                parts = package.split(".")
                parts = parts[: len(parts) - node.level + 1]
                module = ".".join([*parts, module]) if module else ".".join(parts)
            for alias in node.names:
                found[alias.asname or alias.name] = module
        return found

    def resolve(
        self, path: Path, name: str, seen: frozenset[str] = frozenset()
    ) -> tuple[Path, ast.ClassDef] | None:
        """Find class ``name`` as visible from ``path``, with its defining file: base classes must
        be resolved in the scope that declares them, not the one that referenced them."""
        for node in self.tree(path).body:
            if isinstance(node, ast.ClassDef) and node.name == name:
                return path, node
        module = self.imports(path).get(name)
        if module is None or module in seen:
            return None
        target = self.module_file(module)
        return self.resolve(target, name, seen | {module}) if target else None

    def config_fields(self, path: Path, cls: ast.ClassDef, exclude: Iterable[str] = ()) -> list[str]:
        """Names ``ArgumentGroupFactory`` would turn into arguments, in field order, mirroring
        ``build_group``: base-class fields first, ``init=False`` and ``ClassVar`` skipped, an
        ``argparse_meta`` ``dest`` override honoured (``use_nsys_profiler`` -> ``profile``)."""
        skip = set(exclude)
        names: list[str] = []
        for base in cls.bases:
            if not isinstance(base, ast.Name):
                continue
            found = self.resolve(path, base.id)
            if found:
                names.extend(self.config_fields(*found, exclude=skip))
        for stmt in cls.body:
            if not isinstance(stmt, ast.AnnAssign) or not isinstance(stmt.target, ast.Name):
                continue
            if ast.unparse(stmt.annotation).lstrip("\"'").startswith(("ClassVar", "InitVar")):
                continue
            if _has_field_kwarg(stmt.value, "init", False):
                continue
            dest = _argparse_meta_dest(stmt.value) or stmt.target.id
            if dest not in skip and dest not in names:
                names.append(dest)
        return names


def _field_call(value: ast.expr | None) -> ast.Call | None:
    """The ``dataclasses.field(...)`` call of a field default, if any."""
    if not isinstance(value, ast.Call):
        return None
    func = value.func
    called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
    return value if called == "field" else None


def _has_field_kwarg(value: ast.expr | None, name: str, expected: Any) -> bool:
    call = _field_call(value)
    if call is None:
        return False
    return any(
        kw.arg == name and isinstance(kw.value, ast.Constant) and kw.value.value is expected
        for kw in call.keywords
    )


def _argparse_meta_dest(value: ast.expr | None) -> str | None:
    """``metadata={"argparse_meta": {"dest": "profile"}}`` -> ``"profile"``."""
    call = _field_call(value)
    if call is None:
        return None
    for keyword in call.keywords:
        if keyword.arg != "metadata" or not isinstance(keyword.value, ast.Dict):
            continue
        for outer_key, outer in zip(keyword.value.keys, keyword.value.values):
            if not (isinstance(outer_key, ast.Constant) and outer_key.value == "argparse_meta"):
                continue
            if not isinstance(outer, ast.Dict):
                continue
            for inner_key, inner in zip(outer.keys, outer.values):
                if isinstance(inner_key, ast.Constant) and inner_key.value == "dest":
                    if isinstance(inner, ast.Constant):
                        return str(inner.value)
    return None


def _resolve_exclude(tree: ast.Module, call: ast.Call) -> set[str]:
    """The ``exclude=`` list of one ``ArgumentGroupFactory`` call, following a literal assignment
    when the argument is a variable, as ``TransformerConfig``'s call passes. An expression that
    cannot be read statically yields no exclusions, erring towards a missed report."""
    argument = next((kw.value for kw in call.keywords if kw.arg == "exclude"), None)
    literal = _string_elements(argument)
    if literal is not None:
        return literal
    if not isinstance(argument, ast.Name):
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        targets = node.targets if isinstance(node, ast.Assign) else []
        if any(isinstance(t, ast.Name) and t.id == argument.id for t in targets):
            names |= _string_elements(node.value) or set()
    return names


def load_megatron_schema(repo_root: Path) -> BackendSchema | None:
    """Rebuild Megatron's flat argument namespace from ``arguments.py``: roughly 355 literal
    ``add_argument`` calls, plus eleven ``ArgumentGroupFactory(<dataclass>)`` calls generating as
    many again. The generated half holds ``train_iters``, ``global_batch_size`` and
    ``micro_batch_size``, so the literal calls alone would report half of every config as drift."""
    root = repo_root / MEGATRON_ROOT
    src = root / MEGATRON_ARGUMENTS
    if not src.is_file():
        return None
    source = _MegatronSource(root)
    tree = source.tree(src)

    keys = {dest for call in _calls_named(tree, "add_argument") if (dest := _argparse_dest(call))}
    for call in _calls_named(tree, "ArgumentGroupFactory"):
        target = call.args[0] if call.args else None
        if not isinstance(target, ast.Name):
            continue
        found = source.resolve(src, target.id)
        if found is None:
            raise ValueError(f"{MEGATRON_ARGUMENTS}: cannot resolve ArgumentGroupFactory({target.id})")
        keys.update(source.config_fields(*found, exclude=_resolve_exclude(tree, call)))
    for name in MEGATRON_CONFIG_CLASSES:
        found = source.resolve(src, name)
        if found is not None:
            keys.update(source.config_fields(*found))

    return BackendSchema(
        backend="megatron",
        keys=frozenset(keys),
        sections=frozenset(),  # Megatron's namespace is flat: every key is a leaf.
        source=str(MEGATRON_ROOT / MEGATRON_ARGUMENTS),
    )


SCHEMA_LOADERS = {
    "torchtitan": load_torchtitan_schema,
    "maxtext": load_maxtext_schema,
    "megatron": load_megatron_schema,
}


# ---------------------------------------------------------------------------
# Allowlist
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AllowRule:
    pattern: str
    reason: str
    consumer: str
    origin: str

    def matches(self, path: str) -> bool:
        return path == self.pattern or fnmatch.fnmatchcase(path, self.pattern)


@dataclass(frozen=True)
class Allowlist:
    """Rules that widen a backend's legal key set, indexed for lookup: Megatron derives thousands,
    so exact patterns are hashed and only real fnmatch patterns scanned, and exact wins over glob."""

    rules: tuple[AllowRule, ...]

    def __post_init__(self) -> None:
        exact: dict[str, AllowRule] = {}
        globs: list[AllowRule] = []
        prefixes: set[str] = set()
        for rule in self.rules:
            if any(ch in rule.pattern for ch in "*?["):
                globs.append(rule)
            else:
                exact.setdefault(rule.pattern, rule)
            parts = rule.pattern.split(".")
            prefixes.update(".".join(parts[:i]) for i in range(1, len(parts)))
        object.__setattr__(self, "_exact", exact)
        object.__setattr__(self, "_globs", tuple(globs))
        object.__setattr__(self, "_prefixes", frozenset(prefixes))

    def match(self, path: str) -> AllowRule | None:
        rule = self._exact.get(path)
        return rule or next((r for r in self._globs if r.matches(path)), None)

    def is_prefix(self, path: str) -> bool:
        """True when some rule targets a child of ``path`` (so ``path`` is a section)."""
        return path in self._prefixes

    def patterns(self) -> list[str]:
        return sorted({r.pattern for r in self.rules})


def extract_get_param_rules(repo_root: Path, backend: str) -> list[AllowRule]:
    """Collect config paths read by ``get_param(ctx, "<path>", ...)`` in patches."""
    patch_dir = repo_root / PATCH_DIRS.get(backend, Path("does/not/exist"))
    if not patch_dir.is_dir():
        return []
    rules: list[AllowRule] = []
    for py in sorted(patch_dir.rglob("*.py")):
        tree = ast.parse(py.read_text())
        rel = py.relative_to(repo_root)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or len(node.args) < 2:
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name != "get_param":
                continue
            arg = node.args[1]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                rules.append(
                    AllowRule(
                        pattern=arg.value,
                        reason="read by the Primus patch system",
                        consumer=f"{rel}:{node.lineno}",
                        origin="patch:get_param",
                    )
                )
    return rules


_ARGS_OBJECT_HINTS = ("arg", "cfg", "config")


def _is_args_object(node: ast.expr) -> bool:
    """True when ``node`` looks like Megatron's argument or config namespace (``args.x``,
    ``self.args.x``, ``get_args().x``), but not every attribute read: ``tensor.shape`` is not one."""
    while isinstance(node, ast.Call):
        node = node.func
    name = node.attr if isinstance(node, ast.Attribute) else getattr(node, "id", None)
    return bool(name) and any(hint in name.lower() for hint in _ARGS_OBJECT_HINTS)


def extract_args_read_rules(repo_root: Path, backend: str) -> list[AllowRule]:
    """Collect the names a backend package reads off the argument namespace. Two forms count as a
    reader: attribute access on an args-shaped object, and the bare name as a string literal
    anywhere in the package -- reads reach ``args`` through too many indirections (module
    constants, local getters, config-to-env bridges, generated source) for a narrower rule."""
    package = repo_root / BACKEND_PACKAGES.get(backend, Path("does/not/exist"))
    if not package.is_dir():
        return []
    rules: list[AllowRule] = []
    seen: set[str] = set()

    def record(name: str, path: Path, lineno: int, origin: str) -> None:
        # An ALL_CAPS literal is a constant, an enum member or an environment variable, never an
        # argument name -- and a config setting one as if it were (`HSA_NO_SCRATCH_RECLAIM: 1`)
        # is exactly the mistake this check should keep reporting.
        if name in seen or not name.isidentifier() or name == name.upper():
            return
        seen.add(name)
        rules.append(
            AllowRule(
                pattern=name,
                reason=f"read off the {backend} args namespace by the backend package",
                consumer=f"{path.relative_to(repo_root)}:{lineno}",
                origin=origin,
            )
        )

    for py in sorted(package.rglob("*.py")):
        for node in ast.walk(ast.parse(py.read_text())):
            if isinstance(node, ast.Attribute) and _is_args_object(node.value):
                record(node.attr, py, node.lineno, "backend:args_attribute")
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                record(node.value, py, node.lineno, "backend:args_name")
    return rules


def _class_bases(node: ast.ClassDef) -> set[str]:
    return {ast.unparse(base).rsplit(".", 1)[-1] for base in node.bases}


def _class_fields(node: ast.ClassDef) -> set[str]:
    return {
        stmt.target.id
        for stmt in node.body
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
    }


def _primus_config_classes(repo_root: Path, backend: str) -> tuple[dict[str, ast.ClassDef], frozenset[str]]:
    """``(every class in the backend package, those extending its config class)``, grown to a
    fixpoint because the chain can be several classes long: ``FluxConfig`` ->
    ``BaseDiffusionConfig`` -> ``TransformerConfig``."""
    package = repo_root / BACKEND_PACKAGES.get(backend, Path("does/not/exist"))
    if not package.is_dir():
        return {}, frozenset()
    classes: dict[str, ast.ClassDef] = {}
    for py in sorted(package.rglob("*.py")):
        for node in ast.walk(ast.parse(py.read_text())):
            if isinstance(node, ast.ClassDef):
                classes[node.name] = node

    derived: set[str] = set()
    frontier = set(MEGATRON_CONFIG_CLASSES)
    while frontier:
        frontier = {
            n for n, c in classes.items() if n not in derived and _class_bases(c) & (frontier | derived)
        }
        derived |= frontier
    return classes, frozenset(derived)


def extract_backend_config_extensions(repo_root: Path, backend: str) -> tuple[frozenset[str], str]:
    """Fields of the Primus dataclasses that extend the backend's config class, which
    ``core_transformer_config_from_args`` copies off ``args`` and thereby makes legal keys."""
    classes, derived = _primus_config_classes(repo_root, backend)
    if not derived:
        return frozenset(), ""
    keys = frozenset().union(*(_class_fields(classes[name]) for name in derived))
    return keys, str(BACKEND_PACKAGES[backend])


# ---------------------------------------------------------------------------
# Model scopes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelScope:
    """Keys that only reach the model one YAML discriminator selects: live on that model's path,
    a silent no-op on every other."""

    key: str
    """YAML key that decides which model gets built, e.g. ``model_type``."""

    value: str
    """The value of ``key`` that selects this model, e.g. ``deepseek_v4``."""

    default: str | None
    """What ``key`` means when a config omits it; ``None`` when there is no default."""

    config_class: str
    keys: frozenset[str]
    source: str

    def describe(self) -> str:
        return f"{self.key}: {self.value}"


def _is_args_namespace(node: ast.expr) -> bool:
    """True for ``args``, ``self.args``, ``get_args()``, ``backend_args``, ..."""
    while isinstance(node, ast.Call):
        node = node.func
    name = node.attr if isinstance(node, ast.Attribute) else getattr(node, "id", None)
    return bool(name) and "arg" in name.lower()


def _args_namespace_reads(repo_root: Path) -> frozenset[str]:
    """Names read off an *argument* namespace anywhere in ``primus/``, deliberately narrower than
    ``_is_args_object``: ``config.norm_epsilon`` only says the key is a field of whichever config
    class got built, while ``args.moe_use_legacy_grouped_gemm`` says it reaches its reader on
    every model. Only the latter disqualifies a key from being model-scoped."""
    package = repo_root / PRIMUS_PACKAGE
    if not package.is_dir():
        return frozenset()
    names: set[str] = set()
    for py in sorted(package.rglob("*.py")):
        try:
            tree = ast.parse(py.read_text())
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and _is_args_namespace(node.value):
                names.add(node.attr)
            elif isinstance(node, ast.Call) and len(node.args) >= 2:
                called = getattr(node.func, "id", None)
                target = node.args[1]
                if called in ("getattr", "setattr", "hasattr") and _is_args_namespace(node.args[0]):
                    if isinstance(target, ast.Constant) and isinstance(target.value, str):
                        names.add(target.value)
    return frozenset(names)


def _model_provider_branches(repo_root: Path, backend: str) -> tuple[str | None, dict[str, str], str]:
    """Read the model dispatch -- ``(default, {discriminator value: module}, source)`` -- from
    ``get_model_provider``'s branches; only one naming a Primus module reaches a Primus config."""
    src, func_name, key = MODEL_PROVIDERS.get(backend, (None, "", ""))
    if src is None:
        return None, {}, ""
    path = repo_root / src
    if not path.is_file():
        return None, {}, ""
    tree = ast.parse(path.read_text())
    func = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == func_name),
        None,
    )
    if func is None:
        return None, {}, ""

    default = None
    positional = func.args.posonlyargs + func.args.args
    for arg, value in zip(positional[len(positional) - len(func.args.defaults) :], func.args.defaults):
        if arg.arg == key and isinstance(value, ast.Constant) and isinstance(value.value, str):
            default = value.value

    branches: dict[str, str] = {}
    for node in ast.walk(func):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not isinstance(test, ast.Compare) or len(test.ops) != 1:
            continue
        if not isinstance(test.ops[0], ast.Eq) or not isinstance(test.left, ast.Name):
            continue
        if test.left.id != key:
            continue
        right = test.comparators[0]
        if not (isinstance(right, ast.Constant) and isinstance(right.value, str)):
            continue
        for stmt in node.body:
            for call in _calls_named(stmt, "import_module"):
                arg = call.args[0] if call.args else None
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    branches[right.value] = arg.value
    return default, branches, str(src)


def _config_class_of_module(repo_root: Path, module: str) -> str | None:
    """The class a module hands to ``core_transformer_config_from_args``."""
    path = repo_root / (module.replace(".", "/") + ".py")
    if not path.is_file():
        return None
    tree = ast.parse(path.read_text())
    for call in _calls_named(tree, "core_transformer_config_from_args"):
        for keyword in call.keywords:
            if keyword.arg == "config_class" and isinstance(keyword.value, ast.Name):
                return keyword.value.id
    return None


def _inherited_fields(classes: dict[str, ast.ClassDef], derived: frozenset[str], cls: str) -> set[str]:
    """Fields ``cls`` declares plus its Primus-side ancestors', stopping at the backend's own
    config class: an upstream field is upstream's, not this model's."""
    fields: set[str] = set()
    frontier = {cls}
    seen: set[str] = set()
    while frontier:
        name = frontier.pop()
        if name in seen or name not in derived:
            continue
        seen.add(name)
        fields |= _class_fields(classes[name])
        frontier |= _class_bases(classes[name])
    return fields


def extract_model_scopes(
    repo_root: Path, backend: str, upstream: BackendSchema, allow: Allowlist
) -> tuple[ModelScope, ...]:
    """Derive the model-scoped key sets by following the model dispatch. Four things disqualify a
    field, each meaning it reaches something on other models too: upstream defines the argument;
    another Primus config class declares it; something reads it off the argument namespace; or a
    patch or the hand-written allowlist vouches for it globally."""
    default, branches, source = _model_provider_branches(repo_root, backend)
    if not branches:
        return ()
    key = MODEL_PROVIDERS[backend][2]
    classes, derived = _primus_config_classes(repo_root, backend)

    owners: dict[str, str] = {}  # config class -> discriminator value
    for value, module in sorted(branches.items()):
        config_class = _config_class_of_module(repo_root, module)
        if config_class is not None and config_class in derived:
            owners[config_class] = value
    if not owners:
        return ()

    # Which models claim each field. `None` stands for a Primus config class no
    # discriminator selects, so its fields belong to no model in particular.
    claims: dict[str, set[str | None]] = {}
    for name in sorted(derived):
        value = owners.get(name)
        for field in _class_fields(classes[name]):
            claims.setdefault(field, set()).add(value)
    agnostic = _args_namespace_reads(repo_root)

    def is_scoped(field: str, value: str) -> bool:
        if claims.get(field) != {value} or field in upstream.keys or field in agnostic:
            return False
        rule = allow.match(field)
        # A `backend:` rule is the read that made the key legal in the first
        # place, and it is exactly what the scope is about. A patch or a
        # hand-written entry is a claim that the key works everywhere.
        return rule is None or rule.origin.startswith("backend:")

    scopes = []
    for config_class, value in sorted(owners.items(), key=lambda item: item[1]):
        keys = {k for k in _inherited_fields(classes, derived, config_class) if is_scoped(k, value)}
        if not keys:
            continue
        scopes.append(
            ModelScope(
                key=key,
                value=value,
                default=default,
                config_class=config_class,
                keys=frozenset(keys),
                source=source,
            )
        )
    return tuple(scopes)


def build_model_scopes(
    repo_root: Path, backend: str, allowlist_path: Path | None = None
) -> tuple[ModelScope, ...]:
    """``extract_model_scopes`` from a repo root alone; empty when unavailable."""
    upstream = SCHEMA_LOADERS[backend](repo_root)
    if upstream is None:
        return ()
    allow = build_allowlist(repo_root, backend, allowlist_path)
    return extract_model_scopes(repo_root, backend, upstream, allow)


def extract_custom_args_schema(repo_root: Path) -> tuple[frozenset[str], frozenset[str], str]:
    """Follow ``experimental.custom_args_module`` and expand the dataclass it names, mirroring
    ``config_utils.build_job_config_from_namespace``. Returned as schema, not allowlist entries,
    so unknown keys *inside* an extension section stay reportable."""
    empty = (frozenset(), frozenset(), "")
    preset = repo_root / TORCHTITAN_MODULE_PRESET
    if not preset.is_file():
        return empty
    module_path = (parse_yaml(str(preset)).get("experimental") or {}).get("custom_args_module")
    if not module_path:
        return empty
    src = repo_root / (module_path.replace(".", "/") + ".py")
    if not src.is_file():
        return empty
    defs = _dataclass_defs(ast.parse(src.read_text()))
    keys: set[str] = set()
    sections: set[str] = set()
    _expand_dataclass(defs, "JobConfig", "", keys, sections)
    return frozenset(keys), frozenset(sections), module_path


def load_manual_allowlist(path: Path, backend: str) -> list[AllowRule]:
    """Read the hand-written allowlist; ``reason`` and ``consumer`` are mandatory."""
    if not path.is_file():
        return []
    data = yaml.safe_load(path.read_text()) or {}
    rules = []
    for scope in ("common", backend):
        for entry in data.get(scope) or []:
            missing = [f for f in ("key", "reason", "consumer") if not entry.get(f)]
            if missing:
                raise ValueError(f"{path}: entry {entry!r} is missing required field(s) {missing}")
            rules.append(
                AllowRule(
                    pattern=entry["key"],
                    reason=entry["reason"],
                    consumer=entry["consumer"],
                    origin=f"allowlist:{scope}",
                )
            )
    return rules


def check_manual_allowlist_consumers(repo_root: Path, path: Path, backend: str) -> list[str]:
    """Report allowlist entries whose ``consumer`` no longer backs up the claim: the file must
    exist and must still mention the key. Line numbers are deliberately not checked -- they rot
    within weeks and a wrong one says nothing about whether the key is still read."""
    if not path.is_file():
        return []
    data = yaml.safe_load(path.read_text()) or {}
    problems = []
    for scope in ("common", backend):
        for entry in data.get(scope) or []:
            consumer = str(entry.get("consumer", ""))
            # Tolerate a legacy "path:line" form by dropping the line number.
            rel = consumer.split(":", 1)[0].strip()
            if not rel:
                continue
            target = repo_root / rel
            if not target.is_file():
                # Submodules are not always checked out; absence proves nothing.
                if rel.startswith("third_party/"):
                    continue
                problems.append(f"{entry['key']}: consumer {rel} does not exist")
                continue
            token = entry["key"].split(".", 1)[0].rstrip("*")
            if token and token not in target.read_text(errors="ignore"):
                problems.append(f"{entry['key']}: {rel} no longer mentions '{token}'")
    return problems


def build_allowlist(repo_root: Path, backend: str, allowlist_path: Path | None = None) -> Allowlist:
    path = allowlist_path or (repo_root / DEFAULT_ALLOWLIST)
    rules = extract_get_param_rules(repo_root, backend)
    rules += extract_args_read_rules(repo_root, backend)
    rules += load_manual_allowlist(path, backend)
    return Allowlist(tuple(rules))


def extend_schema(repo_root: Path, backend: str, schema: BackendSchema) -> BackendSchema:
    """Add the Primus extension sections declared in-tree to an upstream schema."""
    if backend == "torchtitan":
        keys, sections, module_path = extract_custom_args_schema(repo_root)
        return schema.extended(keys, sections, module_path) if module_path else schema
    keys, source = extract_backend_config_extensions(repo_root, backend)
    return schema.extended(keys, (), source) if keys else schema


def build_schema(repo_root: Path, backend: str) -> BackendSchema | None:
    """Upstream schema plus the Primus extension sections declared in-tree."""
    schema = SCHEMA_LOADERS[backend](repo_root)
    return extend_schema(repo_root, backend, schema) if schema is not None else None


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    backend: str
    key: str
    file: str


@dataclass(frozen=True)
class ScopedFinding:
    """A key that exists, but not for the model this config selects."""

    backend: str
    key: str
    file: str
    scope: ModelScope
    actual: str


@dataclass(frozen=True)
class KeyFinding:
    """All occurrences of one unknown key, collapsed into a single row."""

    backend: str
    key: str
    count: int
    files: tuple[str, ...]
    suggestion: str | None


@dataclass(frozen=True)
class ScopedKeyFinding:
    """All occurrences of one misplaced model-scoped key, collapsed into a row."""

    backend: str
    key: str
    scope: ModelScope
    count: int
    files: tuple[str, ...]
    models: tuple[str, ...]

    def sample(self, limit: int = 3) -> str:
        shown = ", ".join(f"`{f}`" for f in self.files[:limit])
        extra = len(self.files) - limit
        return f"{shown} (+{extra} more)" if extra > 0 else shown


def _walk(
    node: Any,
    prefix: str,
    schema: BackendSchema,
    allow: Allowlist,
    out: list[str],
    leaves: set[str] | None = None,
) -> None:
    """Report the shallowest unknown path; never recurse below a bad or known leaf. ``leaves``
    collects every path the walk settled on, known or not: a scoped key is a *known* one."""
    if not isinstance(node, dict):
        return
    for name, value in node.items():
        path = f"{prefix}.{name}" if prefix else str(name)
        if path in schema.sections or (path not in schema.keys and allow.is_prefix(path)):
            _walk(value, path, schema, allow, out, leaves)
            continue
        if leaves is not None:
            leaves.add(path)
        if path in schema.keys or allow.match(path):
            continue
        out.append(path)


def _module_key_sources(doc: dict) -> Iterator[tuple[str, dict, dict]]:
    """Yield ``(framework, keys, module)`` triples for each module of an experiment YAML."""
    for module in (doc.get("modules") or {}).values():
        if not isinstance(module, dict):
            continue
        framework = module.get("framework")
        keys = {k: v for k, v in module.items() if k not in MODULE_RESERVED_KEYS}
        keys.update(module.get("overrides") or {})
        yield framework, keys, module


def _preset_filename(name: Any) -> str | None:
    if not isinstance(name, str) or not name:
        return None
    return name if name.endswith((".yaml", ".yml")) else f"{name}.yaml"


def build_model_preset_index(repo_root: Path, backend: str, key: str) -> dict[str, str | None]:
    """``<preset file name> -> <its value for key>`` for every model preset. A preset missing from
    the index is one the scan could not read, not one that leaves ``key`` unset."""
    index: dict[str, str | None] = {}
    base = repo_root / MODEL_PRESETS / backend
    if not base.is_dir():
        return index
    for path in sorted(p for p in base.rglob("*.y*ml") if p.is_file()):
        try:
            preset = parse_yaml(str(path))
        except (OSError, ValueError, yaml.YAMLError):
            continue
        if not isinstance(preset, dict):
            continue
        value = preset.get(key)
        index[str(path.relative_to(base))] = None if value is None else str(value)
    return index


def _selected_model(
    keys: dict, module: dict | None, scope: ModelScope, presets: Mapping[str, str | None]
) -> str | None:
    """What ``scope.key`` will be at runtime here, or ``None`` when unknowable. An experiment names
    its model by preset and does not repeat the ``model_type`` the preset declares, so the preset
    has to be consulted; an unreadable one leaves the answer unknown and nothing is reported."""
    if scope.key in keys:
        return str(keys[scope.key])
    if module is not None:
        filename = _preset_filename(module.get("model"))
        if filename is None or filename not in presets:
            return None
        declared = presets[filename]
        if declared is not None:
            return declared
    return scope.default


def scan_file(
    path: Path,
    backend: str,
    schema: BackendSchema,
    allow: Allowlist,
    rel: str,
    scopes: Sequence[ModelScope] = (),
    presets: Mapping[str, str | None] | None = None,
) -> tuple[list[Finding], list[ScopedFinding]]:
    doc = parse_yaml(str(path))
    if not isinstance(doc, dict):
        return [], []
    unknown: list[str] = []
    scoped: list[ScopedFinding] = []
    known_presets = presets if presets is not None else {}

    def check_scopes(keys: dict, module: dict | None, leaves: set[str]) -> None:
        for scope in scopes:
            misplaced = sorted(leaves & scope.keys)
            if not misplaced:
                continue
            actual = _selected_model(keys, module, scope, known_presets)
            if actual is None or actual == scope.value:
                continue
            scoped.extend(ScopedFinding(backend, key, rel, scope, actual) for key in misplaced)

    if "modules" in doc:
        envelope = {k: v for k, v in doc.items() if k != "modules"}
        for framework, keys, module in _module_key_sources(doc):
            if framework != backend:
                continue
            leaves: set[str] = set()
            _walk({**envelope, **keys}, "", schema, allow, unknown, leaves)
            check_scopes({**envelope, **keys}, module, leaves)
    else:
        leaves = set()
        _walk(doc, "", schema, allow, unknown, leaves)
        check_scopes(doc, None, leaves)
    # dict.fromkeys: a file with two modules of the same backend must not count twice.
    return (
        [Finding(backend, key, rel) for key in dict.fromkeys(unknown)],
        list(dict.fromkeys(scoped)),
    )


def iter_config_files(repo_root: Path, roots: Iterable[Path]) -> Iterator[Path]:
    for root in roots:
        base = root if root.is_absolute() else repo_root / root
        if base.is_file():
            yield base
        elif base.is_dir():
            yield from sorted(p for p in base.rglob("*.y*ml") if p.is_file())


@dataclass(frozen=True)
class LoadError:
    """A config the scanner could not read, and therefore did **not** check."""

    file: str
    message: str


def _describe_load_error(path: Path, exc: Exception, repo_root: Path) -> str:
    """Name the file that is actually missing, not just the one being scanned: a broken
    ``extends:`` target is the usual cause and the raw error names an un-normalised path."""
    missing = getattr(exc, "filename", None)
    if not missing or Path(missing).resolve() == path.resolve():
        return f"{type(exc).__name__}: {exc}"
    target = Path(missing).resolve()
    try:
        target = target.relative_to(repo_root.resolve())
    except ValueError:
        pass  # outside repo_root (e.g. a submodule checkout); report the absolute path instead
    return f"cannot resolve `{target}` (check the `extends:` paths)"


@dataclass(frozen=True)
class ScanResult:
    backend: str
    schema: BackendSchema | None
    findings: tuple[Finding, ...] = ()
    scanned: int = 0
    errors: tuple[LoadError, ...] = ()
    scoped: tuple[ScopedFinding, ...] = ()
    scopes: tuple[ModelScope, ...] = ()

    @property
    def available(self) -> bool:
        return self.schema is not None

    @property
    def checked(self) -> int:
        """Configs actually inspected. Anything that failed to load is not one."""
        return self.scanned - len(self.errors)


def scan_backend(
    repo_root: Path,
    backend: str,
    roots: Iterable[Path] | None = None,
    allowlist_path: Path | None = None,
) -> ScanResult:
    """Scan one backend; ``ScanResult.available`` is False when the submodule is absent."""
    upstream = SCHEMA_LOADERS[backend](repo_root)
    if upstream is None:
        return ScanResult(backend=backend, schema=None)
    schema = extend_schema(repo_root, backend, upstream)
    allow = build_allowlist(repo_root, backend, allowlist_path)
    scopes = extract_model_scopes(repo_root, backend, upstream, allow)
    # One discriminator per backend, so every scope shares `key`.
    presets = build_model_preset_index(repo_root, backend, scopes[0].key) if scopes else {}
    findings: list[Finding] = []
    scoped: list[ScopedFinding] = []
    errors: list[LoadError] = []
    scanned = 0
    for path in iter_config_files(repo_root, roots if roots is not None else SCAN_ROOTS[backend]):
        scanned += 1
        try:
            rel = str(path.relative_to(repo_root))
        except ValueError:
            rel = str(path)
        # An unreadable config is an unchecked config: record it instead of
        # raising (no traceback in CI) and instead of ignoring it (no silent pass).
        try:
            unknown, misplaced = scan_file(path, backend, schema, allow, rel, scopes, presets)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            errors.append(LoadError(rel, _describe_load_error(path, exc, repo_root)))
            continue
        findings.extend(unknown)
        scoped.extend(misplaced)
    return ScanResult(
        backend=backend,
        schema=schema,
        findings=tuple(findings),
        scanned=scanned,
        errors=tuple(errors),
        scoped=tuple(scoped),
        scopes=scopes,
    )


def dedupe(findings: Sequence[Finding], schema: BackendSchema | None = None) -> list[KeyFinding]:
    """Collapse per-file findings into one row per key: a single upstream rename shows up in
    dozens of configs, and the per-key view is the actionable one. Widest blast radius first, and
    ``(backend, key)`` is unique per row, so the order is total and does not depend on the order
    the findings arrived in."""
    grouped: dict[tuple[str, str], list[str]] = {}
    for finding in findings:
        grouped.setdefault((finding.backend, finding.key), []).append(finding.file)
    rows = [
        KeyFinding(
            backend=backend,
            key=key,
            count=len(files),
            files=tuple(sorted(set(files))),
            suggestion=schema.suggest(key) if schema is not None else None,
        )
        for (backend, key), files in grouped.items()
    ]
    return sorted(rows, key=lambda r: (r.backend, -r.count, r.key))


def dedupe_scoped(findings: Sequence[ScopedFinding]) -> list[ScopedKeyFinding]:
    """Collapse per-file scoped findings into one row per key, carrying the models it was wrongly
    set on: deleting the key and moving it are different fixes."""
    grouped: dict[tuple[str, str, ModelScope], list[ScopedFinding]] = {}
    for finding in findings:
        grouped.setdefault((finding.backend, finding.key, finding.scope), []).append(finding)
    rows = [
        ScopedKeyFinding(
            backend=backend,
            key=key,
            scope=scope,
            count=len(found),
            files=tuple(sorted({f.file for f in found})),
            models=tuple(sorted({f.actual for f in found})),
        )
        for (backend, key, scope), found in grouped.items()
    ]
    return sorted(rows, key=lambda r: (r.backend, -r.count, r.key))
