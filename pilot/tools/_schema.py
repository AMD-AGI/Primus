"""pilot.tools._schema — single call site for JSON Schema validation.

All Pilot output artifacts (`ClusterProfile`, `RunHandle`, `RunSnapshot`,
`TuningReport`, ...) are persisted as YAML/JSON and validated against a schema
under `pilot/schemas/`. This module exposes one validate-or-die function that
every tool should call before writing the artifact to disk, so that broken
artifacts never escape the tool that produced them.

Design choices
--------------

* Schemas are loaded **once and cached** at first use; subsequent calls reuse
  the compiled validator.
* Validation always raises ``SchemaValidationError`` (Pilot-typed) on failure;
  callers can catch a single exception and translate it to a SubagentResult
  ``failure.kind=TOOL_ERROR`` payload.
* The error message includes (a) which schema, (b) JSON path of the bad
  field, (c) which validator failed (e.g. ``required``, ``type``,
  ``enum``), and (d) the offending value truncated to 200 chars.
* On systems without ``jsonschema`` installed we degrade to a no-op with a
  one-time warning so partial environments don't lose all functionality
  (Pilot already requires PyYAML, so this is a soft fallback rather than a
  hard dependency).
"""

from __future__ import annotations

import json
import sys
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any


_PILOT_ROOT: Path = Path(__file__).resolve().parent.parent  # tools/ -> pilot/
_SCHEMA_DIR: Path = _PILOT_ROOT / "schemas"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SchemaValidationError(Exception):
    """Raised when an artifact fails schema validation.

    Attributes
    ----------
    schema_name : str
        The schema id (e.g. ``"cluster_profile"``).
    json_path : str
        JSON Pointer-ish path to the offending field (e.g.
        ``"/rccl_baseline/intra_node/per_node/n0/sizes_mb"``).
    validator : str
        The jsonschema validator that failed (``required``, ``type``, ...).
    detail : str
        Human-readable detail message.
    """

    def __init__(
        self,
        schema_name: str,
        json_path: str,
        validator: str,
        detail: str,
    ) -> None:
        self.schema_name = schema_name
        self.json_path = json_path
        self.validator = validator
        self.detail = detail
        super().__init__(self._format())

    def _format(self) -> str:
        return (
            f"[schema={self.schema_name}] {self.validator} failed at "
            f"{self.json_path or '<root>'}: {self.detail}"
        )


# ---------------------------------------------------------------------------
# Schema loading (cached)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=32)
def _load_schema(schema_name: str) -> dict[str, Any]:
    """Load ``pilot/schemas/<schema_name>.schema.json`` from disk.

    ``schema_name`` is the bare name (no extension), e.g. ``"run_handle"``.
    """
    path = _SCHEMA_DIR / f"{schema_name}.schema.json"
    if not path.exists():
        raise FileNotFoundError(f"schema not found: {path}")
    with path.open() as f:
        return json.load(f)


@lru_cache(maxsize=32)
def _build_validator(schema_name: str):  # type: ignore[no-untyped-def]
    """Compile a jsonschema validator for ``schema_name``.

    Returns ``None`` if jsonschema isn't installed (caller falls back to
    no-op + warning).
    """
    try:
        import jsonschema  # type: ignore
        from jsonschema import Draft202012Validator
    except ImportError:
        return None
    schema = _load_schema(schema_name)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


_warned_missing_jsonschema = False


def _warn_jsonschema_missing_once() -> None:
    global _warned_missing_jsonschema
    if _warned_missing_jsonschema:
        return
    _warned_missing_jsonschema = True
    warnings.warn(
        "jsonschema is not installed; pilot artifact validation is disabled. "
        "Install with `pip install jsonschema>=4` to re-enable.",
        RuntimeWarning,
        stacklevel=3,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate(
    instance: Any,
    schema_name: str,
    *,
    soft: bool = False,
) -> None:
    """Validate ``instance`` against ``pilot/schemas/<schema_name>.schema.json``.

    Parameters
    ----------
    instance : Any
        The Python object (typically a ``dict`` parsed from YAML) to check.
    schema_name : str
        Bare schema name (e.g. ``"run_handle"``).
    soft : bool, default False
        When True, validation failures only emit a warning (handy during
        bring-up of a new schema). False raises ``SchemaValidationError`` so
        broken artifacts never reach disk.

    Raises
    ------
    SchemaValidationError
        On validation failure when ``soft=False``.
    FileNotFoundError
        If the schema file itself is missing (always hard error).
    """
    validator = _build_validator(schema_name)
    if validator is None:
        _warn_jsonschema_missing_once()
        return

    errors = sorted(validator.iter_errors(instance), key=lambda e: list(e.absolute_path))
    if not errors:
        return

    err = errors[0]
    json_path = "/" + "/".join(str(p) for p in err.absolute_path)
    if json_path == "/":
        json_path = ""
    detail = err.message
    bad = repr(err.instance)
    if len(bad) > 200:
        bad = bad[:197] + "..."
    full_detail = f"{detail} (offending value: {bad})"

    exc = SchemaValidationError(
        schema_name=schema_name,
        json_path=json_path,
        validator=err.validator,
        detail=full_detail,
    )
    if soft:
        warnings.warn(str(exc), RuntimeWarning, stacklevel=2)
        return
    raise exc


def validate_yaml_path(
    path: str | Path,
    schema_name: str,
    *,
    soft: bool = False,
) -> None:
    """Convenience: load YAML from ``path`` and validate.

    Useful for the CLI ``pilot-validate`` shim and for tests.
    """
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(f"PyYAML required for validate_yaml_path: {exc}")
    with Path(path).open() as f:
        data = yaml.safe_load(f)
    validate(data, schema_name, soft=soft)


# ---------------------------------------------------------------------------
# CLI shim — `python -m pilot.tools._schema <schema> <yaml>`
# ---------------------------------------------------------------------------


def _cli() -> int:
    if len(sys.argv) != 3:
        print("usage: python -m pilot.tools._schema <schema_name> <yaml_path>",
              file=sys.stderr)
        return 2
    schema_name, path = sys.argv[1], sys.argv[2]
    try:
        validate_yaml_path(path, schema_name)
    except SchemaValidationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3
    print(f"OK: {path} conforms to {schema_name}.schema.json")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
