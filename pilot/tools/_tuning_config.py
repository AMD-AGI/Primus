"""tuning_config.yaml — the universal "where to write" contract.

Counterpart to ``_cluster_config.py``: every Pilot tool that produces stage
artifacts under ``state/<session_id>/trace/<stage>/`` resolves an input
``tuning.yaml`` (the session-wide bootstrap config produced by
``pilot session init``) before doing anything that touches disk.

Resolution priority (caller passes only the first one explicitly):

1. ``--tuning-config <path>`` CLI flag (highest)
2. ``PILOT_TUNING_CONFIG=<path>`` environment variable
3. ``./tuning.yaml`` in the current working directory (lowest)

If none of those resolves, callers MAY proceed in "legacy mode" — output
falls back to each tool's pre-existing default (e.g. ``state/runs``). When a
tool wires ``required=True`` on its end, missing tuning_config is reported as
``failure.kind=TUNING_CONFIG``.

A loaded ``TuningConfig`` exposes:

* :py:meth:`stage_trace_dir` — absolute path to ``<session>/trace/<stage>``
  (or ``<session>/trace/t<trial_id>`` for the optimize_loop pattern).
* :py:meth:`plan_path_abs`, :py:meth:`cluster_config_abs` — resolved input
  paths the tool can use as defaults when the caller didn't override them.
* :py:attr:`base_overrides` — copy of ``target.base_overrides`` so launching
  tools can auto-merge user-supplied plan-level overrides.
* :py:meth:`stage_default` — opaque ``{"iters", "timeout_s"}`` defaults from
  the matching ``stages.<stage>`` block (None if absent).

Everything is read-only; tools never write back into ``tuning.yaml``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pilot.tools._schema import SchemaValidationError, validate


_PILOT_ROOT: Path = Path(__file__).resolve().parent.parent  # tools/ -> pilot/
_PRIMUS_ROOT: Path = _PILOT_ROOT.parent

_ENV_VAR = "PILOT_TUNING_CONFIG"
_DEFAULT_FILENAME = "tuning.yaml"

# Stages with a fixed ``stages.<stage>.dir`` slot. Anything not here either
# uses ``optimize_loop.dir_pattern`` (the only template) or is rejected.
_FIXED_STAGES: frozenset[str] = frozenset({
    "preflight", "projection", "smoke", "baseline",
    "report", "env_sweep", "correctness",
})
_PATTERNED_STAGES: frozenset[str] = frozenset({"optimize_loop"})


class TuningConfigError(Exception):
    """Raised when tuning.yaml resolution, parsing, or use fails.

    ``kind`` mirrors the SubagentResult ``failure.kind`` taxonomy:

    * ``TUNING_CONFIG`` — file missing / malformed / required field absent.
    * ``USAGE``         — caller passed an unknown stage name or forgot a
                          required argument (e.g. trial_id for optimize_loop).
    """

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _yaml():
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise TuningConfigError("TUNING_CONFIG", f"PyYAML required: {exc}") from exc
    return yaml


def _resolve_path_for_load(cli_arg: str | Path | None) -> Path | None:
    """Apply the 3-tier fallback. Returns ``None`` if nothing resolves."""
    if cli_arg:
        return Path(cli_arg).expanduser()
    env = os.environ.get(_ENV_VAR)
    if env:
        return Path(env).expanduser()
    cwd_candidate = Path.cwd() / _DEFAULT_FILENAME
    if cwd_candidate.exists():
        return cwd_candidate
    return None


def load_tuning_config(
    cli_arg: str | Path | None,
    *,
    required: bool = False,
) -> "TuningConfig | None":
    """Resolve and load a TuningConfig from disk.

    Parameters
    ----------
    cli_arg : str | Path | None
        Value from the caller's ``--tuning-config`` flag (or None).
    required : bool, default False
        When True, missing tuning.yaml raises ``TuningConfigError``; when
        False, returns ``None`` so the tool can fall back to legacy behavior.

    Returns
    -------
    TuningConfig | None
        ``None`` only when ``required=False`` and nothing resolves.
    """
    path = _resolve_path_for_load(cli_arg)
    if path is None:
        if required:
            raise TuningConfigError(
                "TUNING_CONFIG",
                "tuning.yaml not provided. Pass --tuning-config <path>, set "
                f"{_ENV_VAR}=<path>, or place tuning.yaml in the current "
                "directory.",
            )
        return None

    if not path.exists():
        raise TuningConfigError(
            "TUNING_CONFIG",
            f"tuning.yaml not found at: {path}",
        )

    raw = path.read_text()
    try:
        data = _yaml().safe_load(raw)
    except Exception as exc:
        raise TuningConfigError(
            "TUNING_CONFIG", f"{path} is not valid YAML: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise TuningConfigError("TUNING_CONFIG", f"{path} must contain a YAML mapping")

    try:
        validate(data, "tuning_config")
    except SchemaValidationError as exc:
        raise TuningConfigError(
            "TUNING_CONFIG", f"{path} fails tuning_config schema: {exc}"
        ) from exc

    return TuningConfig(path=path.resolve(), data=data)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TuningConfig:
    """Read-only view onto a parsed ``tuning.yaml``.

    Construct via :py:func:`load_tuning_config`; never instantiate directly.
    """

    path: Path
    data: dict[str, Any]

    # -- core identifiers ----------------------------------------------------

    @property
    def source_path(self) -> str:
        return str(self.path)

    @property
    def session_id(self) -> str:
        return self.data["session_id"]

    @property
    def session_dir(self) -> Path:
        """Absolute path to ``state/<session_id>/`` (the directory that owns this file)."""
        return self.path.parent.resolve()

    @property
    def log_dir_prefix_abs(self) -> Path:
        """Absolute ``log_dir_prefix`` (typically equals :py:attr:`session_dir`)."""
        prefix = self.data["log_dir_prefix"]
        p = Path(prefix).expanduser()
        return p if p.is_absolute() else (_PILOT_ROOT / p).resolve()

    @property
    def trace_subdir(self) -> str:
        return self.data.get("trace_subdir", "trace")

    # -- inputs (pointers to upstream YAMLs) --------------------------------

    @property
    def plan_ref(self) -> str:
        return self.data["plan"]

    @property
    def cluster_config_ref(self) -> str:
        return self.data["cluster_config"]

    def plan_path_abs(self) -> Path:
        return self._resolve_repo_path(self.plan_ref)

    def cluster_config_abs(self) -> Path:
        return self._resolve_repo_path(self.cluster_config_ref)

    # -- target / overrides --------------------------------------------------

    @property
    def target(self) -> dict[str, Any]:
        return dict(self.data.get("target", {}))

    @property
    def base_overrides(self) -> dict[str, Any]:
        return dict((self.data.get("target") or {}).get("base_overrides") or {})

    # -- per-stage routing ---------------------------------------------------

    def stage_trace_dir(self, stage: str, *, trial_id: int | None = None) -> Path:
        """Return ``<log_dir_prefix>/<trace_subdir>/<stage>`` as an absolute path.

        For ``stage='optimize_loop'`` the result is
        ``<log_dir_prefix>/<trace_subdir>/t<trial_id>``; ``trial_id`` MUST be
        a positive integer.
        """
        stages = self.data.get("stages") or {}
        if stage in _PATTERNED_STAGES:
            if trial_id is None or trial_id <= 0:
                raise TuningConfigError(
                    "USAGE",
                    f"stage={stage!r} requires a positive --trial-id",
                )
            block = stages.get(stage) or {}
            pattern = block.get("dir_pattern")
            if not pattern or "{trial_id}" not in pattern:
                raise TuningConfigError(
                    "TUNING_CONFIG",
                    f"stages.{stage}.dir_pattern must contain '{{trial_id}}'",
                )
            sub = pattern.format(trial_id=trial_id)
        elif stage in _FIXED_STAGES:
            block = stages.get(stage)
            if not isinstance(block, dict) or "dir" not in block:
                raise TuningConfigError(
                    "TUNING_CONFIG",
                    f"stages.{stage}.dir missing in {self.path}",
                )
            sub = block["dir"]
        else:
            raise TuningConfigError(
                "USAGE",
                f"unknown stage: {stage!r}. Allowed: "
                f"{sorted(_FIXED_STAGES | _PATTERNED_STAGES)}",
            )
        return (self.log_dir_prefix_abs / sub).resolve()

    def stage_default(self, stage: str) -> dict[str, Any]:
        """Return per-stage non-routing defaults (iters / timeout_s / ...).

        Empty dict when nothing is declared.
        """
        block = (self.data.get("stages") or {}).get(stage) or {}
        if not isinstance(block, dict):
            return {}
        return {k: v for k, v in block.items() if k not in ("dir", "dir_pattern")}

    # -- helpers -------------------------------------------------------------

    def _resolve_repo_path(self, ref: str) -> Path:
        p = Path(ref).expanduser()
        if p.is_absolute():
            return p
        # Repo-rooted (e.g. `examples/...` or `pilot/cluster.yaml`).
        return (_PRIMUS_ROOT / p).resolve()


# ---------------------------------------------------------------------------
# CLI failure helper (mirrors `cluster_config_failure` in shape)
# ---------------------------------------------------------------------------


def tuning_config_failure(exc: TuningConfigError, *, stage: str) -> dict[str, Any]:
    """Build a JSON failure payload for a CLI ``--tuning-config`` error.

    ``stage`` is the upstream tool's ``stage`` field (e.g. ``"SUBMIT"``)
    so the orchestrator can route the failure correctly.
    """
    return {
        "stage": stage,
        "status": "failed",
        "failure": {
            "kind": exc.kind,
            "message": str(exc),
            "escalate_to_orchestrator": True,
        },
    }


# ---------------------------------------------------------------------------
# Argparse helpers — DRY across the dozen tools that need this flag
# ---------------------------------------------------------------------------


def add_tuning_config_arg(parser, *, required: bool = False) -> None:
    """Attach ``--tuning-config`` to ``parser``.

    Tools wire ``required=True`` once the migration off the legacy log_dir
    default is complete. During the cut-over both modes coexist.
    """
    parser.add_argument(
        "--tuning-config",
        default=None,
        required=required,
        help=(
            "Path to the session's tuning.yaml. Resolution order: this flag → "
            f"${_ENV_VAR} → ./tuning.yaml. When set, log dirs / plan / "
            "cluster_config / base_overrides default to the values declared "
            "inside it (per-call CLI overrides still win)."
        ),
    )


def add_stage_arg(parser, *, required: bool = False) -> None:
    """Attach ``--stage`` for tools that need to know which trace dir to use."""
    parser.add_argument(
        "--stage",
        default=None,
        required=required,
        choices=sorted(_FIXED_STAGES | _PATTERNED_STAGES),
        help=(
            "Stage name (e.g. smoke / baseline / optimize_loop). Required "
            "when --tuning-config is provided so the tool can pick the right "
            "trace/<stage> dir."
        ),
    )


def add_trial_id_arg(parser) -> None:
    parser.add_argument(
        "--trial-id",
        default=None,
        type=int,
        help="Positive integer for stage=optimize_loop (trace/t<id>). Ignored otherwise.",
    )
