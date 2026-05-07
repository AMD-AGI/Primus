"""Universal Pilot tool input: cluster.yaml loader, LaunchPlan, fast-fail checks.

Every Pilot tool (`pilot.tools.preflight`, `pilot.tools.submit`, `pilot.tools.observe`,
`pilot.tools.constraint`, ...) consumes exactly one mandatory environmental input:
a `cluster.yaml` file conforming to ``schemas/cluster_config.schema.json``.

This module is the single shared implementation for:

  1. Resolving the cluster.yaml path
       (``--cluster-config`` arg > ``$PILOT_CLUSTER_CONFIG`` > ``./cluster.yaml``).
  2. Schema-validating its content (hand-rolled, no jsonschema dep).
  3. Translating it to a ``LaunchPlan`` (mode + nodes + rdzv).
  4. Running the three universal fast-fail checks:
       - cluster.yaml present & valid
       - SLURM allocation alive (mode=slurm only)
       - At least one GPU visible to the current process

See ``AGENTS.md`` §4, ``SETUP.md``, and ``schemas/cluster_config.schema.json``.

Errors raised here all subclass ``ClusterConfigError``; the CLI dispatcher in each
tool maps them to ``failure.kind=CLUSTER`` SubagentResult JSON.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ClusterConfigError(Exception):
    """Base error for cluster.yaml resolution / validation / preflight checks.

    The CLI dispatcher should map this to ``failure.kind=CLUSTER`` and exit
    with a non-zero code while still emitting valid SubagentResult JSON.
    """

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.hint = hint

    def to_message(self) -> str:
        if self.hint:
            return f"{self.args[0]} -- hint: {self.hint}"
        return str(self.args[0])


class ClusterConfigNotFound(ClusterConfigError):
    """Raised when no cluster.yaml can be resolved from the three search locations."""


class ClusterConfigSchemaError(ClusterConfigError):
    """Raised when cluster.yaml exists but does not conform to the schema."""


class SlurmAllocationStaleError(ClusterConfigError):
    """Raised when ``slurm.job_id`` does not point at a RUNNING allocation."""


class NoGpuVisibleError(ClusterConfigError):
    """Raised when no GPU is visible to the current process."""


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ClusterConfig:
    """Parsed contents of cluster.yaml. Validated against the schema."""

    schema_version: str
    cluster_id: str
    mode: Literal["single", "slurm"]
    slurm: dict[str, Any] | None = None
    single: dict[str, Any] | None = None
    runtime: dict[str, Any] = field(default_factory=dict)
    source_path: str | None = None  # absolute path the config was loaded from


@dataclass
class LaunchPlan:
    """Concrete execution plan derived from ClusterConfig + scontrol query.

    Tools only see this; they never re-read cluster.yaml or query SLURM directly.
    """

    mode: Literal["single", "slurm"]
    cluster_id: str
    nnodes: int                    # 1 if single
    nodelist: list[str]            # ["localhost"] if single; expanded hosts otherwise
    head_host: str                 # nodelist[0]
    rdzv_endpoint: str             # f"{head_host}:{rdzv_port}"; for single, port=0 (auto)
    rdzv_id: str                   # f"pf_{slurm_job_id}" or f"local_{cluster_id}_{ts}"
    slurm_job_id: int | None       # for srun --jobid=...; None when mode=single
    rdzv_port: int                 # 29400 default for slurm; 0 for single (auto)
    image_label: str | None        # from runtime.image_label, used in cache filenames

    @property
    def is_multi_node(self) -> bool:
        return self.nnodes > 1


# ---------------------------------------------------------------------------
# Resolution & loading
# ---------------------------------------------------------------------------


def resolve_cluster_config_path(explicit: str | None = None) -> Path:
    """Locate cluster.yaml using the three-tier priority.

    Priority:
      1. ``explicit`` argument (typically from ``--cluster-config``)
      2. ``$PILOT_CLUSTER_CONFIG`` environment variable
      3. ``./cluster.yaml`` in the current working directory

    Returns the resolved absolute Path, or raises ``ClusterConfigNotFound``.
    """
    candidates: list[tuple[str, str | None]] = [
        ("--cluster-config", explicit),
        ("$PILOT_CLUSTER_CONFIG", os.environ.get("PILOT_CLUSTER_CONFIG")),
        ("./cluster.yaml", "cluster.yaml"),
    ]
    tried: list[str] = []
    for source, value in candidates:
        if not value:
            continue
        path = Path(value).expanduser().resolve()
        tried.append(f"{source} -> {path}")
        if path.is_file():
            return path
    raise ClusterConfigNotFound(
        "no cluster.yaml could be resolved",
        hint=(
            "set --cluster-config <path>, export PILOT_CLUSTER_CONFIG, or place "
            "cluster.yaml in the current directory. See pilot/SETUP.md. "
            f"Tried: {tried}"
        ),
    )


def load_cluster_config(explicit: str | None = None) -> ClusterConfig:
    """Resolve, parse, and schema-validate cluster.yaml in one shot.

    Raises ``ClusterConfigNotFound`` or ``ClusterConfigSchemaError``.
    """
    path = resolve_cluster_config_path(explicit)
    try:
        import yaml  # local import: only required when invoked
    except ImportError as exc:  # pragma: no cover
        raise ClusterConfigError(
            f"PyYAML not available: {exc}",
            hint="pip install pyyaml",
        ) from exc

    try:
        with path.open() as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ClusterConfigSchemaError(
            f"cluster.yaml ({path}) is not valid YAML: {exc}",
            hint="run `python -c 'import yaml; yaml.safe_load(open(...))'` to find the syntax error",
        ) from exc

    if not isinstance(raw, dict):
        raise ClusterConfigSchemaError(
            f"cluster.yaml ({path}) must be a YAML mapping at the top level, got {type(raw).__name__}",
        )

    _validate_schema(raw, source=str(path))

    return ClusterConfig(
        schema_version=raw["schema_version"],
        cluster_id=raw["cluster_id"],
        mode=raw["mode"],
        slurm=raw.get("slurm"),
        single=raw.get("single"),
        runtime=raw.get("runtime", {}) or {},
        source_path=str(path),
    )


# ---------------------------------------------------------------------------
# Schema validation (hand-rolled to avoid jsonschema dep)
# ---------------------------------------------------------------------------


_ALLOWED_TOP_KEYS = {"schema_version", "cluster_id", "mode", "slurm", "single", "runtime"}
_ALLOWED_SLURM_KEYS = {"job_id", "nnodes", "nodelist", "partition", "rdzv_port"}
_ALLOWED_SINGLE_KEYS = {"max_local_gpus"}
_ALLOWED_RUNTIME_KEYS = {"image_label", "notes"}


def _validate_schema(raw: dict[str, Any], *, source: str) -> None:
    """Validate against schemas/cluster_config.schema.json (hand-rolled subset)."""
    extras = set(raw.keys()) - _ALLOWED_TOP_KEYS
    if extras:
        raise ClusterConfigSchemaError(
            f"{source}: unknown top-level keys {sorted(extras)}",
            hint=f"allowed keys: {sorted(_ALLOWED_TOP_KEYS)}",
        )

    for key in ("schema_version", "cluster_id", "mode"):
        if key not in raw:
            raise ClusterConfigSchemaError(f"{source}: missing required key '{key}'")

    if raw["schema_version"] != "1.0":
        raise ClusterConfigSchemaError(
            f"{source}: schema_version must be '1.0', got {raw['schema_version']!r}",
        )

    cid = raw["cluster_id"]
    if not isinstance(cid, str) or not cid:
        raise ClusterConfigSchemaError(f"{source}: cluster_id must be a non-empty string")
    if not _is_valid_cluster_id(cid):
        raise ClusterConfigSchemaError(
            f"{source}: cluster_id {cid!r} contains invalid characters",
            hint="must match ^[a-zA-Z0-9][a-zA-Z0-9_-]*$, max 64 chars",
        )

    mode = raw["mode"]
    if mode not in ("single", "slurm"):
        raise ClusterConfigSchemaError(
            f"{source}: mode must be 'single' or 'slurm', got {mode!r}",
        )

    if mode == "slurm":
        if "slurm" not in raw or not isinstance(raw["slurm"], dict):
            raise ClusterConfigSchemaError(
                f"{source}: mode=slurm requires a `slurm:` block with at least job_id",
            )
        _validate_slurm_block(raw["slurm"], source=source)
    else:
        if "slurm" in raw:
            raise ClusterConfigSchemaError(
                f"{source}: mode=single must not declare a `slurm:` block",
            )
        if "single" in raw:
            _validate_single_block(raw["single"], source=source)

    if "runtime" in raw and raw["runtime"] is not None:
        _validate_runtime_block(raw["runtime"], source=source)


def _is_valid_cluster_id(s: str) -> bool:
    if not s or len(s) > 64:
        return False
    if not (s[0].isalnum()):
        return False
    return all(c.isalnum() or c in "_-" for c in s)


def _validate_slurm_block(block: dict[str, Any], *, source: str) -> None:
    extras = set(block.keys()) - _ALLOWED_SLURM_KEYS
    if extras:
        raise ClusterConfigSchemaError(
            f"{source}: unknown keys in slurm block: {sorted(extras)}",
            hint=f"allowed: {sorted(_ALLOWED_SLURM_KEYS)}",
        )
    if "job_id" not in block:
        raise ClusterConfigSchemaError(f"{source}: slurm.job_id is required")
    job_id = block["job_id"]
    if not isinstance(job_id, (int, str)):
        raise ClusterConfigSchemaError(
            f"{source}: slurm.job_id must be int or str, got {type(job_id).__name__}",
        )
    try:
        int(job_id)
    except (TypeError, ValueError) as exc:
        raise ClusterConfigSchemaError(
            f"{source}: slurm.job_id {job_id!r} is not an integer",
        ) from exc

    if "nnodes" in block and (
        not isinstance(block["nnodes"], int) or block["nnodes"] < 1
    ):
        raise ClusterConfigSchemaError(
            f"{source}: slurm.nnodes must be a positive integer",
        )
    if "nodelist" in block and not isinstance(block["nodelist"], str):
        raise ClusterConfigSchemaError(f"{source}: slurm.nodelist must be a string")
    if "rdzv_port" in block:
        port = block["rdzv_port"]
        if not isinstance(port, int) or port < 1024 or port > 65535:
            raise ClusterConfigSchemaError(
                f"{source}: slurm.rdzv_port must be 1024..65535, got {port}",
            )


def _validate_single_block(block: dict[str, Any], *, source: str) -> None:
    extras = set(block.keys()) - _ALLOWED_SINGLE_KEYS
    if extras:
        raise ClusterConfigSchemaError(
            f"{source}: unknown keys in single block: {sorted(extras)}",
            hint=f"allowed: {sorted(_ALLOWED_SINGLE_KEYS)}",
        )
    if "max_local_gpus" in block:
        v = block["max_local_gpus"]
        if not isinstance(v, int) or v < 1:
            raise ClusterConfigSchemaError(
                f"{source}: single.max_local_gpus must be a positive integer",
            )


def _validate_runtime_block(block: Any, *, source: str) -> None:
    if not isinstance(block, dict):
        raise ClusterConfigSchemaError(
            f"{source}: runtime block must be a mapping (got {type(block).__name__})",
        )
    extras = set(block.keys()) - _ALLOWED_RUNTIME_KEYS
    if extras:
        raise ClusterConfigSchemaError(
            f"{source}: unknown keys in runtime block: {sorted(extras)}",
            hint=f"allowed: {sorted(_ALLOWED_RUNTIME_KEYS)}",
        )
    label = block.get("image_label")
    if label is not None:
        if not isinstance(label, str) or len(label) > 64 or not label:
            raise ClusterConfigSchemaError(
                f"{source}: runtime.image_label must be a non-empty string ≤ 64 chars",
            )


# ---------------------------------------------------------------------------
# SLURM helpers (only used when mode=slurm)
# ---------------------------------------------------------------------------


@dataclass
class SlurmJobInfo:
    job_id: int
    job_state: str
    nnodes: int
    nodelist: str
    partition: str | None = None


def scontrol_show_job(job_id: int | str, *, timeout_s: int = 10) -> SlurmJobInfo:
    """Query `scontrol show job <id>` and return the parsed key fields.

    Raises:
        ClusterConfigError: scontrol unavailable / errored / job not found.
    """
    if not shutil.which("scontrol"):
        raise ClusterConfigError(
            "`scontrol` not found on PATH",
            hint="cluster.yaml declares mode=slurm but the SLURM client is not installed in this environment",
        )
    try:
        r = subprocess.run(
            ["scontrol", "show", "job", str(job_id)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise ClusterConfigError(
            f"scontrol timed out after {timeout_s}s",
        ) from exc

    if r.returncode != 0:
        raise SlurmAllocationStaleError(
            f"scontrol show job {job_id} failed (rc={r.returncode}): {r.stderr.strip()[:200]}",
            hint="job id may be wrong or already terminated; check `squeue` and re-salloc if needed",
        )

    fields = _parse_scontrol_kv(r.stdout)
    state = fields.get("JobState")
    if state is None:
        raise ClusterConfigError(
            f"scontrol output missing JobState field; raw: {r.stdout[:200]}",
        )

    return SlurmJobInfo(
        job_id=int(job_id),
        job_state=state,
        nnodes=int(fields.get("NumNodes", "0")),
        nodelist=fields.get("NodeList", ""),
        partition=fields.get("Partition"),
    )


def _parse_scontrol_kv(text: str) -> dict[str, str]:
    """Parse scontrol's space-separated `Key=Value` output into a flat dict.

    Handles values containing `=` correctly by splitting on first `=` only.
    Values containing whitespace (rare in scontrol output) are not supported;
    SLURM avoids them by quoting or replacing with `_`.
    """
    out: dict[str, str] = {}
    for tok in text.split():
        if "=" not in tok:
            continue
        k, _, v = tok.partition("=")
        if k and k not in out:
            out[k] = v
    return out


def expand_slurm_nodelist(nodelist: str) -> list[str]:
    """Expand a SLURM nodelist string into individual hostnames.

    Handles common patterns:
      - "node-1"               -> ["node-1"]
      - "node-[1-3]"           -> ["node-1", "node-2", "node-3"]
      - "node-[1,3,5]"         -> ["node-1", "node-3", "node-5"]
      - "node-[01-04]"         -> ["node-01", "node-02", "node-03", "node-04"]
      - "smc[01-02],aux[1-2]"  -> ["smc01", "smc02", "aux1", "aux2"]

    Falls back to ``scontrol show hostnames`` if the input contains anything
    we can't parse.
    """
    if not nodelist:
        return []

    # Hand parser for the common patterns above.
    try:
        return _expand_nodelist_handparsed(nodelist)
    except ValueError:
        pass

    # Fallback: ask scontrol.
    if shutil.which("scontrol"):
        try:
            r = subprocess.run(
                ["scontrol", "show", "hostnames", nodelist],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if r.returncode == 0:
                return [h for h in r.stdout.splitlines() if h.strip()]
        except subprocess.TimeoutExpired:
            pass

    raise ClusterConfigError(
        f"unable to expand SLURM nodelist {nodelist!r}",
        hint="ensure scontrol is available, or pre-expand `slurm.nodelist` to comma-separated hostnames",
    )


def _expand_nodelist_handparsed(nodelist: str) -> list[str]:
    out: list[str] = []
    # Split on commas that are NOT inside [...]
    groups: list[str] = []
    depth = 0
    cur = ""
    for ch in nodelist:
        if ch == "[":
            depth += 1
            cur += ch
        elif ch == "]":
            depth -= 1
            cur += ch
        elif ch == "," and depth == 0:
            groups.append(cur)
            cur = ""
        else:
            cur += ch
    if cur:
        groups.append(cur)

    for grp in groups:
        if "[" not in grp:
            out.append(grp.strip())
            continue
        prefix, _, rest = grp.partition("[")
        if not rest.endswith("]"):
            raise ValueError(f"unbalanced brackets in {grp!r}")
        body = rest[:-1]
        for piece in body.split(","):
            piece = piece.strip()
            if "-" in piece:
                lo, hi = piece.split("-", 1)
                width = max(len(lo), len(hi))
                try:
                    for n in range(int(lo), int(hi) + 1):
                        out.append(f"{prefix}{str(n).zfill(width)}")
                except ValueError as exc:
                    raise ValueError(f"non-integer range in {grp!r}") from exc
            else:
                out.append(f"{prefix}{piece}")
    return out


# ---------------------------------------------------------------------------
# LaunchPlan derivation
# ---------------------------------------------------------------------------


def to_launch_plan(cfg: ClusterConfig) -> LaunchPlan:
    """Convert a validated ClusterConfig into the executable LaunchPlan.

    For mode=slurm this calls ``scontrol show job`` to confirm the allocation
    is RUNNING and to fill in nnodes/nodelist if the user didn't specify them.

    Raises ``SlurmAllocationStaleError`` if the SLURM job is not RUNNING.
    """
    image_label = cfg.runtime.get("image_label") if cfg.runtime else None

    if cfg.mode == "single":
        return LaunchPlan(
            mode="single",
            cluster_id=cfg.cluster_id,
            nnodes=1,
            nodelist=["localhost"],
            head_host="localhost",
            rdzv_endpoint="127.0.0.1:0",  # auto port via c10d
            rdzv_id=f"local_{cfg.cluster_id}",
            slurm_job_id=None,
            rdzv_port=0,
            image_label=image_label,
        )

    # mode=slurm
    assert cfg.slurm is not None  # schema validator guarantees this
    job_id = int(cfg.slurm["job_id"])

    info = scontrol_show_job(job_id)

    if info.job_state != "RUNNING":
        raise SlurmAllocationStaleError(
            f"slurm job {job_id} is in state {info.job_state}, not RUNNING",
            hint="re-salloc and update slurm.job_id in cluster.yaml",
        )

    declared_nnodes = cfg.slurm.get("nnodes")
    if declared_nnodes is not None and declared_nnodes != info.nnodes:
        raise ClusterConfigError(
            f"slurm.nnodes={declared_nnodes} in cluster.yaml does not match scontrol's "
            f"NumNodes={info.nnodes} for job {job_id}",
            hint="remove slurm.nnodes (Pilot will read it from scontrol) or update it",
        )
    nnodes = info.nnodes

    declared_nodelist = cfg.slurm.get("nodelist")
    nodelist_str = declared_nodelist or info.nodelist
    if (
        declared_nodelist is not None
        and info.nodelist
        and declared_nodelist != info.nodelist
    ):
        raise ClusterConfigError(
            f"slurm.nodelist={declared_nodelist!r} does not match scontrol's "
            f"NodeList={info.nodelist!r}",
            hint="remove slurm.nodelist or update it",
        )
    nodelist = expand_slurm_nodelist(nodelist_str)
    if len(nodelist) != nnodes:
        raise ClusterConfigError(
            f"expanded nodelist has {len(nodelist)} hosts but NumNodes={nnodes}",
            hint=f"raw nodelist: {nodelist_str!r}; expanded: {nodelist}",
        )

    head_host = nodelist[0]
    rdzv_port = int(cfg.slurm.get("rdzv_port", 29400))

    return LaunchPlan(
        mode="slurm",
        cluster_id=cfg.cluster_id,
        nnodes=nnodes,
        nodelist=nodelist,
        head_host=head_host,
        rdzv_endpoint=f"{head_host}:{rdzv_port}",
        rdzv_id=f"pf_{job_id}",
        slurm_job_id=job_id,
        rdzv_port=rdzv_port,
        image_label=image_label,
    )


# ---------------------------------------------------------------------------
# GPU visibility check
# ---------------------------------------------------------------------------


def count_visible_gpus() -> int:
    """Count GPUs visible to the current process.

    Honors ``HIP_VISIBLE_DEVICES`` / ``CUDA_VISIBLE_DEVICES``. Returns 0 when
    neither rocm-smi nor nvidia-smi can find any device.
    """
    visible_env = os.environ.get("HIP_VISIBLE_DEVICES") or os.environ.get(
        "CUDA_VISIBLE_DEVICES"
    )
    if visible_env is not None:
        if visible_env.strip() == "":
            return 0
        try:
            return len([x for x in visible_env.split(",") if x.strip()])
        except ValueError:
            pass  # fall through to smi probe

    if shutil.which("rocm-smi"):
        try:
            r = subprocess.run(
                ["rocm-smi", "--showid"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if r.returncode == 0:
                return sum(1 for ln in r.stdout.splitlines() if "Device Name" in ln)
        except subprocess.TimeoutExpired:
            pass

    if shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(
                ["nvidia-smi", "-L"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if r.returncode == 0:
                return sum(1 for ln in r.stdout.splitlines() if ln.startswith("GPU "))
        except subprocess.TimeoutExpired:
            pass

    return 0


# ---------------------------------------------------------------------------
# Top-level: load + 3 fast-fail checks → LaunchPlan
# ---------------------------------------------------------------------------


def preflight_check(explicit_cluster_config: str | None = None) -> tuple[ClusterConfig, LaunchPlan]:
    """Run the three universal fast-fail checks and return the ready LaunchPlan.

    Used by every Pilot tool's CLI dispatcher as the *first* action, before
    any measurement, state mutation, or subprocess spawn.

    Checks (in order, short-circuiting):
      1. cluster.yaml resolves and validates against the schema
         → raises ``ClusterConfigNotFound`` / ``ClusterConfigSchemaError``
      2. mode=slurm: scontrol confirms allocation is RUNNING and matches
         → raises ``SlurmAllocationStaleError`` / ``ClusterConfigError``
      3. At least one GPU visible
         → raises ``NoGpuVisibleError``

    None of these consume tuning rounds; they are infrastructure validation.

    Returns:
        (ClusterConfig, LaunchPlan): the validated config and its execution plan.
    """
    cfg = load_cluster_config(explicit_cluster_config)

    plan = to_launch_plan(cfg)  # may raise SlurmAllocationStaleError

    n_gpu = count_visible_gpus()
    if n_gpu < 1:
        raise NoGpuVisibleError(
            "no GPU visible to this process",
            hint=(
                "check container GPU passthrough: --device=/dev/kfd /dev/dri (ROCm) "
                "or --gpus all (NVIDIA). For SLURM mode also confirm GRES allocation."
            ),
        )

    return cfg, plan


# ---------------------------------------------------------------------------
# Convenience: SubagentResult-shaped failure dict (for tools to emit)
# ---------------------------------------------------------------------------


def cluster_config_failure(
    exc: ClusterConfigError, *, stage: str = "PREFLIGHT"
) -> dict[str, Any]:
    """Convert a ClusterConfigError into a SubagentResult-shaped failure dict.

    Tools should ``except ClusterConfigError as exc:`` and ``json.dump`` this.
    """
    return {
        "stage": stage,
        "status": "failed",
        "failure": {
            "kind": "CLUSTER",
            "message": exc.to_message(),
            "escalate_to_orchestrator": True,
        },
    }
