###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Route the diffusion recipe's run tracking to MLflow.

AutoModel already ships the machinery (``MLflowConfig.build`` starts or resumes
the run, logs the config and installs a FAILED excepthook; ``mlflow_utils``
supplies the KILLED path), and ``RecipeConfig.mlflow`` already exposes a typed
``mlflow:`` block. The LLM and VLM recipes use all of it; the diffusion recipe
does not.

This patch closes that gap by calling what is already there, and adds the
three things upstream has no opinion about:

* **Provenance** -- the environment Primus runs in (ROCm/RCCL versions, the
  parallelism knobs, the node list). Upstream logs the recipe config; it does
  not know what a training image is.
* **A KILLED handler** -- ``MLflowConfig.build`` installs the FAILED excepthook
  itself, but the SIGTERM path is wired up per-recipe, and the diffusion recipe
  has no such wiring.
* **End-of-run artifacts** -- the logs, profiler traces, TraceLens reports and
  run summary that turn a tracked number into a reproducible one. Uploaded
  after the train loop returns, so nothing here is inside the timed region.
  Ported from the Megatron path rather than shared with it; see the comment
  above ``_TRACE_PATTERNS`` for why sharing is not possible.

Gated on ``primus_mlflow.enable``, which is false by default: a Primus run must
not start contacting a tracking server because this module exists. The gate is
the patch ``condition`` rather than an early return, so a disabled run is
reported as "skipped (condition not met)" in the run log. An early return would
instead log "✓ Applied" and count the patch in the run's applied list, claiming
a run was tracked when nothing was.

Why the loop is wrapped rather than split across two phases
-----------------------------------------------------------
``AutoModelPretrainTrainer`` dispatches ``run_patches`` for ``before_train``
only; no caller in this backend runs the ``after_train`` phase. Teardown
therefore cannot live in an ``after_train`` patch -- it would register and never
fire. Wrapping ``run_train_validation_loop`` keeps setup and teardown in one
place and puts teardown in a ``finally``, which is also what makes the status
reporting below honest. ``torch_profiler_patches`` wraps the same method for
the same reason.

Two config blocks, deliberately
-------------------------------
``mlflow:`` is an AutoModel-native section consumed by ``RecipeConfig.mlflow``
(experiment name, tracking URI, tags, resume). It must reach AutoModel, so it is
*not* a ``primus_`` section, and it must contain only ``MLflowConfig`` fields --
an unknown key there is a ``TypeError`` at config build.

``primus_mlflow:`` holds what AutoModel has no concept of (the gate, the
per-step policy, the environment allow-list) and is stripped before the config
reaches AutoModel.
"""

from __future__ import annotations

import logging
import math
import os
import re
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from primus.backends.automodel import options
from primus.core.patches import PatchContext, register_patch

logger = logging.getLogger(__name__)

_SECTION = "primus_mlflow"
_INSTALLED_FLAG = "_primus_mlflow_installed"

# Experiment directory of the latest install(), i.e. the current trainer's.
# Logs and traces are found relative to it at teardown. Module-level for the
# same reason torch_profiler keeps one: the wrapper closes over the function,
# not over the patch context, and the context is gone by the time it runs.
_exp_root: Optional[str] = None

options.declare(
    _SECTION,
    "enable",
    "per_step",
    "env_allowlist",
    "env_allowlist_extra",
    "upload_logs",
    "upload_traces",
    "upload_tracelens_report",
    "tracelens_ranks",
    "tracelens_output_format",
    "upload_max_bytes",
    "summary",
    "logged_model_name",
    "heartbeat_seconds",
)

# Environment captured as run params. An allow-list, not a deny-list.
# A deny-list fails open on every variable nobody thought of; this
# fails closed on everything not named here.
#
# Exact names first, then prefixes for the families whose members vary by run.
_DEFAULT_ENV_EXACT: Sequence[str] = (
    "ROCM_VERSION",
    "ROCM_PATH",
    "AMDGPU_TARGETS",
    "PYTORCH_ROCM_ARCH",
    "WORLD_SIZE",
    "RANK",
    "LOCAL_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
    "SLURM_JOB_ID",
    "SLURM_JOB_NAME",
    "SLURM_NODELIST",
    "SLURM_NNODES",
    "OMP_NUM_THREADS",
    "PRIMUS_IMAGE_DIGEST",
)

_DEFAULT_ENV_PREFIXES: Sequence[str] = (
    "PRIMUS_",
    "HIP_",
    "HSA_",
    "ROCR_",
    "NCCL_",
    "RCCL_",
    "ROCPROFILER_",
    "TORCHINDUCTOR_",
    "GPU_MAX_",
)

# Value-side backstop. The allow-list is the control; this catches the case
# where an allowed name is given a credential-shaped value anyway (someone
# stuffing a token into PRIMUS_SOMETHING). Cheap, and it only ever drops.
_SECRETISH_NAME = re.compile(r"(TOKEN|SECRET|KEY|PASSWORD|PASSWD|CREDENTIAL|AUTH)", re.IGNORECASE)

# Entropy alone does not separate secrets from provenance: measured on real
# values, the path
# "/workspace/Primus/examples/automodel/configs/MI355X/diffusion" scores 4.42
# bits/char while a real 20-char Databricks PAT scores 4.13. Any threshold that
# catches the token also eats the path.
#
# So entropy is only consulted for values that *look* like an opaque token in
# the first place: one unbroken run of token characters. Excluding "/", ":" and
# "." is what keeps paths, URIs and version strings out of the check entirely --
# they have structure, and structure is the tell.
_OPAQUE_TOKEN = re.compile(r"^[A-Za-z0-9+=_\-]{20,}$")
# Digests are provenance worth keeping and are indistinguishable from secrets by
# entropy (a 40-char SHA scores ~3.9). Lowercase hex of a plausible digest
# length is exempted; credentials are rarely pure hex and almost never are when
# they carry a prefix like "dapi" or mixed case.
_DIGEST = re.compile(r"^[0-9a-f]{7,64}$")
_ENTROPY_MIN_LEN = 20
_ENTROPY_MAX_BITS = 3.5


def _shannon_entropy(value: str) -> float:
    """Bits per character. Only meaningful here for dense, unstructured values."""
    if not value:
        return 0.0
    counts: Dict[str, int] = {}
    for char in value:
        counts[char] = counts.get(char, 0) + 1
    length = len(value)
    return -sum((n / length) * math.log2(n / length) for n in counts.values())


def _looks_like_secret(name: str, value: str) -> bool:
    """Whether to drop an otherwise-allowed variable.

    Two independent reasons: the name reads like a credential, or the value is
    an opaque high-entropy token that is not a digest. Structured values --
    paths, URIs, device lists, node ranges, versions -- never reach the entropy
    test, which is what stops the backstop eating the provenance it exists to
    protect.
    """
    if _SECRETISH_NAME.search(name):
        return True
    if len(value) < _ENTROPY_MIN_LEN or not _OPAQUE_TOKEN.match(value):
        return False
    if _DIGEST.match(value):
        return False
    return _shannon_entropy(value) >= _ENTROPY_MAX_BITS


def _allowed(name: str, exact: Iterable[str], prefixes: Iterable[str]) -> bool:
    return name in set(exact) or any(name.startswith(p) for p in prefixes)


def collect_env_params(
    environ: Optional[Mapping[str, str]] = None,
    extra_names: Optional[Sequence[str]] = None,
    exact: Optional[Sequence[str]] = None,
    prefixes: Optional[Sequence[str]] = None,
) -> Dict[str, str]:
    """Environment variables to log, as ``env__<NAME>`` params.

    Pure and dependency-free on purpose: this is the function that decides what
    leaves the machine, so it must be testable without torch, MLflow, a GPU or a
    tracking server. ``environ`` defaults to ``os.environ``.
    """
    environ = os.environ if environ is None else environ
    exact = list(_DEFAULT_ENV_EXACT if exact is None else exact) + list(extra_names or [])
    prefixes = list(_DEFAULT_ENV_PREFIXES if prefixes is None else prefixes)

    out: Dict[str, str] = {}
    for name, value in environ.items():
        if not _allowed(name, exact, prefixes):
            continue
        text = str(value)
        if _looks_like_secret(name, text):
            logger.debug("[PrimusMLflow] dropping %s: matched the secret backstop", name)
            continue
        out[f"env__{name}"] = text
    return out


def _derive_precision(recipe: Any) -> Dict[str, str]:
    """A direct, searchable answer to "what precision was this run?".

    Derived from the RESOLVED feature gates -- the same values that decide what
    actually executes -- rather than from a filename, an experiment path or a
    dtype field. Each of those lies in a different way:

    * the config filename is not logged, and an EXP can be launched with
      overrides that contradict it;
    * the experiment path (``.../Flux/fp8``) is a naming convention we chose,
      so an FP8 run launched without the override lands in the bf16 experiment
      and the path silently misreports;
    * ``model.compute_dtype`` stays ``bfloat16`` on an FP8 run, because
      Primus-Turbo swaps ``nn.Linear`` internals while the surrounding compute
      stays bf16. Measured on a real pair: compute_dtype, torch_dtype and
      reduce_dtype were IDENTICAL across BF16 and FP8 runs. Anyone checking the
      obvious field concludes both were bf16.

    Ordered most-aggressive first, because a run with both mxfp4 and fp8 gates
    on is doing mxfp4 where they overlap.
    """
    turbo_mxfp4 = options.flag("primus_turbo.mxfp4_linear", False)
    te_mxfp4 = options.flag("primus_te.mxfp4_linear", False)
    fp8_linear = options.flag("primus_turbo.fp8_linear", False)
    granularity = options.get("primus_turbo.fp8_linear_granularity", None)
    fmt = options.get("primus_turbo.fp8_linear_format", None)

    if turbo_mxfp4 or te_mxfp4:
        precision, detail = "mxfp4", "mxfp4_linear"
    elif fp8_linear:
        # granularity mx_blockwise is how MXFP8 is reached through the same gate.
        precision = "mxfp8" if str(granularity) == "mx_blockwise" else "fp8"
        detail = "/".join(str(x) for x in (precision, granularity, fmt) if x)
    else:
        # No low-precision linear gate: the run is whatever compute_dtype says.
        try:
            base = str(recipe.cfg.get("model.compute_dtype", "") or "")
        except Exception:
            base = ""
        precision = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}.get(base, base or "unknown")
        detail = precision

    out = {"precision": precision, "precision_detail": detail}
    if options.flag("primus_turbo.fp8_attention", False):
        # Attention precision is a separate gate from the linears and the two
        # are independently settable, so it gets its own field rather than
        # being folded into the headline.
        out["precision_attention"] = "fp8"
    return out


def _primus_section_params() -> Dict[str, str]:
    """Every resolved ``primus_*`` feature gate, as run params.

    Without this, Primus-side gates are invisible to MLflow. ``_start_run``
    logs ``recipe.cfg.to_dict()``, which is the config AFTER
    ``strip_primus_keys`` has removed every ``primus_*`` section
    (``strip_primus_keys``) so AutoModel sees a clean recipe. The
    result: an FP8 run and a BF16 run logged identical inputs --
    ``primus_turbo.fp8_linear`` appeared nowhere -- and the only hint was an
    ambient ``env__PRIMUS_TURBO_*`` variable that is set either way.

    That is the failure mode this exists to prevent: stale configs once ran
    with every fusion silently off, and the result looked like a real number.

    Namespaced ``primus__`` (double underscore) to match the ``env__`` and
    ``pkg__`` conventions already used here, and so these cannot collide with
    AutoModel's own flattened config keys.
    """
    params: Dict[str, str] = {}
    try:
        sections = getattr(options, "_sections", None) or {}
        for section in sorted(sections):
            values = sections[section] or {}
            if not isinstance(values, Mapping):
                params[f"primus__{section}"] = str(values)
                continue
            for key in sorted(values):
                value = values[key]
                if value is None:
                    continue
                params[f"primus__{section}.{key}"] = str(value)
    except Exception as exc:  # pragma: no cover
        logger.debug("[PrimusMLflow] could not collect primus_* sections: %s", exc)
    return params


def _provenance_params() -> Dict[str, str]:
    """Versions and pins worth having on every run.

    Best-effort by construction: a missing optional package must not stop a
    training run from being tracked, so every probe is independently guarded.
    """
    params: Dict[str, str] = {}

    def _record(key: str, fn: Any) -> None:
        try:
            value = fn()
        except Exception:  # pragma: no cover - probe failures are not interesting
            return
        if value:
            params[key] = str(value)

    def _version(pkg: str) -> Any:
        from importlib import metadata

        return metadata.version(pkg)

    _record("pkg__nemo_automodel", lambda: _version("nemo_automodel"))
    _record("pkg__primus_turbo", lambda: _version("primus-turbo"))
    _record("pkg__mlflow", lambda: _version("mlflow"))

    def _torch_bits() -> str:
        import torch

        return torch.__version__

    _record("pkg__torch", _torch_bits)

    def _gpu_name() -> str:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""

    _record("gpu__name", _gpu_name)
    return params


def _logged_model_name(recipe: Any) -> Optional[str]:
    """What to call this run's model in the experiment's Models column.

    Explicit config wins. Otherwise derive it from the transformer class the
    pipeline spec names, which is the one field that always identifies the
    architecture: FluxTransformer2DModel -> Flux, WanTransformer3DModel -> Wan.
    """
    configured = options.get(f"{_SECTION}.logged_model_name", None)
    if configured:
        return str(configured)
    try:
        cls_name = recipe.cfg.get("model.pipeline_spec.transformer_cls", None)
    except Exception:
        cls_name = None
    if not cls_name:
        return None
    for suffix in ("Transformer2DModel", "Transformer3DModel", "Transformer", "Model"):
        if str(cls_name).endswith(suffix):
            return str(cls_name)[: -len(suffix)] or None
    return str(cls_name)


def _link_logged_model(recipe: Any, run: Any) -> None:
    """Populate the experiment's Models column for this run.

    That column reads the run's *outputs*, not a tag and not a param: its own
    tooltip says "models logged or evaluated by the run". Two calls are needed
    and the second is the one that matters --
    ``create_external_model(source_run_id=...)`` sets the model's view of the
    run, but leaves ``run.outputs.model_outputs`` empty, so the cell still
    renders "-". ``log_outputs`` creates the edge the column actually follows.

    "External" is the honest flavour here: these runs are ``mode: pretrain``
    with ``checkpoint.enabled: false``, so there are no weights anywhere and
    none are uploaded. The entity is a label carrying the architecture and
    precision, which is what the column is for.
    """
    name = _logged_model_name(recipe)
    if not name:
        return
    try:
        import mlflow
        from mlflow.entities import LoggedModelOutput
        from mlflow.tracking import MlflowClient

        params = {}
        for key, param in (
            ("model.pretrained_model_name_or_path", "variant"),
            ("model.compute_dtype", "precision"),
        ):
            try:
                value = recipe.cfg.get(key, None)
            except Exception:
                value = None
            if value:
                params[param] = str(value)

        # Reuse the entity rather than minting one per run: every run would
        # otherwise add another identical "Flux" to the experiment's Models
        # page.
        client = MlflowClient()
        existing = None
        try:
            for candidate in client.search_logged_models([run.info.experiment_id]):
                if candidate.name == name:
                    existing = candidate
                    break
        except Exception as exc:
            logger.debug("[PrimusMLflow] logged-model lookup failed, creating a new one: %s", exc)
        if existing is not None:
            client.log_outputs(run.info.run_id, models=[LoggedModelOutput(existing.model_id, step=0)])
            logger.info("[PrimusMLflow] linked existing logged model %s (%s)", name, existing.model_id)
            return

        logged = mlflow.create_external_model(
            name=name,
            source_run_id=run.info.run_id,
            experiment_id=run.info.experiment_id,
            model_type=str(recipe.cfg.get("model.pipeline_spec.transformer_cls", "") or "") or None,
            params=params or None,
        )
        client.log_outputs(run.info.run_id, models=[LoggedModelOutput(logged.model_id, step=0)])
        logger.info("[PrimusMLflow] created logged model %s (%s)", name, logged.model_id)
    except Exception as exc:
        # A label is not worth failing a tracked run over.
        logger.warning("[PrimusMLflow] could not link a logged model: %s", exc)


def _install_killed_handler() -> None:
    """Report SIGTERM as KILLED rather than FINISHED.

    ``MLflowConfig.build`` installs the FAILED excepthook itself, so uncaught
    Python exceptions are already covered. SIGTERM is not: MLflow's atexit
    handler ends the run FINISHED, which makes a cancelled job and a clean one
    look identical. The LLM and VLM recipes call
    ``end_mlflow_active_run_as_killed`` from their own signal handling; the
    diffusion recipe has none, so install it here.

    Chains the previous handler so job-cancellation cleanup elsewhere still runs.
    """
    import signal

    try:
        from nemo_automodel.components.loggers.mlflow_utils import (
            end_mlflow_active_run_as_killed,
        )
    except Exception:  # pragma: no cover - upstream layout changed
        logger.warning("[PrimusMLflow] no end_mlflow_active_run_as_killed; SIGTERM will report FINISHED")
        return

    previous = signal.getsignal(signal.SIGTERM)
    if getattr(previous, "_primus_mlflow_killed_handler", False):
        return

    def _handler(signum: int, frame: Any) -> None:
        end_mlflow_active_run_as_killed()
        if callable(previous):
            previous(signum, frame)

    _handler._primus_mlflow_killed_handler = True  # type: ignore[attr-defined]
    try:
        signal.signal(signal.SIGTERM, _handler)
    except ValueError:  # pragma: no cover - not on the main thread
        logger.debug("[PrimusMLflow] could not install SIGTERM handler off the main thread")


def _start_run(recipe: Any) -> Any:
    """Start or resume the MLflow run and log the inputs. Returns the run, or None.

    Returns None on every non-rank-0 process, because ``MLflowConfig.build``
    does; downstream ``mlflow.log_*`` calls then no-op naturally since those
    processes have no active run.
    """
    config = getattr(recipe.cfg, "mlflow", None)
    if config is None:
        logger.warning(
            "[PrimusMLflow] primus_mlflow.enable is true but the config has no `mlflow:` block; "
            "nothing will be tracked. Add one with the MLflowConfig fields."
        )
        return None

    import time as _time

    checkpoint_dir = recipe.cfg.get("checkpoint.checkpoint_dir", None)
    run_config = recipe.cfg.to_dict() if hasattr(recipe.cfg, "to_dict") else None
    _t0 = _time.perf_counter()
    run = config.build(checkpoint_dir=checkpoint_dir, run_config=run_config)
    _t_build = _time.perf_counter() - _t0
    if run is None:
        return None

    import mlflow

    params = collect_env_params(extra_names=options.get(f"{_SECTION}.env_allowlist_extra", []) or [])
    params.update(_provenance_params())
    params.update(_primus_section_params())

    # Precision as a TAG as well as a param: tags are what the MLflow UI
    # filters and columns on, e.g. `tags.precision = 'fp8'`
    precision = _derive_precision(recipe)
    params.update({f"primus__{k}": v for k, v in precision.items()})
    try:
        mlflow.set_tags(precision)
    except Exception as exc:
        logger.warning("[PrimusMLflow] could not set precision tags: %s", exc)
    _t2 = _time.perf_counter()
    try:
        mlflow.log_params(params)
    except Exception as exc:
        # Losing provenance is bad; losing the run because provenance failed to
        # upload is worse.
        logger.warning("[PrimusMLflow] failed to log environment params: %s", exc)
    _t_params = _time.perf_counter() - _t2

    _t1 = _time.perf_counter()
    _link_logged_model(recipe, run)
    _t_link = _time.perf_counter() - _t1
    _install_killed_handler()
    # Startup cost is paid on rank 0 while every other rank waits at the first
    # collective, and the recipe's first window is an all-reduce MAX across
    # ranks -- so it lands in the warmup step time. Logged so the split is
    # visible rather than inferred.
    logger.info(
        "[PrimusMLflow] startup %.2fs (build %.2fs, params %.2fs, model-link %.2fs)",
        _t_build + _t_params + _t_link,
        _t_build,
        _t_params,
        _t_link,
    )
    logger.info("[PrimusMLflow] tracking run %s (%d env/provenance params)", run.info.run_id, len(params))
    return run


def _finish_run(status: str) -> None:
    """End the active run with an explicit status.

    MLflow's atexit handler defaults to FINISHED, so a run that ended badly is
    indistinguishable from one that did not unless the status is set here. That
    distinction is the whole point of tracking: evidence that says a crashed run
    succeeded is worse than no evidence.
    """
    try:
        import mlflow
    except ImportError:  # pragma: no cover
        return

    if mlflow.active_run() is None:
        return
    try:
        mlflow.end_run(status=status)
    except Exception as exc:  # pragma: no cover
        logger.warning("[PrimusMLflow] failed to end run as %s: %s", status, exc)


# ---------------------------------------------------------------------------
# End-of-run artifacts
#
# Ported from, not shared with, ``primus/backends/megatron/training/
# mlflow_artifacts.py``. That module is itself backend-agnostic -- it imports
# only stdlib and ``primus.core.utils.module_utils`` -- but it cannot be
# imported from here: ``primus/backends/megatron/__init__.py`` registers the
# Megatron adapter, trainers and *every* Megatron patch as an import side
# effect, and Python runs the parent package before the submodule. Importing it
# from an AutoModel run would register a backend that run has nothing to do
# with. So the logic is reimplemented; the shapes deliberately mirror it.
# ---------------------------------------------------------------------------

_TRACE_PATTERNS: Sequence[str] = ("*.pt.trace.json", "*.pt.trace.json.gz")

# Rank extraction from a trace filename. The first pattern is AutoModel's own
# and is the reason this is not a straight copy: ``torch_profiler`` writes
# ``rank{rank}.pt.trace.json`` -- "rank0.", with no separator before the digit
# (``patched_loop`` in profiling/torch_profiler.py). None of the Megatron patterns
# (``rank_N_``, ``rank_N.``, ``rank[N]``, ``-rankN.``, ``_rankN.``) match that,
# so a copied rank filter would select zero files and look like "no traces
# found" rather than "the filter is broken". The remaining patterns are kept so
# traces produced by other tooling still resolve.
_RANK_PATTERNS: Sequence[Any] = (
    re.compile(r"(?:^|[^A-Za-z0-9])rank(\d+)\."),
    re.compile(r"rank_(\d+)[_.]"),
    re.compile(r"rank\[(\d+)\]"),
)

# Traces from a multinode run reach gigabytes, and they are uploaded from one
# process at the end of a job. Cap the total rather than discovering the limit
# as a timeout. Megatron's version has no cap; this is an addition, not a port.
_DEFAULT_UPLOAD_MAX_BYTES = 2 * 1024**3

# The recipe's per-step log line, which is the same source `auto_benchmark`
# parses -- deliberately, so a summary here and a benchmark table elsewhere
# cannot disagree. Format string in ``run_train_validation_loop``
# (recipes/diffusion/train.py).
# ONE pattern for both consumers: the end-of-run summary parses it out of the
# log file, and the live monitor (below) matches it as the line is emitted. A
# second regex would be a second thing to keep in step, and the whole point of
# the summary reading the recipe's own log is that it cannot disagree about what
# the run did -- a live series derived from a different pattern would reopen
# exactly that gap.
#
# Deliberately lenient (`.*?` between fields rather than fixed spacing) so an
# upstream reordering or an inserted field degrades to a miss on one group
# rather than silently matching nothing. Requires the [TRAIN] marker so it
# cannot latch onto some other line that happens to carry a `loss=`.
_TRAIN_LINE = re.compile(
    r"\[TRAIN\].*?step=(?P<step>\d+).*?"
    r"loss=(?P<loss>[-+0-9.eE]+).*?"
    r"avg_loss=(?P<avg_loss>[-+0-9.eE]+).*?"
    r"lr=(?P<lr>[-+0-9.eE]+).*?"
    r"grad_norm=(?P<grad_norm>[-+0-9.eE]+).*?"
    r"step_time=(?P<step_time>[0-9.]+)s.*?"
    r"samples_per_sec_per_gpu=(?P<sps_gpu>[0-9.]+).*?"
    r"mem=(?P<mem_gb>[0-9.]+)GB"
)


def _extract_rank(filename: str) -> Optional[int]:
    """Rank encoded in a trace filename, or None when it carries no rank."""
    for pattern in _RANK_PATTERNS:
        match = pattern.search(filename)
        if match:
            return int(match.group(1))
    return None


def _normalize_ranks(raw: Any) -> Optional[Sequence[int]]:
    """Rank filter as a sorted list, ``None`` for "every rank", ``[]`` for none.

    Tolerates the string form a YAML or CLI override can produce. An unparseable
    value disables the filter rather than silently uploading nothing, because
    "all traces" is the recoverable mistake and "no traces" is not.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        import ast

        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            logger.warning("[PrimusMLflow] could not parse tracelens_ranks %r; uploading all ranks", raw)
            return None
    if not isinstance(raw, (list, tuple)):
        logger.warning("[PrimusMLflow] tracelens_ranks is not a list; uploading all ranks")
        return None

    ranks = []
    for item in raw:
        if isinstance(item, bool):
            continue
        try:
            value = int(item)
        except (TypeError, ValueError):
            continue
        if value >= 0:
            ranks.append(value)
    return sorted(set(ranks))


def _find_files(root: Optional[str], patterns: Sequence[str]) -> list:
    """Files under ``root`` matching any glob pattern, de-duplicated and sorted."""
    import glob as globmod

    if not root or not os.path.isdir(root):
        return []
    found = []
    escaped = globmod.escape(root)
    for pattern in patterns:
        found.extend(globmod.glob(os.path.join(escaped, "**", pattern), recursive=True))
    return sorted(set(found))


def _filter_by_rank(files: Sequence[str], ranks: Optional[Sequence[int]]) -> list:
    """Keep only traces whose filename names a rank in ``ranks``."""
    if ranks is None:
        return list(files)
    if not ranks:
        return []
    wanted = set(ranks)
    return [f for f in files if _extract_rank(os.path.basename(f)) in wanted]


def _cap_total_size(files: Sequence[str], max_bytes: int) -> tuple:
    """Split ``files`` into (kept, dropped) so the kept total stays under the cap.

    Smallest first, so a single enormous trace cannot starve every other rank.
    Returns the dropped list too because the caller logs it: a silent truncation
    reads as "this is everything" when it is not.
    """
    if max_bytes <= 0:
        return list(files), []

    def _size(path: str) -> int:
        try:
            return os.path.getsize(path)
        except OSError:
            return 0

    kept, dropped, total = [], [], 0
    for path in sorted(files, key=_size):
        size = _size(path)
        if total + size > max_bytes and kept:
            dropped.append(path)
            continue
        kept.append(path)
        total += size
    return sorted(kept), sorted(dropped)


def _upload_files(mlflow: Any, files: Sequence[str], base_dir: str, artifact_path: str) -> int:
    """Upload each file, preserving its path under ``base_dir``. Returns the count.

    Per-file try/except: one unreadable trace must not cost the other nineteen.
    """
    uploaded = 0
    for path in files:
        try:
            rel_dir = os.path.dirname(os.path.relpath(path, base_dir))
            subpath = os.path.join(artifact_path, rel_dir) if rel_dir else artifact_path
            mlflow.log_artifact(path, artifact_path=subpath)
            uploaded += 1
        except Exception as exc:
            logger.warning("[PrimusMLflow] failed to upload %s: %s", path, exc)
    return uploaded


# The summary must read ONE file per rank, from THIS run only. Both halves were
# found by the first real tier-1 run, which reported 16 steps for a 4-step job.
#
#   * the rank directory holds info/debug/error/warning logs, and info.log and
#     debug.log carry byte-identical [TRAIN] lines -- so a "*.log" glob counted
#     every step once per file;
#   * logs are appended across runs in the same workspace, so the second run's
#     summary also included the first run's steps, warmup included.
_SUMMARY_LOG_PATTERNS: Sequence[str] = ("info.log",)

# Byte offset of each summary log at the moment this run started training.
_log_offsets: Dict[str, int] = {}


def _snapshot_log_offsets(exp_root: Optional[str]) -> None:
    """Record how long each log already is, so the summary skips earlier runs.

    Called once as training starts. A file that appears later is absent here and
    reads from offset 0, which is correct: all of it belongs to this run.
    """
    global _log_offsets
    _log_offsets = {}
    for path in _find_files(os.path.join(exp_root or "", "logs"), _SUMMARY_LOG_PATTERNS):
        try:
            _log_offsets[path] = os.path.getsize(path)
        except OSError as exc:
            logger.debug("[PrimusMLflow] could not size %s: %s", path, exc)
            _log_offsets[path] = 0


def _summarise_from_logs(
    log_files: Sequence[str], offsets: Optional[Mapping[str, int]] = None
) -> Dict[str, float]:
    """Run summary derived from the training log after the run has finished.

    Deliberately a log parse rather than a per-step hook. Nothing here may add
    a GPU sync or a collective to the step, and the numbers are
    already in the log at full precision -- so reading them afterwards costs the
    run nothing and cannot perturb the thing it measures. It also uses the same
    source `auto_benchmark` parses, so the two cannot disagree.

    Note the floor this inherits: ``samples_per_sec_per_gpu`` is logged at
    ``%.2f``, so for a low-throughput video model (Wan at ~0.375) the value is
    quantised to ~1.3%. That bounds how tight a regression threshold built on
    this number can be.
    """
    step_times, sps_gpu, mem_gb = [], [], []
    offsets = offsets or {}
    for path in log_files:
        try:
            # Binary, because the start offset comes from getsize() rather than
            # tell(); seeking a text stream to an arbitrary byte is undefined.
            with open(path, "rb") as handle:
                handle.seek(offsets.get(path, 0))
                for raw in handle:
                    match = _TRAIN_LINE.search(raw.decode("utf-8", errors="replace"))
                    if not match:
                        continue
                    step_times.append(float(match.group("step_time")))
                    sps_gpu.append(float(match.group("sps_gpu")))
                    mem_gb.append(float(match.group("mem_gb")))
        except OSError as exc:
            logger.debug("[PrimusMLflow] could not read %s: %s", path, exc)

    if not step_times:
        return {}

    # The first record is excluded from every statistic below, and reported as
    # its own metric instead.
    #
    # It is not merely a slow outlier, the optimizer state is not
    # allocated yet. Including it therefore both distorts the step time and
    # *understates* peak memory, which is the opposite of what a peak is for.
    #
    # A single-record run keeps that record: a wrong number is better than no
    # number here, and perf/warmup_* makes it obvious which case you are in.
    total_records = len(step_times)
    warmup_step_time = step_times[0]
    warmup_mem_gb = mem_gb[0]
    if total_records > 1:
        step_times, sps_gpu, mem_gb = step_times[1:], sps_gpu[1:], mem_gb[1:]

    def _median(values: list) -> float:
        ordered = sorted(values)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[mid]
        return (ordered[mid - 1] + ordered[mid]) / 2.0

    # perf/* names, matching the namespace main already documents in
    # primus_megatron_module.yaml, so a diffusion run and an LLM run are
    # comparable in one experiment.
    #
    # There is deliberately no perf/throughput_tflops_per_gpu here. The
    # diffusion recipe does not log FLOPs (its [TRAIN] line emits step_time,
    # samples_per_sec, samples_per_sec_per_gpu and mem, and nothing else), and
    # inventing one would make a published comparison wrong rather than
    # missing. If FLOPs are wired up later -- nemo_automodel has an unused
    # flux_flops() in components/utils/flops_utils.py -- this is where the
    # resulting value belongs.
    return {
        "perf/iteration_time_ms": _median(step_times) * 1000.0,
        "perf/samples_per_sec_per_gpu": _median(sps_gpu),
        "perf/max_memory_allocated_gb": max(mem_gb),
        "perf/logged_steps": float(total_records),
        # The excluded first record, reported rather than discarded: a compile
        # step that suddenly doubles is worth seeing, and summary_records makes
        # it unambiguous how many samples the medians above are built on.
        "perf/summary_records": float(len(step_times)),
        "perf/warmup_iteration_time_ms": warmup_step_time * 1000.0,
        "perf/warmup_memory_allocated_gb": warmup_mem_gb,
    }


def _trace_dir(exp_root: Optional[str]) -> Optional[str]:
    """Where ``torch_profiler`` writes traces, asked of the profiler itself.

    Resolving the path here independently would mean two definitions of the same
    directory that drift apart the first time either changes.
    """
    try:
        from primus.backends.automodel.profiling import torch_profiler

        out_dir = torch_profiler.resolve_output_dir(
            options.string("primus_profiler.output_dir", "diffusion_profile"), exp_root
        )
    except Exception as exc:
        logger.debug("[PrimusMLflow] could not resolve the profiler output dir: %s", exc)
        return None
    tag = options.string("primus_profiler.tag", "run")
    return os.path.join(out_dir, tag)


def _upload_artifacts(exp_root: Optional[str]) -> None:
    """Attach logs, traces, TraceLens reports and the run summary to the run.

    Called after the train loop returns, so every upload is outside the timed
    region and cannot inflate a step-time measurement.

    Every step is independently guarded and the whole body is best-effort: a
    tracking server having a bad day must not turn a finished training run into
    a failed one. ``run_patches`` also catches patch failures, so this does not
    re-wrap what it already handles -- it keeps a partial upload from discarding
    the parts that did succeed.
    """
    logs_dir = os.path.join(exp_root or "", "logs")
    max_bytes = options.integer(f"{_SECTION}.upload_max_bytes", _DEFAULT_UPLOAD_MAX_BYTES)
    ranks = _normalize_ranks(options.get(f"{_SECTION}.tracelens_ranks", None))

    # Work out what there is to send *before* importing MLflow, so a run with
    # nothing to upload -- and the unit tests, which have no MLflow at all --
    # never reach for it.
    # One file per rank, from this run's offset. Upload below still takes
    # "*.log" on purpose -- every level is worth having as an artifact; it is
    # only the summary that must not count a step twice.
    summary = (
        _summarise_from_logs(_find_files(logs_dir, _SUMMARY_LOG_PATTERNS), _log_offsets)
        if options.flag(f"{_SECTION}.summary", True)
        else {}
    )

    log_files: list = []
    if options.flag(f"{_SECTION}.upload_logs", False):
        log_files, dropped = _cap_total_size(_find_files(logs_dir, ("*.log",)), max_bytes)
        if dropped:
            logger.warning("[PrimusMLflow] log upload capped: %d file(s) omitted", len(dropped))

    trace_dir = None
    trace_files: list = []
    if options.flag(f"{_SECTION}.upload_traces", False):
        trace_dir = _trace_dir(exp_root)
        found = _filter_by_rank(_find_files(trace_dir, _TRACE_PATTERNS), ranks)
        trace_files, dropped = _cap_total_size(found, max_bytes)
        if dropped:
            logger.warning(
                "[PrimusMLflow] trace upload capped at %d bytes: %d file(s) omitted", max_bytes, len(dropped)
            )

    want_tracelens = options.flag(f"{_SECTION}.upload_tracelens_report", False)
    if not (summary or log_files or trace_files or want_tracelens):
        return

    import mlflow

    if summary:
        try:
            mlflow.log_metrics(summary)
            logger.info("[PrimusMLflow] logged run summary over %d steps", int(summary["perf/logged_steps"]))
        except Exception as exc:
            logger.warning("[PrimusMLflow] failed to log the run summary: %s", exc)

    if log_files:
        logger.info(
            "[PrimusMLflow] uploaded %d log file(s)", _upload_files(mlflow, log_files, logs_dir, "logs")
        )

    if trace_files and trace_dir:
        logger.info(
            "[PrimusMLflow] uploaded %d trace file(s)",
            _upload_files(mlflow, trace_files, trace_dir, "traces"),
        )

    if want_tracelens:
        _upload_tracelens(mlflow, exp_root, ranks)


def _upload_tracelens(mlflow: Any, exp_root: Optional[str], ranks: Optional[Sequence[int]]) -> None:
    """Generate TraceLens reports for the selected ranks and upload them.

    TraceLens is an optional extra, not a Primus dependency, so a missing import
    is an expected outcome and not an error. Unlike the Megatron path this does
    not pip-install anything: a training job that reaches for the network at
    teardown is a job that hangs at teardown.
    """
    trace_dir = _trace_dir(exp_root)
    traces = _filter_by_rank(_find_files(trace_dir, _TRACE_PATTERNS), ranks)
    if not traces:
        logger.info("[PrimusMLflow] no traces to analyse; skipping TraceLens")
        return

    try:
        from TraceLens.Reporting import generate_perf_report_pytorch
    except ImportError:
        logger.warning(
            "[PrimusMLflow] upload_tracelens_report is on but TraceLens is not installed; "
            "skipping. Install it in the image rather than at run time."
        )
        return

    reports_dir = os.path.join(exp_root or ".", "tracelens_reports")
    os.makedirs(reports_dir, exist_ok=True)
    fmt = options.string(f"{_SECTION}.tracelens_output_format", "xlsx").strip().lower()
    if fmt not in ("xlsx", "csv"):
        logger.warning("[PrimusMLflow] unsupported tracelens_output_format %r; using xlsx", fmt)
        fmt = "xlsx"

    produced = []
    for trace in traces:
        name = os.path.basename(trace)
        for suffix in (".pt.trace.json.gz", ".pt.trace.json", ".json.gz", ".json"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        try:
            if fmt == "xlsx":
                out = os.path.join(reports_dir, f"{name}_analysis.xlsx")
                generate_perf_report_pytorch(trace, output_xlsx_path=out)
                if os.path.exists(out):
                    produced.append(out)
            else:
                out = os.path.join(reports_dir, name)
                os.makedirs(out, exist_ok=True)
                generate_perf_report_pytorch(trace, output_csvs_dir=out)
                produced.extend(_find_files(out, ("*.csv",)))
        except Exception as exc:
            logger.warning("[PrimusMLflow] TraceLens failed on %s: %s", name, exc)

    # TraceLens also writes sidecars next to the report -- trace_checks.json is
    # its own validation of the trace, and it carries findings the report does
    # not, e.g. "with_stack=False, so call-stack analysis is unavailable". A few
    # KB against a ~380 KB report, and skipping it means uploading an analysis
    # while leaving behind the file that says what the analysis could not see.
    produced.extend(f for f in _find_files(reports_dir, ("*.json",)) if f not in produced)

    kept, dropped = _cap_total_size(
        produced, options.integer(f"{_SECTION}.upload_max_bytes", _DEFAULT_UPLOAD_MAX_BYTES)
    )
    if dropped:
        logger.warning("[PrimusMLflow] TraceLens upload capped: %d file(s) omitted", len(dropped))
    if kept:
        logger.info(
            "[PrimusMLflow] uploaded %d TraceLens report file(s)",
            _upload_files(mlflow, kept, reports_dir, "trace_analysis"),
        )


def install(exp_root: Optional[str] = None) -> bool:
    """Wrap the diffusion train loop so the run is tracked end to end.

    ``exp_root`` is the experiment directory; logs and traces are found relative
    to it at teardown. Idempotent: a second call with the wrapper already in
    place is a no-op, so two jobs in one process do not nest wrappers.
    """
    global _exp_root
    from nemo_automodel.recipes.diffusion.train import TrainDiffusionRecipe

    _exp_root = exp_root
    if getattr(TrainDiffusionRecipe, _INSTALLED_FLAG, False):
        return True

    original = TrainDiffusionRecipe.run_train_validation_loop

    def patched_loop(self: Any) -> Any:
        run = None
        # Before anything writes to the log: everything already in it belongs to
        # an earlier run and must not reach this run's summary.
        _snapshot_log_offsets(_exp_root)
        try:
            run = _start_run(self)
        except Exception as exc:
            # Tracking is observability. It does not get to end a training run.
            logger.warning("[PrimusMLflow] setup failed, continuing untracked: %s", exc)
            return original(self)

        if run is None:
            return original(self)

        # Only after a run exists: capture with nowhere to send is pure cost.
        _start_per_step(getattr(getattr(run, "info", None), "run_id", None))

        try:
            result = original(self)
        except BaseException:
            # FAILED covers the excepthook's blind spots: an exception caught by
            # a caller never reaches sys.excepthook, and KeyboardInterrupt would
            # otherwise land as FINISHED.
            #
            # Artifacts are uploaded here too: the logs and traces of a run that
            # crashed are the ones most worth having.
            _finalize(_exp_root, "FAILED")
            raise
        _finalize(_exp_root, "FINISHED")
        return result

    TrainDiffusionRecipe.run_train_validation_loop = patched_loop
    setattr(TrainDiffusionRecipe, _INSTALLED_FLAG, True)
    return True


# ---------------------------------------------------------------------------
# Per-step metrics, default off.
#
# What actually costs something here is NOT a GPU sync -- it is blocking I/O.
# The recipe already calls .item() on the loss, the elapsed timer and the sample
# count every log window to format its own [TRAIN] line
# (``run_train_validation_loop``, ``_elapsed_seconds_since``,
# ``_count_global_samples``), so those syncs are paid whether or not anything
# is tracked. Reading the results afterwards adds none.
#
# The real hazard is that ``mlflow.log_metrics`` is an HTTPS POST to a tracking
# server in another region. On a ~1.5s step with log_remote_every_steps=2 that
# is a round trip every ~3s on rank 0, while seven other ranks wait at the next
# collective. A 100ms blocking POST is ~3% of a step -- six times the 0.5%
# budget above.
#
# So: capture host-side values at the log window, hand them to a queue, and let
# a daemon thread do the network. Nothing in the training loop ever waits.
_PER_STEP_INSTALLED = "_primus_mlflow_per_step_installed"
_per_step_queue: Any = None
_per_step_thread: Any = None
_PER_STEP_STOP = object()


def _per_step_worker(queue_obj: Any, run_id: str) -> None:
    """Drain the queue onto the tracking server. Runs off the training thread.

    Takes an explicit ``run_id`` and uses ``MlflowClient`` rather than the
    fluent ``mlflow.log_metrics``. That is not a style choice:
    ``mlflow._active_run_stack`` is a ``ThreadLocalVariable``, so the run
    started on the training thread is invisible here and
    ``mlflow.active_run()`` returns None. The first version of this did exactly
    that and dropped every metric silently -- the run finished FINISHED, the
    summary was attached, and the per-step series simply never appeared.
    """
    import time

    from mlflow.entities import Metric
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    while True:
        item = queue_obj.get()
        step = None
        try:
            if item is _PER_STEP_STOP:
                return
            metrics, step = item
            stamp = int(time.time() * 1000)
            client.log_batch(run_id, metrics=[Metric(k, v, stamp, step) for k, v in metrics.items()])
        except Exception as exc:  # pragma: no cover - telemetry must not kill a run
            logger.debug("[PrimusMLflow] per-step flush failed at step %s: %s", step, exc)
        finally:
            queue_obj.task_done()


def _enqueue_per_step(metrics: Dict[str, float], step: int) -> None:
    if _per_step_queue is None:
        return
    try:
        _per_step_queue.put_nowait((metrics, step))
    except Exception:  # pragma: no cover - a full queue drops telemetry, not the run
        logger.debug("[PrimusMLflow] per-step queue full; dropped step %s", step)


class _TrainLineHandler(logging.Handler):
    """Mirror the recipe's own [TRAIN] line to MLflow as it is emitted.

    This replaced a pair of monkey-patches on ``_calculate_throughput_metrics``
    and ``_get_memory_metrics``. Those worked, but could only reach throughput
    and memory: loss, lr and grad_norm live in a local dict the recipe never
    exposes, and loss is the series anyone watching a long run actually wants.

    Hooking the log emission gets all of them at no cost. By the time the
    record exists every value has already been formatted to a Python float --
    the recipe called ``.item()`` on the loss in ``run_train_validation_loop``
    for its own log line -- so there is no tensor here and nothing to
    synchronise. Nothing is
    read from disk either: this is the record on its way out, not the file.

    Rank filtering is free: the recipe emits this line under
    ``if should_log and self.dist_env.is_main``, so only rank 0 ever produces
    one. Verified on a real 8-rank run -- rank-0/info.log carried every [TRAIN]
    line and ranks 1-7 carried none.
    """

    def emit(self, record: Any) -> None:
        try:
            # Reject on the UNFORMATTED template first. record.getMessage()
            # runs %-formatting, and this handler sees every record in the
            # process -- during torch.compile that is thousands of inductor and
            # dynamo records per run. Formatting them all cost +4.24s (+30%) on
            # the warmup step, measured as a clean non-overlapping separation
            # across 5+5 runs. The marker is in the format string itself
            # ("[TRAIN] step=%s ..."), so this check needs no formatting at all.
            if "[TRAIN]" not in str(record.msg):
                return
            message = record.getMessage()
            if "[TRAIN]" not in message:  # formatted-in marker, be strict
                return
            match = _TRAIN_LINE.search(message)
            if match is None:
                return
            found = match.groupdict()
            # Progress is recorded whatever the per_step setting: it is what
            # lets the heartbeat distinguish "alive but wedged" from "alive and
            # training". A timer alone cannot -- a daemon thread keeps ticking
            # happily while the training thread is deadlocked on a collective.
            global _previous_step_seen, _previous_step_wall
            import time as _time

            _previous_step_seen = int(found["step"])
            _previous_step_wall = _time.time()
            if not options.flag(f"{_SECTION}.per_step", False):
                return
            _enqueue_per_step(
                {
                    "train/loss": float(found["loss"]),
                    "train/avg_loss": float(found["avg_loss"]),
                    "train/lr": float(found["lr"]),
                    "train/grad_norm": float(found["grad_norm"]),
                    "perf/iteration_time_ms": float(found["step_time"]) * 1000.0,
                    "perf/samples_per_sec_per_gpu": float(found["sps_gpu"]),
                    "perf/max_memory_allocated_gb": float(found["mem_gb"]),
                },
                int(found["step"]),
            )
        except Exception:  # pragma: no cover
            # A logging handler that raises breaks logging itself, which would
            # take the training run's own output with it. Telemetry is never
            # worth that.
            pass


_train_line_handler: Optional[logging.Handler] = None

# Previous training step observed, and when. Read by the heartbeat.
_previous_step_seen: int = -1
_previous_step_wall: float = 0.0
_heartbeat_thread: Any = None
_heartbeat_stop: Any = None
_DEFAULT_HEARTBEAT_SECONDS = 60.0


def _install_train_line_handler() -> None:
    """Attach the live monitor to the root logger. Idempotent."""
    global _train_line_handler

    if _train_line_handler is not None:
        return
    handler = _TrainLineHandler()
    handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)
    _train_line_handler = handler


def _remove_train_line_handler() -> None:
    global _train_line_handler

    if _train_line_handler is None:
        return
    try:
        logging.getLogger().removeHandler(_train_line_handler)
    except Exception:  # pragma: no cover
        pass
    _train_line_handler = None


def _heartbeat_loop(stop_event: Any, interval: float, started: float) -> None:
    """Emit liveness independently of the training loop.

    ``primus/heartbeat_unixtime`` says the process is alive;
    ``primus/previous_step`` says whether it is getting anywhere. A heartbeat
    that advances while previous_step stands still is a hung run -- which no
    status handler will ever report, because a wedged process never exits and so
    never reaches FAILED or KILLED. A hard kill stops both, which is what
    separates "wedged" from "dead": the two are indistinguishable from a stalled
    metric stream alone.

    ``primus/seconds_since_previous_step`` is that same signal as ONE number,
    and it exists because the other two are close to unreadable in the UI. Every
    heartbeat is logged at the most recently observed step, so on a wedged run
    they all share one x-coordinate and collapse into a single point on MLflow's
    default step axis -- the climb is only visible after manually switching to
    the time axis. This series is flat and small on a healthy run and climbs on
    a wedged one, on whichever axis is in front of you.

    Before the first [TRAIN] line it measures from ``started`` rather than from
    an unset timestamp, so a long compile or warmup reads as elapsed time.
    """
    import time

    while not stop_event.wait(interval):
        now = time.time()
        _enqueue_per_step(
            {
                "primus/heartbeat_unixtime": now,
                "primus/previous_step": float(_previous_step_seen),
                "primus/seconds_since_previous_step": now - (_previous_step_wall or started),
            },
            max(_previous_step_seen, 0),
        )


def _start_heartbeat() -> None:
    """Timer-based liveness, armed on every tracked run.

    Deliberately not conditional on ``per_step``. The per-step stream is a
    liveness signal only in the sense that it stops when training stops -- and
    it stops identically whether the process was hard-killed or is alive and
    wedged on a collective. Only a timer running off the training thread tells
    those apart, which is the case worth catching: a wedged run never exits, so
    no status handler ever reports it, and it holds its GPUs indefinitely.

    The cost is small enough not to trade against that. At the 60s default
    against a per-step stream posting every few seconds it is a rounding error
    on an already-async, already-batched path, and ``heartbeat_seconds: 0``
    turns it off outright.
    """
    global _heartbeat_thread, _heartbeat_stop

    interval = float(options.get(f"{_SECTION}.heartbeat_seconds", _DEFAULT_HEARTBEAT_SECONDS) or 0)
    if interval <= 0:
        return
    import threading
    import time

    _heartbeat_stop = threading.Event()
    _heartbeat_thread = threading.Thread(
        target=_heartbeat_loop,
        args=(_heartbeat_stop, interval, time.time()),
        name="primus-mlflow-heartbeat",
        daemon=True,
    )
    _heartbeat_thread.start()
    logger.info("[PrimusMLflow] heartbeat every %.0fs", interval)


def _stop_heartbeat() -> None:
    global _heartbeat_thread, _heartbeat_stop

    if _heartbeat_stop is not None:
        _heartbeat_stop.set()
    if _heartbeat_thread is not None:
        _heartbeat_thread.join(timeout=5)
    _heartbeat_thread = None
    _heartbeat_stop = None


def _start_per_step(run_id: Optional[str]) -> None:
    """Arm per-step capture and its flusher. No-op unless primus_mlflow.per_step.

    ``run_id`` is read on the training thread and handed to the worker, because
    the worker cannot discover it for itself (see ``_per_step_worker``).
    """
    global _per_step_queue, _per_step_thread

    if not run_id:
        logger.warning("[PrimusMLflow] per-step requested but no run id; continuing without it")
        return
    try:
        import queue as _queue
        import threading

        # Bounded: if the server stalls, telemetry is dropped rather than
        # allowed to grow without limit behind a training loop that cannot wait.
        _per_step_queue = _queue.Queue(maxsize=1024)
        _per_step_thread = threading.Thread(
            target=_per_step_worker,
            args=(_per_step_queue, run_id),
            name="primus-mlflow-perstep",
            daemon=True,
        )
        _per_step_thread.start()
        _install_train_line_handler()
        if options.flag(f"{_SECTION}.per_step", False):
            logger.info(
                "[PrimusMLflow] per-step metrics on, flushed asynchronously at "
                "step_scheduler.log_remote_every_steps"
            )
        # Armed either way: the per-step stream cannot tell a wedged run from a
        # dead one, because both present as a stream that stopped.
        _start_heartbeat()
    except Exception as exc:
        logger.warning("[PrimusMLflow] per-step setup failed, continuing without it: %s", exc)
        _per_step_queue = None


def _stop_per_step(timeout: float = 30.0) -> None:
    """Drain and stop the flusher before the run ends.

    Outside the timed region, and bounded: a tracking server that has stopped
    answering must not hold a finished training job open.
    """
    global _per_step_queue, _per_step_thread

    _stop_heartbeat()
    _remove_train_line_handler()
    if _per_step_queue is None:
        return
    try:
        _per_step_queue.put_nowait(_PER_STEP_STOP)
        if _per_step_thread is not None:
            _per_step_thread.join(timeout=timeout)
            if _per_step_thread.is_alive():
                logger.warning(
                    "[PrimusMLflow] per-step flusher still busy after %.0fs; abandoning it", timeout
                )
        else:
            logger.debug("[PrimusMLflow] per-step flusher was never started")
    except Exception as exc:  # pragma: no cover
        logger.debug("[PrimusMLflow] per-step shutdown: %s", exc)
    finally:
        _per_step_queue = None
        _per_step_thread = None


def _finalize(exp_root: Optional[str], status: str) -> None:
    """Upload artifacts, then end the run with ``status``.

    Order matters: ``mlflow.log_artifact`` needs an active run, so uploading
    after ``end_run`` would silently attach nothing. The per-step flusher is
    drained first for the same reason.
    """
    _stop_per_step()
    try:
        _upload_artifacts(exp_root)
    except Exception as exc:
        logger.warning("[PrimusMLflow] artifact upload failed: %s", exc)
    _finish_run(status)


def _enabled(ctx: PatchContext) -> bool:
    return options.flag(f"{_SECTION}.enable", False)


@register_patch(
    "automodel.mlflow",
    backend="automodel",
    phase="before_train",
    description="Track the diffusion run in MLflow (primus_mlflow.enable)",
    condition=_enabled,
    # Early: it wraps the train loop and should sit outside anything that
    # replaces methods on the recipe later, so the whole run is inside the
    # tracked region. torch_profiler runs at 90 and nests within this.
    priority=20,
)
def apply(ctx: PatchContext) -> None:
    install(exp_root=getattr(ctx.extra.get("primus_config"), "exp_root_path", None))
