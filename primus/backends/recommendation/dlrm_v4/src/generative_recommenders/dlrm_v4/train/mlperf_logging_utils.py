# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe
"""MLPerf Training compliance logging for the DLRMv4 streaming-train-eval path.

Rank-0-gated wrapper around ``mlperf_logging.mllog`` so the streaming loop emits
the MLPerf event stream without every call site re-checking rank or the dep.
"""

import logging
import os
import re
import sys
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import gin
import torch

logger: logging.Logger = logging.getLogger(__name__)

try:
    from mlperf_logging import mllog
    from mlperf_logging.mllog import constants as mllog_constants

    _MLLOG_AVAILABLE = True
except Exception as e:  # pragma: no cover - import-time guard
    mllog = None  # type: ignore[assignment]
    mllog_constants = None  # type: ignore[assignment]
    _MLLOG_AVAILABLE = False
    logger.warning(
        "mlperf_logging not importable (%s); MLPerf logging disabled. "
        "Install via `pip install git+https://github.com/mlcommons/logging.git`.",
        e,
    )


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


class MLPerfLogger:
    """Rank-0-gated facade over ``mllog``.

    Event methods no-op on non-zero ranks and when mlperf_logging is absent.
    ``sync=True`` barriers before emit so the timestamp reflects the slowest rank
    (required for INIT_STOP/RUN_START/RUN_STOP).
    """

    def __init__(
        self,
        rank: Optional[int] = None,
        log_path: Optional[str] = None,
        default_stack_offset: int = 2,
        # MLPerf Training 6.1 benchmark short name. Must match the string
        # registered in mlcommons/logging (constants.py, the closed/open_common
        # allow-lists, benchmark_meta and the rcp_checker tables) or the
        # compliance checker rejects the log before reading anything else.
        benchmark_name: str = "dlrmv4",
        submitter_name: str = "AMD",
        # Empty/None = detect the accelerator on first use (rank 0, after the
        # device is set); see detect_submission_platform().
        submission_platform: Optional[str] = None,
        fresh: bool = True,
    ):
        self.enabled: bool = _MLLOG_AVAILABLE
        # Called before every emit. The host-sync-free step (dlrm_v4/
        # host_sync_free.py) parks the latest train_loss here so it is written
        # before any later event, keeping the stream order of the sync path.
        self.pre_emit_hook: Optional[Callable[[], None]] = None
        # Use the EXPLICIT caller rank: this is built before init_process_group,
        # when dist.get_rank() would return 0 on every rank (all would log).
        self.rank: int = rank if rank is not None else _rank()
        self.benchmark_name: str = benchmark_name
        self.submitter_name: str = submitter_name
        self.submission_platform: Optional[str] = submission_platform or None
        self._logger = None
        if not self.enabled:
            return
        # Only rank 0 emits, so only rank 0 needs the file handler.
        if log_path and self.rank == 0:
            log_dir = os.path.dirname(log_path)
            if log_dir:  # guard: os.makedirs("") raises for a bare filename
                os.makedirs(log_dir, exist_ok=True)
            # mllog's FileHandler APPENDS (mode "a"), which is what a resume needs
            # so the single run's event stream accumulates across relaunches into
            # one file. On a genuine cold start, truncate first so a re-used run
            # dir / a previous crashed-cold-start's orphaned stream can't leave a
            # second run_start in the file (the compliance checker requires
            # EXACTLY_ONE). Resume (fresh=False) appends to continue the stream.
            if fresh:
                open(log_path, "w").close()
            mllog.config(filename=log_path, default_stack_offset=default_stack_offset)
        else:
            mllog.config(default_stack_offset=default_stack_offset)
        self._logger = mllog.get_mllogger()

    def _pre_emit(self) -> None:
        hook = self.pre_emit_hook
        if hook is not None:
            hook()

    @property
    def constants(self):  # pyre-ignore[3]
        return mllog_constants

    def event(
        self,
        key: str,
        value: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        sync: bool = False,
    ) -> None:
        self._pre_emit()
        if sync:
            _barrier()
        if self.enabled and self.rank == 0:
            self._logger.event(key=key, value=value, metadata=metadata or {})

    def start(
        self,
        key: str,
        value: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        sync: bool = False,
    ) -> None:
        self._pre_emit()
        if sync:
            _barrier()
        if self.enabled and self.rank == 0:
            self._logger.start(key=key, value=value, metadata=metadata or {})

    def end(
        self,
        key: str,
        value: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        sync: bool = False,
    ) -> None:
        self._pre_emit()
        if sync:
            _barrier()
        if self.enabled and self.rank == 0:
            self._logger.end(key=key, value=value, metadata=metadata or {})

    def submission_info(self) -> None:
        """Emit the five SUBMISSION_* events required for a valid submission."""
        if not (self.enabled and self.rank == 0):
            return
        c = mllog_constants
        self.event(key=c.SUBMISSION_BENCHMARK, value=self.benchmark_name)
        self.event(key=c.SUBMISSION_ORG, value=self.submitter_name)
        self.event(key=c.SUBMISSION_DIVISION, value=c.CLOSED)
        self.event(key=c.SUBMISSION_STATUS, value=c.ONPREM)
        if not self.submission_platform:
            self.submission_platform = detect_submission_platform()
        self.event(key=c.SUBMISSION_PLATFORM, value=self.submission_platform)

    def log_run_config(
        self,
        micro_batch_size: int,
        config_filename: str,
        env: Optional[Mapping[str, str]] = None,
    ) -> None:
        """Emit the nine MLPerf Training 6.1 precision/parallelism/config keys.

        Each is EXACTLY_ONE in training_6.1.0/common.yaml with no phase
        constraint; log_run_start calls this once, before INIT_STOP/RUN_START.
        Precisions are resolved from what is armed in this process (see
        resolve_numerical_precisions), not from a hard-coded default, so an
        FP8 or quantized-comm opt-out logs what actually ran.
        """
        if not (self.enabled and self.rank == 0):
            return
        c = self.constants
        prec = resolve_numerical_precisions(env=env)
        self.event(
            key=_const(c, "LOWEST_NUMERICAL_PRECISION_IN_LINEAR"), value=prec["linear"]
        )
        self.event(
            key=_const(c, "LOWEST_NUMERICAL_PRECISION_IN_ATTN"), value=prec["attn"]
        )
        self.event(
            key=_const(c, "LOWEST_NUMERICAL_PRECISION_IN_COMM"), value=prec["comm"]
        )
        # Pure data parallelism for the dense model (DistributedModelParallel
        # wraps it in DDP; every rank holds the full HSTU stack and its full
        # sequence). The embedding tables are row-wise sharded across ranks,
        # which is the standard DLRM sparse model-parallel layout, not tensor,
        # pipeline, context (sequence) or expert parallelism. Nothing in this
        # trainer can change that, so these are structurally 1.
        for name in (
            "TENSOR_PARALLELISM",
            "PIPELINE_PARALLELISM",
            "CONTEXT_PARALLELISM",
            "EXPERT_PARALLELISM",
        ):
            self.event(key=_const(c, name), value=1)
        self.event(key=_const(c, "MICRO_BATCH_SIZE"), value=int(micro_batch_size))
        if config_filename:
            self.event(
                key=_const(c, "CONFIG_FILENAME"),
                value=os.path.basename(str(config_filename)),
            )

    def log_run_start(
        self,
        global_batch_size: int,
        seed: int,
        gradient_accumulation_steps: int = 1,
        micro_batch_size: Optional[int] = None,
        config_filename: Optional[str] = None,
    ) -> None:
        """Emit submission info + core hyperparameters, then INIT_STOP + RUN_START.

        Optimizer config and the LR-warmup schedule are read from gin (dense Adam
        + sparse RowWiseAdagrad), resolving env-macro refs to concrete values.
        Every hyperparameter the closed division pins must appear here: the
        compliance checker only evaluates a rule when a line carrying that key is
        in the stream, so an unlogged value is an unenforceable one. Call once on
        a genuine cold start, after the model is built. INIT_STOP/RUN_START
        barrier so the timestamp reflects the slowest rank, so ALL ranks must call
        this together (non-rank-0 / disabled calls no-op the emit but still hit
        the barrier).
        """
        c = self.constants
        self.submission_info()
        self.event(key=c.GLOBAL_BATCH_SIZE, value=int(global_batch_size))
        self.event(
            key=c.GRADIENT_ACCUMULATION_STEPS, value=int(gradient_accumulation_steps)
        )
        self.event(key=c.SEED, value=int(seed))
        self.event(
            key=c.OPT_NAME,
            value=_gin_param("dense_optimizer_factory_and_class.optimizer_name", "Adam"),
        )
        self.event(
            key=c.OPT_BASE_LR,
            value=_gin_param("dense_optimizer_factory_and_class.learning_rate", None),
        )
        betas = _gin_param("dense_optimizer_factory_and_class.betas", (0.95, 0.999))
        self.event(key=c.OPT_ADAM_BETA_1, value=float(betas[0]))
        self.event(key=c.OPT_ADAM_BETA_2, value=float(betas[1]))
        self.event(
            key=c.OPT_ADAM_EPSILON,
            value=float(_gin_param("dense_optimizer_factory_and_class.eps", 1e-8)),
        )
        self.event(
            key=c.OPT_WEIGHT_DECAY,
            value=float(
                _gin_param("dense_optimizer_factory_and_class.weight_decay", 0.0)
            ),
        )
        self.event(
            key=c.OPT_GRADIENT_CLIP_NORM,
            value=float(_gin_param("streaming_train_eval_loop.grad_clip_norm", 0.0)),
        )
        self.event(
            key=c.OPT_LR_WARMUP_STEPS,
            value=int(_gin_param("streaming_train_eval_loop.lr_warmup_steps", 0)),
        )
        # No upstream constant for the absolute LR the warmup ramp starts from;
        # registered alongside the sparse keys in the mlcommons/logging PR.
        self.event(
            key="opt_learning_rate_warmup_start_lr",
            value=float(
                _gin_param("streaming_train_eval_loop.lr_warmup_start_lr", 0.0)
            ),
        )
        self.event(
            key="opt_sparse_name",
            value=_gin_param(
                "sparse_optimizer_factory_and_class.optimizer_name", "RowWiseAdagrad"
            ),
        )
        self.event(
            key="opt_sparse_base_learning_rate",
            value=_gin_param(
                "sparse_optimizer_factory_and_class.learning_rate", None
            ),
        )
        self.event(
            key=c.MAX_SEQUENCE_LENGTH,
            value=int(_gin_param("get_hstu_configs.max_seq_len", 4096)),
        )
        # Eval schedule. Convergence is reported on a fixed grid of
        # eval_every_data_pct * train_samples blocks, so samples-to-converge is
        # quantized to that grid and is only comparable against the RCPs if the
        # grid matches. skip_eval_epoch_pct suppresses the early passes that
        # cannot converge; it moves where the grid starts, not its spacing.
        self.event(
            key="eval_every_data_pct",
            value=float(
                _gin_param("streaming_train_eval_loop.eval_every_data_pct", 0.0)
            ),
        )
        self.event(
            key="skip_eval_epoch_pct",
            value=float(
                _gin_param("streaming_train_eval_loop.skip_eval_epoch_pct", 0.0)
            ),
        )
        # MLPerf Training 6.1 disclosure keys (precision, parallelism, micro
        # batch, config file). micro_batch_size is the per-GPU batch of one
        # optimizer micro-step; derive it from the global batch if the caller
        # did not pass it.
        if micro_batch_size is None:
            world = (
                torch.distributed.get_world_size()
                if torch.distributed.is_available()
                and torch.distributed.is_initialized()
                else 1
            )
            micro_batch_size = int(global_batch_size) // max(
                1, world * int(gradient_accumulation_steps)
            )
        if not config_filename:
            logger.warning(
                "MLPerf: no config_filename passed to log_run_start; "
                "config_filename will be missing from the log"
            )
        self.log_run_config(
            micro_batch_size=int(micro_batch_size),
            config_filename=config_filename or "",
        )
        self.end(key=c.INIT_STOP, sync=True)
        self.start(key=c.RUN_START, sync=True)


def _gin_param(name: str, default: Any) -> Any:
    """Read a gin-bound parameter, resolving env-macro refs to concrete values.

    Returns ``default`` if the parameter is unbound or a macro ref cannot be
    resolved (so env-overridden LRs log as numbers, not unencodable objects).
    """
    try:
        value = gin.query_parameter(name)
    except (ValueError, KeyError):
        return default
    if hasattr(value, "scoped_configurable_fn"):
        try:
            return value.scoped_configurable_fn()
        except Exception:
            return default
    return value


def _const(c: Any, name: str) -> str:
    """mllog constant ``name`` (e.g. TENSOR_PARALLELISM) as its key string.

    The 6.1 keys are only in recent mlperf_logging releases; fall back to the
    literal key (the constant's lower-cased name) on an older install, so the
    stream is the same either way.
    """
    return getattr(c, name, None) or name.lower()


# Checker spellings (training_6.1.0/common.yaml allow-list) for the dtypes this
# trainer can run, ordered lowest precision first. bf16 ranks below fp16: it
# has 7 mantissa bits to fp16's 10 (same width, wider exponent). mxfp6 (E2M3
# elements, per-32 E8M0 block scales) ranks below fp8.
_PRECISION_ORDER = ("mxfp6", "fp8", "bfloat16", "fp16", "fp32")
_COMM_PRECISION_NAMES = {"fp32": "fp32", "fp16": "fp16", "bf16": "bfloat16"}
_TRITON_FP8_MODULE = "generative_recommenders.ops.triton.triton_fp8"
_BWD_FP8_MODULE = "generative_recommenders.ops.triton.bwd_fp8"
_MX6_FWD_MODULE = "generative_recommenders.ops.hip.mx6_fwd"

# FP8 linear GEMMs armed by a switch of their own, which neither
# _HSTU_FP8_GEMM nor _BWD_ENABLED sees. The gin binds these on independently of
# DLRM_HSTU_FP8_FWD/BWD, so opting out of both does not turn them off. One row
# per path: (name, gates); the path is armed when every gate is on. A gate is
# (env switch, owning module, attribute). The attribute is a latched bool or a
# zero-arg function returning the module's own resolved gate; it is read when
# env is None and the module is imported. Otherwise the switch is read as
# "1", as for the bwd_fp8 dgrad.
_FP8_LINEAR_PATHS: Tuple[Tuple[str, Tuple[Tuple[str, str, str], ...]], ...] = (
    # E-M3a (yambda_5b.gin item (52), ops/preproc_fp8_l2.py): the content and
    # additional preprocessor MLPs' L2 GEMMs in FP8 (e4m3 forward, e5m2 x e4m3
    # dgrad, FP8 NN wgrad). It runs only inside the E-M1 grouped tail, so
    # that gate comes first.
    (
        "preproc_l2",
        (
            (
                "DLRM_HSTU_FP8_PREPROC_GROUPED_TAIL",
                "generative_recommenders.modules.preprocessors",
                "_PREPROC_GROUPED_TAIL",
            ),
            (
                "DLRM_HSTU_FP8_PREPROC_L2",
                "generative_recommenders.ops.preproc_fp8_l2",
                "enabled",
            ),
        ),
    ),
    # The e5m2 x e4m3 uvqk / targets-only VK projection dgrad (and the fp8t
    # wgrad) in ops/triton/bwd_fp8.py. Gated on its own switch, not on
    # _BWD_ENABLED.
    (
        "uvqk_bwd",
        (("DLRM_HSTU_FP8_BWD_UVQK", _BWD_FP8_MODULE, "uvqk_fp8_backward_enabled"),),
    ),
    # IX-FP8F (ops/triton/triton_ec_indexed_l1_fp8.py): the e4m3 Indexed-X L1
    # forward of the EC preprocessor (per-row U scales, e4m3 W1). It runs only
    # inside IX, so IX's gate comes first. IX alone
    # (DLRM_HSTU_FP8_EC_INDEXED_L1) is bf16 and is not an FP8 path by itself.
    (
        "ix_fp8f",
        (
            (
                "DLRM_HSTU_FP8_EC_INDEXED_L1",
                "generative_recommenders.ops.triton.triton_ec_indexed_l1",
                "enabled",
            ),
            (
                "DLRM_HSTU_FP8_EC_INDEXED_L1_FP8",
                "generative_recommenders.ops.triton.triton_ec_indexed_l1_fp8",
                "fp8f_enabled",
            ),
        ),
    ),
)


def _lowest(*precisions: str) -> str:
    return min(precisions, key=_PRECISION_ORDER.index)


def _gate_on(gate: Tuple[str, str, str], env: Mapping[str, str], live: bool) -> bool:
    """One gate of an _FP8_LINEAR_PATHS row (see there)."""
    switch, module, attr = gate
    mod = sys.modules.get(module) if live else None
    if mod is not None and hasattr(mod, attr):
        value = getattr(mod, attr)
        return bool(value() if callable(value) else value)
    return (env.get(switch) or "0").strip() == "1"


def armed_fp8_linear_paths(env: Optional[Mapping[str, str]] = None) -> Tuple[str, ...]:
    """Names of the _FP8_LINEAR_PATHS rows armed in this process (or in ``env``)."""
    e = os.environ if env is None else env
    return tuple(
        name
        for name, gates in _FP8_LINEAR_PATHS
        if all(_gate_on(g, e, live=env is None) for g in gates)
    )


def resolve_numerical_precisions(
    env: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """Lowest numerical precision actually armed in linear / attn / comm.

    * base dtype: ``make_model.bf16_training`` (bf16 autocast) -> bfloat16,
      else fp32. This is the floor when no FP8 path is on.
    * linear: fp8 when the e4m3 HSTU projection GEMMs are on. That is
      ``_HSTU_FP8_GEMM`` in ops/triton/triton_fp8.py, latched from
      $DLRM_HSTU_FP8_GEMM at import (set by _env_bootstrap from the gin
      DLRM_HSTU_FP8_FWD arm), or when the e4m3 output-projection dgrad is
      armed (``bwd_fp8._BWD_ENABLED`` / $DLRM_HSTU_FP8_BWD), which is gated on
      its own switch rather than on _HSTU_FP8_GEMM, or when any other FP8
      linear path in ``_FP8_LINEAR_PATHS`` is armed (E-M3a preprocessor L2,
      the uvqk/vk projection backward; the IX-FP8F hook goes there too). Those
      are gated independently of DLRM_HSTU_FP8_FWD/BWD and stay bound on by
      the gin when both are opted out. mxfp6 when the MXFP6 forward K512 trio
      is requested on top of the FP8 forward GEMMs
      (``ops/hip/mx6_fwd.requested()`` / $DLRM_HSTU_FP8_FWD_MXFP6, default
      off): U / VQK / VK then run the scaled MFMA on E2M3 operands with per-32
      E8M0 block scales. It is reported whenever requested, because a site
      that declines at runtime falls back to FP8: the disclosure can overstate
      the lowest precision used, never understate it.
    * attn: fp8 when the e4m3 attention forward is on (``_HSTU_FP8_ATTENTION``:
      GEMM and FUSE_LN and FUSE_QKV and ATTN). The fp8 attention backward
      (k128 qk MFMA, C8 full-FP8) only engages on e4m3 q/k from that forward,
      so it does not change the answer.
    * comm: the lowest wire dtype of any collective. The sparse embedding
      all-to-all uses the TorchRec QComms codec configured by
      make_optimizer_and_shard.sparse_a2a_{forward,backward}_precision
      (gin, $SPARSE_A2A_FWD/BWD); the dense DDP gradient allreduce has no
      compression hook and stays fp32.

    When ``env`` is None and the FP8 module is already imported (the trainer
    case: log_run_start runs after make_model), its latched flags are read
    directly, so the answer is what the kernels will do. For E-M3a that is
    preproc_fp8_l2.enabled(), the cached gate GroupedTail itself asks.
    Otherwise (``env`` given, or not imported yet) the same gating chain is
    evaluated on ``env``.
    """
    base = "bfloat16" if bool(_gin_param("make_model.bf16_training", False)) else "fp32"

    e = os.environ if env is None else env

    def on(name: str) -> bool:
        return (e.get(name) or "0").strip() == "1"

    live = sys.modules.get(_TRITON_FP8_MODULE) if env is None else None
    if live is not None and hasattr(live, "_HSTU_FP8_GEMM"):
        fp8_gemm = bool(getattr(live, "_HSTU_FP8_GEMM"))
        fp8_attn = bool(getattr(live, "_HSTU_FP8_ATTENTION", False))
    else:
        fp8_gemm = on("DLRM_HSTU_FP8_GEMM")
        fp8_attn = (
            fp8_gemm
            and on("DLRM_HSTU_FP8_FUSE_LN")
            and on("DLRM_HSTU_FP8_FUSE_QKV")
            and on("DLRM_HSTU_FP8_ATTN")
        )
    # The e4m3 output-projection dgrad (ops/triton/bwd_fp8.py) is gated on
    # $DLRM_HSTU_FP8_BWD alone, not on _HSTU_FP8_GEMM, so FWD=1 with a pre-set
    # DLRM_HSTU_FP8_GEMM=0 still runs an fp8 linear GEMM in the backward.
    live_bwd = sys.modules.get(_BWD_FP8_MODULE) if env is None else None
    if live_bwd is not None and hasattr(live_bwd, "_BWD_ENABLED"):
        fp8_bwd = bool(getattr(live_bwd, "_BWD_ENABLED"))
    else:
        fp8_bwd = on("DLRM_HSTU_FP8_BWD")
    fp8_other = bool(armed_fp8_linear_paths(env))
    # The MXFP6 trio only replaces FP8 forward GEMMs, so it needs fp8_gemm.
    live_mx6 = sys.modules.get(_MX6_FWD_MODULE) if env is None else None
    if live_mx6 is not None and hasattr(live_mx6, "requested"):
        mx6_fwd = bool(live_mx6.requested())
    else:
        # Parsed as the MX6 gates parse it (mx6_fwd.ENABLED, the call site's
        # _MX6_FWD): exactly "1"; any other value leaves every MX6 site off.
        mx6_fwd = e.get("DLRM_HSTU_FP8_FWD_MXFP6", "0") == "1"
    linear = "fp8" if (fp8_gemm or fp8_bwd or fp8_other) else base
    if mx6_fwd and fp8_gemm:
        linear = "mxfp6"

    comm = ["fp32"]  # dense DDP allreduce
    for direction in ("forward", "backward"):
        raw = _gin_param(
            f"make_optimizer_and_shard.sparse_a2a_{direction}_precision", "fp32"
        )
        p = str(raw or "fp32").strip().lower()
        if p not in _COMM_PRECISION_NAMES:
            # make_optimizer_and_shard raises on this too; never guess.
            raise ValueError(f"unknown sparse a2a {direction} precision {raw!r}")
        comm.append(_COMM_PRECISION_NAMES[p])

    return {
        "linear": linear,
        "attn": "fp8" if fp8_attn else base,
        "comm": _lowest(*comm),
    }


def _platform_from_device(name: str, arch: str = "") -> str:
    """Map a device name (and gcnArchName) to a submission platform string."""
    name = (name or "").strip()
    m = re.search(r"\b(MI\d{3}[A-Z]{0,2})\b", name, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    if name.upper().startswith("NVIDIA "):
        return name[len("NVIDIA "):].strip()
    if name:
        return name
    arch = (arch or "").split(":")[0].strip()
    if arch:
        return f"AMD {arch}"
    return "unknown"


def detect_submission_platform(device_name: Optional[str] = None) -> str:
    """Accelerator model of this rank's current device, e.g. ``MI350X``.

    "AMD Instinct MI350X" -> "MI350X". Reads the CURRENT device (rank-local),
    so it is valid before or after init_process_group. Falls back to the raw
    device name, then the gfx arch, then "unknown" (with a warning) so the
    required key is never empty. $MLPERF_SUBMISSION_PLATFORM and a gin
    get_mlperf_logger.submission_platform binding both bypass this.
    """
    arch = ""
    if device_name is None:
        device_name = ""
        try:
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(torch.cuda.current_device())
                device_name = getattr(props, "name", "") or ""
                arch = getattr(props, "gcnArchName", "") or ""
        except Exception as e:  # noqa: BLE001
            logger.warning("MLPerf: accelerator detection failed: %s", e)
    platform = _platform_from_device(device_name, arch)
    if platform == "unknown":
        logger.warning(
            "MLPerf: could not detect the accelerator; logging "
            "submission_platform='unknown' (set $MLPERF_SUBMISSION_PLATFORM)"
        )
    return platform


class MLPerfRunTracker:
    """Centralized MLPerf run-boundary state machine for the streaming loop.

    Owns the block/eval/run markers, the SAMPLES_COUNT/EPOCH_NUM progress
    metadata, and the convergence decision (per-window AUC vs the configured
    ``auc_threshold``). Every method no-ops when ``logger`` is None, so the
    streaming loop can call them unconditionally. The convergence metric is
    fixed to per-window AUC (higher-is-better).
    """

    # MetricsLogger.compute key short name for per-window AUC.
    _EVAL_METRIC_SHORT = "window_auc"

    def __init__(
        self,
        logger: Optional[MLPerfLogger],
        metric_logger: Any,
        total_train_samples: int,
        rank: int,
        device: Any,
    ):
        self.logger = logger
        self.metric_logger = metric_logger
        self.total_train_samples = int(total_train_samples)
        self.rank = int(rank)
        self.device = device
        self.run_stopped: bool = False
        # Idempotency flag so the boundary helpers and the outer loop can both
        # call start/stop without risking a double BLOCK_START/STOP.
        self._block_open: bool = False

    @property
    def enabled(self) -> bool:
        return self.logger is not None

    def _progress(self) -> Dict[str, Any]:
        c = self.logger.constants
        samples = self.metric_logger.cumulative_train_samples
        epoch = (
            samples / self.total_train_samples if self.total_train_samples > 0 else 0.0
        )
        return {c.SAMPLES_COUNT: samples, c.EPOCH_NUM: epoch}

    def log_dataset_sizes(self, eval_samples: Optional[int] = None) -> None:
        if not self.enabled:
            return
        c = self.logger.constants
        self.logger.event(key=c.TRAIN_SAMPLES, value=self.total_train_samples)
        if eval_samples is not None:
            self.logger.event(key=c.EVAL_SAMPLES, value=int(eval_samples))

    def block_start(self) -> None:
        if self.enabled and not self._block_open:
            self.logger.start(
                key=self.logger.constants.BLOCK_START, metadata=self._progress()
            )
            self._block_open = True

    def block_stop(self) -> None:
        if self.enabled and self._block_open:
            self.logger.end(
                key=self.logger.constants.BLOCK_STOP, metadata=self._progress()
            )
            self._block_open = False

    def eval_start(self) -> None:
        if self.enabled:
            self.logger.start(
                key=self.logger.constants.EVAL_START, metadata=self._progress()
            )

    def _target_metric(self, metrics: Dict[str, float]) -> Optional[float]:
        # Key format `metric/{prefix}_{name}/{task}` (see MetricsLogger.compute);
        # match the per-window AUC short name.
        for key, val in metrics.items():
            short = key.split("/")[-2] if "/" in key else key
            if short == self._EVAL_METRIC_SHORT:
                return float(val)
        return None

    def _meets_target(self, value: Optional[float]) -> bool:
        thr = self.metric_logger.auc_threshold
        if value is None or thr is None:
            return False
        return value >= thr

    def run_stop(self, status: object) -> None:
        # Emit RUN_STOP exactly once, with an all-rank barrier so the timestamp
        # reflects the slowest rank (MLPerf requirement).
        if not self.enabled or self.run_stopped:
            return
        c = self.logger.constants
        self.logger.end(
            key=c.RUN_STOP,
            metadata={c.STATUS: status, **self._progress()},
            sync=True,
        )
        self.run_stopped = True
        # SHIP-13 CANDIDATE (dlstart) TSPEC monitor (TRITON_SPEC_MONITOR=1): the
        # per-rank "[triton-spec] monitor post-run_start: compiles=N ..." line at
        # RUN_STOP. No-op when the monitor is off; never raises.
        try:
            from generative_recommenders.dlrm_v4.train import triton_spec as _tspec

            _tspec.on_run_stop()
        except Exception:
            pass

    def eval_stop(self, eval_metrics: Dict[str, float]) -> bool:
        # Emit EVAL_ACCURACY + EVAL_STOP, early SUCCESS RUN_STOP on target.
        # Rank 0 decides + broadcasts the stop bool so all ranks break in lockstep
        # (a per-rank test could diverge and hang the next all-to-all).
        if not self.enabled:
            return False
        c = self.logger.constants
        eval_value = self._target_metric(eval_metrics)
        if eval_value is not None:
            self.logger.event(
                key=c.EVAL_ACCURACY, value=eval_value, metadata=self._progress()
            )
        self.logger.end(key=c.EVAL_STOP, metadata=self._progress())
        decision = torch.zeros(1, device=self.device)
        if self.rank == 0 and not self.run_stopped and self._meets_target(eval_value):
            decision[0] = 1.0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(decision, src=0)
        should_stop = bool(decision.item() > 0.5)
        if should_stop:
            # All ranks agree -> all reach the RUN_STOP barrier together.
            self.run_stop(c.SUCCESS)
        return should_stop

    def finalize(self, final_metrics: Dict[str, float]) -> None:
        # End-of-run RUN_STOP when the target was never crossed: SUCCESS iff the
        # final eval metric meets the target, else ABORTED.
        if not self.enabled or self.run_stopped:
            return
        c = self.logger.constants
        success = self._meets_target(self._target_metric(final_metrics))
        self.run_stop(c.SUCCESS if success else c.ABORTED)


def mlperf_checkpoint_present(ckpt_path: str) -> bool:
    """True iff ``ckpt_path`` resolves to an existing checkpoint (i.e. a resume).

    A dependency-light mirror of ``checkpoint._resolve_latest_subdir`` so
    ``train_ranker`` can decide cold-start vs resume BEFORE the heavy checkpoint
    import + ``setup()``. This gates the one-time INIT_START/RUN_START markers:
    emit them on a genuine cold start only, and never re-emit on a resume
    relaunch (the MLPerf run spans the resume). Matches the loader's resolution:
    empty path or a base dir with no numeric subdirs => cold start.
    """
    if not ckpt_path:
        return False
    base = ckpt_path.rstrip("/")
    # A leaf save (numeric basename) is a resume iff that dir actually exists.
    if os.path.basename(base).isdigit():
        return os.path.isdir(base)
    if not os.path.isdir(base):
        return False
    for name in os.listdir(base):
        if name.isdigit() and os.path.isdir(os.path.join(base, name)):
            return True
    return False


@gin.configurable
def get_mlperf_logger(
    rank: int = 0,
    log_path: str = "",
    benchmark_name: str = "dlrmv4",
    submitter_name: str = "AMD",
    submission_platform: str = "",
    fresh: bool = True,
) -> Optional[MLPerfLogger]:
    """Build a configured :class:`MLPerfLogger`, or ``None`` if unavailable.

    Path defaults to ``$MLPERF_LOG_PATH``. Returns ``None`` (not a disabled
    logger) so callers' ``is not None`` guards cleanly skip logging.

    Disable knob: set ``$MLPERF_LOGGING=0`` (or false/no/off) to turn the whole
    MLPerf event stream off — returns ``None`` on EVERY rank, so the train loop's
    ``is not None`` guards skip emission AND the cross-rank train-loss all-reduce
    in lockstep. Default (unset / "1") = enabled, preserving prior behavior.
    """
    if not _MLLOG_AVAILABLE:
        return None
    if os.environ.get("MLPERF_LOGGING", "1").strip().lower() in (
        "0", "false", "no", "off",
    ):
        logger.info("MLPerf logging disabled via $MLPERF_LOGGING=0")
        return None
    resolved_path = os.environ.get("MLPERF_LOG_PATH", log_path)
    # SUBMISSION_PLATFORM: $MLPERF_SUBMISSION_PLATFORM wins, then a gin binding;
    # empty means detect the real accelerator at submission_info() time (rank
    # 0, after the device is set) rather than hard-coding a model.
    resolved_platform = (
        os.environ.get("MLPERF_SUBMISSION_PLATFORM", "").strip()
        or submission_platform
    )
    return MLPerfLogger(
        rank=rank,
        log_path=resolved_path,
        benchmark_name=benchmark_name,
        submitter_name=submitter_name,
        submission_platform=resolved_platform,
        fresh=fresh,
    )
