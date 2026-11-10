###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Run the DLRMv4 MLPerf trainer inside the Primus trainer lifecycle.

The trainer spawns its own ranks (one process per GPU, torch.multiprocessing
spawn), so this backend runs as ONE Primus process per node: the prepare hook
(runner/helpers/hooks/train/pretrain/recommendation/prepare.py) selects
`primus-cli direct` single mode. NNODES / NODE_RANK / MASTER_ADDR / MASTER_PORT /
GPUS_PER_NODE reach the trainer through the environment, as with its own
run_and_time.sh.

setup(): environment for the run shape (config -> env vars the gin file reads),
         data and toolchain checks.
init():  the gin file; the MLPerf cache flush when clear_caches is set.
train(): the trainer's launcher; with warm_cache_after_run_start, a background
         reader streams the dataset into the page cache once run_start is logged
         (the "C1" cold-start fix: all of that I/O is inside the timed region).
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

from primus.backends.recommendation.dlrm_v4 import run_config
from primus.core.backend.env_registry import ARCH_GFX950, EnvVar
from primus.core.trainer.base_trainer import BaseTrainer
from primus.core.utils.module_utils import log_rank_0, warning_rank_0
from primus.core.utils.yaml_utils import nested_namespace_to_dict

# The DLRMv4 trainer package (generative_recommenders), imported into Primus.
SRC_ROOT = Path(__file__).resolve().parent / "src"

# The recipe's environment (docs/04-technical-guides/recommendation/dlrm_v4/rocm-port.md).
# setdefault semantics: an image ENV, an outer export or the config `env:` block wins.
ENV_DEFAULTS = [
    EnvVar("HSTU_HAMMER_KERNEL", "TRITON", note="HSTU Triton kernels"),
    EnvVar("FLYDSL_GPU_ARCH", "gfx950", arch=ARCH_GFX950, note="FlyDSL target"),
    EnvVar("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
    EnvVar("FBGEMM_TBE_ROCM_HIP_BACKWARD_KERNEL", "1"),
    # numpy's MADV_HUGEPAGE on loader arrays stalls every rank when THP defrag
    # is `madvise` on a fragmented node (rocm-port.md T29).
    EnvVar("NUMPY_MADVISE_HUGEPAGE", "0", note="no THP for numpy arrays"),
]


class DlrmV4Trainer(BaseTrainer):
    def __init__(self, backend_args: Any = None, **kwargs):
        # BaseTrainer drops the runtime context unless BaseModule is in the MRO.
        self._primus_config = kwargs.get("primus_config")
        super().__init__(backend_args=backend_args, **kwargs)
        self.params = (
            backend_args if isinstance(backend_args, dict) else nested_namespace_to_dict(backend_args)
        )
        self.root: Optional[Path] = None
        self.out: Optional[Path] = None
        self._launch_cwd: Optional[str] = None
        self.gin_path: str = ""
        self.mlperf_log: str = ""
        self.run_start_marker: str = ""
        self._warm_stop = threading.Event()
        self._warm_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ helpers

    def _p(self, key: str, default: Any = None) -> Any:
        value = self.params.get(key, default)
        return default if value is None or value == "" else value

    def _backend_root(self) -> Path:
        """The trainer source root (the folder holding generative_recommenders/),
        put first on sys.path. Only that package lives there, so nothing else in
        it becomes importable as a top-level module."""
        root = Path(self._p("trainer_src") or SRC_ROOT).expanduser().resolve()
        if not (root / "generative_recommenders").is_dir():
            raise FileNotFoundError(f"[Primus:dlrm_v4] no generative_recommenders/ under {root}")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        log_rank_0(f"[Primus:dlrm_v4] trainer source: {root}")
        return root

    def _output_dir(self) -> Path:
        out = self._p("output_dir")
        if not out:
            exp_root = getattr(self._primus_config, "exp_root_path", None)
            out = os.path.join(exp_root or os.getcwd(), "dlrm_v4")
        path = Path(out).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ---------------------------------------------------------------- lifecycle

    def setup(self):
        self.root = self._backend_root()
        # Resolved before the chdir below: a relative workspace is relative to the launch directory.
        self.out = out = self._output_dir()

        env = run_config.build_env(self.params)
        if self._p("mlperf_logging", True) and "MLPERF_LOG_PATH" not in env:
            env["MLPERF_LOG_PATH"] = run_config.default_mlperf_log(str(out))
        for name, value in env.items():
            os.environ[name] = value
            log_rank_0(f"[Primus:dlrm_v4] {name}={value}")
        self.mlperf_log = os.environ.get("MLPERF_LOG_PATH", "")
        if self.mlperf_log:
            Path(self.mlperf_log).parent.mkdir(parents=True, exist_ok=True)
        # Per-node RUN_START marker for the in-run warm: the trainer's local rank 0
        # of each node writes it right after RUN_START. Only global rank 0 writes the
        # MLPerf log, so with one log per node, nodes 1+ never saw run_start and never
        # warmed; and a shared log must not be deleted by another node.
        out.mkdir(parents=True, exist_ok=True)
        self.run_start_marker = str(out / f"run_start.node{os.environ.get('NODE_RANK', '0')}")
        os.environ["DLRM_RUN_START_MARKER"] = self.run_start_marker

        missing = run_config.missing_data(os.environ.get("DLRM_DATA_PATH"), self._p("history_length"))
        if missing:
            raise FileNotFoundError(
                "[Primus:dlrm_v4] dataset not ready (see the recipe README for download/staging): "
                + "; ".join(missing)
            )
        # The trainer's runtime-built HIP kernels call /opt/rocm/bin/hipcc and use
        # /opt/rocm/{include,lib}; TheRock images keep ROCm in the venv (the install
        # hook links it).
        if not os.path.exists("/opt/rocm/bin/hipcc"):
            raise FileNotFoundError(
                "[Primus:dlrm_v4] /opt/rocm/bin/hipcc not found; link /opt/rocm to $ROCM_PATH "
                "(runner/helpers/hooks/train/pretrain/recommendation/00_install_requirements.sh does)"
            )

        # Spawned ranks resolve the trainer package and its relative table paths
        # (e.g. the Triton spec replay table) from the trainer root. The private
        # package directory (packages whose image version the trainer cannot use,
        # installed by the hook) goes first, ahead of the image's site-packages.
        self._launch_cwd = os.getcwd()
        os.chdir(self.root)
        front = [str(self.root)]
        private = self._p("private_site")
        if private and os.path.isdir(private):
            front.insert(0, private)
            sys.path.insert(0, private)
        os.environ["PYTHONPATH"] = os.pathsep.join(front + [os.environ.get("PYTHONPATH", "")]).rstrip(
            os.pathsep
        )
        flydsl = importlib.util.find_spec("flydsl")
        log_rank_0(
            f"[Primus:dlrm_v4] PYTHONPATH front: {front}; flydsl from {flydsl.origin if flydsl else None}"
        )

    def init(self):
        assert self.root is not None, "setup() must run first"
        base = run_config.resolve_gin_file(self.root, self._p("dataset", "yambda-5b"), self._p("gin_file"))
        if not base.is_file():
            raise FileNotFoundError(f"[Primus:dlrm_v4] gin file not found: {base}")
        bindings = self._p("gin_bindings", []) or []
        if bindings:
            path = self.out / "dlrm_v4.gin"
            path.write_text(run_config.gin_file_text(str(base), bindings))
            self.gin_path = str(path)
            log_rank_0(f"[Primus:dlrm_v4] gin: {path} (includes {base}, {len(bindings)} bindings)")
        else:
            self.gin_path = str(base)
            log_rank_0(f"[Primus:dlrm_v4] gin: {base}")

        cleared = "0"
        if self._p("clear_caches", False):
            self._pagecache("flush")
            cleared = "1"
        os.environ["MLPERF_CACHE_CLEARED"] = cleared

    def train(self):
        from generative_recommenders.dlrm_v4.train import train_ranker

        if self._p("warm_cache_after_run_start", False) and self.run_start_marker:
            # A marker left over from an earlier run must not trigger the warm early.
            if os.path.exists(self.run_start_marker):
                os.remove(self.run_start_marker)
            # Node 0 starts a fresh MLPerf log, as before; global rank 0 writes it, so
            # another node must leave it alone (on a shared path it would unlink the
            # log node 0 is writing).
            if (
                self.mlperf_log
                and os.environ.get("NODE_RANK", "0") == "0"
                and os.path.exists(self.mlperf_log)
            ):
                os.remove(self.mlperf_log)
            self._warm_thread = threading.Thread(target=self._warm_after_run_start, daemon=True)
            self._warm_thread.start()
        try:
            train_ranker.launch(
                self._p("dataset", "yambda-5b"),
                self._p("mode", "streaming-train-eval"),
                self.gin_path,
            )
        finally:
            self._warm_stop.set()
        if self.mlperf_log:
            summary = run_config.run_summary(self.mlperf_log)
            log_rank_0(f"[Primus:dlrm_v4] MLPerf summary: {summary} ({self.mlperf_log})")

    def cleanup(self, on_error: bool = False):
        self._warm_stop.set()
        if self._launch_cwd:  # the ranks are gone; Primus continues where it started
            os.chdir(self._launch_cwd)

    # ------------------------------------------------------------- page cache

    def _cache_dir(self) -> str:
        return str(run_config.cache_dir(os.environ["DLRM_DATA_PATH"], self._p("history_length")))

    def _pagecache(self, mode: str) -> None:
        """pagecache.py <mode> on the staged cache, as its own process (the warm
        reader runs while training). Its command line must not contain
        "generative_recommenders": the trainer's tooling cleans up stray trainer
        processes with `pkill -f generative_recommenders`."""
        script = str(Path(__file__).resolve().parent / "pagecache.py")
        log_rank_0(f"[Primus:dlrm_v4] page cache {mode}: {self._cache_dir()}")
        result = subprocess.run([sys.executable, script, mode, self._cache_dir()])
        if result.returncode != 0:
            raise RuntimeError(f"[Primus:dlrm_v4] page cache {mode} failed (exit {result.returncode})")

    def _warm_after_run_start(self) -> None:
        while not self._warm_stop.is_set():
            if os.path.exists(self.run_start_marker):
                break
            time.sleep(0.5)
        else:
            return
        try:
            log_rank_0("[Primus:dlrm_v4] run_start logged; warming the page cache")
            self._pagecache("warm")
        except Exception as error:  # the run goes on cold rather than failing
            warning_rank_0(f"[Primus:dlrm_v4] page cache warm failed: {error}")
