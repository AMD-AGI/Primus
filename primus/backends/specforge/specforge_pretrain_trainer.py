###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
SpecForgePretrainTrainer: Primus wrapper for SpecForge draft-model training.

SpecForge spawns and manages its own distributed workers, so this trainer does
not run a training loop in-process. It replaces the Primus process with the
SpecForge CLI via ``os.execvp``, which keeps a single process tree and lets
SpecForge's exit code propagate to the scheduler unchanged.

Train uses ``os.execvp``, so ``cleanup()`` only runs on the error path.
Capture uses ``subprocess.run`` so the trainer can filter hidden-state shards after
``prepare_hidden_states.py`` exits.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any, Optional

from primus.backends.specforge.argument_builder import (
    build_capture_argv,
    build_specforge_argv,
    flatten_overrides,
    specforge_mode,
)
from primus.backends.specforge.stack_preflight import raise_if_issues
from primus.core.trainer.base_trainer import BaseTrainer
from primus.core.utils.module_utils import log_rank_0, warning_rank_0

# SpecForge treats these as one unit: it accepts all of them or none of them.
TORCHRUN_RANK_VARS = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
TORCHRUN_RENDEZVOUS_VARS = ("MASTER_ADDR", "MASTER_PORT")


def align_visible_devices(env=None) -> Optional[tuple]:
    """Point HIP_VISIBLE_DEVICES at the same devices as CUDA_VISIBLE_DEVICES.

    ``runner/helpers/envs/base_env.sh`` derives HIP_VISIBLE_DEVICES from
    ``GPUS_PER_NODE``, which still defaults to 8 when it runs; the pretrain hook
    narrows GPUS_PER_NODE to 1 afterwards, but HIP is never recomputed. So under
    ``primus-cli`` HIP describes the whole node while CUDA_VISIBLE_DEVICES still
    describes what the job actually allocated.

    Train tolerates that, but capture imports SGLang, which imports vLLM, whose
    ROCm platform module refuses the disagreement at import time (job 99815)::

        File "vllm/platforms/rocm.py", line 80, in <module>
            assert val == cuda_val
        AssertionError

    CUDA_VISIBLE_DEVICES wins because Primus never writes it, so it is the only
    one of the two that still carries the caller's intent.

    Returns ``(old, new)`` when HIP was rewritten.
    """

    environ = os.environ if env is None else env
    hip = environ.get("HIP_VISIBLE_DEVICES")
    cuda = environ.get("CUDA_VISIBLE_DEVICES")
    if not hip or not cuda or hip == cuda:
        return None
    environ["HIP_VISIBLE_DEVICES"] = cuda
    return hip, cuda


def clear_partial_distributed_env(env=None) -> list:
    """Hand SpecForge either a complete torchrun environment or none at all.

    ``runner/helpers/envs/primus-env.sh`` exports MASTER_ADDR/MASTER_PORT even
    under RUN_MODE=single, where no ranks exist. SpecForge refuses to start on
    that partial set::

        distributed environment is incomplete;
        present=['MASTER_ADDR', 'MASTER_PORT'],
        missing=['RANK', 'WORLD_SIZE', 'LOCAL_RANK']

    Returns the variables that were removed.
    """

    environ = os.environ if env is None else env
    if all(environ.get(var) for var in TORCHRUN_RANK_VARS):
        return []
    return [var for var in TORCHRUN_RENDEZVOUS_VARS if environ.pop(var, None) is not None]


class SpecForgePretrainTrainer(BaseTrainer):
    """Trainer that hands off to the SpecForge CLI."""

    def __init__(self, backend_args: Any, **kwargs):
        # Current Primus passes BaseModule-style runtime context to every
        # trainer. BaseTrainer filters it for trainers that do not mix in
        # BaseModule, but subclasses still need to accept and forward it.
        super().__init__(backend_args=backend_args, **kwargs)
        self.argv: Optional[list[str]] = None
        self.workdir: Optional[str] = None
        log_rank_0("Initialized SpecForgePretrainTrainer")

    def setup(self):
        log_rank_0("SpecForgePretrainTrainer.setup()")

    def init(self):
        """Build the SpecForge argv and validate the entrypoint is reachable."""

        raise_if_issues(self.backend_args)
        self.mode = specforge_mode(self.backend_args)
        if self.mode == "capture":
            self.argv = build_capture_argv(self.backend_args)
        else:
            self.argv = build_specforge_argv(self.backend_args)
        self.workdir = getattr(self.backend_args, "specforge_root", None)

        if shutil.which(self.argv[0]) is None:
            raise RuntimeError(
                f"[Primus:specforge] Entrypoint '{self.argv[0]}' not found on PATH. "
                "Install SpecForge in the training image, or set "
                "'specforge_entrypoint' in the pre_trainer module config."
            )

        log_rank_0(f"SpecForge command: {' '.join(self.argv)}")
        if self.workdir:
            log_rank_0(f"SpecForge cwd: {self.workdir}")
        else:
            warning_rank_0(
                "[Primus:specforge] No SpecForge root resolved; relative paths inside the "
                "SpecForge config may not resolve. Set SPECFORGE_ROOT or 'specforge_root'."
            )

    def train(self):
        """Hand off to SpecForge. Train replaces this process; capture returns."""

        if self.argv is None:
            raise RuntimeError("SpecForgePretrainTrainer.init() must be called before train().")

        cleared = clear_partial_distributed_env()
        if cleared:
            log_rank_0(f"Cleared partial distributed env so SpecForge owns the launch: {cleared}")

        realigned = align_visible_devices()
        if realigned:
            old, new = realigned
            log_rank_0(f"Aligned HIP_VISIBLE_DEVICES with CUDA_VISIBLE_DEVICES: {old} -> {new}")

        if self.workdir:
            os.chdir(self.workdir)

        if specforge_mode(self.backend_args) == "capture":
            log_rank_0("Handing off to SpecForge capture (prepare_hidden_states.py).")
            completed = subprocess.run(self.argv, check=False)
            if completed.returncode != 0:
                raise SystemExit(completed.returncode)
            capture = flatten_overrides(getattr(self.backend_args, "specforge_capture", None))
            filter_out = capture.get("filter_output_path")
            raw = capture.get("output_path")
            if filter_out and raw:
                from primus.backends.specforge.filter_hidden_states import (
                    filter_dflash_dir,
                )

                block = int(capture.get("filter_block_size") or 16)
                kept, dropped = filter_dflash_dir(raw, filter_out, block_size=block)
                log_rank_0(f"Filtered hidden states: kept {kept}, dropped {dropped} -> {filter_out}")
                if kept < 8:
                    raise SystemExit(f"too few kept samples ({kept}); need at least 8 for one train step")
            # Return so TrainRuntime finishes cleanup with exit 0. SystemExit(0)
            # is a BaseException and used to be wrapped as a training failure.
            return

        log_rank_0("Handing off to SpecForge (exec).")
        os.execvp(self.argv[0], self.argv)
