###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
DataLoader multiprocessing start-method patch.

PyTorch's ``DataLoader`` defaults to ``fork`` on Linux, which is unsafe
once the main process has initialized fork-hostile native state (HIP/CUDA
runtimes, RDMA MRs with MADV_DONTFORK, long-running pthreads, CUDA/HIP IPC
handles, ...). Workers forked in that state typically SIGSEGV.

This patch monkey-patches ``DataLoader.__init__`` during the ``setup``
phase and injects ``multiprocessing_context=<value>`` when the caller has
``num_workers > 0`` and did not pass a context of its own.

Config (YAML module param, mirrors PyTorch's DataLoader argument):
    multiprocessing_context: "forkserver" | "spawn" | "fork" | null

``null`` leaves PyTorch's default behavior untouched.
"""

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

# Guard against double-patching (e.g. when the patch system runs twice in tests).
_PATCHED_ATTR = "_primus_dataloader_mp_context_patched"

_PARAM_NAME = "dataloader_mp_context"


def _get_mp_context(ctx: PatchContext):
    """Return the configured multiprocessing_context, or None."""
    try:
        args = get_args(ctx)
    except AssertionError:
        return None
    return getattr(args, _PARAM_NAME, None)


def _enabled(ctx: PatchContext) -> bool:
    return _get_mp_context(ctx) is not None


def _preload_forkserver_torch() -> None:
    """Preload ``torch`` in the forkserver daemon to speed up worker startup.
    Pure optimization; safe to skip."""
    try:
        import torch.multiprocessing as tmp

        tmp.get_context("forkserver").set_forkserver_preload(["torch"])
    except Exception:
        pass


_MEGATRON_WORKER_INIT_QUALNAME = "build_pretraining_data_loader.<locals>.worker_init_fn"


def _megatron_worker_init(_worker_id, exit_signal=None) -> None:
    """Picklable copy of the local ``worker_init_fn`` Megatron-core 0.19+ always install."""
    import os

    for fd in os.listdir("/proc/self/fd"):
        try:
            if os.readlink(f"/proc/self/fd/{fd}").startswith("/dev/nvidia"):
                os.close(int(fd))
        except OSError:
            pass
    if exit_signal is not None:
        from megatron.training.dist_signal_handler import DistributedSignalHandler

        DistributedSignalHandler(exit_signal).__enter__()


def _picklable_worker_init_fn(worker_init_fn):
    """Spawn and forkserver pickle ``worker_init_fn``; swap out Megatron's local one."""
    if getattr(worker_init_fn, "__qualname__", None) != _MEGATRON_WORKER_INIT_QUALNAME:
        return worker_init_fn
    import functools

    from megatron.training import get_args as get_megatron_args

    margs = get_megatron_args()
    exit_signal = margs.exit_signal if getattr(margs, "exit_signal_handler", False) else None
    return functools.partial(_megatron_worker_init, exit_signal=exit_signal)


def _install_dataloader_monkeypatch(mp_context) -> None:
    """Patch ``DataLoader.__init__`` to inject ``multiprocessing_context``
    when the caller has ``num_workers > 0`` and did not set one."""
    import inspect

    from torch.utils.data import DataLoader

    if getattr(DataLoader.__init__, _PATCHED_ATTR, False):
        log_rank_0("[Patch:megatron.dataloader_mp_context] DataLoader.__init__ " "already patched; skipping.")
        return

    original_init = DataLoader.__init__
    sig = inspect.signature(original_init)

    def patched_init(self, *args, **kwargs):
        # Resolve args against the real signature to stay version-agnostic.
        bound = sig.bind_partial(self, *args, **kwargs)
        if (
            int(bound.arguments.get("num_workers", 0)) > 0
            and "multiprocessing_context" not in bound.arguments
        ):
            kwargs["multiprocessing_context"] = mp_context
            if mp_context != "fork" and "worker_init_fn" in kwargs:
                kwargs["worker_init_fn"] = _picklable_worker_init_fn(kwargs["worker_init_fn"])
            log_rank_0(f"Setting DataLoader multiprocessing_context='{mp_context}'.")
        return original_init(self, *args, **kwargs)

    setattr(patched_init, _PATCHED_ATTR, True)
    DataLoader.__init__ = patched_init


@register_patch(
    "megatron.dataloader_mp_context",
    backend="megatron",
    phase="before_train",
    priority=50,  # Must run before any DataLoader is built.
    description=(
        "Set DataLoader.multiprocessing_context from the "
        "'multiprocessing_context' module param to avoid SIGSEGV caused by "
        "fork()-hostile native state (RDMA MRs, HIP runtime, IPC handles)."
    ),
    condition=_enabled,
)
def patch_dataloader(ctx: PatchContext) -> None:
    mp_context = _get_mp_context(ctx)
    if mp_context is None:
        return

    log_rank_0(
        f"[Patch:megatron.dataloader_mp_context] "
        f"Setting DataLoader multiprocessing_context='{mp_context}'."
    )

    if mp_context == "forkserver":
        _preload_forkserver_torch()
    _install_dataloader_monkeypatch(mp_context)
