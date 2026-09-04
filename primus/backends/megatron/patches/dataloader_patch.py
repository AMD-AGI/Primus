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
``num_workers > 0`` and either did not pass a context of its own or asked
for ``fork``.

Overriding an explicit ``fork`` matters because the callers that most need
this do not leave the choice open: Megatron-Energon hardcodes
``multiprocessing_context = "fork"`` in both of its dataloader classes, so
only injecting into callers that passed nothing would skip exactly the
dataloader that segfaults. A caller that deliberately asked for ``spawn``
or ``forkserver`` is left alone, and so is one that passed the context
positionally, where replacing it would collide with the positional
argument.

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


def _caller_context_is_fork(value) -> bool:
    """True when the caller's ``multiprocessing_context`` resolves to fork.

    ``None`` counts: it asks for the default, and the default on Linux is fork,
    which is the case this patch exists to fix.
    """
    if value is None:
        return True
    if isinstance(value, str):
        return value == "fork"
    get_start_method = getattr(value, "get_start_method", None)
    if callable(get_start_method):
        try:
            return get_start_method() == "fork"
        except Exception:
            return False
    # Some other context object we cannot interpret; assume it was deliberate.
    return False


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
        if int(bound.arguments.get("num_workers", 0)) > 0:
            if "multiprocessing_context" not in bound.arguments:
                kwargs["multiprocessing_context"] = mp_context
                log_rank_0(f"Setting DataLoader multiprocessing_context='{mp_context}'.")
            elif "multiprocessing_context" in kwargs and _caller_context_is_fork(
                kwargs["multiprocessing_context"]
            ):
                requested = kwargs["multiprocessing_context"]
                kwargs["multiprocessing_context"] = mp_context
                log_rank_0(
                    f"Overriding DataLoader multiprocessing_context={requested!r} "
                    f"with '{mp_context}'."
                )
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
        "fork()-hostile native state (RDMA MRs, HIP runtime, IPC handles). "
        "Also overrides callers that hardcode fork, such as Energon."
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
