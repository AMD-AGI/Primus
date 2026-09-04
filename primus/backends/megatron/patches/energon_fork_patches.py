###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Make the Energon dataloader survive a non-fork start method.

GPU-Direct RDMA registers GPU memory through dmabuf, and a process holding such
a registration cannot ``fork``: the workers inherit device mappings that do not
survive the fork and segfault immediately, inside ``os.fork`` itself. Energon
7.3.2 hardcodes ``fork`` in both of its dataloader classes, so enabling GDR
crashed as soon as the dataloader built its workers.

``dataloader_patch`` handles the start method itself, by overriding the context
Energon asks for. What is left is that ``fork`` shared the dataset by memory
while ``forkserver`` and ``spawn`` must pickle it, which Energon's dataset is
not able to do: it holds a process group, a debug file handle, two lambdas
defined in a class body and three closures built inside ``__init__``, none of
which pickle resolves by qualified name.

This patch installs picklable equivalents from
``primus.backends.megatron.data.energon_picklable``. Two are plain rebinds; the
closures are built in the middle of ``__init__`` where nothing can reach them
from outside, so those need the source-rewrite helper.

Enabled by the same ``dataloader_mp_context`` module param that drives
``dataloader_patch``, since a fork-free start method is the only reason to want
any of this.

An installed tree that already carries the equivalent source changes -- the
vendored ``energon-7.3.2-no-fork.patch`` in tiger-training-internal applies them
to site-packages -- is detected and left alone, so the two mechanisms can
coexist while one is being retired. A tree that has neither the expected
lambdas nor the replacements is a version mismatch and fails loudly here rather
than segfaulting later.
"""

import importlib.util

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

from ._source_patch_utils import patch_method_source_multi

_PARAM_NAME = "dataloader_mp_context"
_PATCHED_ATTR = "_primus_energon_no_fork_patched"
_LOG = "[Patch:megatron.energon_no_fork]"


def _mp_context(ctx: PatchContext):
    """Return the configured start method, or None."""
    try:
        args = get_args(ctx)
    except AssertionError:
        return None
    return getattr(args, _PARAM_NAME, None)


def _enabled(ctx: PatchContext) -> bool:
    """Only for an Energon run that is actually leaving fork behind."""
    mp_context = _mp_context(ctx)
    if mp_context is None or mp_context == "fork":
        return False
    return importlib.util.find_spec("megatron.energon") is not None


def _rewrite_or_skip(label: str, cls, replacements, applied_marker: str) -> None:
    """Rewrite ``cls.__init__``, unless the installed tree already has it.

    Raises:
        AssertionError: If the source shows neither the expected lambdas nor the
            replacements, or shows a mix of both.
    """
    import inspect

    source = inspect.getsource(cls.__init__)
    present = [anchor for anchor, _ in replacements if anchor in source]

    if len(present) == len(replacements):
        patch_method_source_multi(cls, "__init__", replacements)
        log_rank_0(f"{_LOG} rewrote {label}.__init__ ({len(replacements)} fragments)")
        return

    if not present and applied_marker in source:
        log_rank_0(f"{_LOG} {label}.__init__ already carries the fix on disk; leaving it")
        return

    raise AssertionError(
        f"{_LOG} {label}.__init__ matched {len(present)} of {len(replacements)} "
        f"expected fragments and {'does' if applied_marker in source else 'does not'} "
        f"contain {applied_marker!r}. The installed Energon differs from 7.3.2; "
        f"refusing to guess, because getting this wrong segfaults the workers."
    )


def _patch_crude(picklable) -> None:
    """CrudeWebdataset: an identity sample loader and a default part filter."""
    from megatron.energon.flavors import crude

    crude._identity_sample_loader = picklable.identity_sample_loader
    _rewrite_or_skip(
        "CrudeWebdataset",
        crude.CrudeWebdataset,
        [
            (
                "            sample_loader=lambda sample: sample,\n",
                "            sample_loader=_identity_sample_loader,\n",
            )
        ],
        "_identity_sample_loader",
    )

    # part_filter's default is a lambda in the signature, so it is a value rather
    # than code: swap the value instead of rewriting the source.
    init = crude.CrudeWebdataset.__init__
    kwdefaults = init.__kwdefaults__
    assert kwdefaults is not None and "part_filter" in kwdefaults, (
        f"{_LOG} CrudeWebdataset.__init__ has no keyword-only part_filter default; "
        f"installed Energon differs from 7.3.2."
    )
    init.__kwdefaults__ = {**kwdefaults, "part_filter": picklable.accept_all_parts}
    log_rank_0(f"{_LOG} replaced the CrudeWebdataset part_filter default")


def _patch_generic_webdataset(picklable) -> None:
    """DefaultGenericWebdatasetFactory: three closures built inside __init__."""
    from megatron.energon.flavors.webdataset import default_generic_webdataset as dgw

    dgw._PartInSet = picklable.PartInSet
    dgw._FieldMapSampleLoader = picklable.FieldMapSampleLoader
    dgw._KeyedSampleLoader = picklable.KeyedSampleLoader

    # Both part_filter closures are the same line at different depths, so each
    # anchor carries the line above it to stay unique.
    _rewrite_or_skip(
        "DefaultGenericWebdatasetFactory",
        dgw.DefaultGenericWebdatasetFactory,
        [
            (
                "                parts = set(part_filter)\n"
                "                part_filter = lambda part: part in parts\n",
                "                parts = set(part_filter)\n"
                "                part_filter = _PartInSet(parts)\n",
            ),
            (
                "            self._sample_loader = lambda sample: {\n"
                "                k: field_access(sample, v) for k, v in fields.items()\n"
                "            }\n",
                "            self._sample_loader = _FieldMapSampleLoader(fields)\n",
            ),
            (
                "            parts = set(access[0] for options in fields.values() for access in options)\n"
                "            part_filter = lambda part: part in parts\n",
                "            parts = set(access[0] for options in fields.values() for access in options)\n"
                "            part_filter = _PartInSet(parts)\n",
            ),
            (
                "        self._sample_loader = lambda sample: {\n"
                '            "__key__": sample["__key__"],\n'
                "            **inner_sample_loader(sample),\n"
                '            "__restore_key__": sample["__restore_key__"],\n'
                '            "__subflavors__": self.subflavors,\n'
                '            "__sources__": sample["__sources__"],\n'
                "        }\n",
                "        self._sample_loader = _KeyedSampleLoader(inner_sample_loader, self)\n",
            ),
        ],
        "_PartInSet",
    )


def _pin_start_method(mp_context: str) -> None:
    """Pin the process-wide start method as well as the dataloader's.

    Passing a context to the dataloader is not sufficient on its own. Anything
    built earlier under the default context stays bound to it, and handing such
    an object to a worker in another context fails with "A SemLock created in a
    fork context is being shared with a process in a spawn context". Pinning the
    default here, before any dataloader exists, is what the deployment used a
    site-packages ``.pth`` hook for.
    """
    import multiprocessing

    current = multiprocessing.get_start_method(allow_none=True)
    if current == mp_context:
        return
    multiprocessing.set_start_method(mp_context, force=True)
    log_rank_0(f"{_LOG} process start method {current!r} -> {mp_context!r}")


@register_patch(
    "megatron.energon_no_fork",
    backend="megatron",
    phase="before_train",
    priority=50,  # Must run before the dataloader is built.
    description=(
        "Make Energon's dataset picklable so its dataloader can run under "
        "forkserver/spawn, which GPU-Direct RDMA requires because a process "
        "holding a dmabuf registration cannot fork."
    ),
    condition=_enabled,
)
def patch_energon_no_fork(ctx: PatchContext) -> None:
    from megatron.energon.worker import WorkerConfig

    from primus.backends.megatron.data import energon_picklable

    if getattr(WorkerConfig, _PATCHED_ATTR, False):
        log_rank_0(f"{_LOG} already applied; skipping")
        return

    mp_context = _mp_context(ctx)
    _pin_start_method(mp_context)

    # WorkerConfig holds a process group and a debug file handle, neither of
    # which pickles and neither of which a worker needs.
    WorkerConfig.__getstate__ = energon_picklable.worker_config_getstate
    log_rank_0(f"{_LOG} installed WorkerConfig.__getstate__")

    _patch_crude(energon_picklable)
    _patch_generic_webdataset(energon_picklable)

    setattr(WorkerConfig, _PATCHED_ATTR, True)
    log_rank_0(f"{_LOG} done; Energon dataset is picklable for '{mp_context}'")
