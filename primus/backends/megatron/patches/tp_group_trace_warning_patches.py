###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Remove the ``tp_group is None`` deprecation warning from traced forwards.

WHAT THIS FIXES. ``RowParallelLinear.forward`` passes the tensor-parallel group
as a literal ``None`` (``megatron/core/tensor_parallel/layers.py:1265``), unlike
``ColumnParallelLinear.forward``, which passes ``tp_group=self.tp_group``
(line 1017). The ``None`` reaches the deprecation shim
``get_tensor_model_parallel_group_if_none`` (``megatron/core/utils.py:520``),
whose ``tp_group is None`` branch calls ``warnings.warn`` on rank 0.

``warnings.warn`` is the C builtin ``_warnings.warn``, which Dynamo cannot trace
-- it reports "Dynamo does not know how to trace the Python builtin
`_warnings.warn`" and takes a hard graph break. Under ``torch.compile`` that is
one break in every ``RowParallelLinear`` forward, so every block containing a
row-parallel linear is split, and Dynamo must cut each frame between the warning
and the root of the compiled region. A transformer block typically holds several
row-parallel linears (the attention output projection and the second MLP
linear), and activation recomputation replays each forward, so the number of
split frames grows with depth rather than staying constant.

The argument is dead on that path in every configuration: the functional API
only uses ``tp_group`` for a collective when ``allreduce_dgrad`` or
``sequence_parallel`` is set, and ``RowParallelLinear.forward`` hardcodes both to
``False`` (lines 1247 and 1264). The row-parallel output all-reduce that does
need a group uses the correctly resolved ``self.tp_group`` afterwards. So the
break buys nothing at all -- it exists only to emit a warning about an argument
that is never used, and Python's warning registry then dedupes that warning
after its first emission while the graph split is paid on every step forever.

None of this is about tensor parallelism. The branch tests the ``None``
sentinel, not the TP world size, so it fires identically at TP=1.

HOW IT FIXES IT. The only untraceable thing in the shim is the ``warnings.warn``
call itself; Dynamo traces everything ahead of it and breaks exactly there. So
the shim is left alone and its ``warnings`` module global is replaced with a
stand-in whose ``warn`` is a plain Python function that returns early while
Dynamo is tracing. Dynamo inlines a plain Python function instead of breaking on
it, and ``torch.compiler.is_compiling()`` is constant-folded to ``True`` during
tracing, so the early return is all that survives into the graph. In eager the
stand-in forwards to the real ``warnings.warn`` and the deprecation notice is
emitted verbatim -- deliberately narrower than deleting the warning, because the
notice is a real upstream signal (the shim is marked
``# TODO(zijiey): remove this function later.``) and should keep reaching anyone
running eager.

Patching the module global rather than reimplementing the shim keeps upstream's
group-resolution logic authoritative: this patch makes no assumption about how
the group is resolved, so an upstream refactor of that order cannot silently
change what a compiled run resolves to. It also covers every caller of the shim
(``layers.py:229``, ``:419``, ``:693`` and the four call sites in
``extensions/transformer_engine.py``) without editing the Megatron-LM submodule,
because a module global is shared by every reference to the function.
``megatron/core/utils.py:532`` is the only ``warnings.warn`` call site in that
module, so nothing else is affected.

The swap is installed once at ``before_train`` and is never restored, which is
what makes it safe to leave alone: the stand-in holds no state beyond the module
it delegates to, adds no locking of its own, and re-entering it while tracing
returns before touching anything, so there is no ordering between ranks or
threads to get wrong. Installing twice is a no-op rather than a stand-in wrapped
around a stand-in.

The patch installs only while the defect is present. ``_defect_present`` reads
the shim's source and applies the stand-in only if it still contains the
``warnings.warn`` branch, so the patch self-disables the moment upstream removes
or fixes the shim -- on any Megatron-LM version, without a version table to
maintain. Version detection would not work here anyway: Megatron-LM is vendored
as a submodule pinned by commit rather than pip-installed, so
``importlib.metadata.version("megatron-core")`` raises ``PackageNotFoundError``,
and ``megatron.core.__version__`` does not move when the shim changes. The
installed version is logged for diagnostics.

That is also why the framework's own gate, ``backend_versions=[...]`` on
``register_patch`` (as ``megatron.turbo.te_spec_provider`` uses it), is left
unset here. The shim carries ``# TODO(zijiey): remove this function later.``
with no announced removal version, so any bound would be a guess in both
directions: it would keep patching if the warning were fixed inside the pinned
range, and stop patching if the warning outlived it. Reading the shim is exact
in both.

The hardcoded ``None`` at ``layers.py:1265`` is still worth fixing upstream on
its own merits; this patch does not stand in for that.

SCOPE. This is not specific to a model or to a distributed wrapper. Any model
built on Megatron's row-parallel linear hits it under ``torch.compile``, and it
behaves the same under Megatron-FSDP, torch FSDP2 and DDP. It was found on the
diffusion backbones (Wan and Flux), where per-block compilation makes the split
frames easy to see, but nothing about it is diffusion-specific.

Enabled by default: skipping a deprecation warning about an unused argument
while tracing cannot change results, and eager still warns. Set the environment
variable to a false value to restore the unpatched shim, which is useful for
A/B measurement of the graph breaks themselves.

Environment:
    PRIMUS_TP_GROUP_TRACE_FIX=0   disable (default: enabled)
"""

import inspect
import os
from types import ModuleType

import torch

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0

PATCH_ID = "megatron.tp_group_trace_warning"
ENV_VAR = "PRIMUS_TP_GROUP_TRACE_FIX"
_FUNC_NAME = "get_tensor_model_parallel_group_if_none"
_GLOBAL_NAME = "warnings"


class _WarningsSilentWhileTracing:
    """Stand-in for the ``warnings`` module global in ``megatron.core.utils``.

    ``warn`` is a plain Python function, which Dynamo inlines, so replacing the
    module with this object turns the one call Dynamo cannot trace into one it
    can. Everything else defers to the real module.
    """

    def __init__(self, delegate: ModuleType):
        self._delegate = delegate

    def warn(self, *args, **kwargs):
        # Written out in full rather than through an alias so Dynamo recognises
        # the call and folds it to True while tracing.
        if torch.compiler.is_compiling():
            return None

        # Everything is forwarded as given rather than restated through a
        # signature of our own: the builtin declares ``category=None`` and
        # normalises it itself, derives the category from a Warning instance,
        # takes ``source`` positionally and ``skip_file_prefixes`` keyword-only.
        # The one argument that has to change is stacklevel, +1 for this frame,
        # so the notice still points at the caller's line.
        if len(args) >= 3:
            args = (*args[:2], args[2] + 1, *args[3:])
        else:
            kwargs["stacklevel"] = kwargs.get("stacklevel", 1) + 1
        return self._delegate.warn(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._delegate, name)


def _truthy(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def _defect_present() -> bool:
    """Is the untraceable warning still in the upstream shim?

    Reading the shim itself keeps the patch correct across Megatron-LM revisions
    without pinning versions: no shim, no warning branch, or an already-replaced
    global all mean there is nothing to do. Source first, the shim's bytecode as
    a fallback where the source cannot be read.
    """
    try:
        from megatron.core import utils as mcore_utils
    except ImportError:
        return False

    shim = getattr(mcore_utils, _FUNC_NAME, None)
    if shim is None:
        return False

    # Not a module means either an earlier install of this patch or an upstream
    # rename; in both cases leave it alone.
    if not isinstance(getattr(mcore_utils, _GLOBAL_NAME, None), ModuleType):
        return False

    try:
        return f"{_GLOBAL_NAME}.warn" in inspect.getsource(shim)
    except (OSError, TypeError):
        # No .py alongside the module, so fall back to the names the shim's
        # bytecode resolves. Coarser than reading the source -- it cannot tell
        # which branch the call sits in -- but it keeps the patch working on a
        # source-free install rather than silently disabling itself there.
        names = getattr(getattr(shim, "__code__", None), "co_names", ())
        return _GLOBAL_NAME in names and "warn" in names


def _needs_patch(ctx: PatchContext) -> bool:
    del ctx
    if not _truthy(os.getenv(ENV_VAR, "1")):
        return False
    return _defect_present()


@register_patch(
    PATCH_ID,
    backend="megatron",
    phase="before_train",
    description=f"Drop the tp_group=None deprecation warning while tracing ({ENV_VAR}=0 to disable)",
    # Default priority: the swap only has to land before the first forward is
    # traced, which is well after before_train, so ordering is not load-bearing.
    condition=_needs_patch,
)
def patch_tp_group_trace_warning(ctx: PatchContext):
    """Make the deprecation warning in the tp_group shim traceable."""
    del ctx

    import megatron.core as mcore
    from megatron.core import utils as mcore_utils

    # _needs_patch already checked this, but the condition and the patch are
    # separated by the rest of startup, so do not assume it still holds.
    delegate = getattr(mcore_utils, _GLOBAL_NAME, None)
    if not isinstance(delegate, ModuleType):
        log_rank_0(f"[Patch:{PATCH_ID}] skipped: megatron.core.utils.{_GLOBAL_NAME} is not a module")
        return

    setattr(mcore_utils, _GLOBAL_NAME, _WarningsSilentWhileTracing(delegate))
    log_rank_0(
        f"[Patch:{PATCH_ID}] installed on megatron.core "
        f"{getattr(mcore, '__version__', 'unknown')}: {_FUNC_NAME} no longer breaks traced graphs"
    )
