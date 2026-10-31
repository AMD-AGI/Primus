###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Optional: a first iteration on the log schedule ends a logged interval, as any other does.

Upstream ``training_log`` (Megatron-LM ``d3528a2``) does not reset its interval state after the
first iteration of a run (``should_reset = not is_first_iteration``): the loss sums, the advanced /
skipped / nan counters, the ``interval-time`` timer and the timers it logs. This is on purpose
upstream, so the first line documents the start-up overhead. But when that first iteration is also
a scheduled log iteration (``iteration % log_interval == 0``), it is counted again in the next
logged interval: with ``log_interval`` 1 the second logged loss of a run (fresh or resumed) is the
mean of two iterations. A resumed run then does not log the losses of the uninterrupted run.

When enabled, this patch calls upstream with ``is_first_iteration=False`` for a first iteration
that is on the log schedule. Upstream then logs it and resets everything itself, in the same way
as for every other logged iteration (``interval-time`` with ``barrier=True``,
``timers.log(..., reset=True)``, loss sums and counters). A first iteration off the log schedule is
passed through unchanged (upstream logs it without a reset, as before). The flag is passed
as ``False`` for the whole call, so the patch is correct only while upstream reads it in exactly
the places listed in ``_UPSTREAM_FLAG_USES``; the patch reads ``training_log``'s source when it
installs and raises if the uses differ (a Megatron bump that adds or changes one). Besides the
reset and the log condition, the only other upstream
use of the flag is the TensorBoard / WandB ``iteration-time`` write, which upstream skips for the
first iteration (it includes the start-up overhead). The patch keeps that: during the call it hands
upstream TensorBoard / WandB writers that drop only ``iteration-time``. The interval reset is not
limited to stdout: every value upstream derives from the interval timer or sums changes for the
logged interval after the first one, in stdout and in TensorBoard / WandB alike. That covers the
loss and the interval timers, and also ``throughput``, ``iter-energy/gpu`` and ``power/gpu``. They
then cover that interval only, without the first iteration and its start-up time.

Scope: one model-independent switch, the module param ``log_reset_first_iteration`` (default
off: upstream's logs). No model turns it on in code; a config does. The OLMo-3 example
configs (``examples/megatron/configs/*/olmo_3_1025_7B-BF16-pretrain.yaml``) set it to ``true``,
because a run started from them and resumed (``save``/``load`` given as overrides) at an
iteration on the log schedule (a multiple of ``log_interval``) then logs the losses of the
uninterrupted run. A run resumed off the schedule keeps upstream's behaviour: its first logged
interval also counts the first iteration. Any model with the same setting logs the same way. A value that is not a switch is an error at
patch time.

The patch runs at priority 40, before ``megatron.training_log.unified_patch`` (default 50), which
replaces ``training_log`` with a ``(*args, **kwargs)`` wrapper: this patch must bind the arguments
with the signature of upstream ``training_log``. When the function it finds lacks
``iteration`` / ``is_first_iteration`` (another wrapper came first, or upstream changed), the patch
raises instead of silently not resetting. It also raises when the training module lacks
``get_tensorboard_writer`` / ``get_wandb_writer``, since the start-up ``iteration-time`` could then
reach TensorBoard / WandB. It wraps once (a second install is a no-op).

``megatron.training.mlperf_logging`` (priority 15, so it runs first when enabled) replaces
``training_log`` with a ``(*args, **kwargs)`` wrapper, silences Megatron's stdout log and sets the
TensorBoard / WandB writers to None, and logs each step's own loss. There is then no interval log
to reset: the patch logs that and does nothing, instead of refusing the run.
"""

import ast
import inspect
import textwrap

from primus.core.patches import PatchContext, get_param, register_patch
from primus.core.utils.arg_utils import parse_optional_switch
from primus.core.utils.module_utils import log_rank_0

PATCH_ID = "megatron.training_log.first_iteration_reset"
# lower runs first: before megatron.training_log.unified_patch (default priority 50)
PRIORITY = 40
_NEEDED = ("iteration", "is_first_iteration")
SWITCH = "log_reset_first_iteration"
_MARK = "_primus_first_iteration_reset"
# the attribute megatron.training.mlperf_logging sets on its training_log wrapper
# (mlperf_logging_patches.py, ``_mlperf_training_log._primus_mlperf_logging_wrapper = True``)
_MLPERF_MARK = "_primus_mlperf_logging_wrapper"


# Where upstream training_log (Megatron-LM d3528a2) reads is_first_iteration: the enclosing
# assignment, or the test of the enclosing if. The wrapper passes the flag as False for the whole
# call and undoes only the iteration-time write; any other use would change silently.
_UPSTREAM_FLAG_USES = (
    "should_reset = not is_first_iteration",
    "iteration % args.log_interval == 0 or is_first_iteration",
    "args.log_timers_to_tensorboard and not is_first_iteration",
)


def flag_uses(fn) -> tuple:
    """Where the function ``fn`` reads ``is_first_iteration``, as in ``_UPSTREAM_FLAG_USES``: per
    read, the test of the enclosing ``if`` / ``while`` or else the enclosing statement (sorted).
    Raises ``OSError`` / ``TypeError`` when the source cannot be read."""

    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    func = tree.body[0]
    parents = {child: node for node in ast.walk(func) for child in ast.iter_child_nodes(node)}
    uses = []
    for node in ast.walk(func):
        if isinstance(node, ast.Name) and node.id == "is_first_iteration" and isinstance(node.ctx, ast.Load):
            up = node
            while not isinstance(up, ast.stmt):
                up = parents[up]
            if isinstance(up, (ast.If, ast.While)) and _contains(up.test, node):
                uses.append(ast.unparse(up.test))
            else:
                uses.append(ast.unparse(up).splitlines()[0])
    return tuple(sorted(uses))


def expected_flag_uses() -> tuple:
    """``_UPSTREAM_FLAG_USES`` in the form :func:`flag_uses` returns (``ast.unparse``, sorted)."""

    return tuple(sorted(ast.unparse(ast.parse(u)).splitlines()[0] for u in _UPSTREAM_FLAG_USES))


def _contains(tree, node) -> bool:

    return any(n is node for n in ast.walk(tree))


def _missing_in(sig) -> list:
    """The parameters of ``_NEEDED`` that the signature ``sig`` lacks."""
    return [name for name in _NEEDED if name not in sig.parameters]


def reset_enabled(ctx) -> bool:
    """The module param ``log_reset_first_iteration`` (a switch; another value raises); unset is
    off, for every model, and so is a missing context (``ctx`` None)."""
    if ctx is None:
        return False
    return parse_optional_switch(get_param(ctx, SWITCH, None), SWITCH)


def reset_requested(ctx) -> bool:
    """Patch condition, never raises (as ``checkpoint_path.routes_save``): a value that is not a
    switch selects the patch, whose body raises the error inside the patch runner's per-patch
    error handling (``[Patch] ... failed`` context, ``stop_on_error``). The switch is read by
    :func:`reset_enabled` (the module param, with ``get_param``)."""
    try:
        return reset_enabled(ctx)
    except ValueError:
        return True


_DROPPED_SCALAR = "iteration-time"


class _TensorboardWithoutIterationTime:
    """A TensorBoard writer that drops ``add_scalar("iteration-time", ...)``; the rest delegates."""

    def __init__(self, writer):
        self._writer = writer

    def add_scalar(self, tag, *args, **kwargs):
        if tag != _DROPPED_SCALAR:
            return self._writer.add_scalar(tag, *args, **kwargs)
        return None

    def __getattr__(self, name):
        return getattr(self._writer, name)


class _WandbWithoutIterationTime:
    """A WandB writer whose ``log`` drops the ``iteration-time`` key; the rest delegates."""

    def __init__(self, writer):
        self._writer = writer

    def log(self, data, *args, **kwargs):
        if isinstance(data, dict) and _DROPPED_SCALAR in data:
            data = {k: v for k, v in data.items() if k != _DROPPED_SCALAR}
            if not data:
                return None
        return self._writer.log(data, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._writer, name)


_WRITER_GETTERS = ("get_tensorboard_writer", "get_wandb_writer")


def _call_without_iteration_time(fn, module, args, kwargs):
    """``fn(*args, **kwargs)`` with ``module.get_tensorboard_writer`` / ``get_wandb_writer``
    returning writers that drop ``iteration-time`` (restored afterwards). ``module`` None (no
    writers to filter, e.g. a test stand-in): a plain call. ``wrap_training_log`` has checked
    that a given module has both getters."""
    if module is None:
        return fn(*args, **kwargs)
    saved = {n: getattr(module, n) for n in _WRITER_GETTERS}

    def tensorboard():
        writer = saved["get_tensorboard_writer"]()
        return _TensorboardWithoutIterationTime(writer) if writer else writer

    def wandb():
        writer = saved["get_wandb_writer"]()
        return _WandbWithoutIterationTime(writer) if writer else writer

    module.get_tensorboard_writer, module.get_wandb_writer = tensorboard, wandb
    try:
        return fn(*args, **kwargs)
    finally:
        for n, v in saved.items():
            setattr(module, n, v)


def wrap_training_log(original_fn, get_megatron_args, module=None, check_uses=False):
    """``training_log`` that passes ``is_first_iteration=False`` for a first iteration on the log
    schedule, so upstream resets the interval as for any other logged iteration. For that call the
    writers that ``module`` (upstream's training module) hands out drop ``iteration-time``, as
    upstream does for a first iteration. Returns ``original_fn`` if it is already wrapped; raises
    ``RuntimeError`` if its signature lacks the parameters the patch reads, or if ``module`` lacks
    the writer getters (the start-up iteration-time could then reach TensorBoard / WandB), or if
    ``check_uses`` (the patch sets it) and ``original_fn`` reads ``is_first_iteration`` elsewhere
    than in ``_UPSTREAM_FLAG_USES``."""
    if getattr(original_fn, _MARK, False):
        return original_fn
    if module is not None and not all(hasattr(module, n) for n in _WRITER_GETTERS):
        raise RuntimeError(
            f"[Patch:{PATCH_ID}] the training module lacks {list(_WRITER_GETTERS)}; cannot keep the "
            f"first iteration-time out of TensorBoard / WandB. Set {SWITCH}: false to run without it."
        )
    sig = inspect.signature(original_fn, follow_wrapped=False)
    missing = _missing_in(sig)
    if missing:
        raise RuntimeError(
            f"[Patch:{PATCH_ID}] training_log lacks parameter(s) {missing} (another wrapper ran "
            f"first, or upstream changed); cannot reset the first logged interval. Set "
            f"{SWITCH}: false to run without it."
        )
    if check_uses:
        try:
            uses = flag_uses(original_fn)
        except (OSError, TypeError, SyntaxError) as e:
            uses = f"unreadable source ({type(e).__name__}: {e})"
        if uses != expected_flag_uses():
            raise RuntimeError(
                f"[Patch:{PATCH_ID}] training_log reads is_first_iteration in {uses}, not in the "
                f"places this patch is written for {sorted(_UPSTREAM_FLAG_USES)} (upstream changed); "
                f"cannot pass is_first_iteration=False safely. Set {SWITCH}: false to run without it."
            )

    def training_log(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        call = bound.arguments
        if call["is_first_iteration"] and call["iteration"] % get_megatron_args().log_interval == 0:
            call["is_first_iteration"] = False
            return _call_without_iteration_time(original_fn, module, bound.args, bound.kwargs)
        return original_fn(*args, **kwargs)

    setattr(training_log, _MARK, True)
    training_log.__wrapped__ = original_fn
    return training_log


@register_patch(
    PATCH_ID,
    backend="megatron",
    phase="before_train",
    description="A first iteration on the log schedule resets the logged interval (log_reset_first_iteration).",
    condition=reset_requested,
    priority=PRIORITY,
)
def patch_training_log_first_iteration_reset(ctx: PatchContext):
    # the condition (reset_requested) does not raise; a value that is not a switch raises here, and
    # an off switch installs nothing, whoever calls the patch
    if not reset_enabled(ctx):
        return
    import megatron.training.training as training_module
    from megatron.training import get_args as get_megatron_args

    original = training_module.training_log
    if getattr(original, _MLPERF_MARK, False):
        log_rank_0(
            f"[Patch:{PATCH_ID}] MLPerf logging replaces Megatron's interval log; {SWITCH} has nothing "
            "to reset and is not applied"
        )
        return
    # owns "wrap once"; check_uses: upstream reads is_first_iteration only where the patch expects
    wrapped = wrap_training_log(original, get_megatron_args, training_module, check_uses=True)
    if wrapped is original:
        return
    training_module.training_log = wrapped
    log_rank_0(f"[Patch:{PATCH_ID}] a first iteration on the log schedule resets the logged interval")
