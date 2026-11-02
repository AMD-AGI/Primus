###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Backport of Megatron-LM #7092 (27f3c8b, "fix(logging): reset log accumulators
when the first iteration is on a boundary") to the pinned Megatron-LM.

The pinned ``training_log`` sets ``should_reset = not is_first_iteration``: after
the first iteration of a run it keeps the loss sums, the iteration counters and
the timers. When that iteration is also a log point (``iteration % log_interval
== 0``, always with ``log_interval: 1``), the next logged loss and time cover two
iterations. So a run resumed at iteration N logs N+2 as the mean of N+1 and N+2,
and its losses do not match an uninterrupted run. Upstream fixed it with

    should_reset = not is_first_iteration or iteration % args.log_interval == 0

With ``reset_log_averages_after_first_log: true`` this patch makes that one-line
change in the pinned ``training_log`` with ``patch_function_source`` (the line is
in the middle of the function). Everything else is the pinned code. Only the
values after the first iteration change (printed loss, and the printed and
TensorBoard / wandb iteration time and throughput); TensorBoard's loss is
written per iteration and does not change.

* A Megatron whose ``training_log`` already has upstream #7092 needs no patch:
  it logs that and does nothing. The fix is recognised by the parsed
  expression of its one ``should_reset`` assignment (``ast``: spacing, line
  breaks, parentheses and comments do not matter; the two operands of ``or``
  and of ``==`` may be in either order). Any other expression is not taken as
  the fix.
* Otherwise, when the pinned line is not found, ``should_reset`` is assigned
  more than once or to another expression, the source cannot be read, or
  ``training_log`` is not Megatron's function (a wrapper another patch
  installed: not defined in ``megatron.training.training``), the option cannot
  be honoured and the patch raises, so a run never silently logs differently from
  what its config asks for.
"""

import ast
import inspect
import textwrap

from primus.backends.megatron.patches._source_patch_utils import patch_function_source
from primus.core.patches import PatchContext, get_param, register_patch
from primus.core.utils.module_utils import log_rank_0

PATCH_ID = "megatron.training_log.first_iteration_reset"
OPTION = "reset_log_averages_after_first_log"
PINNED_LINE = "    should_reset = not is_first_iteration\n"
FIXED_LINE = "    should_reset = not is_first_iteration or iteration % args.log_interval == 0\n"
_PINNED_EXPR = "not is_first_iteration"
# Upstream #7092's expression (ast.unparse form), in each operand order.
_FIXED_EXPRS = {
    ast.unparse(ast.parse(f"{a} or {b}", mode="eval").body)
    for x, y in (
        ("iteration % args.log_interval", "0"),
        ("0", "iteration % args.log_interval"),
    )
    for a, b in ((_PINNED_EXPR, f"{x} == {y}"), (f"{x} == {y}", _PINNED_EXPR))
}


def _should_reset_exprs(source):
    """The ``ast.unparse`` form of each value assigned to ``should_reset`` in
    ``source`` (nested functions included). Any other write (an annotated or
    augmented assignment, ``:=``, a tuple target, a ``for`` / ``with`` / ``del``
    target) is listed as ``"<other write>"``, so it never matches a known line."""
    tree = ast.parse(textwrap.dedent(source))
    plain = {
        id(node.targets[0]): ast.unparse(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
    }
    return [
        plain.get(id(node), "<other write>")
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id == "should_reset" and not isinstance(node.ctx, ast.Load)
    ]


def check_training_log(training_log, module_name=None):
    """``"patch"`` when the pinned line can be rewritten, ``"fixed"`` when
    ``training_log`` already has upstream's fix; ``RuntimeError`` (naming the
    option) otherwise. ``module_name``: the module ``training_log`` must be
    defined in (a wrapper installed by another patch is defined elsewhere)."""
    defined_in = getattr(training_log, "__module__", None)
    code = getattr(training_log, "__code__", None)
    if (
        hasattr(training_log, "__wrapped__")
        or code is None
        or (module_name is not None and defined_in != module_name)
    ):
        raise RuntimeError(
            f"{OPTION}: training_log is not Megatron's function ({defined_in}."
            f"{getattr(training_log, '__qualname__', training_log)!r}), probably a wrapper; the patch "
            f"{PATCH_ID} must run before other training_log patches, or set {OPTION}: false"
        )
    try:
        source = inspect.getsource(training_log)
        exprs = _should_reset_exprs(source)
    except (OSError, TypeError, SyntaxError) as e:
        raise RuntimeError(
            f"{OPTION}: cannot read the source of training_log ({e}); set {OPTION}: false"
        ) from e
    if len(exprs) == 1 and exprs[0] in _FIXED_EXPRS:
        return "fixed"
    if exprs != [_PINNED_EXPR] or source.count(PINNED_LINE) != 1:
        raise RuntimeError(
            f"{OPTION}: training_log does not have exactly one should_reset assignment, either "
            f"{PINNED_LINE.strip()!r} or upstream #7092's fix (found {exprs}); update {PATCH_ID} "
            f"for this Megatron-LM, or set {OPTION}: false"
        )
    return "patch"


def apply_backport(module):
    """Apply the backport to ``module.training_log``; returns ``"patched"`` or
    ``"fixed"`` (upstream's fix is already there)."""
    if check_training_log(module.training_log, module.__name__) == "fixed":
        return "fixed"
    patch_function_source(module, "training_log", PINNED_LINE, FIXED_LINE)
    return "patched"


def _enabled(ctx: PatchContext) -> bool:
    return bool(get_param(ctx, OPTION, False))


@register_patch(
    PATCH_ID,
    backend="megatron",
    phase="before_train",
    description="Backport Megatron-LM #7092: reset the log accumulators and timers after a first "
    "iteration that is a log point.",
    condition=_enabled,
    # Before the other training_log wrappers (mlperf 15, print_rank_last 50): it
    # rewrites Megatron's function itself, whose source they would hide.
    priority=10,
)
def patch_first_iteration_reset(ctx: PatchContext):
    import megatron.training.training as megatron_training

    from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched

    if is_patched(megatron_training, PATCH_ID):
        return
    if apply_backport(megatron_training) == "fixed":
        log_rank_0(f"[Patch:{PATCH_ID}] training_log already has Megatron-LM #7092; not patched")
    else:
        log_rank_0(f"[Patch:{PATCH_ID}] training_log: {FIXED_LINE.strip()} (Megatron-LM #7092)")
    mark_patched(megatron_training, PATCH_ID)
