###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Opt-in: no weight decay on the token-embedding weight.

Enabled by ``skip_embedding_weight_decay: true`` (default false); used by the SmolLM3
recipe (weight decay 0.1, none on the embedding, after OLMo 2). ``install()`` wraps two
Megatron functions:

  * ``megatron.training.training.setup_model_and_optimizer``: opens a scope for the
    call. When the call returns an optimizer and no param groups were checked in that
    scope (the optimizer came through another path), it raises.
  * ``megatron.core.optimizer._get_param_groups`` (what ``get_megatron_optimizer`` uses,
    through ``_get_param_groups_and_buffers``, to build the param groups for DDP and the
    distributed optimizer). Inside a scope, ``config_overrides`` also set
    ``wd_mult = 0`` for the word-embedding weight and its tied copy
    (``add_embedding_no_wd``). Megatron resolves the overrides itself. The returned
    groups are then checked on the real model chunks; the check raises on a miss and
    logs "verified: [...]". Inside a scope, missing Megatron args or the flag off in
    them is an error. Outside a scope the original runs unchanged.

The ``before_train`` patch installs the wrappers; the ``after_train`` patch removes
them. The installed wrappers are recorded by identity in ``_installed`` (not by a
function attribute, which ``functools.wraps`` copies to outer wrappers). A wrapper that
a later patch wrapped over cannot be unlinked: it stays in that chain, turned off (it
passes calls through).

Errors in the ``before_train`` patch stop the run. They are: Muon or a Primus FSDP2
custom optimizer (see ``unsupported_reason``), the flag off in Megatron's args, and no
args. They are raised as ``SystemExit``, because ``run_patches`` logs and skips an
``Exception`` by default, and training must not then start with weight decay on the
embedding.

Resume: do not change the flag between a checkpoint save and a resume that loads the
optimizer state. The flag moves the embedding weight into a ``wd_mult=0`` param group,
so the optimizer param groups (their number and sizes) differ, and the saved optimizer
state does not load into them. Megatron saves its args in the checkpoint and reads them
(``_load_base_checkpoint(..., rank0=True)``) before it loads the optimizer state; when
the saved ``skip_embedding_weight_decay`` differs from the run's, ``check_resume_flag``
stops the load with a clear error (only for a load that restores the optimizer state:
not with ``finetune`` or ``no_load_optim``). A checkpoint saved before this flag
existed has no value and is treated as off. The check is its own patch
(``...resume_check``), installed in every run that loads a checkpoint, so both
directions are caught: a flag-on run resuming a checkpoint saved with it off, and a
flag-off run (for example a generic recipe) resuming a checkpoint saved with it on. A
run that loads nothing is never wrapped. In a run with the flag off the wrapper only
compares the two values. It stops the load only when the checkpoint was saved with the
flag explicitly on (the saved optimizer state then does not fit). A saved value that is
not a boolean is logged and ignored, a load result it cannot read is passed through, and
an install error is logged, not fatal.

The conditions never raise (a value that is not a boolean makes the install patch run
and stop the run with a clear error), and the uninstall conditions also hold while a
wrapper is installed, so uninstall runs exactly when install left something.

tests/unit_tests/backends/megatron/patches/test_embedding_weight_decay_patches.py
checks that Megatron's optimizer chain looks the function up at call time;
tests/unit_tests/backends/megatron/smollm3/test_smollm3_port.py checks that the flag in
the Primus config reaches Megatron's args; tests/trainer/test_megatron_trainer.py
asserts the "verified" line in a real run.
"""

import functools
import inspect

from primus.backends.megatron.core.optimizer.embedding_weight_decay import (
    add_embedding_no_wd,
    check_embedding_no_wd,
    wd_mult_by_param_id,
    weight_decay_summary,
)
from primus.backends.megatron.core.optimizer.fsdp2_optimizer_flags import (
    active_fsdp2_optimizers,
)
from primus.core.patches import PatchContext, get_param, register_patch
from primus.core.utils.flag_utils import as_bool
from primus.core.utils.module_utils import log_rank_0

_LOG = "[Patch:megatron.optimizer.skip_embedding_weight_decay]"


def unsupported_reason(args):
    """Why this optimizer setup cannot honour skip_embedding_weight_decay, or None.

    ``args`` is Megatron's args namespace (or anything with the same attributes)."""
    optimizer = str(getattr(args, "optimizer", "adam") or "adam")
    # Same (case-sensitive) test as Megatron's setup_model_and_optimizer:
    # 'muon' in args.optimizer picks get_megatron_muon_optimizer.
    if "muon" in optimizer:
        return f"optimizer={optimizer!r} (Muon builds its param groups without the wrapped _get_param_groups)"
    # The Primus FSDP2 custom optimizers replace get_megatron_optimizer, do not call
    # _get_param_groups and drop config_overrides, so the override cannot apply.
    custom = active_fsdp2_optimizers(args)
    if custom:
        return f"{custom} (those optimizers do not take Megatron's config_overrides)"
    return None


_FLAG = "skip_embedding_weight_decay"


def skip_embedding_wd_enabled(ctx: PatchContext) -> bool:
    """The recipe flag ``skip_embedding_weight_decay`` (default false), parsed strictly:
    a value that is not a boolean raises ValueError."""
    return as_bool(get_param(ctx, _FLAG, False), _FLAG)


def skip_embedding_wd_condition(ctx: PatchContext) -> bool:
    """Patch condition: never raises. True if the flag (``get_param``) is on, or set to a
    value that is not a boolean: the patch then runs and stops the run with a clear error (an install
    patch) or removes nothing (an uninstall patch). A condition that raised would abort
    the whole patch phase in ``run_patches``, for every other patch too."""
    try:
        return skip_embedding_wd_enabled(ctx)
    except ValueError:
        return True


def skip_embedding_wd_uninstall_needed(ctx: PatchContext) -> bool:
    """Uninstall condition: a wrapper of this patch is installed, or the flag
    (``get_param``) is on. Reading the installed wrappers means uninstall runs exactly when
    install left something, even if the config changed during the run."""
    return any(k in _installed for k in _MAIN) or skip_embedding_wd_condition(ctx)


def _megatron_args():
    """Megatron's global args, or None before they are set."""
    from megatron.training.global_vars import get_args

    try:
        return get_args()
    except AssertionError:  # "args is not initialized."
        return None


def skip_embedding_wd_in_args(args) -> bool:
    """The flag on Megatron's args, parsed as strictly as the patch condition parses it
    (so ``"false"`` is False, and an unparsable value raises)."""
    return as_bool(getattr(args, _FLAG, False), _FLAG)


def apply_and_check(call, model_chunks, config, config_overrides):
    """``call(overrides + embedding no-wd)`` (the original ``_get_param_groups`` with the
    other arguments bound), checked and logged."""
    param_groups = call(add_embedding_no_wd(config_overrides, config))
    by_id = wd_mult_by_param_id(param_groups)
    try:
        names = check_embedding_no_wd(model_chunks, param_groups, by_id)
    except RuntimeError as e:
        raise RuntimeError(f"{_LOG} {e}") from None
    if names:
        log_rank_0(f"{_LOG} verified: {names} in param group(s) with wd_mult=0.0")
    else:
        log_rank_0(f"{_LOG} no trainable embedding parameter in these model chunks")
    summary = weight_decay_summary(model_chunks, param_groups, by_id)
    log_rank_0(
        f"{_LOG} other 2-D weights: {summary['decayed_2d']} with weight decay, "
        f"{summary['not_decayed_2d']} without; output_layer.weight: "
        f"{'untied (keeps weight decay)' if summary['untied_output_layer'] else 'tied or absent'}"
    )
    return param_groups


# One entry per setup_model_and_optimizer call in progress: the number of param-group
# builds checked in that call. Empty outside such a call.
_scopes = []


def wrap(original, args_getter=None, is_active=None):
    """Return the ``_get_param_groups`` wrapper; ``args_getter`` defaults to Megatron's args.

    Outside a ``setup_model_and_optimizer`` call it calls ``original`` unchanged. Inside
    one, a call without Megatron's args, or with the flag off in them, is an error, not a
    silent fall-back to the original. ``is_active`` (default: always) turns it off."""
    sig = inspect.signature(original)
    if list(sig.parameters)[:3] != ["model_chunks", "config", "config_overrides"]:
        # Checked here, at install (before_train), not as a TypeError in the optimizer build.
        raise RuntimeError(
            f"{_LOG} _get_param_groups{sig} does not start with "
            "(model_chunks, config, config_overrides); update this patch for this Megatron"
        )
    get_args = args_getter or _megatron_args
    active = is_active or (lambda: True)

    @functools.wraps(original)
    def _get_param_groups(*call_args, **call_kwargs):
        # Every argument is passed on (any parameter after the first three too); only
        # config_overrides is replaced.
        if not _scopes or not active():
            return original(*call_args, **call_kwargs)
        args = get_args()
        if args is None:
            raise RuntimeError(
                f"{_LOG} _get_param_groups called before Megatron's args are set; "
                "cannot apply the embedding weight-decay override"
            )
        if not skip_embedding_wd_in_args(args):
            raise RuntimeError(
                f"{_LOG} the patch is installed but {_FLAG} is off in Megatron's args; "
                "the embedding would keep weight decay"
            )
        bound = sig.bind(*call_args, **call_kwargs)
        bound.apply_defaults()
        bound.arguments["model_chunks"] = list(bound.arguments["model_chunks"])

        def call(overrides):
            bound.arguments["config_overrides"] = overrides
            return original(*bound.args, **bound.kwargs)

        groups = apply_and_check(
            call,
            bound.arguments["model_chunks"],
            bound.arguments["config"],
            bound.arguments["config_overrides"],
        )
        _scopes[-1] += 1
        return groups

    return _get_param_groups


def wrap_setup(original, is_active=None):
    """Return a ``setup_model_and_optimizer`` wrapper. It opens a scope for the call, in
    which the ``_get_param_groups`` wrapper applies and counts the checked builds. If
    the call returns an optimizer and no build was checked in its scope (the optimizer
    was built through another path, so the embedding keeps weight decay), it raises.
    ``is_active`` (default: always) turns it off."""
    active = is_active or (lambda: True)

    @functools.wraps(original)
    def setup_model_and_optimizer(*args, **kwargs):
        if not active():
            return original(*args, **kwargs)
        _scopes.append(0)
        try:
            result = original(*args, **kwargs)
        finally:
            checked = _scopes.pop()
        optimizer = result[1] if isinstance(result, tuple) and len(result) > 1 else None
        if optimizer is not None and checked == 0:
            raise RuntimeError(
                f"{_LOG} the optimizer was built without the wrapped _get_param_groups; "
                "the embedding weight-decay override was not applied"
            )
        return result

    return setup_model_and_optimizer


def _flag_in_checkpoint(checkpoint_args) -> bool:
    """The flag in a checkpoint's saved args; absent (saved before the flag existed) is off."""
    return as_bool(getattr(checkpoint_args, _FLAG, False), f"checkpoint {_FLAG}")


def check_resume_flag(args, checkpoint_args):
    """Raise if a load that restores the optimizer state would change the flag.

    ``args`` is the run's Megatron args, ``checkpoint_args`` the args saved in the
    checkpoint (None: nothing to check). The optimizer param groups depend on the flag,
    so the saved optimizer state does not fit a run with the other value. A load without
    the optimizer state (``finetune``, ``no_load_optim``) and a checkpoint saved without
    it (saved ``no_save_optim``) are not checked.

    In a run with the flag off, the only load that is stopped is one whose checkpoint
    was saved with the flag explicitly ``True``; a saved value that is not a boolean is
    ignored there (logged). In a run with the flag on, a saved value that is not a
    boolean raises (the param groups cannot be confirmed to match)."""
    if checkpoint_args is None or getattr(args, "finetune", False) or getattr(args, "no_load_optim", False):
        return
    if getattr(checkpoint_args, "no_save_optim", False):
        # Saved without the optimizer state: there is no optimizer state to load.
        return
    now = skip_embedding_wd_in_args(args)
    try:
        saved = _flag_in_checkpoint(checkpoint_args)
    except ValueError as exc:
        if not now:
            log_rank_0(f"{_LOG} WARNING: {exc}; ignored, {_FLAG} is off in this run")
            return
        raise RuntimeError(f"{_LOG} {exc}; cannot confirm the optimizer param groups match") from exc
    if saved != now:
        raise RuntimeError(
            f"{_LOG} the checkpoint was saved with {_FLAG}={saved} and this run has "
            f"{_FLAG}={now}. The optimizer param groups differ, so the optimizer state cannot "
            f"be loaded. Use {_FLAG}={saved}, or load without the optimizer state "
            "(no_load_optim: true or finetune: true)."
        )


def wrap_load_base_checkpoint(original, is_active=None):
    """Return a ``_load_base_checkpoint`` wrapper that runs ``check_resume_flag`` on the
    args saved in the loaded state dict.

    Megatron's ``load_checkpoint`` first calls ``_load_base_checkpoint(load_dir, args,
    rank0=True)`` to read the common state (with the saved args), and only then builds
    and loads the optimizer state. ``args`` (the second argument) is the run's args. A
    release checkpoint loads no optimizer state and is not checked. In a run with the
    flag on, a result that is not Megatron's ``(state_dict, checkpoint_name, release,
    ckpt_type)`` raises (the check cannot run); in a run with the flag off it is passed
    through unchecked. In a flag-off run the only load it stops is that of a checkpoint
    saved with the flag on (``check_resume_flag``)."""
    sig = inspect.signature(original)
    if "args" not in sig.parameters:
        # The run's args are found by name; without that parameter the check would never
        # run, so refuse to wrap (the install patch decides what that means for the run).
        raise RuntimeError(f"{_LOG} _load_base_checkpoint{sig} has no 'args' parameter")
    active = is_active or (lambda: True)

    @functools.wraps(original)
    def _load_base_checkpoint(*args, **kwargs):
        result = original(*args, **kwargs)
        if not active():
            return result
        run_args = sig.bind_partial(*args, **kwargs).arguments.get("args")
        if not (isinstance(result, tuple) and len(result) >= 3):
            if run_args is not None and skip_embedding_wd_in_args(run_args):
                raise RuntimeError(
                    f"{_LOG} _load_base_checkpoint returned {type(result).__name__}, not "
                    "(state_dict, checkpoint_name, release, ...); cannot check the saved "
                    f"{_FLAG}"
                )
            return result
        state_dict, release = result[0], result[2]
        if isinstance(state_dict, dict) and "args" in state_dict and run_args is not None and not release:
            check_resume_flag(run_args, state_dict["args"])
        return result

    return _load_base_checkpoint


# The installed wrappers: "module.attribute" -> (module, name, original, wrapper, token).
# Recorded by identity, not by a function attribute (functools.wraps copies those).
_installed = {}
_MAIN = ("megatron.training.training.setup_model_and_optimizer", "megatron.core.optimizer._get_param_groups")
_RESUME = ("megatron.training.checkpointing._load_base_checkpoint",)


def _install(module, name, make):
    """Set ``module.name`` to ``make(original, is_active=...)`` once. Returns True if it
    wrapped now, False if this patch's wrapper is already installed."""
    key = f"{module.__name__}.{name}"
    if key in _installed:
        return False
    original = getattr(module, name)
    token = object()  # this install; a later install of the same key gets a new one
    wrapper = make(original, is_active=lambda: _installed.get(key, (None,))[-1] is token)
    setattr(module, name, wrapper)
    _installed[key] = (module, name, original, wrapper, token)
    return True


def _uninstall(keys):
    """Remove the wrappers in ``keys``. A wrapper that is still the attribute is replaced
    by its original; one that a later patch wrapped over stays in that chain, turned off
    (its ``is_active`` is now False, so it passes calls through). Returns
    ``(restored, turned_off)`` lists of keys."""
    restored, turned_off = [], []
    for key in keys:
        entry = _installed.pop(key, None)
        if entry is None:
            continue
        module, name, original, wrapper, _ = entry
        if getattr(module, name) is wrapper:
            setattr(module, name, original)
            restored.append(key)
        else:
            turned_off.append(key)
    return restored, turned_off


def _log_removed(what, removed):
    restored, turned_off = removed
    if restored:
        log_rank_0(f"{_LOG} {what} removed: {restored}")
    if turned_off:
        log_rank_0(f"{_LOG} {what} turned off (a later patch wrapped over them): {turned_off}")


def install():
    """Wrap ``megatron.training.training.setup_model_and_optimizer`` and
    ``megatron.core.optimizer._get_param_groups`` once each. Returns True if
    ``_get_param_groups`` was wrapped now."""
    import megatron.core.optimizer as optimizer_module
    import megatron.training.training as training_module

    _install(training_module, "setup_model_and_optimizer", wrap_setup)
    return _install(optimizer_module, "_get_param_groups", wrap)


def uninstall():
    """Remove this patch's two wrappers; returns ``(restored, turned_off)``."""
    return _uninstall(_MAIN)


def install_resume_check():
    """Wrap ``megatron.training.checkpointing._load_base_checkpoint`` once with
    ``check_resume_flag``. Returns True if wrapped now."""
    import megatron.training.checkpointing as checkpointing_module

    return _install(checkpointing_module, "_load_base_checkpoint", wrap_load_base_checkpoint)


def uninstall_resume_check():
    """Remove the resume-check wrapper; returns ``(restored, turned_off)``."""
    return _uninstall(_RESUME)


def check_and_install(ctx: PatchContext):
    """Check, then install. The checks always run, on ``backend_args`` or else on
    Megatron's ``get_args()``; with neither there is nothing to check and it fails."""
    skip_embedding_wd_enabled(ctx)  # a value that is not a boolean stops here
    args = (ctx.extra or {}).get("backend_args")
    if args is None:
        args = _megatron_args()
    if args is None:
        raise RuntimeError(
            f"{_LOG} no Megatron args (neither backend_args nor get_args()) to check the "
            "optimizer setup against; not installing"
        )
    if not skip_embedding_wd_in_args(args):
        raise RuntimeError(
            f"{_LOG} {_FLAG} is on in the Primus config but not in the "
            "backend args; the embedding would keep weight decay"
        )
    reason = unsupported_reason(args)
    if reason:
        raise NotImplementedError(f"{_LOG} {_FLAG} is not supported with {reason}")
    install()
    log_rank_0(f"{_LOG} installed; the param groups are checked when the optimizer is built")


@register_patch(
    "megatron.optimizer.skip_embedding_weight_decay",
    backend="megatron",
    phase="before_train",
    description="Set wd_mult=0 for the word-embedding weight in Megatron's optimizer param groups",
    priority=40,
    condition=skip_embedding_wd_condition,
)
def patch_skip_embedding_weight_decay(ctx: PatchContext):
    """``check_and_install``; any error stops the run (``SystemExit`` is a
    ``BaseException``, which ``run_patches`` does not catch)."""
    try:
        check_and_install(ctx)
    except Exception as exc:
        log_rank_0(f"{_LOG} FATAL: {exc}")
        raise SystemExit(f"{_LOG} {_FLAG} could not be applied: {exc}") from exc


@register_patch(
    "megatron.optimizer.skip_embedding_weight_decay.uninstall",
    backend="megatron",
    phase="after_train",
    description="Remove the skip_embedding_weight_decay wrappers after training",
    priority=40,
    condition=skip_embedding_wd_uninstall_needed,
)
def patch_skip_embedding_weight_decay_uninstall(ctx: PatchContext):
    _log_removed("wrappers", uninstall())


def resume_check_needed(ctx: PatchContext) -> bool:
    """Patch condition: the run loads a checkpoint (``load`` or ``pretrained_checkpoint``
    from ``get_param`` on the Primus config, or in Megatron's args), whatever the flag
    value, so a resume is checked in both directions (flag on -> off and off -> on). A
    run that loads nothing is never wrapped."""
    if get_param(ctx, "load", None) or get_param(ctx, "pretrained_checkpoint", None):
        return True
    # Megatron's final args (before_train runs after they are built): a load dir set there
    # and not in the Primus config is checked too.
    args = (ctx.extra or {}).get("backend_args")
    return bool(getattr(args, "load", None) or getattr(args, "pretrained_checkpoint", None))


def resume_check_uninstall_needed(ctx: PatchContext) -> bool:
    """Uninstall condition: the resume-check wrapper is installed, or the run loads a
    checkpoint (``resume_check_needed``, which reads ``get_param``)."""
    return any(k in _installed for k in _RESUME) or resume_check_needed(ctx)


@register_patch(
    "megatron.optimizer.skip_embedding_weight_decay.resume_check",
    backend="megatron",
    phase="before_train",
    description="Stop a resume that changes skip_embedding_weight_decay (the optimizer state cannot load)",
    priority=40,
    condition=resume_check_needed,
)
def patch_skip_embedding_weight_decay_resume_check(ctx: PatchContext):
    """``install_resume_check``. With the flag on (or not a boolean), an install error
    stops the run (``SystemExit``): the run must not resume with mismatched param groups
    unchecked. With the flag off, it is logged and the run goes on unchecked."""
    try:
        install_resume_check()
    except Exception as exc:
        if skip_embedding_wd_condition(ctx):
            log_rank_0(f"{_LOG} FATAL: {exc}")
            raise SystemExit(f"{_LOG} the {_FLAG} resume check could not be installed: {exc}") from exc
        log_rank_0(f"{_LOG} WARNING: resume check not installed ({exc}); {_FLAG} is off")
        return
    log_rank_0(f"{_LOG} resume check installed: a load must keep the saved {_FLAG}")


@register_patch(
    "megatron.optimizer.skip_embedding_weight_decay.resume_check.uninstall",
    backend="megatron",
    phase="after_train",
    description="Remove the skip_embedding_weight_decay resume check after training",
    priority=40,
    condition=resume_check_uninstall_needed,
)
def patch_skip_embedding_weight_decay_resume_check_uninstall(ctx: PatchContext):
    _log_removed("resume check", uninstall_resume_check())
