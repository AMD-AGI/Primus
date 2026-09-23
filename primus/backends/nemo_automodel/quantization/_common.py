###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Shared machinery for the low-precision linear swaps.

THE SEAM ALL OF THESE USE:
  AutoModel's diffusion pipeline swaps ``torch.nn.Linear`` for TE Linear through
  one module-level symbol, ``_replace_linear_with_transformer_engine``, gated by
  ``model.transformer_engine_linear``. Rebinding that symbol lets a different
  precision reuse the existing config seam with no fork and no schema change: the
  swap still runs on the built transformer before FSDP2 wrapping, and the
  weight/bias copy and requires_grad handling stay identical to the TE path.

WHY THERE IS A SELECTOR:
  Several precisions can be requested at once, and exactly one can own that
  symbol. Expressing the order as an if-chain inside a shared ``install()`` means
  every precision added later has to edit the same function -- which is both a
  merge conflict and a place to get the ordering subtly wrong. Instead each
  module registers itself with a precedence, and the selector answers one
  question: which requested backend wins. Precedence becomes data, and adding a
  precision means adding a file.

  This mirrors how Primus already dispatches its GEMM backends, so it should read
  as the house pattern rather than a new one.

Kept free of torch at import time so patch conditions can be evaluated, and these
tests can run, without it.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

logger = logging.getLogger(__name__)


class BackendEntry(NamedTuple):
    """A registered low-precision backend."""

    name: str
    # Higher wins when more than one is requested. Spaced so a backend can be
    # slotted between two existing ones without renumbering.
    precedence: int
    is_requested: Callable[[], bool]
    description: str


_BACKENDS: Dict[str, BackendEntry] = {}


def register_backend(
    name: str, *, precedence: int, is_requested: Callable[[], bool], description: str
) -> None:
    """Register a low-precision backend.

    Registration is a side effect of importing the module, and is separate from
    activation: a registered backend does nothing until it is both requested and
    the highest-precedence request.
    """
    if name in _BACKENDS and _BACKENDS[name].precedence != precedence:
        raise ValueError(
            f"backend {name!r} already registered with precedence "
            f"{_BACKENDS[name].precedence}, refusing to re-register with {precedence}"
        )
    _BACKENDS[name] = BackendEntry(name, precedence, is_requested, description)


_discovered = False


def discover_backends() -> None:
    """Import every precision module in this package so all of them register.

    WITHOUT THIS THE SELECTOR IS DECORATIVE, and in a way that is easy to miss.
    Registration is a side effect of importing a module, and the patch conditions
    that consult the selector each import only their own backend. So the first
    condition to be evaluated sees a registry containing nothing but itself,
    concludes it is the highest-precedence request, and installs -- and then the
    next one does the same and overwrites it. Two swaps get installed, the last
    one wins by evaluation order rather than by precedence, and the warning line
    says a backend "will NOT be applied" immediately before applying it.

    An ImportError is not fatal. A precision whose library is absent from the
    image genuinely cannot win, so dropping it from the registry is the right
    outcome; the alternative is that a missing Transformer Engine stops a run that
    only wanted FP8.

    Modules with a leading underscore are skipped as internal. A precision that
    keeps its policy in a private module is still registered, because its public
    module imports that policy -- which keeps 'adding a precision means adding a
    file' true, this file included.
    """
    global _discovered
    if _discovered:
        return
    # Set before importing: a module that raises should not be retried on every
    # subsequent call, and re-entrancy here would be harder to reason about than
    # a backend that is simply absent.
    _discovered = True

    import importlib
    import pkgutil

    package = importlib.import_module(__package__)
    for info in pkgutil.iter_modules(package.__path__):
        if info.name.startswith("_"):
            continue
        try:
            importlib.import_module(f"{__package__}.{info.name}")
        except ImportError as exc:
            logger.debug(
                "[Quantization] %s is unavailable in this image, so it cannot be " "selected: %s",
                info.name,
                exc,
            )


def registered_backends() -> List[BackendEntry]:
    """All registered backends, highest precedence first."""
    discover_backends()
    return sorted(_BACKENDS.values(), key=lambda e: -e.precedence)


def requested_backends() -> List[BackendEntry]:
    """Registered backends whose env gate is on, highest precedence first."""
    return [e for e in registered_backends() if e.is_requested()]


def active_backend() -> Optional[BackendEntry]:
    """The one backend that gets the swap, or None if nothing was requested.

    Logs when a request loses to a higher-precedence one. Silently ignoring the
    loser would leave someone looking at a run they think is FP4 and is not.
    """
    requested = requested_backends()
    if not requested:
        return None
    winner = requested[0]
    if len(requested) > 1:
        losers = ", ".join(e.name for e in requested[1:])
        logger.warning(
            "[Quantization] %s and %s were both requested; %s takes precedence and "
            "%s will NOT be applied.",
            winner.name,
            losers,
            winner.name,
            losers,
        )
    return winner


def is_active(name: str) -> bool:
    """Whether ``name`` is the backend that wins. Use this as a patch condition."""
    active = active_backend()
    return active is not None and active.name == name


def install_linear_swap(replacement: Callable, log_prefix: str) -> None:
    """Rebind AutoModel's TE swap symbol to ``replacement``.

    Imported inside the function so that importing this module does not drag in
    AutoModel, which is what lets the unit tests run without the submodule.
    """
    import nemo_automodel._diffusers.auto_diffusion_pipeline as adp

    adp._replace_linear_with_transformer_engine = replacement
    logger.info(
        "%s installed: the nn.Linear swap is active, triggered by " "model.transformer_engine_linear=true",
        log_prefix,
    )


def copy_linear_params(dst, src) -> None:
    """Copy weight, bias, requires_grad and the training flag from src to dst.

    requires_grad and the training flag are copied rather than assumed, because a
    swap that silently makes a frozen layer trainable (or vice versa) changes what
    the optimizer sees without changing anything visible in the config.
    """
    import torch

    dst.train(src.training)
    with torch.no_grad():
        dst.weight.copy_(src.weight)
        dst.weight.requires_grad_(src.weight.requires_grad)
        if src.bias is not None and dst.bias is not None:
            dst.bias.copy_(src.bias)
            dst.bias.requires_grad_(src.bias.requires_grad)


def replace_linears(
    module,
    module_name: str,
    *,
    factory: Callable,
    should_convert: Callable[[str, object], bool],
    already_converted: Tuple[type, ...],
    log_prefix: str,
) -> Tuple[int, int]:
    """Walk ``module`` and swap eligible ``nn.Linear`` children.

    Args:
        factory: ``(linear) -> replacement_module``, called only for children that
            pass ``should_convert``.
        should_convert: ``(fully_qualified_name, linear) -> bool``. Each precision
            has its own eligibility rules, so this is not decided here.
        already_converted: types to treat as done, so a second pass is a no-op
            rather than wrapping a wrapper.

    Returns ``(converted, skipped)``.
    """
    import torch.nn as nn

    converted = 0
    skipped = 0

    def walk(parent, prefix: str = "") -> None:
        nonlocal converted, skipped
        # list() because children are reassigned during the walk.
        for child_name, child in list(parent.named_children()):
            child_fqn = f"{prefix}.{child_name}" if prefix else child_name
            if already_converted and isinstance(child, already_converted):
                continue
            if isinstance(child, nn.Linear):
                if not should_convert(child_fqn, child):
                    skipped += 1
                    logger.info(
                        "%s keeping %s.%s as torch.nn.Linear (not eligible); weight=%s",
                        log_prefix,
                        module_name,
                        child_fqn,
                        tuple(child.weight.shape),
                    )
                    continue
                replacement = factory(child)
                copy_linear_params(replacement, child)
                setattr(parent, child_name, replacement)
                converted += 1
            else:
                walk(child, child_fqn)

    walk(module)
    return converted, skipped
