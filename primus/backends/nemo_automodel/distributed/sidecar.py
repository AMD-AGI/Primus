###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Primus parallelization sidecars for AutoModel's diffusion path.

AutoModel parallelizes a model through its ``ModelParallelizer`` sidecar: the
class attribute ``parallelizer``, which the diffusion pipeline attaches from a
registry keyed by class name (``_PARALLELIZERS`` in
``nemo_automodel._diffusers.parallelization``) just before parallelizing. A
model with no entry gets the default ``ModelParallelizer``.

``install`` registers a Primus sidecar there, built as a subclass of whatever
upstream registers for the class (or of the default). Upstream's sharding,
tensor-parallel plan and mixed precision therefore stay authoritative; a Primus
sidecar only adds what upstream lacks, and a later upstream sidecar for the same
class is picked up as the new base automatically.

``block_checkpointing`` is the sidecar most models need: activation
checkpointing of whole transformer blocks, across every block list the model
has, applied before delegating with checkpointing switched off so upstream does
not wrap a second time. It runs in ``parallelize`` rather than ``_apply`` so the
FSDP2, unsharded and DDP paths all get the same checkpointing.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Callable, Sequence

from primus.backends.nemo_automodel.distributed import activation_checkpointing as ac

logger = logging.getLogger(__name__)

_SIDECAR_ATTR = "_primus_sidecar"


def _registry():
    from nemo_automodel._diffusers import parallelization

    return parallelization._PARALLELIZERS


def install(model_cls_name: str, build: Callable[[type], type], *, log_prefix: str) -> bool:
    """Register ``build(base)()`` as the sidecar for ``model_cls_name``.

    ``base`` is the class of upstream's registered sidecar, or the default
    ``ModelParallelizer``. Idempotent. Returns False, with a warning, when the
    registry cannot be found, so a layout change upstream degrades to stock
    behaviour rather than failing the run.
    """
    try:
        registry = _registry()
        from nemo_automodel.components.distributed import ModelParallelizer
    except (ImportError, AttributeError) as exc:
        logger.warning(
            "%s AutoModel's diffusion parallelizer registry is not available (%s); "
            "the Primus sidecar for %s is not installed.",
            log_prefix,
            exc,
            model_cls_name,
        )
        return False

    current = registry.get(model_cls_name)
    if getattr(current, _SIDECAR_ATTR, False):
        return True
    base = type(current) if current is not None else ModelParallelizer
    cls = build(base)
    setattr(cls, _SIDECAR_ATTR, True)
    registry[model_cls_name] = cls()
    logger.info(
        "%s installed %s for %s on top of upstream %s",
        log_prefix,
        cls.__name__,
        model_cls_name,
        base.__name__,
    )
    return True


def without_activation_checkpointing(mesh_context):
    """A copy of ``mesh_context`` with checkpointing off everywhere it is read."""
    config = mesh_context.strategy_config
    if getattr(config, "activation_checkpointing", False):
        config = dataclasses.replace(config, activation_checkpointing=False)
    return dataclasses.replace(mesh_context, strategy_config=config, activation_checkpointing=False)


def block_checkpointing(
    base: type,
    *,
    block_attrs: Sequence[str],
    log_prefix: str,
    stride: Callable[[], int] = lambda: 0,
) -> type:
    """A sidecar class that checkpoints whole blocks, then defers to ``base``."""

    class PrimusBlockCheckpointing(base):
        def parallelize(self, model, mesh_context, /):
            config = mesh_context.strategy_config
            requested = ac.normalize(
                mesh_context.activation_checkpointing or getattr(config, "activation_checkpointing", False)
            )
            if not requested:
                return super().parallelize(model, mesh_context)
            ac.apply(
                ac.upstream_helpers(),
                model,
                block_attrs,
                requested,
                enable_compile=bool(getattr(config, "enable_compile", False)),
                stride=stride(),
                log_prefix=log_prefix,
            )
            return super().parallelize(model, without_activation_checkpointing(mesh_context))

    return PrimusBlockCheckpointing
