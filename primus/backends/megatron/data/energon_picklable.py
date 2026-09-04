###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Picklable stand-ins for the lambdas and closures Energon builds inline.

Energon 7.3.2 builds its part filters and sample loaders as lambdas in class
bodies and closures inside ``__init__``. That costs nothing under ``fork``,
which shares memory and never serialises the dataset, but ``forkserver`` and
``spawn`` have to pickle it, and pickle resolves functions by qualified name --
which finds neither a lambda in a class body nor a closure over locals.

Leaving ``fork`` is not optional once GPU-Direct RDMA is in use: a process that
has registered GPU memory through dmabuf cannot fork, because the workers
inherit device mappings that do not survive the fork and segfault immediately.

These deliberately live outside ``primus.backends.megatron.patches``. They are
referenced by name from inside pickled datasets, so a dataloader worker imports
this module while unpickling one; importing the patches package instead would
eagerly pull in every ``*_patches`` module for nothing.

``energon_fork_patches`` is what installs them.
"""

from typing import Any

from megatron.energon.flavors.webdataset.field_access import field_access


def accept_all_parts(_: str) -> bool:
    """Replaces ``lambda _: True``, CrudeWebdataset's default ``part_filter``."""
    return True


def identity_sample_loader(sample):
    """Replaces ``lambda sample: sample``, the loader CrudeWebdataset passes up."""
    return sample


class PartInSet:
    """Replaces ``lambda part: part in parts``."""

    def __init__(self, parts):
        self.parts = parts

    def __call__(self, part):
        return part in self.parts


class FieldMapSampleLoader:
    """Replaces the ``field_map`` sample loader closure."""

    def __init__(self, fields):
        self.fields = fields

    def __call__(self, sample):
        return {k: field_access(sample, v) for k, v in self.fields.items()}


class KeyedSampleLoader:
    """Replaces the outer sample loader closure.

    Holds the factory rather than a copy of its subflavors. The closure read
    ``self.subflavors`` at call time and the attribute is assigned after this
    wrapper is built, so copying the value here would capture it empty.
    """

    def __init__(self, inner, factory):
        self.inner = inner
        self.factory = factory

    def __call__(self, sample):
        return {
            "__key__": sample["__key__"],
            **self.inner(sample),
            "__restore_key__": sample["__restore_key__"],
            "__subflavors__": self.factory.subflavors,
            "__sources__": sample["__sources__"],
        }


def worker_config_getstate(self) -> Any:
    """``WorkerConfig.__getstate__`` that drops what cannot be pickled.

    A dataloader worker runs no collectives and does not write another
    process's debug file, so neither member is needed worker-side, and
    ``global_rank()`` already falls back to the local rank when no group is set.
    """
    state = object.__getstate__(self)
    drop = {"data_parallel_group": None, "_worker_debug_file": None}
    # A slotted dataclass serialises as (dict_state, slot_state).
    if isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], dict):
        return (state[0], {**state[1], **drop})
    if isinstance(state, dict):
        return {**state, **drop}
    return state
