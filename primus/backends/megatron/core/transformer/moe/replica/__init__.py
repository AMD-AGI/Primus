###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Dynamic redundant experts: hot experts are copied to underloaded EP ranks for the current routing, their rows are
computed there, and the copies' weight gradients are added back on the owner. Replicas carry no optimizer state."""

from .planner import (
    ReplicaPlan,
    alloc_fractions,
    make_plan,
    owner_alloc,
    plan_fixed,
    plan_moonep,
    plan_ultraep,
    resplit_alloc,
    resplit_if_balanced,
    slots_from_alloc,
)
from .remap import physical_counts, remap_topk, slot_lookup

__all__ = [
    "ReplicaPlan",
    "alloc_fractions",
    "make_plan",
    "owner_alloc",
    "plan_fixed",
    "plan_moonep",
    "plan_ultraep",
    "resplit_alloc",
    "resplit_if_balanced",
    "slots_from_alloc",
    "physical_counts",
    "remap_topk",
    "slot_lookup",
]
