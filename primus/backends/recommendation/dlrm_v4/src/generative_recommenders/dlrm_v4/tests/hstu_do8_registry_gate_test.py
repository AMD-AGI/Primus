#!/usr/bin/env python3

# pyre-unsafe

"""Arbor correctness-gate shim for the R15 A11 dO8 hook (contract v1.2).

The gate (packs/dlrmv4/validator/correctness.sh) runs pytest on
generative_recommenders/dlrm_v4/tests and dlrm_v4/train/tests only, so the
hook's CPU tests in ops/tests/hstu_do8_registry_test.py would not run there.
This module re-exports their TestCase classes, so the gate collects and runs
them. They need no GPU: stand-in tensors, CPU autograd and source pins. The
forward-gate tests skip without triton and the gin test without gin-config;
the gin test, the only one that clears gin's global config and rewrites the
environment, runs its body in a child interpreter, so this module leaves the
gin state of the other gate tests alone. WiringTest checks that this list
names every TestCase class of the test module.
"""

from generative_recommenders.ops.tests.hstu_do8_registry_test import (
    AutogradRoundTripTest,
    CounterLogTest,
    DeclineTest,
    DiscardBindTest,
    ForwardGateTest,
    KnobTest,
    PublishTest,
    RoundTripTest,
    WiringTest,
)

__all__ = [
    "AutogradRoundTripTest",
    "CounterLogTest",
    "DeclineTest",
    "DiscardBindTest",
    "ForwardGateTest",
    "KnobTest",
    "PublishTest",
    "RoundTripTest",
    "WiringTest",
]
