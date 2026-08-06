###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Launch FlyDSL kernels on torch's *current* stream, not on the null stream.

Every ``@flyc.jit`` launcher in this package declares
``stream: fx.Stream = fx.Stream(None)``, and no caller ever passed one.
``flydsl/expr/typing.py`` resolves ``Stream(None)`` to ``c_void_p(0)`` — the
**null (legacy default) stream** — regardless of what torch is doing.

On this image ``torch.cuda.current_stream().cuda_stream`` is also ``0x0``, so in
ordinary single-stream operation the two coincide and nothing is wrong. They
stop coinciding in exactly the two situations that matter:

* **CUDA graph capture.** ``torch.cuda.graph`` captures on a side stream. The
  FlyDSL kernels keep going to stream 0, are therefore not recorded, and the
  replay silently reproduces a stale output — measured max relative error
  **1.0** at every shape before this fix, **0.0** after
  (``wp9_p5/results/probe_stream.json``). Pass 4's whole "dispatch-free
  ceiling" section was built on that unverified replay.
* **Any caller inside ``with torch.cuda.stream(s)``.** ``torch.cuda.Stream()``
  creates *non-blocking* streams, which do **not** serialise against the legacy
  default stream, so the kernels would race their own inputs.

The second is a correctness bug independent of graphs, which is why this is a
fix rather than a tuning knob. Cost is one ``current_stream()`` call per launch;
measured at production shape the explicit-stream launch is 315 µs against the
null-stream launch's 352 µs, i.e. not slower.
"""

from __future__ import annotations

import torch

import flydsl.expr as fx

__all__ = ["with_current_stream"]


def with_current_stream(launcher):
    """Wrap a ``@flyc.jit`` launcher so it defaults to torch's current stream.

    An explicit ``stream=`` from the caller still wins, so this only fills in
    the default that would otherwise be the null stream.
    """

    def _launch(*args, **kwargs):
        if "stream" not in kwargs:
            kwargs["stream"] = fx.Stream(int(torch.cuda.current_stream().cuda_stream))
        return launcher(*args, **kwargs)

    return _launch
