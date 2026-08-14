"""Did the recompile-limit raise actually land, and how many variants are live?"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch._dynamo.config as cfg  # noqa: E402

from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1 import (  # noqa: E402
    _compile,
)

FIELDS = (
    "recompile_limit",
    "accumulated_recompile_limit",
    "cache_size_limit",
    "accumulated_cache_size_limit",
)
print("before:", {f: getattr(cfg, f, "ABSENT") for f in FIELDS})
_compile._configure()
print("after: ", {f: getattr(cfg, f, "ABSENT") for f in FIELDS})
print("compile_enabled:", _compile.compile_enabled())
