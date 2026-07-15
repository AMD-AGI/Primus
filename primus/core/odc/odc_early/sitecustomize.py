# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""ODC early-init shim (auto-executed by Python's site machinery).

Root cause (verified on MI300X + ROCm 7.2 + PyTorch 2.10a):
    Importing `transformer_engine` BEFORE MORI's C++ runtime is loaded leaves
    the process in a state where MORI init (or even just allocating its
    symmetric heap) aborts with `free(): invalid pointer`. It is a C++
    dynamic-library load-order / global-ctor conflict, NOT an init-order or
    init-method issue.

    The fix is simply to load MORI's C++ runtime (via `import mori`) BEFORE
    Megatron imports transformer_engine. This file runs at interpreter
    startup (site.py imports `sitecustomize` if it is on sys.path), which is
    earlier than any Primus/Megatron/TE import, guaranteeing the correct
    order.

Enable by putting this directory on PYTHONPATH (the ODC launcher does this):
    export PYTHONPATH=<PRIMUS_ROOT>/primus/core/odc/odc_early:$PYTHONPATH

This shim is a pure bootstrap (a launch-time load-order workaround, NOT feature
logic): it only runs when its directory is on PYTHONPATH, which the ODC launcher
adds exactly for ODC runs, so it is a complete no-op for normal runs. Whether ODC
is actually active for training is decided by the enable_odc config item, not by
this file.
"""

import sys

try:
    import mori  # noqa: F401  -- loads libmori / MORI C++ runtime
    import mori.shmem  # noqa: F401

    sys.stderr.write("[ODC sitecustomize] pre-imported mori before TE (load-order fix)\n")
    sys.stderr.flush()
except Exception as _e:  # noqa: BLE001
    sys.stderr.write(f"[ODC sitecustomize] WARNING: pre-import mori failed: {_e}\n")
    sys.stderr.flush()
