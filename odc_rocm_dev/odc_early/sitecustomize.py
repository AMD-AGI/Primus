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

Enable by:
    export ODC_ENABLE=1
    export PYTHONPATH=/workspace/Primus/odc_rocm_dev/odc_early:$PYTHONPATH

This is gated by ODC_ENABLE so it is a complete no-op for normal runs.
"""

import os
import sys

if os.environ.get("ODC_ENABLE", "0") == "1":
    try:
        import mori  # noqa: F401  -- loads libmori / MORI C++ runtime
        import mori.shmem  # noqa: F401

        sys.stderr.write("[ODC sitecustomize] pre-imported mori before TE (load-order fix)\n")
        sys.stderr.flush()
    except Exception as _e:  # noqa: BLE001
        sys.stderr.write(f"[ODC sitecustomize] WARNING: pre-import mori failed: {_e}\n")
        sys.stderr.flush()
