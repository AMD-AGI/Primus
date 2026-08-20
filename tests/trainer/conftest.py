###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
import shutil
import tempfile

import pytest

# Resolved once, before any test can point TMPDIR elsewhere, so each test's cache
# root is a sibling under the real temp dir rather than nested inside the
# previous test's (already deleted) one.
_BASE_TMPDIR = tempfile.gettempdir()

_CACHE_VARS = ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")


@pytest.fixture(autouse=True)
def isolated_compile_caches():
    """Give every E2E test in this directory its own TorchInductor/Triton caches.

    These tests launch real training, and CI runs the whole suite inside one
    long-lived container. With the default locations they all share
    /tmp/torchinductor_<user> and ~/.triton, so a kernel compiled during one test
    is reused by every later test and by every later job in that container.

    TMPDIR is deliberately left alone: the training ranks hand file descriptors
    to DataLoader workers over an AF_UNIX socket underneath it, and sun_path caps
    that address at 107 bytes.

    The caches are deleted afterwards, since the container outlives the job and
    each test writes a few hundred MB.
    """
    cache_root = tempfile.mkdtemp(prefix="primus-ut-cache-", dir=_BASE_TMPDIR)
    saved = {var: os.environ.get(var) for var in _CACHE_VARS}

    os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(cache_root, "torchinductor")
    os.environ["TRITON_CACHE_DIR"] = os.path.join(cache_root, "triton")
    try:
        yield cache_root
    finally:
        for var, value in saved.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value
        shutil.rmtree(cache_root, ignore_errors=True)
