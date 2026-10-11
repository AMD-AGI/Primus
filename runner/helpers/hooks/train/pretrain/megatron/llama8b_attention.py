# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Install the pinned Llama 8B dS layout before torchrun imports Turbo.

The patch changes only the selected gfx950 BF16 A16 Q32/KV256 builder's
physical shared-memory layout. Logical tiles and arithmetic are unchanged.
Set PRIMUS_LLAMA8B_ATTENTION_DS_LAYOUT=0 to leave Turbo untouched.
"""

import fcntl
import hashlib
import importlib.util
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

BASE_SHA256 = "41c855a2d16a4d994a384b196b84c6a276f0b2559634bd94592886946a34c691"
PATCHED_SHA256 = "d895435951c1ddaf9142dfa92d775742537077397b3436836d323eaa9e4ad265"
RELATIVE_SOURCE = Path("primus_turbo/flydsl/attention/flash_attn_bwd.py")
PATCH = Path(__file__).with_name("flash-attn-bwd-ds-layout.patch")


def eligible(config):
    expected = {
        "framework": "megatron",
        "stage": "mlperf_pretrain",
        "num_layers": 32,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_query_groups": 8,
        "seq_length": 8192,
        "micro_batch_size": 4,
        "tensor_model_parallel_size": 1,
        "enable_primus_turbo": True,
        "use_turbo_attention": True,
    }
    return all(getattr(config, key, None) == value for key, value in expected.items())


def apply_layout(source):
    """Apply once, atomically, accepting only the two known source identities."""
    source = Path(source).resolve(strict=True)
    # Serialize multiple launchers sharing an installation. Training processes
    # are launched only after this preparation step completes successfully.
    with source.with_name(source.name + ".primus-ds.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        original = source.read_bytes()
        identity = hashlib.sha256(original).hexdigest()
        if identity == PATCHED_SHA256:
            return "already applied"
        if identity != BASE_SHA256:
            raise RuntimeError("Unqualified Turbo attention source: " + str(source))
        with tempfile.TemporaryDirectory(prefix="primus-ds-") as temporary:
            root = Path(temporary)
            staged = root / RELATIVE_SOURCE
            staged.parent.mkdir(parents=True)
            staged.write_bytes(original)
            subprocess.run(["git", "apply", "--check", str(PATCH)], cwd=root, check=True)
            subprocess.run(["git", "apply", str(PATCH)], cwd=root, check=True)
            patched = staged.read_bytes()
        if hashlib.sha256(patched).hexdigest() != PATCHED_SHA256:
            raise RuntimeError("Unexpected patched Turbo attention source")
        # Same-directory replacement leaves either the complete old or new file.
        fd, name = tempfile.mkstemp(prefix=".primus-ds-", dir=source.parent)
        pending = Path(name)
        try:
            with os.fdopen(fd, "wb") as output:
                output.write(patched)
            shutil.copymode(source, pending)
            if source.read_bytes() != original:
                raise RuntimeError("Turbo attention source changed during preparation")
            pending.replace(source)
        finally:
            pending.unlink(missing_ok=True)
    return "applied"


def prepare_attention(config):
    if os.getenv("PRIMUS_LLAMA8B_ATTENTION_DS_LAYOUT", "1").lower() in ("0", "false"):
        return
    if not eligible(config):
        return
    # Top-level find_spec locates the installed package without importing Turbo
    # or initializing any GPU. Patch before all eight training ranks import it.
    spec = importlib.util.find_spec("primus_turbo")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Llama 8B attention layout requires installed Primus-Turbo")
    source = Path(next(iter(spec.submodule_search_locations))) / "flydsl/attention/flash_attn_bwd.py"
    result = apply_layout(source)
    print(f"[Primus] Llama 8B attention dS layout {result}: {PATCHED_SHA256}")
