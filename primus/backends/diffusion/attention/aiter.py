###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

import importlib.util
import logging
import os

import torch

from ._flash_common import run_flash_attention_backend

AITER_FLASH_ATTN_AVAILABLE = False
aiter = None


def _package_jit_writable() -> bool:
    """True when amd-aiter can compile into its own package directory."""
    spec = importlib.util.find_spec("aiter")
    if spec is None or not spec.origin:
        return True
    package_dir = os.path.dirname(spec.origin)
    package_jit = os.path.join(package_dir, "jit")
    probe = package_jit if os.path.isdir(package_jit) else package_dir
    return os.access(probe, os.W_OK)


def _import_aiter():
    """Import ``aiter``, pointing its JIT loader at the compiled modules.

    amd-aiter compiles ``module_aiter_core`` into ``~/.aiter/jit`` when the
    package directory is not writable (as in a container image). It only adds
    that directory to ``sys.path``, and only imports the module by its bare
    name, when ``AITER_JIT_DIR`` is set before the import. Otherwise the build
    succeeds and the import then dies with
    ``ModuleNotFoundError: aiter.jit.module_aiter_core``.

    A writable package directory must keep its own prebuilt modules. Setting
    ``AITER_JIT_DIR`` just because ``~/.aiter/jit`` exists would rebuild there
    or load a leftover ``.so`` from another image.
    """
    # aiter creates ~/.aiter/jit itself when AITER_JIT_DIR is set, so the
    # directory need not exist yet. Requiring it here left the first run on a
    # fresh home without the variable and broke the first attention call.
    if not _package_jit_writable():
        user_jit = os.path.join(os.path.expanduser("~"), ".aiter", "jit")
        os.environ.setdefault("AITER_JIT_DIR", user_jit)
    import aiter as module

    return module


try:
    _aiter = _import_aiter()

    if hasattr(_aiter, "flash_attn_func") and hasattr(_aiter, "flash_attn_varlen_func"):
        aiter = _aiter
        AITER_FLASH_ATTN_AVAILABLE = True
except Exception as exc:
    AITER_FLASH_ATTN_AVAILABLE = False
    # Installed but unusable is a setup problem worth surfacing. A missing
    # package is the normal non-AITER case, so stay quiet for that.
    if importlib.util.find_spec("aiter") is not None:
        logging.getLogger(__name__).warning(
            "aiter is installed but failed to import; AITER attention is disabled: %s", exc
        )


def _normalize_window_size(window_size: tuple[int, ...]) -> tuple[int, int, int]:
    if len(window_size) == 2:
        left, right = window_size
        return (left, right, 0)
    if len(window_size) == 3:
        left, right, sink = window_size
        return (left, right, sink)
    raise ValueError(f"window_size must have 2 or 3 items, got {window_size}")


def aiter_flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
):
    assert AITER_FLASH_ATTN_AVAILABLE and aiter is not None

    need_lse = torch.is_grad_enabled()
    return run_flash_attention_backend(
        q=q,
        k=k,
        v=v,
        q_lens=q_lens,
        k_lens=k_lens,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        q_scale=q_scale,
        causal=causal,
        window_size=window_size,
        deterministic=deterministic,
        dtype=dtype,
        fixed_attention=aiter.flash_attn_func,
        varlen_attention=aiter.flash_attn_varlen_func,
        window_size_adapter=_normalize_window_size,
        fixed_extra_kwargs={"return_lse": need_lse},
        varlen_extra_kwargs={"return_lse": need_lse},
    )
