# Adapted from ODC (https://github.com/sail-sg/odc), which is distributed under
# the MIT License per its package metadata (pyproject.toml / setup.py
# classifiers). The upstream repository ships no LICENSE file or per-file
# copyright headers; upstream copyright is held by the ODC authors (Sea AI Lab).
#
# Modifications Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# See LICENSE for license information.

from .shmem_triton import (
    LIB_SHMEM_PATH,
    SHMEM_EXTERN_LIBS,
    __syncthreads,
    getmem_nbi_block,
    int_atomic_compare_swap,
    int_atomic_swap,
    int_g,
    int_p,
    int_p_remote,
    int_wait_until_equals,
    int_wait_until_equals_remote,
    putmem_nbi_block,
    quiet,
    tid,
)

__all__ = [
    # shmem_triton
    "int_atomic_compare_swap",
    "int_atomic_swap",
    "putmem_nbi_block",
    "getmem_nbi_block",
    "quiet",
    "int_p",
    "int_p_remote",
    "int_g",
    "int_wait_until_equals",
    "int_wait_until_equals_remote",
    "tid",
    "__syncthreads",
    "LIB_SHMEM_PATH",
    "SHMEM_EXTERN_LIBS",
]
