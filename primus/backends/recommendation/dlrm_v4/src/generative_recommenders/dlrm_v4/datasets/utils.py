# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe
"""
Utility functions for dataset processing.

This module provides helper functions for parsing and processing data
in the DLRMv4 dataset pipeline.
"""

import json
import struct
from typing import Dict, List, Sequence, Tuple

import numpy as np
import xxhash


def json_loads(
    x: str | int | List[int],
) -> List[int]:
    """
    Parse a JSON-like string into a list of integers.

    Handles multiple input formats including JSON arrays, comma-separated
    strings, and single values.

    Args:
        x: Input that can be a JSON array string, a single integer,
           or already a list of integers.

    Returns:
        List of integers parsed from the input.
    """
    if isinstance(x, str):
        if x[0] != "[" and x[-1] != "]":
            x = "[" + x + "]"
        y = json.loads(x)
    else:
        y = x
    y_list = [y] if type(y) == int else list(y)
    return y_list


def separate_uih_candidates(
    x: str | int | List[int],
    candidates_max_seq_len: int,
) -> Tuple[List[int], List[int]]:
    """
    Separate a sequence into user interaction history (UIH) and candidates.

    Splits the input sequence such that the last `candidates_max_seq_len`
    elements become candidates and the rest become UIH.

    Args:
        x: Input sequence as JSON string, single int, or list of ints.
        candidates_max_seq_len: Number of items at the end to use as candidates.

    Returns:
        Tuple of (uih, candidates) where both are lists of integers.
    """
    if isinstance(x, str):
        if x[0] != "[" and x[-1] != "]":
            x = "[" + x + "]"
        y = json.loads(x)
    else:
        y = x
    y_list = [y] if type(y) == int else list(y)
    candidates, uih = (
        y_list[-candidates_max_seq_len:],
        y_list[:-candidates_max_seq_len],
    )
    return uih, candidates


def maybe_truncate_seq(
    y: List[int],
    max_seq_len: int,
) -> List[int]:
    """
    Truncate a sequence if it exceeds the maximum length.

    Args:
        y: Input sequence to potentially truncate.
        max_seq_len: Maximum allowed sequence length.

    Returns:
        The input sequence, truncated to max_seq_len if necessary.
    """
    y_len = len(y)
    if y_len > max_seq_len:
        y = y[:max_seq_len]
    return y


def xxhash_cross(
    anchor: Dict[str, int],
    keys: Sequence[str],
    table_size: int,
    salt: int = 0,
) -> int:
    """xxhash64(seed=salt) over little-endian int64 concat(anchor[k] for k in keys), mod table_size.

    Bit-identical to primus_dlrm.data.hashing.cross_hash_nway — embedding rows
    are interchangeable with Primus-trained ones.
    """
    n = len(keys)
    assert n >= 2, f"xxhash_cross needs >=2 keys, got {n}"
    digest = xxhash.xxh64(seed=salt)
    digest.update(struct.Struct(f"<{n}q").pack(*(int(anchor[k]) for k in keys)))
    return digest.intdigest() % table_size


_XXH_P1 = np.uint64(0x9E3779B185EBCA87)
_XXH_P2 = np.uint64(0xC2B2AE3D27D4EB4F)
_XXH_P3 = np.uint64(0x165667B19E3779F9)
_XXH_P4 = np.uint64(0x85EBCA77C2B2AE63)
_XXH_P5 = 0x27D4EB2F165667C5
_U64 = (1 << 64) - 1


def _rotl64(x: np.ndarray, r: int) -> np.ndarray:
    return (x << np.uint64(r)) | (x >> np.uint64(64 - r))


def xxhash_cross_np_supported(n_keys: int, table_size: int, salt: int) -> bool:
    """True when xxhash_cross_np reproduces xxhash_cross exactly: 2..3 int64
    lanes (16/24-byte input, the XXH64 short-input path), a uint64 seed and a
    table size in [1, 2**63] (every id fits int64). Anything else stays on the
    per-row xxhash call."""
    return (
        2 <= n_keys <= 3
        and isinstance(salt, int)
        and 0 <= salt <= _U64
        and isinstance(table_size, int)
        and 0 < table_size <= (1 << 63)
    )


def xxhash_cross_np(
    cols: Sequence[np.ndarray],
    table_size: int,
    salt: int = 0,
) -> np.ndarray:
    """Bit-exact numpy XXH64 of xxhash_cross for a whole column of anchors.

    ``cols[j][i]`` is key j of row i (int64). Row i equals
    ``xxhash_cross({k_j: cols[j][i]}, [k_0, ...], table_size, salt)``:
    xxh64(seed=salt) over the little-endian int64 concat of the row, mod
    table_size. This is the XXH64 short-input (< 32 byte) path in wrapping
    uint64 arithmetic; callers check xxhash_cross_np_supported first.
    """
    n = len(cols)
    assert xxhash_cross_np_supported(n, table_size, salt)
    rows = int(np.asarray(cols[0]).shape[0])
    h = np.full(rows, (int(salt) + _XXH_P5 + 8 * n) & _U64, dtype=np.uint64)
    for c in cols:
        k = np.ascontiguousarray(c, dtype=np.int64).view(np.uint64)
        h ^= _rotl64(k * _XXH_P2, 31) * _XXH_P1
        h = _rotl64(h, 27) * _XXH_P1 + _XXH_P4
    h ^= h >> np.uint64(33)
    h *= _XXH_P2
    h ^= h >> np.uint64(29)
    h *= _XXH_P3
    h ^= h >> np.uint64(32)
    return (h % np.uint64(table_size)).astype(np.int64)


def xxhash_cross_batch(
    arr_by_key: Dict[str, np.ndarray],
    keys: Sequence[str],
    table_size: int,
    salt: int = 0,
) -> np.ndarray:
    """Vectorised xxhash_cross over equal-length int64 arrays (one per key)."""
    n = len(keys)
    assert n >= 2
    cols = [np.asarray(arr_by_key[k], dtype=np.int64).ravel() for k in keys]
    length = cols[0].shape[0]
    for c in cols:
        assert c.shape[0] == length
    pack = struct.Struct(f"<{n}q").pack
    digest_cls = xxhash.xxh64
    out = np.empty(length, dtype=np.int64)
    for i in range(length):
        d = digest_cls(seed=salt)
        d.update(pack(*(int(c[i]) for c in cols)))
        out[i] = d.intdigest() % table_size
    return out
