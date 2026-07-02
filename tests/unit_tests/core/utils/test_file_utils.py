###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for primus.core.utils.file_utils (filesystem helpers)."""

import pytest

from primus.core.utils.file_utils import (
    PathNotFoundError,
    check_file_exists,
    check_path_exists,
    create_path_if_not_exists,
    is_directory,
    is_file,
    path_exists,
)


def test_path_exists_and_type_checks(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x")
    assert path_exists(f) is True
    assert is_file(f) is True
    assert is_directory(f) is False
    assert is_directory(tmp_path) is True
    assert path_exists(tmp_path / "missing") is False


def test_check_file_exists(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x")
    check_file_exists(f)  # no raise
    with pytest.raises(PathNotFoundError):
        check_file_exists(tmp_path / "missing.txt")


def test_check_path_exists(tmp_path):
    check_path_exists(tmp_path)  # no raise
    with pytest.raises(PathNotFoundError):
        check_path_exists(tmp_path / "missing")


def test_create_path_if_not_exists(tmp_path):
    new_dir = tmp_path / "nested" / "dir"
    assert path_exists(new_dir) is False
    create_path_if_not_exists(new_dir)
    assert is_directory(new_dir) is True
    # idempotent: calling again on an existing path is a no-op
    create_path_if_not_exists(new_dir)
    assert is_directory(new_dir) is True
