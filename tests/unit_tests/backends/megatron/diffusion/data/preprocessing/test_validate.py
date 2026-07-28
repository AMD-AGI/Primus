# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Tests for Energon dataset validation module.

Tests _check_metadata, _check_sample_counts, and validate_energon_dataset
using synthetic dataset directories.
"""

import json
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from primus.backends.megatron.data.diffusion.preprocessing.validate import (
    _check_metadata,
    _check_sample_counts,
    validate_energon_dataset,
)
from tests.utils import PrimusUT


def _build_valid_dataset(base: Path, num_samples: int = 3) -> None:
    """Helper: build a minimal valid dataset directory on disk."""
    meta_dir = base / ".nv-meta"
    meta_dir.mkdir(parents=True)

    shard_name = "000000.tar"
    idx_name = shard_name + ".idx"

    # Write .info.json
    info = {"shard_counts": {shard_name: num_samples}}
    (meta_dir / ".info.json").write_text(json.dumps(info))

    # Write split.yaml
    split = {"split_parts": {"train": [shard_name]}}
    (meta_dir / "split.yaml").write_text(yaml.dump(split))

    # Write dataset.yaml
    (meta_dir / "dataset.yaml").write_text("__module__: megatron.energon\n__class__: CrudeWebdataset\n")

    # Write shard tar with correct number of samples
    tar_path = base / shard_name
    with tarfile.open(str(tar_path), "w") as tar:
        for i in range(num_samples):
            key = f"000000_{i:06d}"
            for ext in ("jpg", "txt"):
                member_name = f"{key}.{ext}"
                data = b"test"
                info_obj = tarfile.TarInfo(name=member_name)
                info_obj.size = len(data)
                import io

                tar.addfile(info_obj, io.BytesIO(data))

    # Write .idx stub
    (base / idx_name).write_text("")


class TestCheckMetadata(PrimusUT):
    """Tests for _check_metadata."""

    def test_valid_metadata(self):
        """Valid .nv-meta with all files returns info dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            _build_valid_dataset(base)
            info = _check_metadata(base)

            assert info is not None
            assert "shard_counts" in info
            assert "_splits" in info

    def test_missing_info_json(self):
        """Missing .info.json returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            _build_valid_dataset(base)
            (base / ".nv-meta" / ".info.json").unlink()

            assert _check_metadata(base) is None

    def test_missing_shard_counts_key(self):
        """Info file without shard_counts key returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            _build_valid_dataset(base)
            (base / ".nv-meta" / ".info.json").write_text(json.dumps({"other": 1}))

            assert _check_metadata(base) is None

    def test_missing_split_yaml(self):
        """Missing split.yaml returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            _build_valid_dataset(base)
            (base / ".nv-meta" / "split.yaml").unlink()

            assert _check_metadata(base) is None

    def test_missing_dataset_yaml(self):
        """Missing dataset.yaml returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            _build_valid_dataset(base)
            (base / ".nv-meta" / "dataset.yaml").unlink()

            assert _check_metadata(base) is None

    def test_missing_shard_file(self):
        """Referenced shard file missing on disk returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            _build_valid_dataset(base)
            (base / "000000.tar").unlink()

            assert _check_metadata(base) is None


class TestCheckSampleCounts(PrimusUT):
    """Tests for _check_sample_counts."""

    def test_matching_counts(self):
        """Tar with correct sample count returns True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            _build_valid_dataset(base, num_samples=3)

            info = _check_metadata(base)
            assert info is not None
            assert _check_sample_counts(base, info) is True

    def test_mismatched_counts(self):
        """Mismatch between .info.json and tar content returns False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            _build_valid_dataset(base, num_samples=3)

            # Overwrite .info.json with wrong count
            wrong_info = {"shard_counts": {"000000.tar": 99}}
            (base / ".nv-meta" / ".info.json").write_text(json.dumps(wrong_info))

            info = _check_metadata(base)
            assert info is not None
            assert _check_sample_counts(base, info) is False


class TestValidateEnergonDataset(PrimusUT):
    """Tests for validate_energon_dataset orchestration."""

    @patch(
        "primus.backends.megatron.data.diffusion.preprocessing.validate._check_sample_load",
        return_value={"jpg": "bytes, len=100", "txt": "str"},
    )
    def test_full_pass(self, mock_load):
        """Complete valid dataset returns True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _build_valid_dataset(Path(tmpdir), num_samples=3)
            assert validate_energon_dataset(tmpdir, encoding="raw") is True

    @patch(
        "primus.backends.megatron.data.diffusion.preprocessing.validate._check_sample_load",
        return_value={"jpg": "bytes, len=100", "txt": "str"},
    )
    def test_metadata_failure_returns_false(self, mock_load):
        """Metadata failure returns False immediately (early exit)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Empty dir has no .nv-meta at all
            assert validate_energon_dataset(tmpdir, encoding="raw") is False
            mock_load.assert_not_called()

    @patch(
        "primus.backends.megatron.data.diffusion.preprocessing.validate._check_sample_load",
        return_value={"jpg": "bytes, len=100", "txt": "str"},
    )
    def test_count_mismatch_returns_false(self, mock_load):
        """Count mismatch returns False but still prints summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _build_valid_dataset(Path(tmpdir), num_samples=3)

            # Corrupt the count
            wrong_info = {"shard_counts": {"000000.tar": 99}}
            (Path(tmpdir) / ".nv-meta" / ".info.json").write_text(json.dumps(wrong_info))

            assert validate_energon_dataset(tmpdir, encoding="raw") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
