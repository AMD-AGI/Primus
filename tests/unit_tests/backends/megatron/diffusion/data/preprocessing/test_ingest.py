# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Tests for the streaming ingest pipeline (pipelines/ingest.py).

Covers:
- StreamingIngestPipeline: parallel download + sequential conversion
- Resume functionality (existing shards are skipped)
- max_files parameter
- Sample offset accumulation
- Arrow file cleanup after conversion
- Skip-and-log for failed downloads and conversions
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from primus.backends.megatron.data.diffusion.preprocessing.pipelines.ingest import (
    StreamingIngestPipeline,
)

_INGEST_MODULE = "primus.backends.megatron.data.diffusion.preprocessing.pipelines.ingest"


class TestStreamingIngestPipeline:
    """Tests for the StreamingIngestPipeline with mocked I/O."""

    @patch(f"{_INGEST_MODULE}._arrow_to_tar")
    @patch(f"{_INGEST_MODULE}.download_with_backoff")
    @patch(f"{_INGEST_MODULE}.fetch_manifest")
    def test_processes_all_files(self, mock_manifest, mock_download, mock_convert):
        """Pipeline processes all entries, creates correct number of shards."""
        entries = [
            ("md5_0", "data-00000.arrow"),
            ("md5_1", "data-00001.arrow"),
            ("md5_2", "data-00002.arrow"),
        ]
        mock_manifest.return_value = ("https://base.url", entries)
        mock_convert.return_value = 100

        with tempfile.TemporaryDirectory() as tmp:
            pipeline = StreamingIngestPipeline(
                manifest_url="http://manifest.uri",
                input_dir=f"{tmp}/arrows",
                output_dir=f"{tmp}/output",
                split_name="train",
                max_workers=2,
                prefetch_depth=2,
            )

            def fake_download(url, dest, **kwargs):
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(b"fake")

            mock_download.side_effect = fake_download

            results = pipeline.run()

        assert results["files_processed"] == 3
        assert results["samples_written"] == 300
        assert results["shards_created"] == 3
        assert mock_download.call_count == 3
        assert mock_convert.call_count == 3

    @patch(f"{_INGEST_MODULE}._arrow_to_tar")
    @patch(f"{_INGEST_MODULE}.download_with_backoff")
    @patch(f"{_INGEST_MODULE}.fetch_manifest")
    def test_max_files_limits_processing(self, mock_manifest, mock_download, mock_convert):
        """max_files parameter limits how many files are processed."""
        entries = [(f"md5_{i}", f"data-{i:05d}.arrow") for i in range(10)]
        mock_manifest.return_value = ("https://base.url", entries)
        mock_convert.return_value = 50

        with tempfile.TemporaryDirectory() as tmp:
            pipeline = StreamingIngestPipeline(
                manifest_url="http://manifest.uri",
                input_dir=f"{tmp}/arrows",
                output_dir=f"{tmp}/output",
                split_name="train",
                max_files=3,
                max_workers=2,
                prefetch_depth=2,
            )

            def fake_download(url, dest, **kwargs):
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(b"fake")

            mock_download.side_effect = fake_download

            results = pipeline.run()

        assert results["files_processed"] == 3
        assert mock_download.call_count == 3

    @patch(f"{_INGEST_MODULE}._arrow_to_tar")
    @patch(f"{_INGEST_MODULE}.download_with_backoff")
    @patch(f"{_INGEST_MODULE}.fetch_manifest")
    def test_download_error_skips_file(self, mock_manifest, mock_download, mock_convert):
        """A failed download is skipped; the pipeline continues with remaining files."""
        entries = [("md5_0", "data-00000.arrow"), ("md5_1", "data-00001.arrow")]
        mock_manifest.return_value = ("https://base.url", entries)
        mock_convert.return_value = 100

        call_count = [0]

        def selective_download(url, dest, **kwargs):
            call_count[0] += 1
            if "data-00000" in url:
                raise RuntimeError("Download failed: HTTP 500")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"fake")

        mock_download.side_effect = selective_download

        with tempfile.TemporaryDirectory() as tmp:
            pipeline = StreamingIngestPipeline(
                manifest_url="http://manifest.uri",
                input_dir=f"{tmp}/arrows",
                output_dir=f"{tmp}/output",
                split_name="train",
                max_workers=1,
                prefetch_depth=2,
            )

            results = pipeline.run()

            assert results["files_processed"] == 1
            assert results["files_failed"] == 1
            assert results["samples_written"] == 100
            assert mock_convert.call_count == 1

            failed_manifest = Path(tmp) / "output" / "train" / "failed_files.json"
            assert failed_manifest.exists()
            failures = json.loads(failed_manifest.read_text())
            assert len(failures) == 1
            assert failures[0]["stage"] == "download"
            assert failures[0]["filename"] == "data-00000.arrow"

    @patch(f"{_INGEST_MODULE}._arrow_to_tar")
    @patch(f"{_INGEST_MODULE}.download_with_backoff")
    @patch(f"{_INGEST_MODULE}.fetch_manifest")
    def test_all_downloads_fail(self, mock_manifest, mock_download, mock_convert):
        """If all downloads fail, the pipeline completes with zero processed."""
        entries = [("md5_0", "data-00000.arrow"), ("md5_1", "data-00001.arrow")]
        mock_manifest.return_value = ("https://base.url", entries)
        mock_download.side_effect = RuntimeError("Download failed: HTTP 500")

        with tempfile.TemporaryDirectory() as tmp:
            pipeline = StreamingIngestPipeline(
                manifest_url="http://manifest.uri",
                input_dir=f"{tmp}/arrows",
                output_dir=f"{tmp}/output",
                split_name="train",
                max_workers=1,
                prefetch_depth=2,
            )

            results = pipeline.run()

        assert results["files_processed"] == 0
        assert results["files_failed"] == 2
        assert results["samples_written"] == 0
        assert mock_convert.call_count == 0

    @patch(f"{_INGEST_MODULE}._arrow_to_tar")
    @patch(f"{_INGEST_MODULE}.download_with_backoff")
    @patch(f"{_INGEST_MODULE}.fetch_manifest")
    def test_conversion_error_skips_file(self, mock_manifest, mock_download, mock_convert):
        """A failed conversion is skipped; partial tar is cleaned up."""
        entries = [
            ("md5_0", "data-00000.arrow"),
            ("md5_1", "data-00001.arrow"),
        ]
        mock_manifest.return_value = ("https://base.url", entries)

        def selective_convert(arrow_path, tar_path, offset):
            if "data-00000" in str(arrow_path):
                raise RuntimeError("Corrupt Arrow file")
            return 100

        mock_convert.side_effect = selective_convert

        with tempfile.TemporaryDirectory() as tmp:
            pipeline = StreamingIngestPipeline(
                manifest_url="http://manifest.uri",
                input_dir=f"{tmp}/arrows",
                output_dir=f"{tmp}/output",
                split_name="train",
                max_workers=1,
                prefetch_depth=2,
            )

            def fake_download(url, dest, **kwargs):
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(b"fake")

            mock_download.side_effect = fake_download

            results = pipeline.run()

            assert results["files_processed"] == 1
            assert results["files_failed"] == 1
            assert results["samples_written"] == 100

            output_train = Path(tmp) / "output" / "train"
            assert not (output_train / "shard_000000.tar").exists()
            assert (output_train / "failed_files.json").exists()
            failures = json.loads((output_train / "failed_files.json").read_text())
            assert len(failures) == 1
            assert failures[0]["stage"] == "conversion"

    @patch(f"{_INGEST_MODULE}._arrow_to_tar")
    @patch(f"{_INGEST_MODULE}.download_with_backoff")
    @patch(f"{_INGEST_MODULE}.fetch_manifest")
    def test_arrow_files_deleted_after_conversion(self, mock_manifest, mock_download, mock_convert):
        """Arrow files are cleaned up after conversion to tar."""
        entries = [("md5_0", "data-00000.arrow")]
        mock_manifest.return_value = ("https://base.url", entries)
        mock_convert.return_value = 10

        with tempfile.TemporaryDirectory() as tmp:
            arrow_dir = Path(tmp) / "arrows"
            pipeline = StreamingIngestPipeline(
                manifest_url="http://manifest.uri",
                input_dir=str(arrow_dir),
                output_dir=f"{tmp}/output",
                split_name="train",
                max_workers=1,
                prefetch_depth=2,
            )

            created_files = []

            def fake_download(url, dest, **kwargs):
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(b"fake arrow data")
                created_files.append(dest)

            mock_download.side_effect = fake_download

            pipeline.run()

            for f in created_files:
                assert not f.exists(), f"Arrow file should have been deleted: {f}"

    @patch(f"{_INGEST_MODULE}._arrow_to_tar")
    @patch(f"{_INGEST_MODULE}.download_with_backoff")
    @patch(f"{_INGEST_MODULE}.fetch_manifest")
    def test_sample_offset_accumulates(self, mock_manifest, mock_download, mock_convert):
        """Global sample offset is passed correctly across shards."""
        entries = [
            ("md5_0", "data-00000.arrow"),
            ("md5_1", "data-00001.arrow"),
            ("md5_2", "data-00002.arrow"),
        ]
        mock_manifest.return_value = ("https://base.url", entries)

        sample_counts = [100, 200, 150]
        mock_convert.side_effect = sample_counts

        with tempfile.TemporaryDirectory() as tmp:
            pipeline = StreamingIngestPipeline(
                manifest_url="http://manifest.uri",
                input_dir=f"{tmp}/arrows",
                output_dir=f"{tmp}/output",
                split_name="train",
                max_workers=1,
                prefetch_depth=2,
            )

            def fake_download(url, dest, **kwargs):
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(b"fake")

            mock_download.side_effect = fake_download

            results = pipeline.run()

        offsets = [c.args[2] for c in mock_convert.call_args_list]
        assert offsets == [0, 100, 300]
        assert results["samples_written"] == 450

    @patch(f"{_INGEST_MODULE}._arrow_to_tar")
    @patch(f"{_INGEST_MODULE}.download_with_backoff")
    @patch(f"{_INGEST_MODULE}.fetch_manifest")
    def test_resume_skips_existing_shards(self, mock_manifest, mock_download, mock_convert):
        """Pre-existing shard tars are skipped; only missing shards are downloaded."""
        entries = [
            ("md5_0", "data-00000.arrow"),
            ("md5_1", "data-00001.arrow"),
            ("md5_2", "data-00002.arrow"),
            ("md5_3", "data-00003.arrow"),
        ]
        mock_manifest.return_value = ("https://base.url", entries)
        mock_convert.return_value = 100

        with tempfile.TemporaryDirectory() as tmp:
            output_train = Path(tmp) / "output" / "train"
            output_train.mkdir(parents=True)

            (output_train / "shard_000000.tar").write_bytes(b"existing")
            (output_train / "shard_000002.tar").write_bytes(b"existing")

            pipeline = StreamingIngestPipeline(
                manifest_url="http://manifest.uri",
                input_dir=f"{tmp}/arrows",
                output_dir=f"{tmp}/output",
                split_name="train",
                max_workers=1,
                prefetch_depth=2,
            )

            def fake_download(url, dest, **kwargs):
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(b"fake")

            mock_download.side_effect = fake_download

            results = pipeline.run()

        assert results["files_processed"] == 2
        assert results["shards_skipped"] == 2
        assert results["shards_created"] == 4  # 2 new + 2 skipped
        assert results["samples_written"] == 200  # only from the 2 new shards
        assert mock_download.call_count == 2
        assert mock_convert.call_count == 2
