# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
End-to-end integration tests for RawDatasetPipeline.

Exercises the full preprocessing data flow (load -> resize -> encode -> shard)
with real files and real I/O. No GPU, no model weights, no HuggingFace downloads.
"""

import tarfile
from unittest.mock import patch

import pytest
from PIL import Image

from primus.backends.megatron.data.diffusion.preprocessing.pipelines.raw import (
    RawDatasetPipeline,
)
from tests.utils import PrimusUT


class TestRawPipelineE2E(PrimusUT):
    """End-to-end tests for RawDatasetPipeline with directory source."""

    @pytest.fixture(autouse=True)
    def setup_test_dataset(self, tmp_path):
        """Create temp directories with test images and captions."""
        self.input_dir = tmp_path / "input"
        self.output_dir = tmp_path / "output"
        images_dir = self.input_dir / "images"
        captions_dir = self.input_dir / "captions"
        images_dir.mkdir(parents=True)
        captions_dir.mkdir(parents=True)
        self.output_dir.mkdir()

        for i in range(8):
            img = Image.new("RGB", (64, 64), color=(i * 30, i * 20, i * 10))
            img.save(images_dir / f"{i:04d}.jpg")
            (captions_dir / f"{i:04d}.txt").write_text(f"caption {i}")

    def test_raw_pipeline_end_to_end(self):
        """Full pipeline run produces correct shards with jpg+txt entries."""
        pipeline = RawDatasetPipeline(
            source_type="directory",
            output_dir=str(self.output_dir),
            image_size=32,
            center_crop=True,
            variable_size=False,
            shard_size=5,
        )

        results = pipeline.run(input_dir=str(self.input_dir))

        assert results["samples_processed"] == 8
        assert results["samples_skipped"] == 0
        assert results["shards_written"] == 2

        tar_files = sorted(self.output_dir.glob("*.tar"))
        assert len(tar_files) == 2

        for tar_path in tar_files:
            with tarfile.open(str(tar_path), "r") as tar:
                members = tar.getmembers()
                assert len(members) > 0

                sample_keys = set()
                extensions_by_key = {}
                for m in members:
                    key, ext = m.name.split(".", 1)
                    sample_keys.add(key)
                    extensions_by_key.setdefault(key, set()).add(ext)

                for key, exts in extensions_by_key.items():
                    assert "jpg" in exts, f"Sample {key} missing jpg"
                    assert "txt" in exts, f"Sample {key} missing txt"

    def test_raw_pipeline_max_samples(self):
        """max_samples limits the number of processed samples."""
        pipeline = RawDatasetPipeline(
            source_type="directory",
            output_dir=str(self.output_dir),
            image_size=32,
            center_crop=True,
            variable_size=False,
            shard_size=1000,
            max_samples=3,
        )

        results = pipeline.run(input_dir=str(self.input_dir))

        assert results["samples_processed"] == 3
        tar_files = list(self.output_dir.glob("*.tar"))
        assert len(tar_files) == 1

    def test_raw_pipeline_empty_directory(self):
        """Empty images/captions dirs produce zero samples and no shards."""
        empty_input = self.output_dir.parent / "empty_input"
        (empty_input / "images").mkdir(parents=True)
        (empty_input / "captions").mkdir(parents=True)
        empty_output = self.output_dir.parent / "empty_output"
        empty_output.mkdir()

        pipeline = RawDatasetPipeline(
            source_type="directory",
            output_dir=str(empty_output),
            image_size=32,
            center_crop=True,
            variable_size=False,
            shard_size=1000,
        )

        results = pipeline.run(input_dir=str(empty_input))

        assert results["samples_processed"] == 0
        assert results["samples_skipped"] == 0
        tar_files = list(empty_output.glob("*.tar"))
        assert len(tar_files) == 0

    def test_raw_pipeline_max_samples_is_total_not_per_rank(self):
        """max_samples=4 with world_size=2 produces 4 total samples (2 per rank), not 8."""
        total_across_ranks = 0
        dist_info_path = (
            "primus.backends.megatron.data.diffusion.preprocessing.pipelines.raw" ".get_distributed_info"
        )

        for rank in range(2):
            rank_output = self.output_dir.parent / f"rank_{rank}_output"
            rank_output.mkdir()

            with patch(dist_info_path, return_value=(rank, 2)):
                pipeline = RawDatasetPipeline(
                    source_type="directory",
                    output_dir=str(rank_output),
                    image_size=32,
                    center_crop=True,
                    variable_size=False,
                    shard_size=1000,
                    max_samples=4,
                )
                results = pipeline.run(input_dir=str(self.input_dir))

            assert (
                results["samples_processed"] == 2
            ), f"Rank {rank} should process 2 samples, got {results['samples_processed']}"
            total_across_ranks += results["samples_processed"]

        assert total_across_ranks == 4, f"Expected 4 total samples across 2 ranks, got {total_across_ranks}"

    def test_raw_pipeline_round_robin_distribution(self):
        """Streaming round-robin split covers all samples exactly once across ranks.

        With world_size=2 and 8 items, each rank should process 4 samples by
        global load index (rank 0 -> even indices, rank 1 -> odd), and the union
        of captions across ranks must equal the full set with no overlap.
        """
        dist_info_path = (
            "primus.backends.megatron.data.diffusion.preprocessing.pipelines.raw" ".get_distributed_info"
        )

        def captions_in(output_dir):
            captions = set()
            for tar_path in sorted(output_dir.glob("*.tar")):
                with tarfile.open(str(tar_path), "r") as tar:
                    for m in tar.getmembers():
                        if m.name.endswith(".txt"):
                            captions.add(tar.extractfile(m).read().decode("utf-8"))
            return captions

        rank_captions = {}
        for rank in range(2):
            rank_output = self.output_dir.parent / f"rr_rank_{rank}_output"
            rank_output.mkdir()

            with patch(dist_info_path, return_value=(rank, 2)):
                pipeline = RawDatasetPipeline(
                    source_type="directory",
                    output_dir=str(rank_output),
                    image_size=32,
                    center_crop=True,
                    variable_size=False,
                    shard_size=1000,
                )
                results = pipeline.run(input_dir=str(self.input_dir))

            assert (
                results["samples_processed"] == 4
            ), f"Rank {rank} should process 4 of 8 samples, got {results['samples_processed']}"
            rank_captions[rank] = captions_in(rank_output)

        # No overlap between ranks
        assert rank_captions[0].isdisjoint(
            rank_captions[1]
        ), f"Ranks processed overlapping samples: {rank_captions[0] & rank_captions[1]}"
        # Complete coverage of all 8 captions
        all_captions = rank_captions[0] | rank_captions[1]
        expected = {f"caption {i}" for i in range(8)}
        assert all_captions == expected, f"Expected {expected}, got {all_captions}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
