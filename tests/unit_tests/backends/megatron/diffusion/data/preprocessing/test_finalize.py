# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Tests for dataset finalization and encoder auth error detection.

Tests finalize_energon_dataset from finalize.py and
_raise_encoder_auth_error from encoded.py.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from primus.backends.megatron.data.diffusion.preprocessing.finalize import (
    finalize_energon_dataset,
)
from primus.backends.megatron.data.diffusion.preprocessing.pipelines.encoded import (
    EncodedDatasetPipeline,
)
from tests.utils import PrimusUT


def _create_dummy_tar(directory: Path, name: str = "000000.tar") -> None:
    """Create an empty .tar file for testing."""
    (directory / name).write_bytes(b"")


class TestDatasetYamlContent(PrimusUT):
    """Tests that finalize_energon_dataset writes correct dataset.yaml."""

    @patch(
        "primus.backends.megatron.data.diffusion.preprocessing.finalize._prepare_programmatic",
        side_effect=ImportError("forced: exercise subprocess fallback path"),
    )
    @patch("primus.backends.megatron.data.diffusion.preprocessing.validate.validate_energon_dataset")
    @patch("subprocess.Popen")
    def test_preencoded_yaml(self, mock_popen, mock_validate, mock_programmatic):
        """dataset.yaml contains CrudeWebdataset with encoding: preencoded."""
        mock_process = MagicMock()
        mock_process.communicate.return_value = ("Done\n", None)
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        with tempfile.TemporaryDirectory() as tmpdir:
            _create_dummy_tar(Path(tmpdir))
            finalize_energon_dataset(output_dir=tmpdir, train_split=1.0, encoding="preencoded")

            dataset_yaml = Path(tmpdir) / ".nv-meta" / "dataset.yaml"
            assert dataset_yaml.exists()
            content = dataset_yaml.read_text()
            assert "__class__: CrudeWebdataset" in content
            assert "encoding: preencoded" in content

            # Verify split_input passed to subprocess stdin
            mock_process.communicate.assert_called_once_with(input="1.0, 0.0, 0.0\nn\n", timeout=300)

    @patch(
        "primus.backends.megatron.data.diffusion.preprocessing.finalize._prepare_programmatic",
        side_effect=ImportError("forced: exercise subprocess fallback path"),
    )
    @patch("primus.backends.megatron.data.diffusion.preprocessing.validate.validate_energon_dataset")
    @patch("subprocess.Popen")
    def test_raw_yaml_and_split_format(self, mock_popen, mock_validate, mock_programmatic):
        """dataset.yaml contains encoding: raw; split ratios are formatted correctly."""
        mock_process = MagicMock()
        mock_process.communicate.return_value = ("Done\n", None)
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        with tempfile.TemporaryDirectory() as tmpdir:
            _create_dummy_tar(Path(tmpdir))
            finalize_energon_dataset(output_dir=tmpdir, train_split=0.8, encoding="raw")

            content = (Path(tmpdir) / ".nv-meta" / "dataset.yaml").read_text()
            assert "encoding: raw" in content

            call_kwargs = mock_process.communicate.call_args
            split_input = call_kwargs.kwargs.get("input") or call_kwargs[1].get("input")
            assert split_input.startswith("0.8, ")
            assert split_input.endswith("\nn\n")
            parts = split_input.split("\n")[0].split(", ")
            self.assertAlmostEqual(float(parts[0]), 0.8)
            self.assertAlmostEqual(float(parts[1]), 0.1)
            self.assertAlmostEqual(float(parts[2]), 0.1)


class TestRaiseEncoderAuthError(PrimusUT):
    """Tests for EncodedDatasetPipeline._raise_encoder_auth_error."""

    def test_401_error_gives_auth_hint(self):
        """Exception containing '401' raises RuntimeError with auth hint."""
        original = Exception("HTTP 401 Unauthorized")
        with self.assertRaises(RuntimeError) as ctx:
            EncodedDatasetPipeline._raise_encoder_auth_error("VAE", "my-model", original)
        assert "HuggingFace authentication" in str(ctx.exception)
        assert "VAE" in str(ctx.exception)

    def test_token_error_gives_auth_hint(self):
        """Exception containing 'token' raises RuntimeError with auth hint."""
        original = Exception("Please pass a valid token")
        with self.assertRaises(RuntimeError) as ctx:
            EncodedDatasetPipeline._raise_encoder_auth_error("T5-XXL", "my-model", original)
        assert "HuggingFace authentication" in str(ctx.exception)
        assert "T5-XXL" in str(ctx.exception)

    def test_non_auth_error_preserves_message(self):
        """Non-auth exception re-raises with original message intact."""
        original = Exception("CUDA out of memory")
        with self.assertRaises(RuntimeError) as ctx:
            EncodedDatasetPipeline._raise_encoder_auth_error("CLIP-L", "my-model", original)
        msg = str(ctx.exception)
        assert "CUDA out of memory" in msg
        assert "CLIP-L" in msg
        assert "HuggingFace authentication" not in msg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
