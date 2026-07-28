# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Tests for download utilities (download.py).

Covers:
- parse_md5_manifest: parsing, filtering, sorting
- download_with_backoff: retry logic, MD5 verification, timeout handling
- fetch_url_text: basic HTTP text fetch
"""

import hashlib
import io
import tempfile
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from primus.backends.megatron.data.diffusion.preprocessing.download import (
    download_with_backoff,
    fetch_url_text,
    parse_md5_manifest,
)

_DOWNLOAD_MODULE = "primus.backends.megatron.data.diffusion.preprocessing.download"


class TestParseMd5Manifest:
    def test_basic_parsing(self):
        text = "abc123 data-00000.arrow\ndef456 data-00001.arrow\n"
        entries = parse_md5_manifest(text)
        assert entries == [("abc123", "data-00000.arrow"), ("def456", "data-00001.arrow")]

    def test_suffix_filter(self):
        text = "abc123 data-00000.arrow\ndef456 readme.txt\n"
        entries = parse_md5_manifest(text, suffix_filter=".arrow")
        assert len(entries) == 1
        assert entries[0][1] == "data-00000.arrow"

    def test_no_filter_returns_all(self):
        text = "abc123 data-00000.arrow\ndef456 readme.txt\n"
        entries = parse_md5_manifest(text)
        assert len(entries) == 2

    def test_sorts_by_filename(self):
        text = "bbb data-00002.arrow\naaa data-00001.arrow\nccc data-00000.arrow\n"
        entries = parse_md5_manifest(text)
        assert [e[1] for e in entries] == [
            "data-00000.arrow",
            "data-00001.arrow",
            "data-00002.arrow",
        ]

    def test_skips_malformed_lines(self):
        text = "abc123 data-00000.arrow\nbadline\n\n  \n"
        entries = parse_md5_manifest(text)
        assert len(entries) == 1


class TestDownloadWithBackoff:
    """Tests for download_with_backoff with mocked HTTP."""

    @patch(f"{_DOWNLOAD_MODULE}.urllib.request.urlopen")
    def test_success_on_first_try(self, mock_urlopen):
        content = b"hello world"
        mock_resp = MagicMock()
        mock_resp.read.side_effect = [content, b""]
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "test.bin"
            download_with_backoff("http://example.com/test.bin", dest, max_retries=0)
            assert dest.exists()

    @patch(f"{_DOWNLOAD_MODULE}.time.sleep")
    @patch(f"{_DOWNLOAD_MODULE}.urllib.request.urlopen")
    def test_retries_on_429(self, mock_urlopen, mock_sleep):
        content = b"success data"
        mock_resp_ok = MagicMock()
        mock_resp_ok.read.side_effect = [content, b""]
        mock_resp_ok.__enter__ = MagicMock(return_value=mock_resp_ok)
        mock_resp_ok.__exit__ = MagicMock(return_value=False)

        mock_urlopen.side_effect = [
            urllib.error.HTTPError("http://x", 429, "Too Many Requests", {}, io.BytesIO(b"")),
            urllib.error.HTTPError("http://x", 429, "Too Many Requests", {}, io.BytesIO(b"")),
            mock_resp_ok,
        ]

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "test.bin"
            download_with_backoff("http://example.com/test.bin", dest, max_retries=3, base_delay=0.01)
            assert dest.exists()
            assert mock_urlopen.call_count == 3
            assert mock_sleep.call_count == 2

    @patch(f"{_DOWNLOAD_MODULE}.time.sleep")
    @patch(f"{_DOWNLOAD_MODULE}.urllib.request.urlopen")
    def test_retries_on_503(self, mock_urlopen, mock_sleep):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "http://x", 503, "Service Unavailable", {}, io.BytesIO(b"")
        )

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "test.bin"
            with pytest.raises(RuntimeError, match="HTTP 503"):
                download_with_backoff("http://example.com/test.bin", dest, max_retries=2, base_delay=0.01)
            assert mock_urlopen.call_count == 3  # initial + 2 retries

    @patch(f"{_DOWNLOAD_MODULE}.urllib.request.urlopen")
    def test_no_retry_on_404(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError("http://x", 404, "Not Found", {}, io.BytesIO(b""))

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "test.bin"
            with pytest.raises(RuntimeError, match="HTTP 404"):
                download_with_backoff("http://example.com/test.bin", dest, max_retries=3, base_delay=0.01)
            assert mock_urlopen.call_count == 1

    @patch(f"{_DOWNLOAD_MODULE}.urllib.request.urlopen")
    def test_md5_verification_pass(self, mock_urlopen):
        content = b"test content"
        expected_md5 = hashlib.md5(content).hexdigest()

        mock_resp = MagicMock()
        mock_resp.read.side_effect = [content, b""]
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "test.bin"
            download_with_backoff("http://example.com/test.bin", dest, expected_md5=expected_md5)
            assert dest.exists()

    @patch(f"{_DOWNLOAD_MODULE}.urllib.request.urlopen")
    def test_md5_verification_fail(self, mock_urlopen):
        content = b"test content"

        mock_resp = MagicMock()
        mock_resp.read.side_effect = [content, b""]
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "test.bin"
            with pytest.raises(RuntimeError, match="MD5 mismatch"):
                download_with_backoff(
                    "http://example.com/test.bin",
                    dest,
                    expected_md5="bad_md5",
                    max_retries=0,
                )
            assert not dest.exists()

    @patch(f"{_DOWNLOAD_MODULE}.time.sleep")
    @patch(f"{_DOWNLOAD_MODULE}.urllib.request.urlopen")
    def test_retries_on_network_error(self, mock_urlopen, mock_sleep):
        content = b"ok"
        mock_resp_ok = MagicMock()
        mock_resp_ok.read.side_effect = [content, b""]
        mock_resp_ok.__enter__ = MagicMock(return_value=mock_resp_ok)
        mock_resp_ok.__exit__ = MagicMock(return_value=False)

        mock_urlopen.side_effect = [
            urllib.error.URLError("Connection refused"),
            mock_resp_ok,
        ]

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "test.bin"
            download_with_backoff("http://example.com/test.bin", dest, max_retries=2, base_delay=0.01)
            assert dest.exists()
            assert mock_urlopen.call_count == 2

    @patch(f"{_DOWNLOAD_MODULE}.time.sleep")
    @patch(f"{_DOWNLOAD_MODULE}.urllib.request.urlopen")
    def test_retries_on_md5_mismatch_then_succeeds(self, mock_urlopen, mock_sleep):
        good_content = b"correct data"
        bad_content = b"corrupted data"
        expected_md5 = hashlib.md5(good_content).hexdigest()

        mock_resp_bad = MagicMock()
        mock_resp_bad.read.side_effect = [bad_content, b""]
        mock_resp_bad.__enter__ = MagicMock(return_value=mock_resp_bad)
        mock_resp_bad.__exit__ = MagicMock(return_value=False)

        mock_resp_good = MagicMock()
        mock_resp_good.read.side_effect = [good_content, b""]
        mock_resp_good.__enter__ = MagicMock(return_value=mock_resp_good)
        mock_resp_good.__exit__ = MagicMock(return_value=False)

        mock_urlopen.side_effect = [mock_resp_bad, mock_resp_good]

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "test.bin"
            download_with_backoff(
                "http://example.com/test.bin",
                dest,
                expected_md5=expected_md5,
                max_retries=3,
                base_delay=0.01,
            )
            assert dest.exists()
            assert dest.read_bytes() == good_content
            assert mock_urlopen.call_count == 2
            assert mock_sleep.call_count == 1

    @patch(f"{_DOWNLOAD_MODULE}.time.sleep")
    @patch(f"{_DOWNLOAD_MODULE}.urllib.request.urlopen")
    def test_md5_mismatch_exhausts_retries(self, mock_urlopen, mock_sleep):
        bad_content = b"always bad"
        expected_md5 = "0000000000000000"

        def make_bad_resp(*args, **kwargs):
            resp = MagicMock()
            resp.read.side_effect = [bad_content, b""]
            resp.__enter__ = MagicMock(return_value=resp)
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        mock_urlopen.side_effect = make_bad_resp

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "test.bin"
            with pytest.raises(RuntimeError, match="MD5 mismatch"):
                download_with_backoff(
                    "http://example.com/test.bin",
                    dest,
                    expected_md5=expected_md5,
                    max_retries=2,
                    base_delay=0.01,
                )
            assert mock_urlopen.call_count == 3  # initial + 2 retries


class TestFetchUrlText:
    @patch(f"{_DOWNLOAD_MODULE}.urllib.request.urlopen")
    def test_returns_decoded_text(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"https://example.com/base"
        mock_urlopen.return_value = mock_resp

        result = fetch_url_text("http://example.com/manifest.uri")
        assert result == "https://example.com/base"
