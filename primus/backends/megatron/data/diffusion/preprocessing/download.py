# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Reusable download utilities for dataset preparation pipelines.

Provides HTTP download with exponential backoff, MD5 verification,
and MLCommons R2 manifest resolution (.uri / .md5 protocol).
Uses urllib.request (stdlib) -- no external HTTP dependencies needed.
"""

import hashlib
import logging
import random
import shutil
import time
import urllib.error
import urllib.request
from http.client import HTTPResponse
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

_DOWNLOAD_TIMEOUT = 300  # 5 min per file
_MAX_RETRIES = 5
_BASE_DELAY = 1.0
_MAX_DELAY = 60.0
_USER_AGENT = "Wget/1.21"


class _MD5MismatchError(Exception):
    """Raised when a downloaded file's MD5 doesn't match the expected value."""


def _is_retryable(status_code: int) -> bool:
    return status_code == 429 or 500 <= status_code < 600


def download_with_backoff(
    url: str,
    dest: Path,
    expected_md5: Optional[str] = None,
    max_retries: int = _MAX_RETRIES,
    base_delay: float = _BASE_DELAY,
    timeout: int = _DOWNLOAD_TIMEOUT,
) -> None:
    """Download a file via HTTP with exponential backoff on 429/5xx.

    Streams the response to disk to avoid holding large files in memory.
    Verifies MD5 checksum if provided.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            resp: HTTPResponse = urllib.request.urlopen(req, timeout=timeout)
            with open(dest, "wb") as f:
                shutil.copyfileobj(resp, f)

            if expected_md5:
                actual_md5 = hashlib.md5(dest.read_bytes()).hexdigest()
                if actual_md5 != expected_md5:
                    dest.unlink(missing_ok=True)
                    raise _MD5MismatchError(
                        f"MD5 mismatch for {dest.name}: " f"expected {expected_md5}, got {actual_md5}"
                    )
            return

        except urllib.error.HTTPError as e:
            last_error = e
            if not _is_retryable(e.code) or attempt == max_retries:
                dest.unlink(missing_ok=True)
                raise RuntimeError(
                    f"HTTP {e.code} downloading {url} " f"(attempt {attempt + 1}/{max_retries + 1})"
                ) from e
            delay = min(base_delay * (2**attempt) + random.uniform(0, 1), _MAX_DELAY)
            logger.warning(f"  HTTP {e.code} on {url}, retry {attempt + 1}/{max_retries} " f"in {delay:.1f}s")
            time.sleep(delay)

        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_error = e
            if attempt == max_retries:
                dest.unlink(missing_ok=True)
                raise RuntimeError(f"Download failed for {url} after {max_retries + 1} attempts: {e}") from e
            delay = min(base_delay * (2**attempt) + random.uniform(0, 1), _MAX_DELAY)
            logger.warning(
                f"  Network error on {url}: {e}, retry {attempt + 1}/{max_retries} " f"in {delay:.1f}s"
            )
            time.sleep(delay)

        except _MD5MismatchError as e:
            last_error = e
            if attempt == max_retries:
                dest.unlink(missing_ok=True)
                raise RuntimeError(str(e)) from e
            delay = min(base_delay * (2**attempt) + random.uniform(0, 1), _MAX_DELAY)
            logger.warning(f"  {e}, retry {attempt + 1}/{max_retries} in {delay:.1f}s")
            time.sleep(delay)

    dest.unlink(missing_ok=True)
    raise RuntimeError(f"Download failed for {url}: {last_error}")


def fetch_url_text(url: str, timeout: int = 30) -> str:
    """Fetch a small text resource (manifest, etc.) via HTTP."""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    resp = urllib.request.urlopen(req, timeout=timeout)
    return resp.read().decode("utf-8")


def parse_md5_manifest(
    manifest_text: str,
    suffix_filter: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """Parse an MLCommons .md5 manifest into (md5, filename) pairs.

    Args:
        manifest_text: Raw text content of the .md5 file.
        suffix_filter: If provided, only include files ending with this
            suffix (e.g. ".arrow"). None means include all files.

    Returns:
        Sorted list of (md5, filename) tuples, sorted by filename
        for deterministic ordering.
    """
    entries = []
    for line in manifest_text.strip().splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        md5, fname = parts
        if suffix_filter is not None and not fname.endswith(suffix_filter):
            continue
        entries.append((md5, fname))
    entries.sort(key=lambda x: x[1])
    return entries


def fetch_manifest(manifest_url: str) -> Tuple[str, List[Tuple[str, str]]]:
    """Fetch .uri and .md5 manifests, return (base_url, [(md5, filename)]).

    The manifest_url should end with '.uri' or '.md5'. The function derives
    the complementary URL by replacing the suffix.
    """
    uri_url = manifest_url.replace(".md5", ".uri")
    md5_url = manifest_url.replace(".uri", ".md5")

    base_url = fetch_url_text(uri_url).strip()
    md5_text = fetch_url_text(md5_url)
    entries = parse_md5_manifest(md5_text)

    logger.info(f"Manifest: {len(entries)} files, base URL: {base_url}")
    return base_url, entries
