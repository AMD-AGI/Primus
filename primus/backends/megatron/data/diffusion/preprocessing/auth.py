# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Authentication utilities for HuggingFace dataset access.

Provides secure token loading from files with permission checks.
"""

import logging
import os
import stat
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class HFAuthError(Exception):
    """Exception raised for HuggingFace authentication errors."""


def check_file_permissions(file_path: Path) -> bool:
    """
    Check if file has secure permissions (600 or 400).

    Args:
        file_path: Path to token file

    Returns:
        True if permissions are secure, False otherwise
    """
    try:
        file_stat = os.stat(file_path)
        mode = stat.S_IMODE(file_stat.st_mode)

        # Check if file is readable by others or group
        if mode & (stat.S_IRGRP | stat.S_IROTH):
            return False

        # Check if file is writable by others or group
        if mode & (stat.S_IWGRP | stat.S_IWOTH):
            return False

        return True
    except OSError as e:
        logger.error(f"Failed to check file permissions: {e}")
        return False


def load_token_from_file(token_file: str) -> str:
    """
    Load HuggingFace token from file with security checks.

    Args:
        token_file: Path to file containing HuggingFace token

    Returns:
        Token string

    Raises:
        HFAuthError: If file has insecure permissions or cannot be read
    """
    token_path = Path(token_file).expanduser().resolve()

    # Check if file exists
    if not token_path.exists():
        raise HFAuthError(
            f"Token file not found: {token_path}\n" f"Please create the file or check the path."
        )

    # Check if it's a file (not directory)
    if not token_path.is_file():
        raise HFAuthError(f"Token path is not a file: {token_path}")

    # Check file permissions
    if not check_file_permissions(token_path):
        raise HFAuthError(
            f"Token file has insecure permissions: {token_path}\n"
            f"Please set secure permissions:\n"
            f"  chmod 600 {token_path}\n"
            f"Current permissions allow read/write by group or others."
        )

    # Read token
    try:
        with open(token_path, "r") as f:
            token = f.read().strip()

        if not token:
            raise HFAuthError(f"Token file is empty: {token_path}")

        # Basic validation (HF tokens start with 'hf_')
        if not token.startswith("hf_"):
            logger.warning(
                f"Token from {token_path} doesn't start with 'hf_' - "
                f"this may not be a valid HuggingFace token"
            )

        logger.info(f"Loaded HuggingFace token from {token_path}")
        return token

    except IOError as e:
        raise HFAuthError(f"Failed to read token file {token_path}: {e}")


def setup_hf_authentication(token_file: Optional[str] = None, use_env: bool = True) -> Optional[str]:
    """
    Setup HuggingFace authentication with multiple fallback options.

    Priority:
    1. Token from file (if token_file provided)
    2. Token from HF_TOKEN environment variable (if use_env=True)
    3. Token from HF CLI login (~/.cache/huggingface/token)
    4. No authentication (public datasets only)

    Args:
        token_file: Optional path to token file
        use_env: Whether to check HF_TOKEN environment variable

    Returns:
        Token string if found, None otherwise

    Side effects:
        Sets HF_TOKEN environment variable if token is found
    """
    # Priority 1: Token file
    if token_file:
        try:
            token = load_token_from_file(token_file)
            os.environ["HF_TOKEN"] = token
            logger.info("Using HuggingFace token from file")
            return token
        except HFAuthError as e:
            logger.error(str(e))
            raise

    # Priority 2: Environment variable
    if use_env and "HF_TOKEN" in os.environ:
        token = os.environ["HF_TOKEN"]
        if token:
            logger.info("Using HuggingFace token from HF_TOKEN environment variable")
            return token

    # Priority 3: HF CLI login
    hf_cache_token = Path.home() / ".cache" / "huggingface" / "token"
    if hf_cache_token.exists():
        if not check_file_permissions(hf_cache_token):
            logger.warning(
                f"HuggingFace CLI token file has insecure permissions: {hf_cache_token}. "
                "Skipping it; run `chmod 600` on the file to use it."
            )
        else:
            try:
                with open(hf_cache_token, "r") as f:
                    token = f.read().strip()
                if token:
                    os.environ["HF_TOKEN"] = token
                    logger.info("Using HuggingFace token from CLI login (~/.cache/huggingface/token)")
                    return token
            except IOError as e:
                logger.debug(f"Could not read HF CLI token: {e}")

    # No authentication found
    logger.info("No HuggingFace authentication found. " "Only public datasets will be accessible.")
    return None


__all__ = [
    "HFAuthError",
    "load_token_from_file",
    "setup_hf_authentication",
    "check_file_permissions",
]
