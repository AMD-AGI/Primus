###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import logging
import os
import sys
from typing import Any, Dict, Optional

import yaml


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Configure logging based on verbosity level.

    Args:
        verbose: Enable debug-level logging.
        quiet: Suppress all but error messages.
    """
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    # Configure root logger
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,  # Override any existing configuration
    )

    # Set library loggers to WARNING by default to reduce noise
    if not verbose:
        logging.getLogger("torch").setLevel(logging.WARNING)
        logging.getLogger("transformers").setLevel(logging.WARNING)


def load_config_file(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to configuration file. If None, tries default locations.

    Returns:
        Configuration dictionary.
    """
    default_paths = [
        os.path.expanduser("~/.primus/config.yaml"),
        os.path.expanduser("~/.config/primus/config.yaml"),
        ".primus.yaml",
    ]

    if config_path:
        paths_to_try = [config_path]
    else:
        paths_to_try = default_paths

    for path in paths_to_try:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    config = yaml.safe_load(f) or {}
                logging.getLogger(__name__).info(f"Loaded configuration from {path}")
                return config
            except Exception as e:
                logging.getLogger(__name__).warning(f"Failed to load config from {path}: {e}")

    return {}


def merge_config_with_args(config: Dict[str, Any], args: Any) -> None:
    """Merge configuration file with command-line arguments.

    Command-line arguments take precedence over config file.

    Args:
        config: Configuration dictionary from file.
        args: Parsed command-line arguments.
    """
    for key, value in config.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)


def is_debug_mode() -> bool:
    """Check if debug mode is enabled via environment variable.

    Returns:
        True if PRIMUS_DEBUG is set to a truthy value.
    """
    return os.getenv("PRIMUS_DEBUG", "").lower() in ("1", "true", "yes", "on")


def is_profile_mode() -> bool:
    """Check if profiling mode is enabled via environment variable.

    Returns:
        True if PRIMUS_PROFILE is set to a truthy value.
    """
    return os.getenv("PRIMUS_PROFILE", "").lower() in ("1", "true", "yes", "on")


def print_error(message: str) -> None:
    """Print error message to stderr with formatting.

    Args:
        message: Error message to print.
    """
    print(f"❌ Error: {message}", file=sys.stderr)


def print_warning(message: str) -> None:
    """Print warning message to stderr with formatting.

    Args:
        message: Warning message to print.
    """
    print(f"⚠️  Warning: {message}", file=sys.stderr)


def print_success(message: str) -> None:
    """Print success message with formatting.

    Args:
        message: Success message to print.
    """
    print(f"✅ {message}")


def print_info(message: str) -> None:
    """Print info message with formatting.

    Args:
        message: Info message to print.
    """
    print(f"ℹ️  {message}")


def safe_exit(code: int = 0, message: Optional[str] = None) -> None:
    """Safely exit the program with an optional message.

    Args:
        code: Exit code (0 for success, non-zero for error).
        message: Optional message to print before exiting.
    """
    if message:
        if code == 0:
            print_success(message)
        else:
            print_error(message)
    sys.exit(code)
