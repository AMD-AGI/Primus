###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Read a HuggingFace ``config.json`` as a JSON object.

The one reader shared by the native checkpoint converters
(``primus.backends.megatron.checkpoint``) and the SFT checkpoint-conversion hook, so both
accept and refuse the same files. Standard library only: the hook can import it before
any requirement is installed and without importing torch or Megatron.
"""

import json
import os

__all__ = ["HFConfigError", "read_hf_config"]


class HFConfigError(ValueError):
    """config.json exists but cannot be read as a JSON object."""


def read_hf_config(hf_dir: str) -> dict:
    """Parse ``<hf_dir>/config.json`` as a JSON object.

    Raises ``FileNotFoundError`` if the file does not exist, and ``HFConfigError`` if it
    exists but cannot be read, is not valid JSON, or is not a JSON object.
    """
    cfg_file = os.path.join(hf_dir, "config.json")
    if not os.path.isfile(cfg_file):
        raise FileNotFoundError(f"no such config file: {cfg_file}")
    try:
        with open(cfg_file) as f:
            cfg = json.load(f)
    except (OSError, ValueError) as exc:
        raise HFConfigError(f"cannot read {cfg_file}: {exc}") from exc
    if not isinstance(cfg, dict):
        raise HFConfigError(f"{cfg_file} is not a JSON object")
    return cfg
