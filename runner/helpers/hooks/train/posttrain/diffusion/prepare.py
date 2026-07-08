###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    primus_root = Path(__file__).resolve().parents[6]
    hook = primus_root / "runner" / "helpers" / "hooks" / "train" / "pretrain" / "diffusion" / "prepare.py"
    sys.argv[0] = str(hook)
    if "--module_name" not in sys.argv:
        sys.argv.extend(["--module_name", "post_trainer"])
    runpy.run_path(str(hook), run_name="__main__")


if __name__ == "__main__":
    main()
