###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Recommendation (DLRMv4) pre-train preparation hook for primus-cli direct.

Invoked by runner/helpers/hooks/train/pretrain/prepare_experiment.sh after the
shell hooks of this folder (00_install_requirements.sh). It selects single run
mode: the DLRMv4 trainer starts one process per GPU itself (torch
multiprocessing spawn), so primus-cli must
start one Primus process per node instead of a torchrun fan-out.
"""

import argparse
import sys
from pathlib import Path

PRIMUS_ROOT = Path(__file__).resolve().parents[6]
if str(PRIMUS_ROOT) not in sys.path:
    sys.path.insert(0, str(PRIMUS_ROOT))

from runner.helpers.hooks.train.pretrain.utils import log_info  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a Primus recommendation run")
    parser.add_argument("--config", required=True)
    parser.parse_known_args()

    # A launcher decision, needed before Python starts: execute_hooks.sh exports
    # it from this line.
    log_info("recommendation: the trainer spawns its own ranks; env.RUN_MODE=single")
    print("env.RUN_MODE=single")


if __name__ == "__main__":
    main()
