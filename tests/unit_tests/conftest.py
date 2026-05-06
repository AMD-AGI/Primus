###############################################################################
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


import os
import sys
from pathlib import Path


def pytest_configure(config):
    # Add project root first to ensure main primus package takes precedence
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Add Megatron-LM after project root (append instead of insert at 0)
    megatron_path = os.environ.get("MEGATRON_PATH")
    if megatron_path is None or not os.path.exists(megatron_path):
        megatron_path = project_root / "third_party" / "Megatron-LM"
    if str(megatron_path) not in sys.path:
        sys.path.append(str(megatron_path))
