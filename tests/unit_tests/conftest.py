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
    megatron_path = os.environ.get("MEGATRON_PATH")
    if megatron_path is None or not os.path.exists(megatron_path):
        megatron_path = Path(__file__).resolve().parent.parent.parent / "third_party" / "Megatron-LM"
    sys.path.insert(0, str(megatron_path))
    print(f"[Primus] sys.path.insert: {megatron_path}")
