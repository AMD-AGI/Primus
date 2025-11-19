###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
from typing import Dict, List

import torch

_backend_streams: Dict[int, List[torch.cuda.Stream]] = {}


def get_backend_stream(size=1, priority=0, prefix=""):
    global _backend_streams

    key = (priority, prefix)
    if key not in _backend_streams or len(_backend_streams[key]) < size:
        _backend_streams[key] = [torch.cuda.Stream(priority=priority) for _ in range(size)]

    return _backend_streams[key][:size]


def is_hip():
    if torch.version.hip is not None:
        return True
    return False
