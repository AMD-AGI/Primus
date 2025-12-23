#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

WORLD_SIZE=4 LOCAL_RANK=0 RANK=0 python ./benchmark_fsdp.py  -dry
WORLD_SIZE=8 LOCAL_RANK=0 RANK=0 python ./benchmark_fsdp.py  -dry