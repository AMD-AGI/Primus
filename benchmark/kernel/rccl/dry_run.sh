#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

WORLD_SIZE=72 LOCAL_RANK=0 RANK=0 python ./benchmark_allreduce.py  -dry