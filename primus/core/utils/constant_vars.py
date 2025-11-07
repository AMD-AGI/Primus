###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# yaml configs
PRIMUS_CONFIG_NAME = "PrimusConfig"

# local platform
LOCAL_NUM_NODES = 1
LOCAL_NODE_RANK = 0
LOCAL_WORLD_SIZE = 1
LOCAL_MASTER_ADDR = "127.0.0.1"
LOCAL_MASTER_PORT = 1024

# modules
PRIMUS_MASTER = "master"
PRE_TRAINER = "pre_trainer"
SFT_TRAINER = "sft_trainer"
MAMBA_TRAINER = "mamba_trainer"
