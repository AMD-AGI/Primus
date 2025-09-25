#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

IP_INTERFACE=$(ip -o -4 addr show | awk -v ip="$(hostname -I | awk '{print $1}')" '$4 ~ ip {print $2}')

if [[ -z "$IP_INTERFACE" ]]; then
    echo "Error: No active ip interface found!" >&2
    exit 1
fi

# Print the result (to be captured by calling script)
echo "$IP_INTERFACE"
