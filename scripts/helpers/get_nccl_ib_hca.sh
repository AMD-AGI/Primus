#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# Get the list of IB devices
IB_DEVICES=$(ls /sys/class/infiniband/ 2>/dev/null)

# Check if no devices are found
if [[ -z "$IB_DEVICES" ]]; then
    echo "Error: No Infiniband devices found!" >&2
    exit 1
fi

# Initialize the NCCL_IB_HCA variable
NCCL_IB_HCA=""

# Loop through all IB devices
for ib_dev in $IB_DEVICES; do
    # Skip bonded devices, e.g. 'mlx5_bond_0/ xeth0'
    if [[ "$ib_dev" == *bond* || "$ib_dev" == *eth0* ]]; then
        echo "Info: Skipping bonded or storage device $ib_dev" >&2
        continue
    fi

    # Get the first available port for each IB device
    port=$(
        find "/sys/class/infiniband/$ib_dev/ports/" -mindepth 1 -maxdepth 1 -printf '%f\n' |
            sort -n | head -n 1
    )

    # If no port is found, continue to the next device
    if [[ -z "$port" ]]; then
        echo "Warning: No port found for device $ib_dev" >&2
        continue
    fi

    NCCL_IB_HCA+="${ib_dev}:${port},"
done

# Remove the trailing comma
NCCL_IB_HCA=${NCCL_IB_HCA%,}

# If NCCL_IB_HCA is still empty, exit with an error
if [[ -z "$NCCL_IB_HCA" ]]; then
    echo "Error: No active Infiniband ports found!" >&2
    exit 1
fi

# Print the result (to be captured by calling script)
echo "$NCCL_IB_HCA"
