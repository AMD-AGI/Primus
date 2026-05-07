#!/bin/bash
echo "Killing training processes to release GPUs..."
pkill -9 -f "primus-cli"
pkill -9 -f "megatron"
pkill -9 -f "python"
echo "Done."
