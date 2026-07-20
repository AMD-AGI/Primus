#!/bin/bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

# shellcheck disable=SC2009  # need full args column; pgrep can't match python -m patterns
# Terminate all residual training/compile processes from the abandoned rocSHMEM
# smoke in THIS container. Patterns live in this file (not in the invoking
# shell's argv), so `pkill -f` cannot match its own parent shell.
echo "=== BEFORE: residual training/compile procs ==="
ps -eo pid,etimes,comm,args 2>/dev/null \
  | grep -iE 'torchrun|primus/cli|torch.distributed|hipcc|clang|cmake|ninja|llvm|multiprocessing' \
  | grep -vE 'grep|cleanup_rs.sh' || echo "(none before)"

for pat in 'torchrun' 'torch.distributed.run' 'primus/cli' 'run_pretrain.sh' \
           'run_odc.sh' 'multiprocessing.spawn' 'multiprocessing.resource_tracker' \
           'hipcc' 'cmake' 'ninja'; do
  pkill -9 -f "$pat" 2>/dev/null
done
pkill -9 -x python3 2>/dev/null
pkill -9 -x pt_main_thread 2>/dev/null
pkill -9 clang 2>/dev/null
pkill -9 'clang++' 2>/dev/null
pkill -9 llvm-as 2>/dev/null
pkill -9 lld 2>/dev/null
sleep 4

echo "=== AFTER: residual training/compile procs (want NONE) ==="
LEFT=$(ps -eo pid,etimes,comm,args 2>/dev/null \
  | grep -iE 'torchrun|primus/cli|torch.distributed|hipcc|clang|cmake|ninja|llvm|multiprocessing|python3' \
  | grep -vE 'grep|cleanup_rs.sh')
if [ -z "$LEFT" ]; then echo "(none)"; else echo "$LEFT"; fi

echo "=== rocm-smi GPU use% ==="
rocm-smi --showuse 2>/dev/null | grep -E 'GPU use \(%\)'
echo "=== rocm-smi compute pids ==="
rocm-smi --showpids 2>/dev/null | sed -n '1,12p'
