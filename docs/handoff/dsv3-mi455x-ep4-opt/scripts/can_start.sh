#!/usr/bin/env bash
# Exit 0 when the GPUs have been free of other users for >= MIN_FREE_MIN (30) minutes and it is
# night in Beijing: experiments start only in [NIGHT_FROM, NIGHT_TO) UTC+8 (user: no GPU in the day).
D=/home/boyan/project/deepep/dsv3opt
MIN_FREE_MIN=${MIN_FREE_MIN:-30}
NIGHT_FROM=${NIGHT_FROM:-0} NIGHT_TO=${NIGHT_TO:-7}
fs=$(sed -n 's/^free_since=//p' "$D/gpu_state" 2>/dev/null)
[ -n "$fs" ] || { echo "no state yet"; exit 1; }
age=$(( ($(date +%s) - fs) / 60 ))
h=$((10#$(TZ=Asia/Shanghai date +%H)))
echo "free for ${age} min, ${h}h UTC+8"
[ "$h" -ge "$NIGHT_FROM" ] && [ "$h" -lt "$NIGHT_TO" ] && [ "$age" -ge "$MIN_FREE_MIN" ]
