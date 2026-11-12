#!/usr/bin/env bash
# Stage the yambda-5b training set from shared NFS onto node-local NVMe.
#
# Why: the streaming loaders mmap a ~350 GB flat-event cache off shared NFS, and
# every window boundary re-scans it. Reading it from local NVMe removes NFS
# latency from the cold path. Pair with:
#
#   LOCAL_DATA_MOUNT=/scratch/$USER/dlrm_data \
#   DLRM_DATA_PATH=/scratch/$USER/dlrm_data \
#   FORCE_PROVISION=1 sbatch ... scripts/launch_slurm.sh
#
# (FORCE_PROVISION is needed once so the container is recreated with the local
# bind mount; docker fixes mounts at `run`.)
#
# What it copies: ONLY the reference data set (see REF_FILES below) -- exactly
# what the MLCommons reference (b5e8e2a) prepares: the hstu_cache_L<N> files in
# its sha256sums_yambda_5b_cache.txt manifest, processed_5b/item_popularity.npy
# and the two shared_metadata item mappings its reader opens. Nothing else is
# staged, in particular no offline-derived file that the reference does not
# prepare (other than the PR 907 window index ts_indices/ts_offsets, which
# the user allowed; never ts_index_manifest, window_index_*,
# positions_*_m<k>/anchor_ts_*_m<k>, embeddings.parquet, session_index,
# test_events, split_meta, *lock). User rule 2026-10-12: no prior knowledge
# about the training data may be generated offline or before RUN_START beyond
# what the reference prepares. The 46 GB train_sessions.parquet is skipped --
# it is the *source* for the npy cache, and the cache's `_READY` sentinel means
# the build is never re-run. Files are copied byte for byte (dd/cp), never
# transformed.
#
# Switch: yambda_5b.gin item (17b), apply_env_bootstrap.
# DLRM_HSTU_FP8_STAGE_REF_ONLY (gin default True = the reference-only staging
# above; must stay True for an MLPerf submission). Read from the gin file of
# this tree (see "Reference-only staging switch" below); a non-empty env var of
# the same name wins (1|0). 0 restores the c68395d stager: the whole
# cache directory, every shared_metadata file and the processed_5b split files.
#
# Idempotent: files whose size already matches the source are skipped, so a
# re-run after a partial copy resumes rather than restarting.
#
# Everything this writes is SCRATCH. The staged copy is a byte-for-byte
# reproduction of $SRC, so it can be deleted at any time and re-made in ~140 s;
# nothing here is ever the only copy of anything. `--purge` removes it.
#
# Usage:  examples/mlperf/dlrm_v4/data/stage_data_local.sh [dest_root] [history_length]
#         examples/mlperf/dlrm_v4/data/stage_data_local.sh --purge [dest_root]
#         srun --nodelist=<node> ... examples/mlperf/dlrm_v4/data/stage_data_local.sh
set -euo pipefail

PURGE=0
if [ "${1:-}" = "--purge" ]; then PURGE=1; shift; fi

SRC=${DLRM_DATA_SRC:-/data}
# Default to /tmp: on this cluster it is the node's local NVMe root fs (~1.6 TB
# free) and world-writable, whereas the larger /scratch RAID is root-owned.
DEST=${1:-${LOCAL_DATA_MOUNT:-/tmp/$(id -un)/dlrm_data}}
HIST=${2:-${HISTORY_LENGTH:-4086}}          # matches max_seq_len 4096 -> hstu_cache_L4086
JOBS=${STAGE_JOBS:-8}                        # parallel copy streams
# Reclaim our own node-local scratch before declaring a node too full. On a
# shared node /tmp accumulates: caches for a max_seq_len nobody trains any more,
# staging roots from a previous experiment, per-job scratch from finished jobs,
# fetched trace exports.
#
# This only ever runs when the copy does not fit, and then removes only as much
# as the shortfall needs, cheapest-to-lose first (see candidates()). A node with
# room is left exactly as it was found. STAGE_CLEAN=0 disables it entirely — the
# space check still fails hard.
STAGE_CLEAN=${STAGE_CLEAN:-1}

CACHE="processed_5b/hstu_cache_L${HIST}"

# Refuse to treat a shared or source path as scratch: everything below is
# subject to rm -rf, and a mistyped DEST would eat the dataset it copies from.
# Checked before anything is touched (--purge included), on DEST as written and
# as resolved: readlink -m resolves every symlink, "//", "." and ".." (missing
# components allowed; readlink -f must agree where the path exists), so a
# symlink to $SRC, "/apps/x//dlrm_data" or a path through a symlinked parent
# cannot pass as scratch. DEST must not be $SRC, lie inside it or contain it;
# the same is checked by device:inode on every existing ancestor, which also
# catches a bind mount of $SRC. (A string compare let the prune delete source
# files through such a DEST.)
canon() {  # resolved absolute path of $1; rc 1 if it cannot be resolved
  local m f
  m=$(readlink -m -- "$1" 2>/dev/null) && [ -n "$m" ] || return 1
  if [ -e "$1" ]; then
    f=$(readlink -f -- "$1" 2>/dev/null) && [ "$f" = "$m" ] || return 1
  fi
  printf '%s\n' "$m"
}
within() {  # resolved path $1 is $2 or lies below it
  [ "$1" = "$2" ] || [ "$2" = / ] || case "$1" in "$2"/*) return 0 ;; *) return 1 ;; esac
}
ids_up() {  # device:inode of $1 and of every existing ancestor, one per line
  local p=$1
  while :; do
    if [ -e "$p" ]; then stat -L -c '%d:%i' -- "$p" 2>/dev/null || true; fi
    if [ "$p" = / ]; then return 0; fi
    p=$(dirname -- "$p")
  done
}
same_dir_in() {  # rc 0 if the directory $1 is $2 or one of $2's ancestors (by device:inode)
  local id ids
  id=$(stat -L -c '%d:%i' -- "$1" 2>/dev/null) || return 1
  ids=$(ids_up "$2")
  case $'\n'"$ids"$'\n' in *$'\n'"$id"$'\n'*) return 0 ;; esac
  return 1
}
R_SRC=$(canon "$SRC") || { echo "[stage] FATAL: cannot resolve the source path '$SRC'"; exit 1; }
R_DEST=$(canon "$DEST") || { echo "[stage] FATAL: cannot resolve the staging root '$DEST'"; exit 1; }
not_scratch=0
case "$DEST" in
  /|/tmp|/home|/apps|/scratch|"$HOME"|"$SRC"|"$SRC"/*) not_scratch=1 ;;
esac
for p in / /tmp /home /apps /scratch "${HOME:-}"; do
  if [ -n "$p" ] && [ "$R_DEST" = "$(canon "$p" || printf '%s' "$p")" ]; then not_scratch=1; fi
done
if [ "$not_scratch" = 1 ]; then
  echo "[stage] FATAL: refusing to use '$DEST' as a staging root — it is not scratch"
  exit 1
fi
src_alias=
if within "$R_DEST" "$R_SRC" || within "$R_SRC" "$R_DEST"; then
  src_alias="resolved: $R_DEST vs $R_SRC"
elif same_dir_in "$R_SRC" "$R_DEST" || same_dir_in "$R_DEST" "$R_SRC"; then
  src_alias="same directory by device:inode, e.g. a bind mount"
fi
if [ -n "$src_alias" ]; then
  echo "[stage] FATAL: refusing to use '$DEST' as a staging root — it is, lies inside or contains the source $SRC ($src_alias); staging, pruning or --purge there would change the data it copies from"
  exit 1
fi

# df needs a path that exists; DEST may not yet, or may have just been purged.
fs_ref() { local p=$1; while [ ! -d "$p" ] && [ "$p" != / ]; do p=$(dirname "$p"); done; printf '%s' "$p"; }
free_gb_of() { df -B1 --output=avail "$(fs_ref "$1")" | tail -1 | awk '{print int($1/1073741824)}'; }

# Remove a scratch path, but only one we own, and say what it bought. Never
# fails the caller: reclaiming is best-effort by definition.
drop() {
  local p=$1 why=$2 sz=0 r
  [ -e "$p" ] || return 0
  # Never the source, anything inside it or anything that holds it (a reclaim
  # candidate that resolves there, or a source kept under /tmp/$USER).
  if ! r=$(canon "$p"); then echo "[stage]   skip $p — cannot resolve it"; return 0; fi
  if within "$r" "$R_SRC" || within "$R_SRC" "$r"; then
    echo "[stage]   skip $p — it is, lies inside or holds the source $SRC"; return 0
  fi
  if [ ! -O "$p" ]; then echo "[stage]   skip $p — not ours"; return 0; fi
  sz=$(du -sB1 -- "$p" 2>/dev/null) || sz=0; sz=${sz%%[!0123456789]*}   # digits only (see unlink_nonref)
  if rm -rf -- "$p" 2>/dev/null; then
    echo "[stage]   removed $p (~$(( ${sz:-0} / 1073741824 )) GiB) — $why"
  else
    echo "[stage]   could not remove $p"
  fi
  return 0
}

# Everything we consider disposable, as "path|why", CHEAPEST TO LOSE FIRST. The
# order is the whole point: a wrong-max_seq_len cache is useless to every run on
# this cluster, while another staging root costs ~140 s to rebuild, so they are
# not interchangeable when only one of them needs to go.
#
# Nothing outside our own uid appears here, and nothing belonging to a job that
# is still alive.
candidates() {
  local d jid
  # Event caches built for a different max_seq_len. No run can use these: the
  # cache name encodes the length it was built for.
  for d in "$DEST"/processed_5b/hstu_cache_L*; do
    [ -d "$d" ] || continue
    [ "$(basename "$d")" = "hstu_cache_L${HIST}" ] && continue
    printf '%s|%s\n' "$d" "cache for another max_seq_len"
  done
  # Trace exports: the harness copies these to results/ on NFS before the arm
  # ends, so a leftover here is a copy of a copy.
  [ -d /tmp/kf_traces ] && printf '%s|%s\n' /tmp/kf_traces "trace exports already fetched to results/"
  # Per-job scratch of jobs that have exited. A live job still shows in squeue,
  # so the CURRENT job's own scratch can never be selected.
  for d in /tmp/hstu-*; do
    [ -d "$d" ] || continue
    jid=${d##*/hstu-}
    case "$jid" in ''|*[!0-9]*) continue ;; esac
    [ -n "$(squeue -h -j "$jid" -o %i 2>/dev/null)" ] && continue
    printf '%s|%s\n' "$d" "scratch of finished job $jid"
  done
  # Other staging roots of ours. LAST, because re-staging one costs a full copy,
  # and on a shared node it may be someone's next run.
  for d in "/tmp/$(id -un)"/*; do
    [ -d "$d" ] || continue
    case "$DEST" in "$d"|"$d"/*) continue ;; esac
    printf '%s|%s\n' "$d" "stale staging root"
  done
  return 0
}

# Reclaim just enough, and no more. Re-checks free space before every removal
# and stops the moment the copy will fit: deleting scratch we did not need to
# delete costs someone else a re-stage for nothing.
reclaim_to() {
  local want=$1 c free left=0 removed=0
  free=$(free_gb_of "$DEST")
  echo "[stage] short on space (${free} GiB free, need ${want}) — reclaiming our own scratch"
  while IFS= read -r c; do
    [ -n "$c" ] || continue
    free=$(free_gb_of "$DEST")
    if [ "$free" -gt "$want" ]; then left=$((left + 1)); continue; fi
    drop "${c%%|*}" "${c#*|}"
    removed=$((removed + 1))
  done < <(candidates)
  free=$(free_gb_of "$DEST")
  if [ "$left" -gt 0 ]; then
    echo "[stage]   stopped at ${free} GiB free — left $left disposable path(s) alone"
  fi
  [ "$removed" = 0 ] && echo "[stage]   nothing of ours to reclaim"
  return 0
}

if [ "$PURGE" = 1 ]; then
  # Explicit request: take everything, no space test. This is the "it is all
  # scratch" path -- the staged copy is a byte-for-byte reproduction of $SRC.
  echo "[stage] purge: host=$(hostname -s) dest=$DEST"
  drop "$DEST" "purge requested"
  while IFS= read -r c; do
    [ -n "$c" ] || continue
    drop "${c%%|*}" "${c#*|}"
  done < <(candidates)
  echo "[stage] purge done — free $(free_gb_of "$DEST") GiB on $(fs_ref "$DEST")"
  exit 0
fi

echo "[stage] host=$(hostname -s)  src=$SRC  dest=$DEST  cache=$CACHE  jobs=$JOBS"

# Reference-only staging switch (yambda_5b.gin item (17b)). The source of truth
# is the gin binding apply_env_bootstrap.DLRM_HSTU_FP8_STAGE_REF_ONLY in the
# gin file the trainer parses (train_ranker --dataset yambda-5b ->
# dlrm_v4/train/gin/yambda_5b.gin of THIS tree; STAGE_GIN=<file> points
# elsewhere, e.g. when this script runs from a copy outside the tree). The file
# must be readable and hold exactly one binding, `= True` or `= False`, or
# staging is FATAL: the stager never guesses the mode. A non-empty
# DLRM_HSTU_FP8_STAGE_REF_ONLY in the environment wins over the binding (as
# _env_bootstrap._bind_bool does; launch_slurm.sh forwards it into the
# container by prefix): 1 = on, 0 = off, any other value is FATAL.
#   1 (gin default; REQUIRED for an MLPerf submission, user rule 2026-10-12: no
#     non-reference data before RUN_START): stage exactly REF_FILES below,
#     every one must exist at the source (FATAL otherwise), verify those,
#     prune non-reference files from $DEST, then check the END STATE: $DEST
#     must hold exactly REF_FILES as regular files, at any depth, or staging
#     is FATAL (see "End state" below).
#   0 (development only): the c68395d stager exactly: every file of
#     the cache directory and of shared_metadata plus the processed_5b split
#     files present at the source, verified by size; nothing pruned; the end
#     state is reported, never FATAL.
# The binding is counted in every form gin resolves to the same parameter
# (plain, scope/..., module-qualified ..._env_bootstrap.apply_env_bootstrap...);
# any other non-comment mention of the knob (e.g. a binding split before its
# "=") and any include/import statement (an included file could rebind it;
# this parse cannot follow it) are FATAL in either mode: they decide the mode.
# Lines are split as gin reads the file (Python universal newlines: CR LF and
# a lone CR are line breaks too), so a carriage return cannot hide an include
# or a binding; the regexes are strings with no backslash in a bracket
# expression, which gawk, mawk and busybox awk all read alike.
STAGE_GIN=${STAGE_GIN:-$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)/primus/backends/recommendation/dlrm_v4/src/generative_recommenders/dlrm_v4/train/gin/yambda_5b.gin}
if [ ! -r "$STAGE_GIN" ]; then
  echo "[stage] FATAL: gin file $STAGE_GIN not readable (needed for DLRM_HSTU_FP8_STAGE_REF_ONLY)"
  exit 1
fi
_ref_gin=$(awk '
  function chk(l,   x) {
    ln++; x = l " "
    if (x ~ inc_re) { if (!inc) inc = ln; return }
    if (l ~ bind_re) {
      v = l; sub(/^[^=]*=[[:space:]]*/, "", v); sub(/[[:space:]]+$/, "", v); n++; return
    }
    sub(/#.*/, "", x); if ((x " ") ~ ment_re) o++
  }
  BEGIN {
    inc_re = "^[[:space:]]*(include|import|from)[^A-Za-z0-9_./]"
    bind_re = "^[[:space:]]*([A-Za-z0-9_./]*[./])?apply_env_bootstrap[.]DLRM_HSTU_FP8_STAGE_REF_ONLY[[:space:]]*="
    ment_re = "DLRM_HSTU_FP8_STAGE_REF_ONLY[^A-Za-z0-9_]"
  }
  { y = $0; sub(/\r$/, "", y); k = split(y, part, "\r"); if (k == 0) ln++
    for (i = 1; i <= k; i++) chk(part[i]) }
  END {
    if (inc) print "<include/import statement at line " inc ">"
    else if (o) print "<" n + 0 " bindings + " o " other non-comment mention(s)>"
    else print (n == 1 ? v : "<" n + 0 " bindings>")
  }' "$STAGE_GIN") \
  || { echo "[stage] FATAL: could not read $STAGE_GIN"; exit 1; }
case "$_ref_gin" in
  True) REF_ONLY=1 ;;
  False) REF_ONLY=0 ;;
  "<include/import"*)
     echo "[stage] FATAL: $STAGE_GIN: gin include/import statement ($_ref_gin): the stager reads only this file, and an included file could rebind apply_env_bootstrap.DLRM_HSTU_FP8_STAGE_REF_ONLY"
     exit 1 ;;
  *) echo "[stage] FATAL: $STAGE_GIN: apply_env_bootstrap.DLRM_HSTU_FP8_STAGE_REF_ONLY must be bound exactly once to True or False (found: $_ref_gin)"
     exit 1 ;;
esac
_ref_src="gin $STAGE_GIN = $_ref_gin"
case "${DLRM_HSTU_FP8_STAGE_REF_ONLY:-}" in
  "") ;;
  1|0) REF_ONLY=$DLRM_HSTU_FP8_STAGE_REF_ONLY
       _ref_src="env DLRM_HSTU_FP8_STAGE_REF_ONLY=$REF_ONLY over $_ref_src" ;;
  *) echo "[stage] FATAL: DLRM_HSTU_FP8_STAGE_REF_ONLY=$DLRM_HSTU_FP8_STAGE_REF_ONLY is not 1 or 0"
     exit 1 ;;
esac
if [ "$REF_ONLY" = 1 ]; then
  echo "[stage] reference-only staging ON ($_ref_src)"
else
  echo "[stage] reference-only staging OFF ($_ref_src): c68395d full-cache staging, NOT for an MLPerf submission"
fi

# REF_FILES (positions_L<N>.npy, ts_*_L<N>_W86400.npy) are the names the trainer
# opens only when its EFFECTIVE min_history equals history_length: any other
# floor is a min_history-tagged file (yambda.py _positions_filename(),
# offline_ts_indices.py), which the b5e8e2a manifest does not hold and the
# trainer would refuse or build later. Effective = $MIN_HISTORY when non-empty
# (utils.env_int), else the gin default mh/env_int.default = 4086
# (yambda_5b.gin; stage_ref_only_hardening_test.py pins the two together).
# Checked before anything in $DEST is touched; FATAL with REF_ONLY=1 only.
# env_int() reads the value with int(), so it is compared as a number: the
# ASCII spellings int() accepts (blanks around it, a sign, leading zeros, a
# single "_" between digits; at most 64 digit characters) are read here the
# same way. Anything else (int() raises, or reads non-ASCII digits) is not
# parsed and so never equals history_length: FATAL with REF_ONLY=1.
MH_EFF=${MIN_HISTORY:-4086}
py_int() {  # canonical decimal of $1 as int($1) reads it; rc 1 if not parsed
  local ws=$' \t\n\v\f\r' re d
  re="^[$ws]*([+-]?)([0123456789]+(_[0123456789]+)*)[$ws]*\$"
  [[ $1 =~ $re ]] && [ "${#BASH_REMATCH[2]}" -le 64 ] || return 1
  d=${BASH_REMATCH[2]//_/}; d=${d#"${d%%[!0]*}"}; d=${d:-0}
  if [ "${BASH_REMATCH[1]}" = - ] && [ "$d" != 0 ]; then d=-$d; fi
  printf '%s\n' "$d"
}
mh_ok=0
if MH_NUM=$(py_int "$MH_EFF"); then
  mh_ok=1; mh_src="MIN_HISTORY=${MIN_HISTORY:-<unset>, gin default}"
  if [ "$MH_NUM" != "$MH_EFF" ]; then mh_src="MIN_HISTORY=$(printf '%q' "$MIN_HISTORY"), read as $MH_NUM"; fi
  mh_why="the trainer would open positions_L${HIST}_m${MH_NUM}.npy, not a reference file"
else
  MH_NUM=$(printf '%q' "$MH_EFF"); mh_src="MIN_HISTORY=$MH_NUM"
  mh_why="not an ASCII decimal integer, so not shown to equal it (the trainer reads it with int())"
fi
if [ "$mh_ok" != 1 ] || [ "$MH_NUM" != "$HIST" ]; then
  if [ "$REF_ONLY" = 1 ]; then
    echo "[stage] FATAL: effective min_history $MH_NUM ($mh_src) != history_length $HIST: $mh_why; reference-only staging needs MIN_HISTORY = HISTORY_LENGTH (min_history runs: DLRM_HSTU_FP8_STAGE_REF_ONLY=0, development only)"
    exit 1
  fi
  echo "[stage] note: effective min_history $MH_NUM != history_length $HIST (reference-only staging OFF: the whole cache directory is staged)"
elif [ "$REF_ONLY" = 1 ]; then
  echo "[stage] effective min_history $MH_NUM = history_length $HIST ($mh_src): reference file names"
fi

[ -d "$SRC/$CACHE" ] || { echo "[stage] FATAL: $SRC/$CACHE not found"; exit 1; }

# The data directories must be directories of the staging root itself, not
# symlinks (or below one): the copy, the prune and the reclaim would act on
# the link target instead (e.g. processed_5b -> $SRC/processed_5b). Checked in
# either mode, before anything in $DEST is touched.
for d in processed_5b "$CACHE" shared_metadata; do
  r=$(canon "$DEST/$d") || r="(unresolvable)"
  if [ "$r" != "$R_DEST/$d" ] || [ -L "$DEST/$d" ]; then
    echo "[stage] FATAL: $DEST/$d is a symlink or lies below one (resolves to $r): a symlinked data directory would make the copy, prune and reclaim act on its target; remove the link or stage into a fresh STAGE_DEST"
    exit 1
  fi
done

mkdir -p "$DEST/$CACHE" "$DEST/processed_5b" "$DEST/shared_metadata"

# The reference data set, relative to the data root. Keep in sync with
# generative_recommenders/dlrm_v4/train/dl_prestart.py reference_data_files()
# (dl_prestart_reference_files_test.py checks the two lists agree). The cache
# entries are the b5e8e2a sha256sums_yambda_5b_cache.txt manifest; anchor_ts is
# in that manifest and is read lazily after RUN_START by the full-scan window
# lookup. ts_indices/ts_offsets are the mlcommons/training PR 907 offline
# time-window index (user ruling 2026-10-12: PR 907 index allowed, DL_WINDOW_INDEX=1);
# they are read only after RUN_START.
REF_FILES=(
  "$CACHE/_READY"
  "$CACHE/store_meta.json"
  "$CACHE/flat_event_types.npy"
  "$CACHE/flat_is_like.npy"
  "$CACHE/flat_is_listen_plus.npy"
  "$CACHE/flat_is_organic.npy"
  "$CACHE/flat_is_skip.npy"
  "$CACHE/flat_item_ids.npy"
  "$CACHE/flat_played_ratio.npy"
  "$CACHE/flat_timestamps.npy"
  "$CACHE/flat_uid.npy"
  "$CACHE/unique_uids.npy"
  "$CACHE/user_start.npy"
  "$CACHE/user_end.npy"
  "$CACHE/positions_L${HIST}.npy"
  "$CACHE/anchor_ts_L${HIST}.npy"
  "$CACHE/ts_indices_L${HIST}_W86400.npy"
  "$CACHE/ts_offsets_L${HIST}_W86400.npy"
  "processed_5b/item_popularity.npy"
  "shared_metadata/artist_item_mapping.parquet"
  "shared_metadata/album_item_mapping.parquet"
)

# Build the copy list: exactly REF_FILES, every one required at the source.
# Anything already present at the right size is dropped from the list.
list=$(mktemp)
if [ "$REF_ONLY" = 1 ]; then
  missing_src=0
  for rel in "${REF_FILES[@]}"; do
    if [ -f "$SRC/$rel" ]; then
      printf '%s\n' "$rel"
    else
      echo "[stage] FATAL: reference file $SRC/$rel not found" >&2
      missing_src=1
    fi
  done > "$list"
  [ "$missing_src" = 0 ] || { rm -f "$list"; exit 1; }
else
  # DLRM_HSTU_FP8_STAGE_REF_ONLY=0: the c68395d list, verbatim: cache
  # dir + metadata + the small per-split files.
  {
    find "$SRC/$CACHE" -maxdepth 1 -type f -printf '%P\0' | while IFS= read -r -d '' f; do
      printf '%s\n' "$CACHE/$f"
    done
    find "$SRC/shared_metadata" -maxdepth 1 -type f -printf '%P\0' | while IFS= read -r -d '' f; do
      printf '%s\n' "shared_metadata/$f"
    done
    for f in item_popularity.npy split_meta.json session_index.parquet test_events.parquet; do
      [ -f "$SRC/processed_5b/$f" ] && printf '%s\n' "processed_5b/$f"
    done
  } > "$list"
fi

todo=$(mktemp)
while IFS= read -r rel; do
  s=$(stat -c %s "$SRC/$rel" 2>/dev/null || echo -1)
  d=$(stat -c %s "$DEST/$rel" 2>/dev/null || echo -2)
  [ "$s" = "$d" ] || printf '%s\n' "$rel" >> "$todo"
done < "$list"

n_all=$(wc -l < "$list"); n_todo=$(wc -l < "$todo" 2>/dev/null || echo 0)
echo "[stage] $n_all files tracked, $n_todo to copy ($((n_all - n_todo)) already staged)"

# The space requirement is exactly the bytes still MISSING, which is why the
# copy list is built first. Sizing it off the dataset total instead would fail a
# node that already holds the full copy; sizing it off `du $DEST` would count
# junk in $DEST as progress and pass a node that then runs out mid-copy.
missing_gb=0
if [ "$n_todo" -gt 0 ]; then
  missing_gb=$(while IFS= read -r rel; do stat -c %s "$SRC/$rel" 2>/dev/null || echo 0
               done < "$todo" | awk '{s+=$1} END {print int(s/1073741824)}')
fi
want_gb=$(( missing_gb + 20 ))            # headroom for the in-flight .part file
free_gb=$(free_gb_of "$DEST")
FS=$(df --output=target "$(fs_ref "$DEST")" | tail -1)
echo "[stage] ${missing_gb} GiB missing -> want ${want_gb} GiB free; have ${free_gb} GiB on $FS"

# Only clean when the copy genuinely does not fit. A node with room is left
# exactly as it was found.
if [ "$free_gb" -le "$want_gb" ] && [ "$STAGE_CLEAN" = "1" ]; then
  reclaim_to "$want_gb"
  free_gb=$(free_gb_of "$DEST")
  echo "[stage] after reclaim: free ${free_gb} GiB (want ${want_gb})"
fi

if [ "$free_gb" -le "$want_gb" ]; then
  echo "[stage] FATAL: not enough node-local space on $FS ($(hostname -s))"
  echo "[stage]   want ${want_gb} GiB free for the ${missing_gb} GiB still missing, have ${free_gb} GiB"
  [ "$STAGE_CLEAN" = "1" ] || echo "[stage]   (STAGE_CLEAN=0, so no scratch was reclaimed)"
  echo "[stage]   largest things on this filesystem:"
  du -shx /tmp/* /var/lib/docker /var/log 2>/dev/null | sort -rh | head -8 | sed 's/^/[stage]     /'
  echo "[stage]   Fatal on purpose. Falling back to NFS here would drop the window-boundary"
  echo "[stage]   behaviour staging exists to provide, and would do it invisibly: the run"
  echo "[stage]   would look fine and measure differently."
  echo "[stage]   Free the node ('$0 --purge' clears our own scratch), use another node, or"
  echo "[stage]   ask for NFS explicitly with STAGE_DATA=0."
  rm -f "$list" "$todo"
  exit 1
fi

if [ "$n_todo" -gt 0 ]; then
  t0=$(date +%s)
  # Copy with O_DIRECT on both ends where possible. These hosts have ~3 TB of RAM,
  # so a plain cp would leave the whole staged copy sitting in page cache and any
  # A/B against NFS would then be measuring RAM, not NVMe. Falls back to cp if the
  # filesystem refuses O_DIRECT.
  export SRC DEST
  # shellcheck disable=SC2016  # expanded by the inner bash, per file
  xargs -a "$todo" -P "$JOBS" -I{} bash -c '
    rel={}; s="$SRC/$rel"; d="$DEST/$rel"
    mkdir -p "$(dirname "$d")"
    rm -f -- "$d.part"   # never write through a pre-existing .part (symlink or hard link to a source file)
    if ! dd if="$s" of="$d.part" bs=8M iflag=direct oflag=direct status=none 2>/dev/null; then
      cp --sparse=never "$s" "$d.part" || exit 1
    fi
    mv "$d.part" "$d"
  '
  t1=$(date +%s)
  echo "[stage] copied in $((t1 - t0))s"
fi
rm -f "$list" "$todo"

# Verify: every reference file present at the source size.
bad=0
if [ "$REF_ONLY" = 1 ]; then
  for rel in "${REF_FILES[@]}"; do
    s=$(stat -c %s "$SRC/$rel"); d=$(stat -c %s "$DEST/$rel" 2>/dev/null || echo -1)
    [ "$s" = "$d" ] || { echo "[stage] MISMATCH $rel (src=$s dest=$d)"; bad=1; }
  done
else
  # DLRM_HSTU_FP8_STAGE_REF_ONLY=0: the c68395d check, verbatim: every
  # tracked file present at the source size.
  while IFS= read -r -d '' src_file; do
    sub=${src_file#"$SRC"/}
    s=$(stat -c %s "$src_file"); d=$(stat -c %s "$DEST/$sub" 2>/dev/null || echo -1)
    [ "$s" = "$d" ] || { echo "[stage] MISMATCH $sub (src=$s dest=$d)"; bad=1; }
  done < <(find "$SRC/$CACHE" "$SRC/shared_metadata" -maxdepth 1 -type f -print0)
fi
[ "$bad" = 0 ] || { echo "[stage] FAILED verification"; exit 1; }

# Optional, default OFF: check the staged PR 907 window index (ts_indices_*/
# ts_offsets_*) against the sidecar that `ts_index_tool.py prebuild/verify`
# wrote next to it on NFS. The NFS sidecar is read (--sidecar-dir), not the
# staged copy, so a stale same-size copy cannot vouch for itself. Sizes, npy
# headers, offsets content and source fingerprints; ~10 MB read, stdlib python
# only. MIN_HISTORY selects the same index tag as training. Runs here, before
# any trainer starts, so it adds nothing to training time. Extra flags via
# STAGE_CHECK_TS_INDEX_ARGS (e.g. "--hash" to also hash the ~18 GB index).
if [ "${STAGE_CHECK_TS_INDEX:-0}" = "1" ]; then
  # shellcheck disable=SC2086
  python3 "$(dirname "$0")/../../../../tools/recommendation/dlrm_v4/ts_index_tool.py" check-staged "$DEST/$CACHE" \
      --history-length "$HIST" ${MIN_HISTORY:+--min-history "$MIN_HISTORY"} \
      --sidecar-dir "$SRC/$CACHE" ${STAGE_CHECK_TS_INDEX_ARGS:-} 2>&1 | sed 's/^/[stage] /' \
    || { echo "[stage] FAILED window-index check (STAGE_CHECK_TS_INDEX=1)"; exit 1; }
fi


# Non-reference files left in $DEST by an older stager (window_index_*,
# *_m0/_m1, embeddings.parquet, ...). The trainer never opens them with the
# default config, but the submission data directory should hold only the
# reference set, so they are pruned by DEFAULT (our own files only; root-owned
# leftovers are reported, not removed). Pruning is skipped while a running
# container on this node bind-mounts $DEST, since that job may still use them;
# STAGE_PRUNE_NONREF=0 disables pruning entirely. A *.part file is never
# pruned: it may be a concurrent stager's in-flight copy (iter47 L3). Whatever
# is left (including anything below depth 1 or outside the three directories)
# makes the end-state check below FATAL.
# Names are NUL-separated throughout (a name may hold a newline), and a pruned
# entry is only ever unlinked (rm -f, never rm -r): a regular file of ours at
# depth 1 of one of the three data directories, re-checked right before.
# DLRM_HSTU_FP8_STAGE_REF_ONLY=0 skips this (c68395d pruned nothing).
shown() {  # $1 for a log line: as is, or shell-quoted if it holds a control character
  case "$1" in *[[:cntrl:]]*) printf '%q' "$1" ;; *) printf '%s' "$1" ;; esac
}
unlink_nonref() {
  local p=$1 sz=0
  if [ -L "$p" ] || [ ! -f "$p" ]; then echo "[stage]   skip $(shown "$p") — not a regular file"; return 0; fi
  if [ ! -O "$p" ]; then echo "[stage]   skip $(shown "$p") — not ours"; return 0; fi
  # Leading digits only: du echoes the name, and a newline in it must not
  # reach the arithmetic below.
  sz=$(du -sB1 -- "$p" 2>/dev/null) || sz=0; sz=${sz%%[!0123456789]*}
  if rm -f -- "$p" 2>/dev/null; then
    echo "[stage]   removed $(shown "$p") (~$(( ${sz:-0} / 1073741824 )) GiB) — non-reference file"
  else
    echo "[stage]   could not remove $(shown "$p")"
  fi
  return 0
}
if [ "$REF_ONLY" = 1 ]; then
  declare -A is_ref=()
  for rel in "${REF_FILES[@]}"; do is_ref[$rel]=1; done
  extra=()
  # shellcheck disable=SC2030  # the inner loop's f is local to it
  while IFS= read -r -d '' rel; do
    [ -n "${is_ref[$rel]:-}" ] || extra+=("$rel")
  done < <(for d in "$CACHE" processed_5b shared_metadata; do
             [ -d "$DEST/$d" ] || continue
             find "$DEST/$d" -mindepth 1 -maxdepth 1 -type f ! -name '*.part' -printf '%P\0' |
               while IFS= read -r -d '' f; do printf '%s/%s\0' "$d" "$f"; done
           done | sort -z)
  dest_in_use() {
    # A running container that bind-mounts $DEST (or a parent of it) may be a live
    # job reading this staging root.
    command -v docker >/dev/null 2>&1 || return 1
    local c src real
    real=$(readlink -f "$DEST")
    for c in $(docker ps -q 2>/dev/null); do
      for src in $(docker inspect -f '{{range .Mounts}}{{.Source}} {{end}}' "$c" 2>/dev/null); do
        src=$(readlink -f "$src")
        case "$real/" in "$src/"*) return 0 ;; esac   # $DEST inside the mount
        case "$src/" in "$real/"*) return 0 ;; esac   # mount inside $DEST
      done
    done
    return 1
  }
  if [ "${#extra[@]}" -gt 0 ]; then
    n_extra=${#extra[@]}
    if [ "${STAGE_PRUNE_NONREF:-1}" != "1" ]; then
      echo "[stage] note: $n_extra non-reference file(s) left in $DEST (STAGE_PRUNE_NONREF=0; never staged, read or snapshotted by this tree):"
      for rel in "${extra[@]}"; do echo "[stage]   $(shown "$rel")"; done
    elif dest_in_use; then
      echo "[stage] note: $n_extra non-reference file(s) left in $DEST: a running container mounts it, so pruning is skipped this time:"
      for rel in "${extra[@]}"; do echo "[stage]   $(shown "$rel")"; done
    else
      for rel in "${extra[@]}"; do unlink_nonref "$DEST/$rel"; done
    fi
  fi
fi

# End state (17b; iter49 REVIEW M2, VERIFY V1): every non-directory entry now
# under $DEST at any depth (regular files, symlinks, *.part, files below the
# three data dirs or outside them), compared with REF_FILES as regular files.
# The count is measured, never assumed, and any find error (an unreadable
# directory, say) means NOT verified. This line, not the trainer's env echo,
# is the per-node record of a reference-only data dir.
#   REF_ONLY=1: anything but exactly REF_FILES is FATAL. Leftovers the prune
#     above cannot or may not remove (STAGE_PRUNE_NONREF=0, a running container
#     mounting $DEST, files not ours or not removable, deeper or outside files,
#     symlinks, *.part) therefore stop the launch instead of passing silently.
#   REF_ONLY=0: one report line, never FATAL (c68395d behaviour).
# Entries are NUL-separated and counted one by one (a newline in a name can
# neither add nor hide an entry), and each reference path must also test as a
# regular file that is not a symlink.
es=$(mktemp)
es_rc=0
find -H "$DEST" -mindepth 1 ! -type d -printf '%y %P\0' > "$es" 2> "$es.err" || es_rc=$?
LC_ALL=C sort -z "$es" > "$es.sorted" || es_rc=1
declare -A es_want=()
for rel in "${REF_FILES[@]}"; do es_want["f $rel"]=1; done
n_found=0; es_left=(); es_miss=()
while IFS= read -r -d '' e; do
  n_found=$((n_found + 1))
  if [ "${es_want[$e]:-}" = 1 ]; then es_want[$e]=2; else es_left+=("$e"); fi
done < "$es.sorted"
for rel in "${REF_FILES[@]}"; do
  # shellcheck disable=SC2031  # "f $rel" is a key string, not $f
  if [ "${es_want["f $rel"]}" != 2 ] || [ -L "$DEST/$rel" ] || [ ! -f "$DEST/$rel" ]; then es_miss+=("$rel"); fi
done
n_left=${#es_left[@]}; n_miss=${#es_miss[@]}; n_err=$(grep -c '' "$es.err" || true)
if [ "$es_rc" -eq 0 ] && [ ! -s "$es.err" ] && [ "$n_left" -eq 0 ] && [ "$n_miss" -eq 0 ] \
   && [ "$n_found" -eq "${#REF_FILES[@]}" ]; then
  echo "[stage] data dir is reference-only: $n_found regular files under $DEST, exactly REF_FILES ($(hostname -s))"
elif [ "$REF_ONLY" = 1 ]; then
  echo "[stage] data dir NOT reference-only ($(hostname -s)): $n_found entries under $DEST, $n_left not in REF_FILES, $n_miss of ${#REF_FILES[@]} reference files missing or not regular, find rc=$es_rc with $n_err error line(s)"
  if [ "$n_left" -gt 0 ]; then
    for e in "${es_left[@]:0:20}"; do
      case "$e" in "f "*) echo "[stage]   extra: $(shown "${e#f }")" ;; *) echo "[stage]   extra: $(shown "${e#* }") (find type ${e%% *})" ;; esac
    done
  fi
  [ "$n_left" -le 20 ] || echo "[stage]   extra: ... and $((n_left - 20)) more"
  if [ "$n_miss" -gt 0 ]; then
    for rel in "${es_miss[@]}"; do echo "[stage]   missing or not regular: $rel"; done
  fi
  head -n 5 "$es.err" | sed 's/^/[stage]   find: /'
  echo "[stage] FATAL: DLRM_HSTU_FP8_STAGE_REF_ONLY=1 needs exactly the reference set in $DEST, nothing else at any depth"
  echo "[stage]   The prune removes only our own regular files at depth 1 of $CACHE, processed_5b and shared_metadata (not with STAGE_PRUNE_NONREF=0, not while a running container mounts $DEST, never *.part)."
  echo "[stage]   Remove the rest by hand (root-owned files as root, e.g. from a root container that mounts $DEST; *.part only once no stager runs on this node; make unreadable directories readable or remove them), or stage into a fresh STAGE_DEST; '$0 --purge $DEST' drops our own copy."
  echo "[stage]   Development only: DLRM_HSTU_FP8_STAGE_REF_ONLY=0 stages the c68395d set and only reports this."
  rm -f "$es" "$es.err" "$es.sorted"
  exit 1
else
  echo "[stage] end state (report only, reference-only staging OFF): $n_found entries under $DEST, $n_left not in REF_FILES, $n_miss of ${#REF_FILES[@]} reference files missing or not regular, find rc=$es_rc with $n_err error line(s) ($(hostname -s))"
fi
rm -f "$es" "$es.err" "$es.sorted"

echo "[stage] OK — $(du -sh "$DEST" | cut -f1) staged at $DEST"
echo "[stage] now run with LOCAL_DATA_MOUNT=$DEST DLRM_DATA_PATH=$DEST FORCE_PROVISION=1"
