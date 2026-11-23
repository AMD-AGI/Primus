#!/usr/bin/env bash
set -uo pipefail

SRC=${DATADIR:?set DATADIR to the shared dataset}
DST=${FLUX_LOCAL_DATADIR:-/data/mlperf_flux1}
STREAMS=${FLUX_STAGE_STREAMS:-12}

# Only the three directories the trainer reads (DATASET_PATH, EVAL_DATASET_PATH,
# EMPTY_ENCODINGS_PATH). Inductor caches and other users' scratch stay on NFS.
SUBDIRS=${FLUX_STAGE_SUBDIRS:-"cc12m_preprocessed coco_preprocessed empty_encodings"}
read -r -a _subdirs <<< "${SUBDIRS}"

echo "[stage] $(hostname -s): ${SRC} -> ${DST} (${STREAMS} streams)"
mkdir -p "${DST}" || { echo "[stage] cannot create ${DST}" >&2; exit 1; }

python3 - "${SRC}" "${DST}" "${STREAMS}" "${_subdirs[@]}" <<'PY'
import os, shutil, sys, time
from concurrent.futures import ThreadPoolExecutor

src, dst, streams, subdirs = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:]
jobs, unreadable = [], []
for sub in subdirs:
    s = os.path.join(src, sub)
    if not os.path.isdir(s):
        sys.exit(f"[stage] missing source directory {s}")
    os.makedirs(os.path.join(dst, sub), exist_ok=True)
    for name in sorted(os.listdir(s)):
        # Other users' private logs live alongside the shards; a dotfile we
        # cannot read must not abort the copy.
        if name.startswith("."):
            continue
        sp = os.path.join(s, name)
        if not os.path.isfile(sp):
            continue
        if not os.access(sp, os.R_OK):
            unreadable.append(sp)
            continue
        dp = os.path.join(dst, sub, name)
        if os.path.exists(dp) and os.path.getsize(dp) == os.path.getsize(sp):
            continue
        jobs.append((sp, dp, os.path.getsize(sp)))

total = sum(j[2] for j in jobs)
print(f"[stage] {len(jobs)} files to copy, {total/1e12:.3f} TB; unreadable skipped: {len(unreadable)}",
      flush=True)

copied, failures, started, marked = [0], [], time.time(), [0]

def copy_one(job):
    sp, dp, size = job
    try:
        # Copy to .part then rename so an interrupted stage never leaves a
        # short file that the size check would later accept as complete.
        shutil.copyfile(sp, dp + ".part")
        os.replace(dp + ".part", dp)
    except Exception as exc:
        failures.append((sp, repr(exc)))
        return
    copied[0] += size
    if copied[0] - marked[0] > (100 << 30):
        marked[0] = copied[0]
        rate = copied[0] / (time.time() - started)
        print(f"[stage]   {copied[0]/1e12:.3f}/{total/1e12:.3f} TB  {rate/1e6:.0f} MB/s  "
              f"eta {(total-copied[0])/max(rate,1)/60:.0f} min", flush=True)

with ThreadPoolExecutor(streams) as pool:
    list(pool.map(copy_one, jobs))

elapsed = time.time() - started
print(f"[stage] copied {copied[0]/1e12:.3f} TB in {elapsed/60:.1f} min "
      f"= {copied[0]/1e6/max(elapsed,1):.0f} MB/s; failures={len(failures)}", flush=True)
for path, err in failures[:10]:
    print(f"[stage]   FAILED {path}: {err}", flush=True)
if failures:
    sys.exit(1)

# Sentinel records the source so run_with_docker_slurm.sh cannot silently feed a
# copy staged from some other dataset.
with open(os.path.join(dst, ".stage_complete"), "w") as fh:
    fh.write(f"source={src}\n")
    for sub in subdirs:
        fh.write(f"{sub}={len(os.listdir(os.path.join(dst, sub)))}\n")
PY
_rc=$?

if [[ ${_rc} -ne 0 ]]; then
    echo "[stage] $(hostname -s): FAILED - local copy left unmarked, training will read ${SRC}" >&2
    exit ${_rc}
fi

echo "[stage] $(hostname -s): complete"
cat "${DST}/.stage_complete"
