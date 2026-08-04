#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# primus-cli --patch script: overlay just the MegaMoE python (flydsl/mega, the
# fused_mega_moe ops and kernels) from a Primus-Turbo working tree onto the
# primus_turbo the image already installed, keeping the image's compiled
# artifacts and the rest of the installed package untouched.
#
# Why: iterating on the flydsl MegaMoE kernels means editing Python only, but
# `pip install` on the turbo source rebuilds the whole CK/HIP kernel library --
# ~20 minutes, and the resulting .so is bound to that container's torch, so it
# has to be redone every time the container is recreated. As long as the image's
# .so already covers the native ops the new Python calls, copying the .py files
# over takes seconds.
#
# Scope: not the whole package. The patch walks primus_turbo imports out from
# the MegaMoE entry points, keeps only what changed since the commit the
# installed .so was built from, and copies that. Unrelated churn in the working
# tree (attention, triton, jax, quantization, ...) stays on the image's version,
# so a broken experiment elsewhere in the tree cannot leak into a MegaMoE run.
#
# Opt in with PRIMUS_MEGAMOE_SRC=<turbo source root> (the PRIMUS_ prefix
# auto-forwards into the container; see primus-cli-container.sh env
# passthrough). Unset means skip, so this is a no-op for everyone else.
#
# The patch also warns when csrc/ diverged since that commit -- that is the
# signal that python-only is no longer enough and a real rebuild is due.
###############################################################################
set -euo pipefail

SRC_ROOT="${PRIMUS_MEGAMOE_SRC:-}"
if [[ -z "$SRC_ROOT" ]]; then
    echo "[overlay_megamoe] PRIMUS_MEGAMOE_SRC not set; nothing to overlay"
    exit 2 # 2 = skip (not an error), per runner/helpers/execute_patches.sh
fi
if [[ ! -d "$SRC_ROOT/primus_turbo" ]]; then
    echo "[overlay_megamoe] no primus_turbo package under $SRC_ROOT" >&2
    exit 1
fi

DST="$(python -c 'import primus_turbo, os; print(os.path.dirname(primus_turbo.__file__))' 2>/dev/null || true)"
if [[ -z "$DST" || ! -d "$DST" ]]; then
    echo "[overlay_megamoe] primus_turbo is not installed; nothing to overlay onto" >&2
    exit 1
fi
if [[ "$DST" -ef "$SRC_ROOT/primus_turbo" ]]; then
    echo "[overlay_megamoe] already importing $SRC_ROOT directly -- skipping"
    exit 2
fi

BUILT_FROM="$(python -c 'import primus_turbo._build_info as b; print(b.__git_commit__)' 2>/dev/null || true)"
if [[ -z "$BUILT_FROM" ]]; then
    echo "[overlay_megamoe] installed primus_turbo has no _build_info; cannot tell what the .so covers" >&2
    exit 1
fi
SRC_HEAD="$(git -c safe.directory='*' -C "$SRC_ROOT" rev-parse --short HEAD)"
echo "[overlay_megamoe] installed .so built from ${BUILT_FROM:0:8}, MegaMoE tree at $SRC_HEAD"

NATIVE_DIFF="$(git -c safe.directory='*' -C "$SRC_ROOT" diff --name-only "$BUILT_FROM"..HEAD -- csrc 2>/dev/null || true)"
if [[ -n "$NATIVE_DIFF" ]]; then
    echo "[overlay_megamoe] WARNING: csrc/ changed since the installed build:" >&2
    echo "$NATIVE_DIFF" | sed 's/^/[overlay_megamoe]   /' >&2
    echo "[overlay_megamoe] WARNING: those changes are NOT in this run; rebuild turbo if you need them" >&2
fi

# The file list: MegaMoE's own modules plus whatever else in primus_turbo they
# import (transitively), narrowed to what actually changed since BUILT_FROM.
# Deriving it instead of hardcoding keeps it right as the MegaMoE code moves.
FILE_LIST=$(
    SRC_ROOT="$SRC_ROOT" BUILT_FROM="$BUILT_FROM" python - <<'PY'
import collections, os, pathlib, re, subprocess

root = pathlib.Path(os.environ["SRC_ROOT"])
base = os.environ["BUILT_FROM"]

changed = set(
    subprocess.run(
        ["git", "-c", "safe.directory=*", "-C", str(root),
         "diff", "--name-only", f"{base}..HEAD", "--", "primus_turbo"],
        capture_output=True, text=True, check=True,
    ).stdout.split()
)

seeds = [
    str(p.relative_to(root))
    for pattern in (
        "primus_turbo/flydsl/mega/**/*.py",
        "primus_turbo/pytorch/kernels/fused_mega_moe/**/*.py",
        "primus_turbo/pytorch/ops/moe/fused_mega_moe*.py",
        "primus_turbo/pytorch/modules/moe/mega_moe*.py",
    )
    for p in root.glob(pattern)
]

imported = re.compile(
    r"^\s*(?:from\s+(primus_turbo[\w\.]*)\s+import|import\s+(primus_turbo[\w\.]*))", re.M
)
seen, queue = set(seeds), collections.deque(seeds)
while queue:
    text = (root / queue.popleft()).read_text()
    for match in imported.finditer(text):
        module = (match.group(1) or match.group(2)).replace(".", "/")
        for candidate in (module + ".py", module + "/__init__.py"):
            if (root / candidate).exists() and candidate not in seen:
                seen.add(candidate)
                queue.append(candidate)

print("\n".join(sorted(seen & changed)))
PY
)

if [[ -z "$FILE_LIST" ]]; then
    echo "[overlay_megamoe] MegaMoE python is unchanged since ${BUILT_FROM:0:8}; nothing to overlay"
    exit 2
fi

echo "$FILE_LIST" | sed 's|^primus_turbo/|[overlay_megamoe]   |'
(cd "$SRC_ROOT" && tar -cf - -T <(echo "$FILE_LIST")) | (cd "$(dirname "$DST")" && tar -xf -)

find "$DST" -name __pycache__ -type d -prune -exec rm -rf {} +

# Import from a neutral cwd: inside the source root the tree itself would shadow
# the installed package and the check would prove nothing.
(cd /tmp && python -c 'from primus_turbo.pytorch.ops.moe.fused_mega_moe import fused_mega_moe_stage1') || {
    echo "[overlay_megamoe] overlaid tree fails to import" >&2
    exit 1
}
echo "[overlay_megamoe] overlaid $(echo "$FILE_LIST" | wc -l) MegaMoE python files onto $DST"
exit 0
