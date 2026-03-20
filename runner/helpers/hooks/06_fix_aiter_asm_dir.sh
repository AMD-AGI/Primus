#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# System hook: fix amd-aiter AITER_ASM_DIR missing gfx prefix for codegen.py.
#
# Problem:
#   amd-aiter has two C++ kernel-loading paths that expect different layouts:
#
#   1. aiter_hip_common.h (load_asm_kernel / AiterAsmKernel):
#        path = AITER_ASM_DIR + "/" + arch_name + "/" + hsaco
#      → Works fine with AITER_ASM_DIR = ".../hsa/"  (adds gfx942/ itself)
#
#   2. codegen.py (fmha_fwd_v3_kernel):
#        path = AITER_ASM_DIR + "fmha_v3_fwd/MI300/" + hsaco
#      → Fails because AITER_ASM_DIR = ".../hsa/" has no gfx942/ prefix.
#        The real files live under .../hsa/gfx942/fmha_v3_fwd/MI300/.
#
#   Changing AITER_ASM_DIR fixes path #2 but breaks path #1 (double gfx942).
#
# Fix:
#   Create symlinks inside .../hsa/ that point into .../hsa/{gfx}/ so that
#   BOTH paths resolve correctly without touching AITER_ASM_DIR.
#
###############################################################################

set -euo pipefail

python3 << 'PYEOF'
import importlib.util, os, re, subprocess, sys

TAG = "[fix_aiter_asm_dir]"


def find_aiter_core_and_hsa():
    """Return (core_py_path, aiter_meta_hsa_dir) or (None, None)."""
    spec = importlib.util.find_spec("aiter")
    if not spec or not spec.submodule_search_locations:
        return None, None

    core_path = None
    hsa_dir = None
    for base in spec.submodule_search_locations:
        candidate = os.path.join(base, "jit", "core.py")
        if os.path.isfile(candidate):
            core_path = candidate
        meta_hsa = os.path.normpath(os.path.join(base, "..", "aiter_meta", "hsa"))
        if os.path.isdir(meta_hsa):
            hsa_dir = meta_hsa
    return core_path, hsa_dir


def detect_gfx():
    """Detect GPU architecture from rocminfo."""
    try:
        out = subprocess.check_output(["rocminfo"], text=True, stderr=subprocess.DEVNULL)
        m = re.search(r"\b(gfx\w+)\b", out, re.IGNORECASE)
        return m.group(1).lower() if m else None
    except Exception:
        return None


def normalize_core_py(core_path):
    """Ensure AITER_ASM_DIR in core.py does NOT include a gfx subdirectory.

    aiter_hip_common.h's load_asm_kernel() already appends "/{arch_name}/" to
    AITER_ASM_DIR, so if core.py also includes it, the final path gets a
    double gfx prefix.  This affects:
      - Hardcoded gfx from a previous hook patch:  .../hsa/gfx942/
      - Upstream aiter >=0.1 using a variable:      .../hsa/{gfx}/
      - Upstream aiter >=0.1 using a function call:  .../hsa/{get_gfx()}/
      - Old hook variant with gfxs[0]:              .../hsa/{gfxs[0]}/

    Normalize all of them back to .../hsa/ so both C++ code paths work.
    """
    if not core_path or not os.path.isfile(core_path):
        return

    with open(core_path) as f:
        content = f.read()

    original = content

    # Match any AITER_ASM_DIR assignment that embeds a gfx component after /hsa/.
    # Covers: .../hsa/gfx942/, .../hsa/{gfx}/, .../hsa/{get_gfx()}/, and
    # the gfxs[0] ternary variant.  The [^"]+ greedily consumes everything
    # between /hsa/ and the closing /" that isn't a quote character.
    FIXED = 'AITER_ASM_DIR = f"{AITER_META_DIR}/hsa/"'
    # Single assignment:  AITER_ASM_DIR = f"{AITER_META_DIR}/hsa/<something>/"
    content = re.sub(
        r'AITER_ASM_DIR\s*=\s*f"[{]AITER_META_DIR[}]/hsa/[^"]+/"',
        FIXED, content,
    )
    # Ternary:  ... if gfxs else f"{AITER_META_DIR}/hsa/"
    content = re.sub(
        r'AITER_ASM_DIR\s*=\s*f"[{]AITER_META_DIR[}]/hsa/"'
        r'\s*if\s+gfxs\s+else\s+f"[{]AITER_META_DIR[}]/hsa/"',
        FIXED, content,
    )

    if content != original:
        try:
            with open(core_path, "w") as f:
                f.write(content)
            print(f"{TAG} Normalized AITER_ASM_DIR in {core_path} (removed gfx prefix)")
        except PermissionError:
            print(f"{TAG} Cannot write {core_path} (read-only)", file=sys.stderr)


def create_symlinks(hsa_dir, gfx):
    """Symlink entries from hsa/{gfx}/ into hsa/ so both code paths resolve.

    Existing symlinks are validated and repaired if they point to the wrong
    target (for example stale links to another gfx directory).
    """
    gfx_dir = os.path.join(hsa_dir, gfx)
    if not os.path.isdir(gfx_dir):
        print(f"{TAG} {gfx_dir} not found, skipping")
        return 0, 0

    created = 0
    repaired = 0
    for entry in os.listdir(gfx_dir):
        src = os.path.join(gfx, entry)           # relative: gfx942/fmha_v3_fwd
        dst = os.path.join(hsa_dir, entry)        # absolute: .../hsa/fmha_v3_fwd

        # If dst is already a symlink, verify target correctness.
        if os.path.islink(dst):
            try:
                cur_target = os.readlink(dst)
            except OSError as e:
                print(f"{TAG} readlink failed for {dst}: {e}", file=sys.stderr)
                continue

            if os.path.normpath(cur_target) == os.path.normpath(src):
                continue

            # Stale/wrong symlink -> repair in place.
            try:
                os.unlink(dst)
                os.symlink(src, dst)
                repaired += 1
                print(f"{TAG} Repaired stale symlink: {dst} -> {src} (was {cur_target})")
            except OSError as e:
                print(f"{TAG} repair symlink {dst} -> {src} failed: {e}", file=sys.stderr)
            continue

        # Real file/dir already exists (non-symlink); keep as-is.
        if os.path.lexists(dst):
            continue

        try:
            os.symlink(src, dst)
            created += 1
        except OSError as e:
            print(f"{TAG} symlink {dst} -> {src} failed: {e}", file=sys.stderr)

    return created, repaired


core_path, hsa_dir = find_aiter_core_and_hsa()
if not hsa_dir:
    print(f"{TAG} aiter_meta/hsa not found, skipping")
    sys.exit(0)

gfx = detect_gfx()
if not gfx:
    print(f"{TAG} Cannot detect GPU arch from rocminfo, skipping", file=sys.stderr)
    sys.exit(0)

normalize_core_py(core_path)

created, repaired = create_symlinks(hsa_dir, gfx)
if created > 0 or repaired > 0:
    print(f"{TAG} Symlink update done in {hsa_dir}: created={created}, repaired={repaired}")
else:
    print(f"{TAG} No symlinks needed (already present or {gfx}/ not found)")
PYEOF
