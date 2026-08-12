#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# primus-cli --patch script: repoint a broken `ld.lld` launcher stub at a
# working LLD at container launch (before torchrun starts).
#
# Why: images that get ROCm from the pip `_rocm_sdk_devel` wheel (ROCm 7.14 and
# newer) put that wheel's llvm/bin early on PATH, where every tool is a ~26 KB
# forwarding stub. The `ld.lld` stub cannot resolve its own program path:
#
#   $ ld.lld --version
#   could not find path component of main program: 'ld.lld'
#
# FlyDSL shells out to `lld` to link the AMDGPU device module it JIT-compiles,
# so every FlyDSL kernel (PrimusTurbo native sparse-MLA attention, gemm_fp8,
# MegaMoE) dies with "lld invocation failed / An error happened while
# serializing the module". On multi-node runs that surfaces as a hang instead:
# the ranks that fail stall while the rest wait in the next collective.
#
# FlyDSL exposes no linker override, so the only lever is the binary that PATH
# resolves. Triton bundles a working LLD in the same image, which this patch
# links over the stub.
#
# The patch probes the toolchain and only acts on an image that is actually
# broken -- it skips when there is no `ld.lld` on PATH and when the one found
# already reports a version -- so it is a no-op on healthy images and needs no
# opt-in. Point it at a specific LLD with PRIMUS_LLD_REPLACEMENT (the PRIMUS_
# prefix auto-forwards into the container; see primus-cli-container.sh env
# passthrough).
#
# This is a runtime stop-gap. The durable fix is for the image to ship a working
# ld.lld in the ROCm SDK it installs.
###############################################################################
set -euo pipefail

# Probe the linker the way FlyDSL calls it: by bare name, through PATH. That
# distinction matters -- the SDK stub resolves itself when handed a path (running
# it as /full/path/ld.lld prints "AMD LLD <version>") and only fails when found
# through PATH, which is exactly how a compiler invokes it.
_lld_ok() {
    "$@" --version 2>&1 | head -1 | grep -qE "LLD [0-9]"
}

STUB="$(command -v ld.lld 2>/dev/null || true)"
if [[ -z "$STUB" ]]; then
    echo "[fix_lld_stub] no ld.lld on PATH; nothing to patch"
    exit 2 # 2 = skip (not an error), per runner/helpers/execute_patches.sh
fi
if _lld_ok ld.lld; then
    echo "[fix_lld_stub] ld.lld already works ($STUB) -- skipping"
    exit 2
fi

echo "[fix_lld_stub] ld.lld broken when invoked through PATH: $STUB"
echo "[fix_lld_stub]   $(ld.lld --version 2>&1 | head -1)"

# Prefer an explicit override, then Triton's bundled LLVM.
CANDIDATES=()
[[ -n "${PRIMUS_LLD_REPLACEMENT:-}" ]] && CANDIDATES+=("$PRIMUS_LLD_REPLACEMENT")
while IFS= read -r c; do CANDIDATES+=("$c"); done < <(
    ls -d /root/.triton/llvm/*/bin/ld.lld 2>/dev/null || true
)

for cand in "${CANDIDATES[@]}"; do
    _lld_ok "$cand" || continue
    ln -sf "$(readlink -f "$cand")" "$STUB"
    # Re-probe through PATH: a candidate that works by path is only useful if the
    # replacement also works by bare name (a real binary does, another stub
    # would not).
    if ! _lld_ok ld.lld; then
        echo "[fix_lld_stub] $cand still fails through PATH after linking" >&2
        continue
    fi
    echo "[fix_lld_stub] repointed $STUB -> $(readlink -f "$cand")"
    echo "[fix_lld_stub] $(ld.lld --version 2>&1 | head -1)"
    exit 0
done

echo "[fix_lld_stub] no working LLD found (tried: ${CANDIDATES[*]:-none})" >&2
exit 1
