#!/bin/bash
# Plan-8 P49 — AOT-compile every plan-8 tilelang kernel ahead of the first
# proxy run so the cold-start JIT cost does not contaminate the EP=8
# steady-iter timer.
#
# At P49 the per-family kernels haven't landed yet, so this script
# only ensures the cache dir exists + the tilelang import works.
# Plan-8 P50..P55 will populate per-phase AOT compile loops.
set -euo pipefail
set -x

cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."
REPO_ROOT="$(pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

CACHE_DIR="${PRIMUS_V4_TILELANG_CACHE_DIR:-output/.tilelang_cache/v4}"
mkdir -p "$CACHE_DIR"

PYTHONPATH="${REPO_ROOT}" python <<'PY'
import sys
sys.path.insert(0, ".")
from primus.backends.megatron.core.transformer.v4_attention_kernels import _tilelang
print('plan-8 tilelang dispatcher available')
print('  pinned version:', _tilelang.TILELANG_VERSION_PIN)
print('  cache dir     :', _tilelang.cache_dir())
print('  env enabled   :', _tilelang.is_tilelang_path_enabled())
for name in ('v4_attention_fwd', 'v4_attention_bwd',
             'v4_csa_attention_fwd', 'v4_csa_attention_bwd'):
    print(f'  {name:<24s} available:', _tilelang.is_tilelang_kernel_available(name))
PY

# Plan-8 P50..P55 will append per-phase AOT compile invocations here.
# See deepseek-v4/develop/plan-8/02-phase-details.md §"Phase 49" for the
# expected shape envelope each phase covers.
