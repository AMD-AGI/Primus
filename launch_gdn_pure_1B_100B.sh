#!/bin/bash
###############################################################################
# All-in-one launcher for the Pure GDN 1B / 100B-tokens run.
#
#   Mirrors FLA's setup_and_train_gdn_pure_1B_100B.sh on the same node:
#     lr=3e-4, warmup=2000, batch=64/GPU, gpus=8, 95368 iters ≈ 100B tokens,
#     FineWeb-Edu sample-100BT (pre-tokenized by FLA).
#
#   Runs a full preflight (repo root, EXP file, FLA cache, tokenizer,
#   placeholder bin/idx, GPU count + free VRAM, disk space, config echo)
#   before invoking examples/run_pretrain.sh, so common config / data /
#   environment problems fail fast with a clear message instead of OOMing
#   or wedging the dataloader 5 minutes in.
#
# Usage (inside the Primus container, from /workspace/Primus):
#
#   bash launch_gdn_pure_1B_100B.sh                 # full preflight + train
#   DRY_RUN=1 bash launch_gdn_pure_1B_100B.sh       # only run preflight, skip training
#   SKIP_DISK_CHECK=1 bash launch_gdn_pure_1B_100B.sh
#   LOG=my_run.log bash launch_gdn_pure_1B_100B.sh
#   EXP=path/to/other-pretrain.yaml bash launch_gdn_pure_1B_100B.sh
#   PRIMUS_FLA_CACHE_DIR=/other/path/sample-100BT/train bash launch_gdn_pure_1B_100B.sh
###############################################################################

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# FLA-parity flags — same set the validated 75% GDN+MLA hybrid run used to
# hit FLA's 1.47 s/iter with bit-perfect iter-1 and <0.05 nat residual gap.
# PRIMUS_FLA_MLA_ATTN is omitted (this is a pure-GDN model, no MLA layers).
#
# NOTE (2026-05-27): The canonical surface for these knobs is now the EXP YAML
# (see `use_fla_fused_swiglu`, `use_fla_fused_rmsnorm`, `use_fla_short_conv`,
# `use_fla_data`, `fused_ce_mode`, `fused_ce_chunks` in
# examples/megatron/configs/MI300X/zebra_llama_1B_gdn_pure_100B-pretrain.yaml).
# The patch primus.backends.megatron.patches.fla_runtime_patches resolves
# each knob (env > YAML > default) onto args.* at phase="before_train".
# All consumers read getattr(get_args(), 'field', default) directly.
#
# These `export` lines are kept here for backward compatibility / quick
# ad-hoc overrides: env vars set on the launcher WIN over the YAML
# (matches the precedence the patch documents).  Comment them out to let
# the YAML be the sole source of truth.
# ─────────────────────────────────────────────────────────────────────────────
export PRIMUS_FUSED_CE=1        # FLA FusedLinearCrossEntropyLoss (chunked logits)
export PRIMUS_FLA_SWIGLU=1      # Triton SwiGLU (matches FLA fuse_swiglu)
export PRIMUS_FLA_NORM=1        # Triton RMSNorm + FusedRMSNormGated + fused MLP-norm
export PRIMUS_FLA_CONV=1        # Triton causal_conv1d for GDN short-conv
export PRIMUS_FLA_DATA=1        # drive Megatron from FLA's DistributedSampler order

# ─────────────────────────────────────────────────────────────────────────────
# empty_cache() interval (Primus patch — see
# primus/backends/megatron/patches/empty_cache_interval_patches.py and
# README_PERF_GDN_1B.md for the full speedup/profile story).
#
# The interval is set in the EXP YAML itself (the authoritative location):
#     overrides:
#       empty_cache_interval: 32
#
# That YAML knob makes torch.cuda.empty_cache() fire only every 32 iters
# instead of every iter, eliminating the 91%-of-wall hipMalloc/hipFree
# thrash measured in the pre-fix profile.
#
# This env var is OPTIONAL and only useful as an ad-hoc override (e.g.
# to A/B values without editing the YAML).  Leave unset and the YAML
# value wins.  Set to e.g. 1 to force passthrough for a single launch.
# ─────────────────────────────────────────────────────────────────────────────
# (intentionally not exported by default — YAML is the source of truth)
# export PRIMUS_EMPTY_CACHE_INTERVAL=32

# ─────────────────────────────────────────────────────────────────────────────
# Fused-CE chunk count.  Each chunk materialises [N/num_chunks, vocab] bf16
# logits — the dominant allocation in the loss step.
#
# Memory math at our config (batch=64, seq=2048, vocab=128256, bf16):
#   per-chunk logits bytes = (64*2048/num_chunks) * 128256 * 2
#     num_chunks=8  →  4.20 GiB/chunk   ← OOMs at 1B (allocator can't find
#                                          a contiguous 4 GiB block when
#                                          PyTorch is already at 185 GB used)
#     num_chunks=16 →  2.10 GiB/chunk   ← tight, may OOM with overlap on
#     num_chunks=32 →  1.05 GiB/chunk   ← safe (peak ~157 GB, 35 GB headroom)
#
# IMPORTANT: FLA also uses 8 chunks by default, but DeepSpeed ZeRO-2 frees
# the gradient buffer mid-backward, leaving more contiguous room than
# Megatron does. We must use 32 here at 1B even though it costs ~100 ms/iter.
# The bigger speedup (~30 ms/iter from overlap_grad_reduce) is in the YAML.
#
# Override here if you ever need to tune (must be a power of 2).
# ─────────────────────────────────────────────────────────────────────────────
export PRIMUS_FUSED_CE_CHUNKS=${PRIMUS_FUSED_CE_CHUNKS:-32}

# ─────────────────────────────────────────────────────────────────────────────
# Memory / allocator tuning — ROCm-specific.
#
# *** CRITICAL FINDING from inspecting torch/include/c10/hip/HIPAllocatorConfig.h
# in this exact PyTorch 2.7.0.dev+rocm6.3 build: ***
#
#     static bool expandable_segments() {
#     #ifndef PYTORCH_C10_DRIVER_API_SUPPORTED
#         if (instance().m_expandable_segments) {
#           TORCH_WARN_ONCE("expandable_segments not supported on this platform")
#         }
#         return false;                          // ← HARD-DISABLED!
#     #else
#         return instance().m_expandable_segments;
#     #endif
#     }
#
# So every prior attempt to set expandable_segments:True on this MI300X build
# was silently ignored — that warning we kept seeing in the log
#     "expandable_segments not supported on this platform"
# was telling us our fragmentation-control flag was a no-op the whole time,
# which is exactly why NCCL kept failing to carve out its 256 MiB workspace.
#
# The same header shows what IS actually parsed by HIPAllocatorConfig::parseArgs:
#     - max_split_size_mb            ← prevent splitting big blocks into junk
#     - garbage_collection_threshold ← release cached blocks back to hipMalloc
#     - roundup_power2_divisions     ← pack small allocs into pow-2 buckets
#     - release_lock_on_hipmalloc    ← let NCCL grab hipMalloc while we're busy
#
# We use all four to maximise the chance that NCCL can satisfy its calloc.
#
#  • max_split_size_mb:512            — never split a >512 MiB block, so the
#                                       1.7 GB grad-buffer blocks remain whole
#                                       and free-able in one go for NCCL.
#  • garbage_collection_threshold:0.8 — when pool > 80% of free VRAM, return
#                                       unused cached blocks to the driver so
#                                       NCCL's direct hipMalloc(256 MiB) can
#                                       succeed (root cause of iter-1
#                                       "Failed to CUDA calloc 268435456 bytes").
#  • roundup_power2_divisions:8       — round small allocs to 1/8-power-of-2
#                                       buckets, drastically cutting fragmentation
#                                       from GDN's many small chunked CE / conv
#                                       buffers.
#  • release_lock_on_hipmalloc:True   — drop the allocator mutex during
#                                       hipMalloc so NCCL on another stream can
#                                       call hipMalloc concurrently.
#
# (expandable_segments deliberately NOT set — would just warn and be ignored.)
# ─────────────────────────────────────────────────────────────────────────────
_ALLOC_CONF=${_ALLOC_CONF:-max_split_size_mb:512,garbage_collection_threshold:0.8,roundup_power2_divisions:8,release_lock_on_hipmalloc:True}
# PyTorch ROCm reads PYTORCH_HIP_ALLOC_CONF first, then falls back to
# PYTORCH_CUDA_ALLOC_CONF (legacy name).  Set both for safety.  The new
# unified PYTORCH_ALLOC_CONF does NOT exist in this 2.7.0.dev+rocm6.3 build.
#
# PRIMUS_FLA_ENV_DEFAULTS controls how much of our allocator+NCCL tuning to
# strip (used by launch_gdn_pure_1B_diag_flaenv*.sh):
#   0 (default)  → keep all four allocator knobs + NCCL_BUFFSIZE override
#   1            → unset PYTORCH_*_ALLOC_CONF entirely (matches FLA exactly,
#                  but caused SIGSEGV on rank 7 at first forward on our
#                  PyTorch 2.10.0a0+rocm6.3 build on 2026-05-26 — see
#                  primus_gdn_pure_1B_diag_flaenv.log)
#   minimal      → keep ONLY release_lock_on_hipmalloc:True (the one most
#                  likely to matter for RCCL workspace allocation safety),
#                  drop the other three knobs which are pure perf-tuning
if [ "${PRIMUS_FLA_ENV_DEFAULTS:-0}" = "1" ]; then
    unset PYTORCH_HIP_ALLOC_CONF PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF
elif [ "${PRIMUS_FLA_ENV_DEFAULTS:-0}" = "minimal" ]; then
    export PYTORCH_HIP_ALLOC_CONF="release_lock_on_hipmalloc:True"
    export PYTORCH_CUDA_ALLOC_CONF="release_lock_on_hipmalloc:True"
    unset PYTORCH_ALLOC_CONF || true
else
    export PYTORCH_HIP_ALLOC_CONF=${PYTORCH_HIP_ALLOC_CONF:-${_ALLOC_CONF}}
    export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-${_ALLOC_CONF}}
    unset PYTORCH_ALLOC_CONF || true   # unused on ROCm 6.3; remove stale carry-over
fi

# ─────────────────────────────────────────────────────────────────────────────
# NCCL workspace size.  Default is 4 MiB per channel; we observed that even
# 2 MiB calloc can fail when the GPU is at the very edge (1B model + batch=64
# on MI300X).  Halving the buffer is harmless to perf at our message sizes
# (we send 1B params × bf16 = 2.4 GB per all-reduce — buffer is just for
# fragmentation), and gives ~8 MB of headroom per channel × ~8 channels.
# Override if you ever see "Failed to CUDA calloc" with a different size.
# ─────────────────────────────────────────────────────────────────────────────
if [ "${PRIMUS_FLA_ENV_DEFAULTS:-0}" = "1" ] || [ "${PRIMUS_FLA_ENV_DEFAULTS:-0}" = "minimal" ]; then
    unset NCCL_BUFFSIZE
    unset NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS NCCL_NCHANNELS_PER_PEER
else
    export NCCL_BUFFSIZE=${NCCL_BUFFSIZE:-2097152}   # 2 MiB (default 4 MiB)

    # ─────────────────────────────────────────────────────────────────────────
    # Megatron-FSDP creates extra NCCL communicator groups (outer HSDP +
    # inner DP) on top of the standard data-parallel group. Each comm
    # allocates BUFFSIZE × #channels of workspace. At 99% VRAM there is no
    # contiguous 256 MiB hole (4 MiB × 64 channels = the prior failure).
    # Clamp channels to keep total comm workspace under ~16 MiB per comm.
    # Verified: with these clamps, FSDP runs without "Failed to CUDA calloc"
    # (50/500-iter diags) and saves 83 ms/iter vs ZeRO-1. See
    # experiments/results/SYNTHESIS.md.
    # ─────────────────────────────────────────────────────────────────────────
    export NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS:-1}
    export NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS:-4}
    export NCCL_NCHANNELS_PER_PEER=${NCCL_NCHANNELS_PER_PEER:-1}
fi

# Defaults (all overridable from the environment)
EXP=${EXP:-examples/megatron/configs/MI300X/zebra_llama_1B_gdn_pure_100B-pretrain.yaml}
LOG=${LOG:-primus_gdn_pure_1B_100B.log}
PRIMUS_FLA_CACHE_DIR=${PRIMUS_FLA_CACHE_DIR:-/home/vanbhati@amd.com/flash-linear-attention/legacy/training/data/HuggingFaceFW/fineweb-edu/sample-100BT/train}
export PRIMUS_FLA_CACHE_DIR

PLACEHOLDER_PREFIX=${PLACEHOLDER_PREFIX:-/home/vanbhati@amd.com/Primus/data/fla_aligned/fla_fineweb_edu_10BT_text_sentence}
TOKENIZER_HF_REPO=${TOKENIZER_HF_REPO:-meta-llama/Llama-3.2-1B}
EXPECTED_GPUS=${EXPECTED_GPUS:-8}
MIN_DISK_FREE_GB=${MIN_DISK_FREE_GB:-100}  # ~20 ckpts × ~5 GB each + headroom

DRY_RUN=${DRY_RUN:-0}
SKIP_DISK_CHECK=${SKIP_DISK_CHECK:-0}
SKIP_GPU_CHECK=${SKIP_GPU_CHECK:-0}
SKIP_TOKENIZER_CHECK=${SKIP_TOKENIZER_CHECK:-0}
SKIP_PATCH_CHECK=${SKIP_PATCH_CHECK:-0}
MEGATRON_DIR=${MEGATRON_DIR:-third_party/Megatron-LM}

# Pretty colours (auto-off when stdout is not a tty)
if [ -t 1 ]; then
    C_RED=$'\033[0;31m';  C_GRN=$'\033[0;32m'
    C_YEL=$'\033[0;33m';  C_CYA=$'\033[0;36m';  C_RST=$'\033[0m'
else
    C_RED=''; C_GRN=''; C_YEL=''; C_CYA=''; C_RST=''
fi
ok()   { echo "${C_GRN}[ OK ]${C_RST} $*"; }
warn() { echo "${C_YEL}[WARN]${C_RST} $*"; }
err()  { echo "${C_RED}[FAIL]${C_RST} $*" >&2; }
info() { echo "${C_CYA}[info]${C_RST} $*"; }

FAILED=0
fail_if() {  # fail_if <0|1> <message>
    if [ "$1" -ne 0 ]; then
        err "$2"
        FAILED=$((FAILED + 1))
    fi
}

echo "═════════════════════════════════════════════════════════════════════════"
echo "  Primus  ·  Pure GDN 1B  ·  100B tokens  ·  FLA-parity launcher"
echo "═════════════════════════════════════════════════════════════════════════"
echo "  EXP                  = ${EXP}"
echo "  LOG                  = ${LOG}"
echo "  PRIMUS_FLA_CACHE_DIR = ${PRIMUS_FLA_CACHE_DIR}"
echo "  PLACEHOLDER_PREFIX   = ${PLACEHOLDER_PREFIX}"
echo "  EXPECTED_GPUS        = ${EXPECTED_GPUS}"
echo "  DRY_RUN              = ${DRY_RUN}"
echo "─── FLA-parity flags ──────────────────────────────────────────────────"
echo "  PRIMUS_FUSED_CE          = ${PRIMUS_FUSED_CE}"
echo "  PRIMUS_FUSED_CE_CHUNKS   = ${PRIMUS_FUSED_CE_CHUNKS}"
echo "  PRIMUS_FLA_SWIGLU        = ${PRIMUS_FLA_SWIGLU}"
echo "  PRIMUS_FLA_NORM          = ${PRIMUS_FLA_NORM}"
echo "  PRIMUS_FLA_CONV          = ${PRIMUS_FLA_CONV}"
echo "  PRIMUS_FLA_DATA          = ${PRIMUS_FLA_DATA}"
echo "  PRIMUS_EMPTY_CACHE_INTERVAL = ${PRIMUS_EMPTY_CACHE_INTERVAL:-<unset, YAML wins>}"
echo "─── Memory (ROCm allocator + NCCL) ─────────────────────────────────────"
echo "  PRIMUS_FLA_ENV_DEFAULTS  = ${PRIMUS_FLA_ENV_DEFAULTS:-0}"
echo "  PYTORCH_HIP_ALLOC_CONF   = ${PYTORCH_HIP_ALLOC_CONF:-<unset, ROCm defaults>}"
echo "  PYTORCH_CUDA_ALLOC_CONF  = ${PYTORCH_CUDA_ALLOC_CONF:-<unset, ROCm defaults>}"
echo "  NCCL_BUFFSIZE            = ${NCCL_BUFFSIZE:-<unset, NCCL chooses>}"
echo "  NCCL channel clamp       = MIN=${NCCL_MIN_NCHANNELS:-<unset>} MAX=${NCCL_MAX_NCHANNELS:-<unset>} PER_PEER=${NCCL_NCHANNELS_PER_PEER:-<unset>}  (FSDP-safe)"
echo "═════════════════════════════════════════════════════════════════════════"
echo

# ───────────────────────── 1. Repo root + EXP file ──────────────────────────
info "1/9  Checking Primus repo root and EXP file..."
if [ ! -f examples/run_pretrain.sh ]; then
    fail_if 1 "examples/run_pretrain.sh not found — run this script from the Primus repo root (typically /workspace/Primus)."
else
    ok "Primus repo root detected at $(pwd)."
fi
if [ ! -f "${EXP}" ]; then
    fail_if 1 "EXP file does not exist: ${EXP}"
else
    ok "EXP found: ${EXP}"
fi

# ───────────────────────── 2. Model config sanity ───────────────────────────
info "2/9  Sanity-checking model config referenced by EXP..."
if [ -f "${EXP}" ]; then
    MODEL_YAML=$(awk '/^[[:space:]]*model:/ {print $2; exit}' "${EXP}" || true)
    if [ -n "${MODEL_YAML:-}" ]; then
        MODEL_PATH="primus/configs/models/megatron/${MODEL_YAML}"
        if [ -f "${MODEL_PATH}" ]; then
            ok "Model YAML: ${MODEL_PATH}"
            HID=$(awk '/^hidden_size:/ {print $2; exit}' "${MODEL_PATH}" || true)
            LYR=$(awk '/^num_layers:/ {print $2; exit}' "${MODEL_PATH}" || true)
            RATIO=$(awk '/^hybrid_attention_ratio:/ {print $2; exit}' "${MODEL_PATH}" || true)
            info "  hidden_size=${HID:-?}  num_layers=${LYR:-?}  hybrid_attention_ratio=${RATIO:-?}"
            if [ "${RATIO:-1.0}" != "0.0" ]; then
                warn "hybrid_attention_ratio is not 0.0 — this YAML may not be a pure-GDN model."
            fi
        else
            fail_if 1 "model YAML referenced by EXP not found: ${MODEL_PATH}"
        fi
    fi
fi

# ───────────────────────── 3. FLA cache directory ───────────────────────────
info "3/9  Checking PRIMUS_FLA_CACHE_DIR (FLA-aligned token order)..."
if [ ! -d "${PRIMUS_FLA_CACHE_DIR}" ]; then
    fail_if 1 "PRIMUS_FLA_CACHE_DIR does not exist: ${PRIMUS_FLA_CACHE_DIR}
       Without it FLAOrderGPTDataset silently falls back to Megatron's shuffler
       and the loss curves are no longer iter-for-iter comparable to FLA.
       Pre-tokenize the 100BT dataset with FLA's preprocess.py first
       (see flash-linear-attention/legacy/training/setup_and_train_gdn_pure_1B_100B.sh)."
else
    NUM_ARROW=$(find "${PRIMUS_FLA_CACHE_DIR}" -maxdepth 1 -name '*.arrow' | wc -l)
    CACHE_SIZE=$(du -sh "${PRIMUS_FLA_CACHE_DIR}" 2>/dev/null | awk '{print $1}')
    info "  arrow files = ${NUM_ARROW}   size = ${CACHE_SIZE:-?}"
    if [ "${NUM_ARROW}" -lt 100 ]; then
        warn "Only ${NUM_ARROW} arrow shards found — sample-100BT should produce ~780. Cache may be truncated."
    else
        ok "FLA cache looks healthy (${NUM_ARROW} shards, ${CACHE_SIZE})."
    fi
fi

# ───────────────────── 4. Placeholder bin/idx for Megatron ──────────────────
info "4/9  Checking Megatron placeholder .bin/.idx (config parser stat() target)..."
if [ ! -f "${PLACEHOLDER_PREFIX}.bin" ] || [ ! -f "${PLACEHOLDER_PREFIX}.idx" ]; then
    fail_if 1 "placeholder ${PLACEHOLDER_PREFIX}.bin/.idx not found.
       Megatron's index builder stat()s these even though FLAOrderGPTDataset
       overrides actual reads. Re-point train_data_path in the EXP yaml at any
       existing bin/idx prefix, or pre-build one with examples/scripts/preprocess_*.py."
else
    BIN_SIZE=$(du -h "${PLACEHOLDER_PREFIX}.bin" | awk '{print $1}')
    IDX_SIZE=$(du -h "${PLACEHOLDER_PREFIX}.idx" | awk '{print $1}')
    ok "placeholder bin/idx present (.bin=${BIN_SIZE}, .idx=${IDX_SIZE})."
fi

# ───────────────────────── 5. Tokenizer availability ────────────────────────
info "5/9  Checking tokenizer availability (${TOKENIZER_HF_REPO})..."
if [ "${SKIP_TOKENIZER_CHECK}" = "1" ]; then
    warn "Tokenizer check skipped (SKIP_TOKENIZER_CHECK=1)."
else
    if python3 - <<PY 2>/dev/null; then
import os
os.environ.setdefault("HF_DATASETS_OFFLINE", "0")
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained("${TOKENIZER_HF_REPO}")
PY
        ok "Tokenizer ${TOKENIZER_HF_REPO} loadable (cached or downloadable)."
    else
        warn "AutoTokenizer.from_pretrained(${TOKENIZER_HF_REPO}) failed.
       The trainer may try to download it on rank 0; set HF_HOME and ensure
       network is available or pre-cache it. Continuing anyway."
    fi
fi

# ───────────────────────── 6. GPU count + free VRAM ─────────────────────────
info "6/9  Checking GPU availability..."
# `set -e` + `pipefail` makes failed pipes inside command substitutions abort
# the script, so each probe below tolerates rocm-smi/nvidia-smi failures (the
# host node has no GPU driver — the real check happens inside the container).
if [ "${SKIP_GPU_CHECK}" = "1" ]; then
    warn "GPU check skipped (SKIP_GPU_CHECK=1)."
elif command -v rocm-smi >/dev/null 2>&1; then
    ROCM_OUT=$(rocm-smi --showid 2>&1 || true)
    NUM_GPUS=$(echo "${ROCM_OUT}" | grep -cE "GPU\[" || true)
    if [ "${NUM_GPUS}" -eq 0 ]; then
        # rocm-smi format varies / driver may not be loaded; fall back to /dev/dri probe.
        NUM_GPUS=$(ls /dev/dri/renderD* 2>/dev/null | wc -l || true)
    fi
    info "  detected ${NUM_GPUS} GPU(s)"
    if echo "${ROCM_OUT}" | grep -q "Driver not initialized"; then
        warn "rocm-smi reports 'Driver not initialized' — likely running outside the container."
        warn "Real GPU check will happen inside the container at training time."
    elif [ "${NUM_GPUS}" -lt "${EXPECTED_GPUS}" ]; then
        fail_if 1 "found only ${NUM_GPUS} GPU(s) but EXP expects ${EXPECTED_GPUS} (global_batch_size=512 = micro_batch_size × dp_size × grad_accum). Adjust micro_batch_size, global_batch_size, or EXPECTED_GPUS."
    else
        ok "${NUM_GPUS} GPU(s) available (≥ ${EXPECTED_GPUS} expected)."
        # Free VRAM probe — warn if any GPU is already >25% used.
        VRAM_CSV=$(rocm-smi --showmeminfo vram --csv 2>/dev/null || true)
        if [ -n "${VRAM_CSV}" ]; then
            BUSY=$(echo "${VRAM_CSV}" | awk -F',' '
                NR>1 && $3 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ {
                    if ($3*4 > $2) print $1
                }' | wc -l || true)
            if [ "${BUSY:-0}" -gt 0 ]; then
                warn "${BUSY} GPU(s) already have >25% VRAM in use. Stale process may OOM training."
            fi
        fi
    fi
elif command -v nvidia-smi >/dev/null 2>&1; then
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || true)
    if [ "${NUM_GPUS}" -lt "${EXPECTED_GPUS}" ]; then
        fail_if 1 "found only ${NUM_GPUS} GPU(s) but EXP expects ${EXPECTED_GPUS}."
    else
        ok "${NUM_GPUS} NVIDIA GPU(s) available."
    fi
else
    warn "Neither rocm-smi nor nvidia-smi found — cannot verify GPUs. Skipping."
fi

# ───────────────────────── 7. Disk-space check on output ────────────────────
info "7/9  Checking free disk space on ./output ..."
if [ "${SKIP_DISK_CHECK}" = "1" ]; then
    warn "Disk-space check skipped (SKIP_DISK_CHECK=1)."
else
    mkdir -p output
    FREE_GB=$(df -BG --output=avail output 2>/dev/null | tail -1 | tr -d 'G ' || true)
    if [ -z "${FREE_GB:-}" ]; then
        warn "Could not determine free space on ./output. Skipping."
    elif [ "${FREE_GB}" -lt "${MIN_DISK_FREE_GB}" ]; then
        fail_if 1 "only ${FREE_GB} GB free on ./output, need ≥ ${MIN_DISK_FREE_GB} GB
       (checkpoint ≈ 5 GB each × save_interval=5000 across 95368 iters,
       limit=3 keeps last 3 → ~15 GB steady-state, plus logs)."
    else
        ok "${FREE_GB} GB free on ./output (≥ ${MIN_DISK_FREE_GB} GB required)."
    fi
fi

# ───────────────────── 8. Megatron-LM submodule patches ────────────────────
# All six patches in megatron_patches/ must be applied or every PRIMUS_FLA_*
# env flag silently no-ops (they're guarded inside vendored Megatron code).
# We verify by grep-ing for unique markers each patch introduces.
info "8/9  Checking Megatron-LM patches are applied..."
if [ "${SKIP_PATCH_CHECK}" = "1" ]; then
    warn "Patch check skipped (SKIP_PATCH_CHECK=1)."
elif [ ! -d "${MEGATRON_DIR}" ]; then
    fail_if 1 "Megatron-LM submodule not found at ${MEGATRON_DIR}.
       Run 'git submodule update --init --recursive' first."
else
    # Each entry: <relative-path-in-submodule>:<unique-marker-the-patch-adds>
    PATCH_MARKERS=(
        "megatron/core/models/mamba/mamba_model.py:fused_ce_mode"             # patch 01
        "megatron/core/optimizer/__init__.py:PRIMUS_TORCH_OPTIM"              # patch 02
        "megatron/core/transformer/mlp.py:use_fla_fused_swiglu"               # patch 03
        "megatron/core/transformer/torch_norm.py:use_fla_fused_rmsnorm"       # patch 04
        "megatron/core/transformer/transformer_config.py:is_hybrid_model"     # patch 05
        "pretrain_mamba.py:use_fla_data"                                      # patch 06
    )
    MISSING=0
    for entry in "${PATCH_MARKERS[@]}"; do
        rel="${entry%%:*}"
        marker="${entry##*:}"
        full="${MEGATRON_DIR}/${rel}"
        if [ ! -f "${full}" ]; then
            warn "  missing file: ${full}"
            MISSING=$((MISSING + 1))
        elif ! grep -q "${marker}" "${full}"; then
            warn "  patch NOT applied to ${rel}  (no '${marker}' marker)"
            MISSING=$((MISSING + 1))
        fi
    done
    if [ "${MISSING}" -gt 0 ]; then
        fail_if 1 "${MISSING}/6 Megatron patches not applied.
       Run 'bash megatron_patch.sh' first.  Without these patches the
       fused CE, SwiGLU, RMSNorm, and FLA-data code paths are missing
       and the run will not match FLA."
    else
        ok "All 6 Megatron-LM patches applied (fused-CE, torch-optim, FLA-swiglu, FLA-rmsnorm, hybrid-init, FLA-data)."
    fi
fi

# ───────────────────────── 9. Effective schedule echo ───────────────────────
info "9/9  Effective schedule (parsed from ${EXP}):"
if [ -f "${EXP}" ]; then
    awk -v gpus="${EXPECTED_GPUS}" '
        /^[[:space:]]*train_iters:/        {ti=$2}
        /^[[:space:]]*micro_batch_size:/   {mb=$2}
        /^[[:space:]]*global_batch_size:/  {gb=$2}
        /^[[:space:]]*seq_length:/         {sl=$2}
        /^[[:space:]]*lr:/                 {if (!lr) lr=$2}
        /^[[:space:]]*min_lr:/             {ml=$2}
        /^[[:space:]]*lr_warmup_iters:/    {wu=$2}
        /^[[:space:]]*lr_decay_iters:/     {di=$2}
        /^[[:space:]]*lr_decay_style:/     {ds=$2}
        /^[[:space:]]*save_interval:/      {si=$2}
        END {
            tokens_per_step = gb * sl
            total_tokens    = ti * tokens_per_step
            grad_accum      = gb / (mb * gpus)
            printf "  train_iters       = %s\n", ti
            printf "  micro_batch_size  = %s\n", mb
            printf "  global_batch_size = %s   (dp_size = %d × grad_accum = %d)\n", gb, gpus, grad_accum
            printf "  seq_length        = %s\n", sl
            printf "  tokens / iter     = %d\n", tokens_per_step
            printf "  total tokens      = %.2f B\n", total_tokens / 1e9
            printf "  lr / min_lr       = %s / %s   (%s)\n", lr, ml, ds
            printf "  warmup / decay    = %s / %s iters\n", wu, di
            printf "  save_interval     = %s\n", si
        }
    ' "${EXP}"
fi
echo

# ───────────────────────── Verdict + launch ────────────────────────────────
if [ "${FAILED}" -gt 0 ]; then
    err "${FAILED} preflight check(s) failed. Fix the issues above and re-run."
    err "Set SKIP_DISK_CHECK=1 / SKIP_GPU_CHECK=1 / SKIP_TOKENIZER_CHECK=1 to bypass individual checks."
    exit 1
fi
ok "All preflight checks passed."
echo

if [ "${DRY_RUN}" = "1" ]; then
    info "DRY_RUN=1 — exiting before training launch."
    exit 0
fi

info "Launching training..."
info "  command: EXP=${EXP} bash examples/run_pretrain.sh"
info "  log    : ${LOG}"
echo

# Make EXP visible to run_pretrain.sh, then exec.
export EXP
bash examples/run_pretrain.sh 2>&1 | tee "${LOG}"
exit_code=${PIPESTATUS[0]}

echo
if [ "${exit_code}" -eq 0 ]; then
    ok "Training exited cleanly (code 0). Log: ${LOG}"
else
    err "Training exited with code ${exit_code}. Log: ${LOG}"
fi
exit "${exit_code}"
