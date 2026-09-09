#!/bin/bash
# One-click FULL DeepSeek-V4-Flash SFT at 4k, multi-node on gfx942 / CDNA3 (MI308X, 192 GB).
#
# This trains the COMPLETE model -- 43 decoder layers + 1 MTP, 256 experts (top-6),
# compress_ratios cycling dense / CSA(4) / HCA(128) -- not the 4-layer cut. On this GPU
# it does not fit on one node: the expert optimizer state is unshardable across data
# parallelism (EP consumes the parallelism DP would use), so a single node cannot hold
# the whole model + optimizer. Three nodes (24 GPUs) is the minimum that does.
#
#   attention domain: TP=1 * PP=3 * CP=1  -> DP = 24/3 = 8
#   expert domain:    ETP=1 * EP=8 * PP=3 -> experts sharded 24-way
#   optimizer:        CPU-offloaded at fraction 0.75 (see the memory note below)
#
# NODE ADDRESSES ARE NOT HARDCODED. Each node exports MASTER_ADDR + NODE_RANK, then runs
# this same script. Example on a 3-node set (run in each node's container):
#
#   # node 0 (also the master):
#   MASTER_ADDR=<node0-ip> NODE_RANK=0 bash examples/deepseek-v4/gfx942/run_full_4k_multinode.sh
#   # node 1:
#   MASTER_ADDR=<node0-ip> NODE_RANK=1 bash examples/deepseek-v4/gfx942/run_full_4k_multinode.sh
#   # node 2:
#   MASTER_ADDR=<node0-ip> NODE_RANK=2 bash examples/deepseek-v4/gfx942/run_full_4k_multinode.sh
#
# Overridable knobs (all have sane defaults):
#   MASTER_ADDR   master node IP/host        (REQUIRED, no default)
#   NODE_RANK     this node's rank 0..N-1    (REQUIRED, no default)
#   NNODES        node count                 (default 3)
#   MASTER_PORT   rendezvous port            (default 29710)
#   NCCL_SOCKET_IFNAME  socket NIC           (default ens50f0 -- the management NIC)
#   V4_TOKENIZER  local tokenizer dir        (default /apps/DeepSeek-V4-Flash)
#   TRAIN_ITERS   steps                      (default 10)
#   PRIMUS_OPTIMIZER_OFFLOAD_FRACTION        (default 0.75)
#
# Why each gfx942 / socket / memory setting is needed is documented in
# examples/deepseek-v4/gfx942/README_full_4k_multinode.md.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"
cd "${REPO}" || exit 1

# ---- required inputs ---------------------------------------------------------------
if [ -z "${MASTER_ADDR:-}" ]; then
  echo "[error] MASTER_ADDR is required (the master node's IP/host). e.g. MASTER_ADDR=10.0.0.1 NODE_RANK=0 bash $0" >&2
  exit 1
fi
if [ -z "${NODE_RANK:-}" ]; then
  echo "[error] NODE_RANK is required (this node's rank, 0..NNODES-1)." >&2
  exit 1
fi

# ---- prerequisites -----------------------------------------------------------------
# Probe the known locations rather than assuming one: the directory has been renamed at
# least once, and the check below aborts the whole run before anything else happens.
if [ -z "${V4_TOKENIZER:-}" ]; then
  for _c in /apps/DeepSeek-V4-Flash /apps/DeepSeek-V4-Flash-FP8; do
    [ -f "${_c}/tokenizer.json" ] && { V4_TOKENIZER="${_c}"; break; }
  done
fi
V4_TOKENIZER="${V4_TOKENIZER:-/apps/DeepSeek-V4-Flash}"
DATA_DIR="${DATA_DIR:-${REPO}/data/sft}"
BACKEND_PATH="${BACKEND_PATH:-${REPO}/third_party/Megatron-LM}"

if [ ! -f "${V4_TOKENIZER}/tokenizer.json" ]; then
  echo "[error] no tokenizer at ${V4_TOKENIZER}. Set V4_TOKENIZER=<dir> (needs tokenizer.json)." >&2
  exit 1
fi

# Megatron-LM: the container image already ships the exact pinned commit; reuse it
# instead of a network fetch when the submodule is not checked out.
if [ ! -f "${BACKEND_PATH}/megatron/training/arguments.py" ]; then
  if [ -f /workspace/Primus/third_party/Megatron-LM/megatron/training/arguments.py ]; then
    echo "[setup] copying Megatron-LM from the image"
    mkdir -p "${REPO}/third_party"
    cp -a /workspace/Primus/third_party/Megatron-LM "${REPO}/third_party/"
    rm -f "${BACKEND_PATH}/.git"
  else
    echo "[setup] fetching Megatron-LM submodule"
    git -C "${REPO}" submodule update --init third_party/Megatron-LM
  fi
fi

SFT_JSONL="${SFT_JSONL:-${DATA_DIR}/sft_4096.jsonl}"
if [ ! -f "${SFT_JSONL}" ]; then
  echo "[setup] building 4k SFT data (one-off)"
  python "${HERE}/prepare_sft_data.py" --tokenizer "${V4_TOKENIZER}" \
         --out-dir "${DATA_DIR}" --lengths 4096 --rows 64
fi

# ---- launcher / rendezvous ---------------------------------------------------------
export PRIMUS_LAUNCHER=direct
# run_deepseek_v4.sh hard-requires DOCKER_IMAGE via `${DOCKER_IMAGE:?...}` even though
# PRIMUS_LAUNCHER=direct never launches a container -- we are already inside one. Without
# this the script depends on an ambient variable that happens to be set in some shells,
# which is how it "worked" before while not being self-contained.
export DOCKER_IMAGE="${DOCKER_IMAGE:-in-container/direct-launcher}"

# ---- primus_turbo compatibility -------------------------------------------------------
# Two independent version mismatches against the primus_turbo in rocm/primus:v26.5:
#   * main's grad-accum fusion passes `fuse_bgrad_accum_pattern` to grouped_gemm, which
#     that build does not accept;
#   * main's MegaMoE imports primus_turbo.pytorch.ops.moe.fused_mega_moe, which that build
#     calls mega_moe_fused (renamed upstream).
# Both default off here so the recipe runs on the documented image; set them back on a
# newer one.
# The effective switch is TURBO_USE_GROUPED_MLP, not the yaml key: run_deepseek_v4.sh
# passes `--use_turbo_grouped_gemm "$TURBO_USE_GROUPED_MLP"` on the command line, which
# overrides whatever the yaml says. It defaults to EP>1, i.e. True for this recipe.
export TURBO_USE_GROUPED_MLP="${TURBO_USE_GROUPED_MLP:-False}"
export YAML_TURBO_GROUPED_GEMM="${YAML_TURBO_GROUPED_GEMM:-false}"
export PRIMUS_USE_TURBO_GROUPED_GEMM="${PRIMUS_USE_TURBO_GROUPED_GEMM:-false}"
export PRIMUS_OPT_MEGA_MOE="${PRIMUS_OPT_MEGA_MOE:-0}"
unset SLURM_JOB_ID SLURM_JOBID SLURM_NODELIST 2>/dev/null || true
export NNODES="${NNODES:-3}"
export NODE_RANK MASTER_ADDR
export MASTER_PORT="${MASTER_PORT:-29710}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export BACKEND_PATH
export PRIMUS_OUTPUT_ROOT="${PRIMUS_OUTPUT_ROOT:-${REPO}/output}"
export PRIMUS_TEAM="${PRIMUS_TEAM:-amd}"
export PRIMUS_USER="${PRIMUS_USER:-$(whoami)}"

# ---- networking: socket over the management NIC ------------------------------------
# RDMA on this fabric is unusable (PFC unconfigured -> IBV_WC_RETRY_EXC_ERR, ionic GDA
# hang), so NCCL runs over TCP on the management NIC. Pin the interface for NCCL, Gloo
# AND torch.distributed's TP-group sockets -- all three, or rendezvous / TP groups bind
# the wrong interface and hang.
export USING_AINIC="${USING_AINIC:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens50f0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME}}"
export TP_SOCKET_IFNAME="${TP_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME}}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export HSA_NO_SCRATCH_RECLAIM="${HSA_NO_SCRATCH_RECLAIM:-0}"

# ROOT-CAUSE FIX for the first-step hang at logical_and_across_model_parallel_group (a
# 1-int all_reduce on the cross-node model-parallel group). NCCL defaults a small
# all_reduce to the Tree algorithm, and Tree-over-TCP-socket across nodes DEADLOCKS on
# this fabric. Forcing Ring fixes it (proven on a working 6-node run on the same fabric:
# all-reduce/all-gather/all-to-all all passed over ens50f0). RCCL_USE_AMD_SMI_LIB=1 is
# the fabric-topology probe that run needs too.
export NCCL_ALGO="${NCCL_ALGO:-Ring}"
export RCCL_USE_AMD_SMI_LIB="${RCCL_USE_AMD_SMI_LIB:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"

# ---- gfx942 (CDNA3) kernel settings ------------------------------------------------
# V4 sparse-MLA backward Triton kernel is gfx950-tuned (needs 160 KB LDS); gfx942 has
# 64 KB, so num_stages=1 disables LDS multi-buffering to let it compile. The full indexer
# Triton path avoids an eager einsum that materialises [B,S,H,P].
export PRIMUS_DSA_BWD_NUM_STAGES="${PRIMUS_DSA_BWD_NUM_STAGES:-1}"
export PRIMUS_INDEXER_TRITON_FULL="${PRIMUS_INDEXER_TRITON_FULL:-1}"

# ---- parallelism: full model over 3 nodes ------------------------------------------
# The flash launcher's NNODES=3 branch sets PP=3/EP=8 and the layout Et*14|t*14|t*15mL.
export PRIMUS_TP="${PRIMUS_TP:-1}"
export PRIMUS_PP="${PRIMUS_PP:-3}"
export PRIMUS_EP="${PRIMUS_EP:-8}"
export PRIMUS_CP="${PRIMUS_CP:-1}"

# ---- optimizer CPU offload (the lever that makes the full model fit) ----------------
# The expert optimizer state is 147 GB/rank and cannot be sharded across DP (expert_dp=1),
# so it must leave the GPU. fraction=0.75 is CALCULATED, not guessed: it puts 113 GB/rank
# on the host (~1178 GB/node peak incl. pinned-memory overhead -> safe on a 3 TB host) and
# 38 GB/rank back on the GPU (~109 GB of 192 -> 83 GB headroom). fraction=1.0 overran the
# host and got a rank OOM-killed; lower fractions push the GPU to OOM. 0.75 is the middle.
export PRIMUS_OPTIMIZER_CPU_OFFLOAD="${PRIMUS_OPTIMIZER_CPU_OFFLOAD:-true}"
export PRIMUS_OPTIMIZER_OFFLOAD_FRACTION="${PRIMUS_OPTIMIZER_OFFLOAD_FRACTION:-0.75}"

# ---- model / data (full 43L + MTP, 256 experts, 4k) --------------------------------
export EXP="${EXP:-examples/megatron/configs/MI355X/deepseek_v4_flash_4layer-BF16-sft.yaml}"
export PRIMUS_SEQ_LENGTH="${PRIMUS_SEQ_LENGTH:-4096}"
export PRIMUS_MAX_POSITION_EMBEDDINGS="${PRIMUS_MAX_POSITION_EMBEDDINGS:-4096}"
export SFT_JSONL
export V4_TOKENIZER
export PRIMUS_LR="${PRIMUS_LR:-1.0e-6}"
export MBS="${MBS:-1}"
export GBS="${GBS:-24}"
export TRAIN_ITERS="${TRAIN_ITERS:-10}"
export PRIMUS_EXP_NAME="${PRIMUS_EXP_NAME:-dsv4_flash_full_4k_${NNODES}node}"

# ---- SFT plumbing (full-model SFT vs pretrain defaults) ----------------------------
# pp_warmup trips SFT forward_step; mock_data installs a NullTokenizer fatal to real SFT;
# expert bias needs sigmoid scoring which conflicts with V4's sqrtsoftplus.
export PP_WARMUP="${PP_WARMUP:-False}"
export MOCK_DATA="${MOCK_DATA:-False}"
export PRIMUS_MOE_ENABLE_EXPERT_BIAS="${PRIMUS_MOE_ENABLE_EXPERT_BIAS:-False}"

# ---- attention backend: triton_v2 (turbo won't compile on gfx942) ------------------
export USE_V4_ATTENTION_BACKEND="${USE_V4_ATTENTION_BACKEND:-triton_v2}"
export USE_V4_CSA_ATTENTION_BACKEND="${USE_V4_CSA_ATTENTION_BACKEND:-triton_v2}"
export USE_TURBO_ATTENTION="${USE_TURBO_ATTENTION:-False}"

LOGDIR="${PRIMUS_OUTPUT_ROOT}/${PRIMUS_TEAM}/${PRIMUS_USER}/${PRIMUS_EXP_NAME}"
mkdir -p "${LOGDIR}"
LOG="${LOGDIR}/log_node${NODE_RANK}.txt"

echo "[run] repo=${REPO}"
echo "[run] FULL model: 43L+MTP, 256 experts, seq=${PRIMUS_SEQ_LENGTH}"
echo "[run] nodes=${NNODES} rank=${NODE_RANK} master=${MASTER_ADDR}:${MASTER_PORT} nic=${NCCL_SOCKET_IFNAME}"
echo "[run] TP=${PRIMUS_TP} PP=${PRIMUS_PP} EP=${PRIMUS_EP} CP=${PRIMUS_CP}  offload_frac=${PRIMUS_OPTIMIZER_OFFLOAD_FRACTION}"
echo "[run] iters=${TRAIN_ITERS} gbs=${GBS} lr=${PRIMUS_LR}"
echo "[run] log=${LOG}"

exec bash examples/deepseek-v4/run_deepseek_v4_flash.sh > "${LOG}" 2>&1
