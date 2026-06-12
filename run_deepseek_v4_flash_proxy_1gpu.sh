#!/bin/bash
###############################################################################
# DeepSeek-V4-Flash single-GPU bring-up proxy on mi455 / gfx1250 (1 GPU).
#
# The upstream launchers (run_deepseek_v4.sh / run_deepseek_v4_flash_proxy.sh)
# go through `primus-cli slurm ...` against a 15-node MI355X SLURM cluster at
# TP=1 PP=1..8 EP=8. That doesn't apply here: this host is a single gfx1250
# box (no SLURM, one GPU we want to use). So this script instead drives the
# SAME `examples/run_pretrain.sh` entrypoint directly inside a local
# `docker run`, mirroring the validated gfx1250 docker / TransformerEngine
# recipe from ../../mi450/Primus/run_dsv3_proxy_4L.sh, and wraps the V4-Flash
# BF16 config with a MINIMUM-size single-GPU proxy:
#
#   - parallel        TP=1 PP=1 EP=1        (single GPU; no SLURM, no DeepEP)
#   - num_layers      4                     (vs production 43; MINIMUM slice
#                                            that still exercises every V4
#                                            attention layer kind)
#   - compress_ratios [0,4,128,0]           layer 0 = dense+SWA+sink (cr=0)
#                                            layer 1 = CSA           (cr=4)
#                                            layer 2 = HCA           (cr=128)
#                                            layer 3 = dense+SWA+sink (cr=0)
#   - num_experts     32  topk 1            (production 256/topk6 div by 8 =
#                                            the per-rank shape of the EP=8
#                                            production run; same scaling the
#                                            dsv3 proxy used. topk ceil(6/8)=1)
#   - moe_ffn_hidden  2048                  (full V4-Flash MoE width; from yaml)
#   - hidden_size     4096  heads 64  head_dim 512   (full V4-Flash; from yaml)
#   - seq_length      1024                  (MINIMUM that keeps the HCA cr=128
#                                            compressed pool non-trivial:
#                                            1024/128 = 8 pooled tokens)
#   - index_topk      128                   (CSA top-K; <= cr=4 pool 1024/4=256)
#   - precision       BF16                  (V4-Flash default; also dodges the
#                                            known tuned-hipBLASLt FP8 backward
#                                            GSU split-K deadlock on this host)
#
# Correctness-first defaults (this is a "does V4 train at all on 1 gfx1250
# GPU" bring-up, not a perf push). Everything below defaults to the robust
# eager / stock path and is ${VAR:-DEFAULT}-guarded so you can flip any knob
# from the command line without editing the script:
#
#   - V4 in-tree Triton attention kernels (dense / HCA / CSA) ........ OFF
#       (USE_V4_TRITON_ATTENTION / USE_V4_TRITON_CSA_ATTENTION). They were
#        tuned for MI355's 160 KiB SMEM at head_dim=512; eager is the safe
#        path for gfx1250 first bring-up. Set =True to A/B them.
#   - PrimusTurbo attention / DeepEP / grouped-MLP .................... OFF
#       (DeepEP is an EP>1 multi-GPU dispatcher; irrelevant at EP=1.)
#   - torch.compile Sinkhorn + plan-6 elementwise Triton fusions ...... OFF
#       (PRIMUS_*_TRITON below; flip individually to =1 for perf A/B.)
#   - tuned hipBLASLt .......................... OFF (hard-pinned, no opt-in)
#       (stock container hipBLASLt only; the tuned feba897 bundle deadlocks
#        this host on a backward-FP8 GSU split-K kernel -> node reboot.)
#   - kineto / pytorch profiler ...................................... OFF
#
# REQUIRED gfx1250 workaround (kept ON, single-GPU-safe): RCCL
# all_reduce(op=AVG) HANGS on this build even for single-rank process groups,
# and Megatron's MoE aux-loss reduce uses AVG. sitecustomize on PYTHONPATH
# rewrites AVG -> SUM/world_size (identity at world_size=1). See
# rccl_avg_workaround/sitecustomize.py.
#
# Usage:
#   ./run_deepseek_v4_flash_proxy_1gpu.sh                              # 10-iter smoke
#   PRIMUS_SEQ_LENGTH=2048 ./run_deepseek_v4_flash_proxy_1gpu.sh       # longer ctx
#   PRIMUS_NUM_EXPERTS=8 PRIMUS_MOE_TOPK=2 ./run_deepseek_v4_flash_proxy_1gpu.sh  # tiny MoE
#   USE_V4_TRITON_ATTENTION=True USE_V4_TRITON_CSA_ATTENTION=True \
#       ./run_deepseek_v4_flash_proxy_1gpu.sh                          # A/B Triton attn
#   HIP_VISIBLE_DEVICES=3 ./run_deepseek_v4_flash_proxy_1gpu.sh        # pin a card
###############################################################################
set -eo pipefail

export DOCKER_IMAGE=${DOCKER_IMAGE:-registry-sc-harbor.amd.com/framework/therock-npi@sha256:feba897e2a32a2465b8b296ed2662b2ad6136b5f1cf6f6c2716a3674aafc30f3}
SCRIPT_DIR=$(realpath -m "$(dirname "$0")")
# Prebuilt gfx1250 TransformerEngine wheel + source from the mi450 sibling tree
# (same image pin feba897). Override TE_DIR / TE_WHEEL_DIR if they live elsewhere.
export TE_DIR=${TE_DIR:-$(realpath -m "$SCRIPT_DIR/../../mi450/TransformerEngine")}
export TE_WHEEL_DIR=${TE_WHEEL_DIR:-$(realpath -m "$SCRIPT_DIR/../../mi450/dist/feba897")}

# ---------- Attention backend env (TE side) --------------------------------
# V4 attention runs its own eager / Triton kernels, not the TE fused-attn path,
# so these only matter for any non-V4 attention TE might still touch. Keep the
# AOTriton fused default from the dsv3 recipe; harmless for the V4 path.
export NVTE_FUSED_ATTN=1
export NVTE_FUSED_ATTN_CK=0
export NVTE_FUSED_ATTN_AOTRITON=1
export NVTE_USE_CK_GEMM=0
export NVTE_FLASH_ATTN=0

# ---------- hipBLASLt: STOCK ONLY (container built-in gfx1250 catalog) ------
# Hard-pinned to stock: no LD_PRELOAD, no tuned Tensile catalog, no env opt-in.
# The tuned feba897 bundle DEADLOCKS on a backward-FP8 GSU split-K kernel on
# this host (GPU wedge -> node reboot, debugged 2026-06-10), so this script
# deliberately has no tuned path. Also do NOT export an empty
# HIPBLASLT_TENSILE_LIBPATH into the container — a missing path breaks even
# stock hipBLASLt ("Cannot read TensileLibrary...", HIPBLASLT Error: 3).
unset HIPBLASLT_DIR HIPBLASLT_LD_PRELOAD HIPBLASLT_TENSILE_LIBPATH
echo "[hipblaslt] STOCK (container built-in gfx1250 catalog); tuned path intentionally not supported here"

# ---------- REQUIRED gfx1250 RCCL AVG->SUM workaround -----------------------
# sitecustomize is auto-imported by every Python worker (dir on PYTHONPATH).
export PYTHONPATH="$SCRIPT_DIR/rccl_avg_workaround:${PYTHONPATH:-}"

# ---------- Distributed / NCCL: single GPU, loopback only -------------------
export HSA_NO_SCRATCH_RECLAIM=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_HCA=
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export RCCL_DISABLE_AMDSMI=1
export NCCL_AMDSMI_DISABLE=1
export USING_AINIC=0          # no AINIC / IB on this host; skip run_pretrain's IB tuning

export GPUS_PER_NODE=1
export NNODES=1
export PYTHONUNBUFFERED=1

# ---------- V4-Flash MINIMUM single-GPU proxy shape -------------------------
export PRIMUS_TP=${PRIMUS_TP:-1}
export PRIMUS_PP=${PRIMUS_PP:-1}
export PRIMUS_EP=${PRIMUS_EP:-1}
export PRIMUS_TOTAL_LAYERS=${PRIMUS_TOTAL_LAYERS:-4}
export PRIMUS_COMPRESS_RATIOS=${PRIMUS_COMPRESS_RATIOS:-"[0,4,128,0]"}
export PRIMUS_NUM_EXPERTS=${PRIMUS_NUM_EXPERTS:-32}
export PRIMUS_MOE_TOPK=${PRIMUS_MOE_TOPK:-1}
export PRIMUS_MOE_FFN_HIDDEN_SIZE=${PRIMUS_MOE_FFN_HIDDEN_SIZE:-2048}
export PRIMUS_INDEX_TOPK=${PRIMUS_INDEX_TOPK:-128}
export PRIMUS_SEQ_LENGTH=${PRIMUS_SEQ_LENGTH:-1024}
export PRIMUS_MAX_POSITION_EMBEDDINGS=${PRIMUS_MAX_POSITION_EMBEDDINGS:-${PRIMUS_SEQ_LENGTH}}
export MBS=${MBS:-1}
export GBS=${GBS:-8}
export TRAIN_ITERS=${TRAIN_ITERS:-10}

# ---------- Perf knobs: OFF for a robust first bring-up ---------------------
export USE_V4_TRITON_ATTENTION=${USE_V4_TRITON_ATTENTION:-False}
export USE_V4_TRITON_CSA_ATTENTION=${USE_V4_TRITON_CSA_ATTENTION:-False}
export USE_V4_TILELANG_ATTENTION=${USE_V4_TILELANG_ATTENTION:-False}
export USE_V4_TILELANG_CSA_ATTENTION=${USE_V4_TILELANG_CSA_ATTENTION:-False}
export USE_TURBO_ATTENTION=${USE_TURBO_ATTENTION:-False}
export USE_TURBO_DEEPEP=${USE_TURBO_DEEPEP:-False}
export TURBO_USE_GROUPED_MLP=${TURBO_USE_GROUPED_MLP:-False}
export USE_V4_COMPILED_SINKHORN=${USE_V4_COMPILED_SINKHORN:-False}
# plan-6 elementwise Triton fusions read straight from os.environ in-kernel;
# default OFF here so the proxy runs the plain eager body. Flip any to 1 to A/B.
export PRIMUS_STACK_GROUPED_WEIGHT_TRITON=${PRIMUS_STACK_GROUPED_WEIGHT_TRITON:-0}
export PRIMUS_ROPE_TRITON=${PRIMUS_ROPE_TRITON:-0}
export PRIMUS_SINKHORN_TRITON=${PRIMUS_SINKHORN_TRITON:-0}
export PRIMUS_HC_TRITON=${PRIMUS_HC_TRITON:-0}
export PRIMUS_INDEXER_TRITON=${PRIMUS_INDEXER_TRITON:-0}
export PRIMUS_INDEXER_TRITON_FULL=${PRIMUS_INDEXER_TRITON_FULL:-0}
export PRIMUS_V4_ROUTER_TRITON=${PRIMUS_V4_ROUTER_TRITON:-0}

# enable_primus_turbo gates the turbo before_train patches; only on if a turbo
# path is requested.
export ENABLE_PRIMUS_TURBO=False
if [ "$USE_TURBO_ATTENTION" = "True" ] || [ "$USE_TURBO_DEEPEP" = "True" ] || [ "$TURBO_USE_GROUPED_MLP" = "True" ]; then
    ENABLE_PRIMUS_TURBO=True
fi

# MoE permute fusion (Triton permute_with_mask_map). VERIFIED working at
# flash proxy shapes (32 experts / hidden 4096, 10/10 iters 2026-06-10), so
# default ON here — but the SAME kernel's BWD autotune wedges the GPU at
# V4-Pro shapes (48 experts / hidden 7168; MES unrecoverable -> node reboot).
# If you grow this proxy's shapes and it hangs at iter-1 backward with GPU 0%,
# flip MOE_PERMUTE_FUSION=False first.
export MOE_PERMUTE_FUSION=${MOE_PERMUTE_FUSION:-True}

export PROFILE=${PROFILE:-False}
# PyTorch profiler writes the chrome trace via tensorboard_trace_handler(
# args.tensorboard_dir), so tensorboard must be enabled to get a trace.
# Auto-enable it when PROFILE=True. Profile window = [START,END); need
# TRAIN_ITERS > PROFILE_STEP_END.
export DISABLE_TENSORBOARD=${DISABLE_TENSORBOARD:-True}
if [ "$PROFILE" = "True" ]; then export DISABLE_TENSORBOARD=False; fi
export PROFILE_STEP_START=${PROFILE_STEP_START:-6}
export PROFILE_STEP_END=${PROFILE_STEP_END:-7}
export PRIMUS_TEAM=${PRIMUS_TEAM:-amd}
export PRIMUS_USER=${PRIMUS_USER:-gfx1250-1gpu}
export PRIMUS_EXP_NAME=${PRIMUS_EXP_NAME:-deepseek_v4_flash_1gpu_L${PRIMUS_TOTAL_LAYERS}_E${PRIMUS_NUM_EXPERTS}_seq${PRIMUS_SEQ_LENGTH}}

PRIMUS_PATH="$SCRIPT_DIR"
DATA_PATH="${PRIMUS_PATH}/data"
mkdir -p "$DATA_PATH"

EXP=${EXP:-examples/megatron/configs/MI355X/deepseek_v4_flash-FP8-pretrain.yaml}
LOG=${LOG:-deepseek-v4-flash-proxy-1gpu.log}

# ---------- FP8 training (matches upstream commit fde4c0d2) ------------------
# V4-Flash trains in FP8 by default now. PRECISION_TYPE=FP8 -> FP8=e4m3 +
# FP8_RECIPE=mxfp8 (the paper's ue8m0 microscaling; TE MXFP8BlockScaling).
# VERIFIED supported on this gfx1250 stock container (mxfp8_probe.py). Use the
# OCP e4m3 format (e4m3fnuz is unsupported here). A/B back to BF16 with
# PRECISION_TYPE=BF16 (or FP8=null).
export PRECISION_TYPE=${PRECISION_TYPE:-FP8}
if [ "$PRECISION_TYPE" = "FP8" ]; then
    export FP8=${FP8:-e4m3}
    export FP8_RECIPE=${FP8_RECIPE:-mxfp8}
else
    export FP8=${FP8:-null}
    export FP8_RECIPE=${FP8_RECIPE:-null}
    EXP=${EXP/deepseek_v4_flash-FP8/deepseek_v4_flash-BF16}
fi

if [ ! -d "$PRIMUS_PATH/third_party/Megatron-LM" ] || \
   [ -z "$(ls -A "$PRIMUS_PATH/third_party/Megatron-LM" 2>/dev/null)" ]; then
    echo "[ERROR] third_party/Megatron-LM missing/empty -> run: git submodule update --init --recursive" >&2
    exit 1
fi

# V4 single-GPU overrides (trailing args -> run_pretrain.sh -> primus cli
# train pretrain --config $EXP ...). Mirrors run_deepseek_v4.sh's CLI set,
# minus the SLURM / DeepEP wiring, scaled down to one GPU / minimum layers.
PROXY_OVERRIDES="\
    --backend_path $PRIMUS_PATH/third_party/Megatron-LM \
    --train_iters $TRAIN_ITERS \
    --lr_warmup_iters 0 \
    --lr_decay_iters $TRAIN_ITERS \
    --num_layers $PRIMUS_TOTAL_LAYERS \
    --compress_ratios $PRIMUS_COMPRESS_RATIOS \
    --micro_batch_size $MBS \
    --global_batch_size $GBS \
    --seq_length $PRIMUS_SEQ_LENGTH \
    --max_position_embeddings $PRIMUS_MAX_POSITION_EMBEDDINGS \
    --rope_type rope \
    --tensor_model_parallel_size $PRIMUS_TP \
    --pipeline_model_parallel_size $PRIMUS_PP \
    --expert_model_parallel_size $PRIMUS_EP \
    --num_experts $PRIMUS_NUM_EXPERTS \
    --moe_router_topk $PRIMUS_MOE_TOPK \
    --moe_router_enable_expert_bias False \
    --moe_ffn_hidden_size $PRIMUS_MOE_FFN_HIDDEN_SIZE \
    --index_topk $PRIMUS_INDEX_TOPK \
    --v4_grouped_experts_support_clamped_swiglu True \
    --mtp_num_layers 0 \
    --moe_permute_fusion $MOE_PERMUTE_FUSION \
    --mock_data True \
    --moe_router_force_load_balancing True \
    --log_avg_skip_iterations 3 \
    --enable_primus_turbo $ENABLE_PRIMUS_TURBO \
    --use_turbo_attention $USE_TURBO_ATTENTION \
    --use_turbo_deepep $USE_TURBO_DEEPEP \
    --use_turbo_grouped_mlp $TURBO_USE_GROUPED_MLP \
    --use_v4_triton_attention $USE_V4_TRITON_ATTENTION \
    --use_v4_triton_csa_attention $USE_V4_TRITON_CSA_ATTENTION \
    --use_v4_tilelang_attention $USE_V4_TILELANG_ATTENTION \
    --use_v4_tilelang_csa_attention $USE_V4_TILELANG_CSA_ATTENTION \
    --use_v4_compiled_sinkhorn $USE_V4_COMPILED_SINKHORN \
    --fp8 $FP8 \
    --fp8_recipe $FP8_RECIPE \
    --recompute_num_layers 0 \
    --recompute_granularity full \
    --recompute_method block \
    --gradient_accumulation_fusion False \
    --overlap_grad_reduce False \
    --overlap_param_gather False \
    --disable_last_saving True \
    --disable_wandb True \
    --disable_tensorboard $DISABLE_TENSORBOARD \
    --profile $PROFILE \
    --use_pytorch_profiler $PROFILE \
    --profile_step_start $PROFILE_STEP_START \
    --profile_step_end $PROFILE_STEP_END"

# Forward every env var the container needs (run_pretrain.sh / V4 builder /
# in-kernel knobs read these from the environment).
ENV_ARGS=()
for v in DOCKER_IMAGE NVTE_FUSED_ATTN NVTE_FUSED_ATTN_CK NVTE_FUSED_ATTN_AOTRITON \
         NVTE_FLASH_ATTN NVTE_USE_CK_GEMM PYTHONPATH HSA_NO_SCRATCH_RECLAIM \
         NCCL_IB_DISABLE NCCL_P2P_DISABLE NCCL_IB_HCA NCCL_SOCKET_IFNAME \
         GLOO_SOCKET_IFNAME RCCL_DISABLE_AMDSMI NCCL_AMDSMI_DISABLE USING_AINIC \
         GPUS_PER_NODE NNODES PYTHONUNBUFFERED TE_DIR TE_WHEEL_DIR \
         PRIMUS_SEQ_LENGTH PRIMUS_MAX_POSITION_EMBEDDINGS \
         PRIMUS_TEAM PRIMUS_USER PRIMUS_EXP_NAME \
         PRIMUS_STACK_GROUPED_WEIGHT_TRITON PRIMUS_ROPE_TRITON \
         PRIMUS_SINKHORN_TRITON PRIMUS_HC_TRITON PRIMUS_INDEXER_TRITON \
         PRIMUS_INDEXER_TRITON_FULL PRIMUS_V4_ROUTER_TRITON; do
    ENV_ARGS+=("--env" "$v")
done
# Pin a physical card if the caller set HIP_VISIBLE_DEVICES (run_pretrain.sh
# otherwise rewrites it to 0..GPUS_PER_NODE-1 = "0").
[[ -n "${HIP_VISIBLE_DEVICES:-}" ]] && ENV_ARGS+=("--env" "HIP_VISIBLE_DEVICES")
# Forward serialization knobs (for measuring AMD_SERIALIZE_COPY perf impact).
for v in AMD_SERIALIZE_COPY AMD_SERIALIZE_KERNEL HIP_LAUNCH_BLOCKING; do
    [[ -n "${!v:-}" ]] && ENV_ARGS+=("--env" "$v")
done

VOLUME_ARGS=(-v "$PRIMUS_PATH":"$PRIMUS_PATH" -v "$DATA_PATH":"$DATA_PATH")
[[ -d "$TE_WHEEL_DIR" ]] && VOLUME_ARGS+=(-v "$TE_WHEEL_DIR":"$TE_WHEEL_DIR")
[[ -d "$TE_DIR" ]] && VOLUME_ARGS+=(-v "$TE_DIR":"$TE_DIR")

# Prebuilt gfx1250 TE wheel + Primus deps, installed before launch.
TE_INSTALL_PREFIX="\
    if ls ${TE_WHEEL_DIR}/transformer_engine-*.whl >/dev/null 2>&1; then \
        echo '[TE] installing prebuilt wheel from ${TE_WHEEL_DIR}' && \
        pip install --quiet --force-reinstall --no-deps ${TE_WHEEL_DIR}/transformer_engine-*.whl && \
        pip install --quiet einops nvdlfw-inspect onnxscript onnx pydantic importlib-metadata packaging transformers pybind11; \
    else \
        echo '[TE] WARNING: no TE wheel found at ${TE_WHEEL_DIR}; run will likely fail'; \
    fi && \
    echo '[deps] installing Primus requirements' && \
    pip install --quiet -r requirements.txt && "

docker run --rm \
    "${ENV_ARGS[@]}" \
    --ipc=host --network=host \
    --device=/dev/kfd --device=/dev/dri \
    --cap-add=SYS_PTRACE --cap-add=CAP_SYS_ADMIN \
    --security-opt seccomp=unconfined --group-add video \
    --privileged \
    --name primus-v4-flash-1gpu \
    "${VOLUME_ARGS[@]}" \
    "$DOCKER_IMAGE" /bin/bash -c "\
        set -e && cd $PRIMUS_PATH && \
        ${TE_INSTALL_PREFIX}\
        echo '==================== V4-FLASH 1-GPU PROXY (gfx1250, BF16, eager, no profiler) ====================' && \
        EXP=$EXP GPUS_PER_NODE=1 NNODES=1 bash examples/run_pretrain.sh \
            ${PROXY_OVERRIDES}" \
    2>&1 | tee "$LOG"
