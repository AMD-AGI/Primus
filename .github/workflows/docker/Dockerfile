# Base image
FROM docker.io/rocm/megatron-lm:v25.9_gfx942

# Specify the commit of Primus-Turbo when building: docker build --build-arg PRIMUS_TURBO_COMMIT=xxx .)
ARG PRIMUS_TURBO_COMMIT

# Install basic dependencies
RUN apt-get update

RUN pip3 install fbgemm-gpu && \
    pip3 install torch==2.10.0.dev20251019+rocm7.0 \
                 torchao==0.15.0.dev20251015+rocm7.0 \
                 torchvision==0.25.0.dev20251020+rocm7.0 \
                 --index-url https://download.pytorch.org/whl/nightly/rocm7.0 \
                 --force-reinstall

# Clone and install the Primus-Turbo
WORKDIR /opt
RUN pip3 uninstall aiter primus_turbo -y && \
    mkdir -p /opt && cd /opt && \
    git clone https://github.com/AMD-AGI/Primus-Turbo.git && \
    cd Primus-Turbo && \
    git checkout ${PRIMUS_TURBO_COMMIT} && \
    git submodule update --init --recursive && \
    pip3 install -r requirements.txt && \
    GPU_ARCHS="gfx942;gfx950" pip3 install --no-build-isolation .

ENV MAX_JOBS=128
ENV PYTORCH_ROCM_ARCH="gfx942;gfx950"
RUN pip uninstall flash_attn transformer_engine grouped_gemm -y

# FA
ARG FA_REPO=https://github.com/ROCm/flash-attention.git
ARG FA_BRANCH=6387433156558135a998d5568a9d74c1778666d8
ENV GPU_ARCHS="${PYTORCH_ROCM_ARCH}"

RUN git clone --recursive ${FA_REPO} \
    && cd flash-attention \
    && git checkout ${FA_BRANCH} \
    && python setup.py install \
    && cd .. \
    && rm -rf flash-attention

# Latest TransformerEngine
ARG TE_REPO=https://github.com/ROCm/TransformerEngine.git
ARG TE_BRANCH=1834247827f47f5eab76ea790841168ac778592e

ENV NVTE_USE_HIPBLASLT=1
ENV NVTE_FRAMEWORK=pytorch
ENV NVTE_ROCM_ARCH=${PYTORCH_ROCM_ARCH}
ENV NVTE_USE_CAST_TRANSPOSE_TRITON=0
ENV NVTE_CK_IS_V3_ATOMIC_FP32=0
ENV NVTE_CK_USES_BWD_V3=1
ENV NVTE_CK_USES_FWD_V3=1
ENV CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=2
ENV NVTE_CK_HOW_V3_BF16_CVT=2
ENV NVTE_USE_ROCM=1

RUN git clone --recursive ${TE_REPO} \
    && cd TransformerEngine \
    && git checkout ${TE_BRANCH} \
    && git submodule update --init --recursive \
    && MAX_JOBS=${MAX_JOBS} pip install . \
    && cd ..

# Groupped GEMM
ARG GROUPED_GEMM_REPO=https://github.com/caaatch22/grouped_gemm.git
ARG GROUPED_GEMM_BRANCH=rocm

RUN git clone ${GROUPED_GEMM_REPO} &&\
    cd grouped_gemm &&\
    git checkout ${GROUPED_GEMM_BRANCH} &&\
    git submodule update --init --recursive &&\
    pip install . && cd ../ && rm -rf grouped_gemm

# APEX 1.7.0
ARG APEX_REPO=https://github.com/rocm/apex
ARG APEX_BRANCH=release/1.7.0

RUN pip uninstall -y apex \
    && git clone ${APEX_REPO} -b ${APEX_BRANCH} \
    && cd apex \
    && pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./ \
    && cd .. && rm -r apex

# Set the default working directory
WORKDIR /opt

# check the installed Primus-Turbo package
RUN python3 -m pip show primus-turbo || true
