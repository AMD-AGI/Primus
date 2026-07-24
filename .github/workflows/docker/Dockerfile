ARG BASE_IMAGE=docker.io/rocm/primus:v26.4
FROM ${BASE_IMAGE}

ARG PRIMUS_TURBO_COMMIT
ARG PRIMUS_TURBO_AITER_COMMIT
ARG PRIMUS_TURBO_FRAMEWORK
# ARG ROCSHMEM_COMMIT # only needed if the rocSHMEM rebuild below is restored
ARG UCCL_COMMIT
ARG TRITON_COMMIT
# Non-interactive APT
ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# Install build dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y rdma-core libibverbs-dev libnuma-dev numactl libgoogle-glog-dev && \
    apt-get install -y --reinstall binutils

RUN rm -rf /var/lib/apt/lists/*
# ---------------------------------------------------------------------------
# Enviroment variables
# ---------------------------------------------------------------------------
ENV PRIMUS_TURBO_FRAMEWORK=${PRIMUS_TURBO_FRAMEWORK}

ENV GPU_ARCHS="gfx942;gfx950"
ENV PYTORCH_ROCM_ARCH="gfx942;gfx950"
ENV HCC_AMDGPU_TARGET="gfx942,gfx950"

# rocSHMEM: used by Primus ODC via Primus-Turbo's odc_rocshmem_* extensions.
# This base image already ships a prebuilt install (commit 17ff985c, version
# 3.2.0) at $ROCSHMEM_HOME=/opt/rocshmem, so building it again is redundant.
#
# If a future base image drops it, restore by uncommenting below (and the
# `ARG ROCSHMEM_COMMIT` above). Note: cmake reads the ROCm version from
# -DROCM_PATH=, not $ROCM_PATH; omitting it falls back to the nonexistent
# /opt/rocm and fails.
# RUN mkdir -p /opt && cd /opt && git clone https://github.com/ROCm/rocSHMEM.git && \
#     cd rocSHMEM && git checkout ${ROCSHMEM_COMMIT} && mkdir build && cd build && \
#     MPI_ROOT=${MPI_HOME} UCX_ROOT=${UCX_HOME} INSTALL_PREFIX=${ROCSHMEM_HOME} \
#     ../scripts/build_configs/gda -DROCM_PATH=${ROCM_PATH} \
#       -DGDA_IONIC=ON -DGDA_MLX5=ON -DGDA_BNXT=ON -DUSE_IPC=ON

# ---------------------------------------------------------------------------
# Install Primus-Turbo
# ---------------------------------------------------------------------------

RUN pip3 uninstall aiter amd-aiter -y || true && \
    cd /opt && \
    git clone https://github.com/ROCm/aiter.git && \
    cd aiter && \
    git checkout -f ${PRIMUS_TURBO_AITER_COMMIT} && \
    git submodule sync && \
    git submodule update --init --recursive && \
    PREBUILD_KERNELS=3 pip install --no-cache-dir --use-pep517 .

RUN cd /opt && \
    git clone https://github.com/AMD-AGI/Primus-Turbo.git && \
    cd Primus-Turbo && \
    git checkout ${PRIMUS_TURBO_COMMIT} && \
    git submodule update --init --recursive && \
    pip3 install -r requirements.txt && \
    pip3 install --no-build-isolation . -v

RUN rm -rf /opt/Primus-Turbo

# ---------------------------------------------------------------------------
# Install Triton
# ---------------------------------------------------------------------------
RUN cd /opt && \
    git clone https://github.com/triton-lang/triton.git && \
    cd triton && \
    git checkout ${TRITON_COMMIT} && \
    pip3 install -r python/requirements.txt && \
    MAX_JOBS=96 pip3 install --no-build-isolation --force-reinstall --no-deps .

RUN rm -rf /opt/triton

# torch's dist-info still pins the base image's stock triton build; without this,
# any later "pip install" (e.g. CI unit tests) sees torch's triton requirement as
# unsatisfiable and silently upgrades torch from PyPI, dragging in a CUDA stack.
RUN TORCH_META=$(python3 -c "import importlib.metadata as m; print(m.distribution('torch')._path / 'METADATA')") && \
    TRITON_VER=$(python3 -c "import importlib.metadata as m; print(m.version('triton'))") && \
    sed -i -E "s/^Requires-Dist: triton==.*/Requires-Dist: triton==${TRITON_VER}/" "${TORCH_META}"

# ---------------------------------------------------------------------------
# Install UCCL-EP (skip for JAX framework)
# ---------------------------------------------------------------------------
RUN if [ "$PRIMUS_TURBO_FRAMEWORK" != "JAX" ]; then \
    cd /opt && \
    git clone https://github.com/uccl-project/uccl.git && \
    cd uccl && \
    git checkout ${UCCL_COMMIT} && \
    cd ep && TORCH_CUDA_ARCH_LIST="gfx942,gfx950" python3 setup.py build && cd .. && \
    cp ep/build/**/*.so uccl && \
    pip3 install --no-build-isolation . -v && \
    cd ep/deep_ep_wrapper && pip3 install --no-build-isolation . -v && \
    rm -rf /opt/uccl; \
    fi

# No origami in this image; primus_turbo falls back to heuristic kernel
# selection for MoE grouped-gemm. To add it back:
#   pip install "git+https://github.com/ROCm/rocm-libraries.git@223648a26928ebed7f3dd0ccdc044c09f1dccf9b#subdirectory=shared/origami/python"

# Set the default working directory
WORKDIR /opt

# check the installed Primus-Turbo package
RUN python3 -m pip show primus-turbo || true
