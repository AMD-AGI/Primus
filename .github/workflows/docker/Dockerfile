ARG BASE_IMAGE=docker.io/rocm/primus:v26.1
FROM ${BASE_IMAGE}

ARG PRIMUS_TURBO_COMMIT
ARG PRIMUS_TURBO_FRAMEWORK
ARG ROCSHMEM_COMMIT
ARG UCCL_COMMIT
# Non-interactive APT
ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# Install build dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y rdma-core libibverbs-dev libnuma-dev numactl&& \
    apt-get install -y --reinstall binutils

RUN rm -rf /var/lib/apt/lists/*
# ---------------------------------------------------------------------------
# Enviroment variables
# ---------------------------------------------------------------------------
ENV ROCSHMEM_HOME=/opt/rocshmem
ENV UCX_HOME=/opt/ucx
ENV MPI_HOME=/opt/ompi
ENV ROCM_HOME=/opt/rocm
ENV PRIMUS_TURBO_FRAMEWORK=${PRIMUS_TURBO_FRAMEWORK}

ENV PATH="/opt/ompi/bin:/opt/ompi/sbin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/ompi/lib:${LD_LIBRARY_PATH}"
ENV GPU_ARCHS="gfx942;gfx950"
ENV PYTORCH_ROCM_ARCH="gfx942;gfx950"
ENV HCC_AMDGPU_TARGET="gfx942,gfx950"
# ---------------------------------------------------------------------------
# Install rocSHMEM
# ---------------------------------------------------------------------------
RUN mkdir -p /opt && cd /opt && \
    git clone https://github.com/ROCm/rocSHMEM.git && \
    cd rocSHMEM && \
    git checkout ${ROCSHMEM_COMMIT} && \
    mkdir build && \
    cd build && \
    MPI_ROOT=${MPI_HOME} UCX_ROOT=${UCX_HOME} INSTALL_PREFIX=${ROCSHMEM_HOME} ../scripts/build_configs/gda \
                                                                                            -DGDA_IONIC=ON \
                                                                                            -DGDA_MLX5=ON  \
                                                                                            -DGDA_BNXT=ON  \
                                                                                            -DUSE_IPC=ON
RUN rm -rf /opt/rocSHMEM

# ---------------------------------------------------------------------------
# Install Primus-Turbo
# ---------------------------------------------------------------------------
RUN cd /opt && \
    git clone https://github.com/AMD-AGI/Primus-Turbo.git && \
    cd Primus-Turbo && \
    git checkout ${PRIMUS_TURBO_COMMIT} && \
    git submodule update --init --recursive && \
    pip3 install -r requirements.txt && \
    pip3 install --no-build-isolation . -v

RUN rm -rf /opt/Primus-Turbo

# ---------------------------------------------------------------------------
# Install UCCL-EP
# ---------------------------------------------------------------------------
RUN cd /opt && \
    git clone https://github.com/uccl-project/uccl.git && \
    cd uccl && \
    git checkout ${UCCL_COMMIT} && \
    cd ep && python3 setup.py build && cd .. && \
    cp ep/build/**/*.so uccl && \
    pip3 install --no-build-isolation . -v \
    cd ep/deep_ep_wrapper && pip3 install --no-build-isolation . -v

RUN rm -rf /opt/uccl

# Set the default working directory
WORKDIR /opt

# check the installed Primus-Turbo package
RUN python3 -m pip show primus-turbo || true
