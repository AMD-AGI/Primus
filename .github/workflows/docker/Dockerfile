ARG BASE_IMAGE=docker.io/rocm/primus:v25.10
FROM ${BASE_IMAGE}

ARG PRIMUS_TURBO_COMMIT
ARG PRIMUS_TURBO_FRAMEWORK
ARG ROCSHMEM_COMMIT

# Non-interactive APT
ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# Install build dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y rdma-core libibverbs-dev libnuma-dev numactl&& \
    apt-get install -y --reinstall binutils


# ---------------------------------------------------------------------------
# Enviroment variables
# ---------------------------------------------------------------------------
ENV ROCSHMEM_HOME=/opt/rocshmem
ENV UCX_HOME=/opt/ucx
ENV MPI_HOME=/opt/ompi
ENV ROCM_HOME=/opt/rocm
ENV PRIMUS_TURBO_FRAMEWORK=${PRIMUS_TURBO_FRAMEWORK}
ENV LD_LIBRARY_PATH=${ROCM_HOME}/lib:${UCX_HOME}/lib:${MPI_HOME}/lib:${LD_LIBRARY_PATH}
ENV PATH=${ROCM_HOME}/bin:${PATH}:${MPI_HOME}/bin:${UCX_HOME}/bin:${PATH}

# ---------------------------------------------------------------------------
# Install rocSHMEM
# ---------------------------------------------------------------------------
RUN mkdir -p /tmp && cd /tmp && \
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


# ---------------------------------------------------------------------------
# Install Primus-Turbo
# ---------------------------------------------------------------------------
RUN cd /tmp && \
    git clone https://github.com/AMD-AGI/Primus-Turbo.git && \
    cd Primus-Turbo && \
    git checkout ${PRIMUS_TURBO_COMMIT} && \
    git submodule update --init --recursive && \
    pip3 install -r requirements.txt && \
    GPU_ARCHS="gfx942;gfx950" pip3 install --no-build-isolation .


# Set the default working directory
WORKDIR /opt

# check the installed Primus-Turbo package
RUN python3 -m pip show primus-turbo || true
