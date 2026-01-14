# Base image
FROM docker.io/rocm/primus:v25.10
ARG ROCSHMEM_COMMIT

# Specify the commit of Primus-Turbo when building: docker build --build-arg PRIMUS_TURBO_COMMIT=xxx .)
ARG PRIMUS_TURBO_COMMIT

# Install basic dependencies
RUN apt-get update && apt-get install -y rdma-core libibverbs-dev libnuma-dev

WORKDIR /opt
# Clone and install rocSHMEM
RUN mkdir -p /opt && cd /opt && \
    git clone https://github.com/ROCm/rocSHMEM.git && \
    cd rocSHMEM && \
    git checkout ${ROCSHMEM_COMMIT} && \
    mkdir build && \
    cd build && \
    MPI_ROOT=/opt/ompi UCX_ROOT=/opt/ucx INSTALL_PREFIX=/opt/rocshmem ../scripts/build_configs/all_backends
    
ENV ROCSHMEM_HOME=/opt/rocshmem
ENV UCX_HOME=/opt/ucx
ENV MPI_HOME=/opt/ompi

# Clone and install the Primus-Turbo
RUN cd /opt && \
    git clone https://github.com/AMD-AGI/Primus-Turbo.git && \
    cd Primus-Turbo && \
    git checkout ${PRIMUS_TURBO_COMMIT} && \
    git submodule update --init --recursive && \
    pip3 install -r requirements.txt && \
    GPU_ARCHS="gfx942;gfx950" pip3 install --no-build-isolation .

RUN apt-get install --reinstall binutils -y && apt-get install numactl -y


# Set the default working directory
WORKDIR /opt

# check the installed Primus-Turbo package
RUN python3 -m pip show primus-turbo || true
