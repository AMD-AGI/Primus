# Base image
# FROM docker.io/rocm/megatron-lm:v25.9_gfx942
FROM docker.io/rocm/primus:v25.9_gfx942

# Specify the commit of Primus-Turbo when building: docker build --build-arg PRIMUS_TURBO_COMMIT=xxx .)
ARG PRIMUS_TURBO_COMMIT

# Install basic dependencies
RUN apt-get update

# Clone and install the Primus-Turbo
WORKDIR /opt
RUN mkdir -p /opt && cd /opt && \
    git clone https://github.com/AMD-AGI/Primus-Turbo.git && \
    cd Primus-Turbo && \
    git checkout ${PRIMUS_TURBO_COMMIT} && \
    git submodule update --init --recursive && \
    pip3 install -r requirements.txt && \
    GPU_ARCHS="gfx942;gfx950" pip3 install --no-build-isolation .

RUN sudo apt update && sudo apt install rdma-core libibverbs-dev \
    ibverbs-utils libopenmpi-dev openmpi-bin -y

# Set the default working directory
WORKDIR /opt

# check the installed Primus-Turbo package
RUN python3 -m pip show primus-turbo || true
