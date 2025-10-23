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


# Set the default working directory
WORKDIR /opt

# check the installed Primus-Turbo package
RUN python3 -m pip show primus-turbo || true
