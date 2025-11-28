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

RUN apt-get install --reinstall binutils -y && apt-get install numactl -y

# LLVM has issue on DeepEP. Need to apply those patches
ARG LLVM_REPO=https://github.com/ROCm/llvm-project.git
ARG LLVM_BRANCH=release/rocm-rel-7.0

WORKDIR /opt
COPY .github/workflows/docker/7.0-deepep-swdev-540592.patch /opt/
RUN git config --global user.email "you@example.com"
RUN git config --global user.name "Your Name"
RUN cd /opt \
    && git clone ${LLVM_REPO} \
    && cd llvm-project \
    && git checkout ${LLVM_BRANCH} \
    && git am /opt/7.0-deepep-swdev-540592.patch \
    && mkdir build; cd build \
    && cmake -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/rocm/llvm/ \
        -DLLVM_ENABLE_PROJECTS="clang;lld" \
        -DLLVM_ENABLE_RUNTIMES=compiler-rt \
        -DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" \
        -DCLANG_DEFAULT_LINKER=lld \
        -DLLVM_PARALLEL_LINK_JOBS=4 \
        ../llvm \
    && ninja; ninja install \
    && cd ../.. \
    && rm -rf llvm-project \
    && cd /opt

# Set the default working directory
WORKDIR /opt

# check the installed Primus-Turbo package
RUN python3 -m pip show primus-turbo || true
