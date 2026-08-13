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

# Primus-Turbo links librocshmem.a with -l: (not --whole-archive) under
# -fgpu-rdc/--hip-link, so rocSHMEM's team.cpp.o (which defines the plain,
# non-namespaced rocshmem::ROCSHMEM_TEAM_WORLD symbol) never gets pulled into
# libprimus_turbo_kernels.so; since it's a shared object the missing symbol
# only surfaces as an ImportError at dlopen/import time, not at build time.
# Work around it by re-exporting that symbol from a tiny shim .so, placed on
# libprimus_turbo_kernels.so's existing rpath and wired in via patchelf.
RUN PT_LIB=$(python3 -c "import primus_turbo, os; print(os.path.join(os.path.dirname(primus_turbo.__file__), 'lib', 'libprimus_turbo_kernels.so'))") && \
    mkdir -p /tmp/rocshmem_team_shim && cd /tmp/rocshmem_team_shim && \
    ar x ${ROCSHMEM_HOME}/lib/librocshmem.a team.cpp.o && \
    g++ -shared -fPIC -o ${ROCSHMEM_HOME}/lib/librocshmem_team_shim.so team.cpp.o \
      -L${ROCSHMEM_HOME}/lib -l:librocshmem.a -L${MPI_HOME}/lib -l:libmpi.so \
      -Wl,-rpath,${ROCSHMEM_HOME}/lib -Wl,-rpath,${MPI_HOME}/lib && \
    patchelf --add-needed librocshmem_team_shim.so "${PT_LIB}" && \
    cd / && rm -rf /tmp/rocshmem_team_shim

# ---------------------------------------------------------------------------
# Remove the CUDA-only CUTLASS DSL stack
# ---------------------------------------------------------------------------
# The base image's mamba-ssm pins quack-kernels, which pulls nvidia-cutlass-dsl.
# Its MLIR Python bindings and the FlyDSL ones installed with Primus-Turbo above
# share one process-wide nanobind type registry, so whichever loads second aborts
# (AMD-AGI/Primus#955). No ROCm path can run these CUDA-only kernels. Done in
# this layer because the base image is already published carrying them; names
# that are not installed are skipped instead of failing the build.
RUN pip3 uninstall -y quack-kernels nvidia-cutlass-dsl \
    nvidia-cutlass-dsl-libs-base nvidia-cutlass-dsl-libs-core \
    nvidia-cutlass-dsl-libs-cu12 nvidia-cutlass-dsl-libs-cu13

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

# Install origami (rocm-libraries@223648a). The v26.4 base image ships no
# origami at all. primus_turbo's MoE grouped-gemm kernel selection tolerates
# its absence (falls back to a heuristic, no measurable perf loss on gfx942),
# but Primus's performance-projection tool (primus projection performance)
# hard-depends on origami as its only supported GEMM simulation backend and
# has no fallback (see primus/core/projection/simulation_backends/factory.py)
# — dropping this install broke TestProjectionSimulate::* outright. Skipped
# for JAX, where the projection tool isn't used.
RUN if [ "$PRIMUS_TURBO_FRAMEWORK" != "JAX" ]; then \
    rm -rf /tmp/rocm-libraries && \
    git clone --filter=blob:none --no-checkout https://github.com/ROCm/rocm-libraries.git /tmp/rocm-libraries && \
    cd /tmp/rocm-libraries && \
    git sparse-checkout init --cone && \
    git sparse-checkout set shared/origami && \
    git checkout 223648a26928ebed7f3dd0ccdc044c09f1dccf9b && \
    (pip uninstall -y origami || true) && \
    pip install --no-cache-dir ./shared/origami/python && \
    rm -rf /tmp/rocm-libraries; \
    fi

# Set the default working directory
WORKDIR /opt

# check the installed Primus-Turbo package
RUN python3 -m pip show primus-turbo || true
