ARG BASE_IMAGE=docker.io/rocm/primus:v26.5
FROM ${BASE_IMAGE}

ARG PRIMUS_TURBO_COMMIT
ARG PRIMUS_TURBO_AITER_COMMIT
ARG PRIMUS_TURBO_FRAMEWORK
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

# v26.5 exposes rocSHMEM through the pip-installed ROCm SDK under $ROCM_PATH
# instead of exporting a standalone $ROCSHMEM_HOME. Primus-Turbo auto-detects
# that library and builds its optional odc_rocshmem_* extensions. The v26.4
# team-symbol shim is no longer needed: the v26.5 static library links and
# imports cleanly.

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

# Install origami (rocm-libraries@223648a). The v26.5 base image ships no
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
