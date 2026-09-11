# SpecForge ROCm runtime: SGLang ROCm stack + SpecForge's spec-capture.patch
# + SpecForge (editable, --no-deps) + this Primus checkout (editable, --no-deps).
# Do not install Primus's full dependency tree or run primus-cli deps sync.
#
# Build, ownership, and docker run: examples/specforge/docker/README.md
# Build from the Primus repository root:
#   docker build -f examples/specforge/docker/Dockerfile \
#     -t primus-specforge:v0.5.14-rocm700-mi35x .

ARG BASE_IMAGE=lmsysorg/sglang:v0.5.14-rocm700-mi35x
FROM ${BASE_IMAGE}

ARG SPECFORGE_REF=ec0988d527c828a24b8a5d639857f676e6e7f575

LABEL org.opencontainers.image.title="Primus SpecForge ROCm runtime"
LABEL org.opencontainers.image.description="SGLang v0.5.14 ROCm base with SpecForge capture patch and Primus SpecForge backend"

# The SGLang base image already ships git.
RUN command -v git >/dev/null

WORKDIR /workspace
RUN git clone https://github.com/sgl-project/SpecForge.git SpecForge \
    && git -C SpecForge checkout --detach "${SPECFORGE_REF}"

WORKDIR /sgl-workspace/sglang
RUN git apply --check /workspace/SpecForge/patches/sglang/v0.5.14/spec-capture.patch \
    && git apply /workspace/SpecForge/patches/sglang/v0.5.14/spec-capture.patch \
    && python -m py_compile python/sglang/srt/server_args.py \
       python/sglang/srt/spec_capture_sink.py

WORKDIR /workspace/SpecForge
RUN python -m pip install -e . --no-deps

# This checkout, not the published primus wheel (that wheel has no SpecForge backend).
WORKDIR /opt/primus
COPY pyproject.toml LICENSE README.md primus_cli.py /opt/primus/
COPY primus /opt/primus/primus
COPY runner /opt/primus/runner
COPY examples/specforge /opt/primus/examples/specforge
# hatchling is the build backend for an editable Primus install. It is not the
# Primus runtime dependency tree (do not run primus-cli deps sync).
RUN python -m pip install "hatchling>=1.21" \
    && python -m pip install -e /opt/primus --no-deps

COPY examples/specforge/docker/install-light-deps.sh examples/specforge/docker/smoke-check.sh /tmp/
RUN bash /tmp/install-light-deps.sh \
    && bash /tmp/smoke-check.sh

ENV SPECFORGE_ROOT=/workspace/SpecForge
ENV PRIMUS_PATH=/opt/primus
WORKDIR /opt/primus
