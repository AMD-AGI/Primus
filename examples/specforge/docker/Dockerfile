# SpecForge ROCm runtime: SGLang ROCm stack + SpecForge's spec-capture.patch
# + SpecForge (editable, --no-deps) + this Primus checkout (editable, --no-deps).
# Do not install Primus's full dependency tree or run primus-cli deps sync.
#
# Build, ownership, and docker run: docs/04-technical-guides/specforge/runtime-image.md
# Build from the Primus repository root:
#   docker build -f examples/specforge/docker/Dockerfile \
#     -t primus-specforge:v0.5.14-rocm700-mi35x .

# Digest is the pin. A floating tag here would silently pick up a new push of
# v0.5.14-rocm700-mi35x. Override with --build-arg for MI300X (mi30x tag).
ARG BASE_IMAGE=lmsysorg/sglang:v0.5.14-rocm700-mi35x@sha256:f1d38fd8076e69fdd103858149f9953a936cf9b322b2aa2cc97b08474d9a3623
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
RUN bash /tmp/install-light-deps.sh

# install-light-deps.sh uses --no-deps. SummaryWriter imports tensorboard's own
# dependency tree, so a no-deps install leaves --report-to tensorboard broken.
# Constrain the resolver to distributions already in the image: it may add
# missing tensorboard dependencies, and it must not upgrade the SGLang stack
# (numpy, protobuf, grpcio, setuptools, pillow). tensorboard 2.21 requires
# protobuf>=6.31.1 and grpcio>=1.74; stay below that so the constraint holds.
RUN python - <<'PY'
import json
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

watched = ["numpy", "protobuf", "grpcio", "setuptools", "pillow", "packaging"]
before = {}
for name in watched:
    try:
        before[name] = version(name)
    except PackageNotFoundError:
        before[name] = None
Path("/tmp/base-versions.json").write_text(json.dumps(before), encoding="utf-8")

freeze = subprocess.check_output(["python", "-m", "pip", "freeze"], text=True)
lines = []
for line in freeze.splitlines():
    if not line or line.startswith("-e ") or " @ " in line:
        continue
    dist = line.split("==", 1)[0].strip().lower().replace("_", "-")
    if dist in {"tensorboard", "tb-nightly"}:
        continue
    lines.append(line)
Path("/tmp/base-constraints.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
RUN python -m pip install --upgrade-strategy only-if-needed \
      -c /tmp/base-constraints.txt "tensorboard>=2.14,<2.21"
RUN python - <<'PY'
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

before = json.loads(Path("/tmp/base-versions.json").read_text(encoding="utf-8"))
changed = []
for name, old in before.items():
    if old is None:
        continue
    try:
        new = version(name)
    except PackageNotFoundError:
        new = None
    if new != old:
        changed.append(f"{name} {old} -> {new}")
if changed:
    raise SystemExit("tensorboard install changed base packages:\n" + "\n".join(changed))
Path("/tmp/base-constraints.txt").unlink(missing_ok=True)
Path("/tmp/base-versions.json").unlink(missing_ok=True)
PY

RUN bash /tmp/smoke-check.sh

ENV SPECFORGE_ROOT=/workspace/SpecForge
ENV PRIMUS_PATH=/opt/primus
WORKDIR /opt/primus
