# SpecForge ROCm runtime: SGLang ROCm stack + SpecForge capture plugin
# + SpecForge (editable, --no-deps) + this Primus checkout (editable, --no-deps).
# Do not install Primus's full dependency tree or run primus-cli deps sync.
#
# Build, ownership, and docker run: docs/04-technical-guides/specforge/runtime-image.md
# Build from the Primus repository root:
#   docker build -f examples/specforge/docker/Dockerfile \
#     -t primus-specforge:v0.5.23-rocm720-mi35x-20261103 .

# Nightly with the required generic capture extension APIs.
# Pin the tested digest so a re-pushed nightly tag cannot change the stack.
# Override BASE_IMAGE only with an image providing the required plugin APIs.
ARG BASE_IMAGE=lmsysorg/sglang-rocm:v0.5.23-rocm720-mi35x-20261103@sha256:4c9b1497e3793b2f5d216440688a53151ad67d656b8dcdc99311e79fbf734577
FROM ${BASE_IMAGE}

ARG SPECFORGE_REF=af1c3f3868e0ad1d1bc4a3bc4c577f2c9d3fa7b5

LABEL org.opencontainers.image.title="Primus SpecForge ROCm runtime"
LABEL org.opencontainers.image.description="SGLang v0.5.23 ROCm base with SpecForge capture plugin and Primus SpecForge backend"

# The SGLang base image already ships git.
RUN command -v git >/dev/null

WORKDIR /workspace
RUN git clone https://github.com/sgl-project/SpecForge.git SpecForge \
    && git -C SpecForge checkout --detach "${SPECFORGE_REF}"

WORKDIR /workspace/SpecForge
# Keep the base image's ROCm stack. The tested image's Python is older than
# SpecForge declares; do not resolve SpecForge's training dependency pins here.
RUN python -m pip install -e . --no-deps --ignore-requires-python \
    && python -m pip install -e plugins/sglang-spec-capture --no-deps

# This checkout, not the published primus wheel (that wheel has no SpecForge backend).
WORKDIR /opt/primus
COPY pyproject.toml LICENSE README.md primus_cli.py /opt/primus/
COPY primus /opt/primus/primus
COPY runner /opt/primus/runner
COPY examples/specforge /opt/primus/examples/specforge
# hatchling is the build backend for an editable Primus install. It is not the
# Primus runtime dependency tree (do not run primus-cli deps sync).
# Normalize Windows runner scripts (outside docker/*.sh in .gitattributes).
# primus-cli resolves its packaged launcher under primus/runner in editable installs.
RUN find /opt/primus/runner -type f -print0 | xargs -0 sed -i 's/\r$//' \
    && ln -sfn /opt/primus/runner /opt/primus/primus/runner \
    && python -m pip install "hatchling>=1.21" \
    && python -m pip install -e /opt/primus --no-deps

COPY examples/specforge/docker/install-light-deps.sh examples/specforge/docker/smoke-check.sh /tmp/
RUN sed -i 's/\r$//' /tmp/install-light-deps.sh /tmp/smoke-check.sh \
    && bash /tmp/install-light-deps.sh

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
