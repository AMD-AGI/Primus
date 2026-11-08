# syntax=docker/dockerfile:1.7
#
# MI455X (gfx1250) training image for the Kimi K3.1 Pro 4-layer proxy (Primus / Megatron) and
# the MLPerf DLRMv4 HSTU ranker (generative_recommenders). Build with ./build.sh, which supplies
# the two named build contexts:
#   primus   a `git archive` of this Primus checkout, with third_party/Megatron-LM
#   dlrmv4   the DLRMv4 `recommendation/` tree
# See README.md in this directory for what each layer fixes and how to reproduce the results.

ARG BASE_IMAGE=amdprimus/amdprimus:tas-20261019-turbo-4c74631
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG SP=/opt/venv/lib/python3.12/site-packages

# Primus-Turbo at the head of AMD-AGI/Primus-Turbo#602: FlyDSL DeepEP V1/V2 split with the gfx1250
# V2 correctness fixes (illegal memory access, wrong rows, combine hang). The base image's
# 4c74631e has the V2 backend (#583) without them. Same steps as the base image's Turbo layer.
ARG TURBO_COMMIT=8824b44788b45a8ab5fd6a1d65f02fd11f174090
WORKDIR /workspace/Primus-Turbo
RUN git fetch --no-tags origin "${TURBO_COMMIT}" \
    && git checkout --detach "${TURBO_COMMIT}" \
    && git submodule update --init --recursive \
    && pip uninstall -y primus_turbo \
    && PATH=$(tr : "\n" <<< "$PATH" | grep -v openmpi | paste -sd:) \
       GPU_ARCHS="gfx1250" MAX_JOBS=64 pip install --no-build-isolation --no-deps . \
    && rm -rf build ./*.egg-info \
    && grep -q "${TURBO_COMMIT}" "${SP}/primus_turbo/_build_info.py" \
    && test -d "${SP}/primus_turbo/flydsl/attention/gfx1250"

# Packages the DLRMv4 tree needs on top of the TAS image.
RUN pip install --no-cache-dir "gin-config==0.5.0" "polars-u64-idx==1.33.1"

# Triton 3.8.0+git7ff97e31 (triton-lang/triton main, 2026-09-02) replaces the TAS 3.8.0+git669b31ac,
# which miscompiles some gfx1250 loop-tail stores (VMEM WAR hazard; DLRMv4's
# _add_timestamp_position_embeddings_kernel faults on every rank).
ARG TRITON_COMMIT=7ff97e310935b4a79794878dbc911f9af25d38d9
RUN pip uninstall -y triton \
    && git clone https://github.com/triton-lang/triton.git /tmp/triton \
    && cd /tmp/triton && git checkout "${TRITON_COMMIT}" \
    && pip install --no-cache-dir -r python/requirements.txt \
    && MAX_JOBS=64 pip install --no-cache-dir --no-build-isolation . \
    && if [ -d python/triton_kernels ]; then pip install --no-cache-dir --no-deps python/triton_kernels; fi \
    && cd / && rm -rf /tmp/triton /root/.triton \
    && python -c "import triton; from triton.backends import backends; assert 'amd' in backends, backends; print('triton', triton.__version__)"

# Triton turns on amdgpu expert scheduling for gfx1250 by default; LLVM then inserts too few
# s_wait_alu and Triton kernels fault or produce NaN at random. Default it to off.
# TRITON_HIP_USE_EXPERT_SCHEDULING=1 still turns it back on.
RUN python - <<'EOF'
import os, re, triton
p = os.path.join(os.path.dirname(triton.__file__), "backends", "amd", "compiler.py")
s = open(p).read()
old = '        return knobs.amd.use_expert_scheduling\n    return arch in ["gfx1250"]\n'
new = '        return knobs.amd.use_expert_scheduling\n    return False\n'
assert s.count(old) == 1, "unexpected is_expert_scheduling_enabled() in " + p
open(p, "w").write(s.replace(old, new))
import importlib, triton.backends.amd.compiler as c
importlib.reload(c)
assert c.is_expert_scheduling_enabled("gfx1250") is False
print("expert scheduling default off:", p)
EOF

# fbgemm_gpu rebuilt for gfx1250 wave32 from the TAS image's /workspace/FBGEMM build tree. The TAS
# build sizes host code with kWarpSize 64 against wave32 device code: illegal 64x32 TBE backward
# block and a wrong forward. The three scripts run in order (CPU only, ~10 min).
COPY --from=dlrmv4 docker/mi455/fbgemm_cta_per_row_gfx1250_rebuild.sh \
                   docker/mi455/fbgemm_host_warp32_gfx1250_rebuild.sh \
                   docker/mi455/fbgemm_codegen_warp32_gfx1250_rebuild.sh /fb/
RUN bash /fb/fbgemm_cta_per_row_gfx1250_rebuild.sh \
    && bash /fb/fbgemm_host_warp32_gfx1250_rebuild.sh \
    && bash /fb/fbgemm_codegen_warp32_gfx1250_rebuild.sh

# DLRMv4's gfx1250 FlyDSL HSTU attention kernels are written against FlyDSL 0.2.4 (0.3.x changed
# the rocdl WMMA builders), while Primus-Turbo's FlyDSL kernels need the image's 0.3.4.1. Keep 0.2.4
# out of site-packages; run_dlrmv4.sh puts /opt/flydsl-0.2.4 first on PYTHONPATH.
RUN pip install --no-cache-dir --no-deps --target /opt/flydsl-0.2.4 flydsl==0.2.4 \
    && PYTHONPATH=/opt/flydsl-0.2.4 python -c "import flydsl, sys; assert flydsl.__file__.startswith('/opt/flydsl-0.2.4'), flydsl.__file__" \
    && python -c "import flydsl; assert not flydsl.__file__.startswith('/opt/flydsl-0.2.4')"

# Stack dumps of hung ranks; not imported by training.
RUN pip install --no-cache-dir py-spy

# Workload sources.
RUN rm -rf /workspace/Primus
COPY --from=primus . /workspace/Primus
COPY --from=dlrmv4 . /workspace/recommendation

# fla is not imported here: it pulls in mamba_ssm, which queries the GPU at import time.
RUN python -c "import torch, triton, fbgemm_gpu, torchrec, primus_turbo, gin, polars; \
from triton.backends import backends; assert 'amd' in backends; \
print('torch', torch.__version__, '| triton', triton.__version__, '| fbgemm_gpu', fbgemm_gpu.__version__, \
'| torchrec', torchrec.__version__, '| primus_turbo', primus_turbo.__version__)"

# torch's metadata pins the TAS Triton, so pip treats the Triton swap above as a broken torch and
# would replace torch from PyPI. Every dependency is installed here: skip Primus's runtime
# `pip install -r requirements-megatron.txt` hook.
ENV PRIMUS_SKIP_PIP=1

ARG BASE_IMAGE
ARG PRIMUS_COMMIT=unknown
ARG DLRMV4_COMMIT=unknown
LABEL org.amd.base-image="${BASE_IMAGE}" \
      org.amd.primus.commit="${PRIMUS_COMMIT}" \
      org.amd.primus-turbo.commit="${TURBO_COMMIT}" \
      org.amd.triton.commit="${TRITON_COMMIT}" \
      org.amd.dlrmv4.commit="${DLRMV4_COMMIT}" \
      org.amd.workloads="kimi_k3.1_pro_4L;dlrmv4_yambda_5b"

WORKDIR /workspace/
ENTRYPOINT []
CMD ["/usr/bin/bash"]
