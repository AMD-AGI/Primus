###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Per-PR, CPU-only numeric check of the Granite model against HF.

A tiny Granite GPTModel (GraniteTransformerConfig + GraniteTransformerLayer, local
spec) is loaded with random HF-layout weights through the converter mapping
(``native_loader_granite.build_mcore_state_dict``) and compared with the real
``transformers`` ``GraniteForCausalLM`` (when the installed transformers has it) and with an
independent plain-torch transcription of its forward. This checks all four multipliers
numerically: embedding_multiplier, attention_multiplier, residual_multiplier and
logits_scaling.

Scope: the model is built with Megatron's local attention and norms, because TE needs a
GPU. The production TE path (TE attention softmax scale, TE layernorm-linear, the TE key
names the converter writes) is checked on a GPU by the l3-logprob-parity gate with the
released weights.

It runs in a subprocess, because the CPU set-up (gloo process group, CUDA device
stubs, TE CPU-offload context disabled) must not leak into other tests. All GPUs are
hidden from the subprocess; no megatron.training is needed.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[5]
WORKER = Path(__file__).with_name("_granite_cpu_parity_worker.py")


def test_granite_gptmodel_matches_hf_reference_on_cpu():
    pytest.importorskip("torch")
    pytest.importorskip("megatron.core")
    env = dict(os.environ)
    paths = [str(ROOT), str(ROOT / "third_party" / "Megatron-LM")]
    env["PYTHONPATH"] = os.pathsep.join(paths + [env.get("PYTHONPATH", "")])
    # Hide every GPU (CUDA and ROCm), so the test runs the same on GPU and CPU-only hosts.
    for var in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        env[var] = ""
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(key, None)
    proc = subprocess.run(
        [sys.executable, str(WORKER)], cwd=ROOT, env=env, capture_output=True, text=True, timeout=600
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out[-4000:]
    lines = [l for l in proc.stdout.splitlines() if l.startswith("GRANITE_PARITY_RESULT=")]
    assert len(lines) == 1, out[-4000:]
    result = json.loads(lines[0].split("=", 1)[1])
    # fp32 on CPU: a correct port differs by ~1e-6; a wrong multiplier by O(1).
    tol = 1e-4 * max(1.0, result["ref_scale"])
    assert result["max_abs_err_plain_torch"] <= tol, result
    if result["transformers_available"]:
        assert result["max_abs_err_transformers"] <= tol, result
