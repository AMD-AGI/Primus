###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MEGATRON_REQUIREMENTS_HOOK = (
    ROOT / "runner" / "helpers" / "hooks" / "train" / "pretrain" / "megatron" / "00_install_requirements.sh"
)
POSTTRAIN_REQUIREMENTS_HOOK = (
    ROOT / "runner" / "helpers" / "hooks" / "train" / "posttrain" / "megatron" / "00_install_requirements.sh"
)


def test_primus_skip_pip_makes_megatron_hook_clean_noop(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    pip_marker = tmp_path / "pip-was-called"
    fake_pip = fake_bin / "pip"
    fake_pip.write_text(
        f"#!/bin/bash\ntouch '{pip_marker}'\nexit 99\n",
        encoding="utf-8",
    )
    fake_pip.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "PRIMUS_SKIP_PIP": "1",
        }
    )
    result = subprocess.run(
        ["/bin/bash", str(MEGATRON_REQUIREMENTS_HOOK), "--data_path", str(tmp_path / "data")],
        cwd=ROOT,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "PRIMUS_SKIP_PIP=1" in result.stdout
    assert not pip_marker.exists()
    assert not (tmp_path / "data").exists()


# The posttrain hook skips the Megatron-Bridge install when the native converter
# will run (01_convert_checkpoints.py decides). Mistral-7B-v0.3's config, cut down.
_MISTRAL = {
    "architectures": ["MistralForCausalLM"],
    "model_type": "mistral",
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "num_key_value_heads": 8,
    "max_position_embeddings": 32768,
    "rms_norm_eps": 1e-05,
    "rope_theta": 1000000.0,
    "sliding_window": None,
    "vocab_size": 32768,
}


@pytest.mark.parametrize(
    "hf_config,native,want",
    [
        (_MISTRAL, None, "skip"),
        (dict(_MISTRAL, sliding_window=4096), None, "install"),  # Mistral-7B-v0.1: Bridge fallback
        (_MISTRAL, "false", "install"),  # Bridge forced
        (dict(_MISTRAL, sliding_window=4096), "true", "skip"),  # native forced (the hook then stops)
    ],
)
def test_posttrain_hook_installs_bridge_deps_only_for_bridge(tmp_path, hf_config, native, want):
    pytest.importorskip("yaml")
    hf = tmp_path / "hf"
    hf.mkdir()
    (hf / "config.json").write_text(json.dumps(hf_config))
    native_key = f"      native_ckpt_convert: {native}\n" if native else ""
    config = tmp_path / "exp.yaml"
    config.write_text(
        "work_group: amd\nuser_name: root\nexp_name: hook\n"
        f"workspace: {tmp_path / 'ws'}\n"
        "modules:\n  post_trainer:\n    framework: megatron\n    config: sft_trainer.yaml\n"
        "    model: mistral_7B_v0_3.yaml\n    overrides:\n      stage: sft\n"
        f"      hf_path: {hf}\n" + native_key
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    pip_log = tmp_path / "pip.log"
    (fake_bin / "pip").write_text(f"#!/bin/bash\necho \"$*\" >> '{pip_log}'\n")
    (fake_bin / "pip").chmod(0o755)
    # the interpreter (and environment) that runs the tests
    (fake_bin / "python3").write_text(f'#!/bin/bash\nexec "{sys.executable}" "$@"\n')
    (fake_bin / "python3").chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k != "PRIMUS_SKIP_PIP"}
    env.update(PATH=f"{fake_bin}:{env['PATH']}", DATA_PATH=str(tmp_path / "data"), PYTHONPATH=str(ROOT))
    result = subprocess.run(
        ["/bin/bash", str(POSTTRAIN_REQUIREMENTS_HOOK), "--config", str(config)],
        cwd=ROOT,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    if want == "skip":
        assert "skipping Megatron-Bridge dependency install" in result.stdout
        assert not pip_log.exists()
    else:
        assert "Installing Megatron-Bridge" in result.stdout
        assert "transformers==4.57.6" in pip_log.read_text()
