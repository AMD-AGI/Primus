#!/usr/bin/env bash
set -euo pipefail

########################################
# CONFIG
########################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# primus/backends/megatron_bridge -> repo root
PRIMUS_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PRIMUS_THIRD_PARTY="${PRIMUS_ROOT}/third_party"
MEGATRON_BRIDGE_DIR="${PRIMUS_THIRD_PARTY}/megatron-bridge"

TRANSFORMERS_REPO="https://github.com/huggingface/transformers.git"
TRANSFORMERS_BRANCH="v4.57-release"
TRANSFORMERS_DIR="${PRIMUS_THIRD_PARTY}/transformers"

MEGATRON_BRIDGE_TOML_SRC="${PRIMUS_ROOT}/primus/backends/megatron_bridge/patches/pyproject.toml"
MEGATRON_BRIDGE_TOML_DST="${MEGATRON_BRIDGE_DIR}/pyproject.toml"

GPT_PROVIDER_FILE="${MEGATRON_BRIDGE_DIR}/src/megatron/bridge/models/gpt_provider.py"
TRAINING_SETUP_FILE="${MEGATRON_BRIDGE_DIR}/src/megatron/bridge/training/setup.py"

########################################
# 1) Transformers (installed in third_party)
########################################
mkdir -p "$PRIMUS_THIRD_PARTY"
cd "$PRIMUS_THIRD_PARTY"

if [ ! -d "$TRANSFORMERS_DIR/.git" ]; then
  echo "[+] Cloning transformers..."
  git clone "$TRANSFORMERS_REPO" "$TRANSFORMERS_DIR"
else
  echo "[=] transformers already exists"
fi

cd "$TRANSFORMERS_DIR"
git fetch --tags
git checkout "$TRANSFORMERS_BRANCH"

########################################
# 2) Patch transformers modeling_utils.py (idempotent)
########################################
python - <<'PY'
from pathlib import Path

file = Path("src/transformers/modeling_utils.py")
text = file.read_text(encoding="utf-8")

block = """from .pytorch_utils import (  # noqa: F401
    Conv1D,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    id_tensor_storage,
    prune_conv1d_layer,
    prune_layer,
    prune_linear_layer,
)
"""

if block in text:
    print("[=] modeling_utils.py already patched")
else:
    lines = text.splitlines(True)
    insert_at = 0
    for i, ln in enumerate(lines[:350]):
        if ln.startswith("import ") or ln.startswith("from "):
            insert_at = i + 1
    lines.insert(insert_at, "\n" + block + "\n")
    file.write_text("".join(lines), encoding="utf-8")
    print("[+] Patched modeling_utils.py")
PY

########################################
# 3) Install Transformers from source (editable)
########################################
pip install -U pip
pip install -e .

########################################
# 4) Replace Megatron-Bridge pyproject.toml
########################################
echo "[+] Replacing Megatron-Bridge pyproject.toml"

if [ ! -f "$MEGATRON_BRIDGE_TOML_SRC" ]; then
  echo "[ERROR] Source pyproject.toml not found: $MEGATRON_BRIDGE_TOML_SRC"
  exit 1
fi

if [ ! -f "$MEGATRON_BRIDGE_TOML_DST" ]; then
  echo "[ERROR] Destination pyproject.toml not found: $MEGATRON_BRIDGE_TOML_DST"
  exit 1
fi

cp "$MEGATRON_BRIDGE_TOML_SRC" "$MEGATRON_BRIDGE_TOML_DST"
echo "[OK] pyproject.toml replaced"

########################################
# 5) Install remaining packages (any directory)
########################################
cd "$HOME" || cd /

pip install "onnx==1.20.0rc1"
pip install -U nvidia-modelopt
pip install -U nvidia_resiliency_ext

########################################
# 6) Export required env var (current shell + persist)
########################################
export HSA_NO_SCRATCH_RECLAIM=1
echo "[OK] Exported HSA_NO_SCRATCH_RECLAIM=1 (current shell)"

SHELL_RC="$HOME/.bashrc"
if [ -n "${ZSH_VERSION:-}" ]; then
  SHELL_RC="$HOME/.zshrc"
fi

if ! grep -q "^export HSA_NO_SCRATCH_RECLAIM=1" "$SHELL_RC" 2>/dev/null; then
  echo "export HSA_NO_SCRATCH_RECLAIM=1" >> "$SHELL_RC"
  echo "[OK] Persisted HSA_NO_SCRATCH_RECLAIM=1 in $SHELL_RC"
else
  echo "[=] HSA_NO_SCRATCH_RECLAIM already present in $SHELL_RC"
fi

########################################
# 7) Replace cuda_graph_impl -> cuda_graph_scope in Megatron-Bridge
########################################
echo "[+] Replacing cuda_graph_impl -> cuda_graph_scope under ${MEGATRON_BRIDGE_DIR}"

if [ ! -d "$MEGATRON_BRIDGE_DIR" ]; then
  echo "[ERROR] megatron-bridge folder not found at: $MEGATRON_BRIDGE_DIR"
  exit 1
fi

cd "$PRIMUS_THIRD_PARTY"
grep -rlZ "cuda_graph_impl" megatron-bridge/ | xargs -0 sed -i 's/cuda_graph_impl/cuda_graph_scope/g'
echo "[OK] Replaced occurrences in megatron-bridge"

########################################
# 8) Patch gpt_provider.py: use_te_rng_tracker -> True
########################################
python - <<PY
from pathlib import Path

file = Path("${GPT_PROVIDER_FILE}")
if not file.exists():
    raise RuntimeError(f"File not found: {file}")

text = file.read_text(encoding="utf-8")
old = "use_te_rng_tracker: bool = False"
new = "use_te_rng_tracker: bool = True"

if new in text:
    print("[=] use_te_rng_tracker already enabled")
elif old in text:
    file.write_text(text.replace(old, new), encoding="utf-8")
    print("[+] Enabled use_te_rng_tracker = True")
else:
    raise RuntimeError("Expected line not found: 'use_te_rng_tracker: bool = False'")
PY

########################################
# 9) Comment out line 25 and lines 115-118 in training/setup.py
########################################
python - <<PY
from pathlib import Path

file = Path("${TRAINING_SETUP_FILE}")
if not file.exists():
    raise RuntimeError(f"File not found: {file}")

lines = file.read_text(encoding="utf-8").splitlines()

def comment_line(idx0: int):
    if idx0 < 0 or idx0 >= len(lines):
        raise RuntimeError(f"Line {idx0+1} out of range (file has {len(lines)} lines)")
    if not lines[idx0].lstrip().startswith("#"):
        lines[idx0] = "# " + lines[idx0]

for ln in [25, 115, 116, 117, 118]:
    comment_line(ln - 1)

file.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("[+] Commented out lines 25 and 115-118 in training/setup.py")
PY

########################################
# 10) Install megatron-bridge (editable)
########################################
echo "[+] Installing megatron-bridge editable"
cd "$MEGATRON_BRIDGE_DIR"
pip install -e .
echo "[OK] Installed megatron-bridge editable"

echo "========================================"
echo "Setup complete."
echo "========================================"

