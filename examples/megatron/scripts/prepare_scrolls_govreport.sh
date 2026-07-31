#!/usr/bin/env bash
# Download and unpack SCROLLS GovReport for native Megatron SFT (Qwen3 LoRA).
#
# Produces:
#   ${SCROLLS_GOVREPORT_DIR}/train.jsonl
#   ${SCROLLS_GOVREPORT_DIR}/validation.jsonl
#
# Usage:
#   export SCROLLS_GOVREPORT_DIR=/data/scrolls_govreport
#   bash examples/megatron/scripts/prepare_scrolls_govreport.sh
#
# Then train with:
#   examples/megatron/configs/MI355X/qwen3_235B_A22B-BF16-lora-govreport.yaml
#
# Tokenization (Qwen chat template + pack to 8192) runs inside Primus on first
# training launch; cache under $HF_DATASETS_CACHE/primus_packed/.

set -euo pipefail

SCROLLS_GOVREPORT_DIR="${SCROLLS_GOVREPORT_DIR:-/data/scrolls_govreport}"
GOVREPORT_ZIP_URL="${GOVREPORT_ZIP_URL:-https://huggingface.co/datasets/tau/scrolls/resolve/main/gov_report.zip}"

mkdir -p "${SCROLLS_GOVREPORT_DIR}"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

ZIP_PATH="${WORK_DIR}/gov_report.zip"
echo "[govreport] Downloading ${GOVREPORT_ZIP_URL}"
if command -v curl >/dev/null 2>&1; then
  curl -fsSL -o "${ZIP_PATH}" "${GOVREPORT_ZIP_URL}"
elif command -v wget >/dev/null 2>&1; then
  wget -q -O "${ZIP_PATH}" "${GOVREPORT_ZIP_URL}"
else
  echo "Need curl or wget to download GovReport zip." >&2
  exit 1
fi

echo "[govreport] Unpacking and writing JSONL (stdlib zipfile; no unzip required)"
python3 - <<'PY' "${ZIP_PATH}" "${SCROLLS_GOVREPORT_DIR}"
import json
import shutil
import sys
import zipfile
from pathlib import Path

zip_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

extract_root = zip_path.parent / "extract"
extract_root.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(extract_root)
print(f"[govreport] Extracted archive to {extract_root}")

def find_split_files(root: Path, split: str):
    names = [f"{split}.jsonl", f"{split}.json", f"{split}_data.jsonl"]
    for path in root.rglob("*"):
        if path.is_file() and path.name in names:
            return path
    return None

def copy_as_jsonl(src: Path, dst: Path):
    if src.suffix == ".jsonl":
        shutil.copy2(src, dst)
        return
    with open(src, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(f"Expected list in {src}, got {type(data)}")
    with open(dst, "w", encoding="utf-8") as out:
        for row in data:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

for split in ("train", "validation"):
    found = find_split_files(extract_root, split)
    if found is None:
        raise SystemExit(
            f"Could not find {split}.jsonl/json under {extract_root}. "
            "Inspect the zip layout and copy files manually."
        )
    dst = out_dir / f"{split}.jsonl"
    copy_as_jsonl(found, dst)
    print(f"[govreport] Wrote {dst} ({found})")

print(f"[govreport] Ready: {out_dir}")
PY

echo "[govreport] Done. Set SCROLLS_GOVREPORT_DIR=${SCROLLS_GOVREPORT_DIR} and launch posttrain."
