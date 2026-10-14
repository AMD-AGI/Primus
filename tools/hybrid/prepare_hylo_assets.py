#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Download and convert the public assets used by native HyLo training.

Authentication is read only from ``HF_TOKEN``. The token is never written by
this tool. Both Llama checkpoints are converted to TP=PP=1 legacy Megatron
format, which is the topology supported by the native distillation runtime.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _download(repo: str, target: Path, token: str) -> Path:
    from huggingface_hub import snapshot_download

    target.mkdir(parents=True, exist_ok=True)
    return Path(snapshot_download(repo, local_dir=target, token=token))


def _convert(hf_dir: Path, output: Path) -> None:
    from primus.backends.megatron.checkpoint import native_convert_common as common
    from primus.backends.megatron.checkpoint import native_loader_llama

    common.ensure_megatron_on_path()
    common.apply_convert_patches(module_name="checkpoint_convert")
    native_loader_llama.convert(
        str(hf_dir),
        str(output),
        dtype="bf16",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        expert_parallel_size=1,
    )


def _convert_isolated(hf_dir: Path, output: Path) -> None:
    """Convert in a fresh process because Megatron globals initialize once."""
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--convert-source",
            str(hf_dir),
            "--convert-output",
            str(output),
        ],
        check=True,
        env=os.environ,
    )


def _export_dataset(output: Path, token: str) -> None:
    from datasets import load_dataset

    output.parent.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("JunxiongWang/sftdatasetv3", split="train", token=token)
    dataset.to_json(str(output), orient="records", lines=True, force_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("./data/hylo"))
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument("--skip-dataset", action="store_true")
    parser.add_argument("--convert-source", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--convert-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.convert_source or args.convert_output:
        if not args.convert_source or not args.convert_output:
            parser.error("--convert-source and --convert-output must be used together")
        _convert(args.convert_source, args.convert_output)
        return
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN must be exported before running this tool")

    hf_root = args.root / "hf"
    megatron_root = args.root / "megatron"
    repos = {
        "llama32-1b": "meta-llama/Llama-3.2-1B-Instruct",
        "llama31-8b": "meta-llama/Llama-3.1-8B-Instruct",
    }
    for name, repo in repos.items():
        hf_dir = hf_root / name
        if not args.skip_download:
            print(f"Downloading {repo} to {hf_dir}")
            _download(repo, hf_dir, token)
        if not args.skip_convert:
            print(f"Converting {hf_dir} to {megatron_root / name}")
            _convert_isolated(hf_dir, megatron_root / name)
    if not args.skip_dataset:
        dataset_path = args.root / "hylo-sft.jsonl"
        print(f"Exporting JunxiongWang/sftdatasetv3 to {dataset_path}")
        _export_dataset(dataset_path, token)


if __name__ == "__main__":
    main()
