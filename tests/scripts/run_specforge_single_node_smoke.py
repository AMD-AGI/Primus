#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
"""Run a bounded 200-step online smoke in a single-node-enabled Primus runtime."""

import argparse
import ast
import hashlib
import json
import math
import os
import re
import subprocess
from pathlib import Path

STEPS = 200
METRICS = ("train/loss", "train/acc", "train/grad_norm")


def validate_run(run_dir):
    control = run_dir / "control"
    for name in ("consumer.done", "inference.done"):
        if (control / name).read_text(encoding="utf-8").strip() != "0":
            raise ValueError(f"Unsuccessful role status: {name}")
    text = (control / "consumer.log").read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    observations = {}
    for match in re.finditer(r"\bstep\s+(\d+):\s*(\{[^\n]*\})", text):
        step = int(match[1])
        if step in observations:
            raise ValueError(f"Duplicate metric record at step {step}")
        metrics = ast.literal_eval(match[2])
        for key in METRICS:
            if key not in metrics or not math.isfinite(float(metrics[key])):
                raise ValueError(f"Missing/non-finite {key} at step {step}")
        observations[step] = {key: float(metrics[key]) for key in METRICS}
    expected = set(range(5, STEPS + 1, 5))
    if set(observations) != expected:
        raise ValueError(
            f"Expected metrics at every fifth step through 200; missing={sorted(expected - observations.keys())}, unexpected={sorted(observations.keys() - expected)}"
        )
    checkpoints = []
    for shared in (run_dir / "output").glob("**/*-step200/training_state.pt"):
        rank = shared.with_name("training_state_rank0.pt")
        if shared.stat().st_size > 0 and rank.is_file() and rank.stat().st_size > 0:
            checkpoints.append(str(shared.parent.relative_to(run_dir)))
    if not checkpoints:
        raise ValueError("Missing nonempty step-200 shared/rank-0 checkpoint files")
    window = [metrics for step, metrics in observations.items() if 150 < step <= STEPS]
    return {
        "step": STEPS,
        "final_metrics": observations[STEPS],
        "final_window_observations": len(window),
        "final_window_mean": {key: sum(row[key] for row in window) / len(window) for key in METRICS},
        "checkpoints": checkpoints,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primus-root", type=Path, default=Path("/opt/primus"))
    parser.add_argument("--data", type=Path, required=True, help="Frozen conversation JSONL")
    parser.add_argument("--data-sha256", required=True, help="Recorded reference checksum, not regenerated")
    parser.add_argument("--run-dir", type=Path, required=True, help="New directory; never reused")
    parser.add_argument("--timeout-seconds", type=int, default=5400)
    parser.add_argument(
        "--manifest", type=Path, default=Path(__file__).with_name("specforge_smoke_fixture.json")
    )
    parser.add_argument(
        "--baseline", type=Path, help="Frozen calibration baseline; omitted only for reference collection"
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    root, data, run_dir = args.primus_root.resolve(), args.data.resolve(), args.run_dir.resolve()
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", args.data_sha256):
        parser.error("--data-sha256 must contain 64 hex characters")
    digest = hashlib.sha256()
    with data.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != args.data_sha256.lower():
        parser.error("Dataset SHA-256 differs from the recorded reference")
    if digest.hexdigest() != manifest["dataset"]["fixture_sha256"]:
        parser.error("Dataset differs from the canonical fixture manifest")
    supervisor = root / "primus/backends/specforge/online_supervisor.py"
    if "def _run_single_node(" not in supervisor.read_text(encoding="utf-8"):
        parser.error("This runtime needs the single-node enablement from Primus PR #1421")
    # Refuse stale control, logs, and checkpoints before launching anything.
    run_dir.mkdir(parents=True, exist_ok=False)
    # Resolve only the exact predownloaded snapshot; never follow a mutable Hub ref.
    import torch
    import yaml
    from huggingface_hub import snapshot_download

    devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if len(devices) != 2 or any("MI355X" not in device for device in devices):
        parser.error("The canonical reference requires exactly two visible MI355X GPUs")

    model_path = snapshot_download(
        manifest["model"]["repository"], revision=manifest["model"]["revision"], local_files_only=True
    )
    recipe_path = root / "examples/specforge/configs/qwen3.5-4b-dflash-online.yaml"
    recipe_bytes = recipe_path.read_bytes()
    recipe = yaml.safe_load(recipe_bytes)
    overrides = recipe["modules"]["pre_trainer"]["overrides"]
    specforge_root = Path("/workspace/SpecForge")
    specforge_config = (
        specforge_root / "examples/configs/online/disaggregated/external/qwen3.5-4b-dflash-amd.yaml"
    )
    overrides["specforge_root"] = str(specforge_root)
    overrides["specforge_config"] = str(specforge_config)
    overrides["specforge_online"]["target_model_path"] = model_path
    overrides["specforge_overrides"].update(
        {
            "model.target_model_path": model_path,
            "training.seed": manifest["training"]["seed"],
            "training.prompt_seed": manifest["training"]["prompt_seed"],
            "training.log_interval": 5,
            "data.cache_dir": str(run_dir / "dataset-cache"),
        }
    )
    config_path = run_dir / "recipe.yaml"
    config_path.write_text(yaml.safe_dump(recipe, sort_keys=False), encoding="utf-8")
    env = dict(os.environ)
    env.update(
        PRIMUS_SPECFORGE_SINGLE_NODE="true",
        CAPTURE_NNODES="1",
        TRAINER_NNODES="1",
        SERVER_COUNT="1",
        SERVER_TP="1",
        SERVER_GPUS="0",
        TRAINER_GPUS="1",
        NPROC_PER_NODE="1",
        MAX_STEPS=str(STEPS),
        RUN_ID=run_dir.name,
        RUN_ROOT=str(run_dir / "control"),
        CONSUMER_STATE_DIR=str(run_dir / "consumer-state"),
        OUTPUT_DIR=str(run_dir / "output"),
        PRIMUS_WORKSPACE=str(run_dir / "primus"),
        TRAIN_DATA_PATH=str(data),
        TARGET_MODEL=model_path,
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
    )
    metadata = {
        "data_sha256": digest.hexdigest(),
        "steps": STEPS,
        "timeout_seconds": args.timeout_seconds,
        "primus_root": str(root),
        "manifest": manifest,
        "recipe_sha256": hashlib.sha256(recipe_bytes).hexdigest(),
        "specforge_config_sha256": hashlib.sha256(specforge_config.read_bytes()).hexdigest(),
        "draft_config_sha256": hashlib.sha256(
            (specforge_root / "configs/qwen3.5-4b-dflash.json").read_bytes()
        ).hexdigest(),
        "model_snapshot": model_path,
        "runtime": {"torch": torch.__version__, "hip": torch.version.hip, "devices": devices},
    }
    identity = {
        "manifest": manifest,
        "recipe_sha256": metadata["recipe_sha256"],
        "specforge_config_sha256": metadata["specforge_config_sha256"],
        "draft_config_sha256": metadata["draft_config_sha256"],
        "runtime": metadata["runtime"],
    }
    metadata["reference_identity"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if args.baseline:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        if baseline["reference_identity"] != metadata["reference_identity"]:
            parser.error("Baseline inputs differ from this run")
    (run_dir / "inputs.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    # GNU timeout sends TERM to the command group and bounds teardown with KILL.
    # The supervisor handles TERM and cleans up its separately launched sidecars.
    command = [
        "timeout",
        "--signal=TERM",
        "--kill-after=60s",
        f"{args.timeout_seconds}s",
        "bash",
        "./runner/primus-cli",
        "direct",
        "--log_file",
        str(run_dir / "launcher.log"),
        "--",
        "train",
        "pretrain",
        "--config",
        str(config_path),
    ]
    subprocess.run(command, cwd=root, env=env, check=True)
    summary = validate_run(run_dir)
    summary["run_id"] = run_dir.name
    summary["reference_identity"] = metadata["reference_identity"]
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.baseline:
        from calibrate_specforge_smoke import compare

        compare(summary, baseline)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
