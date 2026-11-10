#!/usr/bin/env python3
"""Resolve image pins and build the same SpecForge runtime locally and in CI."""

import argparse
import ast
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = Path(__file__).with_name("images.json")


def image_config(variant):
    pins = json.loads(MANIFEST.read_text(encoding="utf-8"))
    # Read Primus's existing version source without importing its dependencies.
    module = ast.parse((ROOT / "primus/__init__.py").read_text(encoding="utf-8"))
    version = next(
        ast.literal_eval(node.value)
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets)
    )
    digest = pins["variants"][variant]
    base_tag = f"v{pins['sglang_version']}-{pins['rocm_tag']}-{variant}"
    return {
        "base_image": f"{pins['sglang_repository']}:{base_tag}@{digest}",
        "specforge_ref": pins["specforge_ref"],
        "specforge_version": pins["specforge_version"],
        "sglang_version": pins["sglang_version"],
        "version_tag": (
            f"v{version}-sglang{pins['sglang_version']}-specforge{pins['specforge_version']}"
            f"-g{pins['specforge_ref'][:12]}-{variant}"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=json.loads(MANIFEST.read_text())["variants"])
    parser.add_argument("--field", help="Print one resolved field instead of JSON")
    parser.add_argument("--build", action="store_true", help="Build with Docker (requires BuildKit)")
    parser.add_argument("--tag", help="Override the local image name/tag")
    parser.add_argument("--revision", help="Source revision for the OCI image label")
    args = parser.parse_args()
    config = image_config(args.variant)
    if not args.build:
        print(config[args.field] if args.field else json.dumps(config, indent=2))
        return

    tag = args.tag or f"primus-specforge:{config['version_tag']}"
    revision = (
        args.revision or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    )
    command = ["docker", "build", "--network=host", "-f", "examples/specforge/docker/Dockerfile"]
    for field in ("base_image", "specforge_ref", "specforge_version", "sglang_version"):
        command.extend(["--build-arg", f"{field.upper()}={config[field]}"])
    command.extend(
        [
            "--label",
            f"org.opencontainers.image.revision={revision}",
            "--label",
            "org.opencontainers.image.source=https://github.com/AMD-AGI/Primus",
            "-t",
            tag,
            "-",
        ]
    )
    # An allowlisted tar stream avoids host-specific Docker daemon mount paths.
    context = subprocess.Popen(
        [
            "tar",
            "-cf",
            "-",
            "pyproject.toml",
            "LICENSE",
            "README.md",
            "primus_cli.py",
            "primus",
            "runner",
            "examples/specforge",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
    )
    try:
        subprocess.run(command, stdin=context.stdout, env={**os.environ, "DOCKER_BUILDKIT": "1"}, check=True)
        context.stdout.close()
        if context.wait() != 0:
            raise RuntimeError("Failed to create Docker build context")
    finally:
        context.stdout.close()
        if context.poll() is None:
            context.terminate()
        context.wait()
    print(f"Built {tag}")


if __name__ == "__main__":
    main()
