###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared helpers for the release-docs tooling: snapshot IO and the parsers for
the artifacts a training image ships under `/workspace/.manifest/`.

Stdlib-only, because `release_notes.py check` runs in the lint job, which
installs nothing but pre-commit (same constraint as
`tools/ci/check_version_consistency.py`).
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(__file__).resolve().parent / "data"
RELEASE_NOTES = ROOT / "docs/01-getting-started/release-notes.md"
DOCKER_RELEASE_DIR = ROOT / ".github/workflows/docker-release"

SCHEMA_VERSION = 1

# The two published image families. `dockerfile_stem` is the basename convention
# in .github/workflows/docker-release/, `manifest_dockerfile` the name the image
# uses for its baked-in copy (the JAX image calls it docker-build-recipe.txt).
FAMILIES = {
    "primus": {
        "image": "rocm/primus:{version}",
        "dockerfile_stem": "Dockerfile.primus-{version}",
        "backends": "Megatron-LM, TorchTitan, and Megatron Bridge backends.",
    },
    "jax": {
        "image": "rocm/jax-training:maxtext-{version}",
        "dockerfile_stem": "Dockerfile.jax-{version}",
        "backends": "MaxText (JAX) backend.",
    },
}

MANIFEST_DIR = "/workspace/.manifest"
MANIFEST_DOCKERFILE_NAMES = ("Dockerfile", "docker-build-recipe.txt")


def canon(name):
    """Canonical package key: PEP 503-ish normalisation, as pip itself compares."""
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def family_for_image(image):
    """Infer family from an image reference, so callers need not pass --family."""
    if "jax-training" in image or "maxtext" in image:
        return "jax"
    if "primus" in image:
        return "primus"
    return None


def image_for(family, version):
    return FAMILIES[family]["image"].format(version=version)


def dockerfile_for(family, version):
    return DOCKER_RELEASE_DIR / FAMILIES[family]["dockerfile_stem"].format(version=version)


def parse_pip_list(text):
    """Parse the column-formatted `pip list` the images ship.

        Package                        Version           Editable project location
        ------------------------------ ----------------- -------------------------
        absl-py                        2.5.0

    Returns {canonical_name: version}. Package names never contain spaces, so a
    plain split is sufficient and tolerates the optional third column.
    """
    packages = {}
    for line in text.splitlines():
        line = line.rstrip()
        if not line or line.startswith("-") or line.startswith("Package "):
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        packages[canon(fields[0])] = fields[1]
    return packages


def parse_env(text):
    """Parse the `env` dump. Values may contain '=', keys never do."""
    env = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key and not key[0].isdigit():
            env[key.strip()] = value.strip()
    return env


def parse_dpkg(text):
    """Parse `dpkg -l` rows into {package: version}, keeping only installed ones."""
    packages = {}
    for line in text.splitlines():
        if not line.startswith("ii"):
            continue
        fields = line.split()
        if len(fields) >= 3:
            packages[fields[1].split(":")[0]] = fields[2]
    return packages


def parse_version_header(text, macros):
    """Pull integer/hex version macros out of a ROCm C header.

    Native libraries are not pip packages, so hipBLASLt and RCCL versions come
    from `#define` lines instead. Returns {macro_suffix: value}.
    """
    found = {}
    for macro in macros:
        match = re.search(rf"#define\s+{re.escape(macro)}\s+(\S+)", text)
        if match:
            found[macro] = match.group(1).strip()
    return found


def snapshot_path(version, family):
    return DATA_DIR / f"{version}-{family}.json"


def save_snapshot(snapshot):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = snapshot_path(snapshot["version"], snapshot["family"])
    path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    return path


def read_snapshot_file(path):
    """Load one snapshot, naming the file if it is unreadable.

    These are consumed by a lint-job check, where a bare JSONDecodeError
    traceback would say nothing about which snapshot needs regenerating.
    """
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise SystemExit(
            f"ERROR: {path} is not valid JSON ({error}). Re-run probe_image.py for that release."
        ) from error


def load_snapshot(version, family):
    path = snapshot_path(version, family)
    if not path.exists():
        return None
    return read_snapshot_file(path)


def available_snapshots():
    """Every committed snapshot, newest version first."""
    if not DATA_DIR.exists():
        return []
    found = []
    for path in sorted(DATA_DIR.glob("*.json")):
        data = read_snapshot_file(path)
        if "version" not in data or "family" not in data:
            raise SystemExit(f"ERROR: {path} is missing 'version'/'family'; re-run probe_image.py.")
        found.append((data["version"], data["family"], data))
    return sorted(found, key=lambda item: (version_key(item[0]), item[1]), reverse=True)


def version_key(version):
    """Sort key for 'v26.7' / 'v26.5.1' style versions."""
    parts = re.findall(r"\d+", version)
    return tuple(int(part) for part in parts)
