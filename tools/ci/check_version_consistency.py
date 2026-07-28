###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fail CI on cross-file config drift: a value duplicated across files that fell
out of sync. Runs in the lint job. Covers BASE_IMAGE (ci.yaml vs Dockerfile
ARG), primus.__version__ (PEP 440), pyproject deps vs requirements.txt, action
SHA-pinning, and workflow python-version. Stdlib-only.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CI_YAML = ROOT / ".github/workflows/ci.yaml"
DOCKERFILE = ROOT / ".github/workflows/docker/Dockerfile"
INIT_PY = ROOT / "primus/__init__.py"
PYPROJECT = ROOT / "pyproject.toml"
REQUIREMENTS = ROOT / "requirements.txt"
WORKFLOWS = sorted((ROOT / ".github/workflows").glob("*.y*ml"))

PEP440 = re.compile(r"^\d+(\.\d+)*((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?$")
SHA40 = re.compile(r"^[0-9a-f]{40}$")
NAME_SPEC = re.compile(r"^([A-Za-z0-9._-]+)\s*(.*)$")


def _canon(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def _find(text, pattern, what, where):
    match = re.search(pattern, text, re.MULTILINE)
    if match is None:
        sys.exit(f"ERROR: could not find {what} in {where}")
    return match.group(1).strip()


def _parse_dep(line):
    line = line.split("#", 1)[0].strip()
    match = NAME_SPEC.match(line) if line else None
    return (_canon(match.group(1)), match.group(2).replace(" ", "")) if match else None


def check_base_image(errors):
    ci = _find(CI_YAML.read_text(), r"^\s*BASE_IMAGE:\s*(\S+)", "BASE_IMAGE env", CI_YAML)
    docker = _find(DOCKERFILE.read_text(), r"^ARG\s+BASE_IMAGE=(\S+)", "ARG BASE_IMAGE default", DOCKERFILE)
    if ci != docker:
        errors.append(f"BASE_IMAGE drift: ci.yaml={ci!r} vs Dockerfile default={docker!r}.")


def check_version(errors):
    version = _find(INIT_PY.read_text(), r'__version__\s*=\s*["\']([^"\']+)["\']', "__version__", INIT_PY)
    if not PEP440.match(version):
        errors.append(f"primus.__version__={version!r} is not a valid PEP 440 version.")


def check_deps(errors):
    block = re.search(r"^dependencies\s*=\s*\[(.*?)\]", PYPROJECT.read_text(), re.S | re.M)
    if block is None:
        errors.append("could not find [project].dependencies in pyproject.toml")
        return
    pyproject_deps = dict(filter(None, (_parse_dep(d) for d in re.findall(r'"([^"]+)"', block.group(1)))))
    req = dict(filter(None, (_parse_dep(line) for line in REQUIREMENTS.read_text().splitlines())))
    # pyproject deps must be a subset of requirements with matching specifiers;
    # requirements may carry extra dev/CI-only entries (e.g. hip-python).
    for name, spec in sorted(pyproject_deps.items()):
        if name not in req:
            errors.append(f"{name!r} is in pyproject.toml but missing from requirements.txt.")
        elif req[name] != spec:
            errors.append(f"spec drift for {name!r}: pyproject={spec!r} vs requirements={req[name]!r}.")


def check_pinned_actions(errors):
    for wf in WORKFLOWS:
        for raw in re.findall(r"uses:\s*(\S+)", wf.read_text()):
            uses = raw.strip().strip('"').strip("'")
            if uses.startswith("./"):
                continue
            if "@" not in uses:
                errors.append(f"{wf.name}: action {uses!r} is not pinned (no @ref).")
            elif not SHA40.match(uses.rsplit("@", 1)[1]):
                errors.append(f"{wf.name}: action {uses!r} is not pinned to a 40-hex commit SHA.")


def check_python_versions(errors):
    requires = _find(
        PYPROJECT.read_text(), r'requires-python\s*=\s*["\']([^"\']+)["\']', "requires-python", PYPROJECT
    )
    floor = re.search(r">=\s*(\d+)\.(\d+)", requires)
    min_ver = (int(floor.group(1)), int(floor.group(2))) if floor else None
    versions = set()
    for wf in WORKFLOWS:
        versions.update(re.findall(r'python-version:\s*\[?\s*["\'](\d+\.\d+)["\']', wf.read_text()))
    if len(versions) > 1:
        errors.append(f"workflow python-version values disagree: {sorted(versions)}.")
    for v in versions if min_ver else []:
        if tuple(int(x) for x in v.split(".")) < min_ver:
            errors.append(
                f"workflow python-version {v} is below requires-python >={min_ver[0]}.{min_ver[1]}."
            )


def main():
    errors = []
    check_base_image(errors)
    check_version(errors)
    check_deps(errors)
    check_pinned_actions(errors)
    check_python_versions(errors)
    if errors:
        print("CI consistency check FAILED:")
        for err in errors:
            print(f"  - {err}")
        return 1
    print("CI consistency OK (base image, version, deps, action pinning, python-version).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
