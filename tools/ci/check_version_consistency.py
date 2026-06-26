###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fail CI when version/commit pins drift between ci.yaml and the Dockerfile.

Checks that ci.yaml's BASE_IMAGE env matches the Dockerfile's ARG BASE_IMAGE
default (else a local `docker build` uses a different base than CI), and that
primus.__version__ is a valid PEP 440 version. Stdlib-only, runs in the lint job.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CI_YAML = ROOT / ".github/workflows/ci.yaml"
DOCKERFILE = ROOT / ".github/workflows/docker/Dockerfile"
INIT_PY = ROOT / "primus/__init__.py"

# Deliberately permissive PEP 440 subset (release + optional pre/post/dev).
PEP440 = re.compile(r"^\d+(\.\d+)*((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?$")


def _extract(text, pattern, what, where):
    match = re.search(pattern, text, re.MULTILINE)
    if match is None:
        sys.exit(f"ERROR: could not find {what} in {where}")
    return match.group(1).strip()


def main():
    errors = []

    ci_base = _extract(CI_YAML.read_text(), r"^\s*BASE_IMAGE:\s*(\S+)", "BASE_IMAGE env", CI_YAML)
    docker_base = _extract(
        DOCKERFILE.read_text(), r"^ARG\s+BASE_IMAGE=(\S+)", "ARG BASE_IMAGE default", DOCKERFILE
    )
    if ci_base != docker_base:
        errors.append(
            f"BASE_IMAGE drift: ci.yaml={ci_base!r} vs Dockerfile default={docker_base!r}. "
            "Keep ci.yaml `env` and the Dockerfile `ARG` default in sync."
        )

    version = _extract(INIT_PY.read_text(), r'__version__\s*=\s*["\']([^"\']+)["\']', "__version__", INIT_PY)
    if not PEP440.match(version):
        errors.append(f"primus.__version__={version!r} is not a valid PEP 440 version.")

    if errors:
        print("Version consistency check FAILED:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print(f"Version consistency OK (BASE_IMAGE={ci_base}, primus.__version__={version}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
