###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Read a published training image and write a committed JSON snapshot of its
software stack, so the release notes are rendered from machine-extracted facts
instead of hand-transcribed ones.

Everything comes out of one `docker run`: the artifacts the image ships under
/workspace/.manifest/ (full `pip list`, `dpkg -l`, baked env, build tag, and the
Dockerfile it was built from), the native library versions that are not pip
packages (hipBLASLt, RCCL, read from ROCm headers), and the git HEAD of every
checkout under /workspace/ -- which is where the per-family build commit comes
from when the Dockerfile pins PRIMUS_BRANCH=main rather than a commit.

Also verifies the in-repo release Dockerfile against the image's baked-in copy.
The release notes claim v26.4-v26.6 were "cross-checked against the release
Dockerfiles" by hand; this makes that automatic and catches a committed
Dockerfile that is not what the image was actually built from.

Usage:
    python tools/release_docs/probe_image.py --version v26.7 --family jax
    python tools/release_docs/probe_image.py --image rocm/primus:v26.7
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common as C  # noqa: E402

# One shell script, one container start. Sections are delimited so a single
# stdout capture yields every artifact.
DELIM = "@@@RELEASE_DOCS_SECTION:"

PROBE_SCRIPT = r"""
set -u
emit() { printf '%s%s@@@\n' "@@@RELEASE_DOCS_SECTION:" "$1"; }

emit requirements
cat /workspace/.manifest/requirements.txt 2>/dev/null

emit pip_live
pip list 2>/dev/null

emit dpkg
cat /workspace/.manifest/dpkg-list.txt 2>/dev/null

emit env
cat /workspace/.manifest/env.txt 2>/dev/null

emit manifest_version
cat /workspace/.manifest/training_docker_version 2>/dev/null

emit dockerfile_name
for candidate in Dockerfile docker-build-recipe.txt; do
    if [ -f "/workspace/.manifest/$candidate" ]; then echo "$candidate"; break; fi
done

emit dockerfile
for candidate in Dockerfile docker-build-recipe.txt; do
    if [ -f "/workspace/.manifest/$candidate" ]; then cat "/workspace/.manifest/$candidate"; break; fi
done

emit python
python --version 2>&1 || true

emit hipblaslt
header="$(find ${ROCM_PATH:-/opt/rocm} /opt/rocm /opt/venv -name hipblaslt-version.h 2>/dev/null | head -1)"
[ -n "$header" ] && grep -hE 'HIPBLASLT_VERSION_(MAJOR|MINOR|PATCH|TWEAK)' "$header" 2>/dev/null

emit rccl
header="$(find ${ROCM_PATH:-/opt/rocm} /opt/rocm /opt/venv -name rccl.h 2>/dev/null | head -1)"
[ -n "$header" ] && grep -hE 'define NCCL_(MAJOR|MINOR|PATCH)' "$header" 2>/dev/null

emit workspace_repos
for dir in /workspace/*/; do
    if [ -d "$dir/.git" ]; then
        name="$(basename "$dir")"
        rev="$(git -C "$dir" rev-parse HEAD 2>/dev/null || echo unknown)"
        echo "$name $rev"
    fi
done

emit rocm_path
echo "${ROCM_PATH:-}"
"""


def run(cmd, **kwargs):
    return subprocess.run(cmd, capture_output=True, text=True, check=False, **kwargs)


def split_sections(stdout):
    """Split the delimited probe output into {section_name: body}."""
    sections = {}
    current = None
    lines = []
    for line in stdout.splitlines():
        match = re.match(rf"^{re.escape(DELIM)}(\w+)@@@$", line)
        if match:
            if current is not None:
                sections[current] = "\n".join(lines).strip("\n")
            current = match.group(1)
            lines = []
        elif current is not None:
            lines.append(line)
    if current is not None:
        sections[current] = "\n".join(lines).strip("\n")
    return sections


def inspect_image(image):
    """Short image id, build date and size, as the release-notes tables show them.

    Size comes from `docker images`, not `docker image inspect`. Inspect only
    exposes the content (compressed) size -- 14.3 GB for rocm/primus:v26.6 --
    while the release notes quote on-disk usage, 54.0 GB. Verified against the
    committed v26.4-v26.6 tables for both families.
    """
    result = run(["docker", "image", "inspect", image, "--format", "{{json .}}"])
    if result.returncode != 0:
        raise SystemExit(f"ERROR: docker image inspect {image} failed:\n{result.stderr.strip()}")
    data = json.loads(result.stdout)
    image_id = data.get("Id", "")
    if image_id.startswith("sha256:"):
        image_id = image_id[len("sha256:") : len("sha256:") + 12]
    created = (data.get("Created") or "")[:10]
    content_size = data.get("Size") or 0

    disk = run(["docker", "images", "--format", "{{.Size}}", image])
    size_display = None
    match = re.match(r"([\d.]+)\s*([KMGT]?B)", disk.stdout.strip())
    if match:
        value, unit = float(match.group(1)), match.group(2)
        size_display = f"{value:.1f} {unit}"
    return image_id, created, content_size, size_display


def probe_container(image):
    result = run(["docker", "run", "--rm", "--entrypoint", "bash", image, "-c", PROBE_SCRIPT])
    if result.returncode != 0 and not result.stdout.strip():
        raise SystemExit(f"ERROR: probe container failed for {image}:\n{result.stderr.strip()}")
    return split_sections(result.stdout)


def hipblaslt_version(section):
    """1.4.1-8d1ae90e -- MAJOR.MINOR.PATCH with the TWEAK hash appended."""
    macros = C.parse_version_header(
        section,
        [
            "HIPBLASLT_VERSION_MAJOR",
            "HIPBLASLT_VERSION_MINOR",
            "HIPBLASLT_VERSION_PATCH",
            "HIPBLASLT_VERSION_TWEAK",
        ],
    )
    parts = [macros.get(f"HIPBLASLT_VERSION_{name}") for name in ("MAJOR", "MINOR", "PATCH")]
    # All three or nothing: a partial read would render as "1.4.?" and then be
    # reported as a version mismatch rather than as the missing data it is.
    if any(part is None for part in parts):
        return None
    core = ".".join(parts)
    tweak = macros.get("HIPBLASLT_VERSION_TWEAK")
    return f"{core}-{tweak}" if tweak else core


def rccl_version(section):
    macros = C.parse_version_header(section, ["NCCL_MAJOR", "NCCL_MINOR", "NCCL_PATCH"])
    if not macros:
        return None
    return ".".join(macros.get(f"NCCL_{part}", "?") for part in ("MAJOR", "MINOR", "PATCH"))


def verify_dockerfile(family, version, in_image_text, in_image_name):
    """Compare the committed release Dockerfile with the image's baked-in copy."""
    path = C.dockerfile_for(family, version)
    rel = path.relative_to(C.ROOT).as_posix()
    if not in_image_text:
        return {"path": rel, "status": "absent-in-image", "in_image_name": in_image_name}
    if not path.exists():
        return {"path": rel, "status": "missing-in-repo", "in_image_name": in_image_name}

    def normalise(text):
        return [line.rstrip() for line in text.strip().splitlines()]

    repo_lines = normalise(path.read_text())
    image_lines = normalise(in_image_text)
    if repo_lines == image_lines:
        return {"path": rel, "status": "match", "in_image_name": in_image_name}
    repo_set, image_set = set(repo_lines), set(image_lines)
    only_repo = len([line for line in repo_lines if line not in image_set])
    only_image = len([line for line in image_lines if line not in repo_set])
    return {
        "path": rel,
        "status": "differs",
        "in_image_name": in_image_name,
        "repo_lines": len(repo_lines),
        "image_lines": len(image_lines),
        "lines_only_in_repo": only_repo,
        "lines_only_in_image": only_image,
    }


def build_snapshot(image, version, family, sections, image_id, created, content_size, size_display):
    # The build-time manifest is not always what ships. rocm/jax-training:maxtext-v26.6
    # records transformers 5.14.1 in /workspace/.manifest/requirements.txt, but the
    # MaxDiffusion stage runs afterwards and pins it back to 4.57.3 -- which is what
    # the release notes document, because it is what a user actually gets. So the live
    # `pip list` is authoritative and the manifest is kept for provenance; where they
    # disagree, the divergence is itself worth a note in the docs.
    pip_manifest = C.parse_pip_list(sections.get("requirements", ""))
    pip_live = C.parse_pip_list(sections.get("pip_live", ""))
    pip = pip_live or pip_manifest
    divergences = {
        name: {"manifest": pip_manifest[name], "installed": pip_live[name]}
        for name in sorted(set(pip_manifest) & set(pip_live))
        if pip_manifest[name] != pip_live[name]
    }
    env = C.parse_env(sections.get("env", ""))
    repos = {}
    for line in sections.get("workspace_repos", "").splitlines():
        fields = line.split()
        if len(fields) == 2:
            repos[fields[0]] = fields[1]

    python_version = None
    match = re.search(r"(\d+\.\d+\.\d+)", sections.get("python", ""))
    if match:
        python_version = match.group(1)

    native = {}
    hipblaslt = hipblaslt_version(sections.get("hipblaslt", ""))
    if hipblaslt:
        native["hipblaslt"] = hipblaslt
    rccl = rccl_version(sections.get("rccl", ""))
    if rccl:
        native["rccl"] = rccl

    build_commit = repos.get("Primus")

    return {
        "schema_version": C.SCHEMA_VERSION,
        "source": "image",
        "version": version,
        "family": family,
        "image": image,
        "image_id": image_id,
        "built": created,
        "size": size_display,
        "content_size_bytes": content_size,
        "manifest_version": sections.get("manifest_version", "").strip() or None,
        "build_commit": build_commit,
        "build_commit_short": build_commit[:8] if build_commit else None,
        "python": python_version,
        "pip": pip,
        "pip_source": "installed" if pip_live else "manifest",
        "pip_manifest_divergences": divergences,
        "dpkg": C.parse_dpkg(sections.get("dpkg", "")),
        "env": env,
        "native": native,
        "workspace_repos": repos,
        "rocm_path": sections.get("rocm_path", "").strip() or None,
        "dockerfile": verify_dockerfile(
            family, version, sections.get("dockerfile", ""), sections.get("dockerfile_name", "").strip()
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--version", help="Release version, e.g. v26.7")
    parser.add_argument("--family", choices=sorted(C.FAMILIES), help="Image family (inferred from --image)")
    parser.add_argument("--image", help="Explicit image reference; defaults to the family convention")
    parser.add_argument(
        "--dockerfile-version",
        help="Version to use when locating the in-repo Dockerfile (defaults to --version)",
    )
    args = parser.parse_args()

    family = args.family or (C.family_for_image(args.image) if args.image else None)
    if not family:
        parser.error("could not infer --family; pass it explicitly")
    if not args.version:
        parser.error("--version is required")

    image = args.image or C.image_for(family, args.version)
    print(f"Probing {image} (family={family}, version={args.version})")

    image_id, created, content_size, size_display = inspect_image(image)
    sections = probe_container(image)
    if not sections.get("requirements"):
        raise SystemExit(
            f"ERROR: {image} has no {C.MANIFEST_DIR}/requirements.txt. "
            "Images older than v26.3 predate the manifest and cannot be probed."
        )

    snapshot = build_snapshot(
        image, args.version, family, sections, image_id, created, content_size, size_display
    )
    if args.dockerfile_version:
        snapshot["dockerfile"] = verify_dockerfile(
            family,
            args.dockerfile_version,
            sections.get("dockerfile", ""),
            sections.get("dockerfile_name", "").strip(),
        )

    path = C.save_snapshot(snapshot)
    print(f"  packages       {len(snapshot['pip'])} (from {snapshot['pip_source']})")
    if snapshot["pip_manifest_divergences"]:
        names = ", ".join(sorted(snapshot["pip_manifest_divergences"]))
        print(f"  post-manifest  {len(snapshot['pip_manifest_divergences'])} changed after capture: {names}")
    print(f"  id / built     {snapshot['image_id']} / {snapshot['built']} / {snapshot['size']}")
    print(f"  python         {snapshot['python']}")
    print(f"  native         {snapshot['native']}")
    print(f"  build commit   {snapshot['build_commit_short']}")
    print(f"  manifest       {snapshot['manifest_version']}")
    print(f"  dockerfile     {snapshot['dockerfile']['status']} ({snapshot['dockerfile']['path']})")
    print(f"  wrote          {path.relative_to(C.ROOT)}")

    if snapshot["dockerfile"]["status"] == "differs":
        print("\nWARNING: the committed Dockerfile is not what this image was built from.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
