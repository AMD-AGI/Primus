###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Establish what a release run has to work with, and record it in state.json.

The important output is the **build commit per family** -- the commit each image
was actually built from. Everything that needs a git range anchors on that
rather than on a tag, a release branch, or HEAD:

  - HEAD is wrong. For v26.6 there were 67 commits between the build commit
    (2aa05ead) and the doc-update commit, so anchoring on HEAD would credit the
    release with two weeks of work that is not in the image.
  - The tag is usually right but need not exist yet, and is not on `main`
    (v26.6.0 is not an ancestor of main -- release tags live on the release
    branch lineage).
  - The release branch often does not exist at doc-writing time. release/v26.7
    did not exist while v26.7.0 was already tagged.

Resolution order per family: `ARG PRIMUS_BRANCH` from the release Dockerfile
when it is a commit (the primus image pins one), otherwise the image's own
/workspace/Primus HEAD from the snapshot (the JAX Dockerfile pins
PRIMUS_BRANCH=main, so only the image knows).

Also decides the release shape (full / single-family / patch) and which shape
the user-facing checkout instruction must take, since writing
`git checkout release/v26.7` while that branch does not exist ships a broken
instruction.

Usage:
    python tools/release_docs/preflight.py --version v26.7
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common as C  # noqa: E402

SHA40 = re.compile(r"^[0-9a-f]{40}$")
STATE_DIR = C.ROOT / "output/release-docs"


def run(cmd, **kwargs):
    return subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=C.ROOT, **kwargs)


def git(*args):
    return run(["git", *args])


def dockerfile_args(path):
    """ARG/ENV defaults declared in a Dockerfile, for pin and branch lookups."""
    if not path.exists():
        return {}
    found = {}
    for match in re.finditer(
        r"^(?:ARG|ENV)\s+([A-Za-z_][A-Za-z0-9_]*)=(\S+)", path.read_text(), re.MULTILINE
    ):
        found.setdefault(match.group(1), match.group(2))
    return found


def image_present(image):
    return run(["docker", "image", "inspect", image, "--format", "{{.Id}}"]).returncode == 0


def ref_exists(ref):
    return git("rev-parse", "--verify", "--quiet", ref).returncode == 0


def git_remote_readable():
    result = git("ls-remote", "--heads", "origin", "refs/heads/main")
    return result.returncode == 0, (result.stderr.strip().splitlines() or [""])[0]


def resolve_family(version, family, previous):
    """Everything known about one family for this release."""
    image = C.image_for(family, version)
    dockerfile = C.dockerfile_for(family, version)
    args = dockerfile_args(dockerfile)
    snapshot = C.load_snapshot(version, family)

    pinned = args.get("PRIMUS_BRANCH", "")
    if SHA40.match(pinned):
        build_commit, origin = pinned, "Dockerfile ARG PRIMUS_BRANCH"
    elif snapshot and snapshot.get("build_commit"):
        build_commit = snapshot["build_commit"]
        origin = f"image /workspace/Primus (Dockerfile pins PRIMUS_BRANCH={pinned or 'unset'})"
    else:
        build_commit, origin = None, "unresolved: probe the image first"

    previous_snapshot = C.load_snapshot(previous, family) if previous else None
    return {
        "image": image,
        "image_present": image_present(image),
        "dockerfile": dockerfile.relative_to(C.ROOT).as_posix(),
        "dockerfile_present": dockerfile.exists(),
        "dockerfile_verified": (snapshot or {}).get("dockerfile", {}).get("status"),
        "maxtext_branch": args.get("MAXTEXT_BRANCH"),
        "snapshot_present": snapshot is not None,
        "build_commit": build_commit,
        "build_commit_short": build_commit[:8] if build_commit else None,
        "build_commit_source": origin,
        "build_commit_known_locally": bool(build_commit) and ref_exists(build_commit),
        "previous_build_commit": (previous_snapshot or {}).get("build_commit"),
    }


def previous_version_for(version):
    others = sorted(
        {snap_version for snap_version, _, _ in C.available_snapshots() if snap_version != version},
        key=C.version_key,
        reverse=True,
    )
    return others[0] if others else None


def release_shape(families, version):
    """Full, single-family, or patch -- the two families are not in lockstep."""
    present = [name for name, info in families.items() if info["image_present"]]
    if len(re.findall(r"\d+", version)) > 2:
        return "patch", present
    if len(present) == len(C.FAMILIES):
        return "full", present
    if len(present) == 1:
        return "single-family", present
    return "unknown", present


def build_state(version, previous, ssh_key):
    if ssh_key:
        os.environ["GIT_SSH_COMMAND"] = f"ssh -i {ssh_key} -o IdentitiesOnly=yes"

    previous = previous or previous_version_for(version)
    families = {name: resolve_family(version, name, previous) for name in sorted(C.FAMILIES)}
    shape, present = release_shape(families, version)

    tag = f"{version.lstrip('v')}.0"
    tag_ref = f"v{tag}"
    branch = f"release/{version}"
    remote_ok, remote_error = git_remote_readable()

    # The user-facing checkout instruction: branch form when the branch exists,
    # otherwise the v26.4-style commit form. Never a blind string substitution.
    branch_exists = ref_exists(f"refs/remotes/origin/{branch}")
    checkout = "branch" if branch_exists else "commit"

    return {
        "version": version,
        "previous_version": previous,
        "release_shape": shape,
        "families_present": present,
        "families": families,
        "tag": {"ref": tag_ref, "exists": ref_exists(tag_ref)},
        "release_branch": {"ref": branch, "exists_on_origin": branch_exists},
        "checkout_instruction_shape": checkout,
        "git_remote_readable": remote_ok,
        "git_remote_error": None if remote_ok else remote_error,
        "phases_completed": [],
    }


def report(state):
    def mark(ok):
        return "ok  " if ok else "MISS"

    print(f"Release preflight: {state['version']}  (previous: {state['previous_version']})")
    print(f"  release shape           {state['release_shape']} {state['families_present']}")
    for name, info in state["families"].items():
        print(f"  --- {name}")
        print(f"    {mark(info['image_present'])} image                {info['image']}")
        print(f"    {mark(info['dockerfile_present'])} dockerfile           {info['dockerfile']}")
        print(
            f"    {mark(info['dockerfile_verified'] == 'match')} dockerfile vs image  {info['dockerfile_verified']}"
        )
        print(f"    {mark(info['snapshot_present'])} snapshot")
        print(f"    {mark(bool(info['build_commit']))} build commit         {info['build_commit_short']}")
        print(f"         via                  {info['build_commit_source']}")
        if info["maxtext_branch"]:
            print(f"         MAXTEXT_BRANCH       {info['maxtext_branch']}")
    print(f"  {mark(state['tag']['exists'])} tag                  {state['tag']['ref']}")
    print(
        f"  {mark(state['release_branch']['exists_on_origin'])} release branch       {state['release_branch']['ref']} (on origin)"
    )
    print(f"  ---> checkout instructions use the {state['checkout_instruction_shape'].upper()} form")
    print(f"  {mark(state['git_remote_readable'])} git remote readable")
    if not state["git_remote_readable"]:
        print(f"         {state['git_remote_error']}")

    blockers = []
    # Without an image there is no release to document, and nothing downstream can
    # resolve a build commit. Say so instead of reporting a tidy list of misses and
    # exiting clean.
    if not state["families_present"]:
        blockers.append(
            f"no image found for {state['version']}: pull "
            + " or ".join(info["image"] for info in state["families"].values())
        )
    for name, info in state["families"].items():
        if info["image_present"] and not info["build_commit"]:
            blockers.append(f"{name}: build commit unresolved (run probe_image.py)")
        if info["image_present"] and not info["dockerfile_present"]:
            blockers.append(f"{name}: {info['dockerfile']} is missing")
        if info["dockerfile_verified"] == "differs":
            blockers.append(f"{name}: committed Dockerfile is not what the image was built from")
    if not state["release_branch"]["exists_on_origin"]:
        print(
            f"\n  NOTE: {state['release_branch']['ref']} does not exist on origin. Not a blocker -- "
            "docs land on main first and are cherry-picked to the branch later (v26.6 did this in #1160). "
            "Checkout instructions will use the commit form and the PR will carry the follow-up."
        )
    if blockers:
        print("\nBLOCKERS:")
        for blocker in blockers:
            print(f"  - {blocker}")
    return blockers


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--version", required=True)
    parser.add_argument("--previous", help="previous release (defaults to the newest other snapshot)")
    parser.add_argument("--ssh-key", help="SSH key for origin; sets GIT_SSH_COMMAND")
    parser.add_argument("--no-write", action="store_true", help="report without writing state.json")
    args = parser.parse_args()

    state = build_state(args.version, args.previous, args.ssh_key)
    blockers = report(state)

    if not args.no_write:
        path = STATE_DIR / args.version / "state.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {path.relative_to(C.ROOT)}")
    return 1 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
