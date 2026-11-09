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

Resolution order per family: `--build-commit FAMILY=SHA` when given, then
`ARG PRIMUS_BRANCH` from the release Dockerfile when it is a commit (the primus
image pins one), otherwise the image's own /workspace/Primus HEAD from the
snapshot (the JAX Dockerfile pinned PRIMUS_BRANCH=main through v26.7, so only the
image knew).

The v26.8 JAX image shipped no Primus at all -- it ran from a mounted
checkout -- so there was no build commit to read (v26.9 pins one again). The Primus commit that pairs
with such an image is an editorial choice (normally the release-branch commit
that aligns the backend with the image), so it has to be passed explicitly with
--build-commit rather than guessed.

Also decides the release shape (full / single-family / patch) and which shape
the user-facing checkout instruction must take, since writing
`git checkout release/v26.7` while that branch does not exist ships a broken
instruction.

Works from the snapshots, not from docker: a family is part of the release once
`probe_image.py` has written its snapshot (from the CI image that passed QA,
`--publish-as` the release tag), so neither the published image nor docker is
needed here. `--family` limits the run to one family when the two images are
released at different times; the other family's section is left alone.

Usage:
    python tools/release_docs/preflight.py --version v26.7
    python tools/release_docs/preflight.py --version v26.9 --family primus --build-commit primus=<sha>
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


def ref_exists(ref):
    return git("rev-parse", "--verify", "--quiet", ref).returncode == 0


def remote_branch_exists(url, branch):
    """Whether a branch exists in another repository.

    The MaxText branch lives in ROCm/maxtext, not in Primus, so checking it against
    `origin` conflates two repositories. During the first release run that made the
    tool hold `.gitmodules` back even though `release/v26.7` did exist upstream --
    at exactly the commit the published image ships.
    """
    result = git("ls-remote", "--heads", url, f"refs/heads/{branch}")
    if result.returncode != 0:
        return None  # unknown: no network or no access
    return bool(result.stdout.strip())


def git_remote_readable():
    result = git("ls-remote", "--heads", "origin", "refs/heads/main")
    return result.returncode == 0, (result.stderr.strip().splitlines() or [""])[0]


def previous_snapshot_for(version, family, previous=None):
    """The previous release of this family: `--previous` when given, else the
    newest older snapshot of the family (MaxText ships patch releases the
    primus family does not have)."""
    if previous:
        return C.load_snapshot(previous, family)
    older = [
        (snap_version, data)
        for snap_version, snap_family, data in C.available_snapshots()
        if snap_family == family and C.version_key(snap_version) < C.version_key(version)
    ]
    return older[0][1] if older else None


def resolve_family(version, family, previous, override=None):
    """Everything known about one family for this release."""
    dockerfile = C.dockerfile_for(family, version)
    args = dockerfile_args(dockerfile)
    snapshot = C.load_snapshot(version, family)
    image = (snapshot or {}).get("image") or C.image_for(family, version)
    ships_primus = snapshot is None or "Primus" in snapshot.get("workspace_repos", {})

    pinned = args.get("PRIMUS_BRANCH", "")
    if override:
        resolved = git("rev-parse", "--verify", "--quiet", f"{override}^{{commit}}").stdout.strip()
        build_commit = resolved or None
        origin = (
            f"--build-commit {override}"
            if resolved
            else f"unresolved: --build-commit {override} is not a local commit (fetch origin)"
        )
    elif SHA40.match(pinned):
        build_commit, origin = pinned, "Dockerfile ARG PRIMUS_BRANCH"
    elif snapshot and snapshot.get("build_commit"):
        build_commit = snapshot["build_commit"]
        origin = f"image /workspace/Primus (Dockerfile pins PRIMUS_BRANCH={pinned or 'unset'})"
    elif not ships_primus:
        build_commit = None
        origin = "unresolved: the image ships no /workspace/Primus; pass --build-commit " f"{family}=<sha>"
    else:
        build_commit, origin = None, "unresolved: probe the image first"

    previous_snapshot = previous_snapshot_for(version, family, previous)
    previous_commit = (previous_snapshot or {}).get("build_commit")
    previous_origin = "previous snapshot" if previous_commit else None
    if previous_snapshot and not previous_commit:
        # A JAX image without Primus (v26.8 on) records no build commit. The primus
        # image of the same release was built from that release's branch, which is
        # the Primus a user pairs with the JAX image, so it stands in -- reported, so
        # a reviewer can override it with collect_changes.py --from.
        other = C.load_snapshot(previous_snapshot["version"], "primus" if family == "jax" else "jax")
        previous_commit = (other or {}).get("build_commit")
        if previous_commit:
            previous_origin = f"the other family's {previous_snapshot['version']} build commit (check it)"

    # The MaxText branch is a branch of ROCm/maxtext; resolve it there.
    maxtext_branch = args_maxtext = args.get("MAXTEXT_BRANCH")
    maxtext_repo = args.get("MAXTEXT_REPO", "https://github.com/ROCm/maxtext.git")
    maxtext_exists = remote_branch_exists(maxtext_repo, args_maxtext) if args_maxtext else None

    return {
        "image": image,
        "probed_image": (snapshot or {}).get("probed_image"),
        "dockerfile": dockerfile.relative_to(C.ROOT).as_posix(),
        "dockerfile_present": dockerfile.exists(),
        "dockerfile_verified": (snapshot or {}).get("dockerfile", {}).get("status"),
        "maxtext_branch": maxtext_branch,
        "maxtext_repo": maxtext_repo,
        "maxtext_branch_exists_upstream": maxtext_exists,
        "snapshot_present": snapshot is not None,
        "build_commit": build_commit,
        "build_commit_short": build_commit[:8] if build_commit else None,
        "build_commit_source": origin,
        "build_commit_known_locally": bool(build_commit) and ref_exists(build_commit),
        "image_ships_primus": ships_primus,
        "previous_version": (previous_snapshot or {}).get("version"),
        "previous_build_commit": previous_commit,
        "previous_build_commit_source": previous_origin,
    }


def previous_version_for(version):
    others = sorted(
        {snap_version for snap_version, _, _ in C.available_snapshots() if snap_version != version},
        key=C.version_key,
        reverse=True,
    )
    return others[0] if others else None


def release_shape(families, version):
    """Full, single-family, or patch -- the two families are not in lockstep.

    `families` is the run's scope; a family is present once its snapshot exists."""
    present = [name for name, info in families.items() if info["snapshot_present"]]
    if len(re.findall(r"\d+", version)) > 2:
        return "patch", present
    if len(present) == len(C.FAMILIES):
        return "full", present
    if len(present) == 1:
        return "single-family", present
    return "unknown", present


def build_state(version, previous, ssh_key, overrides=None, scope=None):
    if ssh_key:
        os.environ["GIT_SSH_COMMAND"] = f"ssh -i {ssh_key} -o IdentitiesOnly=yes"

    overrides = overrides or {}
    scope = sorted(scope or C.FAMILIES)
    # Only an explicit --previous goes to the families: without one, each falls
    # back to its own newest older snapshot (MaxText has patch releases primus lacks).
    families = {name: resolve_family(version, name, previous, overrides.get(name)) for name in scope}
    previous = previous or previous_version_for(version)
    shape, present = release_shape(families, version)
    released_earlier = sorted(
        name for name in C.FAMILIES if name not in scope and C.load_snapshot(version, name) is not None
    )

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
        "scope": scope,
        "families_present": present,
        "families_released_earlier": released_earlier,
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
    if state["families_released_earlier"]:
        print(f"  released earlier        {state['families_released_earlier']} (not touched by this run)")
    for name, info in state["families"].items():
        print(f"  --- {name}")
        print(f"    {mark(info['snapshot_present'])} snapshot             {info['image']}")
        if info["probed_image"]:
            print(f"         probed from          {info['probed_image']}")
        print(f"    {mark(info['dockerfile_present'])} dockerfile           {info['dockerfile']}")
        print(
            f"    {mark(info['dockerfile_verified'] == 'match')} dockerfile vs image  {info['dockerfile_verified']}"
        )
        print(f"    {mark(bool(info['build_commit']))} build commit         {info['build_commit_short']}")
        print(f"         via                  {info['build_commit_source']}")
        previous = info["previous_build_commit"]
        print(
            f"    {mark(bool(previous))} previous             {info['previous_version']} "
            f"{previous[:8] if previous else None} ({info['previous_build_commit_source']})"
        )
        if info["maxtext_branch"]:
            upstream = info["maxtext_branch_exists_upstream"]
            verdict = {True: "exists upstream", False: "NOT in ROCm/maxtext", None: "unverified"}[upstream]
            print(f"    {mark(upstream is True)} MAXTEXT_BRANCH       {info['maxtext_branch']} ({verdict})")
    print(f"  {mark(state['tag']['exists'])} tag                  {state['tag']['ref']}")
    print(
        f"  {mark(state['release_branch']['exists_on_origin'])} release branch       {state['release_branch']['ref']} (on origin)"
    )
    print(f"  ---> checkout instructions use the {state['checkout_instruction_shape'].upper()} form")
    print(f"  {mark(state['git_remote_readable'])} git remote readable")
    if not state["git_remote_readable"]:
        print(f"         {state['git_remote_error']}")

    blockers = []
    # Without a snapshot there is nothing to document, and nothing downstream can
    # resolve a build commit. Say so instead of reporting a tidy list of misses and
    # exiting clean.
    for name, info in state["families"].items():
        if not info["snapshot_present"]:
            blockers.append(
                f"{name}: no snapshot for {state['version']}: probe_image.py --version {state['version']} "
                f"--image <CI tag> --publish-as {info['image']} --write-dockerfile"
            )
            continue
        if not info["build_commit"]:
            blockers.append(f"{name}: build commit {info['build_commit_source']}")
        elif not info["build_commit_known_locally"]:
            blockers.append(
                f"{name}: build commit {info['build_commit_short']} is not in this clone (fetch origin)"
            )
        if not info["dockerfile_present"]:
            blockers.append(
                f"{name}: {info['dockerfile']} is missing (probe_image.py --write-dockerfile saves the image's)"
            )
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
    parser.add_argument(
        "--build-commit",
        action="append",
        default=[],
        metavar="FAMILY=REV",
        help="Primus commit to anchor a family on when its image ships no Primus checkout",
    )
    parser.add_argument(
        "--family",
        action="append",
        choices=sorted(C.FAMILIES),
        help="limit the run to this family (repeatable; default: both)",
    )
    parser.add_argument("--no-write", action="store_true", help="report without writing state.json")
    args = parser.parse_args()

    overrides = {}
    for item in args.build_commit:
        family, _, rev = item.partition("=")
        if family not in C.FAMILIES or not rev:
            parser.error(
                f"--build-commit expects FAMILY=REV with FAMILY in {sorted(C.FAMILIES)}, got {item!r}"
            )
        overrides[family] = rev

    state = build_state(args.version, args.previous, args.ssh_key, overrides, args.family)
    blockers = report(state)

    if not args.no_write:
        path = STATE_DIR / args.version / "state.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {path.relative_to(C.ROOT)}")
    return 1 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
