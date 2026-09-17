###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Replay the tooling against release work that was already reviewed and shipped.

Two past commits serve as ground truth:

  ebca87c4  "docs: bump current release tags from v26.5 to v26.6 (#1114)" -- a
            commit whose entire purpose was the version bump, which makes it an
            exact target for bump_version.py. Measured as precision/recall over
            the version-transition lines it changed.

  3aa6a458  "update baremetal install v26.6 and update docs to v26.6 (#1120)" --
            aligned the install scripts to the v26.6 Dockerfiles. Before it the
            scripts were v26.5-era, so install_parity.py must report drift at the
            parent and none at the commit itself. The negative half matters most:
            a parity checker that cannot see drift it was built to catch is worse
            than nothing.

Each replay runs today's tools against the tree as it was, in a throwaway
worktree, so the check is a regression test rather than a self-consistency one.

Usage:
    python tools/release_docs/golden_replay.py
"""

import argparse
import ast
import json
import re
import shutil
import subprocess
import sys
import tempfile
import tokenize
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common as C  # noqa: E402

TOOLS_REL = "tools/release_docs"

# Ground truth is the whole v26.6 documentation train on main, not a single
# commit: the work landed across #1114 (version bump), #1120 (install/docs
# alignment) and #1052. Scoring against one commit counts the others' lines as
# false positives. `base` is the tree before any of it, `head` after all of it.
BUMP_FIXTURE = {
    "base": "ebca87c4^",
    "head": "3aa6a458",
    "old": "v26.5",
    "new": "v26.6",
    "branch_existed": True,
    "why": "every v26.5 -> v26.6 transition across the release train (#1114, #1120, #1052)",
}

PARITY_FIXTURE = {
    "commit": "3aa6a458",
    "release": "v26.6",
    "why": "aligned tools/installation* with the v26.6 Dockerfiles",
}


def run(cmd, cwd=None):
    return subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=cwd or C.ROOT)


def git(*args, cwd=None):
    result = run(["git", *args], cwd=cwd)
    if result.returncode != 0:
        raise SystemExit(f"ERROR: git {' '.join(args)} failed:\n{result.stderr.strip()}")
    return result.stdout


class Worktree:
    """A detached worktree at a commit, with today's tools copied in."""

    def __init__(self, commit):
        self.commit = commit
        self.path = Path(tempfile.mkdtemp(prefix="golden-replay-"))

    def __enter__(self):
        shutil.rmtree(self.path, ignore_errors=True)
        git("worktree", "add", "--detach", str(self.path), self.commit)
        shutil.rmtree(self.path / TOOLS_REL, ignore_errors=True)
        shutil.copytree(C.ROOT / TOOLS_REL, self.path / TOOLS_REL)
        return self

    def __exit__(self, *exc):
        run(["git", "worktree", "remove", "--force", str(self.path)])
        shutil.rmtree(self.path, ignore_errors=True)
        run(["git", "worktree", "prune"])
        return False


def stale_counts(tree_root, old, excluded):
    """Per-file count of surviving old-version references under a tree.

    Comparing counts rather than matching lines is what makes this measurable:
    the release often moves a version reference *inside* prose it also rewrote
    (megatron-lm-training.md:23 both bumps the tag and reworks the sentence), so
    no line-level substitution match exists even though the version did move.
    """
    counts = {}
    pattern = re.compile(rf"{re.escape(old)}\b")
    for line in git("ls-files", cwd=tree_root).splitlines():
        if not line or any(line.startswith(prefix) for prefix in excluded):
            continue
        path = Path(tree_root) / line
        if path.suffix.lower() in {".png", ".jpg", ".pdf", ".whl", ".so", ".gif"} or not path.is_file():
            continue
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        hits = len(pattern.findall(text))
        if hits:
            counts[line] = hits
    return counts


def replay_bump():
    fixture = BUMP_FIXTURE
    old, new = fixture["old"], fixture["new"]
    print(f"=== bump_version replay: {fixture['base']}..{fixture['head']} ({old} -> {new})")
    print(f"    {fixture['why']}")

    excluded, _ = None, None
    with Worktree(fixture["base"]) as tree:
        cmd = [sys.executable, f"{TOOLS_REL}/bump_version.py", "--from", old, "--to", new]
        if fixture["branch_existed"]:
            cmd.append("--branch-exists")
        report_json = run(cmd + ["--json"], cwd=tree.path)
        if report_json.returncode != 0:
            print(report_json.stderr)
            return False
        data = json.loads(report_json.stdout)
        excluded = json.loads((C.ROOT / TOOLS_REL / "version_bump_rules.json").read_text())["excluded_paths"]

        applied = run(cmd + ["--apply"], cwd=tree.path)
        if applied.returncode != 0:
            print(applied.stderr)
            return False
        after_tool = stale_counts(tree.path, old, excluded)
        tree.path
        base_files_snapshot = set(git("ls-files", cwd=tree.path).splitlines())

    with Worktree(fixture["head"]) as tree:
        after_release = stale_counts(tree.path, old, excluded)

    held_paths = {hit["path"] for hit in data["hold"]}
    # Only files the tool could actually have seen. The release also adds and
    # renames files (examples/mlperf/gpt_oss_20b/run_with_docker.sh is new in
    # v26.6), and counting those as over-reach would be measuring the wrong thing.
    base_files = base_files_snapshot
    over_reach, under_reach = [], []
    for path in sorted((set(after_tool) | set(after_release)) & base_files):
        tool_left = after_tool.get(path, 0)
        release_left = after_release.get(path, 0)
        if tool_left < release_left:
            over_reach.append((path, tool_left, release_left))
        elif tool_left > release_left:
            under_reach.append((path, tool_left, release_left, path in held_paths))

    print(f"    rewrites proposed   {len(data['rewrite'])}")
    print(f"    held for review     {len(data['hold'])}")
    print(
        f"    files agreeing with the release on what stays: "
        f"{len(set(after_tool) | set(after_release)) - len(over_reach) - len(under_reach)}"
    )

    if over_reach:
        print(f"    OVER-REACH ({len(over_reach)}) -- rewrote references the release deliberately kept:")
        for path, tool_left, release_left in over_reach:
            print(f"      {path}: tool left {tool_left}, release left {release_left}")
    if under_reach:
        unheld = [item for item in under_reach if not item[3]]
        print(f"    under-reach ({len(under_reach)}) -- left references the release updated:")
        for path, tool_left, release_left, held in under_reach:
            marker = "held for review" if held else "NOT SURFACED"
            print(f"      [{marker}] {path}: tool left {tool_left}, release left {release_left}")
        if unheld:
            print(f"    FAIL: {len(unheld)} file(s) left stale without being surfaced for review.")
            return False

    # Over-reach is the dangerous direction: it silently rewrites history. Leaving
    # something for review is safe as long as it is actually surfaced.
    return not over_reach


def replay_parity():
    fixture = PARITY_FIXTURE
    print(f"\n=== install_parity replay: {fixture['commit']} ({fixture['why']})")
    outcomes = {}
    for label, commit in (("before", f"{fixture['commit']}^"), ("after", fixture["commit"])):
        with Worktree(commit) as tree:
            result = run(
                [
                    sys.executable,
                    f"{TOOLS_REL}/install_parity.py",
                    "--release",
                    fixture["release"],
                    "--check",
                ],
                cwd=tree.path,
            )
            drift = sum(
                int(match) for match in re.findall(r"^\s*drift\s+(\d+)$", result.stdout, re.MULTILINE)
            )
            outcomes[label] = drift
            print(f"    {label:6} {commit:12} drift={drift}  (exit {result.returncode})")

    ok = outcomes["before"] > 0 and outcomes["after"] == 0
    if outcomes["before"] == 0:
        print("    FAIL: no drift detected before the alignment commit; the checker is blind.")
    if outcomes["after"] != 0:
        print("    FAIL: drift reported after the alignment commit; the checker has false positives.")
    return ok


def check_sha_abbreviation():
    """Refuse any tool that reads a SHA git abbreviated for itself.

    git scales the width to the local object count -- 7 characters in a fresh clone,
    8 in a long-lived one -- so slicing one makes the output depend on the machine
    that produced it. `submodule_bumps` shipped exactly that: `[:8]` of
    `git diff --raw` returned 7 characters in CI, where it read as documentation
    drift against the full SHAs probed from the image. Ask for `--abbrev=40` and
    truncate here instead.

    `--submodule=short` is banned for the older reason recorded in `submodule_bumps`:
    it emits no summary line, so the parse silently matched nothing and reported no
    bumps forever.

    Only string literals are inspected, because the comments that explain these
    constructs necessarily contain them.
    """
    banned = {"%h", "--short", "--submodule=short"}
    offenders = []
    for path in sorted((C.ROOT / TOOLS_REL).glob("*.py")):
        # This file names the banned forms as data; scanning it finds only itself.
        if path.resolve() == Path(__file__).resolve():
            continue
        with open(path, "rb") as handle:
            for token in tokenize.tokenize(handle.readline):
                if token.type != tokenize.STRING:
                    continue
                try:
                    value = ast.literal_eval(token.string)
                except (ValueError, SyntaxError):
                    continue
                if not isinstance(value, str):
                    continue
                if value in banned or (value.startswith("--abbrev=") and value != "--abbrev=40"):
                    offenders.append(f"{path.name}: {value!r}")

    print("\n=== sha abbreviation: no tool may read git's variable-width SHAs")
    for item in offenders:
        print(f"    {item}")
    print(f"    {len(offenders)} offender(s)")
    return not offenders


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--only", choices=["bump", "parity", "abbrev"], help="run a single check")
    args = parser.parse_args()

    results = {}
    if args.only in (None, "bump"):
        results["bump_version"] = replay_bump()
    if args.only in (None, "parity"):
        results["install_parity"] = replay_parity()
    if args.only in (None, "abbrev"):
        results["sha_abbreviation"] = check_sha_abbreviation()

    print("\n=== golden replay summary")
    for name, ok in results.items():
        print(f"    {name:16} {'PASS' if ok else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
