###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Collect the raw material for the release-notes Highlights section.

Pure `git log`, no network and no `gh`. GitHub squash-merges put the whole PR
description into the commit message, so the subject, PR number and body are all
already in the repository -- verified across the v26.7 range, where e.g. the
SpecForge merge carries its full "## Summary" / "## Test plan" body. Only labels
are unavailable, and conventional-commit prefixes carry the bucketing.

Two range subtleties this handles:

  - Endpoints are **build commits**, not tags or branches, so the changelog
    describes what actually shipped in the images.
  - `--cherry-pick --right-only` drops commits whose patch already exists on the
    other side. release/v26.6 carries 12 cherry-picks whose originals are on
    main, and without this they would be re-reported as new in v26.7.

Also extracts the added `examples/` recipes and submodule bumps, because a
release that adds models needs those propagated into the model lists and the
support matrix -- changes no version-string bump would ever notice.

Usage:
    python tools/release_docs/collect_changes.py --version v26.7
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common as C  # noqa: E402

RECORD = "\x1e"
FIELD = "\x1f"

PR_NUMBER = re.compile(r"\(#(\d+)\)\s*$")

# Conventional prefixes first, then the bracket styles this repo also uses
# ("[Perf Optimization] ...", "[OOB Release] ...", "[Fix] ...").
BUCKET_RULES = [
    ("dependencies", r"^chore\(deps"),
    ("features", r"^feat\b|^feature\b|^\[?feat"),
    ("fixes", r"^fix\b|^bugfix\b|^\[fix\]|^\[[\w\s-]*\]\s*fix\b|^revert\b"),
    ("performance", r"^perf\b|^\[perf|^\[performance"),
    ("docs", r"^docs?\b"),
    ("maintenance", r"^chore\b|^ci\b|^test\b|^tests\b|^refactor\b|^build\b|^style\b|^\[oob release\]"),
]

# Checked only when no prefix rule matched. This repo lands a lot of un-prefixed
# config tuning ("[maxtext] v26.6 mi300x batch size tuning") that belongs under
# performance rather than in the unclassified pile.
SECONDARY_RULES = [
    ("performance", r"\btun(e|ed|ing)\b|\bbatch size\b|\boccupancy\b|\bthroughput\b|\buplift\b"),
    ("features", r"^add\b|\bsupport for\b|^enable\b"),
]

AREA_RULES = [
    ("megatron", r"^primus/backends/megatron/|^primus/configs/.*megatron|^examples/megatron/"),
    ("torchtitan", r"^primus/backends/torchtitan/|^examples/torchtitan/"),
    ("maxtext", r"^primus/backends/(jax|maxtext)/|^examples/maxtext/|^primus/configs/models/maxtext/"),
    ("maxdiffusion", r"^primus/backends/diffusion/|^examples/maxdiffusion/|^examples/diffusion/"),
    ("examples", r"^examples/"),
    ("docs", r"^docs/|\.md$"),
    ("tools", r"^tools/|^runner/|^\.github/"),
    ("tests", r"^tests/"),
    ("submodules", r"^third_party/"),
    ("core", r"^primus/"),
]

BUCKET_ORDER = ["features", "performance", "fixes", "docs", "maintenance", "dependencies", "other"]


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=C.ROOT)


def git(*args):
    result = run(["git", *args])
    if result.returncode != 0:
        raise SystemExit(f"ERROR: git {' '.join(args)} failed:\n{result.stderr.strip()}")
    return result.stdout


def classify_bucket(subject, author):
    if "dependabot" in author.lower():
        return "dependencies"
    lowered = subject.strip().lower()
    for bucket, pattern in BUCKET_RULES + SECONDARY_RULES:
        if re.search(pattern, lowered):
            return bucket
    return "other"


def classify_areas(files):
    areas = []
    for path in files:
        for area, pattern in AREA_RULES:
            if re.search(pattern, path):
                if area not in areas:
                    areas.append(area)
                break
    return areas


def collapse_body(body, bucket):
    """Dependabot bodies are 200-line upstream changelogs; keep one line."""
    body = body.strip()
    if bucket == "dependencies":
        return body.splitlines()[0].strip() if body else ""
    # Drop the trailing co-author/signature block, which is noise for highlights.
    body = re.sub(r"\n-{3,}\s*\n.*$", "", body, flags=re.DOTALL)
    body = re.sub(r"\n(Co-authored-by|Signed-off-by):.*$", "", body, flags=re.DOTALL)
    return body.strip()


def parse_log(from_commit, to_commit, cherry_pick=True):
    # %h is deliberately absent: it abbreviates to whatever the local object count
    # warrants, which would make changelog.json differ between clones. The short
    # form is sliced from %H below instead.
    fmt = FIELD.join(["%H", "%ad", "%an", "%s", "%b"]) + FIELD
    args = ["log", f"--format={RECORD}{fmt}", "--name-only", "--date=short"]
    if cherry_pick:
        args += ["--cherry-pick", "--right-only", f"{from_commit}...{to_commit}"]
    else:
        args += [f"{from_commit}..{to_commit}"]
    raw = git(*args)

    commits = []
    for chunk in raw.split(RECORD):
        if not chunk.strip():
            continue
        parts = chunk.split(FIELD)
        if len(parts) < 6:
            continue
        sha, date, author, subject, body, tail = parts[:6]
        files = [line.strip() for line in tail.splitlines() if line.strip()]
        bucket = classify_bucket(subject, author)
        pr_match = PR_NUMBER.search(subject)
        commits.append(
            {
                "sha": sha.strip(),
                "short": sha.strip()[:8],
                "date": date.strip(),
                "author": author.strip(),
                "subject": subject.strip(),
                "pr": int(pr_match.group(1)) if pr_match else None,
                "body": collapse_body(body, bucket),
                "bucket": bucket,
                "areas": classify_areas(files),
                "files": files,
            }
        )
    return commits


def diff_names(from_commit, to_commit):
    """Aggregate added/modified/deleted paths across the range."""
    added, modified, deleted = [], [], []
    for line in git("diff", "--name-status", f"{from_commit}..{to_commit}").splitlines():
        fields = line.split("\t")
        if len(fields) < 2:
            continue
        status, path = fields[0], fields[-1]
        if status.startswith("A"):
            added.append(path)
        elif status.startswith("D"):
            deleted.append(path)
        else:
            modified.append(path)
    return added, modified, deleted


def submodule_bumps(from_commit, to_commit):
    """Gitlink moves under third_party/, read from `git diff --raw`.

    --raw gives `:160000 160000 <old> <new> M<TAB><path>`, where 160000 is the
    gitlink mode. Not `--submodule=short`, which emits an ordinary
    `-Subproject commit ...` diff with no summary line: an earlier version of this
    looked for `Submodule <path> a..b`, which that format never produces, so it
    always reported no bumps and did so silently.
    """
    bumps = {}
    # --abbrev=40 because git otherwise scales the abbreviation to the object
    # count, so the same range yields 7 characters in a fresh CI clone and 8 in a
    # long-lived one. Truncating below is only deterministic on a full SHA.
    raw = git("diff", "--raw", "--abbrev=40", f"{from_commit}..{to_commit}", "--", "third_party/")
    for line in raw.splitlines():
        match = re.match(r"^:(\d{6}) (\d{6}) ([0-9a-f]+) ([0-9a-f]+) (\w+)\t(.+)$", line)
        if match and "160000" in (match.group(1), match.group(2)):
            bumps[match.group(6)] = {"from": match.group(3)[:8], "to": match.group(4)[:8]}
    return bumps


def collect(version, previous, from_commit, to_commit):
    commits = parse_log(from_commit, to_commit)
    unfiltered = len(parse_log(from_commit, to_commit, cherry_pick=False))
    added, modified, deleted = diff_names(from_commit, to_commit)

    buckets = {name: [] for name in BUCKET_ORDER}
    for commit in commits:
        buckets[commit["bucket"]].append(commit["short"])

    areas = {}
    for commit in commits:
        for area in commit["areas"]:
            areas.setdefault(area, []).append(commit["short"])

    # Recipes and model presets a release adds; these need to reach the
    # "pre-optimized models" list and the support matrix.
    new_recipes = [
        path
        for path in added
        if re.match(r"^(examples/|primus/configs/models/)", path) and path.endswith((".yaml", ".yml", ".sh"))
    ]

    return {
        "version": version,
        "previous_version": previous,
        "range": {
            "from": from_commit,
            "to": to_commit,
            "commits": len(commits),
            "commits_before_cherry_pick_filter": unfiltered,
            "cherry_picks_suppressed": unfiltered - len(commits),
        },
        "buckets": {name: items for name, items in buckets.items() if items},
        "areas": areas,
        "new_recipes": sorted(new_recipes),
        "files": {"added": len(added), "modified": len(modified), "deleted": len(deleted)},
        "submodule_bumps": submodule_bumps(from_commit, to_commit),
        "commits": commits,
    }


def resolve_endpoints(version, previous, args):
    """Endpoints default to the primus build commits, which are the pinned
    release commits; the JAX image is built from main at a different point and is
    recorded separately by preflight."""
    from_commit = args.from_commit
    to_commit = args.to_commit
    if not to_commit:
        snapshot = C.load_snapshot(version, "primus") or C.load_snapshot(version, "jax")
        to_commit = (snapshot or {}).get("build_commit")
    if not from_commit and previous:
        snapshot = C.load_snapshot(previous, "primus") or C.load_snapshot(previous, "jax")
        from_commit = (snapshot or {}).get("build_commit")
    return from_commit, to_commit


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--version", required=True)
    parser.add_argument("--previous")
    parser.add_argument("--from", dest="from_commit", help="override the start commit")
    parser.add_argument("--to", dest="to_commit", help="override the end commit")
    args = parser.parse_args()

    previous = args.previous
    if not previous:
        others = sorted(
            {v for v, _, _ in C.available_snapshots() if v != args.version}, key=C.version_key, reverse=True
        )
        previous = others[0] if others else None

    from_commit, to_commit = resolve_endpoints(args.version, previous, args)
    if not from_commit or not to_commit:
        raise SystemExit(
            "ERROR: could not resolve the commit range. Probe both releases first, "
            "or pass --from/--to explicitly."
        )

    data = collect(args.version, previous, from_commit, to_commit)
    path = C.ROOT / f"output/release-docs/{args.version}/changelog.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")

    rng = data["range"]
    print(f"changelog {previous} -> {args.version}  ({rng['from'][:8]}..{rng['to'][:8]})")
    print(
        f"  commits              {rng['commits']} ({rng['cherry_picks_suppressed']} cherry-picks suppressed)"
    )
    print(f"  buckets              {ordered_summary(data['buckets'])}")
    print(f"  areas                {', '.join(f'{k}:{len(v)}' for k, v in sorted(data['areas'].items()))}")
    print(f"  new recipes          {len(data['new_recipes'])}")
    for name, bump in sorted(data["submodule_bumps"].items()):
        print(f"  submodule            {name} {bump['from']}..{bump['to']}")
    print(f"  wrote                {path.relative_to(C.ROOT)}")
    return 0


def ordered_summary(buckets):
    return ", ".join(f"{name}:{len(buckets[name])}" for name in BUCKET_ORDER if name in buckets)


if __name__ == "__main__":
    raise SystemExit(main())
