###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Check in-repo markdown links and heading anchors.

A release update renames headings and removes whole sections, and both break
inbound links silently. In the first real run, bumping "Important notes for v26.6"
to v26.7 killed `#important-notes-for-v266` in another page, three links had their
text bumped to v26.7 while still pointing at the v26.6 anchor, and rotating v26.4
out of the detailed sections orphaned `#primus-source-for-v264`. None of that is
visible to a version-string check, and none of it fails a build.

Only relative links are followed; external URLs are left alone deliberately, so
this is fast and offline.

Pre-existing breakage is recorded in a baseline so this can gate CI without
demanding an unrelated docs cleanup first. New breakage fails; fixing a
baselined entry prompts you to shrink the baseline.

Stdlib-only: this runs in the lint job.

Usage:
    python tools/release_docs/check_links.py
    python tools/release_docs/check_links.py --update-baseline
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common as C  # noqa: E402

BASELINE = Path(__file__).resolve().parent / "link_baseline.json"

SEARCH_ROOTS = ("docs", "tools/installation", "tools/installation-jax", "skills")
EXTRA_FILES = ("README.md",)

LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$", re.MULTILINE)
FENCE = re.compile(r"^\s*(```|~~~)")


def slugify(heading):
    """GitHub's heading-anchor rules, closely enough for link checking.

    Lowercase, drop anything that is not alphanumeric, space, hyphen or
    underscore, then map each remaining space to a hyphen. Note *each* space, not
    runs of them: "## Highlights - v26.7" with a stripped dash leaves two spaces
    and therefore a double hyphen, which is exactly the anchor that tripped this
    up during the release.
    """
    text = heading.strip().lower()
    text = re.sub(r"<[^>]+>", "", text)  # inline HTML
    text = re.sub(r"[`*_~]", "", text)  # markdown emphasis
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # links keep their text
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    return text.replace(" ", "-")


def headings_and_anchors(text):
    """Anchors a markdown file exposes, skipping fenced blocks."""
    anchors = set()
    in_fence = False
    for line in text.splitlines():
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = HEADING.match(line)
        if match:
            anchors.add(slugify(match.group(2)))
    # Explicit <a name=...> / id=... targets.
    anchors.update(re.findall(r'<a[^>]+(?:name|id)="([^"]+)"', text))
    return anchors


def markdown_files():
    files = []
    for root in SEARCH_ROOTS:
        base = C.ROOT / root
        if base.exists():
            files.extend(sorted(base.rglob("*.md")))
    for name in EXTRA_FILES:
        path = C.ROOT / name
        if path.exists():
            files.append(path)
    return [f for f in files if "third_party" not in f.parts]


def check(files):
    """Every broken relative link, as 'path: kind -> target' strings."""
    anchors_cache = {}

    def anchors_of(path):
        if path not in anchors_cache:
            try:
                anchors_cache[path] = headings_and_anchors(path.read_text())
            except (OSError, UnicodeDecodeError):
                anchors_cache[path] = set()
        return anchors_cache[path]

    problems = []
    for path in files:
        try:
            text = path.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        try:
            rel = path.relative_to(C.ROOT).as_posix()
        except ValueError:
            rel = path.as_posix()  # a file outside the repo (tests, ad-hoc checks)
        for target in LINK.findall(text):
            if target.startswith(("http://", "https://", "mailto:", "#!")):
                continue
            file_part, _, anchor = target.partition("#")
            if file_part:
                resolved = (path.parent / file_part).resolve()
                if not resolved.exists():
                    problems.append(f"{rel}: missing file -> {target}")
                    continue
                if anchor and resolved.suffix == ".md":
                    if anchor not in anchors_of(resolved):
                        problems.append(f"{rel}: dead anchor -> {target}")
            elif anchor and anchor not in anchors_of(path):
                problems.append(f"{rel}: dead anchor -> {target}")
    return sorted(set(problems))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--update-baseline", action="store_true", help="record the current problems as the accepted baseline"
    )
    args = parser.parse_args()

    problems = check(markdown_files())

    if args.update_baseline:
        BASELINE.write_text(json.dumps({"known_problems": problems}, indent=2) + "\n")
        print(f"baseline updated: {len(problems)} known problem(s) recorded in {BASELINE.name}")
        return 0

    baseline = set(json.loads(BASELINE.read_text())["known_problems"]) if BASELINE.exists() else set()
    new = [p for p in problems if p not in baseline]
    fixed = sorted(baseline - set(problems))

    print(f"link check: {len(problems)} problem(s), {len(baseline)} baselined, {len(new)} new")
    for problem in new:
        print(f"  NEW  {problem}")
    if fixed:
        print(
            f"  {len(fixed)} baselined problem(s) are now fixed; re-run with --update-baseline to shrink it:"
        )
        for problem in fixed[:10]:
            print(f"    fixed: {problem}")
    if new:
        print("\nFAILED: new broken links. A renamed heading or a removed section orphans anchors.")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
