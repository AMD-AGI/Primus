###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Move the repository's version references from one release to the next.

Roughly 120 references to a release are spread across 30+ files, and they are
not all the same kind of thing. Image tags and Dockerfile filenames always move.
Sentences like "v26.6 dropped the ROCm git fork" are history that must stay.
`release/vX.Y` is neither: it depends on whether that branch exists yet, because
writing `git checkout release/v26.7` before the branch is cut ships a broken
instruction (the v26.4 section of the release notes shows the commit-form
alternative).

So every hit is classified by tools/release_docs/version_bump_rules.json into
rewrite / conditional / hold, and only the first two are ever touched. Held hits
are reported for a decision rather than guessed at.

Matching is deliberately narrow -- `v26.6`, `primus==26.6`, `__version__ =
"26.6"` -- so that `26.6GB` in a memory log and `rpds-py==2026.6.3` are never
candidates in the first place.

Usage:
    python tools/release_docs/bump_version.py --from v26.6 --to v26.7          # dry run
    python tools/release_docs/bump_version.py --from v26.6 --to v26.7 --apply
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common as C  # noqa: E402

RULES_FILE = Path(__file__).resolve().parent / "version_bump_rules.json"

EXAMPLES_PER_FILE = 3

BINARY_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".pdf",
    ".whl",
    ".so",
    ".pyc",
    ".ico",
    ".woff",
    ".woff2",
}


# A release token must not match a patch release that merely starts with it:
# without this, bumping v26.5 rewrites `Dockerfile.primus-v26.5.1` to
# `Dockerfile.primus-v26.6.1`, a file that does not exist. `\b` alone is not
# enough because a '.' is a word boundary. Patch releases are real: v26.3.1,
# v26.3.2 and v26.5.1 all shipped.
NOT_A_PATCH_SUFFIX = r"(?!\.\d)"


def version_tokens(version):
    """The forms a release is referenced in, as regexes.

    Narrow on purpose: requiring the `v`, or an explicit `primus==` /
    `__version__` context, keeps `26.6GB` and `2026.6.3` out of the candidate
    set entirely.
    """
    bare = version.lstrip("v")
    escaped = re.escape(version)
    escaped_bare = re.escape(bare)
    # The anchor form is scoped to a '#...' fragment on purpose: 'v266' on its own
    # is too generic to treat as a release reference.
    escaped_slug = re.escape(slug_token(version))
    return [
        rf"{escaped}{NOT_A_PATCH_SUFFIX}\b",
        rf"primus==\s*{escaped_bare}(\.\d+)?",
        rf"__version__\s*=\s*([\"']){escaped_bare}\1",
        rf"#[a-z0-9-]*{escaped_slug}\b",
    ]


def slug_token(version):
    """'v26.6' -> 'v266', the form a version takes inside a heading anchor."""
    return version.replace(".", "")


def _substitute(template, old, new, escape):
    """Fill {old}/{new}/{old_bare}/{new_bare}/{old_slug}/{new_slug} in a pattern.

    In patterns (escape=True) the old v-prefixed token carries the patch-release
    guard, so a rule matching `Dockerfile.primus-{old}` cannot also consume
    `Dockerfile.primus-v26.5.1`. `{old_bare}` deliberately does not, because the
    pip-spec rule matches `primus==26.5.0` and needs the trailing `.0`.
    """
    prepare = re.escape if escape else (lambda value: value)
    old_token = prepare(old) + (NOT_A_PATCH_SUFFIX if escape else "")
    return (
        template.replace("{old_slug}", prepare(slug_token(old)))
        .replace("{new_slug}", prepare(slug_token(new)) if escape else slug_token(new))
        .replace("{old_bare}", prepare(old.lstrip("v")))
        .replace("{new_bare}", prepare(new.lstrip("v")))
        .replace("{old}", old_token)
        .replace("{new}", prepare(new))
    )


def in_rewrite_scope(rel_path, scope):
    """Whether a file is part of the documentation surface this tool may edit.

    Outside it, references pin an image deliberately -- examples/mlperf/ records
    what a submission was validated against -- so they are held instead.
    """
    return any(rel_path == prefix or rel_path.startswith(prefix) for prefix in scope)


def load_rules(old, new):
    """Compile the rule table: match pattern, path scope, and the rule's own
    replacements. Replacements are per-rule so a line mixing a live image tag
    with a historical sentence only has the tag rewritten."""
    config = json.loads(RULES_FILE.read_text())
    compiled = []
    for rule in config["rules"]:
        match = re.compile(_substitute(rule["match"], old, new, escape=True), re.IGNORECASE)
        replace = [
            (
                re.compile(_substitute(pattern, old, new, escape=True)),
                _substitute(replacement, old, new, escape=False),
            )
            for pattern, replacement in rule.get("replace", [])
        ]
        compiled.append(
            {
                "name": rule["name"],
                "action": rule["action"],
                "match": match,
                "paths": tuple(rule.get("paths", ())),
                "replace": replace,
                "why": rule["why"],
            }
        )
    return config["excluded_paths"], compiled, config["rewrite_scope"]


def tracked_files():
    result = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=False, cwd=C.ROOT)
    if result.returncode != 0:
        raise SystemExit("ERROR: git ls-files failed; run inside the repository.")
    return [line for line in result.stdout.splitlines() if line]


def classify(line, rules, suffix):
    for rule in rules:
        if rule["paths"] and suffix not in rule["paths"]:
            continue
        if rule["match"].search(line):
            return rule
    return {"name": "default-prose", "action": "hold", "replace": [], "why": "unclassified"}


def scan(old, new, branch_exists):
    excluded, rules, scope = load_rules(old, new)
    tokens = [re.compile(token) for token in version_tokens(old)]
    hits = []

    for rel_path in tracked_files():
        if any(rel_path.startswith(prefix) for prefix in excluded):
            continue
        path = C.ROOT / rel_path
        if path.suffix.lower() in BINARY_SUFFIXES or not path.is_file():
            continue
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        if not any(token.search(text) for token in tokens):
            continue

        in_fence = False
        for number, line in enumerate(text.splitlines(), start=1):
            # Inside a fenced block '#' starts a shell comment, not a heading, so
            # the heading rule must not see these lines: bare-metal-installation-jax.md
            # embeds '# v26.6: ROCm lives in the pip SDK' in a bash fence.
            if re.match(r"\s*(```|~~~)", line):
                in_fence = not in_fence
            if not any(token.search(line) for token in tokens):
                continue
            suffix = path.suffix.lower()
            rule = classify(line, rules, "" if in_fence and suffix == ".md" else suffix)
            action, why = rule["action"], rule["why"]
            if action != "hold" and not in_rewrite_scope(rel_path, scope):
                action = "hold"
                why = (
                    "outside the documentation surface: example, benchmark and launcher "
                    "scripts pin the image they were validated against"
                )
            if action == "conditional":
                # release/vX.Y: rewrite only when the new branch exists.
                action = "rewrite" if branch_exists else "hold"
                if not branch_exists:
                    why = f"{why} (branch missing: use the commit form)"
            hits.append(
                {
                    "path": rel_path,
                    "line": number,
                    "text": line.strip(),
                    "rule": rule["name"],
                    "action": action,
                    "why": why,
                    "replace": rule["replace"],
                }
            )
    return hits


def apply_rewrites(hits):
    """Apply each hit's own replacements to its own line.

    Line-scoped and rule-scoped: a held hit sharing a file -- or a line that
    mixes a live image tag with a historical sentence -- is never collaterally
    rewritten, because only the matching construct is substituted.
    """
    by_file = {}
    for hit in hits:
        if hit["action"] == "rewrite":
            by_file.setdefault(hit["path"], []).append(hit)

    changed = []
    for rel_path, file_hits in sorted(by_file.items()):
        path = C.ROOT / rel_path
        lines = path.read_text().splitlines(keepends=True)
        edits = 0
        for hit in sorted(file_hits, key=lambda item: item["line"]):
            index = hit["line"] - 1
            original = lines[index]
            updated = original
            for pattern, replacement in hit["replace"]:
                updated = pattern.sub(replacement, updated)
            if updated != original:
                lines[index] = updated
                edits += 1
        if edits:
            path.write_text("".join(lines))
            changed.append((rel_path, edits))
    return changed


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--from", dest="old", required=True, help="current version, e.g. v26.6")
    parser.add_argument("--to", dest="new", required=True, help="new version, e.g. v26.7")
    parser.add_argument("--apply", action="store_true", help="write the rewrites (default is a dry run)")
    parser.add_argument(
        "--branch-exists",
        action="store_true",
        help="the new release/vX.Y exists on origin, so checkout instructions use the branch form",
    )
    parser.add_argument("--json", dest="as_json", action="store_true", help="emit the hit list as JSON")
    args = parser.parse_args()

    hits = scan(args.old, args.new, args.branch_exists)
    rewrites = [hit for hit in hits if hit["action"] == "rewrite"]
    holds = [hit for hit in hits if hit["action"] == "hold"]

    if args.as_json:
        strip = lambda items: [  # noqa: E731 - compiled patterns are not serialisable
            {key: value for key, value in item.items() if key != "replace"} for item in items
        ]
        print(json.dumps({"rewrite": strip(rewrites), "hold": strip(holds)}, indent=2))
        return 0

    print(f"version bump {args.old} -> {args.new}: {len(hits)} references found")
    print(f"  rewrite {len(rewrites)}")
    print(f"  hold    {len(holds)} (need your decision)")
    print()

    by_rule = {}
    for hit in rewrites:
        by_rule.setdefault(hit["rule"], []).append(hit)
    print("MECHANICAL (will be rewritten):")
    for rule, items in sorted(by_rule.items()):
        files = sorted({item["path"] for item in items})
        print(f"  {rule:22} {len(items):3} hit(s) in {len(files)} file(s)")
        for path in files:
            count = sum(1 for item in items if item["path"] == path)
            print(f"      {path} ({count})")

    print("\nHELD (not rewritten -- decide per item):")
    by_rule = {}
    for hit in holds:
        by_rule.setdefault(hit["rule"], []).append(hit)
    for rule, items in sorted(by_rule.items()):
        print(f"  {rule} ({len(items)}) -- {items[0]['why']}")
        by_file = {}
        for hit in items:
            by_file.setdefault(hit["path"], []).append(hit)
        for path, file_hits in sorted(by_file.items()):
            print(f"    {path} ({len(file_hits)})")
            for hit in file_hits[:EXAMPLES_PER_FILE]:
                text = hit["text"] if len(hit["text"]) <= 96 else hit["text"][:93] + "..."
                print(f"      :{hit['line']}  {text}")
            if len(file_hits) > EXAMPLES_PER_FILE:
                print(f"      ... {len(file_hits) - EXAMPLES_PER_FILE} more (use --json for all)")

    install_holds = sum(
        1
        for hit in holds
        if hit["path"].startswith(("tools/installation", "docs/01-getting-started/bare-metal"))
    )
    if install_holds:
        print(
            f"\n  NOTE: {install_holds} of the held hits are in the bare-metal install scripts and their"
            "\n  docs. Those are rewritten in context by the install-alignment phase, which re-pins them"
            "\n  against the new release Dockerfile, so they do not need a decision here."
        )

    if args.apply:
        changed = apply_rewrites(hits)
        total = sum(count for _, count in changed)
        print(f"\nAPPLIED: {total} line(s) across {len(changed)} file(s)")
        for rel_path, count in changed:
            print(f"  {rel_path} ({count})")
    else:
        print("\nDry run. Re-run with --apply to write the mechanical rewrites.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
