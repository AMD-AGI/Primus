###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Render and verify the image tables in docs/01-getting-started/release-notes.md.

    render  emit the markdown for a release: per-image tables plus the
            "Changes since" delta against the previous release.
    check   re-read the committed markdown and assert every version it states
            matches the image snapshot it claims to describe.

`check` is the load-bearing half. The release notes carry ~25 version strings
per image; a wrong hipBLASLt tweak hash is invisible in review but a customer
hits it immediately. Running this in CI means the page cannot drift from the
images it documents, and prose an editor adds to a cell (`7.15.0 (rocm-sdk
7.15.0a20260727)`) is preserved because a cell only has to *contain* the
extracted value, not equal it.

Stdlib-only: this runs in the lint job.

Usage:
    python tools/release_docs/release_notes.py check
    python tools/release_docs/release_notes.py render --version v26.7
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common as C  # noqa: E402
import components  # noqa: E402

IMAGE_HEADING = re.compile(r"^### `(rocm/[^`]+)`\s*$", re.MULTILINE)
TABLE_ROW = re.compile(r"^\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*$")

# Metadata rows we can verify against a snapshot; others (e.g. "Also published
# as") are informational and skipped rather than flagged.
META_FIELDS = {
    "Image ID": "image_id",
    "Built": "built",
    "Size": "size",
    "Manifest": "manifest_version",
}


def parse_image_ref(ref):
    """'rocm/jax-training:maxtext-v26.6' -> ('jax', 'v26.6')."""
    tag = ref.split(":", 1)[1] if ":" in ref else ""
    version = tag.replace("maxtext-", "")
    family = C.family_for_image(ref)
    return family, version


def strip_cell(value):
    """Drop backticks and markdown link wrappers so comparisons see the text."""
    value = value.strip()
    value = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", value)
    return value.replace("`", "").strip()


def parse_image_sections(text):
    """Every '### `rocm/...`' block with its metadata and component rows."""
    sections = []
    matches = list(IMAGE_HEADING.finditer(text))
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        # Stop at the next heading of any level. Without this the block runs on
        # into sibling '### Primus source for vX.Y' and '### Changes since vX.Y'
        # sections, whose tables are not image contents.
        next_heading = re.search(r"^#{2,6} ", text[start:end], re.MULTILINE)
        if next_heading:
            end = start + next_heading.start()
        body = text[start:end]

        meta = {}
        components_rows = []
        in_components = False
        for line in body.splitlines():
            row = TABLE_ROW.match(line)
            if not row:
                continue
            left, right = row.group(1).strip(), row.group(2).strip()
            if set(left) <= set("- ") and set(right) <= set("- "):
                continue
            if left == "Software component" and right == "Version":
                in_components = True
                continue
            if in_components:
                components_rows.append((left, strip_cell(right)))
            elif left:
                meta[left] = strip_cell(right)

        family, version = parse_image_ref(match.group(1))
        sections.append(
            {
                "image": match.group(1),
                "family": family,
                "version": version,
                "meta": meta,
                "components": components_rows,
                "line": text[: match.start()].count("\n") + 1,
            }
        )
    return sections


def check(args):
    text = C.RELEASE_NOTES.read_text()
    sections = parse_image_sections(text)
    failures = []
    warnings = []
    skipped = []
    verified = 0

    for section in sections:
        snapshot = C.load_snapshot(section["version"], section["family"])
        where = f"{C.RELEASE_NOTES.name}:{section['line']} {section['image']}"
        if snapshot is None:
            skipped.append(f"{where}: no snapshot committed")
            continue
        if snapshot.get("source") != "image":
            skipped.append(f"{where}: snapshot source is {snapshot.get('source')!r}, not image-derived")
            continue

        for label, field in META_FIELDS.items():
            if label not in section["meta"]:
                continue
            stated = section["meta"][label]
            actual = snapshot.get(field)
            if actual is None:
                warnings.append(f"{where}: '{label}' stated as {stated!r} but snapshot has no {field}")
                continue
            if str(actual) not in stated:
                failures.append(f"{where}: '{label}' says {stated!r}, image has {actual!r}")
            else:
                verified += 1

        dockerfile = section["meta"].get("Dockerfile")
        if dockerfile:
            expected = C.dockerfile_for(section["family"], section["version"]).name
            if expected not in dockerfile:
                failures.append(f"{where}: 'Dockerfile' row points at {dockerfile!r}, expected {expected!r}")
            else:
                verified += 1

        for label, stated in section["components"]:
            spec = components.spec_for_label(section["family"], label)
            if spec is None:
                warnings.append(f"{where}: no rule for component row {label!r} (stated {stated!r})")
                continue
            actual = components.resolve(spec, snapshot)
            if actual is None:
                warnings.append(f"{where}: {label!r} stated {stated!r} but not found in the image")
                continue
            missing = [part for part in str(actual).split(" / ") if part != "?" and part not in stated]
            if missing:
                failures.append(f"{where}: {label!r} says {stated!r}, image has {actual!r}")
            else:
                verified += 1

    print(f"release-notes check: {verified} values verified against image snapshots")
    for note in skipped:
        print(f"  SKIP  {note}")
    for note in warnings:
        print(f"  WARN  {note}")
    for note in failures:
        print(f"  FAIL  {note}")
    if failures:
        print(f"\nFAILED: {len(failures)} stated value(s) do not match the images.")
        return 1
    if warnings and args.strict:
        print(f"\nFAILED (strict): {len(warnings)} warning(s).")
        return 1
    print("OK")
    return 0


def render_image_block(snapshot, previous=None):
    """The '### `image`' block: metadata table then component table."""
    family = snapshot["family"]
    lines = [f"### `{snapshot['image']}`", ""]
    lines.append(C.FAMILIES[family]["backends"])
    lines.append("")
    lines.append("| | |")
    lines.append("| --- | --- |")
    lines.append(f"| Image ID | `{snapshot['image_id']}` |")
    lines.append(f"| Built | {snapshot['built']} |")
    lines.append(f"| Size | {snapshot['size']} |")
    lines.append(f"| Manifest | `{snapshot['manifest_version']}` |")
    dockerfile = C.dockerfile_for(family, snapshot["version"])
    rel = dockerfile.relative_to(C.ROOT).as_posix()
    lines.append(f"| Dockerfile | [`{dockerfile.name}`](https://github.com/AMD-AGI/Primus/blob/main/{rel}) |")
    lines.append("")
    lines.append("| Software component | Version |")
    lines.append("| ------------------ | ------- |")
    for label, value in components.rows_for(snapshot):
        lines.append(f"| {label} | {value} |")
    return "\n".join(lines)


def render_delta(current, previous):
    """The 'Changes since' table: only rows whose value actually moved."""
    if previous is None:
        return None
    rows = []
    for spec in components.SPECS[current["family"]]:
        new_value = components.resolve(spec, current)
        old_value = components.resolve(spec, previous)
        if new_value is None and old_value is None:
            continue
        if new_value != old_value:
            label = components.label_for(spec, current)
            rows.append((label, old_value or "-", new_value or "-"))
    if current.get("size") != previous.get("size"):
        rows.append(("Image size", previous.get("size", "-"), current.get("size", "-")))
    if not rows:
        return None
    header = [
        f"| Component | {previous['version']} | {current['version']} |",
        "| --------- | ----- | ----- |",
    ]
    return "\n".join(header + [f"| {label} | {old} | {new} |" for label, old, new in rows])


def render(args):
    previous_version = args.previous
    if not previous_version:
        others = sorted(
            {version for version, _, _ in C.available_snapshots() if version != args.version},
            key=C.version_key,
            reverse=True,
        )
        previous_version = others[0] if others else None

    out = []
    for family in ("primus", "jax"):
        snapshot = C.load_snapshot(args.version, family)
        if snapshot is None:
            continue
        out.append(render_image_block(snapshot))
        out.append("")

    deltas = []
    for family in ("primus", "jax"):
        snapshot = C.load_snapshot(args.version, family)
        previous = C.load_snapshot(previous_version, family) if previous_version else None
        delta = render_delta(snapshot, previous) if snapshot else None
        if delta:
            deltas.append((family, delta))

    # Under its own heading, matching the committed structure, so the rendered
    # markdown round-trips through parse_image_sections() instead of the delta
    # rows being read as image contents.
    if deltas:
        out.append(f"### Changes since {previous_version}")
        out.append("")
        for family, delta in deltas:
            out.append(f"`{C.FAMILIES[family]['image'].format(version='')[:-1]}`:")
            out.append("")
            out.append(delta)
            out.append("")

    if not out:
        raise SystemExit(f"ERROR: no snapshots found for {args.version}. Run probe_image.py first.")
    print("\n".join(out).rstrip())
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    check_parser = sub.add_parser("check", help="verify the committed notes against image snapshots")
    check_parser.add_argument("--strict", action="store_true", help="treat warnings as failures")
    check_parser.set_defaults(func=check)

    render_parser = sub.add_parser("render", help="emit markdown for a release")
    render_parser.add_argument("--version", required=True)
    render_parser.add_argument("--previous", help="previous version for the delta table")
    render_parser.set_defaults(func=render)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
