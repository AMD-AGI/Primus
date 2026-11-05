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
hits it immediately. Run it after any edit to the page and the page cannot drift
from the images it documents, and prose an editor adds to a cell (`7.15.0 (rocm-sdk
7.15.0a20260727)`) is preserved because a cell only has to *contain* the
extracted value, not equal it.

Nothing runs this automatically -- it is a Phase 5 gate in the release-docs skill,
so a release that skips it ships unverified numbers.

Stdlib-only, so it needs no virtualenv wherever it is run.

    rotate  insert the release section (or one family's part of it), move the
            previous release's Highlights into that release's own section, and
            keep three detailed releases.

The two families are not released in lockstep, so `render --family` and
`rotate --family` work on one family's image block, its "Changes since" table
and its Highlights subsection inside a release section that may already hold the
other family's.

Usage:
    python tools/release_docs/release_notes.py check
    python tools/release_docs/release_notes.py render --version v26.7
    python tools/release_docs/release_notes.py render --version v26.9 --family primus
    python tools/release_docs/release_notes.py rotate --version v26.9 --family primus \
        --body body.md --highlights highlights.md
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


def next_heading_offset(text):
    """Character offset of the next markdown heading, ignoring fenced blocks."""
    offset = 0
    in_fence = False
    for line in text.splitlines(keepends=True):
        if re.match(r"\s*(```|~~~)", line):
            in_fence = not in_fence
        elif not in_fence and re.match(r"#{2,6} ", line):
            return offset
        offset += len(line)
    return None


def parse_image_sections(text):
    """Every '### `rocm/...`' block with its metadata and component rows."""
    sections = []
    matches = list(IMAGE_HEADING.finditer(text))
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        # Stop at the next heading of any level. Without this the block runs on
        # into sibling '### Primus source for vX.Y' and '### Changes since vX.Y'
        # sections, whose tables are not image contents. Fenced blocks are skipped
        # while looking, because '## ' inside a bash fence is a comment, not a
        # heading, and would truncate the section early.
        offset = next_heading_offset(text[start:end])
        if offset is not None:
            end = start + offset
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

    # An interrupted Phase 5 can leave two sections for the same image if the
    # rotation is re-run and inserts rather than updates. Both would verify
    # individually, so the duplicate has to be caught structurally.
    seen = {}
    for section in sections:
        key = (section["family"], section["version"])
        if key in seen:
            failures.append(
                f"{C.RELEASE_NOTES.name}:{section['line']} duplicate section for {section['image']} "
                f"(already at line {seen[key]}) -- the rotation inserted instead of updating"
            )
        else:
            seen[key] = section["line"]

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

    print(
        f"release-notes check: {verified} values verified across "
        f"{len(sections) - len(skipped)}/{len(sections)} documented images"
    )
    for note in skipped:
        print(f"  SKIP  {note}")
    for note in warnings:
        print(f"  WARN  {note}")
    for note in failures:
        print(f"  FAIL  {note}")
    if failures:
        print(f"\nFAILED: {len(failures)} stated value(s) do not match the images.")
        return 1
    # A skip means a documented image was not checked at all. Reporting that as a
    # pass would let coverage quietly fall to zero if snapshots went missing, so
    # the count is always in the headline and --strict refuses to tolerate it.
    if args.strict and (warnings or skipped):
        print(f"\nFAILED (strict): {len(warnings)} warning(s), {len(skipped)} unchecked image(s).")
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


def previous_snapshot(version, family, previous_version=None):
    """The release this family's delta is against: --previous, else the newest
    older snapshot of the same family (MaxText has patch releases primus lacks)."""
    if previous_version:
        return C.load_snapshot(previous_version, family)
    older = [
        data
        for snap_version, snap_family, data in C.available_snapshots()
        if snap_family == family and C.version_key(snap_version) < C.version_key(version)
    ]
    return older[0] if older else None


def render(args):
    families = args.family or ["primus", "jax"]
    out = []
    for family in families:
        snapshot = C.load_snapshot(args.version, family)
        if snapshot is None:
            continue
        out.append(render_image_block(snapshot))
        out.append("")

    deltas = []
    for family in families:
        snapshot = C.load_snapshot(args.version, family)
        previous = previous_snapshot(args.version, family, args.previous) if snapshot else None
        delta = render_delta(snapshot, previous) if snapshot else None
        if delta:
            deltas.append((family, previous["version"], delta))

    # Under its own heading, matching the committed structure, so the rendered
    # markdown round-trips through parse_image_sections() instead of the delta
    # rows being read as image contents.
    if deltas:
        previous_version = args.previous or max((v for _, v, _ in deltas), key=C.version_key)
        out.append(f"### Changes since {previous_version}")
        out.append("")
        for family, _, delta in deltas:
            out.append(f"{family_label(family)}:")
            out.append("")
            out.append(delta)
            out.append("")

    if not out:
        raise SystemExit(f"ERROR: no snapshots found for {args.version}. Run probe_image.py first.")
    print("\n".join(out).rstrip())
    return 0


def family_label(family):
    """The label a family's part of 'Changes since' starts with: `rocm/primus`:"""
    return f"`{C.FAMILIES[family]['image'].format(version='')[:-1]}`"


DETAILED_SECTIONS = 3


def release_section_bounds(text):
    """(version, start, end) for every '## vX.Y' release section, in page order."""
    heads = [
        (match.group(1), match.start())
        for match in re.finditer(r"^## (v[\d.]+)(?: \(current\))?\s*$", text, re.MULTILINE)
    ]
    bounds = []
    for index, (version, start) in enumerate(heads):
        if index + 1 < len(heads):
            end = heads[index + 1][1]
        else:
            following = re.search(r"^## (?!v[\d.])", text[start + 1 :], re.MULTILINE)
            end = start + 1 + following.start() if following else len(text)
        bounds.append((version, start, end))
    return bounds


HEADING = re.compile(r"^(#{1,6}) ")
FENCE = re.compile(r"\s*(```|~~~)")
HIGHLIGHTS_HEADING = re.compile(r"^## Highlights for (v[\d.]+)\s*$", re.MULTILINE)
CHANGE_LABEL = re.compile(r"^`(rocm/[^`]+)`:\s*$")


def split_blocks(text, level):
    """Split markdown into (preamble, [block, ...]) at headings of exactly `level`,
    ignoring '#' lines inside fenced code. A block runs from its heading to the
    next heading of the same or a higher level."""
    preamble, blocks, current = [], [], None
    in_fence = False
    for line in text.splitlines(keepends=True):
        if FENCE.match(line):
            in_fence = not in_fence
        match = None if in_fence else HEADING.match(line)
        if match and len(match.group(1)) <= level:
            if len(match.group(1)) < level:
                raise ValueError(f"unexpected heading inside a block: {line.strip()}")
            current = [line]
            blocks.append(current)
            continue
        (current if current is not None else preamble).append(line)
    return "".join(preamble), ["".join(block) for block in blocks]


def heading_of(block):
    return block.splitlines()[0].strip()


def strip_rule(text):
    """Drop a trailing '---' separator and surrounding blank lines."""
    text = text.rstrip()
    if text.endswith("\n---") or text == "---":
        text = text[: -len("---")].rstrip()
    return text


def block_family(heading):
    """'### `rocm/primus:v26.9`' or '### `rocm/primus` — ...' -> 'primus'."""
    match = re.search(r"`(rocm/[^`]+)`", heading)
    return C.family_for_image(match.group(1)) if match else None


def section_rank(heading):
    """Where a block goes inside a release section."""
    family = block_family(heading)
    if family and IMAGE_HEADING.match(heading):
        return 0 if family == "primus" else 1
    if heading.startswith("### Changes since"):
        return 3
    return 2


def highlights_rank(heading):
    family = block_family(heading)
    if heading.lower().startswith("### both families"):
        return 0
    if family:
        return 1 if family == "primus" else 2
    return 3


def merge_blocks(existing, incoming, rank, merge_one=None):
    """Replace the blocks of `existing` whose heading matches one in `incoming`,
    insert the others before the first existing block of a higher rank."""
    merged = list(existing)
    for block in incoming:
        heading = heading_of(block)
        same = [i for i, old in enumerate(merged) if heading_of(old) == heading]
        if not same and heading.startswith("### Changes since"):
            same = [i for i, old in enumerate(merged) if heading_of(old).startswith("### Changes since")]
        if same:
            i = same[0]
            merged[i] = merge_one(merged[i], block) if merge_one else block
            continue
        position = next(
            (i for i, old in enumerate(merged) if rank(heading_of(old)) > rank(heading)), len(merged)
        )
        merged.insert(position, block)
    return merged


def merge_changes(existing, incoming):
    """Merge one family's part of a 'Changes since' block into the existing one."""

    def parts(block):
        head, items, current = [], [], None
        for line in block.splitlines(keepends=True):
            match = CHANGE_LABEL.match(line.strip())
            if match:
                current = [C.family_for_image(match.group(1)), [line]]
                items.append(current)
            elif current is not None:
                current[1].append(line)
            else:
                head.append(line)
        return "".join(head), [(family, "".join(lines)) for family, lines in items]

    head, old_parts = parts(existing)
    _, new_parts = parts(incoming)
    order = {"primus": 0, "jax": 1}
    merged = dict(old_parts)
    merged.update(new_parts)
    body = "".join(
        merged[family].rstrip() + "\n\n" for family in sorted(merged, key=lambda f: order.get(f, 9))
    )
    return head.rstrip() + "\n\n" + body


def join_section(preamble, blocks):
    parts = [preamble.rstrip()] if preamble.strip() else []
    parts += [block.rstrip() for block in blocks]
    return "\n\n".join(parts)


def shift_headings(text, levels=1):
    """Every heading one level deeper, fenced code left alone."""
    out, in_fence = [], False
    for line in text.splitlines(keepends=True):
        if FENCE.match(line):
            in_fence = not in_fence
        if not in_fence and HEADING.match(line):
            line = "#" * levels + line
        out.append(line)
    return "".join(out)


def highlights_bounds(text):
    """(version, start, end) of the top '## Highlights for vX.Y' block, or None."""
    match = HIGHLIGHTS_HEADING.search(text)
    if not match:
        return None
    following = re.search(r"^## ", text[match.end() :], re.MULTILINE)
    end = match.end() + following.start() if following else len(text)
    return match.group(1), match.start(), end


def demote_highlights(text):
    """Move the top Highlights into its own release's section as '### Highlights
    for vX.Y', below the section heading, where the page keeps its history."""
    bounds = highlights_bounds(text)
    if not bounds:
        return text, None
    version, start, end = bounds
    body = strip_rule(text[start:end])
    text = text[:start] + text[end:]
    sections = {entry[0]: entry for entry in release_section_bounds(text)}
    if version not in sections:
        print(f"  NOTE: the Highlights for {version} were dropped: it has no detailed section")
        return text, version
    _, section_start, _ = sections[version]
    heading_end = text.index("\n", section_start) + 1
    demoted = shift_headings(body).strip() + "\n\n"
    text = text[:heading_end] + "\n" + demoted + text[heading_end:].lstrip("\n")
    return text, version


def place_highlights(text, version, body, family):
    """Insert or merge '## Highlights for <version>' above the first release section."""
    body = strip_rule(body)
    if body.startswith("## Highlights for"):
        body = body.split("\n", 1)[1] if "\n" in body else ""
    bounds = highlights_bounds(text)
    if bounds and bounds[0] == version:
        _, start, end = bounds
        current = strip_rule(text[start:end]).split("\n", 1)[1]
        if family:
            old_pre, old_blocks = split_blocks(current, 3)
            new_pre, new_blocks = split_blocks(body, 3)
            merged = merge_blocks(old_blocks, new_blocks, highlights_rank)
            body = join_section(new_pre if new_pre.strip() else old_pre, merged)
        text = text[:start] + text[end:]
    sections = release_section_bounds(text)
    position = sections[0][1] if sections else len(text)
    block = f"## Highlights for {version}\n\n{body.strip()}\n\n---\n\n"
    return text[:position] + block + text[position:]


def rotate(args):
    """Insert the new release, demote the previous one, evict the fourth-oldest.

    Doing this by hand is where the first real run went wrong: a hand-rolled splice
    left the previous release's image blocks under the new heading, and a stray
    search-and-replace reached into the historical sections. The edit is purely
    structural, so it belongs here rather than in an editor.

    With --family only that family's blocks of the body are merged into an
    existing section (its image block, its part of 'Changes since', and any other
    block the body carries, such as 'Primus source'); the other family's are kept.
    --highlights does the same for the Highlights, and the previous release's
    Highlights move into that release's section either way.

    Idempotent: re-running replaces an existing section for the same version
    instead of inserting a second one, which is the failure `check` guards against.
    """
    version = args.version
    if args.family and len(args.family) > 1:
        raise SystemExit("ERROR: rotate --family takes one family; leave it out to write both")
    family = args.family[0] if args.family else None
    text = C.RELEASE_NOTES.read_text()
    body = Path(args.body).read_text().strip() if args.body else ""
    if not body:
        raise SystemExit("ERROR: --body is required (the rendered section, from `render`)")

    sections = release_section_bounds(text)
    if not sections:
        raise SystemExit("ERROR: found no '## vX.Y' release sections to rotate")

    bounds = highlights_bounds(text)
    if bounds and bounds[0] != version:
        text, moved = demote_highlights(text)
        print(f"moved the Highlights for {moved} into its own section")
        sections = release_section_bounds(text)

    known = [entry[0] for entry in sections]
    if version in known and family:
        _, start, end = next(entry for entry in sections if entry[0] == version)
        current = strip_rule(text[start:end])
        heading, _, rest = current.partition("\n")
        old_pre, old_blocks = split_blocks(rest, 3)
        new_pre, new_blocks = split_blocks(body, 3)
        merged = merge_blocks(old_blocks, new_blocks, section_rank, merge_changes_if_changes)
        section = f"{heading}\n\n{join_section(new_pre if new_pre.strip() else old_pre, merged)}\n\n---\n\n"
        text = text[:start] + section + text[end:]
        print(f"merged the {family} part into the existing {version} section")
    elif version in known:
        _, start, end = next(entry for entry in sections if entry[0] == version)
        text = text[:start] + f"## {version} (current)\n\n{body}\n\n---\n\n" + text[end:]
        print(f"replaced the existing {version} section in place")
    else:
        text = (
            text[: sections[0][1]] + f"## {version} (current)\n\n{body}\n\n---\n\n" + text[sections[0][1] :]
        )
        print(f"inserted {version} as the current release")

    for other in known:
        if other != version:
            text = text.replace(f"## {other} (current)\n", f"## {other}\n", 1)

    if args.highlights:
        text = place_highlights(text, version, Path(args.highlights).read_text(), family)
        print(f"  highlights for {version} {'merged (' + family + ')' if family else 'written'}")

    evicted = []
    sections = release_section_bounds(text)
    while len(sections) > DETAILED_SECTIONS:
        old_version, start, end = sections[-1]
        text = text[:start] + text[end:]
        evicted.append(old_version)
        sections = release_section_bounds(text)

    C.RELEASE_NOTES.write_text(text)
    print(f"  detailed sections now: {', '.join(entry[0] for entry in release_section_bounds(text))}")
    if evicted:
        print(f"  evicted: {', '.join(evicted)}")
        print("  NEXT: add a headline row for each evicted release under '## Earlier releases',")
        print("        then run check_links.py -- removing a section breaks inbound anchors.")
    return 0


def merge_changes_if_changes(old, new):
    if heading_of(old).startswith("### Changes since"):
        return merge_changes(old, new)
    return new


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
    render_parser.add_argument(
        "--family", action="append", choices=["primus", "jax"], help="render only this family (repeatable)"
    )
    render_parser.set_defaults(func=render)

    rotate_parser = sub.add_parser(
        "rotate", help="splice a release section in and evict the oldest detailed one"
    )
    rotate_parser.add_argument("--version", required=True)
    rotate_parser.add_argument(
        "--body", required=True, help="file holding the section body (render output plus your prose)"
    )
    rotate_parser.add_argument(
        "--family",
        action="append",
        choices=["primus", "jax"],
        help="merge only this family's part into an existing section (one family)",
    )
    rotate_parser.add_argument(
        "--highlights", help="file holding the Highlights body (subsections per family, as on the page)"
    )
    rotate_parser.set_defaults(func=rotate)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
