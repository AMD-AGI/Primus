###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fail CI when a Primus YAML sets a backend key upstream no longer defines.

Every backend adapter accepts unknown keys silently, so a field that upstream
renamed or moved keeps parsing while doing nothing. Prints the drift as Markdown
for `$GITHUB_STEP_SUMMARY`:

    python tools/ci/check_config_schema.py --warn-only >> "$GITHUB_STEP_SUMMARY"

A key upstream looks to have merely renamed gets a table and a row of its own:
that is a fix someone applies one key at a time. The rest share a table where a
row is a set of configs, because dozens of dead keys are usually one preset's
worth of rot, and repeating the same three long paths for each of them buries
the part a reader has to act on.

TorchTitan, MaxText and Megatron each have their schema read from their own
`third_party/` submodule. A backend whose submodule is not checked out has
nothing to compare against and is skipped, which exits non-zero even under
`--warn-only`: a checkout that fetches only some of them drops that part of the
check, and a check that did not run must not look like one that passed.

A config that cannot be loaded (broken `extends:` path, unparseable YAML) fails
the check just like drift does -- an unchecked config must never be counted as a
passing one. `--warn-only` suppresses both.

Misplaced model-scoped keys get their own table. Such a key is not unknown --
some Primus config class does declare it -- but that class is built for one
model only, so setting the key on any other model is the same silent no-op.
This table never fails the check: fixing one moves training numerics for
whichever model owns the key, so it needs that owner and a convergence run.
"""

import argparse
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from primus.core.config.schema_check import (  # noqa: E402
    BACKENDS,
    DEFAULT_ALLOWLIST,
    KeyFinding,
    build_allowlist,
    check_manual_allowlist_consumers,
    dedupe,
    dedupe_scoped,
    scan_backend,
)

SHOWN_PATTERNS = 40
SHOWN_FILES = 3


def shorten_paths(paths: Iterable[str]) -> dict[str, str]:
    """Map every path to the fewest trailing segments no other path here also ends with.

    Almost every config sits under one of a handful of roots, so the shared
    `primus/configs/models/megatron/` is pure repetition -- but only where
    dropping it still names one file, and `MI300X/` and `MI355X/` hold configs
    with identical names.
    """
    parts = {path: path.split("/") for path in set(paths)}
    tails = Counter("/".join(s[-d:]) for s in parts.values() for d in range(1, len(s) + 1))
    short = {}
    for path, segments in parts.items():
        # The whole path always names itself, so it is the fallback rather than a candidate.
        candidates = ("/".join(segments[-d:]) for d in range(1, len(segments)))
        short[path] = next((tail for tail in candidates if tails[tail] == 1), path)
    return short


@dataclass(frozen=True)
class DriftRow:
    """One row of the drift table: the keys that live in exactly the same configs."""

    backend: str
    keys: tuple[str, ...]
    count: int
    files: tuple[str, ...]
    suggestion: str | None

    def sample(self, short: dict[str, str], limit: int = SHOWN_FILES) -> str:
        shown = ", ".join(f"`{short.get(f, f)}`" for f in self.files[:limit])
        extra = len(self.files) - limit
        return f"{shown} (+{extra} more)" if extra > 0 else shown


def group_rows(rows: Sequence[KeyFinding]) -> tuple[list[DriftRow], list[DriftRow]]:
    """Split the findings into ``(renamed, merged)``, the two tables.

    A key with a suggested rename keeps its own row: that one is a direct fix
    someone applies key by key, and burying it in a list of twenty-five is the
    opposite of the point. The rest say the same thing many times over, so the
    ones that drifted out of the very same configs share a row -- which loses
    nothing, the row's configs being every member's.
    """
    renamed, shared = [], {}
    for row in rows:
        if row.suggestion:
            renamed.append(DriftRow(row.backend, (row.key,), row.count, row.files, row.suggestion))
        else:
            shared.setdefault((row.backend, row.count, row.files), []).append(row.key)
    merged = [
        DriftRow(backend, tuple(sorted(keys)), count, files, None)
        for (backend, count, files), keys in shared.items()
    ]

    def order(row: DriftRow) -> tuple[str, int, str]:
        return row.backend, -row.count, row.keys[0]

    return sorted(renamed, key=order), sorted(merged, key=order)


def subsection(title: str, lead: str) -> list[str]:
    """A `###` heading and its lead-in, spaced the way Markdown wants them."""
    return ["", f"### {title}", "", lead, ""]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", action="append", choices=BACKENDS, help="Limit to one backend.")
    parser.add_argument(
        "--path",
        action="append",
        type=Path,
        help="Scan these files/dirs instead of the default roots (requires a single --backend).",
    )
    parser.add_argument("--allowlist", type=Path, help="Override the allowlist YAML path.")
    parser.add_argument(
        "--warn-only",
        action="store_true",
        help="Report unknown-key drift/errors/stale allowlist entries but exit 0. "
        "A skipped backend still fails, and model-scoped findings never gate either way.",
    )
    parser.add_argument(
        "--show-allowlist", action="store_true", help="Also print the derived allowlist patterns."
    )
    args = parser.parse_args()
    if args.path and len(args.backend or BACKENDS) != 1:
        parser.error("--path requires exactly one --backend")
    return args


def main():
    args = parse_args()
    backends = args.backend or list(BACKENDS)
    started = time.perf_counter()

    lines = ["## Backend config schema drift", ""]
    rows, skipped, errors, checked = [], [], [], 0
    stale, scoped, model_scopes = [], [], []

    allowlist_path = args.allowlist or (ROOT / DEFAULT_ALLOWLIST)
    for backend in backends:
        for problem in check_manual_allowlist_consumers(ROOT, allowlist_path, backend):
            if problem not in stale:  # the `common` scope is read once per backend
                stale.append(problem)

        result = scan_backend(ROOT, backend, args.path, args.allowlist)
        if not result.available:
            skipped.append(backend)
            continue
        checked += result.checked
        errors.extend(result.errors)
        rows.extend(dedupe(result.findings, result.schema))
        scoped.extend(dedupe_scoped(result.scoped))
        model_scopes.extend((backend, scope) for scope in result.scopes)

    # Loud, and its own section: a backend nobody compared against anything is
    # the one outcome that must not read like a pass. It is also the one that
    # `--warn-only` does not forgive, so say so where the reader is looking.
    if skipped:
        lines.append(f"### FAILED -- {len(skipped)} backend(s) not checked")
        lines.append("")
        lines.append(
            f"**{', '.join(skipped)}**: upstream schema not found under `third_party/`, so no "
            "config of theirs was compared against anything. Run `git submodule update --init` "
            "to restore the check. Drift is warn-only; a check that never ran is not."
        )
        lines.append("")
    # Reported even when no schema is available: an allowlist entry whose
    # consumer is gone widens the legal key set, which is the blindness this
    # check exists to remove, and verifying it needs no submodule.
    if stale:
        lines.append(f"**{len(stale)} allowlist entr(y/ies) no longer backed by their consumer:**")
        lines.append("")
        for problem in stale:
            lines.append(f"- {problem}")
        lines.append("")

    if len(skipped) == len(backends):
        lines.append("Nothing to check." if not stale else "No schema available; allowlist checked.")
        print("\n".join(lines))
        return 1

    if rows:
        renamed, merged = group_rows(rows)
        short = shorten_paths(f for row in rows for f in row.files)
        shared = len(rows) - len(renamed)
        lines.append(
            f"**{len(rows)} unknown key(s)** across {sum(r.count for r in rows)} occurrence(s). "
            "Each one parses and then does nothing: fix the config, or add the key to "
            "`tools/ci/config_schema_allowlist.yaml` with a reason and a consumer. Example files "
            "are shortened to the shortest trailing path that names one config here."
        )
        # Two tables, because the two halves are two different jobs. A rename is
        # a key-by-key edit with a known destination; the rest is a decision per
        # set of configs, and a `Likely moved to` column of nothing but dashes.
        if renamed:
            lines += subsection(
                "Keys that look renamed rather than dropped",
                f"{len(renamed)} of them: upstream still defines a name this close, so each is one "
                "direct edit and gets a row to itself.",
            )
            lines.append("| Backend | Unknown key | Configs | Likely moved to | Example files |")
            lines.append("| --- | --- | ---: | --- | --- |")
            for row in renamed:
                lines.append(
                    f"| {row.backend} | `{row.keys[0]}` | {row.count} | "
                    f"`{row.suggestion}` | {row.sample(short)} |"
                )
        if merged:
            lines += subsection(
                "Keys with no obvious replacement",
                f"{'The other' if renamed else 'All'} {shared} drifted out of only {len(merged)} "
                "distinct set(s) of configs, so a row here is one such set and every key that "
                "drifted out of exactly it.",
            )
            lines.append("| Backend | Configs | Example files | Unknown key(s) |")
            lines.append("| --- | ---: | --- | --- |")
            for row in merged:
                keys = ", ".join(f"`{k}`" for k in row.keys)
                lines.append(f"| {row.backend} | {row.count} | {row.sample(short)} | {keys} |")
    elif not errors:
        lines.append(
            f"No drift: every key in {checked} config(s) exists upstream " "or is a known Primus extension."
        )
    elif checked:
        lines.append(f"No drift in the {checked} config(s) that loaded.")

    if scoped:
        if lines[-1]:
            lines.append("")
        lines.append("### Keys set on a model that cannot read them")
        lines.append("")
        lines.append("| Backend | Key | Only reaches | Set instead on | Configs | Example |")
        lines.append("| --- | --- | --- | --- | ---: | --- |")
        for row in scoped:
            models = ", ".join(f"`{m}`" for m in row.models)
            lines.append(
                f"| {row.backend} | `{row.key}` | `{row.scope.describe()}` "
                f"({row.scope.config_class}) | {models} | {row.count} | {row.sample(2)} |"
            )
        lines.append("")
        lines.append(
            f"**{len(scoped)} model-scoped key(s)** set on {sum(r.count for r in scoped)} config(s) "
            "that build a different model. The key parses, no config class carries it, and the "
            "value is silently replaced by the upstream default: drop it, or use the field the "
            "model in question actually reads."
        )

    # An unreadable config was never checked, so it cannot count towards a pass.
    if errors:
        if lines[-1]:
            lines.append("")
        lines.append(f"**{len(errors)} config(s) could not be loaded and were NOT checked:**")
        lines.append("")
        for error in errors:
            lines.append(f"- `{error.file}`: {error.message}")

    if args.show_allowlist:
        lines.append("")
        lines.append("<details><summary>Derived allowlist</summary>")
        lines.append("")
        for backend in backends:
            # Megatron derives one pattern per name read off `args`, thousands of
            # them, so show a sample rather than flooding the summary.
            patterns = build_allowlist(ROOT, backend, args.allowlist).patterns()
            shown = ", ".join(f"`{p}`" for p in patterns[:SHOWN_PATTERNS])
            extra = len(patterns) - SHOWN_PATTERNS
            lines.append(
                f"- **{backend}** ({len(patterns)}): {shown}"
                + (f", ... (+{extra} more)" if extra > 0 else "")
            )
        lines.append("")
        lines.append("</details>")

        # A scope narrows the legal key set rather than widening it, so it earns
        # the same scrutiny as an allowlist entry: show what it covers.
        if model_scopes:
            lines.append("")
            lines.append("<details><summary>Derived model scopes</summary>")
            lines.append("")
            for backend, scope in model_scopes:
                keys = ", ".join(f"`{k}`" for k in sorted(scope.keys))
                lines.append(
                    f"- **{backend}** `{scope.describe()}` -> `{scope.config_class}` "
                    f"(from `{scope.source}`): {keys}"
                )
            lines.append("")
            lines.append("</details>")

    lines.append("")
    scope = f"{checked} config(s)" if not errors else f"{checked} of {checked + len(errors)} config(s)"
    lines.append(f"_Checked {scope} in {time.perf_counter() - started:.2f}s._")
    print("\n".join(lines))
    # A missing submodule is an infrastructure failure, not a config one, so it
    # gets none of the grace period `--warn-only` buys the findings.
    if skipped:
        return 1
    # `scoped` is deliberately out of the gate: fixing one moves training
    # numerics for whichever model owns the key, so it needs that owner, not a
    # CI gate blocking unrelated PRs. An unknown key costs nothing to remove.
    return 1 if (rows or errors or stale) and not args.warn_only else 0


if __name__ == "__main__":
    raise SystemExit(main())
