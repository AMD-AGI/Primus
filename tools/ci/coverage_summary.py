###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Render coverage.py JSON as a compact Markdown table for the CI run summary.

usage: coverage_summary.py TITLE --unit REPORT... [--e2e REPORT...]

Modes:
  --unit only -> single "Coverage" column (e.g. JAX MaxText E2E).
  with --e2e  -> "Unit" vs "Unit+E2E". Each column is the union of its reports'
                 executed lines per file (Unit+E2E also takes the unit reports),
                 so runs on other GPUs or frameworks add only the lines they
                 alone reach. Every report must come from the same commit.

Layout: top-level groups are bold rows sorted by coverage; core/backends also
get indented per-area detail rows. The headline percentage gets a tier emoji
(_TIER_THRESHOLDS), and the "Notes" column, derived from the numbers, marks
Primus core, the modules where more tests gain the most (_RAISE_SHARE) and why a
low one is low -- both are reading aids, not quality gates.

What counts toward a total vs. what gets its own row are deliberately
decoupled decisions -- see classify().

With --unit only, the table hides untouched modules and totals only the
executed ones, so a partial run (e.g. MaxText E2E) isn't diluted by code it
can't reach; with --e2e it keeps the full denominator.
"""

import argparse
import json
from collections import defaultdict

OMIT_MODULES = {"tools", "platforms"}
# Top-level groups whose sub-packages are shown as indented detail rows; every
# other group (agents, cli, ...) is a single bold row.
DETAILED_GROUPS = ("core", "backends")


def _relpath(path: str) -> str:
    return path[path.find("primus/") :] if "primus/" in path else path


def classify(path: str):
    """Return (group, detail) for a file, or None to drop it (not counted).

    group is the top-level row key; detail is the sub-row key within
    DETAILED_GROUPS (e.g. "core/projection"). detail=None means "counted
    toward group's total, no row of its own" -- true for every
    non-DETAILED_GROUPS group, and for a bare file with no sub-package of its
    own (pretrain.py -> folds into "primus (top-level)"; core/base_module.py ->
    folds into "core"). Folding is by path *depth*, not filename, so a future
    file added the same way folds the same way for free.

    Dropping entirely (OMIT_MODULES above) is the only *policy* exclusion --
    orthogonal to the structural folding here.
    """
    seg = _relpath(path).split("/")
    if seg[-1] == "__init__.py":
        return None
    if len(seg) <= 2:  # outside primus/ entirely, or primus/<file>.py directly (no sub-package)
        return "primus (top-level)", None
    if seg[1] in DETAILED_GROUPS:
        if len(seg) == 3:  # primus/<group>/<file>.py directly: no sub-package of its own
            return seg[1], None
        return seg[1], seg[1] + "/" + seg[2]
    return seg[1], None


def _pct(covered: int, total: int) -> float:
    return (100.0 * covered / total) if total else 0.0


# Best-effort visual highlight for the headline column: GitHub strips CSS color
# from Action run summaries, so emoji is the portable substitute. Thresholds
# are a rough reading aid, not a quality gate.
_TIER_THRESHOLDS = ((50.0, "\U0001F7E2"), (25.0, "\U0001F7E1"))  # >=50% green, >=25% yellow, else red
_TIER_RED = "\U0001F534"


def _tier(pct: float) -> str:
    for threshold, emoji in _TIER_THRESHOLDS:
        if pct >= threshold:
            return emoji
    return _TIER_RED


# Coverage priority: lower sorts first (ties keep coverage order), pinning the
# most important groups to the top. core (Primus's core) ranks highest.
PRIORITY = {"core": 0}
_DEFAULT_PRIORITY = 1


def _priority(group: str) -> int:
    return PRIORITY.get(group, _DEFAULT_PRIORITY)


# A module below green that holds at least this share of all uncovered lines is
# marked as the place where more tests gain the most.
_RAISE_SHARE = 0.05

_LEGEND = (
    "\U0001F7E2 >=50% / \U0001F7E1 >=25% / \U0001F534 <25% (next to module) "
    "&nbsp;\u00b7&nbsp; \U0001F511 Primus core, imported by every run "
    f"&nbsp;\u00b7&nbsp; \u2b06\ufe0f below 50% and has >={_RAISE_SHARE:.0%} of all uncovered lines: worth raising\n"
)


def _merge(reports: list) -> dict:
    """Return {path: (statements, executed lines)}, the union over the reports."""
    files = {}
    for report in reports:
        for fpath, info in report.get("files", {}).items():
            path = _relpath(fpath)
            stmts, lines = files.get(path, (0, set()))
            files[path] = (max(stmts, info["summary"]["num_statements"]), lines | set(info["executed_lines"]))
    return files


def _aggregate(files: dict):
    """Return {group: {detail|group: [covered, statements]}} for kept modules."""
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for path, (stmts, lines) in files.items():
        result = classify(path)
        if result is None:
            continue
        group, detail = result
        if group in OMIT_MODULES:
            continue
        a = agg[group][detail or group]
        a[0] += min(len(lines), stmts)
        a[1] += stmts
    return agg


def _total_pct(files: dict) -> float:
    covered = sum(min(len(lines), stmts) for stmts, lines in files.values())
    return _pct(covered, sum(stmts for stmts, _ in files.values()))


def render(units: list, title: str, e2es: list = None) -> str:
    unit_files = _merge(units)
    two = bool(e2es)
    all_files = _merge(units + e2es) if two else {}
    if two:  # Both columns count the same files against the same statements.
        unit_files = {
            path: (stmts, unit_files.get(path, (0, set()))[1]) for path, (stmts, _) in all_files.items()
        }
    pa, ca = _aggregate(unit_files), _aggregate(all_files)

    def e2e_cov(group, key):
        return ca.get(group, {}).get(key, [0])[0] if two else 0

    def note(group, cov, stmts, e2e, leaf):
        # Rows with detail rows below them leave the reasons to those.
        parts = ["\U0001F511"] if group in PRIORITY else []
        hit = e2e if two else cov
        if leaf and stmts and _pct(hit, stmts) < _TIER_THRESHOLDS[0][0]:
            if gap and stmts - hit >= _RAISE_SHARE * gap:
                parts.append("\u2b06\ufe0f has %.0f%% of all uncovered lines" % (100.0 * (stmts - hit) / gap))
            if not hit:
                parts.append("no test runs it")
            elif two and e2e == cov:
                parts.append("E2E adds no lines")
        return "; ".join(parts)

    def row(label, cov, stmts, e2e, bold=False, notes=""):
        # Tier dot rides next to the module name (leftmost), so it reads as a
        # per-module health mark and leaves the right-aligned % columns clean.
        # It goes *after* any &emsp; indent so detail-row dots stay indented.
        headline = _pct(e2e if two else cov, stmts)
        dot = _tier(headline) + " " if stmts else ""
        if two:
            vals = [format(stmts, ","), "%.1f%%" % _pct(cov, stmts), "%.1f%%" % _pct(e2e, stmts)]
        else:
            vals = [format(cov, ","), format(stmts, ","), "%.1f%%" % _pct(cov, stmts)]
        w = "**" if bold else ""
        indent = "&emsp;" if label.startswith("&emsp;") else ""
        rest = label[len(indent) :]
        label_cell = "%s%s%s%s%s" % (indent, dot, w, rest, w)
        cells = [label_cell] + ["%s%s%s" % (w, x, w) for x in vals]
        cells.append(notes)
        return "| " + " | ".join(cells) + " |"

    def group_totals(group):
        # --e2e mode keeps every entry (full denominator); --unit-only
        # mode counts only executed ones so a partial run isn't diluted.
        entries = [(k, v) for k, v in pa[group].items() if two or v[0] > 0]
        cov = sum(v[0] for _, v in entries)
        stmts = sum(v[1] for _, v in entries)
        e2e = sum(e2e_cov(group, k) for k, _ in entries) if two else 0
        return cov, stmts, e2e

    def group_executed(group):  # --unit-only mode: hide groups a partial run never touched
        return group_totals(group)[0] > 0 if not two else True

    groups = [g for g in pa if group_totals(g)[1] > 0 and group_executed(g)]

    tc = sum(group_totals(g)[0] for g in groups)
    tn = sum(group_totals(g)[1] for g in groups)
    te = sum(group_totals(g)[2] for g in groups) if two else 0
    gap = tn - (te if two else tc)
    excl = ", ".join(sorted(OMIT_MODULES))

    out = ["## Primus coverage - %s\n" % title]
    if two:
        out.append(
            "**Unit %.1f%% -> Unit+E2E %.1f%%** (%s statements; excludes %s)\n"
            % (_pct(tc, tn), _pct(te, tn), format(tn, ","), excl)
        )
        out.append(
            "_Including all modules (nothing excluded): Unit %.1f%% -> Unit+E2E %.1f%%._\n"
            % (_total_pct(unit_files), _total_pct(all_files))
        )
        out.append(_LEGEND)
        out += ["| Module | Stmts | Unit | Unit+E2E | Notes |", "|---|--:|--:|--:|---|"]
    else:
        out.append(
            "**Total line coverage: %.1f%%** (%s / %s statements; excludes %s)\n"
            % (_pct(tc, tn), format(tc, ","), format(tn, ","), excl)
        )
        out.append(_LEGEND)
        out += ["| Module | Covered | Stmts | Coverage | Notes |", "|---|--:|--:|--:|---|"]

    for group in sorted(groups, key=lambda g: (_priority(g), -_pct(group_totals(g)[0], group_totals(g)[1]))):
        cov, stmts, e2e = group_totals(group)
        leaf = group not in DETAILED_GROUPS
        out.append(row("`%s`" % group, cov, stmts, e2e, True, note(group, cov, stmts, e2e, leaf)))
        if group in DETAILED_GROUPS:
            # k == group is a folded loose file (see classify()), already
            # counted in group_totals() above -- no row of its own here.
            details = ((k, v) for k, v in pa[group].items() if k != group and v[1] > 0 and (two or v[0] > 0))
            for k, v in sorted(details, key=lambda kv: -_pct(kv[1][0], kv[1][1])):
                e2e = e2e_cov(group, k)
                out.append(row("&emsp;`%s`" % k, v[0], v[1], e2e, notes=note(k, v[0], v[1], e2e, True)))

    out.append(row("TOTAL", tc, tn, te, bold=True))
    return "\n".join(out)


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render coverage.py JSON reports as a Markdown table.")
    parser.add_argument("title")
    parser.add_argument("--unit", nargs="+", required=True, help="unit-test coverage JSON reports")
    parser.add_argument("--e2e", nargs="+", default=[], help="E2E coverage JSON reports")
    args = parser.parse_args()
    print(render([_load(p) for p in args.unit], args.title, [_load(p) for p in args.e2e]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
