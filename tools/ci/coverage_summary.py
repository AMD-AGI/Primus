###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Render coverage.py JSON as a compact Markdown table for the CI run summary.

Modes:
  1 report  -> single "Coverage" column (e.g. JAX MaxText E2E).
  2 reports -> "Unit" vs "Unit+E2E" columns. The 2nd report must come from
               `coverage combine` of the unit + E2E data (line-level merge);
               two summary percentages cannot simply be added.

Layout: top-level groups (core, backends, modules, agents, cli, pretrain.py, ...)
are bold rows at the same level, sorted by coverage. core/ and backends/ also
get indented sub-rows per area, sorted by coverage. __init__.py is dropped;
tools/ and platforms/ are excluded (ops tooling / env abstraction, not
unit-testable). runner/ is bash, covered by the tests/runner/ shell tests.

Single-report mode reflects what a partial run (e.g. MaxText E2E) actually
executed: modules with zero covered lines are hidden and the total is computed
over the executed modules only, so the headline number is meaningful instead of
diluted by code that run can never touch. The two-report comparison keeps every
module (unit gives the full denominator).
"""

import json
import sys
from collections import defaultdict

OMIT_MODULES = {"tools", "platforms"}
# Top-level groups whose sub-packages are shown as indented detail rows; every
# other group (modules, agents, cli, pretrain.py, ...) is a single bold row.
DETAILED_GROUPS = ("core", "backends")


def classify(path: str):
    """Return (group, detail) for a covered file, or None to skip it.

    group is the top-level row key; detail is the sub-row key for
    DETAILED_GROUPS (e.g. core/projection), else None.
    """
    seg = (path[path.find("primus/") :] if "primus/" in path else path).split("/")
    if seg[-1] == "__init__.py":
        return None
    if len(seg) < 2:
        return "(top-level)", None
    if len(seg) == 2:  # primus/<file>.py, e.g. pretrain.py
        return seg[1], None
    if seg[1] in DETAILED_GROUPS:
        return seg[1], seg[1] + "/" + seg[2]
    return seg[1], None


def _pct(covered: int, total: int) -> float:
    return (100.0 * covered / total) if total else 0.0


def _aggregate(report: dict):
    """Return {group: {detail|group: [covered, statements]}} for kept modules."""
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for fpath, info in report.get("files", {}).items():
        result = classify(fpath)
        if result is None:
            continue
        group, detail = result
        if group in OMIT_MODULES:
            continue
        s = info["summary"]
        a = agg[group][detail or group]
        a[0] += s["covered_lines"]
        a[1] += s["num_statements"]
    return agg


def render(primary: dict, title: str, secondary: dict = None) -> str:
    pa = _aggregate(primary)
    sa = _aggregate(secondary) if secondary is not None else None
    two = sa is not None

    def e2e_cov(group, key):
        return sa.get(group, {}).get(key, [0])[0] if two else 0

    def row(label, cov, stmts, e2e, bold=False):
        if two:
            vals = [format(stmts, ","), "%.1f%%" % _pct(cov, stmts), "%.1f%%" % _pct(e2e, stmts)]
        else:
            vals = [format(cov, ","), format(stmts, ","), "%.1f%%" % _pct(cov, stmts)]
        w = "**" if bold else ""
        return "| " + " | ".join("%s%s%s" % (w, x, w) for x in [label] + vals) + " |"

    def group_totals(group):
        # Single-report mode counts only executed entries (cov > 0) so a partial
        # run isn't diluted by sub-modules it never touched; two-report keeps all.
        entries = [(k, v) for k, v in pa[group].items() if two or v[0] > 0]
        cov = sum(v[0] for _, v in entries)
        stmts = sum(v[1] for _, v in entries)
        e2e = sum(e2e_cov(group, k) for k, _ in entries) if two else 0
        return cov, stmts, e2e

    # Single-report mode hides modules with zero coverage and totals over the
    # executed modules only, so a partial run (e.g. MaxText E2E) isn't diluted by
    # code it can never touch. The two-report comparison keeps the full denominator.
    def group_executed(group):
        return group_totals(group)[0] > 0 if not two else True

    groups = [g for g in pa if group_totals(g)[1] > 0 and group_executed(g)]

    tc = sum(group_totals(g)[0] for g in groups)
    tn = sum(group_totals(g)[1] for g in groups)
    te = sum(group_totals(g)[2] for g in groups) if two else 0
    excl = ", ".join(sorted(OMIT_MODULES))

    out = ["## Primus coverage - %s\n" % title]
    if two:
        out.append(
            "**Unit %.1f%% -> Unit+E2E %.1f%%** (%s statements; excludes %s)\n"
            % (_pct(tc, tn), _pct(te, tn), format(tn, ","), excl)
        )
        out += ["| Module | Stmts | Unit | Unit+E2E |", "|---|--:|--:|--:|"]
    else:
        out.append(
            "**Total line coverage: %.1f%%** (%s / %s statements; excludes %s)\n"
            % (_pct(tc, tn), format(tc, ","), format(tn, ","), excl)
        )
        out += ["| Module | Covered | Stmts | Coverage |", "|---|--:|--:|--:|"]

    # Top-level groups, sorted by coverage (desc).
    for group in sorted(groups, key=lambda g: -_pct(group_totals(g)[0], group_totals(g)[1])):
        cov, stmts, e2e = group_totals(group)
        out.append(row("`%s`" % group, cov, stmts, e2e, bold=True))
        if group in DETAILED_GROUPS:
            # In single-report mode, hide sub-rows that were never executed.
            details = ((k, v) for k, v in pa[group].items() if v[1] > 0 and (two or v[0] > 0))
            for k, v in sorted(details, key=lambda kv: -_pct(kv[1][0], kv[1][1])):
                out.append(row("&emsp;`%s`" % k, v[0], v[1], e2e_cov(group, k)))

    out.append(row("TOTAL", tc, tn, te, bold=True))
    return "\n".join(out)


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def main() -> int:
    primary = _load(sys.argv[1])
    title = sys.argv[2] if len(sys.argv) > 2 else "tests"
    secondary = _load(sys.argv[3]) if len(sys.argv) > 3 else None
    print(render(primary, title, secondary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
