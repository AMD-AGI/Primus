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

Grouping: core/* and backends/* get a bold subtotal row; other modules are
listed flat. __init__.py is dropped; tools/ and platforms/ are excluded (ops
tooling / env abstraction, not unit-testable). runner/ is bash, covered by the
tests/runner/ shell tests.
"""

import json
import sys
from collections import defaultdict

OMIT_MODULES = {"tools", "platforms"}


def classify(path: str):
    """Return (section, module_name) for a covered file, or None to skip it."""
    seg = (path[path.find("primus/") :] if "primus/" in path else path).split("/")
    if seg[-1] == "__init__.py":
        return None
    if len(seg) < 2:
        return "other", "(top-level)"
    if len(seg) == 2:  # primus/<file>.py, e.g. pretrain.py
        return "other", seg[1]
    if seg[1] in ("core", "backends"):
        return seg[1], seg[1] + "/" + seg[2]
    return "other", seg[1]


def _pct(covered: int, total: int) -> float:
    return (100.0 * covered / total) if total else 0.0


def _aggregate(report: dict):
    """Return {module_name: [covered, statements, section]} for kept modules."""
    agg = defaultdict(lambda: [0, 0, ""])
    for fpath, info in report.get("files", {}).items():
        result = classify(fpath)
        if result is None:
            continue
        sec, name = result
        if name in OMIT_MODULES:
            continue
        s = info["summary"]
        a = agg[name]
        a[0] += s["covered_lines"]
        a[1] += s["num_statements"]
        a[2] = sec
    return agg


def render(primary: dict, title: str, secondary: dict = None) -> str:
    pa = _aggregate(primary)
    sa = _aggregate(secondary) if secondary is not None else None
    two = sa is not None

    def e2e_cov(names):
        return sum(sa.get(n, [0])[0] for n in names) if two else 0

    def row(label, names, bold=False):
        """One Markdown row. `label` is rendered as-is (caller adds backticks)."""
        c = sum(pa[n][0] for n in names)
        n = sum(pa[n][1] for n in names)
        if two:
            vals = [format(n, ","), "%.1f%%" % _pct(c, n), "%.1f%%" % _pct(e2e_cov(names), n)]
        else:
            vals = [format(c, ","), format(n, ","), "%.1f%%" % _pct(c, n)]
        w = "**" if bold else ""
        cells = ["%s%s%s" % (w, x, w) for x in [label] + vals]
        return "| " + " | ".join(cells) + " |"

    names_all = [n for n in pa if pa[n][1] > 0]
    tc = sum(pa[n][0] for n in names_all)
    tn = sum(pa[n][1] for n in names_all)
    excl = ", ".join(sorted(OMIT_MODULES))

    out = ["## Primus coverage - %s\n" % title]
    if two:
        out.append(
            "**Unit %.1f%% -> Unit+E2E %.1f%%** (%s statements; excludes %s)\n"
            % (_pct(tc, tn), _pct(e2e_cov(names_all), tn), format(tn, ","), excl)
        )
        out += ["| Module | Stmts | Unit | Unit+E2E |", "|---|--:|--:|--:|"]
    else:
        out.append(
            "**Total line coverage: %.1f%%** (%s / %s statements; excludes %s)\n"
            % (_pct(tc, tn), format(tc, ","), format(tn, ","), excl)
        )
        out += ["| Module | Covered | Stmts | Coverage |", "|---|--:|--:|--:|"]

    for sec, stitle in [("core", "Core (framework)"), ("backends", "Backends")]:
        names = sorted([n for n in names_all if pa[n][2] == sec], key=lambda n: -pa[n][1])
        if names:
            out.append(row(stitle, names, bold=True))
            out += [row("`%s`" % n, [n]) for n in names]

    # other modules (modules, agents, cli, pretrain.py, ...) listed flat
    others = sorted([n for n in names_all if pa[n][2] == "other"], key=lambda n: -pa[n][1])
    out += [row("`%s`" % n, [n]) for n in others]

    out.append(row("TOTAL", names_all, bold=True))
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
