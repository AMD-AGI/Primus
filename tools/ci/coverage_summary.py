###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Render coverage.py JSON as a compact Markdown table for the CI run summary.

Modes (by number of report arguments):
  1 report   -> single "Coverage" column (e.g. JAX MaxText E2E).
  2+ reports -> "Unit" vs "Unit+E2E". The 1st is unit; the rest are E2E reports,
                merged per module by taking the max covered lines. Each E2E
                report should be a `coverage combine` of unit + that job's E2E
                data (line-level). Taking the max across jobs avoids double
                counting and lets torch (megatron/torchtitan) and jax (maxtext)
                E2E - which cover near-disjoint modules - share one table.

Layout: top-level groups (core, backends, modules, agents, cli, ...) are bold
rows at the same level, sorted by coverage. core/ and backends/ also get
indented sub-rows per area, sorted by coverage. __init__.py is dropped;
tools/, platforms/ and the top-level pretrain.py entrypoint are excluded (ops
tooling / env abstraction / thin CLI glue, exercised by E2E and shell tests
rather than unit tests). runner/ is bash, covered by the tests/runner/ shell
tests.

Single-report mode reflects what a partial run (e.g. MaxText E2E) actually
executed: modules with zero covered lines are hidden and the total is computed
over the executed modules only, so the headline number is meaningful instead of
diluted by code that run can never touch. The two-report comparison keeps every
module (unit gives the full denominator).
"""

import json
import sys
from collections import defaultdict

OMIT_MODULES = {"tools", "platforms", "pretrain.py"}
# Top-level groups whose sub-packages are shown as indented detail rows; every
# other group (modules, agents, cli, ...) is a single bold row.
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


def render(primary: dict, title: str, secondaries: list = None) -> str:
    pa = _aggregate(primary)
    sas = [_aggregate(s) for s in (secondaries or [])]
    two = bool(sas)

    def e2e_cov(group, key):
        # Merge E2E reports by max covered lines (jobs cover near-disjoint
        # modules, so max avoids double counting the shared core code).
        return max((sa.get(group, {}).get(key, [0])[0] for sa in sas), default=0) if two else 0

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

    # coverage's own totals over every primus file (nothing excluded), for context.
    p_all = primary.get("totals", {}).get("percent_covered", 0.0)
    s_all = (
        max((s.get("totals", {}).get("percent_covered", 0.0) for s in secondaries), default=0.0)
        if two
        else 0.0
    )

    out = ["## Primus coverage - %s\n" % title]
    if two:
        out.append(
            "**Unit %.1f%% -> Unit+E2E %.1f%%** (%s statements; excludes %s)\n"
            % (_pct(tc, tn), _pct(te, tn), format(tn, ","), excl)
        )
        out.append(
            "_Including all modules (nothing excluded): Unit %.1f%% -> Unit+E2E %.1f%%._\n" % (p_all, s_all)
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
    secondaries = [_load(p) for p in sys.argv[3:]]
    print(render(primary, title, secondaries or None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
