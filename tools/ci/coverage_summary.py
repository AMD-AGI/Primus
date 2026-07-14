###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Render coverage.py JSON as a compact Markdown table for the CI run summary.

Modes (by number of report arguments):
  1 report   -> single "Coverage" column (e.g. JAX MaxText E2E).
  2+ reports -> "Unit" vs "Unit+E2E". The 1st is unit; the rest are E2E reports,
                merged per module by max covered lines (avoids double-counting
                torch's near-disjoint megatron/torchtitan E2E).

Layout: top-level groups are bold rows sorted by coverage; core/backends also
get indented per-area detail rows. The headline percentage gets a tier emoji
(_TIER_THRESHOLDS) and a curated "Notes" column flags widely-shared infra or a
known low-coverage cause (NOTES) -- both are reading aids, not quality gates.

What counts toward a total vs. what gets its own row are deliberately
decoupled decisions -- see classify().

Single-report mode hides untouched modules and totals only the executed ones,
so a partial run (e.g. MaxText E2E) isn't diluted by code it can't reach; the
two-report comparison keeps the full denominator.
"""

import json
import sys
from collections import defaultdict

OMIT_MODULES = {"tools", "platforms"}
# Top-level groups whose sub-packages are shown as indented detail rows; every
# other group (agents, cli, ...) is a single bold row.
DETAILED_GROUPS = ("core", "backends")


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
    seg = (path[path.find("primus/") :] if "primus/" in path else path).split("/")
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


# Curated, best-effort context -- not exhaustive. Flags infra whose coverage
# matters more than its size suggests, and low-coverage areas with a known,
# persistent cause, so it isn't re-litigated every read. Keyed like the table
# (group, or "<group>/<detail>"); update alongside any fix or new finding.
# <br> forces cell wrapping so one long note can't stretch the whole column.
NOTES = {
    "primus (top-level)": "loose primus/ modules, no sub-package;<br>auto-folded by path depth",
    "core": "\U0001F511 shared infra;<br>imported by every run",
    "backends/megatron": (
        "100+ patches gated by fp8 / MoE /<br>zero-bubble-pp / fsdp2 flags;<br>"
        "CI E2E runs only 1-2 configs"
    ),
    "backends/transformer_engine": "fp8 GEMM / attn-overlap kernels;<br>only hit when an E2E enables fp8",
    "backends/diffusion": "no E2E trainer suite yet<br>(unit-tested only)",
}

_LEGEND = (
    "\U0001F7E2 >=50% / \U0001F7E1 >=25% / \U0001F534 <25% (next to module) "
    "&nbsp;\u00b7&nbsp; \U0001F511 widely-shared infra\n"
)


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

    def row(label, key, cov, stmts, e2e, bold=False):
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
        cells.append(NOTES.get(key, ""))
        return "| " + " | ".join(cells) + " |"

    def group_totals(group):
        # Two-report mode keeps every entry (full denominator); single-report
        # mode counts only executed ones so a partial run isn't diluted.
        entries = [(k, v) for k, v in pa[group].items() if two or v[0] > 0]
        cov = sum(v[0] for _, v in entries)
        stmts = sum(v[1] for _, v in entries)
        e2e = sum(e2e_cov(group, k) for k, _ in entries) if two else 0
        return cov, stmts, e2e

    def group_executed(group):  # single-report mode: hide groups a partial run never touched
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
        out.append(_LEGEND)
        out += ["| Module | Stmts | Unit | Unit+E2E | Notes |", "|---|--:|--:|--:|---|"]
    else:
        out.append(
            "**Total line coverage: %.1f%%** (%s / %s statements; excludes %s)\n"
            % (_pct(tc, tn), format(tc, ","), format(tn, ","), excl)
        )
        out.append(_LEGEND)
        out += ["| Module | Covered | Stmts | Coverage | Notes |", "|---|--:|--:|--:|---|"]

    for group in sorted(groups, key=lambda g: -_pct(group_totals(g)[0], group_totals(g)[1])):
        cov, stmts, e2e = group_totals(group)
        out.append(row("`%s`" % group, group, cov, stmts, e2e, bold=True))
        if group in DETAILED_GROUPS:
            # k == group is a folded loose file (see classify()), already
            # counted in group_totals() above -- no row of its own here.
            details = ((k, v) for k, v in pa[group].items() if k != group and v[1] > 0 and (two or v[0] > 0))
            for k, v in sorted(details, key=lambda kv: -_pct(kv[1][0], kv[1][1])):
                out.append(row("&emsp;`%s`" % k, k, v[0], v[1], e2e_cov(group, k)))

    out.append(row("TOTAL", None, tc, tn, te, bold=True))
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
