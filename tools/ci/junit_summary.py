###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Render pytest JUnit XML as a Markdown table for the CI run summary.

GitHub does not render JUnit on its own, so uploading it only yields a download
link. This turns one or more reports into a per-suite pass/fail/error/skip/time
table (plus a collapsible list of failures) to append next to the coverage
table:

    python junit_summary.py --title torch test-reports/*.xml >> "$GITHUB_STEP_SUMMARY"

Each report is labelled by its filename stem; a missing/unparseable file
renders as a "no report" row instead of failing the step.
"""

import argparse
import glob
import os
import xml.etree.ElementTree as ET

COLUMNS = ("tests", "passed", "failed", "errors", "skipped")


def _fmt_time(seconds):
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def parse_file(path):
    """Return (label, stats|None, failures) for one JUnit XML file."""
    label = os.path.splitext(os.path.basename(path))[0]
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        return label, None, []

    stats = dict.fromkeys(COLUMNS, 0)
    stats["time"] = 0.0
    failures = []
    for suite in root.iter("testsuite"):  # root is <testsuites> or a lone <testsuite>
        stats["tests"] += int(suite.get("tests", 0) or 0)
        stats["failed"] += int(suite.get("failures", 0) or 0)
        stats["errors"] += int(suite.get("errors", 0) or 0)
        stats["skipped"] += int(suite.get("skipped", 0) or 0)
        stats["time"] += float(suite.get("time", 0) or 0)
        for case in suite.iter("testcase"):
            # find(...) "or" is unsafe: an empty Element is falsy, so test explicitly.
            bad, kind = case.find("failure"), "failure"
            if bad is None:
                bad, kind = case.find("error"), "error"
            if bad is not None:
                name = (case.get("classname", "") + "::" + case.get("name", "")).strip(":")
                msg = (bad.get("message") or "").strip().splitlines()
                failures.append((name, kind, msg[0] if msg else ""))
    stats["passed"] = stats["tests"] - stats["failed"] - stats["errors"] - stats["skipped"]
    return label, stats, failures


def render(reports, title=None):
    lines = ["## Test results - %s\n" % title] if title else []
    lines += [
        "| Suite | Tests | Passed | Failed | Errors | Skipped | Time |",
        "|---|--:|--:|--:|--:|--:|--:|",
    ]
    total = dict.fromkeys(COLUMNS, 0)
    total["time"] = 0.0
    failures = []
    for label, st, fails in reports:
        if st is None:
            lines.append("| `%s` | _no report_ |  |  |  |  |  |" % label)
            continue
        lines.append(
            "| `%s` | %d | %d | %d | %d | %d | %s |"
            % (
                label,
                st["tests"],
                st["passed"],
                st["failed"],
                st["errors"],
                st["skipped"],
                _fmt_time(st["time"]),
            )
        )
        for k in total:
            total[k] += st[k]
        failures += [(label, *f) for f in fails]
    lines.append(
        "| **TOTAL** | **%d** | **%d** | **%d** | **%d** | **%d** | **%s** |"
        % (
            total["tests"],
            total["passed"],
            total["failed"],
            total["errors"],
            total["skipped"],
            _fmt_time(total["time"]),
        )
    )
    lines.append("")

    if failures:
        lines.append("<details><summary>%d failing test(s)</summary>\n" % len(failures))
        for label, name, kind, msg in failures:
            lines.append(("- `%s` **%s** (%s): %s" % (label, name, kind, msg))[:300])
        lines.append("\n</details>")
    else:
        lines.append("_All tests passed._")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Render JUnit XML as a Markdown CI summary.")
    ap.add_argument("xml", nargs="+", help="JUnit XML file(s) or glob(s).")
    ap.add_argument("--title", default=None, help="Optional section title (e.g. torch).")
    args = ap.parse_args()

    paths = []
    for pat in args.xml:
        paths += sorted(glob.glob(pat)) or [pat]  # keep literal -> "no report" row
    print(render([parse_file(p) for p in paths], args.title))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
