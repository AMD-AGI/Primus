###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Render CI stage wall-clock times (a stage<TAB>seconds TSV) as a Markdown table.

Complements junit_summary (test time) by surfacing the heavy build/install
stages. A missing/empty TSV renders nothing and exits 0 so the step never fails
the job; stage order (i.e. execution order) is preserved.
"""

import argparse


def fmt(seconds):
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def parse(path):
    rows = []
    try:
        with open(path) as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 2 or not parts[0].strip():
                    continue
                try:
                    rows.append((parts[0].strip(), float(parts[1].strip())))
                except ValueError:
                    continue
    except OSError:
        return []
    return rows


def render(rows, title=None):
    suffix = f" - {title}" if title else ""
    lines = [f"## CI runtime{suffix}\n", "| Stage | Time |", "|---|--:|"]
    total = 0.0
    for stage, secs in rows:
        lines.append(f"| {stage} | {fmt(secs)} |")
        total += secs
    lines.append(f"| **TOTAL (timed stages)** | **{fmt(total)}** |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Render a stage<TAB>seconds TSV as a Markdown runtime table.")
    ap.add_argument("tsv", help="Path to the runtime TSV (stage<TAB>seconds per line).")
    ap.add_argument("--title", default=None, help="Optional section title (e.g. torch).")
    args = ap.parse_args()

    rows = parse(args.tsv)
    if not rows:
        return 0  # nothing timed; don't emit an empty table or fail the step
    print(render(rows, args.title))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
