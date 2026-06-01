#!/usr/bin/env python3
"""Build the aggregated *monthly* report dashboard index.

Thin CLI wrapper over the shared periodic-report core in
``periodic_reports.py``. It validates
``docs/monthly_reports/dashboard-data/reports/*.json`` and rewrites
``docs/monthly_reports/dashboard-data/index.json``.

Weekly and monthly reports share the same schema and tooling; the dashboard
consumes a combined index assembled at bundle-build time (see
``build_site_bundle.py``). Run this directly to refresh just the monthly index.
"""

from __future__ import annotations

import sys

from periodic_reports import CADENCES, REPO_ROOT, write_cadence_index


def main() -> int:
    cadence = CADENCES["monthly"]
    count = write_cadence_index(cadence)
    print(
        f"Wrote monthly-report dashboard index with {count} report(s) to "
        f"{cadence.index_path.relative_to(REPO_ROOT)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
