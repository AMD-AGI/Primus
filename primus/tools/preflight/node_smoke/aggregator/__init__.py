###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Aggregator -- combines per-node JSONs into a cluster report.

* :mod:`.summarizers` -- pure data-shaping helpers used by the report
  writer (one ``_*_rows`` / ``_*_summary`` function per Markdown
  section). They never raise; missing data degrades to empty rows.
* :mod:`.report`      -- the markdown writer. The single
  :func:`.report.write_smoke_report` entry point composes the report
  by calling small per-section ``_write_<section>`` helpers, in the
  exact order the original monolithic block produced.
"""

from __future__ import annotations
