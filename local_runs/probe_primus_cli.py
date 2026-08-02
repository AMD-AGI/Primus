#!/usr/bin/env python3
"""Run the Primus CLI while preserving BaseException tracebacks per rank."""

import runpy
import sys
import traceback

sys.argv[0] = "primus"
try:
    runpy.run_module("primus.cli.main", run_name="__main__")
except BaseException:
    traceback.print_exc()
    raise
