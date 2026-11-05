#!/bin/bash
# Build-time and post-build smoke checks (no GPU required).
set -euo pipefail

python - <<'PY'
from primus.backends.specforge.capture_plugin import capture_plugin_issues

issues = capture_plugin_issues()
if issues:
    raise SystemExit("Capture plugin checks failed:\n" + "\n".join(issues))
PY
bash "$(dirname "${BASH_SOURCE[0]}")/smoke-check-runtime.sh"
