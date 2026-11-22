#!/usr/bin/env bash
# Runs safe_job.py (see there for usage) with paramiko from a venv kept for the rest of the job: the
# runner may lack ssh and curl and may not be allowed to install them.
set -euo pipefail
venv=$RUNNER_TEMP/safe-venv
if [[ ! -x $venv/bin/python ]]; then
  python3 -m venv "$venv"
  "$venv/bin/pip" install -q --no-cache-dir paramiko
fi
exec "$venv/bin/python" "$(dirname "${BASH_SOURCE[0]}")/safe_job.py" "$@"
