#!/bin/bash
# Build-time and post-build smoke checks (no GPU required).
set -euo pipefail

python - <<'PY'
from primus.backends.specforge.capture_plugin import capture_plugin_issues

issues = capture_plugin_issues()
if issues:
    raise SystemExit("Capture plugin checks failed:\n" + "\n".join(issues))
PY
python -c "import specforge"
python - <<'PY'
import tempfile
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

with tempfile.TemporaryDirectory() as tmp:
    writer = SummaryWriter(log_dir=tmp)
    writer.add_scalar("smoke/ok", 1.0, 0)
    writer.flush()
    writer.close()
    events = list(Path(tmp).rglob("events.out.tfevents.*"))
    if not events:
        raise SystemExit("SummaryWriter did not write an event file")
PY

primus-cli --help >/dev/null
specforge --help >/dev/null
python -m primus.backends.specforge.offline_capture --help >/dev/null

python - <<'PY'
import pathlib

import primus
import sglang
import torch

print("torch", torch.__version__)
print("sglang", sglang.__version__)
print("primus", getattr(primus, "__version__", "?"), pathlib.Path(primus.__file__).parent)

version = torch.__version__.lower()
hip = getattr(torch.version, "hip", None)
if hip:
    print("hip", hip)
elif "git" in version or "+rocm" in version:
    # Editable ROCm builds in the SGLang image may omit +rocm in __version__.
    print("torch appears to be the SGLang ROCm build")
elif "+cu" in version:
    raise SystemExit(
        "Detected a CUDA torch wheel; the base ROCm stack may have been clobbered"
    )
else:
    raise SystemExit(
        "Unexpected torch build; expected HIP metadata or the SGLang ROCm git build"
    )
PY

echo "Smoke checks passed."
