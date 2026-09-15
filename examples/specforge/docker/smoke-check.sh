#!/bin/bash
# Build-time and post-build smoke checks (no GPU required).
set -euo pipefail

python -c "from sglang.srt.server_args import ServerArgs; assert hasattr(ServerArgs, 'enable_spec_capture')"
python -c "import specforge"

primus-cli --help >/dev/null
specforge --help >/dev/null

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
