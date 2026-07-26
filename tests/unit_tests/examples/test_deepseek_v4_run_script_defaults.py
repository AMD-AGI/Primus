###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Default-value gate for the DeepSeek-V4 run scripts.

The Flash launcher overrides a handful of knobs on top of ``run_deepseek_v4.sh``.
One of them -- ``USE_V4_FP8_INDEXER`` -- silently changes model numerics rather
than just performance: the indexer picks which compressed KV entries each query
attends to, so quantizing its QK inputs changes the selection itself. On top of
that the knob is a fake quant (quantize/dequantize around a BF16 GEMM), so it
costs throughput without buying any. Low-precision indexer QK is a legitimate
thing to explore, but it should be opt-in rather than on by default in one
launcher only.

These tests parse the scripts as text: they need neither torch nor a GPU, so
they run in any environment.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXAMPLES = _REPO_ROOT / "examples" / "deepseek-v4"

_FLASH_SCRIPT = _EXAMPLES / "run_deepseek_v4_flash.sh"
_BASE_SCRIPT = _EXAMPLES / "run_deepseek_v4.sh"


def _default_of(script: Path, var: str) -> str | None:
    """Return the default in ``export VAR=${VAR:-default}``, or None.

    The last assignment wins, mirroring shell semantics.
    """
    pattern = re.compile(rf"^\s*export\s+{re.escape(var)}=\$\{{{re.escape(var)}:-([^}}]*)\}}", re.MULTILINE)
    matches = pattern.findall(script.read_text(encoding="utf-8"))
    return matches[-1] if matches else None


@pytest.mark.parametrize("script", [_FLASH_SCRIPT, _BASE_SCRIPT], ids=["flash", "base"])
def test_fp8_indexer_defaults_off(script: Path) -> None:
    """Both launchers must leave the indexer QK in high precision by default."""
    assert script.is_file(), f"missing run script: {script}"
    default = _default_of(script, "USE_V4_FP8_INDEXER")
    assert default is not None, f"{script.name} must define USE_V4_FP8_INDEXER with a ${{VAR:-...}} default"
    assert default == "False", (
        f"{script.name} defaults USE_V4_FP8_INDEXER to {default!r}. The indexer selects which "
        "compressed KV entries each query sees, so enabling the fake-quant path by default "
        "changes model numerics -- and costs throughput while doing so, since the GEMM stays "
        "BF16. Keep it opt-in."
    )


def test_fp8_indexer_stays_overridable() -> None:
    """The knob keeps the ``${VAR:-default}`` form so QAT runs can opt back in."""
    text = _FLASH_SCRIPT.read_text(encoding="utf-8")
    assert "USE_V4_FP8_INDEXER=${USE_V4_FP8_INDEXER:-" in text


def test_flash_defaults_agree_with_base_script() -> None:
    """Flash must not silently disagree with the generic launcher on this knob.

    The Flash script is a thin wrapper that execs ``run_deepseek_v4.sh``; a
    divergent default here is exactly how the FP8 indexer ended up enabled in
    production while the yaml and the generic script both said otherwise.
    """
    assert _default_of(_FLASH_SCRIPT, "USE_V4_FP8_INDEXER") == _default_of(
        _BASE_SCRIPT, "USE_V4_FP8_INDEXER"
    )
