###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Every FlyDSL kernel of the MSA backend is built with a name, its symbol in profiles and ISA
dumps: the kernel and the tags of the build options it carries (``kernel_name``)."""

import ast
import pathlib

import pytest

FLYDSL_DIR = (
    pathlib.Path(__file__).resolve().parents[4]
    / "primus"
    / "backends"
    / "megatron"
    / "core"
    / "transformer"
    / "minimax_m3"
    / "flydsl"
)


def _kernel_decorators():
    """``(file, function, decorator call)`` of every ``@flyc.kernel(...)`` under ``FLYDSL_DIR``."""
    found = []
    for path in sorted(FLYDSL_DIR.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if not isinstance(node, ast.FunctionDef):
                continue
            for dec in node.decorator_list:
                call = dec if isinstance(dec, ast.Call) else None
                func = call.func if call is not None else dec
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "kernel"
                    and getattr(func.value, "id", "") == "flyc"
                ):
                    found.append((path.name, node.name, call))
    return found


def test_every_kernel_is_named():
    found = _kernel_decorators()
    assert len(found) >= 17
    unnamed = [
        f"{file}:{fn}"
        for file, fn, call in found
        if call is None or not any(k.arg == "name" for k in call.keywords)
    ]
    assert not unnamed, f"@flyc.kernel without name=: {unnamed}"


def test_kernel_name_joins_the_tags_it_is_given():
    pytest.importorskip("flydsl")
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_fwd import (
        kernel_name,
    )

    assert kernel_name("msa_fwd") == "msa_fwd"
    assert kernel_name("msa_fwd_w32", "wg4", "", "slse", "", "lean") == "msa_fwd_w32_wg4_slse_lean"
    assert kernel_name("msa_bwd_dkdv", "") == "msa_bwd_dkdv"
