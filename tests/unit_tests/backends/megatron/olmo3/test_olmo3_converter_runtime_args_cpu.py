###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CPU-only (no torch, no Megatron): the converter runtime / save args the OLMo-3 converter sets
(native_loader_olmo3.set_converter_runtime_args / set_save_iteration) equal the values the Llama,
Qwen3 and DeepSeek converters set inline in their build_margs. Both sides are read from the
source with ``ast``, so the copies cannot drift without this test failing, also where
megatron.core is not importable."""

import ast
from pathlib import Path

import pytest

_CKPT = Path(__file__).resolve().parents[5] / "primus/backends/megatron/checkpoint"

# the runtime / save attributes every native converter sets to a constant
_KEYS = (
    "use_cpu_initialization",
    "perform_initialization",
    "transformer_impl",
    "gradient_accumulation_fusion",
    "async_tensor_model_parallel_allreduce",
    "bias_gelu_fusion",
    "bias_swiglu_fusion",
    "masked_softmax_fusion",
    "bias_dropout_fusion",
    "apply_query_key_layer_scaling",
    "ckpt_format",
    "no_save_optim",
    "no_save_rng",
    "no_load_optim",
    "no_load_rng",
    "save_interval",
    "async_save",
    "ckpt_assume_constant_structure",
    "iteration",
)


def _constant_assignments(func: ast.FunctionDef, target: str) -> dict:
    """``{attr: value}`` for every ``<target>.<attr> = <constant>`` in ``func`` (last one wins)."""
    out = {}
    for node in ast.walk(func):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
            continue
        for t in node.targets:
            if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == target:
                out[t.attr] = node.value.value
    return out


def _function(module: str, name: str) -> ast.FunctionDef:
    tree = ast.parse((_CKPT / f"{module}.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{module}.{name} not found")


def _olmo3_values() -> dict:
    values = _constant_assignments(_function("native_loader_olmo3", "set_converter_runtime_args"), "margs")
    values.update(_constant_assignments(_function("native_loader_olmo3", "set_save_iteration"), "margs"))
    return values


def test_olmo3_sets_every_runtime_key():
    values = _olmo3_values()
    assert set(_KEYS) <= set(values), sorted(set(_KEYS) - set(values))


@pytest.mark.parametrize("family", ["llama", "qwen3", "deepseek"])
def test_other_converters_set_the_same_runtime_args(family):
    inline = _constant_assignments(_function(f"native_loader_{family}", "build_margs"), "m")
    olmo3 = _olmo3_values()
    assert {k: inline.get(k) for k in _KEYS} == {k: olmo3[k] for k in _KEYS}
