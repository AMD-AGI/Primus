###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Unit tests for the tuning-agent workload resolver (``workload.py``).

The resolver flattens a Primus pretrain YAML plus the ``extends:`` chain of
its model file into a single :class:`ArchitectureRecord`.  These tests cover:

  * the small templating / coercion helpers (``_strip_env``,
    ``_coerce_opt_int``, ``_expand_primus_templates``, ``_parse_layer_ids``);
  * ``_vpp_from_layout`` stage-count inference;
  * ``resolve_workload`` end-to-end against tmp YAML trees, including the
    model ``extends:`` chain, MoE detection, FP8 precision detection, env
    templating in overrides, and VPP-from-layout inference.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("primus.agents.tuning_agent.workload")

from primus.agents.tuning_agent.workload import (  # noqa: E402
    ArchitectureRecord,
    _coerce_opt_int,
    _expand_primus_templates,
    _parse_layer_ids,
    _strip_env,
    _vpp_from_layout,
    resolve_workload,
)

# ─────────────────────────────────────────────────────────────────────────────
# _strip_env
# ─────────────────────────────────────────────────────────────────────────────


def test_strip_env_returns_default_segment():
    assert _strip_env("${PRIMUS_TP:4}") == "4"
    assert _strip_env("${PRIMUS_MODEL:gpt_oss_120B}") == "gpt_oss_120B"


def test_strip_env_passthrough_for_non_template():
    assert _strip_env("8") == "8"
    assert _strip_env(8) == 8
    assert _strip_env(None) is None
    # Not a ``${...:...}`` template → untouched.
    assert _strip_env("${NOCOLON}") == "${NOCOLON}"


# ─────────────────────────────────────────────────────────────────────────────
# _coerce_opt_int  (the fix this branch added for templated VPP values)
# ─────────────────────────────────────────────────────────────────────────────


def test_coerce_opt_int_handles_none_and_empty():
    assert _coerce_opt_int(None) is None
    assert _coerce_opt_int("") is None


def test_coerce_opt_int_strips_template_then_casts():
    assert _coerce_opt_int("${PRIMUS_VP:2}") == 2
    assert _coerce_opt_int("4") == 4
    assert _coerce_opt_int(3) == 3


def test_coerce_opt_int_raises_on_garbage():
    with pytest.raises(ValueError):
        _coerce_opt_int("not-an-int")


# ─────────────────────────────────────────────────────────────────────────────
# _expand_primus_templates
# ─────────────────────────────────────────────────────────────────────────────


def test_expand_primus_templates_replaces_with_defaults():
    assert _expand_primus_templates("${PRIMUS_MODEL:foo}.yaml") == "foo.yaml"
    # Multiple segments in one string.
    assert _expand_primus_templates("${A:1}_${B:2}") == "1_2"


def test_expand_primus_templates_noop_when_absent():
    assert _expand_primus_templates("plain.yaml") == "plain.yaml"


# ─────────────────────────────────────────────────────────────────────────────
# _parse_layer_ids
# ─────────────────────────────────────────────────────────────────────────────


def test_parse_layer_ids_from_string():
    assert _parse_layer_ids("0,3,4,7") == [0, 3, 4, 7]
    # Whitespace + trailing commas tolerated.
    assert _parse_layer_ids(" 1 , 2 ,") == [1, 2]


def test_parse_layer_ids_from_list():
    assert _parse_layer_ids([0, 1, 2]) == [0, 1, 2]


def test_parse_layer_ids_none_and_unparseable():
    assert _parse_layer_ids(None) is None
    assert _parse_layer_ids("a,b,c") is None
    assert _parse_layer_ids(3.5) is None


# ─────────────────────────────────────────────────────────────────────────────
# _vpp_from_layout
# ─────────────────────────────────────────────────────────────────────────────


def test_vpp_from_layout_basic():
    # 4 stages, PP=2 → VPP=2.
    assert _vpp_from_layout("t*1|t*1|t*1|t*1", 2) == 2


def test_vpp_from_layout_strips_trailing_loss_head():
    # Trailing ",L" marks the loss head attached to the last stage and must
    # not count as an extra stage.  4 stages, PP=4 → VPP=1.
    assert _vpp_from_layout("Et*1|t*1|t*1|t*1,L", 4) == 1


def test_vpp_from_layout_rejects_non_divisor_and_malformed():
    # 3 stages, PP=2 → not divisible.
    assert _vpp_from_layout("t*1|t*1|t*1", 2) is None
    assert _vpp_from_layout(None, 4) is None
    assert _vpp_from_layout("", 4) is None
    assert _vpp_from_layout("t*1|t*1", 0) is None


# ─────────────────────────────────────────────────────────────────────────────
# ArchitectureRecord.as_prompt_dict
# ─────────────────────────────────────────────────────────────────────────────


def test_as_prompt_dict_drops_raw_blocks():
    rec = ArchitectureRecord(
        model_name="m",
        raw_overrides={"a": 1},
        raw_model={"b": 2},
    )
    d = rec.as_prompt_dict()
    assert "raw_overrides" not in d
    assert "raw_model" not in d
    assert d["model_name"] == "m"


# ─────────────────────────────────────────────────────────────────────────────
# resolve_workload end-to-end
# ─────────────────────────────────────────────────────────────────────────────


def _write_primus_tree(root: Path, model_name: str, *, model_body: dict, base_body: dict | None = None):
    """Lay out a minimal primus/configs/models/<framework>/ tree.

    Returns the model filename (relative) the workload yaml should reference.
    """
    import yaml

    models_dir = root / "primus" / "configs" / "models" / "megatron"
    models_dir.mkdir(parents=True, exist_ok=True)
    if base_body is not None:
        (models_dir / "base.yaml").write_text(yaml.safe_dump(base_body))
        model_body = {**model_body, "extends": "base.yaml"}
    (models_dir / model_name).write_text(yaml.safe_dump(model_body))
    return model_name


def _write_workload(path: Path, *, model: str, overrides: dict, extra_top: dict | None = None):
    import yaml

    doc = {"modules": {"pre_trainer": {"framework": "megatron", "model": model, "overrides": overrides}}}
    if extra_top:
        doc.update(extra_top)
    path.write_text(yaml.safe_dump(doc))


def test_resolve_workload_dense_with_extends_chain(tmp_path):
    root = tmp_path / "repo"
    _write_primus_tree(
        root,
        "dense.yaml",
        model_body={"num_layers": 32, "hidden_size": 4096},
        base_body={"num_attention_heads": 32, "seq_length": 4096, "vocab_size": 50000},
    )
    wl = tmp_path / "workload.yaml"
    _write_workload(
        wl,
        model="dense.yaml",
        overrides={
            "tensor_model_parallel_size": 2,
            "pipeline_model_parallel_size": 4,
            "micro_batch_size": 2,
            "global_batch_size": 128,
        },
    )

    arch = resolve_workload(wl, primus_root=root)

    assert arch.model_name == "dense"
    assert arch.num_layers == 32
    assert arch.hidden_size == 4096
    # Pulled from the extends base.
    assert arch.num_attention_heads == 32
    assert arch.seq_length == 4096
    assert arch.vocab_size == 50000
    assert arch.is_moe is False
    assert arch.num_experts == 0
    assert arch.tensor_model_parallel_size == 2
    assert arch.pipeline_model_parallel_size == 4
    assert arch.micro_batch_size == 2
    assert arch.global_batch_size == 128
    assert arch.precision == "bf16"


def test_resolve_workload_detects_moe_and_fp8(tmp_path):
    root = tmp_path / "repo"
    _write_primus_tree(
        root,
        "moe.yaml",
        model_body={
            "num_layers": 24,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_experts": 8,
            "moe_router_topk": 2,
            "seq_length": 4096,
        },
    )
    wl = tmp_path / "moe_workload.yaml"
    _write_workload(
        wl,
        model="moe.yaml",
        overrides={"expert_model_parallel_size": 8, "fp8": "hybrid"},
    )

    arch = resolve_workload(wl, primus_root=root)
    assert arch.is_moe is True
    assert arch.num_experts == 8
    assert arch.moe_router_topk == 2
    assert arch.expert_model_parallel_size == 8
    # ``fp8`` present in overrides → precision flips to fp8 and the fp8 knob
    # is captured.
    assert arch.precision == "fp8"
    assert arch.fp8 == "hybrid"


def test_resolve_workload_expands_env_templates_in_model_and_overrides(tmp_path):
    root = tmp_path / "repo"
    _write_primus_tree(
        root,
        "templated.yaml",
        model_body={"num_layers": 16, "hidden_size": 2048, "num_attention_heads": 16, "seq_length": 2048},
    )
    wl = tmp_path / "templated_workload.yaml"
    _write_workload(
        wl,
        # Model name itself uses a ${PRIMUS_MODEL:...} template.
        model="${PRIMUS_MODEL:templated}.yaml",
        overrides={
            "tensor_model_parallel_size": "${PRIMUS_TP:2}",
            "pipeline_model_parallel_size": "${PRIMUS_PP:1}",
        },
    )

    arch = resolve_workload(wl, primus_root=root)
    assert arch.model_name == "templated"
    assert arch.tensor_model_parallel_size == 2
    assert arch.pipeline_model_parallel_size == 1


def test_resolve_workload_infers_vpp_from_layout(tmp_path):
    root = tmp_path / "repo"
    _write_primus_tree(
        root,
        "layout.yaml",
        model_body={"num_layers": 32, "hidden_size": 4096, "num_attention_heads": 32, "seq_length": 4096},
    )
    wl = tmp_path / "layout_workload.yaml"
    # PP=2 with a 4-stage layout → VPP should be inferred as 2 even though no
    # explicit virtual_pipeline_model_parallel_size is set.
    _write_workload(
        wl,
        model="layout.yaml",
        overrides={
            "pipeline_model_parallel_size": 2,
            "pipeline_model_parallel_layout": "t*1|t*1|t*1|t*1,L",
        },
    )

    arch = resolve_workload(wl, primus_root=root)
    assert arch.pipeline_model_parallel_size == 2
    assert arch.virtual_pipeline_model_parallel_size == 2


def test_resolve_workload_explicit_templated_vpp(tmp_path):
    """A ``${PRIMUS_VP:2}``-templated explicit VPP must be coerced to int.

    This is the exact path the ``_coerce_opt_int`` fix added — previously the
    raw template string leaked into ``virtual_pipeline_model_parallel_size``.
    """
    root = tmp_path / "repo"
    _write_primus_tree(
        root,
        "vpp.yaml",
        model_body={"num_layers": 32, "hidden_size": 4096, "num_attention_heads": 32, "seq_length": 4096},
    )
    wl = tmp_path / "vpp_workload.yaml"
    _write_workload(
        wl,
        model="vpp.yaml",
        overrides={
            "pipeline_model_parallel_size": 4,
            "virtual_pipeline_model_parallel_size": "${PRIMUS_VP:2}",
        },
    )

    arch = resolve_workload(wl, primus_root=root)
    assert arch.virtual_pipeline_model_parallel_size == 2
    assert isinstance(arch.virtual_pipeline_model_parallel_size, int)


def test_resolve_workload_missing_model_raises(tmp_path):
    import yaml

    wl = tmp_path / "bad.yaml"
    wl.write_text(yaml.safe_dump({"modules": {"pre_trainer": {"framework": "megatron"}}}))
    with pytest.raises(ValueError, match="no modules.pre_trainer.model"):
        resolve_workload(wl, primus_root=tmp_path)


def test_resolve_workload_unknown_model_file_raises(tmp_path):
    root = tmp_path / "repo"
    (root / "primus" / "configs" / "models" / "megatron").mkdir(parents=True)
    wl = tmp_path / "wl.yaml"
    _write_workload(wl, model="does_not_exist.yaml", overrides={})
    with pytest.raises(FileNotFoundError):
        resolve_workload(wl, primus_root=root)
