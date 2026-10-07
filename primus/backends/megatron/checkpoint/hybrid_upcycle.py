###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Upcycle a dense Megatron Transformer checkpoint into a non-MoE hybrid model.

Primus hybrid stacks (Mamba, Gated DeltaNet, Kimi Delta Attention) are a flat
list of sublayers. ``HybridStack.allocate_layers`` pairs every sequence mixer
with its own MLP::

    dense layer i  ->  hybrid sublayers (2i, 2i + 1) = (mixer, MLP)

Weight initialization follows HyLo's from-teacher upcycling
(``HybridModelWrapper`` on ``AMD-Hybrid-Models`` ``feat/HyLo``):

* MLA ``*`` slots run the SVD query/KV reinit from ``DeepseekV3Attention.re_init_q``
  and ``re_init_kv``, and copy the overlapping output projection.
* GDN ``M`` slots copy Q, K, V, and O with ``_copy_llama_attn_to_gdn``, including
  the GQA repeat, into Megatron's fused ``mixer.in_proj`` / ``mixer.out_proj``.
* Mamba ``M`` slots copy V, K, and Q into the ``x``, ``B``, and ``C`` slices of
  ``in_proj`` the way ``HybridModelWrapper`` does, only when that tensor uses
  HyLo's ``[z | x | B | C | dt]`` widths.

Embeddings, the final norm, and every MLP are still copied in full. KDA has no
HyLo recipe, and a Mamba ``in_proj`` whose width is not HyLo's stays at the
hybrid initialization.

This is separate from Megatron's ``--moe-use-upcycling``, which duplicates a
dense MLP into experts. Expert patterns (``E``), pipeline markers (``|``), and
MTP markers (``/``) are rejected here.
"""

from __future__ import annotations

import argparse
import json
import re
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Optional

import torch

ATTENTION = "*"
MAMBA = "M"
MLP = "-"
MOE = "E"

_LAYER_RE = re.compile(r"^decoder\.layers\.(\d+)\.")

# TE fuses the pre-MLP norm into fc1; the no-TE hybrid spec stores it separately.
_MLP_NORM_ALIASES = (
    ("pre_mlp_layernorm.weight", "mlp.linear_fc1.layer_norm_weight"),
    ("pre_mlp_layernorm.bias", "mlp.linear_fc1.layer_norm_bias"),
)
_ATTN_NORM_ALIASES = (
    ("input_layernorm.weight", "self_attention.linear_qkv.layer_norm_weight"),
    ("input_layernorm.bias", "self_attention.linear_qkv.layer_norm_bias"),
)
_FINAL_NORM_ALIASES = (
    "decoder.final_norm.weight",
    "decoder.final_layernorm.weight",
)
_DROPPED_CHECKPOINT_KEYS = (
    "optimizer",
    "opt_param_scheduler",
    "rng_state",
    "rerun_state_machine",
    "num_floating_point_operations_so_far",
)


@dataclass
class UpcycleReport:
    """Record of what was transplanted and what stayed at hybrid init."""

    pattern: str
    num_dense_layers: int
    copied: list[str] = field(default_factory=list)
    attention_left_initialized: list[str] = field(default_factory=list)
    attention_shape_mismatch: list[str] = field(default_factory=list)
    mixer_layers_left_initialized: list[int] = field(default_factory=list)
    hylo_partial: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pattern": self.pattern,
            "num_dense_layers": self.num_dense_layers,
            "num_copied": len(self.copied),
            "copied": self.copied,
            "attention_left_initialized": self.attention_left_initialized,
            "attention_shape_mismatch": self.attention_shape_mismatch,
            "mixer_layers_left_initialized": self.mixer_layers_left_initialized,
            "hylo_partial": self.hylo_partial,
        }


@dataclass
class UpcycleLayout:
    """Dense and hybrid widths needed to apply the HyLo init recipes.

    Built from the two checkpoints' ``args`` namespaces. Fields that HyLo reads
    off ``HybridConfig`` / the teacher Llama config are optional here so a
    checkpoint that does not use that recipe can still be converted.
    """

    num_attention_heads: int
    num_query_groups: int
    head_dim: int
    q_lora_rank: Optional[int] = None
    kv_lora_rank: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None
    linear_type: Optional[str] = None
    gdn_num_key_heads: Optional[int] = None
    gdn_key_head_dim: Optional[int] = None
    gdn_num_value_heads: Optional[int] = None
    gdn_value_head_dim: Optional[int] = None
    mamba_d_inner: Optional[int] = None
    mamba_d_xb: Optional[int] = None
    mamba_nheads: Optional[int] = None


def layout_from_args(dense_args: object, hybrid_args: object) -> Optional[UpcycleLayout]:
    """Read HyLo widths from Megatron checkpoint args. Missing head counts skip the recipes."""
    heads = _first_arg(dense_args, hybrid_args, "num_attention_heads")
    if heads is None:
        return None
    query_groups = _first_arg(dense_args, hybrid_args, "num_query_groups") or heads
    hidden = _first_arg(dense_args, hybrid_args, "hidden_size")
    head_dim = _first_arg(dense_args, hybrid_args, "kv_channels")
    if head_dim is None and hidden is not None:
        head_dim = int(hidden) // int(heads)
    if head_dim is None:
        return None

    linear_type = _first_arg(hybrid_args, dense_args, "hybrid_linear_type")
    if linear_type is None and _first_arg(hybrid_args, dense_args, "linear_num_key_heads") is not None:
        linear_type = "gdn"
    elif linear_type is None and _first_arg(hybrid_args, dense_args, "mamba_expand") is not None:
        linear_type = "mamba"

    d_inner = None
    expand = _first_arg(hybrid_args, dense_args, "mamba_expand")
    if expand is not None and hidden is not None:
        d_inner = int(expand) * int(hidden)
    d_xb = _first_arg(hybrid_args, dense_args, "mamba_d_xb")
    if d_xb is None:
        d_xb = d_inner
    nheads = None
    mamba_head_dim = _first_arg(hybrid_args, dense_args, "mamba_head_dim")
    if d_inner is not None and mamba_head_dim:
        nheads = int(d_inner) // int(mamba_head_dim)

    return UpcycleLayout(
        num_attention_heads=int(heads),
        num_query_groups=int(query_groups),
        head_dim=int(head_dim),
        q_lora_rank=_optional_int(_first_arg(hybrid_args, dense_args, "q_lora_rank")),
        kv_lora_rank=_optional_int(_first_arg(hybrid_args, dense_args, "kv_lora_rank")),
        qk_nope_head_dim=_optional_int(_first_arg(hybrid_args, dense_args, "qk_head_dim")),
        qk_rope_head_dim=_optional_int(_first_arg(hybrid_args, dense_args, "qk_pos_emb_head_dim")),
        v_head_dim=_optional_int(_first_arg(hybrid_args, dense_args, "v_head_dim")),
        linear_type=None if linear_type is None else str(linear_type).lower(),
        gdn_num_key_heads=_optional_int(_first_arg(hybrid_args, dense_args, "linear_num_key_heads")),
        gdn_key_head_dim=_optional_int(_first_arg(hybrid_args, dense_args, "linear_key_head_dim")),
        gdn_num_value_heads=_optional_int(_first_arg(hybrid_args, dense_args, "linear_num_value_heads")),
        gdn_value_head_dim=_optional_int(_first_arg(hybrid_args, dense_args, "linear_value_head_dim")),
        mamba_d_inner=None if d_inner is None else int(d_inner),
        mamba_d_xb=None if d_xb is None else int(d_xb),
        mamba_nheads=nheads,
    )


def hybrid_pattern_from_ratio(num_sublayers: int, attention_ratio: float) -> str:
    """Build the paired mixer/MLP pattern used by ``HybridStack.allocate_layers``.

    ``num_sublayers`` is the hybrid stack length (twice the dense layer count).
    ``attention_ratio`` is the fraction of *mixer* slots that stay attention,
    matching ``hybrid_attention_ratio`` in the model YAML. A ratio of 0.25 on
    32 sublayers yields ``(*-M-M-M-)`` repeated four times.
    """
    if num_sublayers <= 0 or num_sublayers % 2 != 0:
        raise ValueError(f"num_sublayers must be a positive even integer, got {num_sublayers}")
    if not 0.0 <= attention_ratio <= 1.0:
        raise ValueError(f"attention_ratio must be in [0, 1], got {attention_ratio}")

    num_blocks = num_sublayers // 2
    num_attention = int(num_blocks * attention_ratio)
    num_mamba = num_blocks - num_attention
    if num_attention == 0:
        return (MAMBA + MLP) * num_blocks

    if attention_ratio <= 0.5:
        per_attention = num_mamba // num_attention
        base = [ATTENTION, MLP] + [MAMBA, MLP] * per_attention
        layers = base * num_attention
        layers += [MAMBA, MLP] * (num_mamba % num_attention)
    else:
        layers = [ATTENTION, MLP] * (num_attention - num_mamba)
        base = [ATTENTION, MLP, MAMBA, MLP]
        layers += base * num_mamba

    pattern = "".join(layers)
    if len(pattern) != num_sublayers:
        raise ValueError(
            f"attention_ratio={attention_ratio} does not tile {num_sublayers} sublayers "
            f"(produced {pattern!r}). Pass an explicit hybrid pattern."
        )
    return pattern


def validate_upcycle_pattern(pattern: str, num_dense_layers: int) -> str:
    """Require a paired ``(*|M)-`` pattern with no MoE, pipeline, or MTP symbols."""
    pattern = "".join(pattern.split())
    if not pattern:
        raise ValueError("hybrid pattern is empty")
    unsupported = sorted(set(pattern) & set(MOE + "|/"))
    if unsupported:
        raise ValueError(
            "dense-to-hybrid upcycling only supports attention ('*'), recurrent ('M'), "
            f"and MLP ('-') sublayers; found {unsupported}"
        )
    expected = 2 * num_dense_layers
    if len(pattern) != expected:
        raise ValueError(
            f"pattern length {len(pattern)} != 2 * dense layers ({expected}). "
            "Each dense block becomes one mixer sublayer and one MLP sublayer."
        )
    for index, symbol in enumerate(pattern):
        if index % 2 == 0 and symbol not in (ATTENTION, MAMBA):
            raise ValueError(f"pattern[{index}] must be '{ATTENTION}' or '{MAMBA}', got {symbol!r}")
        if index % 2 == 1 and symbol != MLP:
            raise ValueError(f"pattern[{index}] must be '{MLP}', got {symbol!r}")
    return pattern


def count_decoder_layers(state: Mapping[str, torch.Tensor]) -> int:
    """Return the contiguous ``decoder.layers.{i}`` count."""
    indices = set()
    for key in state:
        match = _LAYER_RE.match(key)
        if match:
            indices.add(int(match.group(1)))
    if not indices:
        raise ValueError("checkpoint has no decoder.layers.* parameters")
    expected = set(range(max(indices) + 1))
    if indices != expected:
        missing = sorted(expected - indices)
        raise ValueError(f"decoder layer indices are not contiguous; missing {missing}")
    return max(indices) + 1


def upcycle_state_dict(
    dense_state: Mapping[str, torch.Tensor],
    hybrid_state: Mapping[str, torch.Tensor],
    pattern: str,
    layout: Optional[UpcycleLayout] = None,
) -> tuple[dict[str, torch.Tensor], UpcycleReport]:
    """Overlay transplanted dense weights onto a copy of ``hybrid_state``.

    MLP and embedding transplants that disagree in shape raise. When ``layout``
    is set, MLA and GDN/Mamba slots follow the HyLo recipes. Without it, or
    when a mixer is not a HyLo GDN or Mamba layout, attention is copied only on
    exact shape match and recurrent mixers stay at hybrid init.
    """
    num_dense = count_decoder_layers(dense_state)
    pattern = validate_upcycle_pattern(pattern, num_dense)
    num_hybrid = count_decoder_layers(hybrid_state)
    if num_hybrid != len(pattern):
        raise ValueError(
            f"hybrid checkpoint has {num_hybrid} decoder sublayers, pattern has {len(pattern)}"
        )

    upcycled = {key: value for key, value in hybrid_state.items()}
    report = UpcycleReport(pattern=pattern, num_dense_layers=num_dense)

    _copy_global(dense_state, upcycled, report)
    for dense_index in range(num_dense):
        mixer_symbol = pattern[2 * dense_index]
        mixer_index = 2 * dense_index
        mlp_index = mixer_index + 1
        _copy_mlp(dense_state, upcycled, dense_index, mlp_index, report)
        if mixer_symbol == ATTENTION:
            _copy_attention(dense_state, upcycled, dense_index, mixer_index, report, layout)
        elif not _copy_mixer_hylo(dense_state, upcycled, dense_index, mixer_index, report, layout):
            report.mixer_layers_left_initialized.append(mixer_index)
    return upcycled, report


def build_upcycle_checkpoint(hybrid_checkpoint: dict, upcycled_state: Mapping[str, torch.Tensor]) -> dict:
    """Return a legacy Megatron checkpoint envelope at iteration 0.

    Optimizer and RNG state are dropped. Load the result with ``finetune: true``,
    ``no_load_optim: true``, and ``no_load_rng: true``.
    """
    checkpoint = {
        key: value for key, value in hybrid_checkpoint.items() if key not in _DROPPED_CHECKPOINT_KEYS
    }
    checkpoint["model"] = dict(upcycled_state)
    checkpoint["iteration"] = 0
    checkpoint.setdefault("checkpoint_version", 3.0)
    args = checkpoint.get("args")
    if isinstance(args, Namespace) and hasattr(args, "iteration"):
        args.iteration = 0
    return checkpoint


def pattern_from_checkpoint_args(args: object) -> Optional[str]:
    """Read a hybrid pattern off a Megatron ``args`` namespace or dict."""
    for name in ("hybrid_layer_pattern", "hybrid_override_pattern"):
        value = _arg(args, name)
        if isinstance(value, str) and value.strip():
            return "".join(value.split())
    return None


def attention_ratio_from_args(args: object) -> Optional[float]:
    value = _arg(args, "hybrid_attention_ratio")
    if value is None:
        return None
    return float(value)


def resolve_checkpoint_file(path: Path) -> Path:
    """Find ``mp_rank_00/model_optim_rng.pt`` under a file, iter dir, or load root."""
    path = Path(path).expanduser()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(path)

    if _is_distcp_dir(path):
        raise ValueError(
            f"{path} is a distributed checkpoint. Consolidate it to the legacy torch "
            "format with tools/hybrid/consolidate_distcp_to_torch.py first."
        )

    ranks = sorted(p for p in path.glob("mp_rank_*") if p.is_dir())
    if len(ranks) > 1:
        raise ValueError(
            f"{path} has {len(ranks)} model-parallel shards ({ranks[0].name} .. {ranks[-1].name}). "
            "Upcycling expects a single mp_rank_00 checkpoint (TP=PP=1)."
        )
    direct = path / "mp_rank_00" / "model_optim_rng.pt"
    if direct.is_file():
        return direct

    tracker = path / "latest_checkpointed_iteration.txt"
    if tracker.is_file():
        iteration = tracker.read_text().strip()
        iter_name = "release" if iteration == "release" else f"iter_{int(iteration):07d}"
        candidate = path / iter_name
        if _is_distcp_dir(candidate):
            raise ValueError(
                f"{candidate} is a distributed checkpoint. Consolidate it to the legacy torch "
                "format with tools/hybrid/consolidate_distcp_to_torch.py first."
            )
        resolved = candidate / "mp_rank_00" / "model_optim_rng.pt"
        if resolved.is_file():
            return resolved
        raise FileNotFoundError(resolved)
    raise FileNotFoundError(f"no mp_rank_00/model_optim_rng.pt under {path}")


def load_checkpoint(path: Path) -> tuple[dict[str, torch.Tensor], dict]:
    """Load a legacy torch checkpoint and return ``(model state, envelope)``."""
    # Checkpoint args unpickle Megatron classes. Training adds Megatron to
    # PYTHONPATH; a standalone conversion does not.
    from primus.backends.megatron.checkpoint.native_convert_common import ensure_megatron_on_path

    ensure_megatron_on_path()
    ckpt_file = resolve_checkpoint_file(path)
    checkpoint = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{ckpt_file} did not contain a checkpoint dict")
    state = unwrap_model_state(checkpoint)
    _require_plain_tensors(state, ckpt_file)
    return state, checkpoint


def unwrap_model_state(checkpoint: Mapping) -> dict:
    if "model" not in checkpoint:
        if any(str(key).startswith(("embedding.", "decoder.")) for key in checkpoint):
            return dict(checkpoint)
        raise ValueError("checkpoint has no 'model' entry")
    model = checkpoint["model"]
    if isinstance(model, (list, tuple)):
        if len(model) != 1:
            raise ValueError(
                "virtual pipeline checkpoints store model0, model1, ... ; "
                "re-save the hybrid init with pipeline_model_parallel_size: 1"
            )
        model = model[0]
    if not isinstance(model, Mapping):
        raise ValueError(f"checkpoint['model'] must be a state dict, got {type(model).__name__}")
    return dict(model)


def save_upcycle_checkpoint(checkpoint: dict, output_dir: Path, report: UpcycleReport) -> Path:
    """Write ``iter_0000000`` plus the iteration tracker and a JSON report."""
    iter_dir = output_dir / "iter_0000000" / "mp_rank_00"
    iter_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = iter_dir / "model_optim_rng.pt"
    torch.save(checkpoint, ckpt_path)
    (output_dir / "latest_checkpointed_iteration.txt").write_text("0\n")
    (output_dir / "upcycle_report.json").write_text(json.dumps(report.to_dict(), indent=2) + "\n")
    return ckpt_path


def resolve_pattern(
    pattern: Optional[str],
    attention_ratio: Optional[float],
    hybrid_args: object,
    num_sublayers: int,
) -> str:
    if pattern:
        return "".join(pattern.split())
    from_args = pattern_from_checkpoint_args(hybrid_args)
    if from_args:
        return from_args
    ratio = attention_ratio if attention_ratio is not None else attention_ratio_from_args(hybrid_args)
    if ratio is None:
        raise ValueError(
            "pass --hybrid-pattern, or save hybrid_override_pattern / hybrid_attention_ratio "
            "in the hybrid init checkpoint"
        )
    return hybrid_pattern_from_ratio(num_sublayers, ratio)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Upcycle a dense Megatron checkpoint into a non-MoE hybrid checkpoint. "
            "Embeddings and MLPs are copied. MLA, GDN, and Mamba slots are initialized "
            "with the HyLo from-teacher recipes when the checkpoint args carry those widths."
        )
    )
    parser.add_argument("--dense-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--hybrid-init-checkpoint",
        type=Path,
        required=True,
        help=(
            "Legacy torch checkpoint of the target hybrid model. "
            "A mock run with lr: 0, train_iters: 1, save_interval: 1 writes one."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--hybrid-pattern",
        default=None,
        help="Paired pattern such as '*-M-M-M-'. Defaults to the pattern stored on the hybrid checkpoint.",
    )
    parser.add_argument(
        "--attention-ratio",
        type=float,
        default=None,
        help="Used only when the hybrid checkpoint has no pattern. Matches hybrid_attention_ratio.",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> UpcycleReport:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    dense_state, dense_ckpt = load_checkpoint(args.dense_checkpoint)
    hybrid_state, hybrid_ckpt = load_checkpoint(args.hybrid_init_checkpoint)
    pattern = resolve_pattern(
        args.hybrid_pattern,
        args.attention_ratio,
        hybrid_ckpt.get("args"),
        count_decoder_layers(hybrid_state),
    )
    layout = layout_from_args(dense_ckpt.get("args"), hybrid_ckpt.get("args"))
    upcycled, report = upcycle_state_dict(dense_state, hybrid_state, pattern, layout)
    checkpoint = build_upcycle_checkpoint(hybrid_ckpt, upcycled)
    ckpt_path = save_upcycle_checkpoint(checkpoint, args.output_dir, report)
    print(f"pattern: {report.pattern}")
    print(f"copied {len(report.copied)} tensors")
    print(f"attention tensors left at hybrid init: {len(report.attention_left_initialized)}")
    print(f"attention shape mismatches left at hybrid init: {len(report.attention_shape_mismatch)}")
    print(f"HyLo partial slice copies: {len(report.hylo_partial)}")
    print(f"recurrent mixer sublayers left at hybrid init: {report.mixer_layers_left_initialized}")
    print(f"wrote {ckpt_path}")
    print("Load this directory with finetune: true, no_load_optim: true, no_load_rng: true.")
    return report


def _copy_global(dense: Mapping, hybrid: dict, report: UpcycleReport) -> None:
    if "embedding.word_embeddings.weight" not in dense or "embedding.word_embeddings.weight" not in hybrid:
        raise ValueError("both checkpoints must contain embedding.word_embeddings.weight")
    _assign_matching(
        dense,
        hybrid,
        "embedding.word_embeddings.weight",
        "embedding.word_embeddings.weight",
        report,
        required=True,
    )
    if "output_layer.weight" in hybrid:
        _assign_matching(dense, hybrid, "output_layer.weight", "output_layer.weight", report, required=False)

    target_norm = next((key for key in _FINAL_NORM_ALIASES if key in hybrid), None)
    if target_norm is None:
        raise ValueError(
            "hybrid checkpoint has no decoder.final_norm.weight or decoder.final_layernorm.weight"
        )
    source_norm = next((key for key in _FINAL_NORM_ALIASES if key in dense), None)
    if source_norm is None:
        raise ValueError("dense checkpoint has no final norm")
    _assign_matching(dense, hybrid, source_norm, target_norm, report, required=True)


def _copy_mlp(
    dense: Mapping, hybrid: dict, dense_index: int, hybrid_index: int, report: UpcycleReport
) -> None:
    source = _relative_keys(dense, dense_index)
    target = _relative_keys(hybrid, hybrid_index)
    copied_weights = set()
    for relative, target_key in sorted(target.items()):
        if _skip_key(relative) or not _is_mlp_key(relative):
            continue
        source_relative = _alias_lookup(source, relative, _MLP_NORM_ALIASES)
        if source_relative is None:
            raise ValueError(
                f"dense layer {dense_index} has no tensor for hybrid MLP key {target_key}. "
                "The dense and hybrid MLP layouts must match (including SwiGLU and bias)."
            )
        source_key = source[source_relative]
        _assign_matching(dense, hybrid, source_key, target_key, report, required=True)
        if relative in ("mlp.linear_fc1.weight", "mlp.linear_fc2.weight"):
            copied_weights.add(relative)
    missing = {"mlp.linear_fc1.weight", "mlp.linear_fc2.weight"} - copied_weights
    if missing:
        raise ValueError(f"hybrid MLP sublayer {hybrid_index} is missing {sorted(missing)}")


def _copy_attention(
    dense: Mapping,
    hybrid: dict,
    dense_index: int,
    hybrid_index: int,
    report: UpcycleReport,
    layout: Optional[UpcycleLayout],
) -> None:
    source = _relative_keys(dense, dense_index)
    target = _relative_keys(hybrid, hybrid_index)
    if layout is not None and _has_mla_down_proj(target):
        _copy_mla_hylo(dense, hybrid, dense_index, hybrid_index, source, target, report, layout)
        return
    for relative, target_key in sorted(target.items()):
        if _skip_key(relative) or _is_mlp_key(relative):
            continue
        source_relative = _alias_lookup(source, relative, _ATTN_NORM_ALIASES)
        if source_relative is None:
            report.attention_left_initialized.append(target_key)
            continue
        source_key = source[source_relative]
        if not _same_shape(dense[source_key], hybrid[target_key]):
            report.attention_shape_mismatch.append(target_key)
            continue
        _assign_matching(dense, hybrid, source_key, target_key, report, required=True)


def _copy_mla_hylo(
    dense: Mapping,
    hybrid: dict,
    dense_index: int,
    hybrid_index: int,
    source: Mapping[str, str],
    target: Mapping[str, str],
    report: UpcycleReport,
    layout: UpcycleLayout,
) -> None:
    """Port of HyLo ``init_with_svd`` onto Megatron MLA tensor names.

    ``re_init_q`` / ``re_init_kv`` fill the compressed nope projections. RoPE
    channels in ``linear_q_up_proj`` and ``linear_kv_down_proj`` are not part of
    that SVD, so they stay at hybrid init. ``out_proj`` copies the overlapping
    columns of the dense output projection.
    """
    _copy_norm_alias(dense, hybrid, source, target, "input_layernorm.weight", _ATTN_NORM_ALIASES, report)
    if not _mla_ranks_ready(layout):
        for relative, target_key in sorted(target.items()):
            if relative.startswith("self_attention.linear_q_") or relative.startswith("self_attention.linear_kv_"):
                report.attention_left_initialized.append(target_key)
        return

    q, k, v = _dense_qkv(dense, dense_index, source, layout)
    q_down, q_up_nope = hylo_reinit_q(
        q,
        layout.q_lora_rank,
        layout.num_attention_heads,
        layout.head_dim,
        layout.qk_nope_head_dim,
    )
    _assign_tensor(hybrid, target["self_attention.linear_q_down_proj.weight"], q_down, report, partial=False)
    q_up_key = target["self_attention.linear_q_up_proj.weight"]
    _write_q_up_nope(hybrid, q_up_key, q_up_nope, layout, report)

    kv_down, kv_up = hylo_reinit_kv(
        k,
        v,
        layout.num_query_groups,
        layout.head_dim,
        _mla_kv_heads(hybrid[target["self_attention.linear_kv_up_proj.weight"]], layout),
        layout.kv_lora_rank,
        layout.qk_nope_head_dim,
        layout.v_head_dim,
    )
    kv_down_key = target["self_attention.linear_kv_down_proj.weight"]
    _write_kv_down_prefix(hybrid, kv_down_key, kv_down, report)
    _assign_tensor(hybrid, target["self_attention.linear_kv_up_proj.weight"], kv_up, report, partial=False)

    if "self_attention.linear_proj.weight" in target:
        dense_proj = _dense_proj_key(source)
        if dense_proj is None:
            report.attention_left_initialized.append(target["self_attention.linear_proj.weight"])
        else:
            _copy_overlapping_columns(
                hybrid, target["self_attention.linear_proj.weight"], dense[dense_proj], report
            )


def _copy_mixer_hylo(
    dense: Mapping,
    hybrid: dict,
    dense_index: int,
    hybrid_index: int,
    report: UpcycleReport,
    layout: Optional[UpcycleLayout],
) -> bool:
    """Apply HyLo's GDN or Mamba QKVO copy. Return False when no recipe matches."""
    if layout is None:
        return False
    source = _relative_keys(dense, dense_index)
    target = _relative_keys(hybrid, hybrid_index)
    if "mixer.in_proj.weight" not in target:
        return False
    linear_type = _mixer_linear_type(target, layout)
    if linear_type == "gdn":
        _copy_gdn_hylo(dense, hybrid, dense_index, source, target, report, layout)
        return True
    if linear_type == "mamba":
        return _copy_mamba_hylo(dense, hybrid, dense_index, source, target, report, layout)
    return False


def _copy_gdn_hylo(
    dense: Mapping,
    hybrid: dict,
    dense_index: int,
    source: Mapping[str, str],
    target: Mapping[str, str],
    report: UpcycleReport,
    layout: UpcycleLayout,
) -> None:
    """Port of ``_copy_llama_attn_to_gdn`` into fused Megatron ``in_proj`` slices.

    Slice order matches ``convert_gdn_hybrid_to_fla_hf.convert_gdn_block``:
    q, k, v, g, b, a. Gate, beta, and A stay at hybrid init, as in HyLo.
    """
    key_dim = layout.gdn_num_key_heads * layout.gdn_key_head_dim
    value_dim = layout.gdn_num_value_heads * layout.gdn_value_head_dim
    expected_rows = key_dim * 2 + value_dim * 2 + layout.gdn_num_value_heads * 2
    in_proj_key = target["mixer.in_proj.weight"]
    if hybrid[in_proj_key].shape[0] != expected_rows:
        raise ValueError(
            f"GDN mixer.in_proj rows {hybrid[in_proj_key].shape[0]} != HyLo/Primus "
            f"fused width {expected_rows} (q, k, v, g, b, a)"
        )
    q, k, v = _repeat_gqa(
        *_dense_qkv(dense, dense_index, source, layout),
        layout.num_attention_heads,
        layout.num_query_groups,
        layout.head_dim,
    )
    updated = hybrid[in_proj_key].detach().to(device="cpu").clone()
    partial = False
    for start, block, slot in ((0, q, key_dim), (key_dim, k, key_dim), (key_dim * 2, v, value_dim)):
        partial = _overlay_rows(updated, start, block, slot) or partial
    hybrid[in_proj_key] = updated.contiguous()
    report.copied.append(in_proj_key)
    if partial:
        report.hylo_partial.append(in_proj_key)
    _copy_norm_alias(
        dense,
        hybrid,
        source,
        target,
        "mixer.in_proj.layer_norm_weight",
        (("mixer.in_proj.layer_norm_weight", "input_layernorm.weight"),),
        report,
    )
    if "mixer.out_proj.weight" in target:
        dense_proj = _dense_proj_key(source)
        if dense_proj is not None:
            _copy_overlapping_columns(hybrid, target["mixer.out_proj.weight"], dense[dense_proj], report)


def _copy_mamba_hylo(
    dense: Mapping,
    hybrid: dict,
    dense_index: int,
    source: Mapping[str, str],
    target: Mapping[str, str],
    report: UpcycleReport,
    layout: UpcycleLayout,
) -> bool:
    """Port of HyLo's Mamba ``init_with_kqvo`` slice copy.

    HyLo writes V into ``x``, K into ``B``, and Q into ``C`` of an ``in_proj``
    whose width is ``2 * d_inner + 2 * d_xb + nheads``. Primus Mamba2 checkpoints
    use that width only when ``d_xb == d_inner == n_groups * d_state``. Any other
    width is left untouched.
    """
    if layout.mamba_d_inner is None or layout.mamba_d_xb is None or layout.mamba_nheads is None:
        return False
    d_inner = layout.mamba_d_inner
    d_xb = layout.mamba_d_xb
    expected = 2 * d_inner + 2 * d_xb + layout.mamba_nheads
    in_proj_key = target["mixer.in_proj.weight"]
    in_proj = hybrid[in_proj_key]
    if in_proj.shape[0] != expected:
        return False
    q, k, v = _dense_qkv(dense, dense_index, source, layout)
    updated = in_proj.detach().to(device="cpu").clone()
    partial = _overlay_rows(updated, d_inner, v, d_xb)
    partial = _overlay_rows(updated, d_inner + d_xb, k, d_xb) or partial
    partial = _overlay_rows(updated, d_inner + 2 * d_xb, q, d_inner) or partial
    hybrid[in_proj_key] = updated.contiguous()
    report.copied.append(in_proj_key)
    if partial:
        report.hylo_partial.append(in_proj_key)
    if "mixer.out_proj.weight" in target:
        dense_proj = _dense_proj_key(source)
        if dense_proj is not None and _same_shape(dense[dense_proj], hybrid[target["mixer.out_proj.weight"]]):
            _assign_matching(
                dense, hybrid, dense_proj, target["mixer.out_proj.weight"], report, required=True
            )
    return True


def hylo_reinit_q(
    q_matrix: torch.Tensor,
    q_lora_rank: int,
    num_heads: int,
    head_dim: int,
    qk_nope_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``DeepseekV3Attention.re_init_q`` without the module.

    Returns ``(q_a_proj, q_b_nope)`` with shapes ``(q_lora_rank, hidden)`` and
    ``(num_heads * qk_nope_head_dim, q_lora_rank)``.
    """
    u_q, s_q, v_q = torch.linalg.svd(q_matrix.float(), full_matrices=True)
    q_down = (torch.diag(s_q[:q_lora_rank]) @ v_q[:q_lora_rank, :]).to(dtype=q_matrix.dtype)
    q_up = u_q[:, :q_lora_rank].to(dtype=q_matrix.dtype)
    q_up = q_up.view(num_heads, head_dim, q_lora_rank)[:, :qk_nope_head_dim, :]
    q_up = q_up.reshape(num_heads * qk_nope_head_dim, q_lora_rank).contiguous()
    return q_down.contiguous(), q_up


def hylo_reinit_kv(
    k_matrix: torch.Tensor,
    v_matrix: torch.Tensor,
    num_kv_heads_init: int,
    head_dim: int,
    num_kv_heads: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    v_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``DeepseekV3Attention.re_init_kv`` without the module.

    Returns ``(kv_a, kv_b)``. ``kv_a`` is the compressed rows only; Megatron's
    ``linear_kv_down_proj`` also stores RoPE rows after those.
    """
    if num_kv_heads % num_kv_heads_init != 0:
        raise ValueError(
            f"MLA kv heads ({num_kv_heads}) must be a multiple of dense kv heads ({num_kv_heads_init})"
        )
    group_size = num_kv_heads // num_kv_heads_init
    k_matrix = k_matrix.view(num_kv_heads_init, head_dim, -1).repeat_interleave(group_size, dim=0)
    v_matrix = v_matrix.view(num_kv_heads_init, head_dim, -1).repeat_interleave(group_size, dim=0)
    k_matrix = k_matrix.reshape(num_kv_heads * head_dim, -1)
    v_matrix = v_matrix.reshape(num_kv_heads * head_dim, -1)
    kv_matrix = torch.cat((k_matrix, v_matrix), dim=0)
    u_kv, s_kv, v_kv = torch.linalg.svd(kv_matrix.float(), full_matrices=True)
    kv_down = (torch.diag(s_kv[:kv_lora_rank]) @ v_kv[:kv_lora_rank, :]).to(dtype=k_matrix.dtype)

    k_new = u_kv[:, :kv_lora_rank][: num_kv_heads * head_dim, :]
    k_new = k_new.view(num_kv_heads, head_dim, kv_lora_rank)[:, :qk_nope_head_dim, :]
    v_new = u_kv[:, :kv_lora_rank][num_kv_heads * head_dim :, :]
    v_new = v_new.view(num_kv_heads, head_dim, kv_lora_rank)[:, :v_head_dim, :]
    kv_up = torch.cat((k_new, v_new), dim=1).reshape(num_kv_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank)
    return kv_down.contiguous(), kv_up.to(dtype=k_matrix.dtype).contiguous()


def _dense_qkv(
    dense: Mapping, dense_index: int, source: Mapping[str, str], layout: UpcycleLayout
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if "self_attention.linear_qkv.weight" in source:
        return split_fused_qkv(
            dense[source["self_attention.linear_qkv.weight"]],
            layout.num_attention_heads,
            layout.num_query_groups,
            layout.head_dim,
        )
    try:
        return (
            dense[source["self_attention.linear_q_proj.weight"]],
            dense[source["self_attention.linear_k_proj.weight"]],
            dense[source["self_attention.linear_v_proj.weight"]],
        )
    except KeyError as exc:
        raise ValueError(
            f"dense layer {dense_index} has no fused linear_qkv or separate q/k/v projections"
        ) from exc


def split_fused_qkv(
    fused: torch.Tensor, num_heads: int, num_kv_heads: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split Megatron's grouped ``[Q, K, V]`` ``linear_qkv`` into separate projections."""
    if num_heads % num_kv_heads != 0:
        raise ValueError(f"num_attention_heads ({num_heads}) must divide into query groups ({num_kv_heads})")
    heads_per_group = num_heads // num_kv_heads
    group = (heads_per_group + 2) * head_dim
    if fused.shape[0] != num_kv_heads * group:
        raise ValueError(
            f"linear_qkv rows {fused.shape[0]} != {num_kv_heads} groups * {group} "
            f"(heads={num_heads}, head_dim={head_dim})"
        )
    grouped = fused.view(num_kv_heads, group, fused.shape[1])
    q_width = heads_per_group * head_dim
    q = grouped[:, :q_width, :].reshape(num_heads * head_dim, fused.shape[1])
    k = grouped[:, q_width : q_width + head_dim, :].reshape(num_kv_heads * head_dim, fused.shape[1])
    v = grouped[:, q_width + head_dim :, :].reshape(num_kv_heads * head_dim, fused.shape[1])
    return q.contiguous(), k.contiguous(), v.contiguous()


def _repeat_gqa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Repeat K and V out to ``num_heads``, matching ``_copy_llama_attn_to_gdn``."""
    if num_kv_heads >= num_heads:
        return q, k, v
    if num_heads % num_kv_heads != 0:
        raise ValueError(f"cannot repeat {num_kv_heads} kv heads onto {num_heads} query heads")
    repeats = num_heads // num_kv_heads
    k = k.view(num_kv_heads, head_dim, -1).repeat_interleave(repeats, dim=0)
    v = v.view(num_kv_heads, head_dim, -1).repeat_interleave(repeats, dim=0)
    hidden = q.shape[1]
    return q, k.reshape(num_heads * head_dim, hidden), v.reshape(num_heads * head_dim, hidden)


def _has_mla_down_proj(target: Mapping[str, str]) -> bool:
    return "self_attention.linear_q_down_proj.weight" in target


def _mla_ranks_ready(layout: UpcycleLayout) -> bool:
    return None not in (
        layout.q_lora_rank,
        layout.kv_lora_rank,
        layout.qk_nope_head_dim,
        layout.v_head_dim,
    )


def _mla_kv_heads(kv_up: torch.Tensor, layout: UpcycleLayout) -> int:
    per_head = layout.qk_nope_head_dim + layout.v_head_dim
    if kv_up.shape[0] % per_head != 0:
        raise ValueError(
            f"linear_kv_up_proj rows {kv_up.shape[0]} are not a multiple of "
            f"qk_nope_head_dim + v_head_dim ({per_head})"
        )
    return kv_up.shape[0] // per_head


def _mixer_linear_type(target: Mapping[str, str], layout: UpcycleLayout) -> Optional[str]:
    if layout.linear_type == "kda":
        return "kda"
    if layout.linear_type == "gdn" and None not in (
        layout.gdn_num_key_heads,
        layout.gdn_key_head_dim,
        layout.gdn_num_value_heads,
        layout.gdn_value_head_dim,
    ):
        return "gdn"
    if layout.linear_type == "mamba":
        return "mamba"
    if "mixer.out_proj.weight" in target and layout.gdn_num_key_heads is not None:
        return "gdn"
    return layout.linear_type


def _copy_norm_alias(
    dense: Mapping,
    hybrid: dict,
    source: Mapping[str, str],
    target: Mapping[str, str],
    relative: str,
    pairs: tuple[tuple[str, str], ...],
    report: UpcycleReport,
) -> None:
    if relative not in target:
        return
    source_relative = _alias_lookup(source, relative, pairs)
    if source_relative is None:
        report.attention_left_initialized.append(target[relative])
        return
    _assign_matching(dense, hybrid, source[source_relative], target[relative], report, required=True)


def _write_q_up_nope(
    hybrid: dict, key: str, q_up_nope: torch.Tensor, layout: UpcycleLayout, report: UpcycleReport
) -> None:
    current = hybrid[key].detach().to(device="cpu").clone()
    nope = layout.qk_nope_head_dim
    rope = layout.qk_rope_head_dim or 0
    head_span = nope + rope
    if rope == 0 and tuple(current.shape) == tuple(q_up_nope.shape):
        hybrid[key] = q_up_nope.to(dtype=current.dtype).contiguous()
        report.copied.append(key)
        return
    if current.shape[0] != layout.num_attention_heads * head_span:
        raise ValueError(
            f"{key} rows {current.shape[0]} != {layout.num_attention_heads} * {head_span}"
        )
    viewed = current.view(layout.num_attention_heads, head_span, current.shape[1])
    viewed[:, :nope, :] = q_up_nope.to(dtype=current.dtype).view(layout.num_attention_heads, nope, -1)
    hybrid[key] = viewed.reshape(current.shape).contiguous()
    report.copied.append(key)
    if rope:
        report.hylo_partial.append(key)


def _write_kv_down_prefix(hybrid: dict, key: str, kv_down: torch.Tensor, report: UpcycleReport) -> None:
    current = hybrid[key].detach().to(device="cpu").clone()
    rows = kv_down.shape[0]
    if current.shape[0] < rows or current.shape[1] != kv_down.shape[1]:
        raise ValueError(
            f"{key} is {tuple(current.shape)}, HyLo kv_a is {tuple(kv_down.shape)}"
        )
    current[:rows, :] = kv_down.to(dtype=current.dtype)
    hybrid[key] = current.contiguous()
    report.copied.append(key)
    if current.shape[0] != rows:
        report.hylo_partial.append(key)


def _dense_proj_key(source: Mapping[str, str]) -> Optional[str]:
    for relative in ("self_attention.linear_proj.weight", "self_attention.linear_o_proj.weight"):
        if relative in source:
            return source[relative]
    return None


def _copy_overlapping_columns(
    hybrid: dict, key: str, dense_weight: torch.Tensor, report: UpcycleReport
) -> None:
    current = hybrid[key].detach().to(device="cpu").clone()
    rows = min(current.shape[0], dense_weight.shape[0])
    cols = min(current.shape[1], dense_weight.shape[1])
    current[:rows, :cols] = dense_weight[:rows, :cols].to(dtype=current.dtype)
    hybrid[key] = current.contiguous()
    report.copied.append(key)
    if rows != dense_weight.shape[0] or cols != dense_weight.shape[1] or tuple(current.shape) != tuple(dense_weight.shape):
        report.hylo_partial.append(key)


def _assign_tensor(
    hybrid: dict, key: str, value: torch.Tensor, report: UpcycleReport, *, partial: bool
) -> None:
    current = hybrid[key]
    if tuple(current.shape) != tuple(value.shape):
        raise ValueError(f"{key} is {tuple(current.shape)}, HyLo init produced {tuple(value.shape)}")
    hybrid[key] = value.detach().to(dtype=current.dtype, device="cpu").contiguous().clone()
    report.copied.append(key)
    if partial:
        report.hylo_partial.append(key)


def _overlay_rows(destination: torch.Tensor, start: int, source: torch.Tensor, slot_rows: int) -> bool:
    """Copy ``source`` into a ``slot_rows``-tall window. Return True if either side was clipped."""
    rows = min(source.shape[0], slot_rows, destination.shape[0] - start)
    cols = min(source.shape[1], destination.shape[1])
    if rows <= 0 or cols <= 0:
        raise ValueError(
            f"cannot overlay source {tuple(source.shape)} onto {tuple(destination.shape)} at row {start}"
        )
    destination[start : start + rows, :cols] = source[:rows, :cols].to(dtype=destination.dtype)
    return rows != source.shape[0] or rows != slot_rows or cols != source.shape[1]


def _first_arg(primary: object, secondary: object, name: str):
    value = _arg(primary, name)
    if value is None:
        value = _arg(secondary, name)
    return value


def _optional_int(value) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def _assign_matching(
    dense: Mapping,
    hybrid: dict,
    source_key: str,
    target_key: str,
    report: UpcycleReport,
    *,
    required: bool,
) -> None:
    if source_key not in dense:
        if required:
            raise ValueError(f"dense checkpoint is missing {source_key}")
        return
    if target_key not in hybrid:
        if required:
            raise ValueError(f"hybrid checkpoint is missing {target_key}")
        return
    source = dense[source_key]
    target = hybrid[target_key]
    if not torch.is_tensor(source) or not torch.is_tensor(target):
        raise ValueError(f"{target_key} is not a plain tensor; re-save with ckpt_format: torch and TP=PP=1")
    if not _same_shape(source, target):
        raise ValueError(
            f"shape mismatch for {target_key}: dense {source_key} is {tuple(source.shape)}, "
            f"hybrid is {tuple(target.shape)}. hidden size, FFN size, and padded vocab must match."
        )
    hybrid[target_key] = source.detach().to(dtype=target.dtype, device="cpu").contiguous().clone()
    report.copied.append(target_key)


def _relative_keys(state: Mapping, layer_index: int) -> dict[str, str]:
    prefix = f"decoder.layers.{layer_index}."
    return {key[len(prefix) :]: key for key in state if key.startswith(prefix)}


def _alias_lookup(
    source_relative: Mapping[str, str],
    relative: str,
    pairs: tuple[tuple[str, str], ...],
) -> Optional[str]:
    if relative in source_relative:
        return relative
    for left, right in pairs:
        if relative == left and right in source_relative:
            return right
        if relative == right and left in source_relative:
            return left
    return None


def _is_mlp_key(relative: str) -> bool:
    return relative.startswith("mlp.") or relative.startswith("pre_mlp_layernorm.")


def _skip_key(relative: str) -> bool:
    return relative.endswith("_extra_state")


def _same_shape(left: torch.Tensor, right: torch.Tensor) -> bool:
    return tuple(left.shape) == tuple(right.shape)


def _arg(args: object, name: str):
    if args is None:
        return None
    if isinstance(args, Mapping):
        return args.get(name)
    return getattr(args, name, None)


def _is_distcp_dir(path: Path) -> bool:
    return path.is_dir() and ((path / ".metadata").is_file() or any(path.glob("*.distcp")))


def _require_plain_tensors(state: Mapping, ckpt_file: Path) -> None:
    for key, value in state.items():
        if key.endswith("_extra_state"):
            continue
        if not torch.is_tensor(value):
            raise ValueError(
                f"{ckpt_file} parameter {key} is {type(value).__name__}, not a torch.Tensor. "
                "Re-save with ckpt_format: torch and tensor_model_parallel_size: 1, "
                "pipeline_model_parallel_size: 1."
            )
