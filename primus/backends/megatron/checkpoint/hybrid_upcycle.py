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

``*`` keeps the dense attention block when the target module has the same
tensor shapes. ``M`` is a new recurrent mixer: those parameters stay at the
hybrid model's own initialization. Embeddings, the final norm, and every MLP
are copied from the dense checkpoint.

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

    def to_dict(self) -> dict:
        return {
            "pattern": self.pattern,
            "num_dense_layers": self.num_dense_layers,
            "num_copied": len(self.copied),
            "copied": self.copied,
            "attention_left_initialized": self.attention_left_initialized,
            "attention_shape_mismatch": self.attention_shape_mismatch,
            "mixer_layers_left_initialized": self.mixer_layers_left_initialized,
        }


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
) -> tuple[dict[str, torch.Tensor], UpcycleReport]:
    """Overlay transplanted dense weights onto a copy of ``hybrid_state``.

    MLP and embedding transplants that disagree in shape raise. Attention
    parameters that are absent from the dense model, or whose shapes differ
    (GQA into MLA is the usual case), stay at the hybrid initialization.
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
            _copy_attention(dense_state, upcycled, dense_index, mixer_index, report)
        else:
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
            "Embeddings and MLPs are copied. New Mamba/GDN/KDA layers keep the "
            "weights from --hybrid-init-checkpoint."
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
    dense_state, _dense_ckpt = load_checkpoint(args.dense_checkpoint)
    hybrid_state, hybrid_ckpt = load_checkpoint(args.hybrid_init_checkpoint)
    pattern = resolve_pattern(
        args.hybrid_pattern,
        args.attention_ratio,
        hybrid_ckpt.get("args"),
        count_decoder_layers(hybrid_state),
    )
    upcycled, report = upcycle_state_dict(dense_state, hybrid_state, pattern)
    checkpoint = build_upcycle_checkpoint(hybrid_ckpt, upcycled)
    ckpt_path = save_upcycle_checkpoint(checkpoint, args.output_dir, report)
    print(f"pattern: {report.pattern}")
    print(f"copied {len(report.copied)} tensors")
    print(f"attention tensors left at hybrid init: {len(report.attention_left_initialized)}")
    print(f"attention shape mismatches left at hybrid init: {len(report.attention_shape_mismatch)}")
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
    dense: Mapping, hybrid: dict, dense_index: int, hybrid_index: int, report: UpcycleReport
) -> None:
    source = _relative_keys(dense, dense_index)
    target = _relative_keys(hybrid, hybrid_index)
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
