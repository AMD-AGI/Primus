###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Find the ops a model config declares that the projection does not price.

A config key is *consumed* when the projection can act on it: it is a
``ModelConfig`` field, the projection source reads it by name, or a registered
op model claims or consumes it.  Everything else that looks like it defines an
op (attention variants, linear / state-space layers, gates, convolutions,
routers, hybrid layer schedules, ...) and is switched on is reported:

* ``gap``      -- a selector names an op variant (``experimental_attention_variant:
  gated_delta_net``), or a boolean / string switch turns an op on.  The
  projection is pricing this layer as something else; the tuning agent brings
  these up.
* ``suspect``  -- only numeric parameters of an op family are set (e.g.
  ``mamba_state_dim`` inherited from a base file).  Reported, not acted on.

Works on any flat ``{key: value}`` view of a model: the resolved YAML chain
the tuning agent reads, or the args namespace the projection is built from.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, fields
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .base import (
    SLOT_ATTENTION,
    SLOT_ATTENTION_EXTRA,
    SLOT_LAYER_EXTRA,
    SLOT_MLP,
    is_active,
)
from .registry import op_models_enabled, registered_op_models

GAP_MARKER = "PRIMUS_OP_GAP"

# Keys whose *value* names an op variant.  ``modeled`` are the values the
# built-in profilers price; anything else is a gap unless an op model claims
# it.  ``absorbs`` are key prefixes that parameterise the selected op and are
# reported with it instead of as separate gaps.
_SELECTORS: Tuple[Tuple[str, frozenset, str, Tuple[str, ...]], ...] = (
    ("experimental_attention_variant", frozenset({""}), SLOT_ATTENTION, ("linear_",)),
    ("linear_attention_type", frozenset({""}), SLOT_ATTENTION, ("linear_",)),
    ("linear_attention_kind", frozenset({""}), SLOT_ATTENTION, ("linear_",)),
    (
        "attention_type",
        frozenset({"", "standard", "mha", "gqa", "mla", "full", "full_attention"}),
        SLOT_ATTENTION,
        (),
    ),
    # HF configs: one entry per layer.
    (
        "layer_types",
        frozenset({"full_attention", "attention", "sliding_attention"}),
        SLOT_ATTENTION,
        ("linear_",),
    ),
    ("lfm_layer_types", frozenset({"full_attention", "attention"}), SLOT_ATTENTION, ("conv_",)),
    (
        "hybrid_override_pattern",
        frozenset({"*", "-", "e"}),
        SLOT_ATTENTION,
        ("mamba_", "hybrid_", "linear_", "use_short_conv", "is_hybrid_model"),
    ),
)
# Megatron hybrid layer symbols ('*' attention, '-' MLP, 'E' MoE are built in).
_PATTERN_SYMBOLS = {"m": "mamba", "g": "gated_delta_net", "k": "kda"}

# Boolean switches whose True means "this layer is not a plain transformer layer".
_SWITCH_OPS: Dict[str, Tuple[str, str, Tuple[str, ...]]] = {
    "is_hybrid_model": (
        "hybrid_linear_layers",
        SLOT_ATTENTION,
        ("mamba_", "hybrid_", "linear_", "use_short_conv"),
    ),
}

_OP_WORDS = re.compile(
    r"(attention|attn|linear_|mamba|ssm|conv|gate|gating|gated|delta|recurren|state_dim|"
    r"expert|router|lora|latent|window|sparse|index|compress|hyper|hc_|mtp|engram|"
    r"retention|rwkv|hyena|kda|gdn|mla|nsa|dsa|layer_types|hybrid|variant|residual|"
    r"memory_layer|kv_|qk_|chunk)"
)
_NOT_OPS = re.compile(
    r"(dropout|init|fusion|fused|_fp32|in_fp32|softmax_scale|eps|epsilon|recompute|dtype|"
    r"tokenizer|load_balancing|aux_loss|coeff|bias|dispatcher|permute|z_loss|token_drop|"
    r"capacity|pad|layernorm_1p|apply_wd|scaling_factor|loss|backend|score_function|"
    r"enable_expert_bias|router_topk_scaling|grouped_gemm|legacy|overlap|deepep|turbo|"
    r"checkpoint|offload|log|print|profile|save|load|ckpt|seed|_path|_dir|file|interval|"
    r"warmup|lr|weight_decay|clip|noise|freeze|requires_grad|frozen|pre_process|post_process|"
    r"mscale|rotary|rope|yarn|beta_fast|beta_slow|original_max|extended|"
    r"use_mem_eff|disable_mamba_mem_eff|cuda_graph|te_|transformer_impl|spec|expert_tensor|"
    r"router_dtype|expert_model_parallel|expert_parallel|num_experts_per|moe_layer_recompute|"
    r"shared_expert_overlap|router_pre_softmax|router_group_limit|router_num_groups|"
    r"router_enable_expert_bias|router_bias_update|router_fusion|router_force|_tp_comm|"
    r"_sequence_parallel|mtp_loss|token_dispatcher|use_flash_attn|kv_channels|qk_layernorm|"
    r"clip_qkv|qk_clip|gated_linear_unit|mla_use_nope|apply_query_key_layer_scaling|"
    r"attention_softmax|calculate_per_token_loss|window_size_for|attn_mask|attention_mask)"
)


# Keys that look like an op but are priced through another field, or are too
# small to matter. Each entry says where the cost lives, so the decision is
# reviewable. The bring-up stage appends to ``generated/acknowledged.yaml``
# when it concludes an op is negligible instead of writing a model for it.
_ACKNOWLEDGED: Dict[str, str] = {
    "hybrid_attention_enabled": "DeepSeek-V4 attention, priced from compress_ratios in AttentionProfiler",
    "hc_use_sinkhorn": "mHC Sinkhorn, priced in the V4 attention memory-bound block (PRIMUS_HC_SINKHORN_ITERS)",
    "hc_sinkhorn_iters": "mHC Sinkhorn iterations, read from PRIMUS_HC_SINKHORN_ITERS",
    "hc_post_mult": "mHC output scale: elementwise, inside the mHC expand pass",
    "attn_sink": "one learned logit per head added to the softmax: no extra pass",
    "v4_grouped_experts_support_clamped_swiglu": "kernel capability flag; clamped SwiGLU costs as SwiGLU",
    "moe_router_group_topk": "group-limited routing constrains which experts are picked, not how many",
    "moe_router_num_groups": "group-limited routing constrains which experts are picked, not how many",
}

# Filler words dropped before grouping keys into an op family.
_FILLER = {"use", "enable", "apply", "is", "num", "with", "add", "has", "do"}
# First tokens too generic to name a family on their own.
_GENERIC = {"moe", "attn", "attention", "mla", "kv", "qk", "router", "v4", "model", "layer", "head"}


@dataclass
class OpGap:
    op_id: str
    severity: str  # "gap" | "suspect"
    slot: str
    reason: str
    keys: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)

    def marker_line(self) -> str:
        return f"{GAP_MARKER} " + json.dumps(self.as_dict(), default=str, sort_keys=True)


@dataclass
class CoverageReport:
    gaps: List[OpGap] = field(default_factory=list)
    covered_by: Dict[str, str] = field(default_factory=dict)  # op model name -> what it covers

    @property
    def blocking(self) -> List[OpGap]:
        return [g for g in self.gaps if g.severity == "gap"]

    def summary_lines(self) -> List[str]:
        lines = []
        for name, what in sorted(self.covered_by.items()):
            lines.append(f"modeled by op model '{name}': {what}")
        for g in self.gaps:
            keys = ", ".join(f"{k}={_short(v)}" for k, v in g.keys.items())
            lines.append(f"{g.severity.upper()} {g.op_id} [{g.slot}]: {g.reason} ({keys})")
        return lines

    def as_dict(self) -> dict:
        return {"gaps": [g.as_dict() for g in self.gaps], "covered_by": dict(self.covered_by)}


def _short(v: Any, n: int = 40) -> str:
    s = json.dumps(v, default=str) if not isinstance(v, str) else v
    return s if len(s) <= n else s[: n - 3] + "..."


@lru_cache(maxsize=1)
def _projection_tokens() -> frozenset:
    """Identifiers the projection source mentions (attribute reads and string keys)."""
    root = Path(__file__).resolve().parents[1]
    toks: set = set()
    word = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
    for path in root.rglob("*.py"):
        if "op_models" in path.parts:
            continue
        try:
            toks.update(word.findall(path.read_text(errors="ignore")))
        except OSError:
            continue
    return frozenset(toks)


@lru_cache(maxsize=1)
def _model_config_fields() -> frozenset:
    from primus.core.projection.training_config import (
        ModelConfig,
        ModelParallelConfig,
        RuntimeConfig,
    )

    return frozenset(f.name for cls in (ModelConfig, ModelParallelConfig, RuntimeConfig) for f in fields(cls))


def _op_shaped(key: str) -> bool:
    return bool(_OP_WORDS.search(key)) and not _NOT_OPS.search(key)


def _values(v: Any) -> List[str]:
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                v = json.loads(s.replace("'", '"'))
            except ValueError:
                return [t.strip(" '\"") for t in s.strip("[]").split(",") if t.strip()]
        else:
            return [s]
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    return [str(v)]


def _family(key: str) -> str:
    parts = [p for p in key.lower().split("_") if p and p not in _FILLER] or [key]
    if parts[0] in _GENERIC and len(parts) > 1:
        return "_".join(parts[:2])
    return parts[0]


def _switches_on(key: str, val: Any) -> bool:
    """A value that turns an op on rather than sizing one."""
    if val is True:
        return True
    if isinstance(val, str) and not _numeric(val):
        return True
    if isinstance(val, (list, tuple)) and val:
        return True
    # A layer count: ``mtp_num_layers: 1`` adds a layer, it does not size one.
    if isinstance(val, int) and not isinstance(val, bool) and val > 0:
        return key.endswith("_layers") or key.endswith("_num_layers")
    return False


def _load_chain(path: Path, seen=None) -> Dict[str, Any]:
    import yaml

    seen = seen or set()
    path = path.resolve()
    if path in seen or not path.is_file():
        return {}
    seen.add(path)
    doc = yaml.safe_load(path.read_text()) or {}
    merged: Dict[str, Any] = {}
    parents = doc.get("extends") or doc.get("bases") or []
    for parent in [parents] if isinstance(parents, str) else parents:
        merged.update(_load_chain(path.parent / parent, seen))
    merged.update({k: v for k, v in doc.items() if k not in ("extends", "bases")})
    return merged


@lru_cache(maxsize=4)
def default_baseline(framework: str = "megatron") -> Dict[str, Any]:
    """Values every model of *framework* inherits: base model YAML + trainer module."""
    root = Path(__file__).resolve().parents[3] / "configs"
    out: Dict[str, Any] = {}
    out.update(_load_chain(root / "modules" / framework / "pre_trainer.yaml"))
    out.update(_load_chain(root / "models" / framework / "language_model.yaml"))
    return out


def _acknowledged() -> Dict[str, str]:
    out = dict(_ACKNOWLEDGED)
    path = Path(__file__).resolve().parent / "generated" / "acknowledged.yaml"
    if path.is_file():
        try:
            import yaml

            for k, v in (yaml.safe_load(path.read_text()) or {}).items():
                out[str(k)] = str(v)
        except Exception:
            pass
    return out


def detect_op_gaps(
    cfg: Dict[str, Any],
    *,
    extra_consumed: Iterable[str] = (),
    baseline: Optional[Dict[str, Any]] = None,
    framework: str = "megatron",
) -> CoverageReport:
    """Coverage of a flat model config against the projection and the registry.

    Keys whose value equals *baseline* (default: the framework's base model and
    trainer YAMLs) are inherited defaults, not something this model turns on.
    """
    if baseline is None:
        try:
            baseline = default_baseline(framework)
        except Exception:
            baseline = {}
    flat = {
        k: v
        for k, v in (cfg or {}).items()
        if isinstance(k, str) and not (k in baseline and baseline[k] == v)
    }
    report = CoverageReport()
    ops = registered_op_models() if op_models_enabled() else []

    class _View:
        def __init__(self, d):
            self.extra_args = d

        def __getattr__(self, item):
            return None

    view = _View(flat)
    claimed: Dict[str, set] = {}
    consumed = (
        set(_model_config_fields()) | set(_projection_tokens()) | set(extra_consumed) | set(_acknowledged())
    )
    for cls in ops:
        try:
            present = cls.matches(view)
        except Exception:
            present = False
        for k, vals in cls.claims.items():
            claimed.setdefault(k, set()).update(str(v).lower() for v in vals)
        if present:
            consumed.update(cls.consumes)
            consumed.update(cls.claims)
            consumed.update(getattr(cls, "ignores", {}))
            report.covered_by[cls.name] = ", ".join(dict.fromkeys((*cls.claims, *cls.consumes)))

    def _claimed(key: str, val: str) -> bool:
        vals = claimed.get(key, set())
        return "*" in vals or val.lower() in vals

    absorbed: set = set()
    for key, modeled, slot, absorbs in _SELECTORS:
        if key not in flat or not is_active(flat[key]):
            continue
        if key == "hybrid_override_pattern":
            vals = {
                _PATTERN_SYMBOLS.get(c.lower(), f"hybrid_{c.lower()}")
                for c in str(flat[key])
                if c.lower() not in modeled and not c.isspace() and c != "|"
            }
        else:
            vals = set(_values(flat[key]))
        unknown = sorted({v for v in vals if v.lower() not in modeled and not _claimed(key, v)})
        unknown = [v for v in unknown if v.lower() not in ("none", "null")]
        if not unknown:
            absorbed.add(key)
            continue
        params = {
            k: v
            for k, v in flat.items()
            if k != key and any(k.startswith(p) for p in absorbs) and is_active(v) and k not in consumed
        }
        absorbed.update(params)
        absorbed.add(key)
        for v in unknown:
            op_id = re.sub(r"[^a-z0-9]+", "_", v.lower()).strip("_") or key
            if len(op_id) < 6:
                op_id = f"{key.split('_')[0]}_{op_id}"
            report.gaps.append(
                OpGap(
                    op_id=op_id,
                    severity="gap",
                    slot=slot,
                    reason=f"{key} selects '{v}', which no built-in profiler or op model prices",
                    keys={key: flat[key], **params},
                )
            )

    for key, (op_id, slot, absorbs) in _SWITCH_OPS.items():
        if flat.get(key) is not True or key in consumed or key in absorbed or _claimed(key, "true"):
            continue
        if any(
            g.slot == slot and set(g.keys) & {k for k in flat if any(k.startswith(p) for p in absorbs)}
            for g in report.gaps
        ):
            absorbed.add(key)
            continue
        params = {
            k: v
            for k, v in flat.items()
            if any(k.startswith(p) for p in absorbs) and is_active(v) and k not in consumed
        }
        absorbed.update(params)
        absorbed.add(key)
        report.gaps.append(
            OpGap(
                op_id,
                "gap",
                slot,
                f"{key}=true but the projection prices every layer as attention + MLP",
                {key: True, **params},
            )
        )

    families: Dict[str, Dict[str, Any]] = {}
    for key, val in flat.items():
        if key in absorbed or key in consumed or not is_active(val) or not _op_shaped(key):
            continue
        families.setdefault(_family(key), {})[key] = val

    # One slot of a layer holds one new op, so families landing in the same
    # slot are its switch and its parameters (Kimi-K3's ``kda_*`` and
    # ``linear_*``): merge them, into a selector's gap when there is one.
    by_slot: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for fam, keys in sorted(families.items()):
        by_slot.setdefault(_guess_slot(" ".join(keys)), []).append((fam, keys))
    for slot, fams in by_slot.items():
        merged = {k: v for _, keys in fams for k, v in keys.items()}
        owner = next((g for g in report.gaps if g.slot == slot), None)
        if owner is not None:
            owner.keys.update(merged)
            continue
        switched = [(fam, keys) for fam, keys in fams if any(_switches_on(k, v) for k, v in keys.items())]
        lead = max(switched or fams, key=lambda fk: len(fk[1]))[0]
        report.gaps.append(
            OpGap(
                op_id=lead,
                severity="gap" if switched else "suspect",
                slot=slot,
                reason=(
                    "op switch is set but nothing in the projection reads it"
                    if switched
                    else "op parameters are set but nothing in the projection reads them"
                ),
                keys=merged,
            )
        )
    return report


def _numeric(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _guess_slot(name: str) -> str:
    if re.search(r"(expert|moe|router|mlp|ffn|activation)", name):
        return SLOT_MLP
    if re.search(r"(attn_res|_residual|residual_)", name):
        return SLOT_LAYER_EXTRA
    if "gate" in name and re.search(r"(attention|attn|mla|output_gate)", name):
        return SLOT_ATTENTION_EXTRA
    if re.search(
        r"(attention|attn|linear|mamba|ssm|conv|delta|kda|gdn|window|sparse|index|compress|mla)", name
    ):
        return SLOT_ATTENTION
    return SLOT_LAYER_EXTRA


def config_view_from_args(args) -> Dict[str, Any]:
    """Flat ``{key: value}`` from an args namespace (or a dict)."""
    raw = args if isinstance(args, dict) else vars(args)
    return {k: v for k, v in raw.items() if not k.startswith("_")}


_REPORTED: set = set()


def report_op_coverage(
    cfg: Dict[str, Any], *, prefix: str = "[Primus:OpCoverage]"
) -> Optional[CoverageReport]:
    """Print the coverage once per process; gaps also as machine-readable markers."""
    import os

    if int(os.getenv("RANK", "0")) != 0 or os.getenv("PRIMUS_OP_COVERAGE", "1") == "0":
        return None
    report = detect_op_gaps(cfg)
    key = json.dumps(report.as_dict(), sort_keys=True, default=str)
    if key in _REPORTED:
        return report
    _REPORTED.add(key)
    for line in report.summary_lines():
        print(f"{prefix} {line}")
    for g in report.blocking:
        print(g.marker_line())
    if report.blocking:
        print(
            f"{prefix} {len(report.blocking)} op(s) above are priced by a built-in fallback, not their own "
            "model. Bring them up with: python -m primus.agents.tuning_agent.op_bringup --workload <yaml>"
        )
    return report


def parse_gap_markers(text: str) -> List[OpGap]:
    """Recover :class:`OpGap` records from projection stdout."""
    out = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line.startswith(GAP_MARKER + " "):
            continue
        try:
            d = json.loads(line[len(GAP_MARKER) + 1 :])
            out.append(OpGap(**d))
        except (ValueError, TypeError):
            continue
    return out
