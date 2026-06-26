"""Workload resolver — flattens a Primus pretrain YAML + the extends-chain
of its model file into a single ArchitectureRecord that the agent uses.

The workload YAML looks like:

    modules:
      pre_trainer:
        framework: megatron
        model: mixtral_8x22B_v0.1.yaml
        overrides:
          tensor_model_parallel_size: 1
          ...

The model file lives at primus/configs/models/<framework>/<model> and
typically `extends:` a base file. We chain those, then merge the workload
overrides, and project a small dict that is safe to put into an LLM prompt.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Architecture record
# ---------------------------------------------------------------------------


@dataclass
class ArchitectureRecord:
    """Flat view of model architecture + baseline runtime overrides.

    Anything the LLM needs to reason about per-axis legality lives here.
    """

    framework: str = "megatron"
    model_name: str = ""

    # architecture
    num_layers: int = 0
    hidden_size: int = 0
    ffn_hidden_size: int = 0
    num_attention_heads: int = 0
    num_query_groups: int | None = None  # GQA / MQA
    kv_channels: int | None = None
    seq_length: int = 0
    max_position_embeddings: int = 0
    attention_type: str = "standard"  # "standard" | "mla"
    is_moe: bool = False
    num_experts: int = 0
    moe_router_topk: int = 0
    moe_ffn_hidden_size: int | None = None
    moe_shared_expert_intermediate_size: int | None = None
    vocab_size: int | None = None  # padded vocab size if known
    precision: str = "bf16"

    # baseline runtime overrides
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    context_parallel_size: int = 1
    virtual_pipeline_model_parallel_size: int | None = None
    micro_batch_size: int = 1
    global_batch_size: int = 1
    recompute_granularity: str | None = None
    recompute_method: str | None = None
    recompute_num_layers: int = 0
    # ``recompute_layer_ids`` (e.g. DeepSeek V3) is a comma-separated list of
    # specific layer indices that should be recomputed. Captured as a list so
    # the seed planner can probe ``recompute_num_layers ≈ len(layer_ids)``,
    # matching the count the published reference uses.
    recompute_layer_ids: list[int] | None = None
    # Workloads with a non-divisor PP (DeepSeek V3 PP=16 / 61-layers) ship an
    # explicit per-stage layer assignment via ``pipeline_model_parallel_layout``
    # of the form ``Et*1|t*2|...|t*1,L``. Captured here so the agent can re-
    # generate an analogous layout when proposing other (PP, VPP) pairs.
    pipeline_model_parallel_layout: str | None = None
    overlap_grad_reduce: bool = True
    overlap_param_gather: bool = True
    # Tier-A/B knobs from the Primus Projection skill — captured here so the
    # planner can avoid redundant sweeps (e.g. don't toggle FP8 if already on).
    fp8: str | None = None
    cross_entropy_loss_fusion: bool | None = None
    use_torch_fsdp2: bool | None = None
    use_distributed_optimizer: bool | None = None
    use_turbo_deepep: bool | None = None
    sync_free_stage: int | None = None

    # raw blocks for evaluator use
    raw_overrides: dict = field(default_factory=dict)
    raw_model: dict = field(default_factory=dict)
    workload_path: str = ""
    model_path: str = ""

    def as_prompt_dict(self) -> dict:
        d = asdict(self)
        d.pop("raw_overrides", None)
        d.pop("raw_model", None)
        return d


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _vpp_from_layout(layout: str | None, pp: int) -> int | None:
    """Infer VPP from a Primus `pipeline_model_parallel_layout` string.

    The layout encodes per-stage layer assignments separated by ``|``. The
    total stage count must equal ``PP * VPP``. For DeepSeek V3 the workload
    yaml ships e.g. ``Et*1|t*1|t*2|...|t*1,L`` with 32 stages and PP=16,
    implying VPP=2. Returns None when the layout is missing or malformed.
    """
    if not isinstance(layout, str) or not layout or pp <= 0:
        return None
    # Layout stages are top-level pipe-separated entries; trailing ``,L``
    # marks the loss head, attached to the last stage.
    core = layout.rstrip()
    if core.endswith(",L"):
        core = core[:-2]
    n_stages = len([s for s in core.split("|") if s.strip()])
    if n_stages % pp != 0 or n_stages == 0:
        return None
    vpp = n_stages // pp
    return vpp if vpp > 0 else None


def _load_yaml(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text())
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} is not a YAML mapping")
    return raw


def _resolve_model_chain(start: Path) -> dict:
    """Resolve `extends:` chain. Later extends override earlier ones; the
    "leaf" file overrides everything in its bases."""
    chain: list[dict] = []
    visited: set[Path] = set()
    stack: list[Path] = [start]
    while stack:
        cur = stack.pop().resolve()
        if cur in visited:
            continue
        visited.add(cur)
        d = _load_yaml(cur)
        ext = d.get("extends") or []
        if isinstance(ext, str):
            ext = [ext]
        for parent in ext:
            stack.append((cur.parent / parent).resolve())
        chain.append(d)
    # chain order: leaf first, base last → merge from base→leaf so leaf wins
    merged: dict = {}
    for d in reversed(chain):
        merged.update({k: v for k, v in d.items() if k != "extends"})
    return merged


def _detect_attention_type(model: dict) -> str:
    if any(
        k.startswith("q_lora_rank") or k.startswith("kv_lora_rank") or k == "multi_latent_attention"
        for k in model
    ):
        return "mla"
    return "standard"


def resolve_workload(workload_yaml: Path, primus_root: Path | None = None) -> ArchitectureRecord:
    """Resolve a Primus pretrain YAML into an ArchitectureRecord."""
    workload_yaml = workload_yaml.resolve()
    raw = _load_yaml(workload_yaml)

    pre_trainer = ((raw.get("modules") or {}).get("pre_trainer")) or {}
    framework = pre_trainer.get("framework", "megatron")
    model_filename = pre_trainer.get("model")
    if not model_filename:
        raise ValueError(f"{workload_yaml} has no modules.pre_trainer.model")
    if isinstance(model_filename, str):
        model_filename = _expand_primus_templates(model_filename)

    overrides = pre_trainer.get("overrides") or {}

    # Find the model file. Convention: primus/configs/models/<framework>/<file>
    if primus_root is None:
        primus_root = _find_primus_root(workload_yaml)
    model_path = primus_root / "primus" / "configs" / "models" / framework / model_filename
    if not model_path.is_file():
        # fallback: try resolving relative to workload
        candidate = workload_yaml.parent / model_filename
        if candidate.is_file():
            model_path = candidate
        else:
            raise FileNotFoundError(
                f"Could not find model YAML for '{model_filename}' " f"under {model_path} or {candidate}"
            )

    model = _resolve_model_chain(model_path)

    _ne = model.get("num_experts", 0)
    is_moe = "num_experts" in model and _ne is not None and int(_ne) > 0

    # precision: BF16 by default; FP8 if `fp8` block present at workload level
    precision = "bf16"
    if any("fp8" in k.lower() for k in raw.keys()) or any("fp8" in k.lower() for k in overrides.keys()):
        precision = "fp8"

    return ArchitectureRecord(
        framework=framework,
        model_name=model_path.stem,
        num_layers=int(model.get("num_layers", 0)),
        hidden_size=int(model.get("hidden_size", 0)),
        ffn_hidden_size=int(model.get("ffn_hidden_size", 0)),
        num_attention_heads=int(model.get("num_attention_heads", 0)),
        num_query_groups=model.get("num_query_groups"),
        kv_channels=model.get("kv_channels"),
        seq_length=int(_strip_env(overrides.get("seq_length", model.get("seq_length", 0)))),
        max_position_embeddings=int(
            _strip_env(
                overrides.get(
                    "max_position_embeddings",
                    model.get("max_position_embeddings", 0),
                )
            )
        ),
        attention_type=_detect_attention_type(model),
        is_moe=is_moe,
        num_experts=int(model.get("num_experts", 0) or 0),
        moe_router_topk=int(model.get("moe_router_topk", 0) or 0),
        moe_ffn_hidden_size=model.get("moe_ffn_hidden_size"),
        moe_shared_expert_intermediate_size=model.get("moe_shared_expert_intermediate_size"),
        vocab_size=model.get("padded_vocab_size") or model.get("vocab_size"),
        precision=precision,
        tensor_model_parallel_size=int(_strip_env(overrides.get("tensor_model_parallel_size", 1))),
        pipeline_model_parallel_size=int(_strip_env(overrides.get("pipeline_model_parallel_size", 1))),
        expert_model_parallel_size=int(_strip_env(overrides.get("expert_model_parallel_size", 1))),
        context_parallel_size=int(_strip_env(overrides.get("context_parallel_size", 1))),
        # VPP can be set explicitly OR implied by ``pipeline_model_parallel_layout``
        # (number of pipe-pipeline stages / PP). DeepSeek V3 uses the layout
        # form, so when no explicit VPP is set we count the layout stages and
        # divide by PP. Layout stages are pipe-separated entries, e.g.
        # "Et*1|t*2|...|t*1,L" → 32 stages.
        virtual_pipeline_model_parallel_size=(
            _coerce_opt_int(
                overrides.get("num_virtual_stages_per_pipeline_rank")
                or overrides.get("virtual_pipeline_model_parallel_size")
            )
            or _vpp_from_layout(
                overrides.get("pipeline_model_parallel_layout"),
                int(_strip_env(overrides.get("pipeline_model_parallel_size", 1))),
            )
        ),
        micro_batch_size=int(_strip_env(overrides.get("micro_batch_size", 1))),
        global_batch_size=int(_strip_env(overrides.get("global_batch_size", 1))),
        recompute_granularity=overrides.get("recompute_granularity"),
        recompute_method=overrides.get("recompute_method"),
        recompute_num_layers=int(overrides.get("recompute_num_layers", 0) or 0),
        recompute_layer_ids=_parse_layer_ids(overrides.get("recompute_layer_ids")),
        pipeline_model_parallel_layout=overrides.get("pipeline_model_parallel_layout"),
        overlap_grad_reduce=bool(overrides.get("overlap_grad_reduce", True)),
        overlap_param_gather=bool(overrides.get("overlap_param_gather", True)),
        fp8=(overrides.get("fp8") or raw.get("fp8") or model.get("fp8")) or None,
        cross_entropy_loss_fusion=overrides.get("cross_entropy_loss_fusion"),
        use_torch_fsdp2=overrides.get("use_torch_fsdp2"),
        use_distributed_optimizer=overrides.get("use_distributed_optimizer"),
        use_turbo_deepep=overrides.get("use_turbo_deepep"),
        sync_free_stage=overrides.get("sync_free_stage"),
        raw_overrides=overrides,
        raw_model=model,
        workload_path=str(workload_yaml),
        model_path=str(model_path),
    )


def _parse_layer_ids(value: Any) -> list[int] | None:
    """Parse a Primus ``recompute_layer_ids`` field into a list of layer ints.

    Accepts both the published string form (``"0,3,4,7,...,51"`` — what
    DeepSeek V3 ships) and a YAML list. Returns None when the field is
    absent or not parseable, so the agent simply skips the targeted-recompute
    seed for that workload.
    """
    if value is None:
        return None
    if isinstance(value, list):
        try:
            return [int(v) for v in value]
        except (TypeError, ValueError):
            return None
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        try:
            return [int(p) for p in parts]
        except ValueError:
            return None
    return None


def _strip_env(value: Any) -> Any:
    """Some configs use ${PRIMUS_TP:1} style. Strip to the default."""
    if isinstance(value, str) and value.startswith("${") and ":" in value and value.endswith("}"):
        return value.split(":", 1)[1].rstrip("}")
    return value


def _coerce_opt_int(value: Any) -> int | None:
    """Strip ``${VAR:default}`` templating and coerce to int, or None when unset."""
    value = _strip_env(value)
    if value in (None, ""):
        return None
    return int(value)


def _expand_primus_templates(s: str) -> str:
    """Expand ``${PRIMUS_*:default}`` segments (e.g. ``${PRIMUS_MODEL:foo}.yaml``)."""
    return re.sub(r"\$\{[A-Za-z0-9_]+:([^}]*)\}", r"\1", s)


def _find_primus_root(start: Path) -> Path:
    """Find the Primus repository root.

    Resolution order:
      1. ``$PRIMUS_ROOT`` env var (handy when the workload yaml lives outside
         the Primus tree, e.g. patched yamls in a side directory).
      2. Walk up from ``start`` looking for ``primus/configs/models``.
      3. Fall back to CWD (last resort).
    """
    env_root = os.environ.get("PRIMUS_ROOT")
    if env_root:
        p = Path(env_root)
        if (p / "primus" / "configs" / "models").is_dir():
            return p
    cur = start.parent
    for _ in range(10):
        if (cur / "primus" / "configs" / "models").is_dir():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return Path.cwd()
