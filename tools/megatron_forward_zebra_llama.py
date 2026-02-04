"""
Run a single forward pass with a Megatron (mcore) Zebra-Llama checkpoint.

This is intended for *numerical parity* checks against the HF implementation in
`tools/modeling_zebra_llama.py`:
  - same tokenizer ids
  - same logits for a fixed prompt (within dtype tolerance)

Usage (1 GPU):
  cd /vfs/silo/mingyyan/home_backup/Primus
  export PYTHONPATH="$(pwd):$(pwd)/third_party/Megatron-LM:${PYTHONPATH}"

  torchrun --nproc_per_node=1 tools/megatron_forward_zebra_llama.py \
    --load output/zebra_llama_1B-pretrain/iter_0150000 \
    --prompt "The capital of France is" \
    --topk 10

Notes:
  - For TP/PP > 1, you must launch with the matching world size and pass the
    parallelism args; this script focuses on the common TP=1, PP=1 debug case.
"""

from __future__ import annotations

import os
import sys
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import Any

import torch


def _setup_sys_path() -> None:
    """Make sure Primus + Megatron are importable when run from anywhere."""
    primus_root = Path(__file__).resolve().parent.parent
    megatron_root = primus_root / "third_party" / "Megatron-LM"
    tools_root = primus_root / "tools"

    for p in (str(primus_root), str(megatron_root), str(tools_root)):
        if p not in sys.path:
            sys.path.insert(0, p)


def _ensure_primus_logger(rank: int, world_size: int) -> None:
    """
    Primus helpers (e.g., tokenizer builder) log via `primus.core.utils.logger`.
    In full Primus training runs, the logger is initialized by the launcher/BaseModule.
    This standalone script must initialize it explicitly.
    """
    from primus.core.utils import logger as primus_logger
    from primus.modules.module_utils import set_logging_rank

    # Make log_rank_0 / log_rank_last behave correctly before torch.distributed init.
    set_logging_rank(rank, world_size)

    # Avoid double-init (setup_logger is call_once, but also keep it cheap).
    if getattr(primus_logger, "_logger", None) is not None:
        return

    exp_root = os.environ.get("PRIMUS_EXP_ROOT", "/tmp/primus_megatron_forward")
    work_group = os.environ.get("PRIMUS_TEAM", "local")
    user_name = os.environ.get("PRIMUS_USER", os.environ.get("USER", "user"))
    exp_name = os.environ.get("PRIMUS_EXP_NAME", "megatron_forward")

    cfg = primus_logger.LoggerConfig(
        exp_root_path=exp_root,
        work_group=work_group,
        user_name=user_name,
        exp_name=exp_name,
        module_name="megatron_forward",
        file_sink_level="INFO",
        stderr_sink_level="INFO",
        node_ip=os.environ.get("MASTER_ADDR", "localhost"),
        rank=rank,
        world_size=world_size,
    )
    primus_logger.setup_logger(cfg, is_head=False)


def _ensure_rocm_validate_args_compat(args) -> None:
    """
    `primus.modules.trainer.megatron.utils.validate_args_on_rocm()` assumes a Primus
    YAML-backed args object that contains some Primus-specific flags.
    When running this standalone script, those attributes may not exist.
    Set safe defaults so validation can run without crashing.
    """

    def _setdefault(name: str, value) -> None:
        if not hasattr(args, name):
            setattr(args, name, value)

    # Determinism / MoE flags
    _setdefault("deterministic_mode", False)
    _setdefault("moe_grouped_gemm", False)

    # FP8 / turbo linear flags
    _setdefault("fp8", False)
    _setdefault("use_turbo_parallel_linear", False)
    _setdefault("fp8_recipe", "tensorwise")

    # Pipeline debug
    _setdefault("dump_pp_data", False)

    # PrimusTurbo / MoE extras (keep disabled)
    _setdefault("turbo_sync_free_moe_stage", 0)
    _setdefault("enable_primus_turbo", False)
    _setdefault("moe_use_legacy_grouped_gemm", False)
    _setdefault("use_turbo_deepep", False)
    _setdefault("moe_shared_expert_overlap", False)
    _setdefault("moe_router_dtype", "fp32")
    _setdefault("expert_model_parallel_size", 1)
    _setdefault("turbo_deepep_num_cu", 0)


def _apply_zebra_defaults_if_needed(args) -> None:
    """
    Megatron's argparse provides many non-None defaults (e.g. hybrid_attention_ratio=0.0),
    which means `validate_args(args, args_defaults)` will not override them.
    In Primus training, these values come from YAML and are set correctly.

    For this standalone script, if the user is using the Zebra hybrid spec and has not
    set key Zebra knobs explicitly, apply the Zebra defaults so model construction matches
    the training config.
    """

    zebra_spec = [
        "primus.backends.megatron.core.models.hybrid.hybrid_mamba_mla_layer_specs",
        "hybrid_stack_spec",
    ]
    if not hasattr(args, "spec") or args.spec != zebra_spec:
        return

    # ------------------------------------------------------------------
    # Zebra-Llama 1B config parity knobs (from primus/configs/models/megatron/zebra_llama_1B.yaml)
    # Ensure these are set before model construction.
    # ------------------------------------------------------------------
    # Model size
    args.num_layers = 32
    args.hidden_size = 2048
    args.ffn_hidden_size = 8192
    args.num_attention_heads = 32

    # Hybrid / Mamba
    args.is_hybrid_model = True
    args.hybrid_attention_ratio = 0.25
    args.mamba_state_dim = 64
    args.mamba_head_dim = 64
    args.mamba_num_groups = 8

    # MLA
    args.group_query_attention = False
    args.swiglu = True
    args.num_query_groups = None
    args.multi_latent_attention = True
    args.q_lora_rank = 1344
    args.kv_lora_rank = 128
    args.qk_head_dim = 32
    args.qk_pos_emb_head_dim = 32
    args.v_head_dim = 64

    # RoPE / position settings
    args.normalization = "RMSNorm"
    args.rotary_base = 500000
    args.rotary_scaling_factor = 1.0
    args.mscale = 1.0
    args.mscale_all_dim = 1.0
    args.position_embedding_type = "none"
    args.add_position_embedding = True
    args.use_rotary_position_embeddings = False
    args.original_max_position_embeddings = 2048

    args.add_bias_linear = False
    args.mamba_hidden_act = "silu"

    # Extra Mamba knobs (not in zebra_llama_1B.yaml but required by spec)
    if getattr(args, "mamba_expand", None) is None:
        args.mamba_expand = 1
    if getattr(args, "mamba_d_conv", None) is None:
        args.mamba_d_conv = 4


def _normalize_load_path(args) -> None:
    """
    Megatron expects `--load` to point to the *root* checkpoint directory that contains
    `latest_checkpointed_iteration.txt`. Users often pass an `iter_XXXXXXX/` subdir.
    If so, rewrite:
      --load <root>/iter_XXXXXXX  ->  --load <root>  and set --ckpt-step XXXX
    """
    load_dir = getattr(args, "load", None)
    if not load_dir:
        return
    p = Path(str(load_dir)).expanduser()
    name = p.name
    if name.startswith("iter_"):
        try:
            step = int(name[len("iter_") :])
        except ValueError:
            return
        # Only override if user didn't already request a specific step.
        if getattr(args, "ckpt_step", None) in (None, 0):
            setattr(args, "ckpt_step", step)
        setattr(args, "load", str(p.parent))


def _add_script_args(parser: ArgumentParser) -> ArgumentParser:
    group = parser.add_argument_group(title="zebra forward debug")
    group.add_argument("--prompt", type=str, default="The capital of France is")
    group.add_argument("--topk", type=int, default=10)
    group.add_argument("--max-prompt-tokens", type=int, default=256)
    group.add_argument("--save-logits", type=str, default=None, help="Optional .pt path to save logits tensor.")
    group.add_argument(
        "--hf-dir",
        type=str,
        default=None,
        help="If set, also run the HuggingFace Zebra-Llama forward pass from this converted HF checkpoint dir "
        "(must contain config.json + pytorch_model.bin) and compare logits.",
    )
    group.add_argument(
        "--hf-dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="dtype for HF model weights/compute during comparison.",
    )
    group.add_argument(
        "--compare-atol",
        type=float,
        default=1e-2,
        help="Abs tolerance used only for reporting (not an assert).",
    )
    group.add_argument(
        "--compare-layerwise",
        action="store_true",
        default=False,
        help="If set with --hf-dir, compare intermediate hidden states layer-by-layer.",
    )
    group.add_argument(
        "--compare-layer-isolated",
        action="store_true",
        default=False,
        help="If set with --hf-dir, feed each layer the SAME hidden-state tensor (taken from Megatron's layer input) "
        "and compare that layer's output. This isolates per-layer numeric drift from upstream accumulation.",
    )
    group.add_argument(
        "--layerwise-token",
        type=str,
        default="last",
        choices=["last", "mean"],
        help="Which token representation to compare per layer: last token vector or mean over sequence.",
    )
    group.add_argument(
        "--runtime-gather-output",
        action="store_true",
        default=True,
        help="Gather full-vocab logits at runtime (useful for TP>1). Default: true.",
    )
    group.add_argument(
        "--position-ids-mode",
        type=str,
        default="normal",
        choices=["normal", "zeros"],
        help="Test RoPE/positional-embedding impact by controlling position_ids. "
        "'zeros' forces all tokens to use position 0 (often makes rotary a no-op).",
    )
    group.add_argument(
        "--torch-profiler",
        type=str,
        default="off",
        choices=["off", "megatron", "hf", "both"],
        help="Enable torch.profiler tracing for selected forward pass(es). Exports Chrome trace JSON.",
    )
    group.add_argument(
        "--torch-profiler-dir",
        type=str,
        default=None,
        help="Directory to write torch.profiler traces to. Default: ./profiler_traces",
    )
    group.add_argument(
        "--torch-profiler-all-ranks",
        action="store_true",
        default=False,
        help="If set, write one trace per rank. Default: only rank0 writes traces.",
    )
    group.add_argument(
        "--torch-profiler-record-shapes",
        action="store_true",
        default=False,
        help="Record operator input shapes (larger traces).",
    )
    group.add_argument(
        "--torch-profiler-memory",
        action="store_true",
        default=False,
        help="Record memory usage (larger traces).",
    )
    group.add_argument(
        "--torch-profiler-with-stack",
        action="store_true",
        default=False,
        help="Record Python stacks (much larger/slower).",
    )
    return parser


def _profile_forward(
    *,
    enabled: bool,
    name: str,
    out_path: Path,
    record_shapes: bool,
    profile_memory: bool,
    with_stack: bool,
    fn,
):
    """Run `fn()` under torch.profiler and export a Chrome trace JSON."""
    if not enabled:
        return fn()

    from torch.profiler import ProfilerActivity, profile, record_function

    activities = [ProfilerActivity.CPU]
    # On ROCm, torch.cuda APIs still work and profiler uses CUDA activity to mean GPU.
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    with profile(
        activities=activities,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
    ) as prof:
        with record_function(name):
            out = fn()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(out_path))
    return out


@torch.inference_mode()
def main() -> None:
    _setup_sys_path()

    # Megatron imports (after sys.path is set)
    import megatron
    from megatron.training import get_args, get_model, get_tokenizer, print_rank_0
    from megatron.training.arguments import parse_args, validate_args
    from megatron.training.checkpointing import checkpoint_exists, load_checkpoint
    from megatron.training.global_vars import set_global_variables
    from megatron.training.initialize import (
        _init_autoresume,
        _initialize_distributed,
        _set_random_seed,
        setup_logging,
    )
    from megatron.training.utils import get_ltor_masks_and_position_ids

    # Primus adds a thin wrapper around Megatron tokenizer building (used in training).
    from primus.backends.megatron.training.tokenizer.tokenizer import build_tokenizer
    from primus.backends.megatron.training.global_vars import set_primus_global_variables
    from primus.modules.trainer.megatron.utils import set_wandb_writer_patch, validate_args_on_rocm

    # Builders for MCore Mamba/hybrid models
    from model_provider import model_provider
    from mamba_builders import mamba_builder

    # ---------------------------------------------------------------------
    # Primus-style initialization (matches MegatronTrainer.initialize_megatron)
    # ---------------------------------------------------------------------
    args_defaults = {
        # inference-ish defaults
        "no_load_rng": True,
        "no_load_optim": True,
        "micro_batch_size": 1,
        "global_batch_size": 1,
        "exit_on_missing_checkpoint": True,
        # Primus training default: skip compile_dependencies (avoids CUDA-only nvcc fused kernels)
        "disable_compile_dependencies": True,
        # zebra_llama_1B defaults (can be overridden from CLI)
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "num_layers": 32,
        "hidden_size": 2048,
        "ffn_hidden_size": 8192,
        "num_attention_heads": 32,
        "seq_length": 2048,
        "max_position_embeddings": 2048,
        "position_embedding_type": "none",
        "normalization": "RMSNorm",
        "tokenizer_type": "HuggingFaceTokenizer",
        "tokenizer_model": "meta-llama/Llama-3.2-1B",
        # Hybrid + MLA
        "is_hybrid_model": True,
        "hybrid_attention_ratio": 0.25,
        "multi_latent_attention": True,
        "q_lora_rank": 1344,
        "kv_lora_rank": 128,
        "qk_head_dim": 32,
        "qk_pos_emb_head_dim": 32,
        "v_head_dim": 64,
        # Mamba
        "mamba_state_dim": 64,
        "mamba_head_dim": 64,
        "mamba_num_groups": 8,
        "mamba_expand": 1,
        "mamba_d_conv": 4,
        "mamba_hidden_act": "silu",
        # Spec that defines the hybrid block layout
        "spec": [
            "primus.backends.megatron.core.models.hybrid.hybrid_mamba_mla_layer_specs",
            "hybrid_stack_spec",
        ],
    }

    args = parse_args(extra_args_provider=_add_script_args, ignore_unknown_args=False)
    _normalize_load_path(args)
    validate_args(args, args_defaults)
    _apply_zebra_defaults_if_needed(args)

    # Primus utilities used below require logger to be initialized.
    _ensure_primus_logger(rank=int(args.rank), world_size=int(args.world_size))

    # Monkey-patch wandb writer hook (Primus does this before set_global_variables).
    megatron.training.global_vars._set_wandb_writer = set_wandb_writer_patch

    # Global vars (args/timers/etc), but build tokenizer ourselves (Primus wrapper).
    set_global_variables(args, build_tokenizer=False)
    set_primus_global_variables(args)
    args = get_args()

    # Build and register tokenizer the same way Primus does.
    import megatron.training.global_vars as global_vars

    global_vars._ensure_var_is_not_initialized(global_vars._GLOBAL_TOKENIZER, "tokenizer")
    global_vars._GLOBAL_TOKENIZER = build_tokenizer(args)

    setup_logging()

    # Distributed init + seeding (Primus finish_mpu_init path, without compile_dependencies).
    _initialize_distributed(None, None, None)
    _set_random_seed(
        args.seed,
        args.data_parallel_random_init,
        args.te_rng_tracker,
        args.inference_rng_tracker,
        use_cudagraphable_rng=bool(getattr(args, "enable_cuda_graph", False))
        or bool(getattr(args, "external_cuda_graph", False)),
    )
    _init_autoresume()

    # Mirror Primus trainer extra ROCm validations.
    _ensure_rocm_validate_args_compat(args)
    validate_args_on_rocm(args)

    # Build model and load checkpoint
    model_list = get_model(partial(model_provider, mamba_builder), wrap_with_ddp=False)

    # Prove whether we actually load weights (common confusion is passing --load=.../iter_XXXXXXX).
    def _fingerprint_first_param(m) -> tuple[float, float]:
        p0 = next(m.parameters())
        x = p0.detach().float()
        return float(x.mean().item()), float(x.std().item())

    if torch.distributed.get_rank() == 0:
        print_rank_0(f"Checkpoint --load: {getattr(args, 'load', None)}")
        if getattr(args, "load", None) is not None:
            print_rank_0(f"Checkpoint tracker exists? {checkpoint_exists(getattr(args, 'load'))}")
            tracker = Path(str(getattr(args, "load"))) / "latest_checkpointed_iteration.txt"
            if tracker.exists():
                print_rank_0(f"latest_checkpointed_iteration.txt: {tracker.read_text().strip()!r}")
            if getattr(args, "ckpt_step", None):
                print_rank_0(f"--ckpt-step: {getattr(args, 'ckpt_step')}")

    fp_before = _fingerprint_first_param(model_list[0])
    it, _ = load_checkpoint(model_list, None, None, strict=False)
    fp_after = _fingerprint_first_param(model_list[0])
    if torch.distributed.get_rank() == 0:
        print_rank_0(f"load_checkpoint() iteration={it}")
        print_rank_0(f"param_fingerprint mean/std before: {fp_before[0]:.6g}/{fp_before[1]:.6g}")
        print_rank_0(f"param_fingerprint mean/std after:  {fp_after[0]:.6g}/{fp_after[1]:.6g}")
    model = model_list[0]
    model.eval()

    # Build tokenizer
    if args.legacy_tokenizer:
        tokenizer = get_tokenizer()
    else:
        tokenizer = build_tokenizer(args)

    prompt: str = getattr(args, "prompt")
    topk: int = int(getattr(args, "topk"))
    max_prompt_tokens: int = int(getattr(args, "max_prompt_tokens"))

    # Tokenize (truncate for safety)
    token_ids = tokenizer.tokenize(prompt)
    if len(token_ids) > max_prompt_tokens:
        token_ids = token_ids[:max_prompt_tokens]

    input_ids = torch.tensor([token_ids], device=torch.cuda.current_device(), dtype=torch.long)

    # Create standard causal attention mask + position_ids (no resets)
    eos = getattr(tokenizer, "eos_id", None)
    pad = getattr(tokenizer, "pad_id", None)
    if eos is None:
        eos = 0
    if pad is None:
        pad = eos

    attention_mask, _loss_mask, position_ids = get_ltor_masks_and_position_ids(
        data=input_ids,
        eod_token=eos,
        pad_token=pad,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        pad_mask_loss=False,
    )

    # Optional: control position_ids to isolate rotary/positional effects.
    if str(getattr(args, "position_ids_mode", "normal")) == "zeros":
        position_ids = torch.zeros_like(position_ids)

    # Forward (logits)
    prof_mode = str(getattr(args, "torch_profiler", "off"))
    prof_dir = Path(str(getattr(args, "torch_profiler_dir", None) or "profiler_traces")).expanduser().resolve()
    rank = int(torch.distributed.get_rank()) if torch.distributed.is_initialized() else 0
    do_profile_rank = bool(getattr(args, "torch_profiler_all_ranks", False)) or (rank == 0)

    logits = _profile_forward(
        enabled=do_profile_rank and prof_mode in ("megatron", "both"),
        name="megatron_forward_logits",
        out_path=prof_dir / f"megatron_rank{rank}.json",
        record_shapes=bool(getattr(args, "torch_profiler_record_shapes", False)),
        profile_memory=bool(getattr(args, "torch_profiler_memory", False)),
        with_stack=bool(getattr(args, "torch_profiler_with_stack", False)),
        fn=lambda: model(
            input_ids,
            position_ids,
            attention_mask,
            labels=None,
            runtime_gather_output=bool(getattr(args, "runtime_gather_output", True)),
        ),
    )

    # logits: [b, s, vocab]
    if torch.distributed.get_rank() == 0:
        print_rank_0(f"Prompt tokens: {input_ids.shape[1]}")
        print_rank_0(f"Logits shape:  {tuple(logits.shape)}")

        # Megatron may use a padded vocab size; restrict to the *actual* tokenizer vocab
        # so top-k and comparisons are meaningful.
        tok_vocab_size = getattr(tokenizer, "vocab_size", None)
        if tok_vocab_size is None:
            tok_vocab_size = logits.shape[-1]
        vocab_limit = min(int(tok_vocab_size), int(logits.shape[-1]))
        print_rank_0(f"Tokenizer vocab_size: {int(tok_vocab_size)} (using logits[:{vocab_limit}])")

        last_logits = logits[0, -1, :vocab_limit]
        vals, idxs = torch.topk(last_logits, k=min(topk, last_logits.numel()))
        print_rank_0("Top-k next-token candidates:")
        for v, i in zip(vals.tolist(), idxs.tolist()):
            # tokenizer.detokenize() may drop special tokens; still useful as a hint.
            try:
                piece = tokenizer.detokenize([i])
            except Exception:
                piece = ""
            print_rank_0(f"  id={i:6d}  logit={v: .6f}  text={piece!r}")

        # Optional HF comparison (requires a converted HF checkpoint dir)
        hf_dir = getattr(args, "hf_dir", None)
        if hf_dir:
            from transformers import AutoTokenizer

            from modeling_zebra_llama import ZebraLlamaConfig, ZebraLlamaForCausalLM

            hf_dir_path = Path(str(hf_dir)).expanduser().resolve()
            cfg_path = hf_dir_path / "config.json"
            weights_path = hf_dir_path / "pytorch_model.bin"
            if not cfg_path.exists() or not weights_path.exists():
                raise FileNotFoundError(
                    f"--hf-dir must contain config.json and pytorch_model.bin. Got: {hf_dir_path}"
                )

            cfg_dict = __import__("json").loads(cfg_path.read_text())
            hf_config = ZebraLlamaConfig(**cfg_dict)

            hf_model = ZebraLlamaForCausalLM(hf_config).eval()
            # `use_return_dict` is a read-only property in Transformers configs.
            # We pass `return_dict=True` in the forward call below; also try setting
            # `return_dict` for any code paths that consult config defaults.
            try:
                hf_model.config.return_dict = True
            except Exception:
                pass

            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            missing, unexpected = hf_model.load_state_dict(state, strict=False)
            if missing:
                print_rank_0(f"[HF warn] Missing keys: {len(missing)} (showing first 10)")
                for k in missing[:10]:
                    print_rank_0(f"  - {k}")
            if unexpected:
                print_rank_0(f"[HF warn] Unexpected keys: {len(unexpected)} (showing first 10)")
                for k in unexpected[:10]:
                    print_rank_0(f"  - {k}")

            hf_tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-3.2-1B", trust_remote_code=True
            )
            hf_ids = hf_tokenizer(prompt, add_special_tokens=False).input_ids
            if len(hf_ids) > max_prompt_tokens:
                hf_ids = hf_ids[:max_prompt_tokens]

            if hf_ids != token_ids:
                print_rank_0("[HF compare] Tokenization mismatch between Megatron and HF tokenizers!")
                print_rank_0(f"  Megatron ids[:32]: {token_ids[:32]}")
                print_rank_0(f"  HF ids[:32]:       {hf_ids[:32]}")

            # Build HF inputs from the *Megatron* ids to ensure identical tokens.
            hf_input_ids = torch.tensor([token_ids], device=torch.cuda.current_device(), dtype=torch.long)
            hf_attention_mask = torch.ones_like(hf_input_ids, dtype=torch.long)
            hf_position_ids = torch.arange(
                hf_input_ids.shape[1], device=hf_input_ids.device, dtype=torch.long
            ).unsqueeze(0)
            if str(getattr(args, "position_ids_mode", "normal")) == "zeros":
                hf_position_ids = torch.zeros_like(hf_position_ids)

            dtype_name = str(getattr(args, "hf_dtype", "bfloat16"))
            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            hf_model = hf_model.to(device=torch.device("cuda"), dtype=dtype_map[dtype_name])

            def _extract_hidden(x):
                """
                Extract a tensor-like hidden state from various module outputs.
                Megatron modules sometimes return tuples where the first element is not the hidden state.
                """
                # Common: WrappedTensor-like object.
                if hasattr(x, "tensor") and isinstance(getattr(x, "tensor"), torch.Tensor):
                    return x.tensor
                # Common: tuple outputs (pick the first tensor-ish entry).
                if isinstance(x, tuple):
                    for y in x:
                        if hasattr(y, "tensor") and isinstance(getattr(y, "tensor"), torch.Tensor):
                            return y.tensor
                        if isinstance(y, torch.Tensor):
                            return y
                    return x
                return x

            def _to_bsh(x: torch.Tensor, seq_len: int, batch_size: int = 1) -> torch.Tensor:
                # Normalize hidden states to [B,S,H] for comparison.
                if x.dim() != 3:
                    raise ValueError(f"Expected 3D hidden states, got shape={tuple(x.shape)}")
                # Heuristic: Megatron often uses [S,B,H], HF uses [B,S,H]
                if x.shape[0] == seq_len and x.shape[1] == batch_size:
                    return x.transpose(0, 1).contiguous()
                return x

            def _repr_from_bsh(x_bsh: torch.Tensor, mode: str) -> torch.Tensor:
                if mode == "last":
                    return x_bsh[:, -1, :]
                # mode == "mean"
                return x_bsh.mean(dim=1)

            class _StopForward(Exception):
                """Internal control-flow to stop after a specific layer."""

            # Optional: layerwise comparison of intermediate activations.
            if bool(getattr(args, "compare_layerwise", False)):
                print_rank_0("Layerwise comparison enabled: collecting intermediate activations…")
                seq_len = int(hf_input_ids.shape[1])

                mg_layer_vecs: list[torch.Tensor] = []
                mg_layer_names: list[str] = []

                def _mk_mg_hook(name: str):
                    def _hook(_m, _inp, out):
                        h = _extract_hidden(out)
                        if not isinstance(h, torch.Tensor):
                            return
                        h_bsh = _to_bsh(h, seq_len=seq_len, batch_size=1).float()
                        mg_layer_vecs.append(_repr_from_bsh(h_bsh, getattr(args, "layerwise_token", "last")).cpu())
                        mg_layer_names.append(name)

                    return _hook

                hf_layer_vecs: list[torch.Tensor] = []
                hf_layer_names: list[str] = []

                def _mk_hf_hook(name: str):
                    def _hook(_m, _inp, out):
                        h = _extract_hidden(out)
                        if not isinstance(h, torch.Tensor):
                            return
                        h_bsh = _to_bsh(h, seq_len=seq_len, batch_size=1).float()
                        hf_layer_vecs.append(_repr_from_bsh(h_bsh, getattr(args, "layerwise_token", "last")).cpu())
                        hf_layer_names.append(name)

                    return _hook

                mg_hooks = []
                try:
                    mg_layers = getattr(model, "decoder", None)
                    mg_layers = getattr(mg_layers, "layers", None)
                    if mg_layers is None:
                        raise RuntimeError("Megatron model.decoder.layers not found for hooks.")
                    for i, layer in enumerate(mg_layers):
                        mg_hooks.append(layer.register_forward_hook(_mk_mg_hook(f"mg[{i}]:{layer.__class__.__name__}")))
                except Exception as e:
                    print_rank_0(f"[Layerwise] Failed to attach Megatron hooks: {e}")

                hf_hooks = []
                try:
                    hf_layers = getattr(hf_model, "model", None)
                    hf_layers = getattr(hf_layers, "layers", None)
                    if hf_layers is None:
                        raise RuntimeError("HF model.model.layers not found for hooks.")
                    for i, layer in enumerate(hf_layers):
                        hf_hooks.append(layer.register_forward_hook(_mk_hf_hook(f"hf[{i}]:{layer.__class__.__name__}")))
                except Exception as e:
                    print_rank_0(f"[Layerwise] Failed to attach HF hooks: {e}")

                # Run forwards again to collect activations
                _ = model(
                    input_ids,
                    position_ids,
                    attention_mask,
                    labels=None,
                    runtime_gather_output=bool(getattr(args, "runtime_gather_output", True)),
                )
                _ = hf_model(
                    input_ids=hf_input_ids,
                    attention_mask=hf_attention_mask,
                    position_ids=hf_position_ids,
                    use_cache=False,
                    return_dict=True,
                )

                for h in mg_hooks:
                    try:
                        h.remove()
                    except Exception:
                        pass
                for h in hf_hooks:
                    try:
                        h.remove()
                    except Exception:
                        pass

                n = min(len(mg_layer_vecs), len(hf_layer_vecs))
                print_rank_0(f"Layerwise vectors collected: mg={len(mg_layer_vecs)} hf={len(hf_layer_vecs)} compare_n={n}")
                if n > 0:
                    print_rank_0("Per-layer diff (vector max/mean abs, cosine):")
                    for i in range(n):
                        a = mg_layer_vecs[i].squeeze(0)
                        b = hf_layer_vecs[i].squeeze(0)
                        d = (a - b).abs()
                        # cosine similarity
                        cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
                        print_rank_0(
                            f"  {i:03d} {mg_layer_names[i]}  vs  {hf_layer_names[i]} | "
                            f"max={d.max().item():.6g} mean={d.mean().item():.6g} cos={cos:.6g}"
                        )

            # Optional: pre/post of first norm inside each block.
            if bool(getattr(args, "compare_prenorm", False)):
                print_rank_0("Pre/Post first-norm comparison enabled: collecting norm IO…")
                seq_len = int(hf_input_ids.shape[1])
                mode = str(getattr(args, "layerwise_token", "last"))

                mg_pre, mg_post, mg_names = [], [], []
                hf_pre, hf_post, hf_names = [], [], []

                def _mk_io_hook(store_pre, store_post, store_name, name: str):
                    def _hook(_m, inp, out):
                        if not inp:
                            return
                        x_in = _extract_hidden(inp[0])
                        x_out = _extract_hidden(out)
                        if not (isinstance(x_in, torch.Tensor) and isinstance(x_out, torch.Tensor)):
                            return
                        x_in_bsh = _to_bsh(x_in, seq_len=seq_len, batch_size=1).float()
                        x_out_bsh = _to_bsh(x_out, seq_len=seq_len, batch_size=1).float()
                        store_pre.append(_repr_from_bsh(x_in_bsh, mode).cpu())
                        store_post.append(_repr_from_bsh(x_out_bsh, mode).cpu())
                        store_name.append(name)

                    return _hook

                mg_io_hooks = []
                try:
                    mg_layers = getattr(model, "decoder", None)
                    mg_layers = getattr(mg_layers, "layers", None)
                    for i, layer in enumerate(mg_layers):
                        nname, nmod = _find_first_norm_like(layer)
                        if nmod is None:
                            continue
                        mg_io_hooks.append(
                            nmod.register_forward_hook(_mk_io_hook(mg_pre, mg_post, mg_names, f"mg[{i}].{nname}"))
                        )
                except Exception as e:
                    print_rank_0(f"[PreNorm] Failed to attach Megatron norm hooks: {e}")

                hf_io_hooks = []
                try:
                    hf_layers = getattr(hf_model, "model", None)
                    hf_layers = getattr(hf_layers, "layers", None)
                    for i, layer in enumerate(hf_layers):
                        nname, nmod = _find_first_norm_like(layer)
                        if nmod is None:
                            continue
                        hf_io_hooks.append(
                            nmod.register_forward_hook(_mk_io_hook(hf_pre, hf_post, hf_names, f"hf[{i}].{nname}"))
                        )
                except Exception as e:
                    print_rank_0(f"[PreNorm] Failed to attach HF norm hooks: {e}")

                # Re-run forwards to collect norm IO.
                _ = model(
                    input_ids,
                    position_ids,
                    attention_mask,
                    labels=None,
                    runtime_gather_output=bool(getattr(args, "runtime_gather_output", True)),
                )
                _ = hf_model(
                    input_ids=hf_input_ids,
                    attention_mask=hf_attention_mask,
                    position_ids=hf_position_ids,
                    use_cache=False,
                    return_dict=True,
                )

                for h in mg_io_hooks:
                    try:
                        h.remove()
                    except Exception:
                        pass
                for h in hf_io_hooks:
                    try:
                        h.remove()
                    except Exception:
                        pass

                n = min(len(mg_pre), len(hf_pre), len(mg_post), len(hf_post))
                print_rank_0(f"Pre/post norm vectors collected: mg={len(mg_pre)} hf={len(hf_pre)} compare_n={n}")
                if n > 0:
                    print_rank_0("Per-norm diff (PRE then POST): max/mean abs, cosine")
                    for i in range(n):
                        a0 = mg_pre[i].squeeze(0)
                        b0 = hf_pre[i].squeeze(0)
                        # PRE should generally be comparable (both are the incoming hidden state vector).
                        d0 = (a0 - b0).abs()
                        cos0 = torch.nn.functional.cosine_similarity(a0, b0, dim=0).item()
                        a1 = mg_post[i].squeeze(0)
                        b1 = hf_post[i].squeeze(0)
                        # POST may be non-comparable if Megatron uses a fused LN+Linear module
                        # (output dim != hidden_size). In that case, report PRE only and skip POST.
                        if a1.numel() != b1.numel():
                            print_rank_0(
                                f"  {i:03d} {mg_names[i]} vs {hf_names[i]} | "
                                f"PRE max={d0.max().item():.6g} mean={d0.mean().item():.6g} cos={cos0:.6g} || "
                                f"POST skipped (dim mismatch: mg={a1.numel()} hf={b1.numel()})"
                            )
                        else:
                            d1 = (a1 - b1).abs()
                            cos1 = torch.nn.functional.cosine_similarity(a1, b1, dim=0).item()
                            print_rank_0(
                                f"  {i:03d} {mg_names[i]} vs {hf_names[i]} | "
                                f"PRE max={d0.max().item():.6g} mean={d0.mean().item():.6g} cos={cos0:.6g} || "
                                f"POST max={d1.max().item():.6g} mean={d1.mean().item():.6g} cos={cos1:.6g}"
                            )

            # Optional: isolated per-layer compare (force identical input hidden per layer).
            if bool(getattr(args, "compare_layer_isolated", False)):
                print_rank_0("Isolated layer comparison enabled: capturing Megatron per-layer inputs/outputs…")
                seq_len = int(hf_input_ids.shape[1])
                mode = str(getattr(args, "layerwise_token", "last"))

                mg_in_bsh_cpu: dict[int, torch.Tensor] = {}
                mg_out_bsh_cpu: dict[int, torch.Tensor] = {}
                mg_names: dict[int, str] = {}

                # Capture Megatron layer input+output in a single forward pass.
                mg_io_hooks = []
                try:
                    mg_layers = getattr(model, "decoder", None)
                    mg_layers = getattr(mg_layers, "layers", None)
                    if mg_layers is None:
                        raise RuntimeError("Megatron model.decoder.layers not found for hooks.")

                    def _mk_mg_pre_hook(i: int, name: str):
                        def _pre(_m, inp, kwargs=None):
                            # Megatron may call layers with positional args or kwargs.
                            x_src = None
                            if inp:
                                x_src = inp[0]
                            elif kwargs:
                                # Prefer common hidden-state kwarg names.
                                for k in ("hidden_states", "hidden", "x", "input", "input_tensor"):
                                    if k in kwargs:
                                        x_src = kwargs[k]
                                        break
                                if x_src is None:
                                    # Fallback: first tensor-ish value.
                                    for v in kwargs.values():
                                        if isinstance(v, torch.Tensor) or (hasattr(v, "tensor") and isinstance(getattr(v, "tensor"), torch.Tensor)):
                                            x_src = v
                                            break
                            if x_src is None:
                                return

                            x = _extract_hidden(x_src)
                            if not isinstance(x, torch.Tensor):
                                return
                            x_bsh = _to_bsh(x, seq_len=seq_len, batch_size=1)
                            # store float32 on CPU for stable injection + low GPU memory
                            mg_in_bsh_cpu[i] = x_bsh.detach().float().cpu()
                            mg_names[i] = name

                        return _pre

                    def _mk_mg_post_hook(_i: int):
                        def _post(_m, _inp, out):
                            x = _extract_hidden(out)
                            if not isinstance(x, torch.Tensor):
                                return
                            x_bsh = _to_bsh(x, seq_len=seq_len, batch_size=1)
                            mg_out_bsh_cpu[_i] = x_bsh.detach().float().cpu()

                        return _post

                    for i, layer in enumerate(mg_layers):
                        name = f"mg[{i}]:{layer.__class__.__name__}"
                        # Some Megatron layers are invoked with kwargs; capture those too.
                        try:
                            mg_io_hooks.append(layer.register_forward_pre_hook(_mk_mg_pre_hook(i, name), with_kwargs=True))
                        except TypeError:
                            mg_io_hooks.append(layer.register_forward_pre_hook(_mk_mg_pre_hook(i, name)))
                        mg_io_hooks.append(layer.register_forward_hook(_mk_mg_post_hook(i)))
                except Exception as e:
                    print_rank_0(f"[Isolated] Failed to attach Megatron IO hooks: {e}")

                # Run Megatron once to populate mg_in_bsh_cpu / mg_out_bsh_cpu.
                _ = model(
                    input_ids,
                    position_ids,
                    attention_mask,
                    labels=None,
                    runtime_gather_output=bool(getattr(args, "runtime_gather_output", True)),
                )

                for h in mg_io_hooks:
                    try:
                        h.remove()
                    except Exception:
                        pass

                common = sorted(set(mg_in_bsh_cpu.keys()) & set(mg_out_bsh_cpu.keys()))
                print_rank_0(
                    f"[Isolated] Captured Megatron IO: pre={len(mg_in_bsh_cpu)} post={len(mg_out_bsh_cpu)} common={len(common)}"
                )
                if len(common) == 0:
                    print_rank_0("[Isolated] No Megatron layers captured; skipping isolated compare.")
                else:
                    # Run HF once per layer, overriding that layer's input with Megatron's captured input.
                    hf_layers = getattr(hf_model, "model", None)
                    hf_layers = getattr(hf_layers, "layers", None)
                    if hf_layers is None:
                        raise RuntimeError("HF model.model.layers not found for hooks.")

                    # Compare only up to common layer count.
                    n = min(len(common), len(hf_layers))
                    common = common[:n]
                    print_rank_0(f"[Isolated] Comparing {n} layers (mg_common={len(common)}, hf={len(hf_layers)}).")
                    print_rank_0("Per-layer isolated diff (output vector max/mean abs, cosine):")

                    hf_param_dtype = next(hf_model.parameters()).dtype
                    hf_device = next(hf_model.parameters()).device

                    for i in common:
                        inj_bsh = mg_in_bsh_cpu[i].to(device=hf_device, dtype=hf_param_dtype)
                        got_out: dict[str, torch.Tensor] = {}

                        def _hf_pre(_m, inp, kwargs=None):
                            # HF layers are usually called positionally, but support kwargs for safety.
                            if inp:
                                new_args = (inj_bsh,) + tuple(inp[1:])
                                # When registered with `with_kwargs=True`, must return (new_args, new_kwargs).
                                if kwargs is not None:
                                    return new_args, kwargs
                                return new_args
                            if kwargs is not None:
                                kwargs = dict(kwargs)
                                # Try common hidden-state names.
                                for k in ("hidden_states", "hidden", "x", "input", "input_tensor"):
                                    if k in kwargs:
                                        kwargs[k] = inj_bsh
                                        return (), kwargs
                                # If we couldn't find a name to overwrite, still satisfy hook contract.
                                return (), kwargs
                            return inp

                        def _hf_post(_m, _inp, out):
                            x = _extract_hidden(out)
                            if isinstance(x, torch.Tensor):
                                got_out["out"] = x
                            raise _StopForward()

                        try:
                            h_pre = hf_layers[i].register_forward_pre_hook(_hf_pre, with_kwargs=True)
                        except TypeError:
                            h_pre = hf_layers[i].register_forward_pre_hook(_hf_pre)
                        h_post = hf_layers[i].register_forward_hook(_hf_post)
                        try:
                            try:
                                _ = hf_model(
                                    input_ids=hf_input_ids,
                                    attention_mask=hf_attention_mask,
                                    position_ids=hf_position_ids,
                                    use_cache=False,
                                    return_dict=True,
                                )
                            except _StopForward:
                                pass

                            if "out" not in got_out:
                                print_rank_0(f"  {i:03d} {mg_names[i]}  vs  hf[{i}] | no output captured")
                                continue

                            hf_out = _extract_hidden(got_out["out"])
                            if not isinstance(hf_out, torch.Tensor):
                                print_rank_0(f"  {i:03d} {mg_names[i]}  vs  hf[{i}] | non-tensor output")
                                continue

                            hf_out_bsh = _to_bsh(hf_out, seq_len=seq_len, batch_size=1).float().cpu()
                            mg_out_bsh = mg_out_bsh_cpu[i]

                            a = _repr_from_bsh(mg_out_bsh, mode).squeeze(0)
                            b = _repr_from_bsh(hf_out_bsh, mode).squeeze(0)
                            d = (a - b).abs()
                            cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
                            print_rank_0(
                                f"  {i:03d} {mg_names.get(i, f'mg[{i}]')}  vs  hf[{i}]:{hf_layers[i].__class__.__name__} | "
                                f"max={d.max().item():.6g} mean={d.mean().item():.6g} cos={cos:.6g}"
                            )
                        finally:
                            try:
                                h_pre.remove()
                            except Exception:
                                pass
                            try:
                                h_post.remove()
                            except Exception:
                                pass

            hf_out = _profile_forward(
                enabled=do_profile_rank and prof_mode in ("hf", "both"),
                name="hf_forward_logits",
                out_path=prof_dir / f"hf_rank{rank}.json",
                record_shapes=bool(getattr(args, "torch_profiler_record_shapes", False)),
                profile_memory=bool(getattr(args, "torch_profiler_memory", False)),
                with_stack=bool(getattr(args, "torch_profiler_with_stack", False)),
                fn=lambda: hf_model(
                    input_ids=hf_input_ids,
                    attention_mask=hf_attention_mask,
                    position_ids=hf_position_ids,
                    use_cache=False,
                    return_dict=True,
                ),
            )
            hf_logits = hf_out.logits  # [b, s, vocab]

            # Compare on shared vocab range (Megatron vocab may be padded)
            v = min(hf_logits.shape[-1], last_logits.shape[-1])
            mg_last = last_logits[:v].float()
            hf_last = hf_logits[0, -1, :v].float()
            diff = (mg_last - hf_last).abs()

            atol = float(getattr(args, "compare_atol", 1e-2))
            print_rank_0("HF comparison (last-token logits):")
            print_rank_0(f"  shared_vocab={v}")
            print_rank_0(f"  max_abs_diff={diff.max().item():.6g}")
            print_rank_0(f"  mean_abs_diff={diff.mean().item():.6g}")
            print_rank_0(f"  pct(|diff|<=atol {atol:g})={(diff <= atol).float().mean().item()*100:.2f}%")

            mg_top1 = int(torch.argmax(mg_last).item())
            hf_top1 = int(torch.argmax(hf_last).item())
            print_rank_0(f"  top1_match={mg_top1 == hf_top1}  (mg={mg_top1}, hf={hf_top1})")

            mg_vals, mg_idxs = torch.topk(mg_last, k=min(topk, mg_last.numel()))
            hf_vals, hf_idxs = torch.topk(hf_last, k=min(topk, hf_last.numel()))
            print_rank_0("  top-k (Megatron vs HF):")
            for rank, (mvi, mii, hvi, hii) in enumerate(
                zip(mg_vals.tolist(), mg_idxs.tolist(), hf_vals.tolist(), hf_idxs.tolist()),
                start=1,
            ):
                try:
                    mtxt = hf_tokenizer.decode([mii])
                except Exception:
                    mtxt = ""
                try:
                    htxt = hf_tokenizer.decode([hii])
                except Exception:
                    htxt = ""
                print_rank_0(
                    f"   #{rank:02d}  mg id={mii:6d} logit={mvi: .6f} text={mtxt!r} | "
                    f"hf id={hii:6d} logit={hvi: .6f} text={htxt!r}"
                )

        if getattr(args, "save_logits", None):
            out_path = Path(str(getattr(args, "save_logits"))).expanduser()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"prompt": prompt, "input_ids": input_ids.cpu(), "logits": logits.cpu()}, out_path)
            print_rank_0(f"Saved logits to: {str(out_path)}")


if __name__ == "__main__":
    # Megatron relies on torchrun/torch.distributed init; initialize_megatron handles it.
    main()

