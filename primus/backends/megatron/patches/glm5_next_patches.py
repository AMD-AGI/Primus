###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""GLM-5.3 (``model_type: glm5_next``) runtime patches.

1. ``megatron.glm5_next.pp_tensor_shape``: the decoder block carries its
   ``hc_mult`` mHC streams across PP boundaries folded into the sequence axis
   (``[hc_mult * s, b, h]``), so the PP wire shape is scaled the same way the
   Kimi K3 / DeepSeek-V4 patches do it.

2. ``megatron.glm5_next.hf_load``: wraps ``setup_model_and_optimizer`` so that
   ``glm5_hf_load_path`` (yaml) / ``PRIMUS_GLM5_HF_LOAD`` (env) loads a
   HuggingFace checkpoint online after the model is built (and refreshes the
   optimizer's main params).

   With ``glm5_logprob_eval_input`` / ``PRIMUS_GLM5_LOGPROB_EVAL_INPUT`` set,
   it instead builds the model **without** DDP / optimizer (the full model in
   bf16 fits one node at EP8, its optimizer state does not), loads the HF
   checkpoint, runs ``forward_only`` over a jsonl of token sequences, writes the
   per-token log-probabilities to ``glm5_logprob_eval_output`` and exits.

   Input line: ``{"id": ..., "tokens": [int, ...]}``. Output line:
   ``{"id": ..., "logprobs": [...]}`` where ``logprobs[i] = log p(tokens[i+1] | tokens[:i+1])``.

3. ``megatron.glm5_next.te_gemm_workspace``: enlarges TE's hipBLASLt GEMM
   workspace (``PRIMUS_TE_GEMM_WORKSPACE_MB``, default 512).
"""

from __future__ import annotations

import contextlib
import json
import os
import sys

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0


def _is_glm5_next(ctx: PatchContext) -> bool:
    return getattr(get_args(ctx), "model_type", None) == "glm5_next"


def _opt(args, name: str, env: str):
    return getattr(args, name, None) or os.environ.get(env) or None


@register_patch(
    "megatron.glm5_next.pp_tensor_shape",
    backend="megatron",
    phase="before_train",
    description="GLM-5.3: fold the hc_mult mHC streams into the PP wire sequence axis.",
    condition=lambda ctx: _is_glm5_next(ctx)
    and int(getattr(get_args(ctx), "pipeline_model_parallel_size", 1) or 1) > 1,
)
def patch_glm5_next_pp_tensor_shape(ctx: PatchContext):
    import megatron.core.pipeline_parallel.schedules as schedules_module

    from primus.backends.megatron.patches.kimi_k3_pp_shape_patches import (
        _make_k3_get_tensor_shapes,
        _make_k3_interleaved_schedule,
    )

    args = get_args(ctx)
    seq_mult = int(getattr(args, "hc_mult", 4) or 4)
    for flag in ("patch_zero_bubble", "patch_primus_pipeline"):
        if getattr(args, flag, False):
            raise NotImplementedError(f"GLM-5.3 with pipeline parallelism does not support {flag}=True.")
    schedules_module.get_tensor_shapes = _make_k3_get_tensor_shapes(
        schedules_module.get_tensor_shapes, seq_mult
    )
    schedules_module.forward_backward_pipelining_with_interleaving = _make_k3_interleaved_schedule(
        schedules_module.forward_backward_pipelining_with_interleaving, seq_mult
    )
    log_rank_0(f"[Patch:megatron.glm5_next.pp_tensor_shape] PP wire seq_len * hc_mult = {seq_mult}")


def _run_logprob_eval(model_chunks, in_path: str, out_path: str) -> None:
    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state, tensor_parallel

    assert (
        len(model_chunks) == 1 and parallel_state.get_pipeline_model_parallel_world_size() == 1
    ), "GLM-5.3 logprob eval supports PP=1 only."
    model = model_chunks[0]
    model.eval()
    with open(in_path) as f:
        samples = [json.loads(line) for line in f if line.strip()]
    tp = parallel_state.get_tensor_model_parallel_world_size()
    pad_to = 64 * tp
    rank0 = dist.get_rank() == 0
    out_ctx = open(out_path, "w") if rank0 else contextlib.nullcontext(None)
    device = torch.cuda.current_device()
    with out_ctx as out_f:
        for i, sample in enumerate(samples):
            tokens = [int(t) for t in sample["tokens"]]
            n = len(tokens)
            padded = n + (-n) % pad_to
            ids = torch.zeros(1, padded, dtype=torch.long, device=device)
            ids[0, :n] = torch.tensor(tokens, device=device)
            target = torch.zeros_like(ids)
            target[0, : n - 1] = ids[0, 1:n]
            with torch.no_grad():
                logits = model(input_ids=ids, position_ids=None, attention_mask=None)  # [1, S, V/tp]
                nll = tensor_parallel.vocab_parallel_cross_entropy(
                    logits.float().transpose(0, 1).contiguous(), target.t()
                )
            logprobs = (-nll[: n - 1, 0]).tolist()
            if rank0:
                out_f.write(json.dumps({"id": sample.get("id", i), "logprobs": logprobs}) + "\n")
                out_f.flush()
            log_rank_0(
                f"[GLM5-Next logprob eval] sample {i}: {n} tokens, mean logprob {sum(logprobs) / max(1, len(logprobs)):.4f}"
            )


@register_patch(
    "megatron.glm5_next.hf_load",
    backend="megatron",
    phase="before_train",
    description="GLM-5.3: online HF checkpoint load and the forward-only logprob eval mode.",
    condition=lambda ctx: _is_glm5_next(ctx)
    and bool(
        _opt(get_args(ctx), "glm5_hf_load_path", "PRIMUS_GLM5_HF_LOAD")
        or _opt(get_args(ctx), "glm5_logprob_eval_input", "PRIMUS_GLM5_LOGPROB_EVAL_INPUT")
    ),
)
def patch_glm5_next_hf_load(ctx: PatchContext):
    import megatron.training.training as training_module

    args = get_args(ctx)
    hf_path = _opt(args, "glm5_hf_load_path", "PRIMUS_GLM5_HF_LOAD")
    eval_in = _opt(args, "glm5_logprob_eval_input", "PRIMUS_GLM5_LOGPROB_EVAL_INPUT")
    eval_out = (
        _opt(args, "glm5_logprob_eval_output", "PRIMUS_GLM5_LOGPROB_EVAL_OUTPUT") or "glm5_logprobs.jsonl"
    )
    original = training_module.setup_model_and_optimizer

    def _load(model):
        from primus.backends.megatron.core.models.glm5_next.glm5_next_hf_loader import (
            load_glm5_next_hf_checkpoint,
        )

        assert hf_path, "glm5_hf_load_path / PRIMUS_GLM5_HF_LOAD is required"
        n = load_glm5_next_hf_checkpoint(model, hf_path)
        log_rank_0(f"[GLM5-Next] loaded {n} tensors from HF checkpoint {hf_path}")

    def setup_model_and_optimizer(model_provider_func, model_type, *a, **kw):
        if eval_in:
            import torch.distributed as dist

            model = training_module.get_model(model_provider_func, model_type, wrap_with_ddp=False)
            _load(model)
            _run_logprob_eval(model, eval_in, eval_out)
            log_rank_0(f"[GLM5-Next] logprob eval written to {eval_out}; exiting.")
            dist.barrier()
            dist.destroy_process_group()
            sys.exit(0)
        model, optimizer, scheduler = original(model_provider_func, model_type, *a, **kw)
        if hf_path and int(getattr(training_module.get_args(), "iteration", 0) or 0) == 0:
            _load(model)
            if optimizer is not None and hasattr(optimizer, "reload_model_params"):
                optimizer.reload_model_params()
        return model, optimizer, scheduler

    training_module.setup_model_and_optimizer = setup_model_and_optimizer
    log_rank_0(f"[Patch:megatron.glm5_next.hf_load] hf={hf_path} eval_in={eval_in} eval_out={eval_out}")


@register_patch(
    "megatron.glm5_next.te_gemm_workspace",
    backend="megatron",
    phase="before_train",
    description="GLM-5.3: enlarge the TE hipBLASLt GEMM workspace (64 MiB is too small for some shapes).",
    condition=_is_glm5_next,
)
def patch_glm5_next_te_gemm_workspace(ctx: PatchContext):
    # On gfx950, hipBLASLt picks split-K algorithms for large-K / small-N GEMMs
    # (e.g. KDA o_proj 8192->4096 at 960..1280 tokens) whose workspace exceeds
    # TE's 64 MiB default and fails with "HIPBLASLT Error: 6".
    del ctx
    import torch

    if torch.version.hip is None:
        return
    import transformer_engine.pytorch.cpp_extensions.gemm as te_gemm

    size = int(os.environ.get("PRIMUS_TE_GEMM_WORKSPACE_MB", "512")) * 1024 * 1024
    if te_gemm.get_cublas_workspace_size_bytes() >= size:
        return
    te_gemm.get_cublas_workspace_size_bytes = lambda: size
    te_gemm.get_cublas_workspace.cache_clear()
    log_rank_0(f"[Patch:megatron.glm5_next.te_gemm_workspace] TE GEMM workspace -> {size >> 20} MiB")
