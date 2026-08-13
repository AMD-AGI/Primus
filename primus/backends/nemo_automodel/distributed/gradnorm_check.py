###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Verify that the reported gradient norm is the true global one.

WHY:
  ``clip_grad_norm`` collects ``p.grad`` over ``model.parameters()`` and groups by
  device mesh and placement. That is correct as long as every parameter's grad is
  either a plain replicated tensor or a DTensor whose placements describe how it is
  actually sharded. A backend that hands it something else -- a bare local shard, a
  main-grad buffer that is not the parameter's ``.grad``, a DTensor on a mesh the
  grouping does not recognise -- gets a norm computed over a fraction of the model.

  Nothing catches this. With ``clip_grad_norm.max_norm`` at its default 1.0 the
  gradients are clipped every step of a from-scratch run, so an under-reported norm
  does not shrink the update, it INFLATES it: clipping scales by
  ``max_norm / reported``, and if ``reported`` is half of true then the post-clip
  norm is twice ``max_norm``. The effective learning rate is wrong by that factor,
  the loss curve still descends, and a short benchmark looks perfectly healthy.

WHAT:
  Wrap the recipe's ``clip_grad_norm`` and, for the first N steps, recompute the
  norm from first principles: square the local gradient of every parameter, sum the
  sharded contributions across the data-parallel group, count each replicated
  contribution once, and compare. Logs both values and their ratio.

  This is a diagnostic, not a guard: it costs an all-reduce and a full pass over the
  gradients per checked step, so it is off unless asked for, and it stops after N
  steps rather than running for a whole job.

Activation (env, no config schema change):
    PRIMUS_GRADNORM_CHECK=<n>   verify the first <n> steps (0 / unset = off)
"""
from __future__ import annotations

import functools
import logging
import os

logger = logging.getLogger(__name__)

_checked = 0


def check_steps() -> int:
    try:
        return int(os.getenv("PRIMUS_GRADNORM_CHECK", "0"))
    except ValueError:
        return 0


def _reference_grad_norm(model_parts) -> float:
    """Recompute the global L2 gradient norm without trusting placement grouping."""
    import torch
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor, Replicate

    sharded_sq = 0.0
    replicated_sq = 0.0
    missed_sq = 0.0
    n_sharded = n_replicated = n_nograd = n_main_grad = 0

    # Element counts, not tensor counts. Megatron-FSDP exposes flat bucketed
    # parameters, so "fewer tensors carry a gradient" is expected and means nothing
    # on its own; whether the covered ELEMENTS add up to the whole model is the
    # question. DTensor.numel() is already the global count; a plain tensor here is
    # replicated, so its local count is also the global one.
    covered_numel = 0
    missing_numel = 0

    missing_names = []
    # Fingerprints of one covered and one uncovered parameter. A parameter that is
    # invisible to the clipper might still be training (its gradient living
    # somewhere the clipper cannot see) or might be frozen outright, and those are
    # very different failures. Watching the values move across steps separates them.
    covered_fp = missing_fp = None
    has_grad, param_numel = [], []
    for module in model_parts:
        for name, p in module.named_parameters():
            if not p.requires_grad:
                continue
            uncovered = p.grad is None and getattr(p, "main_grad", None) is None
            has_grad.append(0 if uncovered else 1)
            param_numel.append(p.numel())
            if uncovered and len(missing_names) < 8:
                missing_names.append(f"{name}{tuple(p.shape)}")
            # Local element count matters as much as the value here: a parameter
            # whose local shard is EMPTY lives entirely on another rank, which is
            # ordinary under flat-buffer sharding and means the other rank
            # contributes it to the all-reduce. A parameter with a full-size local
            # tensor that reads as all zeros is a different and much worse thing.
            local_p = p.to_local() if isinstance(p, DTensor) else p
            fp = (name, local_p.numel(), p.numel(), local_p.detach().float().abs().sum().item())
            if uncovered and missing_fp is None:
                missing_fp = fp
            elif not uncovered and covered_fp is None:
                covered_fp = fp
            g = p.grad
            if g is None:
                # Megatron-FSDP accumulates into its own main-grad buffer. A
                # parameter whose gradient lives only there is invisible to
                # clip_grad_norm, which reads p.grad -- so its contribution is
                # missing from the norm the clip is scaled by.
                main = getattr(p, "main_grad", None)
                if main is not None:
                    local = main.to_local() if isinstance(main, DTensor) else main
                    missed_sq += local.detach().float().pow(2).sum().item()
                    covered_numel += main.numel()
                    n_main_grad += 1
                else:
                    missing_numel += p.numel()
                    n_nograd += 1
                continue
            if isinstance(g, DTensor):
                local = g.to_local()
                # Replicated on every mesh axis means every rank holds the same
                # values, so summing across ranks would count them world_size times.
                is_replicated = all(isinstance(pl, Replicate) for pl in g.placements)
            else:
                local = g
                is_replicated = True
            sq = local.detach().float().pow(2).sum().item()
            covered_numel += g.numel()
            if is_replicated:
                replicated_sq += sq
                n_replicated += 1
            else:
                sharded_sq += sq
                n_sharded += 1

    if dist.is_available() and dist.is_initialized():
        t = torch.tensor([sharded_sq, missed_sq], dtype=torch.float64, device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        sharded_sq, missed_sq = t[0].item(), t[1].item()

    for label, fp in (("covered", covered_fp), ("uncovered", missing_fp)):
        if fp is not None:
            logger.info(
                "[PrimusGradNormCheck] %s param %s: local %d of %d elements, |w|_1 = %.6f",
                label,
                fp[0],
                fp[1],
                fp[2],
                fp[3],
            )

    # A parameter with no gradient on THIS rank is not necessarily missing from the
    # norm: under flat-buffer sharding a whole parameter can live on another rank,
    # which contributes it to the all-reduce. Only a parameter that no rank has a
    # gradient for is genuinely absent, so decide it with a per-parameter MAX across
    # ranks rather than from one rank's view. named_parameters() iterates in the same
    # order everywhere, so the index into this vector means the same thing on each rank.
    total_numel = covered_numel + missing_numel
    logger.info(
        "[PrimusGradNormCheck] tensors: %d sharded, %d replicated, %d main_grad-only, "
        "%d without grad | elements: %.3fB of %.3fB covered (%.2f%%)",
        n_sharded,
        n_replicated,
        n_main_grad,
        n_nograd,
        covered_numel / 1e9,
        total_numel / 1e9,
        100.0 * covered_numel / total_numel if total_numel else float("nan"),
    )
    if missing_numel:
        logger.error(
            "[PrimusGradNormCheck] %d parameter(s) totalling %.3fB elements require grad "
            "but have none at clip time: they contribute nothing to the norm and may not "
            "be training at all. First few: %s",
            n_nograd,
            missing_numel / 1e9,
            ", ".join(missing_names),
        )
    if n_main_grad:
        visible = (sharded_sq + replicated_sq) ** 0.5
        whole = (sharded_sq + replicated_sq + missed_sq) ** 0.5
        logger.error(
            "[PrimusGradNormCheck] %d parameter(s) carry a gradient ONLY in main_grad, "
            "where clip_grad_norm cannot see it: norm over visible grads %.6f vs %.6f "
            "over all grads (%.4fx). Clipping is scaled by the smaller number.",
            n_main_grad,
            visible,
            whole,
            whole / visible if visible else float("nan"),
        )
    return (sharded_sq + replicated_sq) ** 0.5


def install() -> bool:
    """Wrap the diffusion recipe's clip_grad_norm. No-op unless PRIMUS_GRADNORM_CHECK is set."""
    n = check_steps()
    if n <= 0:
        return False

    import nemo_automodel.recipes.diffusion.train as train_mod

    orig = train_mod.clip_grad_norm
    if getattr(orig, "_primus_gradnorm_checked", False):
        return True

    @functools.wraps(orig)
    def _clip_and_check(max_grad_norm, model_parts, **kwargs):
        global _checked

        if _checked >= n:
            return orig(max_grad_norm, model_parts, **kwargs)

        reference = _reference_grad_norm(model_parts)
        # Called before clipping mutates the gradients, so the reference above and
        # the norm below are taken from the same values.
        reported = orig(max_grad_norm, model_parts, **kwargs)
        _checked += 1

        reported_f = float(reported)
        ratio = reported_f / reference if reference else float("nan")
        verdict = "OK" if 0.99 <= ratio <= 1.01 else "MISMATCH"
        logger.info(
            "[PrimusGradNormCheck] step %d: reported=%.6f reference=%.6f " "ratio=%.4f -> %s",
            _checked,
            reported_f,
            reference,
            ratio,
            verdict,
        )
        if verdict == "MISMATCH":
            logger.error(
                "[PrimusGradNormCheck] the clipped update is scaled by max_norm/reported, "
                "so a ratio of %.4f means the post-clip gradient norm is %.4fx max_norm "
                "and the effective learning rate is wrong by that factor.",
                ratio,
                1.0 / ratio if ratio else float("nan"),
            )
        return reported

    _clip_and_check._primus_gradnorm_checked = True
    train_mod.clip_grad_norm = _clip_and_check
    logger.info("[PrimusGradNormCheck] verifying the first %d step(s)", n)
    return True
