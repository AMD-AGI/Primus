"""Switch A6W4 forwards to an FP8 forward at FLUX_FPROP_SWITCH_STEP."""

import logging
import os
import time
from dataclasses import replace

import torch

from primus.backends.diffusion.models.quantization import mxfp4_linear as mx

logger = logging.getLogger(__name__)

_PRECOMPILE = os.getenv("FLUX_FPROP_SWITCH_PRECOMPILE", "1") == "1"
# (module, A6W4 config, FP8 config, FP8 gather eligible)
_MODS: list = []
# id(FSDPParam) -> the local shard wrapper for each mode, kept so a switch back restores the
# original object.
_A6W4_LOCAL: dict = {}
_FP8_LOCAL: dict = {}
_READY = False


def _log(msg):
    logger.info(msg)
    print(msg, flush=True)


def switch_step():
    value = os.getenv("FLUX_FPROP_SWITCH_STEP", "").strip()
    if not value:
        return None
    step = int(value)
    if step <= 0:
        raise ValueError(f"FLUX_FPROP_SWITCH_STEP must be > 0, got {step}")
    fp4_ag = os.getenv("FLUX_FP4_ALL_GATHER", "1") == "1"
    if not _PRECOMPILE and fp4_ag:
        raise RuntimeError("FLUX_FPROP_SWITCH_PRECOMPILE=0 needs FLUX_FP4_ALL_GATHER=0")
    if _PRECOMPILE and fp4_ag and os.getenv("FLUX_FP8_ALL_GATHER", "0") != "1":
        raise RuntimeError(
            "FLUX_FPROP_SWITCH_STEP with the FP4 gather needs FLUX_FP8_ALL_GATHER=1, which runs "
            "the per-step amax refresh the FP8 gather quantizes with"
        )
    return step


def _raise_dynamo_limits():
    import torch._dynamo.config as dc

    for name, floor in (
        ("recompile_limit", 64),
        ("cache_size_limit", 64),
        ("accumulated_recompile_limit", 2048),
        ("accumulated_cache_size_limit", 2048),
    ):
        if hasattr(dc, name):
            setattr(dc, name, max(getattr(dc, name), floor))


def prepare(model):
    """Before the warmup: the config pair and FP8 scale slots of every A6W4 linear."""
    if not _PRECOMPILE:
        return
    _MODS.clear()
    for module in model.modules():
        if isinstance(module, mx.MXFP4Linear) and module.config.fprop == "a6w4":
            a6w4 = module.config
            fp8 = replace(a6w4, fprop="fp8")
            module.config = fp8
            module._build_fp8_slots(module.weight.device)
            eligible = mx._fp8_all_gather_eligible(module)
            module.config = a6w4
            _MODS.append((module, a6w4, fp8, eligible))
    _raise_dynamo_limits()
    _log(f"FLUX_FPROP_SWITCH_STEP: {len(_MODS)} A6W4 linears prepared for the FP8 switch "
         f"({sum(m[3] for m in _MODS)} FP8-gather eligible)")


def _fsdp_params(model):
    """id(owning module) -> its FSDPParam for ``weight``. Keyed on the module rather than on
    ``module.weight``, which is the unsharded parameter whenever the group is unsharded."""
    out = {}
    for module in model.modules():
        get = getattr(module, "_get_fsdp_state", None)
        if get is None:
            continue
        group = get()._fsdp_param_group
        if group is None:
            continue
        for fp in group.fsdp_params:
            info = fp._module_info
            if info.param_name == "weight":
                out[id(info.module)] = fp
    return out


def _reset_gather_state(fp):
    from torch.distributed.fsdp._fully_shard._fsdp_param import ShardedState

    if fp.sharded_state != ShardedState.SHARDED:
        raise RuntimeError(f"FLUX_FPROP_SWITCH_STEP: gather swap on a {fp.sharded_state} parameter")
    fp.all_gather_outputs = []
    if hasattr(fp, "_unsharded_param"):
        del fp._unsharded_param
    fp._unsharded_inner_tensors = []
    ext = getattr(fp, "_extensions_data", None)
    if ext is not None:
        ext.clear()


def _swap_gather(model, to_fp8):
    # Groups with reshard_after_forward=False (the root) stay unsharded between steps; the swap
    # needs every gather state to be the sharded one.
    for module in model.modules():
        if hasattr(module, "_get_fsdp_state") and hasattr(module, "reshard"):
            module.reshard()
    fps = _fsdp_params(model)
    swapped = plain = 0
    skipped: dict = {}
    for module, _, _, eligible in _MODS:
        fp = fps.get(id(module))
        if fp is None:
            skipped["no FSDPParam"] = skipped.get("no FSDPParam", 0) + 1
            continue
        weight = fp.sharded_param
        cur = weight._local_tensor
        key = id(fp)
        if to_fp8:
            if not isinstance(cur, mx._A6W4AllGatherWeight):
                kind = f"local {type(cur).__name__}"
                skipped[kind] = skipped.get(kind, 0) + 1
                continue
            _A6W4_LOCAL[key] = cur
            if eligible:
                new = _FP8_LOCAL.get(key)
                if new is None or new._tensor is not cur._tensor:
                    new = mx._FP8AllGatherWeight(cur._tensor)
                    _FP8_LOCAL[key] = new
            else:
                # Gathers the bf16 weight, as with FLUX_FP4_ALL_GATHER=0.
                new = cur._tensor
                plain += 1
        else:
            old = _A6W4_LOCAL.get(key)
            if old is None:
                continue
            if cur is old:
                continue
            if old._tensor is not getattr(cur, "_tensor", cur):
                raise RuntimeError("FLUX_FPROP_SWITCH_STEP: a gathered shard was rebound since the swap")
            new = old
        weight._local_tensor = new
        _reset_gather_state(fp)
        swapped += 1
    if skipped:
        _log(f"FLUX_FPROP_SWITCH_STEP: gather swap to {'fp8' if to_fp8 else 'a6w4'} skipped {skipped}")
    if to_fp8 and os.getenv("FLUX_FP4_ALL_GATHER", "1") == "1" and swapped == 0 and _MODS:
        raise RuntimeError("FLUX_FPROP_SWITCH_STEP: the FP4 gather is on but no gather was swapped")
    return swapped, plain


def _set_mode(model, fp8):
    for module, a6w4_cfg, fp8_cfg, _ in _MODS:
        module.config = fp8_cfg if fp8 else a6w4_cfg
    return _swap_gather(model, fp8)


def _refresh_fp8_gather(model):
    if os.getenv("FLUX_FP8_ALL_GATHER", "0") == "1":
        mx.precompute_fp8_all_gather_amax(model)


def precompile(trainer):
    """After the warmup: compile the FP8 graphs and the FP8 gather, then return to A6W4."""
    global _READY
    if not _PRECOMPILE:
        return
    t0 = time.time()
    model = trainer.model
    saved = (trainer.mlperf_warmup_train_steps, trainer.mlperf_warmup_validation_steps)
    trainer.mlperf_warmup_train_steps = max(1, saved[0])
    trainer.mlperf_warmup_validation_steps = max(1, saved[1])
    try:
        swapped, plain = _set_mode(model, True)
        _refresh_fp8_gather(model)
        trainer._mlperf_warmup()
        if mx._FP8_AG_PREQUANT:
            # The second refresh captures the prequant graphs; the second warmup replays them.
            _refresh_fp8_gather(model)
            trainer._mlperf_warmup()
        _set_mode(model, False)
    finally:
        trainer.mlperf_warmup_train_steps, trainer.mlperf_warmup_validation_steps = saved
    dropped = 0
    for module, _, _, _ in _MODS:
        for slot in (module.fp8_act_slot, module.fp8_wgt_slot):
            if slot is not None and mx._FP8_DUAL_SCALE.pop(slot.data_ptr(), None) is not None:
                dropped += 1
    torch.cuda.synchronize()
    _READY = True
    _log(f"FLUX_FPROP_SWITCH_STEP: precompiled the FP8 forward for {len(_MODS)} linears in "
         f"{time.time() - t0:.1f}s ({swapped} gathers swapped, {plain} of them to bf16; "
         f"{dropped} warmup FP8 scales dropped)")


def switch_fprop(model, global_step, target="fp8"):
    t0 = time.time()
    if _PRECOMPILE:
        if not _READY:
            raise RuntimeError("FLUX_FPROP_SWITCH_STEP: the FP8 forward was not precompiled "
                               "(the MLPerf warmup did not run)")
        swapped, plain = _set_mode(model, True)
        _refresh_fp8_gather(model)
        switched = len(_MODS)
        how = f"precompiled; {swapped} gathers swapped to FP8, {plain} to bf16"
    else:
        switched = 0
        for module in model.modules():
            if isinstance(module, mx.MXFP4Linear) and module.config.fprop == "a6w4":
                module.config = replace(module.config, fprop=target)
                module._build_fp8_slots(module.weight.device)
                switched += 1
        torch._dynamo.reset()
        how = "dynamo reset"
    _log(f"FLUX_FPROP_SWITCH_STEP: {switched} linears a6w4 -> {target} fprop after step "
         f"{global_step} ({how}, {time.time() - t0:.2f}s)")
    return switched


def guard_eval_scales(trainer):
    """Restore the FP8 delayed scales after each validation, so no scale derived from eval
    activations reaches a training step."""
    validate = trainer.validate_loss

    def validate_loss():
        saved = dict(mx._FP8_DUAL_SCALE)
        try:
            return validate()
        finally:
            mx._FP8_DUAL_SCALE.clear()
            mx._FP8_DUAL_SCALE.update(saved)

    trainer.validate_loss = validate_loss
