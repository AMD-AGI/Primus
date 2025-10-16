###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""megatron utils"""

import inspect
import json
import os

import megatron
import torch
from megatron.core import parallel_state

from primus.core.utils import logger

_GLOBAL_PP_VIS_EVENTS = []
_GLOBAL_PP_VIS_EVENTS_PER_ITER = None


######################################################log after torch distributed initialized
def is_last_rank():
    return torch.distributed.get_rank() == (torch.distributed.get_world_size() - 1)


def print_rank_last(msg):
    """If distributed is initialized, print only on last rank."""
    log_func = logger.info_with_caller

    caller = inspect.stack()[1]
    caller_frame = caller.frame
    function_name = caller_frame.f_code.co_name
    module_name = caller_frame.f_globals["__name__"].split(".")[-1]
    line = caller.lineno

    if torch.distributed.is_initialized():
        if is_last_rank():
            log_func(msg, module_name, function_name, line)
    else:
        log_func(msg, module_name, function_name, line)


def set_wandb_writer_patch(args):  # monkey patch
    """
    This function is adapted from the original Megatron implementation, with an additional
    wandb argument `entity` be added.
    Monkey-patch note:
    - The original function will be replaced at runtime by this implementation.

    """

    megatron.training.global_vars._ensure_var_is_not_initialized(
        megatron.training.global_vars._GLOBAL_WANDB_WRITER, "wandb writer"
    )

    if getattr(args, "wandb_project", "") and args.rank == (args.world_size - 1):
        if args.wandb_exp_name == "":
            raise ValueError("Please specify the wandb experiment name!")

        import wandb

        if args.wandb_save_dir:
            save_dir = args.wandb_save_dir
        else:
            # Defaults to the save dir.
            save_dir = os.path.join(args.save, "wandb")
        wandb_kwargs = {
            "dir": save_dir,
            "name": args.wandb_exp_name,
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "config": vars(args),
        }
        os.makedirs(wandb_kwargs["dir"], exist_ok=True)
        wandb.init(**wandb_kwargs)
        megatron.training.global_vars._GLOBAL_WANDB_WRITER = wandb


def validate_manual_split(args):
    """
    The use of decoder_pipeline_manual_split_list is to relax the divisibility
    restriction of the current (interleaved) 1f1b pipeline schedule. The layer
    split or number of each pp rank is
    decoder_pipeline_manual_split_list[pp_rank*vp_size:(pp_rank+1)*vp_size] or
    decoder_pipeline_manual_split_list[pp_rank] when interleaved pipeline is
    used or not. For example, the split list could be "[2,3,2,2,2,2,2,1]"
    in layer16-pp4-vpp2 config, where the vpp split of
    pp_rank0/pp_rank1/pp_rank2/pp_rank3 is [2,3]/[2,2]/[2,2]/[2,1].

    """

    if (
        args.num_layers_per_virtual_pipeline_stage is not None
        or args.decoder_first_pipeline_num_layers is not None
        or args.decoder_last_pipeline_num_layers is not None
        or args.account_for_embedding_in_pipeline_split
        or args.account_for_loss_in_pipeline_split
    ):
        raise ValueError(
            "decoder_pipeline_manual_split_list is not compatible "
            "with num_layers_per_virtual_pipeline_stage/"
            "decoder_first_pipeline_num_layers/"
            "decoder_last_pipeline_num_layers/"
            "account_for_embedding_in_pipeline_split/"
            "account_for_loss_in_pipeline_split yet"
        )

    num_layers = args.num_layers
    pp_size = args.pipeline_model_parallel_size
    vp_size = args.virtual_pipeline_model_parallel_size
    pp_split = args.decoder_pipeline_manual_split_list

    if pp_size <= 1:
        raise ValueError(
            f"pipeline_model_parallel_size={pp_size} should be larger "
            f"than 1 when decoder_pipeline_manual_split_list is used"
        )

    if not isinstance(pp_split, list):
        raise ValueError(f"decoder_pipeline_manual_split_list={pp_split} should be a list")

    split_size = pp_size if vp_size is None else pp_size * vp_size
    if len(pp_split) != split_size:
        raise ValueError(
            f"the size of decoder_pipeline_manual_split_list="
            f"{pp_split} should be {split_size} "
            f"given pipeline_model_parallel_size={pp_size} and "
            f"virtual_pipeline_model_parallel_size={vp_size}"
        )

    if not all(x > 0 for x in pp_split):
        raise ValueError(
            f"layer numbers in decoder_pipeline_manual_split_list={pp_split} should all be larger than 0"
        )

    if sum(pp_split) != num_layers:
        raise ValueError(
            f"the sum of decoder_pipeline_manual_split_list="
            f"{pp_split} is {sum(pp_split)} and "
            f"should be equal to num_layers={num_layers}"
        )

    return True


def validate_args_modified(*args, **kwargs):
    def validate_args_modifier(func, modification):
        import inspect

        source = inspect.getsource(func)
        modified_source = modification(source)
        namespace = {}
        exec(modified_source, func.__globals__, namespace)
        return namespace[func.__name__]

    ori_code = (
        "if args.decoder_first_pipeline_num_layers is None and args.decoder_last_pipeline_num_layers is None:"
    )
    new_code = "if args.decoder_pipeline_manual_split_list is None and " + ori_code.split("if ")[-1]
    megatron.training.arguments.validate_args = validate_args_modifier(
        megatron.training.arguments.validate_args, lambda s: s.replace(ori_code, new_code)
    )
    megatron.training.arguments.validate_args(*args, **kwargs)


def set_manual_pipeline_split_patch(args):
    """
    Monkey-patch note:
    - The original function will be replaced at runtime by this implementation.

    """

    megatron.core.transformer.TransformerConfig.decoder_pipeline_manual_split_list = (
        args.decoder_pipeline_manual_split_list
    )

    # patch get_num_layers_to_build
    def get_num_layers_to_build_patch(config, vp_stage):
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        vp_size = config.virtual_pipeline_model_parallel_size
        pp_idx = pp_rank if vp_size is None else pp_rank * vp_size + vp_stage
        num_layers_to_build = config.decoder_pipeline_manual_split_list[pp_idx]
        return num_layers_to_build

    megatron.core.transformer.transformer_block.get_num_layers_to_build = get_num_layers_to_build_patch
    megatron.core.models.gpt.gpt_layer_specs.get_num_layers_to_build = get_num_layers_to_build_patch

    # patch get_transformer_layer_offset
    def get_transformer_layer_offset_patch(config, vp_stage):
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        pp_size = config.pipeline_model_parallel_size
        vp_size = config.virtual_pipeline_model_parallel_size

        if not parallel_state.is_inside_encoder():
            pp_decoder_start = parallel_state.get_pipeline_model_parallel_decoder_start()
            if pp_decoder_start is not None:
                pp_rank = pp_rank - pp_decoder_start

        offset = 0
        if vp_stage is not None:
            for vp_idx in range(vp_stage):
                for pp_idx in range(pp_size):
                    offset += config.decoder_pipeline_manual_split_list[pp_idx * vp_size + vp_idx]
            for pp_idx in range(pp_rank):
                offset += config.decoder_pipeline_manual_split_list[pp_idx * vp_size + vp_stage]
        else:
            offset = sum(config.decoder_pipeline_manual_split_list[:pp_rank])
        return offset

    megatron.core.transformer.transformer_layer.get_transformer_layer_offset = (
        get_transformer_layer_offset_patch
    )
    megatron.core.transformer.transformer_block.get_transformer_layer_offset = (
        get_transformer_layer_offset_patch
    )
    megatron.core.models.gpt.gpt_layer_specs.get_transformer_layer_offset = get_transformer_layer_offset_patch


def pp_warmup(args, config, model, optimizer):
    for model_chunk in model:
        with model_chunk.no_sync():
            if model_chunk.use_forward_hook:
                model_chunk.disable_forward_pre_hook()
            dtype = torch.float32
            if config.bf16:
                dtype = torch.bfloat16
            elif config.fp16:
                dtype = torch.float16
            seq_len = args.seq_length // args.tensor_model_parallel_size // args.context_parallel_size

            for layer in model_chunk.module.module.decoder.layers:
                attn_input = torch.randn(seq_len, 1, config.hidden_size, device="cuda", dtype=dtype)
                attention_mask = (
                    torch.tril(torch.ones((seq_len, seq_len), device="cuda")).unsqueeze(0).unsqueeze(0) == 0
                )
                attn_output = layer.self_attention(attn_input, attention_mask=attention_mask)
                attn_output[0].backward(torch.ones_like(attn_output[0]))

                mlp_input = torch.randn(seq_len, 1, config.hidden_size, device="cuda", dtype=dtype)
                mlp_output = layer.mlp(mlp_input)
                mlp_output[0].backward(torch.ones_like(mlp_output[0]))

            if model_chunk.use_forward_hook:
                model_chunk.enable_forward_pre_hook()
            optimizer.zero_grad()
    torch.cuda.empty_cache()


def schedule_wrapper(func):
    def wrapper(*args, **kwargs):
        global _GLOBAL_PP_VIS_EVENTS_PER_ITER
        _GLOBAL_PP_VIS_EVENTS_PER_ITER = {
            "start": None,
            "end": None,
            "memory": None,
            "fwd_start": [],
            "fwd_end": [],
            "fwd_minibatch": [],
            "fwd_chunk": [],
            "bwd_start": [],
            "bwd_end": [],
            "bwd_minibatch": [],
            "bwd_chunk": [],
            "wgrad_start": [],
            "wgrad_end": [],
            "wgrad_minibatch": [],
            "wgrad_chunk": [],
        }

        _GLOBAL_PP_VIS_EVENTS_PER_ITER["start"] = torch.cuda.Event(enable_timing=True)
        _GLOBAL_PP_VIS_EVENTS_PER_ITER["start"].record()
        res = func(*args, **kwargs)
        _GLOBAL_PP_VIS_EVENTS_PER_ITER["end"] = torch.cuda.Event(enable_timing=True)
        _GLOBAL_PP_VIS_EVENTS_PER_ITER["end"].record()

        _GLOBAL_PP_VIS_EVENTS_PER_ITER["memory"] = torch.cuda.max_memory_reserved() / 1024**3

        global _GLOBAL_PP_VIS_EVENTS
        _GLOBAL_PP_VIS_EVENTS.append(_GLOBAL_PP_VIS_EVENTS_PER_ITER)

        return res

    return wrapper


def fwd_bwd_wrapper(func, mode, minibatch=None, chunk=None):
    def wrapper(*args, **kwargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        res = func(*args, **kwargs)
        end.record()

        global _GLOBAL_PP_VIS_EVENTS_PER_ITER
        _GLOBAL_PP_VIS_EVENTS_PER_ITER[mode + "_start"].append(start)
        _GLOBAL_PP_VIS_EVENTS_PER_ITER[mode + "_end"].append(end)

        if minibatch is not None:
            _GLOBAL_PP_VIS_EVENTS_PER_ITER[mode + "_minibatch"].append(minibatch)
        if chunk is not None:
            _GLOBAL_PP_VIS_EVENTS_PER_ITER[mode + "_chunk"].append(chunk)
        return res

    return wrapper


def set_dump_pp_data_patch():
    from megatron.core.pipeline_parallel import schedules

    schedules.forward_step = fwd_bwd_wrapper(schedules.forward_step, "fwd")
    schedules.backward_step = fwd_bwd_wrapper(schedules.backward_step, "bwd")


def dump_pp_data(args, num_mbs, pp_data_dir):
    torch.cuda.synchronize()

    global _GLOBAL_PP_VIS_EVENTS
    all_iter_data = {}
    for iter_idx, iter_events in enumerate(_GLOBAL_PP_VIS_EVENTS):
        iter_data = {
            "total": None,
            "memory": None,
            "fwd_start": [],
            "fwd_end": [],
            "fwd_minibatch": [],
            "fwd_chunk": [],
            "bwd_start": [],
            "bwd_end": [],
            "bwd_minibatch": [],
            "bwd_chunk": [],
            "wgrad_start": [],
            "wgrad_end": [],
            "wgrad_minibatch": [],
            "wgrad_chunk": [],
        }
        iter_data["total"] = iter_events["start"].elapsed_time(iter_events["end"])
        iter_data["memory"] = iter_events["memory"]

        for i in range(len(iter_events["fwd_start"])):
            for key in ["fwd_start", "fwd_end", "bwd_start", "bwd_end", "wgrad_start", "wgrad_end"]:
                if i >= len(iter_events[key]):
                    continue
                event_time = iter_events["start"].elapsed_time(iter_events[key][i])
                iter_data[key].append(event_time)
            for key in [
                "fwd_minibatch",
                "fwd_chunk",
                "bwd_minibatch",
                "bwd_chunk",
                "wgrad_minibatch",
                "wgrad_chunk",
            ]:
                if i >= len(iter_events[key]):
                    continue
                iter_data[key].append(iter_events[key][i])

        all_iter_data[iter_idx + 1] = iter_data

    rank = torch.distributed.get_rank()
    dp_rank = parallel_state.get_data_parallel_rank()
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    os.makedirs(pp_data_dir, exist_ok=True)
    if dp_rank == 0:
        log_path = os.path.join(pp_data_dir, f"pp_rank_{pp_rank}.json")
        with open(log_path, "w") as f:
            json.dump(all_iter_data, f, indent=2)

    if rank == 0:
        vp_size = args.virtual_pipeline_model_parallel_size
        vp_size = 1 if vp_size is None else vp_size
        config_dict = {
            "world_size": args.world_size,
            "dp_size": args.data_parallel_size,
            "tp_size": args.tensor_model_parallel_size,
            "ep_size": args.expert_model_parallel_size,
            "pp_size": args.pipeline_model_parallel_size,
            "vp_size": vp_size,
            "num_mbs": num_mbs,
            "train_iters": args.train_iters,
        }
        log_path = os.path.join(pp_data_dir, f"config.json")
        with open(log_path, "w") as f:
            json.dump(config_dict, f, indent=2)


def _get_sync_free_moe_options(stage: int) -> dict:
    if stage > 3 or stage < 0:
        raise ValueError("turbo_sync_free_moe_stage only support [0-3]")

    sync_free_moe = {
        1: {"moe_use_fused_router_with_aux_score": True, "moe_permute_fusion": True},
        2: {
            "moe_use_fused_router_with_aux_score": True,
            "use_turbo_deepep": True,
            "moe_permute_fusion": True,
            "use_turbo_grouped_mlp": True,
        },
    }

    return sync_free_moe[stage]


def validate_args_on_rocm(args):
    # Deterministic mode
    if args.deterministic_mode:
        assert not args.moe_grouped_gemm, "MoE Grouped GEMM can't be used in deterministic mode."

    # Turbo FP8 linear check
    if args.fp8 and args.use_turbo_parallel_linear:
        support_fp8_recipe = ["tensorwise", "blockwise"]
        assert (
            args.fp8_recipe in support_fp8_recipe
        ), f"{args.fp8_recipe} recipe is not support when enable `use_turbo_parallel_linear`."

    # dump pp data
    if args.dump_pp_data and args.pipeline_model_parallel_size == 1:
        args.dump_pp_data = False
        print_rank_last(f"Disable args.dump_pp_data since args.pipeline_model_parallel_size=1")

    # sync-free MoE
    if args.turbo_sync_free_moe_stage > 0:
        assert args.enable_primus_turbo, "use `enable_primus_turbo=True` to enable sync-free MoE."

        if args.turbo_sync_free_moe_stage > 2:
            raise NotImplementedError("not support fully sync-free moe.")

        options = _get_sync_free_moe_options(args.turbo_sync_free_moe_stage)
        print_rank_last(
            f"========== Enable Sync-Free MoE Stage {args.turbo_sync_free_moe_stage} (Auto-Enabled Options) =========="
        )
        for flag, value in options.items():
            print_rank_last(f"{flag}={value}")
            setattr(args, flag, value)
        print_rank_last(
            f"========== Enable Sync-Free MoE Stage {args.turbo_sync_free_moe_stage} (Auto-Enabled Options) =========="
        )

    # turbo deepep
    if args.use_turbo_deepep:
        assert not args.moe_shared_expert_overlap, "MoE Shared Expert Overlap can't be used with deepep."
        assert (
            args.moe_router_dtype == "fp32"
        ), "DeepEP only supports float32 probs, please set --moe-router-dtype=fp32"
        if args.expert_model_parallel_size >= 16:
            assert args.turbo_deepep_num_cu <= 32, "Set `turbo_deepep_num_cu<=32` when using ep_size >= 16."
