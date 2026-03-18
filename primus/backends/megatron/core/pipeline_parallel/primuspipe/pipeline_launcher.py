###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import contextlib
from typing import Any, Callable, Iterator, List, Optional, Union

import torch
from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.schedules import (
    clear_embedding_activation_buffer,
    finish_embedding_wgrad_compute,
    get_tensor_shapes,
)
from megatron.core.pipeline_parallel.utils import set_streams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import get_model_config, get_model_type
from megatron.training import get_args

from primus.backends.megatron.core.pipeline_parallel.primuspipe.handlers import (
    megatron_primuspipe_handler_dict,
)
from primus.core.pipeline_parallel.handler.wgrad_handler import WGradRunningCache
from primus.core.pipeline_parallel.scheduler.schedule_table_factory import (
    produce_schedule_instance,
)
from primus.core.pipeline_parallel.scheduler.scheduler import ScheduleRunner
from primus.core.pipeline_parallel.scheduler.scheduler_node import FuncType
from primus.modules.module_utils import warning_rank_0


class PrimusPipelineParallelLauncher:
    def __init__(self):
        # init Primus pp algorithm runtime
        self.pp_size = parallel_state.get_pipeline_model_parallel_world_size()
        self.vpp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()
        if self.vpp_size is None:
            self.vpp_size = 1
        self.pp_rank = parallel_state.get_pipeline_model_parallel_rank()

        args = get_args()

        self.pp_algorithm = args.pp_algorithm

        assert self.pp_algorithm in (
            "1f1b",
            "1f1b-interleaved",
            "zero-bubble",
            "zbv-formatted",
            "v-half",
            "v-min",
        )

        if self.pp_algorithm in ("1f1b", "zero-bubble"):
            assert self.vpp_size == 1, "1f1b and zero-bubble require vpp_size to be 1"
        if self.pp_algorithm in ("zbv-formatted", "v-half", "v-min"):
            assert self.vpp_size == 2, "zbv-formatted, v-half, and v-min require vpp_size to be 2"

        self.handler_dict = megatron_primuspipe_handler_dict
        self.schedule_runner = ScheduleRunner(self.handler_dict)

        self.forward_data_store = []

        set_streams()

    def _init_check_pg_collection(self, model_type, pg_collection: Optional[ProcessGroupCollection] = None):

        if pg_collection is None:
            tp_group = parallel_state.get_tensor_model_parallel_group()
            cp_group = parallel_state.get_context_parallel_group()
            embd_group = parallel_state.get_embedding_group(check_initialized=False)
            pp_group = parallel_state.get_pipeline_model_parallel_group()
            pos_emb_group = parallel_state.get_position_embedding_group(check_initialized=False)

            pg_collection = ProcessGroupCollection()
            pg_collection.tp = tp_group
            pg_collection.cp = cp_group
            pg_collection.embd = embd_group
            pg_collection.pos_embd = pos_emb_group
            pg_collection.pp = pp_group
            pg_collection.dp_cp = parallel_state.get_data_parallel_group(
                with_context_parallel=True, partial_data_parallel=False
            )
        else:
            assert model_type != ModelType.encoder_and_decoder, (
                "encoder PP stages not yet supported when passing custom process groups. "
                "support coming soon!"
            )
            assert hasattr(pg_collection, "tp"), "pg_collection must have a tp_group"
            assert hasattr(pg_collection, "cp"), "pg_collection must have a cp_group"
            assert hasattr(pg_collection, "embd"), (
                "pg_collection must have a embd. In previous version, it is used default "
                "`parallel_state.default_embedding_ranks` to create the process group. If you are "
                "using the default process group, please use `parallel_state.get_embedding_group()` "
                "to get the process group. If you don't need explicitly set it to None."
            )
            assert hasattr(pg_collection, "pos_embd"), (
                "pg_collection must have a pos_embd. In previous version, it is used default "
                "`parallel_state.default_position_embedding_ranks` to create the process group."
                " If you are using the default process group, please use "
                "`parallel_state.get_position_embedding_group()` "
                "If you don't need pos_embd_group, you need to explicitly set it to None."
            )
            assert hasattr(pg_collection, "pp"), "pg_collection must have a pp_group"
            assert hasattr(pg_collection, "dp_cp"), "pg_collection must have a dp_cp_group"

        return pg_collection

    def _get_send_recv_tensor_shapes(
        self, config, model_type, seq_length, micro_batch_size, decoder_seq_length, pg_collection
    ):
        tp_group = pg_collection.tp
        cp_group = pg_collection.cp
        recv_tensor_shapes = get_tensor_shapes(
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=decoder_seq_length,
            config=config,
            tp_group=tp_group,
            cp_group=cp_group,
        )
        send_tensor_shapes = get_tensor_shapes(
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=decoder_seq_length,
            config=config,
            tp_group=tp_group,
            cp_group=cp_group,
        )

        return recv_tensor_shapes, send_tensor_shapes

    def run(
        self,
        *,
        forward_step_func,
        data_iterator: Union[Iterator, List[Iterator]],
        model: Union[torch.nn.Module, List[torch.nn.Module]],
        num_microbatches: int,
        seq_length: int,
        micro_batch_size: int,
        decoder_seq_length: Optional[int] = None,
        forward_only: bool = False,
        collect_non_loss_data: bool = False,
        first_val_step: Optional[bool] = None,
        adjust_tensor_shapes_fn: Optional[Callable] = None,
        p2p_communicator: Optional[P2PCommunicator] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        kwargs = {}
        if self.pp_algorithm == "zbv-formatted":
            kwargs["combined_forward_backward"] = get_args().overlap_moe_expert_parallel_comm

        offload = get_args().offload
        if self.pp_algorithm in ("zbv-formatted", "v-half", "v-min"):
            kwargs["offload"] = offload
        else:
            assert not offload, f"offload is not supported for {self.pp_algorithm} pp algorithm"

        self.schedule_instance = produce_schedule_instance(
            self.pp_algorithm, self.pp_size, self.vpp_size, num_microbatches, **kwargs
        )

        if not hasattr(self.schedule_instance, "schedule_table"):
            self.schedule_table = self.schedule_instance.generate_schedule_table()
            setattr(self.schedule_instance, "schedule_table", self.schedule_table)
        else:
            self.schedule_table = getattr(self.schedule_instance, "schedule_table")

        self.last_pp_stage_rank = self.schedule_instance.last_pp_stage_rank()

        if parallel_state.get_pipeline_model_parallel_rank() == 0 and get_args().debug_scheduler_table:
            self.schedule_instance.print_schedule_table(self.schedule_table)

        assert not forward_only, "forward_only is not supported yet"
        assert adjust_tensor_shapes_fn is None, "adjust_tensor_shapes_fn is not supported yet"
        if not isinstance(model, list):
            model = [model]
        config = get_model_config(model[0])
        model_type = get_model_type(model[0])
        pg_collection = self._init_check_pg_collection(model_type, pg_collection)

        if config.overlap_p2p_comm and config.batch_p2p_comm:
            raise ValueError("Can not use both overlap_p2p_comm and batch_p2p_comm")

        if p2p_communicator is not None:
            warning_rank_0("p2p_communicator is provided, but PrimusPipelineParallelLauncher will not use it")

        # Needed only when gradients are finalized in M-Core
        if config.finalize_model_grads_func is not None and not forward_only:
            # vp is ignored for clear_embedding_activation_buffer
            embedding_module = clear_embedding_activation_buffer(
                config, model, self.schedule_instance.last_pp_stage_rank() == self.pp_rank
            )

        # Disable async grad reductions
        no_sync_func = config.no_sync_func
        if isinstance(no_sync_func, list):

            def multi_no_sync():
                stack = contextlib.ExitStack()
                for model_chunk_no_sync_func in config.no_sync_func:
                    stack.enter_context(model_chunk_no_sync_func())
                return stack

            no_sync_func = multi_no_sync

        if no_sync_func is None:
            no_sync_func = contextlib.nullcontext
        no_sync_context = None

        if config.grad_sync_func is not None and not isinstance(config.grad_sync_func, list):
            config.grad_sync_func = [config.grad_sync_func for _ in model]

        if config.param_sync_func is not None and not isinstance(config.param_sync_func, list):
            config.param_sync_func = [config.param_sync_func for _ in model]

        if config.timers is not None:
            config.timers("forward-backward", log_level=1).start(barrier=config.barrier_with_L1_time)

        def disable_grad_sync():
            """Disable asynchronous grad reductions"""
            nonlocal no_sync_context
            if no_sync_context is None:
                no_sync_context = no_sync_func()
                no_sync_context.__enter__()

        def enable_grad_sync():
            """Enable asynchronous grad reductions"""
            nonlocal no_sync_context
            if no_sync_context is not None:
                no_sync_context.__exit__(None, None, None)
                no_sync_context = None

        disable_grad_sync()

        recv_tensor_shapes, send_tensor_shapes = self._get_send_recv_tensor_shapes(
            config, model_type, seq_length, micro_batch_size, decoder_seq_length, pg_collection
        )

        total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")
        for node in self.schedule_table[self.pp_rank]:
            if node.args is None:
                node.args = dict[str, Any]()

            if node.meta is None:
                node.meta = dict[str, Any]()

            node.meta["pp_size"] = self.pp_size
            node.meta["vpp_size"] = self.vpp_size
            node.meta["last_pp_stage_rank"] = self.last_pp_stage_rank
            node.meta["pp_rank"] = self.pp_rank
            node.args["config"] = config

            if node.func_type == FuncType.F:
                node.args["forward_step_func"] = forward_step_func
                node.args["data_iterator"] = data_iterator
                node.args["models"] = model
                node.args["num_microbatches"] = num_microbatches
                node.args["forward_data_store"] = self.forward_data_store
                node.args["cp_group_size"] = pg_collection.cp.size()
                node.args["collect_non_loss_data"] = collect_non_loss_data
                node.args["is_last_stage"] = parallel_state.is_pipeline_last_stage()
                node.args["total_num_tokens"] = total_num_tokens
                node.args["recv_tensor_shapes"] = recv_tensor_shapes
            elif node.func_type in [FuncType.B, FuncType.BW]:
                node.args["model_type"] = model_type
                node.args["send_tensor_shapes"] = send_tensor_shapes
            elif node.func_type in [FuncType.RF, FuncType.RB]:
                node.args["recv_tensor_shapes"] = recv_tensor_shapes
                node.args["dtype"] = config.pipeline_dtype
                node.args["pp_group"] = pg_collection.pp
            elif node.func_type in [FuncType.SF, FuncType.SB]:
                node.args["send_tensor_shapes"] = send_tensor_shapes
                node.args["pp_group"] = pg_collection.pp

        self.schedule_runner.run(self.schedule_table, self.pp_rank)

        # Launch any remaining grad reductions
        if no_sync_context is not None:
            enable_grad_sync()

        if config.finalize_model_grads_func is not None and not forward_only:
            # If defer_embedding_wgrad_compute is enabled we need to do the
            # weight gradient GEMM's here.
            finish_embedding_wgrad_compute(
                config,
                embedding_module,
                self.schedule_instance.last_pp_stage_rank() == self.pp_rank,
                pg_collection.tp,
            )

            # Finalize model grads (perform full grad all-reduce / reduce-scatter for
            # data parallelism, layernorm all-reduce for sequence parallelism, and
            # embedding all-reduce for pipeline parallelism).
            config.finalize_model_grads_func(
                model,
                total_num_tokens if config.calculate_per_token_loss else None,
                pg_collection=pg_collection,
            )

        assert WGradRunningCache.is_empty(), "WGradRunningCache is not empty"

        return self.forward_data_store
