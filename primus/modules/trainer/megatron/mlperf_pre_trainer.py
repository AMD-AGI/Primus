###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import inspect
import json
import time

import collections
from functools import partial

import torch
from megatron.training import get_args
from primus.modules.module_utils import log_rank_0
from megatron.core.num_microbatches_calculator import get_num_microbatches
from primus.core.utils.flops_estimator import num_floating_point_operations
from megatron.training.global_vars import get_timers
from megatron.training import get_timers
import statistics
from .pre_trainer import MegatronPretrainTrainer


class MLPerfMegatronPretrainTrainer(MegatronPretrainTrainer):
    """
    MLPerf-compliant trainer that extends MegatronPretrainTrainer.
    
    Adds MLCommons JSON format logging for benchmark compliance.
    """
    
    def __init__(self, *args, **kwargs):

        # Get current file and line number
        frame = inspect.currentframe()

        # init_start
        mlperf_log = {
            "namespace": "",
            "time_ms": int(time.time() * 1000),
            "event_type": "INTERVAL_START",
            "key": "init_start",
            "metadata": {
                "file_name": __file__,
                "lineno": frame.f_lineno,
            },
        }
        log_rank_0(f":::MLLOG {json.dumps(mlperf_log)}")

        super().__init__(*args, **kwargs)
        

        megatron_args = get_args()
        # Log submission info
        mlperf_log = {
            "namespace": "",
            "time_ms": int(time.time() * 1000),
            "event_type": "POINT_IN_TIME",
            "key": "submission_benchmark",
            "value": "llama3.1_8b",
            "metadata": {
                "file_name": __file__,
                "lineno": frame.f_lineno,
            },
        }
        log_rank_0(f":::MLLOG {json.dumps(mlperf_log)}")

        # submission_org
        mlperf_log = {
            "namespace": "",
            "time_ms": int(time.time() * 1000),
            "event_type": "POINT_IN_TIME",
            "key": "submission_org",
            "value": "AMD",
            "metadata": {
                "file_name": __file__,
                "lineno": frame.f_lineno,
            },            
        }
        log_rank_0(f":::MLLOG {json.dumps(mlperf_log)}")

        # submission_division
        mlperf_log = {
            "namespace": "",
            "time_ms": int(time.time() * 1000),
            "event_type": "POINT_IN_TIME",
            "key": "submission_division",
            "value": "AMD-MLPerf",
            "metadata": {
                "file_name": __file__,
                "lineno": frame.f_lineno,
            },            
        }
        log_rank_0(f":::MLLOG {json.dumps(mlperf_log)}")    

        # submission_status   
        mlperf_log = {
            "namespace": "",
            "time_ms": int(time.time() * 1000),
            "event_type": "POINT_IN_TIME",
            "key": "submission_version",
            "value": "1.0",
            "metadata": {
                "file_name": __file__,
                "lineno": frame.f_lineno,
            },            
        }
        log_rank_0(f":::MLLOG {json.dumps(mlperf_log)}")    

        # submission_platform
        mlperf_log = {
            "namespace": "",
            "time_ms": int(time.time() * 1000),
            "event_type": "POINT_IN_TIME",
            "key": "submission_platform",
            "value": "AMD-MLPerf",
            "metadata": {
                "file_name": __file__,
                "lineno": frame.f_lineno,
            },
        }
        log_rank_0(f":::MLLOG {json.dumps(mlperf_log)}")    

        # seed
        mlperf_log = {
            "namespace": "",
            "time_ms": int(time.time() * 1000),
            "event_type": "POINT_IN_TIME",
            "key": "seed",
            "value": megatron_args.seed,
            "metadata": {
                "file_name": __file__,
                "lineno": frame.f_lineno,
            },
        }
        log_rank_0(f":::MLLOG {json.dumps(mlperf_log)}")    

        # global_batch_size
        mlperf_log = {
            "namespace": "",
            "time_ms": int(time.time() * 1000),
            "event_type": "POINT_IN_TIME",
            "key": "global_batch_size",
            "value": megatron_args.micro_batch_size * megatron_args.data_parallel_size * megatron_args.num_microbatches if hasattr(megatron_args, 'num_microbatches') else megatron_args.global_batch_size,
        }
        log_rank_0(f":::MLLOG {json.dumps(mlperf_log)}")    

        # gradient_accumulation_steps
        mlperf_log = {
            "namespace": "",
            "time_ms": int(time.time() * 1000),
            "event_type": "POINT_IN_TIME",
            "key": "gradient_accumulation_steps",
            "value": megatron_args.gradient_accumulation_steps,
        }
        log_rank_0(f":::MLLOG {json.dumps(mlperf_log)}")    

        # max_sequence_length
        mlperf_log = {
            "namespace": "",
            "time_ms": int(time.time() * 1000),
            "event_type": "POINT_IN_TIME",
            "key": "max_sequence_length",
            "value": megatron_args.max_sequence_length,
        }
        log_rank_0(f":::MLLOG {json.dumps(mlperf_log)}")    

        # eval_samples
        mlperf_log = {
            "namespace": "",
            "time_ms": int(time.time() * 1000),
            "event_type": "POINT_IN_TIME",
            "key": "eval_samples",
            "value": megatron_args.eval_samples,
        }
        log_rank_0(f":::MLLOG {json.dumps(mlperf_log)}")    

        # train_samples
        mlperf_log = {
            "namespace": "",
            "time_ms": int(time.time() * 1000),
            "event_type": "POINT_IN_TIME",
            "key": "train_samples",
            "value": megatron_args.train_samples,
        }
        log_rank_0(f":::MLLOG {json.dumps(mlperf_log)}")    

        # init_checkpoint_step  
        mlperf_log = {
            "namespace": "",
            "time_ms": int(time.time() * 1000),
            "event_type": "POINT_IN_TIME",
            "key": "init_checkpoint_step",
            "value": megatron_args.init_checkpoint_step,
        }
        log_rank_0(f":::MLLOG {json.dumps(mlperf_log)}")

        # opt_name    
        mlperf_log = {
            "namespace": "",
            "time_ms": int(time.time() * 1000),
            "event_type": "POINT_IN_TIME",
            "key": "opt_name",
            "value": megatron_args.opt_name,
        }
        log_rank_0(f":::MLLOG {json.dumps(mlperf_log)}")

        # opt_base_learning_rate
        mlperf_log = {
            "namespace": "",
            "time_ms": int(time.time() * 1000),
            "event_type": "POINT_IN_TIME",
            "key": "opt_base_learning_rate",
            "value": megatron_args.opt_base_learning_rate,
        }
        log_rank_0(f":::MLLOG {json.dumps(mlperf_log)}")

        # opt_adamw_beta1
        mlperf_log = {
            "namespace": "",
            "time_ms": int(time.time() * 1000),
            "event_type": "POINT_IN_TIME",
            "key": "opt_adamw_beta1",
            "value": megatron_args.opt_adamw_beta1,
        }
        log_rank_0(f":::MLLOG {json.dumps(mlperf_log)}")

        # opt_adamw_beta2
        mlperf_log = {
            "namespace": "",
            "time_ms": int(time.time() * 1000),
            "event_type": "POINT_IN_TIME",
            "key": "opt_adamw_beta2",
            "value": megatron_args.opt_adamw_beta2,
        }
        log_rank_0(f":::MLLOG {json.dumps(mlperf_log)}")

        # opt_adamw_eps
        mlperf_log = {
            "namespace": "",
            "time_ms": int(time.time() * 1000),
            "event_type": "POINT_IN_TIME",
            "key": "opt_adamw_epsilon" ,
            "value": megatron_args.opt_adamw_epsilon,
        }
        log_rank_0(f":::MLLOG {json.dumps(mlperf_log)}")

        # opt_adamw_weight_decay
        mlperf_log = {
            "namespace": "",
            "time_ms": int(time.time() * 1000),
            "event_type": "POINT_IN_TIME",
            "key": "opt_adamw_weight_decay",
            "value": megatron_args.opt_adamw_weight_decay,
        }
        log_rank_0(f":::MLLOG {json.dumps(mlperf_log)}")

        # opt_gradient_clip_norm    
        mlperf_log = {
            "namespace": "",
            "time_ms": int(time.time() * 1000),
            "event_type": "POINT_IN_TIME",
            "key": "opt_gradient_clip_norm",
            "value": megatron_args.opt_gradient_clip_norm,
        }
        log_rank_0(f":::MLLOG {json.dumps(mlperf_log)}")

        # opt_end_learning_rate
        mlperf_log = {
            "namespace": "",
            "time_ms": int(time.time() * 1000),
            "event_type": "POINT_IN_TIME",
            "key": "opt_end_learning_rate",
            "value": megatron_args.opt_end_learning_rate,
        }
        log_rank_0(f":::MLLOG {json.dumps(mlperf_log)}")

        # opt_learning_warmup_steps
        mlperf_log = {
            "namespace": "",
            "time_ms": int(time.time() * 1000),
            "event_type": "POINT_IN_TIME",
            "key": "opt_learning_warmup_steps",
            "value": megatron_args.opt_learning_warmup_steps,
        }
        log_rank_0(f":::MLLOG {json.dumps(mlperf_log)}")

        # opt_learning_rate_decay_steps
        mlperf_log = {
            "namespace": "",
            "time_ms": int(time.time() * 1000),
            "event_type": "POINT_IN_TIME",
            "key": "opt_learning_rate_decay_steps",
            "value": megatron_args.opt_learning_rate_decay_steps,
        }
        log_rank_0(f":::MLLOG {json.dumps(mlperf_log)}")

        # opt_learning_max_steps
        mlperf_log = {
            "namespace": "",
            "time_ms": int(time.time() * 1000),
            "event_type": "POINT_IN_TIME",
            "key": "opt_learning_max_steps",
            "value": megatron_args.opt_learning_max_steps,
        }
        log_rank_0(f":::MLLOG {json.dumps(mlperf_log)}")

        # opt_learning_rate_decay_schedule
        mlperf_log = {
            "namespace": "",
            "time_ms": int(time.time() * 1000),
            "event_type": "POINT_IN_TIME",
            "key": "opt_learning_rate_decay_schedule",
            "value": megatron_args.opt_learning_rate_decay_schedule,
        }
        log_rank_0(f":::MLLOG {json.dumps(mlperf_log)}")

        # target_accuracy
        mlperf_log = {
            "namespace": "",
            "time_ms": int(time.time() * 1000),
            "event_type": "POINT_IN_TIME",
            "key": "target_accuracy",
            "value": megatron_args.target_accuracy,
        }
        log_rank_0(f":::MLLOG {json.dumps(mlperf_log)}")

        # init_stop
        mlperf_log = {
            "namespace": "",
            "time_ms": int(time.time() * 1000),
            "event_type": "INTERVAL_END",
            "key": "init_stop",
            "metadata": {
                "status": "success",
            },
        }
        log_rank_0(f":::MLLOG {json.dumps(mlperf_log)}")


    def run(self, *args, **kwargs):
        # Get args for MLPerf logging
        megatron_args = get_args()
        
        # Log training run start
        mlperf_log = {
            "namespace": "llama2_70b",  # TODO: Make this configurable
            "time_ms": int(time.time() * 1000),
            "event_type": "INTERVAL_START",
            "key": "run_start",
            "metadata": {
                "global_batch_size": megatron_args.micro_batch_size * megatron_args.data_parallel_size * megatron_args.num_microbatches if hasattr(megatron_args, 'num_microbatches') else megatron_args.global_batch_size,
                "world_size": megatron_args.world_size,
            },
        }
        log_rank_0(f":::MLLOG {json.dumps(mlperf_log)}")

        # block_start
        mlperf_log = {
            "namespace": "",
            "time_ms": int(time.time() * 1000),
            "event_type": "INTERVAL_START",
            "key": "block_start",
            "metadata": {
                "block_type": "train",
            },
        }
        log_rank_0(f":::MLLOG {json.dumps(mlperf_log)}")
        
        super().run(*args, **kwargs)

        # Log training run stop
        mlperf_log = {
            "namespace": "llama2_70b",
            "time_ms": int(time.time() * 1000),
            "event_type": "INTERVAL_END",
            "key": "run_stop",
            "metadata": {
                "status": "success",
                "final_iteration": megatron_args.iteration,   
            },
        }
        log_rank_0(f":::MLLOG {json.dumps(mlperf_log)}")
        
        return None

    def train(
        self,
        forward_step_func,
        model,
        optimizer,
        opt_param_scheduler,
        train_data_iterator,
        valid_data_iterator,
        process_non_loss_data_func,
        config,
        checkpointing_context,
        non_loss_data_func,
    ):
        """
        Override train method to add MLPerf-specific logging.
        
        This method adds MLCommons logging at key training lifecycle points:
        - run_start: Beginning of training
        - run_stop: End of training
        """
        args = get_args()
        
        # Log training run start
        mlperf_log = {
            "namespace": "llama2_70b",  # TODO: Make this configurable
            "time_ms": int(time.time() * 1000),
            "event_type": "INTERVAL_START",
            "key": "run_start",
            "metadata": {
                "global_batch_size": args.micro_batch_size * args.data_parallel_size * args.num_microbatches if hasattr(args, 'num_microbatches') else args.global_batch_size,
                "world_size": args.world_size,
            },
        }
        log_rank_0(f":::MLLOG {json.dumps(mlperf_log)}")
        
        # Log global batch size
        mlperf_log = {
            "namespace": "llama2_70b",
            "time_ms": int(time.time() * 1000),
            "event_type": "POINT_IN_TIME",
            "key": "global_batch_size",
            "value": args.micro_batch_size * args.data_parallel_size * args.num_microbatches if hasattr(args, 'num_microbatches') else args.global_batch_size,
        }
        log_rank_0(f":::MLLOG {json.dumps(mlperf_log)}")
        
        # Call parent train method
        iteration, num_floating_point_operations = super().train(
            forward_step_func,
            model,
            optimizer,
            opt_param_scheduler,
            train_data_iterator,
            valid_data_iterator,
            process_non_loss_data_func,
            config,
            checkpointing_context,
            non_loss_data_func,
        )
        
        # Log training run stop
        mlperf_log = {
            "namespace": "llama2_70b",
            "time_ms": int(time.time() * 1000),
            "event_type": "INTERVAL_END",
            "key": "run_stop",
            "metadata": {
                "status": "success",
                "final_iteration": iteration,
            },
        }
        log_rank_0(f":::MLLOG {json.dumps(mlperf_log)}")
        
        return iteration, num_floating_point_operations

    def training_log(
        self,
        loss_dict,
        total_loss_dict,
        learning_rate,
        decoupled_learning_rate,
        iteration,
        loss_scale,
        report_memory_flag,
        skipped_iter,
        grad_norm,
        params_norm,
        num_zeros_in_grad,
    ):
        """
        Override training_log to add MLPerf-specific logging.
        
        This method calls the parent's training_log to preserve all normal
        logging functionality, then adds MLCommons JSON format logs.
        """
        
        # Add MLCommons JSON format logging
        args = get_args()
        timers = get_timers()
        batch_size = args.micro_batch_size * args.data_parallel_size * get_num_microbatches()

        advanced_iters_key = "advanced iterations"
        skipped_iters_key = "skipped iterations"

        if not skipped_iter:
            total_loss_dict[advanced_iters_key] = total_loss_dict.get(advanced_iters_key, 0) + 1
        else:
            if advanced_iters_key not in total_loss_dict:
                total_loss_dict[advanced_iters_key] = 0
        # Skipped iterations.
        total_loss_dict[skipped_iters_key] = total_loss_dict.get(skipped_iters_key, 0) + skipped_iter

        total_iterations = total_loss_dict[advanced_iters_key] + total_loss_dict[skipped_iters_key]

        elapsed_time = timers("interval-time").elapsed(barrier=True)
        elapsed_time_per_iteration = elapsed_time / total_iterations

        if (
            iteration == self.log_avg_skip_iterations + 1
            or len(self.recent_iteration_times) >= self.log_avg_reset_interval
        ):
            self.recent_iteration_times.clear()
        self.recent_iteration_times.append(elapsed_time_per_iteration * 1000.0)        

        flops_calc = (
            num_floating_point_operations
            if not args.multi_latent_attention
            else self.num_floating_point_operations_mla_moe
        )
        throughput = flops_calc(args, batch_size) / (
                elapsed_time_per_iteration * 10**12 * args.world_size
            )

        log_string = " Throughput (samples/s): {:.1f}/{:.1f} |".format(
                batch_size / float(elapsed_time_per_iteration),
                batch_size / float(statistics.mean(self.recent_iteration_times) / 1000.0),
            )
        log_rank_0(f":::MLLOG {log_string}")            

        return None
