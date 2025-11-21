###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


import os
from argparse import ArgumentParser, Namespace
from types import SimpleNamespace

from primus.core.utils import checker
from primus.core.utils.env import get_torchrun_env
from primus.modules.module_utils import log_kv_rank_0, log_rank_0, warning_rank_0


class MegatronLauncherAdapter:
    def __init__(self, config: SimpleNamespace, exp_root_path: str, exp_meta_info: str):
        self.config = config
        self.exp_root_path = exp_root_path
        self.exp_meta_info = exp_meta_info

    def apply_all(self):
        self._patch_args()
        self._patch_parse_args()
        self._patch_megatron_runtime_hooks()
        self._patch_flops_calculator()

    def _patch_parse_args(self):
        self._parser = self._build_parser()

        flat_args = vars(self.config)

        # Patch: Replace unsupported tokenizer
        tokenizer_type = flat_args.get("tokenizer_type")
        if tokenizer_type:
            log_rank_0(
                f"[MegatronLauncherAdapter] tokenizer_type '{tokenizer_type}' "
                "replaced with 'HuggingFaceTokenizer'"
            )
            flat_args["tokenizer_type"] = "HuggingFaceTokenizer"

        self.known_args = self._filter_known_args(flat_args)

        import megatron.training.arguments as megatron_args
        import megatron.training.initialize as megatron_init

        dist_env = get_torchrun_env()
        self.known_args.world_size = dist_env["world_size"]
        self.known_args.rank = dist_env["rank"]
        self.known_args.local_rank = dist_env["local_rank"]

        patched = lambda *_, **__: self.known_args

        megatron_args.parse_args = patched
        megatron_init.parse_args = patched

        log_rank_0("[MegatronLauncherAdapter] Successfully patched Megatron parse_args")

    def _patch_megatron_runtime_hooks(self):
        # Example: skip CUDA fused kernel compilation for AMD GPUs
        import megatron.training.initialize as megatron_initialize

        megatron_initialize._compile_dependencies = lambda: log_rank_0(
            "[MegatronLauncherAdapter] Skipped Megatron _compile_dependencies()"
        )

    def _patch_flops_calculator(self):
        import megatron.training.training as megatron_training

        import primus.core.utils.flops_estimator as primus_flops

        megatron_training.num_floating_point_operations = primus_flops.num_floating_point_operations

    def _patch_args(self):
        # cuda
        if not self.config.use_torch_fsdp2 and not self.config.use_custom_fsdp:
            CUDA_DEVICE_MAX_CONNECTIONS = "1"
        else:
            CUDA_DEVICE_MAX_CONNECTIONS = "8"
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = CUDA_DEVICE_MAX_CONNECTIONS
        log_rank_0(f"[PatchArgs] set env 'CUDA_DEVICE_MAX_CONNECTIONS' to {CUDA_DEVICE_MAX_CONNECTIONS}")

        # profile
        if self.config.profile:
            self.config.disable_tensorboard = False
            log_rank_0(f"[PatchArgs] set args.disable_tensorboard to 'False'")

        # checkpoint
        ckpt_path = os.path.abspath(os.path.join(self.exp_root_path, "checkpoints"))
        if self.config.save is not None:
            warning_rank_0(f"[PatchArgs] args.save is deprecated, the checkpoint path is: {ckpt_path}")
        self.config.save = ckpt_path
        log_kv_rank_0(f"[PatchArgs] -save", f"{self.config.save}")

        # tensorboard
        if not self.config.disable_tensorboard:
            tb_path = os.path.abspath(os.path.join(self.exp_root_path, "tensorboard"))
            if self.config.tensorboard_dir is not None:
                warning_rank_0(f"args.tensorboard_dir is deprecated, the tensorboard path is: {tb_path}")
            self.config.tensorboard_dir = tb_path
        else:
            self.config.tensorboard_dir = None
        log_kv_rank_0(f"[PatchArgs] -disable_tensorboard", f"{self.config.disable_tensorboard}")
        log_kv_rank_0(f"[PatchArgs] -tensorboard_dir", f"{self.config.tensorboard_dir}")

        # wandb
        if not self.config.disable_wandb:
            wandb_path = self.exp_root_path
            if self.config.wandb_save_dir is not None:
                warning_rank_0(
                    f"[PatchArgs] args.wandb_save_dir is deprecated, the wandb path is: {wandb_path}/wandb"
                )
            if not hasattr(self.config, "wandb_project") or self.config.wandb_project is None:
                self.config.wandb_project = (
                    f"{self.exp_meta_info['work_group']}_{self.exp_meta_info['user_name']}"
                )
                warning_rank_0(f"[PatchArgs] -create new wandb project name: {self.config.wandb_project}")
            if not hasattr(self.config, "wandb_exp_name") or self.config.wandb_exp_name is None:
                self.config.wandb_exp_name = self.exp_meta_info["exp_name"]
                warning_rank_0(f"[PatchArgs] - create new exp name: {self.config.wandb_exp_name}")
            self.config.wandb_save_dir = wandb_path
        elif self.config.wandb_project is not None:
            self.config.wandb_project = None
            warning_rank_0(f"[PatchArgs] args.wandb_project is disabled, as args.disable_wandb=True.")
        log_kv_rank_0(f"[PatchArgs] -disable_wandb", f"{self.config.disable_wandb}")
        if not self.config.disable_wandb and "WANDB_API_KEY" not in os.environ:
            warning_rank_0(
                "The environment variable WANDB_API_KEY is not set. "
                "Please set it before proceeding or enable 'disable_wandb' in yaml config"
            )
        log_kv_rank_0(f"[PatchArgs]  -wandb_project", f"{self.config.wandb_project}")
        log_kv_rank_0(f"[PatchArgs]  -wandb_exp_name", f"{self.config.wandb_exp_name}")
        log_kv_rank_0(f"[PatchArgs]  -wandb_save_dir", f"{self.config.wandb_save_dir}")
        log_kv_rank_0(f"[PatchArgs]  -wandb_entity", f"{self.config.wandb_entity}")

        # sink_level: logging_level
        level_map = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
        checker.check_true(self.config.stderr_sink_level in level_map)
        logging_level = level_map[self.config.stderr_sink_level]
        if self.config.logging_level is not None:
            warning_rank_0(
                f"[PatchArgs] -args.logging_level is deprecated, set args.logging_level={logging_level} [stderr_sink_level]"
            )
        self.config.logging_level = logging_level

        # update data path
        # "data1 data2 data3" -> ['data1', 'data2', 'data3']
        if self.config.data_path is not None:
            self.config.data_path = self.config.data_path.split(" ")
            log_rank_0(f"-data_path: {self.config.data_path}")

        if self.config.train_data_path is not None:
            self.config.train_data_path = self.config.train_data_path.split(" ")
            log_rank_0(f"[PatchArgs] -train_data_path: {self.config.train_data_path}")
        if self.config.valid_data_path is not None:
            self.config.valid_data_path = self.config.valid_data_path.split(" ")
            log_rank_0(f"[PatchArgs] -valid_data_path: {self.config.valid_data_path}")
        if self.config.test_data_path is not None:
            self.config.test_data_path = self.config.test_data_path.split(" ")
            log_rank_0(f"[PatchArgs] -test_data_path: {self.config.test_data_path}")

        if self.config.mock_data:
            self.config.data_path = None
            self.config.train_data_path = None
            self.config.valid_data_path = None
            self.config.test_data_path = None

    def _build_parser(self) -> ArgumentParser:
        from megatron.training.arguments import add_megatron_arguments

        parser = ArgumentParser(description="Megatron-LM args", allow_abbrev=False)
        parser = add_megatron_arguments(parser)
        return parser

    def _filter_known_args(self, flat_args: dict) -> Namespace:
        import enum
        from argparse import _StoreFalseAction, _StoreTrueAction

        cli_args = []

        action_map = {
            action.dest: action
            for action in self._parser._actions
            if action.option_strings  # skip positional args
        }

        for key, val in flat_args.items():
            if val is None:
                continue

            action = action_map.get(key)
            if not action:
                warning_rank_0(f"[PatchParseArgs] Unknown key ignored: {key}:{val}")
                continue

            opt = action.option_strings[0]

            # Handle booleans flags: store_true / store_false
            if isinstance(val, bool):
                if isinstance(action, _StoreTrueAction) and val is True:
                    cli_args.append(opt)
                elif isinstance(action, _StoreFalseAction) and val is False:
                    cli_args.append(opt)

            # List-type flags (nargs +/*), ignore empty lists
            elif action.nargs in ("+", "*") and isinstance(val, list):
                if val:
                    cli_args += [opt] + [str(v) for v in val]

            elif action.choices and all(isinstance(c, enum.Enum) for c in action.choices):
                try:
                    input_str = str(val).lower()
                    matched = [
                        e
                        for e in action.choices
                        if e.name.lower() == input_str or str(e.value).lower() == input_str
                    ]
                    if not matched:
                        raise ValueError(
                            f"[PatchParseArgs] Invalid enum value '{val}' for {key}, "
                            f"expected one of {[e.name for e in action.choices]}"
                        )
                    cli_args += [opt, matched[0].name]
                except Exception as e:
                    raise RuntimeError(f"[PatchParseArgs] Failed to match enum for {key}={val}") from e

            # Typed single value
            elif action.type:
                try:
                    cli_args += [opt, str(action.type(val))]
                except Exception as e:
                    raise RuntimeError(f"[PatchParseArgs] Failed to cast {key}={val} to {action.type}") from e

            # Fallback: treat as string
            else:
                cli_args += [opt, str(val)]

        known_args, unknown_args = self._parser.parse_known_args(cli_args)
        if unknown_args:
            log_rank_0(f"[PatchParseArgs] Ignored unknown args: {unknown_args}")

        return known_args
