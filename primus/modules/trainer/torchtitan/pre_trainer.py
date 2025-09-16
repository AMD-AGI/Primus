###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional

from primus.core.utils.yaml_utils import nested_namespace_to_dict
from primus.modules.base_module import BaseModule


class TorchTitanPretrainTrainer(BaseModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # important: make sure patch torchtitan logger first
        self.patch_torchtitan_logger()

        from torchtitan.config_manager import JobConfig
        from torchtitan.train import Trainer

        self.TrainerClass = Trainer
        self.JobConfigClass = JobConfig

        self.primus_cfg = kwargs.pop("primus_config", None)
        if self.primus_cfg is None:
            raise ValueError("primus_config is required")

        pre_trainer_cfg = self.primus_cfg.get_module_config("pre_trainer")
        cfg_dict = nested_namespace_to_dict(pre_trainer_cfg)

        self.titan_config = self.build_job_config(cfg_dict, self.JobConfigClass)
        self.log_config(self.titan_config)
        self.trainer = None

        if hasattr(self.titan_config, "primus_turbo") and self.titan_config.primus_turbo.enable_primus_turbo:
            self.enable_primus_turbo_extension()
        else:
            self.patch_torch_async_tp()

    def setup(self):
        pass

    def init(self, *init_args, **kwargs):
        self.trainer = self.TrainerClass(self.titan_config)

    def run(self, *args, **kwargs):
        if self.trainer is None:
            raise RuntimeError("Trainer has not been initialized. Call init() first.")
        self.trainer.train()

    def patch_torchtitan_logger(self):
        from primus.core.utils.logger import _logger as primus_logger

        primus_logger.info("Mokey patch torchtitan logger...")

        import torchtitan.tools.logging as titan_logging

        titan_logging.logger = primus_logger
        titan_logging.init_logger = lambda: None

    def enable_primus_turbo_extension(self):
        """
        Enable Primus-Turbo features and extensions.
        """
        try:
            import primus_turbo  # noqa: F401
        except ImportError:
            raise ImportError("Module 'primus_turbo' is not installed. Please install it")

        # ******* Model Converters Container *******
        import torchtitan.protocols.model_converter

        from primus.backends.torchtitan.protocols.model_converter import (
            ModelConvertersContainer,
        )

        torchtitan.protocols.model_converter.ModelConvertersContainer = ModelConvertersContainer

        # ******* llama3 Attention Model *******
        import torchtitan.models.llama3.model

        from primus.backends.torchtitan.models.llama3.model import Attention

        torchtitan.models.llama3.model.Attention = Attention

        # ******* MXLinear *******
        import torchtitan.components.quantization.mx
        from torchtitan.protocols.model_converter import _registry_model_converter_cls

        from primus.backends.torchtitan.components.quantization.mx import (
            PrimusTubroMXConverter,
        )

        _registry_model_converter_cls["mx"] = PrimusTubroMXConverter
        torchtitan.components.quantization.mx.MXConverter = PrimusTubroMXConverter

        # ******* Async TP *******
        self.patch_torch_async_tp()

    def patch_torch_async_tp(self):
        import torch
        import torch.distributed._symmetric_memory as symm_module
        import torch.distributed.distributed_c10d as c10d
        from torchtitan.tools.logging import logger

        if not self.titan_config.parallelism.enable_async_tensor_parallel:
            return

        try:
            import primus_turbo.pytorch as pt

            from primus.backends.torchtitan.tools.utils import get_backend_stream

            def _fused_all_gather_matmul_impl(
                mm_out_op: torch._ops.OpOverload,
                A_shard: torch.Tensor,
                Bs: list[torch.Tensor],
                A_scale: Optional[torch.Tensor],
                kwargs_list: list[dict[str, Any]],
                out_dtypes: list[Optional[torch.dtype]],
                gather_dim: int,
                group_name: str,
                return_A: bool,
            ) -> tuple[Optional[torch.Tensor], list[torch.Tensor]]:
                assert A_scale is None, "fused_all_gather_matmul not support for fp8"

                layouts = ["NN" for _ in range(len(Bs))]
                group = c10d._resolve_process_group(group_name)
                gemm_streams = [torch.cuda.current_stream()]
                comm_streams = get_backend_stream(size=group.size() - 1, priority=0, prefix="comm")

                copy_streams = get_backend_stream(size=1, priority=0, prefix="copy")
                A, outputs = pt.ops.fused_all_gather_matmul(
                    A_shard,
                    Bs,
                    layouts,
                    gather_dim=gather_dim,
                    group_name=group_name,
                    gemm_streams=gemm_streams,
                    comm_streams=comm_streams,
                    copy_streams=copy_streams,
                    comm_method="pipeline",
                    num_splits=4,
                    return_A=return_A,
                    out_dtypes=out_dtypes,
                )

                return A, outputs

            def _fused_matmul_reduce_scatter_impl(
                mm_out_op: torch._ops.OpOverload,
                A: torch.Tensor,
                B: torch.Tensor,
                kwargs: dict[str, Any],
                out_dtype: Optional[torch.dtype],
                reduce_op: str,
                scatter_dim: int,
                group_name: str,
            ) -> torch.Tensor:
                comm_method = "pipeline"
                group = c10d._resolve_process_group(group_name)
                # Move the scatter_dim to the front and flatten the tensor into a 2D matrix
                if comm_method == "pipeline":
                    gemm_streams = [torch.cuda.current_stream()]
                    comm_streams = get_backend_stream(size=group.size(), priority=0, prefix="comm")
                elif comm_method == "tile":
                    gemm_streams = []
                    comm_streams = []
                else:
                    raise ValueError(f"Only pipeline and tile supported, but {comm_method} provided")

                rs_output = pt.ops.fused_matmul_reduce_scatter(
                    A,
                    B,
                    layout="NN",
                    reduce_op=reduce_op,
                    scatter_dim=scatter_dim,
                    group_name=group_name,
                    gemm_streams=gemm_streams,
                    comm_streams=comm_streams,
                    comm_method=comm_method,
                    num_splits=4,
                    out_dtype=out_dtype,
                )
                return rs_output.contiguous()

            symm_module._fused_all_gather_matmul_impl = _fused_all_gather_matmul_impl
            symm_module._fused_matmul_reduce_scatter_impl = _fused_matmul_reduce_scatter_impl
            logger.warning(f"TorchtitanPretrainTrainer: Patch Async TP")

        except ImportError as e:
            logger.warning(f"TorchtitanPretrainTrainer: Patch Async TP failed - {e}")

    def flatten_config(self, obj: Any, prefix: str = "") -> Dict[str, Any]:
        flat_dict = {}
        if is_dataclass(obj):
            obj = asdict(obj)

        if isinstance(obj, dict):
            for key, value in obj.items():
                full_key = f"{prefix}.{key}" if prefix else key
                if is_dataclass(value) or isinstance(value, dict):
                    flat_dict.update(self.flatten_config(value, full_key))
                else:
                    flat_dict[full_key] = value
        else:
            flat_dict[prefix] = obj

        return flat_dict

    def log_config(self, obj: Any, header: str = "TorchTitan Config"):
        from torchtitan.tools.logging import logger

        logger.info("========== %s ==========" % header)
        flat = self.flatten_config(obj)
        max_key_len = max(len(k) for k in flat.keys())
        for key in sorted(flat):
            val = flat[key]
            formatted_line = f"arguments {key.ljust(max_key_len, '.')} {val}"
            logger.info(formatted_line)

    def build_job_config(self, cfg_dict: dict, JobConfigType) -> Any:
        import importlib

        from torchtitan.tools.logging import logger

        from third_party.torchtitan.torchtitan.config_manager import Experimental

        # Step 1: Parse the experimental section to check for a custom JobConfig extension
        experimental_cfg = cfg_dict.get("experimental", {})
        experimental = Experimental(**experimental_cfg)

        # Step 2: If a custom_args_module is defined, import and merge with JobConfig
        custom_job_config_cls = JobConfigType
        if experimental and getattr(experimental, "custom_args_module", None):
            try:
                module = importlib.import_module(experimental.custom_args_module)
                ExtendedJobConfig = getattr(module, "JobConfig")
                custom_job_config_cls = self.merge_configs(JobConfigType, ExtendedJobConfig)
                logger.info(f"Loaded and merged custom JobConfig from {experimental.custom_args_module}")
            except Exception as e:
                logger.warning(f"Failed to load custom_args_module '{experimental.custom_args_module}': {e}")

        # Step 3: Parse config dict (including custom fields) into dataclass recursively
        return self._dict_to_dataclass(custom_job_config_cls, cfg_dict)

    @staticmethod
    def merge_configs(base_cls, custom_cls):
        """
        Merges two dataclass types into one unified dataclass.

        Merge logic:
        - If a field exists in both:
            - If both fields are dataclasses, recursively merge them.
            - Otherwise, the custom field overrides the base.
        - Fields only in base or only in custom are included as-is.
        """
        from dataclasses import field, fields, make_dataclass

        base_fields = {f.name: f for f in fields(base_cls)}
        custom_fields = {f.name: f for f in fields(custom_cls)}

        merged = []

        # Merge overlapping and base-only fields
        for name, base_f in base_fields.items():
            if name in custom_fields:
                custom_f = custom_fields[name]
                if is_dataclass(base_f.type) and is_dataclass(custom_f.type):
                    merged_type = TorchTitanPretrainTrainer.merge_configs(base_f.type, custom_f.type)
                    merged.append((name, merged_type, field(default_factory=merged_type)))
                else:
                    merged.append((name, custom_f.type, custom_f))
            else:
                merged.append((name, base_f.type, base_f))

        # Add custom-only fields
        for name, custom_f in custom_fields.items():
            if name not in base_fields:
                merged.append((name, custom_f.type, custom_f))

        return make_dataclass(f"Merged{base_cls.__name__}", merged, bases=(base_cls,))

    def _dict_to_dataclass(self, cls, data: dict[str, Any]) -> Any:
        """Recursively convert dictionary to dataclass, handling nested and custom fields."""
        from dataclasses import fields, is_dataclass

        if not is_dataclass(cls):
            return data

        init_values = {}
        for f in fields(cls):
            if f.name in data:
                val = data[f.name]
                if is_dataclass(f.type) and isinstance(val, dict):
                    init_values[f.name] = self._dict_to_dataclass(f.type, val)
                else:
                    init_values[f.name] = val
        return cls(**init_values)
