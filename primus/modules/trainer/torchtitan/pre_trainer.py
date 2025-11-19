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
        extra_args = kwargs.pop("extra_args", None)
        super().__init__(*args, **kwargs)

        # important: make sure patch torchtitan logger first
        self.patch_torchtitan_logger()

        self.primus_cfg = kwargs.pop("primus_config", None)
        if self.primus_cfg is None:
            raise ValueError("primus_config is required")

        pre_trainer_cfg = self.primus_cfg.get_module_config("pre_trainer")
        cfg_dict = nested_namespace_to_dict(pre_trainer_cfg)

        patch_mock = getattr(pre_trainer_cfg.training, "mock_data", False)
        if patch_mock:
            from primus.modules.trainer.torchtitan.patch_utils import (
                patch_mock_hf_dataset,
            )

            patch_mock_hf_dataset()

        self.patch_torchtitan_embedding_amp(cfg_dict["primus_turbo"]["enable_embedding_autocast"])
        self.patch_titan_train_spec(pre_trainer_cfg.model.name, pre_trainer_cfg.model.flavor, extra_args)

        # ensure checkpoint patch applied before import torchtitan
        # background: consolidate_safetensors_files_on_every_rank is a new DCP
        # utility introduced in newer torch versions. our current build does not
        # include it yet. this patch safely skips safetensors consolidation and
        # issues a warning so Titan checkpoints can still work normally.
        self.patch_torch_dcp_consolidate()

        # ensure ScheduleDualPipeV is available
        # background: ScheduleDualPipeV is a newer pipeline schedule recently
        # introduced in torch.distributed; our current torch build does not
        # include it yet. this patch injects a temporary alias to fall back to
        # Schedule1F1B or ScheduleGPipe so Titan imports can succeed.
        self.patch_torch_pipelining_schedules()

        # ensure AuxOutput exists in flex_attention for model imports
        # background: AuxOutput is a newly introduced optional return type in
        # torch.nn.attention.flex_attention, used for debug or profiling data
        # (e.g., attention probabilities or mask stats). our current torch build
        # does not yet include it. this patch injects a lightweight stub class
        # so model imports succeed. Titan does not rely on AuxOutput in its
        # attention or training logic, so this patch does not affect behavior.
        self.patch_torch_flex_attention_auxoutput()

        from torchtitan.config.job_config import JobConfig
        from torchtitan.train import Trainer

        self.TrainerClass = Trainer
        self.JobConfigClass = JobConfig

        self.titan_config = self.build_job_config(cfg_dict, self.JobConfigClass)
        
        # patch torchtitan moe
        # background: we use turbo grouped mm for moe, so we need to patch the torchtitan moe
        self.patch_torchtitan_moe()
        
        self.log_config(self.titan_config)
        self.trainer = None

        if hasattr(self.titan_config, "primus_turbo") and self.titan_config.primus_turbo.enable_primus_turbo:
            self.enable_primus_turbo_extension()

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
    
    def patch_torchtitan_moe(self):
        if not self.titan_config.primus_turbo.use_turbo_grouped_mm:
            return
        from primus.core.utils.logger import _logger as primus_logger
        primus_logger.info("Monkey patch torchtitan moe...")
        try:
            import functools
            import torchtitan.models.moe.moe
            from primus.backends.torchtitan.models.moe.moe import _run_experts_grouped_mm
            
            # Get MoE FP8 configuration and create a partial function
            use_moe_fp8 = self.titan_config.primus_turbo.use_moe_fp8
            primus_logger.info(f"Set MoE FP8 mode: {use_moe_fp8}")
            
            # Patch the grouped_mm function with use_fp8 parameter pre-set
            torchtitan.models.moe.moe._run_experts_grouped_mm = functools.partial(
                _run_experts_grouped_mm, use_fp8=use_moe_fp8
            )
            primus_logger.info("Successfully patched torchtitan moe with turbo grouped_mm")
        except ImportError as e:
            raise ImportError(
                f"Failed to import primus_turbo for MoE grouped_mm patch. "
                f"Please ensure primus_turbo is installed or set use_turbo_grouped_mm=False. "
                f"Original error: {e}"
            ) from e

    def patch_torch_dcp_consolidate(self):
        """
        Monkey patch for torch.distributed.checkpoint._consolidate_hf_safetensors
        when current torch build does not export consolidate_safetensors_files_on_every_rank.
        This avoids ImportError in TorchTitan when last_save_in_hf=True.
        """
        import sys
        import types
        import warnings

        mod_name = "torch.distributed.checkpoint._consolidate_hf_safetensors"
        func_name = "consolidate_safetensors_files_on_every_rank"

        try:
            mod = __import__(mod_name, fromlist=["*"])
            if hasattr(mod, func_name):
                primus_logger.info("[PrimusPatch][DCP] consolidate available, no patch needed.")
                return  # OK, torch build supports it
        except Exception:
            pass

        # Patch missing module/function
        dummy_mod = types.ModuleType(mod_name)

        def _warn_and_skip(*args, **kwargs):
            warnings.warn(
                "[PrimusPatch][DCP] Current PyTorch build does not support "
                f"{mod_name}.{func_name}; safetensors export will be skipped.",
                UserWarning,
            )
            return None

        setattr(dummy_mod, func_name, _warn_and_skip)
        sys.modules[mod_name] = dummy_mod

        from primus.core.utils.logger import _logger as primus_logger

        primus_logger.warning(
            f"[PrimusPatch][DCP] Installed fallback for missing {mod_name}.{func_name}, "
            "HuggingFace safetensors export will be disabled."
        )

    def patch_torch_pipelining_schedules(self):
        """
        Ensure torch.distributed.pipelining.schedules.ScheduleDualPipeV exists.

        If this class is missing in the current PyTorch build (common in ROCm 7.0 / 2.9),
        we create a fallback alias that inherits from Schedule1F1B or ScheduleGPipe.
        This prevents ImportError in TorchTitan pipeline modules.
        """

        from primus.core.utils.logger import _logger as primus_logger

        try:
            import torch.distributed.pipelining.schedules as sched
        except Exception as e:
            primus_logger.warning(f"[PrimusPatch][Pipe] failed to import schedules: {e}")
            return

        # Check if DualPipeV is already provided
        if hasattr(sched, "ScheduleDualPipeV"):
            primus_logger.info("[PrimusPatch][Pipe] ScheduleDualPipeV available, no patch needed.")
            return  # No patch needed

        # Pick a safe fallback
        fallback = getattr(sched, "Schedule1F1B", None) or getattr(sched, "ScheduleGPipe", None)

        if fallback is None:
            primus_logger.warning(
                "[PrimusPatch][Pipe] No pipeline schedule available; pipeline parallel may be unsupported."
            )
            return

        # Define the fallback class with identical signature
        class ScheduleDualPipeV(fallback):  # type: ignore[misc]
            def __init__(self, *args, **kwargs):
                primus_logger.warning(
                    f"[PrimusPatch][Pipe] ScheduleDualPipeV not found, using fallback {fallback.__name__}. "
                    f"This is a temporary compatibility patch; functionality may differ from the official DualPipeV."
                )
                super().__init__(*args, **kwargs)

        # Inject into torch namespace
        setattr(sched, "ScheduleDualPipeV", ScheduleDualPipeV)
        primus_logger.warning(
            f"[PrimusPatch][Pipe] Installed fallback: ScheduleDualPipeV -> {fallback.__name__}"
        )

    def patch_torch_flex_attention_auxoutput(self):
        """
        Ensure torch.nn.attention.flex_attention has an AuxOutput symbol.
        Some PyTorch builds (e.g., certain ROCm 2.9 dev builds) rename or drop it.
        We provide a safe alias so Titan's imports won't fail.
        """
        from primus.core.utils.logger import _logger as primus_logger

        try:
            import torch.nn.attention.flex_attention as flex_mod
        except Exception as e:
            primus_logger.warning(f"[PrimusPatch][FlexAttn] flex_attention import failed: {e}")
            return

        # If AuxOutput already exists, nothing to do.
        if hasattr(flex_mod, "AuxOutput"):
            primus_logger.info("[PrimusPatch][FlexAttn] AuxOutput available, no patch needed.")
            return

        primus_logger.warning(
            "[PrimusPatch][FlexAttn] AuxOutput not found. "
            "This torch build predates the new debug/profiling return type. "
            "Injecting a lightweight stub so Titan model imports can succeed."
        )

        from dataclasses import dataclass

        import torch

        @dataclass
        class _AuxOutput:
            attn_probs: torch.Tensor = torch.empty(0)
            block_mask: torch.Tensor | None = None
            stats: dict | None = None
            extra: dict | None = None

            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        setattr(flex_mod, "AuxOutput", _AuxOutput)
        primus_logger.info(
            "[PrimusPatch][FlexAttn] Injected fallback AuxOutput stub (Titan does not rely on this)."
        )

    def enable_primus_turbo_extension(self):
        """
        Enable Primus-Turbo features and extensions.
        """
        from torchtitan.tools.logging import logger

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

        if self.titan_config.primus_turbo.use_turbo_attention:
            # ******* llama3 Attention Model *******
            import torchtitan.models.llama3.model.model

            from primus.backends.torchtitan.models.llama3.model.model import Attention

            torchtitan.models.llama3.model.model.Attention = Attention
            logger.warning(f"TorchtitanPretrainTrainer: Patch Turbo Attention")

        if self.titan_config.primus_turbo.use_turbo_mx_linear:
            # ******* MXLinear *******
            import torchtitan.components.quantization.mx
            from torchtitan.protocols.model_converter import (
                _registry_model_converter_cls,
            )

            from primus.backends.torchtitan.components.quantization.mx import (
                PrimusTubroMXConverter,
            )

            _registry_model_converter_cls["mx"] = PrimusTubroMXConverter
            torchtitan.components.quantization.mx.MXConverter = PrimusTubroMXConverter
            logger.warning(f"TorchtitanPretrainTrainer: Patch Turbo MXLinear")

        if self.titan_config.primus_turbo.use_turbo_async_tp:
            # ******* Async TP *******
            self.patch_torch_async_tp()

        from primus.core.utils.logger import _logger as primus_logger

        primus_logger.info("Enable primus turbo extension...")

    def patch_torch_async_tp(self):
        import torch
        import torch.distributed._symmetric_memory as symm_module
        import torch.distributed.distributed_c10d as c10d

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

        from torchtitan.config.job_config import Experimental
        from torchtitan.tools.logging import logger

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

        # collect valid field names
        field_names = {f.name for f in fields(cls)}
        init_values = {}

        # only use known fields for constructor
        for f in fields(cls):
            if f.name in data:
                val = data[f.name]
                if is_dataclass(f.type) and isinstance(val, dict):
                    init_values[f.name] = self._dict_to_dataclass(f.type, val)
                else:
                    init_values[f.name] = val

        # instantiate dataclass
        obj = cls(**init_values)

        # attach unknown fields dynamically
        for k, v in data.items():
            if k not in field_names:
                setattr(obj, k, v)

        return obj

    def patch_torchtitan_embedding_amp(self, enable_patch: bool):
        """
        Monkey patch for AMP precision mismatch in nn.Embedding.

        Behavior:
            Globally patches nn.Embedding.__init__ to register a forward hook that:
            - When AMP/autocast is active, casts outputs to AMP dtype (bf16/fp16).
            - Otherwise, uses mixed_precision_param from Titan config.
            - Can be disabled via env: export PRIMUS_EMBED_AUTOCAST_DTYPE=off
        """

        import torch
        import torch.nn as nn

        from primus.core.utils.logger import _logger as primus_logger

        if not enable_patch:
            primus_logger.info("[PrimusPatch][AMP] Embedding AMP patch disabled via config.")
            return

        def _hook(module, inp, out):
            if not isinstance(out, torch.Tensor) or not out.is_floating_point():
                return out

            if torch.is_autocast_enabled():
                runtime_dtype = torch.get_autocast_gpu_dtype()
                if out.dtype != runtime_dtype:
                    return out.to(runtime_dtype)
            return out

        orig_init = nn.Embedding.__init__

        def new_init(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            self.register_forward_hook(_hook)

        nn.Embedding.__init__ = new_init
        primus_logger.info(
            "[PrimusPatch][AMP] nn.Embedding.__init__ patched for AMP/mixed precision alignment."
        )

    def patch_titan_train_spec(self, model_name: str, flavor: str, model_overrides: Dict[str, Any]):
        """
        Monkey patch torchtitan.train_spec.get_train_spec to override model args dynamically.
        All override keys MUST start with "model." (e.g., {"model.n_layers": 8}).
        """
        from primus.core.utils.logger import _logger as primus_logger

        if not model_overrides:
            primus_logger.info("[PrimusPatch][ModelOverride] No model_overrides provided, skip patch.")
            return

        primus_logger.info(f"[PrimusPatch][ModelOverride] Applying model_overrides: {model_overrides}")

        # --- flatten nested form {"model": {"n_layers": 4}} → {"model.n_layers": 4}
        flat_overrides = {}
        for k, v in model_overrides.items():
            if k == "model" and isinstance(v, dict):
                for subk, subv in v.items():
                    flat_overrides[f"model.{subk}"] = subv
            else:
                flat_overrides[k] = v
        model_overrides = flat_overrides

        # Enforce `model.` prefix strictly
        bad_keys = [k for k in model_overrides.keys() if not k.startswith("model.")]
        if bad_keys:
            raise ValueError(
                # f"[PrimusPatch][ModelOverride] Unsupported override keys (must start with 'model.'): {bad_keys}"
                f"[PrimusPatch][ModelOverride] Invalid override keys detected: {bad_keys}. "
                "These parameters belong to the model configuration and must be specified "
                "with the 'model.' prefix (e.g., 'model.n_layers', 'model.dim')."
            )

        primus_logger.info(
            f"[PrimusPatch][ModelOverride] model_overrides provided for '{model_name}' (flavor={flavor}): {model_overrides}"
        )

        import torchtitan.protocols.train_spec as train_spec_module

        orig_get_train_spec = train_spec_module.get_train_spec

        def patched_get_train_spec(name: str):
            spec = orig_get_train_spec(name)
            if name != model_name:
                return spec  # only patch targeted model

            assert hasattr(
                spec, "model_args"
            ), f"[PrimusPatch][ModelOverride] train_spec for '{name}' missing model_args"
            model_args_root = spec.model_args
            assert isinstance(
                model_args_root, dict
            ), f"[PrimusPatch][ModelOverride] train_spec.model_args must be dict, got {type(model_args_root)}"

            if flavor not in model_args_root:
                raise KeyError(
                    f"[PrimusPatch][ModelOverride] flavor '{flavor}' not found in model_args for '{name}'. "
                    f"Available flavors: {list(model_args_root.keys())}"
                )

            target_args = model_args_root[flavor]
            assert is_dataclass(
                target_args
            ), f"[PrimusPatch][ModelOverride] Expected dataclass model_args, got {type(target_args)}"

            before = asdict(target_args)
            for k, v in model_overrides.items():
                field_name = k[len("model.") :]
                if not hasattr(target_args, field_name):
                    raise AttributeError(
                        f"[PrimusPatch][ModelOverride] '{type(target_args).__name__}' has no field '{field_name}'"
                    )
                setattr(target_args, field_name, v)

            primus_logger.info(
                f"[PrimusPatch][ModelOverride] Patched dataclass model_args['{flavor}'] "
                f"for '{name}' with {model_overrides} (before={before})"
            )
            return spec

        # Apply the patch globally
        train_spec_module.get_train_spec = patched_get_train_spec
        primus_logger.info(
            f"[PrimusPatch][ModelOverride] get_train_spec for '{model_name}' successfully monkey patched (flavor={flavor})."
        )
