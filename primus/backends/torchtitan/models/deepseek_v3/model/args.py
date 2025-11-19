###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from dataclasses import dataclass

from torchtitan.config import JobConfig
from torchtitan.models.deepseek_v3.model.args import DeepSeekV3ModelArgs
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability

from primus.backends.torchtitan.tools.utils import is_hip


@dataclass
class PrimusDeepSeekV3ModelArgs(DeepSeekV3ModelArgs):
    def update_from_config(self, job_config: JobConfig, **kwargs) -> None:
        seq_len = job_config.training.seq_len
        if seq_len > self.max_seq_len:
            logger.warning(
                f"Sequence length {seq_len} exceeds original maximum {self.max_seq_len}."
            )
        self.max_seq_len = seq_len

        if self.moe_args.use_grouped_mm:
            if is_hip():
                if not has_cuda_capability(9, 4) or not job_config.primus_turbo.enable_primus_turbo:
                    logger.warning(
                        "Failed to use grouped mm, which is only supported on AMD GFX94 or later and needs to enable_primus_turbo.",
                    )
                    self.moe_args.use_grouped_mm = False
            else:
                if not has_cuda_capability(9, 0):
                    logger.warning(
                        "Failed to use grouped mm, which is only supported on NV SM90 or later.",
                    )
                    self.moe_args.use_grouped_mm = False

        if job_config.parallelism.context_parallel_degree > 1 and self.use_flex_attn:
            raise NotImplementedError(
                "CP support for FlexAttention is still in progress."
            )

        self.moe_args._debug_force_load_balance = (
            job_config.training.debug_moe_force_load_balance
        )
