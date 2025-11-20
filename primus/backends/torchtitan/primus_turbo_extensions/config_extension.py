###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from dataclasses import dataclass, field

from torchtitan.config.job_config import JobConfig as TTJobConfig
from torchtitan.config.job_config import Profiling as TTProfilingConfig

# TODO: float8 quant config
# Tensorwise / Rowwise / Blockwise  etc.
# @dataclass
# class PrimusTurboFloat8Config:
#     pass


@dataclass
class ProfilingConfig(TTProfilingConfig):
    """
    Extended Profiling configuration with Primus extensions.
    
    Additional attributes:
        profile_ranks (list[int]): List of ranks to profile. Defaults to [0].
    """
    profile_ranks: list[int] = field(default_factory=lambda: [0])


@dataclass
class PrimusTurboConfig:
    enable_primus_turbo: bool = False
    enable_attention_float8: bool = False
    use_turbo_attention: bool = False
    use_turbo_async_tp: bool = False
    use_turbo_mx_linear: bool = False
    use_turbo_grouped_mm: bool = False
    use_moe_fp8: bool = True
    enable_embedding_autocast: bool = True
    # float8_config: PrimusTurboFloat8Config = field(default_factory=PrimusTurboFloat8Config)


@dataclass
class JobConfig(TTJobConfig):
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    primus_turbo: PrimusTurboConfig = field(default_factory=PrimusTurboConfig)
