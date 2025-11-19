###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from dataclasses import dataclass, field

from torchtitan.config.job_config import JobConfig as TTJobConfig

# TODO: float8 quant config
# Tensorwise / Rowwise / Blockwise  etc.
# @dataclass
# class PrimusTurboFloat8Config:
#     pass


@dataclass
class PrimusTurboConfig:
    enable_primus_turbo: bool = False
    enable_attention_float8: bool = False
    use_turbo_attention: bool = False
    use_turbo_async_tp: bool = False
    use_turbo_mx_linear: bool = False
    enable_embedding_autocast: bool = True
    # float8_config: PrimusTurboFloat8Config = field(default_factory=PrimusTurboFloat8Config)


@dataclass
class JobConfig(TTJobConfig):
    primus_turbo: PrimusTurboConfig = field(default_factory=PrimusTurboConfig)
