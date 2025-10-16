###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from torchtitan.config.job_config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.model_converter import (
    ModelConvertersContainer as TTModelConvertersContainer,
)
from torchtitan.protocols.model_converter import (
    _registry_model_converter_cls as registry_model_converter_cls,
)


class ModelConvertersContainer(TTModelConvertersContainer):
    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        super().__init__(job_config=job_config, parallel_dims=parallel_dims)

        if job_config.primus_turbo.enable_primus_turbo:
            self.primus_turbo_entension(job_config, ParallelDims)

    def primus_turbo_entension(self, job_config, ParallelDims):
        # Append different converts according to the primus turbo config.
        self.converters.append(registry_model_converter_cls["primus_turbo"](job_config, ParallelDims))
