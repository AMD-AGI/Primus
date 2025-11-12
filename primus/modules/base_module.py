###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import builtins
import os
from abc import ABC, abstractmethod

from primus.core.launcher.config import PrimusConfig
from primus.core.utils import checker, logger
from primus.core.utils.global_vars import (
    get_cli_args,
    get_target_platform,
    set_global_variables,
)

from .module_utils import debug_rank_all, set_logging_rank


class BaseModule(ABC):
    def __init__(
        self,
        module_name: str,
        primus_config: PrimusConfig,
        module_rank: int,
        module_world_size: int,
        module_master_addr: str = None,
        module_master_port: int = None,
    ):
        # module will be initialized by multiple worker processes
        self.module_name = module_name

        assert primus_config is not None
        self.primus_config = primus_config
        self.module_config = primus_config.get_module_config(module_name)

        # set config into the global vars of worker process
        set_global_variables(primus_config)
        self.platform = get_target_platform()
        self.cli_args = get_cli_args()

        # distributed module worker infos
        checker.check_true(
            module_master_port is not None,
            msg=f"Must provide a master port for all module workers",
        )
        if module_rank > 0:
            checker.check_true(
                module_master_addr is not None,
                msg=f"Must provide master addr for workers with rank > 0",
            )
        self.module_rank = module_rank
        self.module_world_size = module_world_size
        if module_rank == 0 and module_master_addr is not None:
            self.module_master_addr = self.platform.get_addr()
        else:
            self.module_master_addr = module_master_addr
        self.module_master_port = module_master_port

        # Set the environment variables to make them accessible for workers if needed later.
        os.environ["MASTER_ADDR"] = str(self.module_master_addr)
        os.environ["MASTER_PORT"] = str(self.module_master_port)
        os.environ["WORLD_SIZE"] = str(self.module_world_size)
        os.environ["RANK"] = str(self.module_rank)
        self.num_gpus_per_node = self.platform.get_gpus_per_node()
        self.module_local_rank = self.module_rank % self.num_gpus_per_node
        os.environ["LOCAL_RANK"] = str(self.module_local_rank)

        # setup logger for worker
        # self.setup_worker_logger(module_rank, module_world_size)

    @abstractmethod
    def init(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def setup(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError

    def setup_worker_logger(self, rank, world_size):
        # setup logger of dist module
        if self.module_config.sink_level is not None:
            self.module_config.file_sink_level = self.module_config.sink_level
            self.module_config.stderr_sink_level = self.module_config.sink_level
        logger_cfg = logger.LoggerConfig(
            exp_root_path=self.primus_config.exp_root_path,
            work_group=self.primus_config.exp_meta_info["work_group"],
            user_name=self.primus_config.exp_meta_info["user_name"],
            exp_name=self.primus_config.exp_meta_info["exp_name"],
            module_name=self.module_name,
            file_sink_level=self.module_config.file_sink_level,
            stderr_sink_level=self.module_config.stderr_sink_level,
            node_ip=self.platform.get_addr(),
            rank=rank,
            world_size=world_size,
        )
        logger.setup_logger(logger_cfg, is_head=False)

        # use log_rank_0 before torch distributed initialize
        set_logging_rank(rank, world_size)

        # monkey patch print function of builtins
        self.original_print = builtins.print
        # builtins.print = log_rank_all
        builtins.print = debug_rank_all

    @property
    def exp_root_path(self) -> str:
        return self.primus_config.exp_root_path

    @property
    def exp_meta_info(self) -> dict:
        return self.primus_config.exp_meta_info

    def get_module_master_address(self) -> str:
        return self.module_master_addr

    def get_module_master_port(self) -> int:
        return self.module_master_port

    def get_module_rank(self) -> int:
        return self.module_rank

    def get_module_world_size(self) -> int:
        raise self.module_world_size

    @property
    def get_module_config(self):
        return self.module_config

    @property
    def trainable(self):
        return self.module_config.trainable
