###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
import socket
from abc import ABC, abstractmethod

from primus.core.utils.global_vars import get_primus_config


class BasePlatform(ABC):
    def __init__(self, name="local"):
        self.global_args = get_primus_config()
        self.args = self.global_args.platform_config
        self.enable_autotask_manager = False

    @property
    def sink_level(self):
        return self.args.master_sink_level

    def get_platform_env(self, key, default_value=None):
        if default_value is not None:
            return os.environ.get(key, default_value)
        assert key in os.environ, f"Cannot find {key} in {self.__class__}"

        return os.environ[key]

    def get_gpus_per_node(self):
        if self.args.gpus_per_node_env_key is None:
            return 8
        return int(self.get_platform_env(self.args.gpus_per_node_env_key, 8))

    def get_addr(self):
        """
        get ip address in current node
        """
        hostname = socket.gethostname()
        ip_addr = socket.gethostbyname(hostname)
        # If IP starts with 127.*, try to get real IP via socket
        if ip_addr.startswith("127."):
            try:
                # UDP socket connect doesn't send data, just determines routing
                # Works without actual network connectivity
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                    s.connect(("8.8.8.8", 80))
                    ip_addr = s.getsockname()[0]
            except Exception:
                pass

        return ip_addr

    @abstractmethod
    def setup(self):
        """
        setup platform
        """
        raise NotImplementedError

    @abstractmethod
    def teardown(self):
        """
        teardown platform
        """
        raise NotImplementedError

    @abstractmethod
    def get_num_nodes(self):
        """
        get node number
        """
        raise NotImplementedError

    @abstractmethod
    def get_master_addr(self):
        """
        get ip address in master node
        """
        raise NotImplementedError

    @abstractmethod
    def get_master_port(self):
        """
        get port in master node
        """
        raise NotImplementedError

    @abstractmethod
    def get_node_rank(self):
        """
        get node rank in current node
        """
        raise NotImplementedError
