###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import ctypes

import torch
from megatron.core.dist_checkpointing.strategies.filesystem_async import (
    FileSystemWriterAsync,
)

from primus.modules.module_utils import log_rank_0, warning_rank_0


class PrimusFileSystemWriterAsync(FileSystemWriterAsync):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def preload_tensors(*args, **kwargs):
        # (limou)
        # change argument non_blocking to False on HIP platform
        # the tensors will be stored in pinned memory if non_blocking=True
        # on the ROCm platform (hip_runtime_version < 7.1)
        # forking a subprocess afterward with pinned_memory=True will trigger segmentation fault
        if torch.version.hip:
            major, minor = PrimusFileSystemWriterAsync.get_hip_runtime_version()
            log_rank_0(f"hip runtime version : {major}.{minor}")
            if major < 7 or (major == 7 and minor < 1):
                log_rank_0("HIP env detected, change argument non_blocking in FileSystemWriterAsync to False")
                if "non_blocking" in kwargs:
                    kwargs["non_blocking"] = False
                elif len(args) > 0 and type(args[-1]) == type(True):
                    # TODO (limou)
                    # non_blocking may NOT always be the last argument in the future
                    args = args[:-1] + (False,)
                else:
                    warning_rank_0("found argument non_blocking failed")

        return super(PrimusFileSystemWriterAsync, PrimusFileSystemWriterAsync).preload_tensors(
            *args, **kwargs
        )

    # unlike torch.version.hip
    # hipRuntimeGetVersion() can return the HIP runtime version instead of build-time
    @staticmethod
    def get_hip_runtime_version():
        try:
            libhip = ctypes.CDLL("libamdhip64.so")
            hipRuntimeGetVersion = libhip.hipRuntimeGetVersion
            hipRuntimeGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
            hipRuntimeGetVersion.restype = ctypes.c_int
            version = ctypes.c_int()
            error_code = hipRuntimeGetVersion(ctypes.byref(version))
            if error_code != 0:
                return (-1, -1)
            # (major_version, minor_version)
            return (version.value // 10000000, (version.value // 100000) % 100)
        except Exception as e:
            print(e)
            return (-1, -1)
