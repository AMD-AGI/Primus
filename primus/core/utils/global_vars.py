###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


from primus.core.launcher.config import PrimusConfig

_GLOBAL_CLI_ARGS = None
_GLOBAL_PRIMUS_CFG = None
_GLOBAL_TARGET_PLATFORM = None


def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is not None, "{} is not initialized.".format(name)


def is_initialized():
    if _GLOBAL_PRIMUS_CFG is None:
        return False
    return _GLOBAL_PRIMUS_CFG.initialized


def set_initialized():
    _GLOBAL_PRIMUS_CFG.initialized = True


def get_cli_args():
    """Return cli arguments."""
    _ensure_var_is_initialized(_GLOBAL_CLI_ARGS, "cli args")
    return _GLOBAL_PRIMUS_CFG.cli_args


def get_primus_config():
    """Return primus config."""
    _ensure_var_is_initialized(_GLOBAL_PRIMUS_CFG, "primus config")
    return _GLOBAL_PRIMUS_CFG


def get_target_platform():
    """Return target platform."""
    _ensure_var_is_initialized(_GLOBAL_TARGET_PLATFORM, "target_platform")
    return _GLOBAL_TARGET_PLATFORM


def set_global_variables(cfg: PrimusConfig):
    """Set global vars"""
    assert cfg is not None

    global _GLOBAL_PRIMUS_CFG
    if _GLOBAL_PRIMUS_CFG:
        return
    _GLOBAL_PRIMUS_CFG = cfg

    _set_cli_args(cfg)
    _set_target_platform(cfg)


def _set_cli_args(cfg: PrimusConfig):
    global _GLOBAL_CLI_ARGS
    if _GLOBAL_CLI_ARGS:
        return
    _GLOBAL_CLI_ARGS = cfg.cli_args


def _set_target_platform(cfg: PrimusConfig):
    global _GLOBAL_TARGET_PLATFORM
    if _GLOBAL_TARGET_PLATFORM:
        return

    platform_config = cfg.platform_config
    if platform_config.name and platform_config.name != "local":
        from primus.platforms import RemotePlatform

        _GLOBAL_TARGET_PLATFORM = RemotePlatform(platform_config.name)
    else:
        from primus.platforms import LocalPlatform

        _GLOBAL_TARGET_PLATFORM = LocalPlatform("local")
