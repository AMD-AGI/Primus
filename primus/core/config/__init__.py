# Avoid circular import - users should import directly from submodules
# from primus.core.config.primus_config import ModuleConfig, PrimusConfig

__all__ = ["PrimusConfig", "ModuleConfig"]


def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    if name == "PrimusConfig" or name == "ModuleConfig":
        from primus.core.config.primus_config import ModuleConfig, PrimusConfig

        return PrimusConfig if name == "PrimusConfig" else ModuleConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
