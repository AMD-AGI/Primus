from primus.core.config.primus_config import PrimusConfig


class PrimusConfigLoader:
    """
    Legacy loader - now deprecated.
    Use PrimusConfig.from_file() instead.
    """

    @staticmethod
    def load(path: str) -> PrimusConfig:
        """
        Load configuration from YAML file.

        This is a legacy method - prefer using PrimusConfig.from_file() directly.
        """
        import argparse

        # Create a minimal cli_args namespace for compatibility
        cli_args = argparse.Namespace()

        return PrimusConfig.from_file(path, cli_args)
