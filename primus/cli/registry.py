###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import importlib
import inspect
import os
import pkgutil
from typing import Dict, Type

from primus.cli.base import CommandBase


class CommandRegistry:
    """Registry for CLI commands that handles auto-discovery."""

    _commands: Dict[str, Type[CommandBase]] = {}

    @classmethod
    def discover_commands(cls) -> None:
        """Discover all command classes in the CLI package."""
        # Get the directory containing CLI subcommands
        subcommands_dir = os.path.join(os.path.dirname(__file__), "subcommands")

        # Discover all Python modules in the subcommands directory
        for _, name, _ in pkgutil.iter_modules([subcommands_dir]):
            if not name.startswith("_"):  # Skip __init__ and other internal modules
                module = importlib.import_module(f"primus.cli.subcommands.{name}")

                # Find all classes in the module
                for _, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and issubclass(obj, CommandBase) and obj != CommandBase:
                        # Register the command
                        cls._commands[obj.name()] = obj

    @classmethod
    def get_command(cls, name: str) -> Type[CommandBase]:
        """Get a command by name.

        Args:
            name: The name of the command.

        Returns:
            The command class.

        Raises:
            KeyError: If the command is not found.
        """
        return cls._commands[name]

    @classmethod
    def get_all_commands(cls) -> Dict[str, Type[CommandBase]]:
        """Get all registered commands.

        Returns:
            A dictionary mapping command names to command classes.
        """
        return cls._commands
