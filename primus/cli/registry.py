###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import importlib
import inspect
import logging
import os
import pkgutil
from typing import Dict, Type

from primus.cli.base import CommandBase

logger = logging.getLogger(__name__)


class CommandRegistry:
    """Registry for CLI commands that handles auto-discovery.

    This registry supports both eager and lazy loading of commands:
    - Eager loading: All commands are imported immediately during discovery
    - Lazy loading: Commands are only imported when first accessed

    Lazy loading improves CLI startup time for operations like --help.
    """

    _commands: Dict[str, Type[CommandBase]] = {}
    _lazy_commands: Dict[str, str] = {}  # Maps command name to module name
    _discovered: bool = False

    @classmethod
    def discover_commands(cls, lazy: bool = False) -> None:
        """Discover all command classes in the CLI package.

        Args:
            lazy: If True, register commands without importing them.
                 Commands will be imported on first access.
        """
        if cls._discovered:
            logger.debug("Commands already discovered, skipping")
            return

        # Get the directory containing CLI subcommands
        subcommands_dir = os.path.join(os.path.dirname(__file__), "subcommands")

        if not os.path.exists(subcommands_dir):
            logger.warning(f"Subcommands directory not found: {subcommands_dir}")
            return

        discovered_count = 0

        # Discover all Python modules in the subcommands directory
        for _, module_name, _ in pkgutil.iter_modules([subcommands_dir]):
            if module_name.startswith("_"):
                # Skip __init__ and other internal modules
                continue

            if lazy:
                # Lazy loading: just record the module name
                logger.debug(f"Registering command module for lazy loading: {module_name}")
                cls._lazy_commands[module_name] = f"primus.cli.subcommands.{module_name}"
            else:
                # Eager loading: import and register immediately
                try:
                    cls._load_module(module_name)
                    discovered_count += 1
                except Exception as e:
                    logger.error(f"Failed to load command module '{module_name}': {e}")
                    if os.getenv("PRIMUS_DEBUG"):
                        raise

        cls._discovered = True
        logger.debug(f"Discovered {discovered_count} command module(s)")

    @classmethod
    def _load_module(cls, module_name: str) -> None:
        """Load a command module and register its commands.

        Args:
            module_name: Name of the module (without package prefix).
        """
        full_module_name = f"primus.cli.subcommands.{module_name}"
        logger.debug(f"Loading command module: {full_module_name}")

        try:
            module = importlib.import_module(full_module_name)

            # Find all CommandBase subclasses in the module
            for obj_name, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, CommandBase)
                    and obj != CommandBase
                    and obj.__module__ == full_module_name  # Only classes defined in this module
                ):
                    command_name = obj.name()
                    if command_name in cls._commands:
                        logger.warning(f"Command '{command_name}' already registered, skipping duplicate")
                        continue

                    cls._commands[command_name] = obj
                    logger.debug(f"Registered command: {command_name} from {obj_name}")

        except Exception as e:
            logger.error(f"Error loading module {full_module_name}: {e}")
            raise

    @classmethod
    def _ensure_command_loaded(cls, name: str) -> None:
        """Ensure a command is loaded from lazy registry.

        Args:
            name: Command name to ensure is loaded.
        """
        # Check if command is already loaded
        if name in cls._commands:
            return

        # Try to find and load from lazy registry
        for module_name, full_module_name in cls._lazy_commands.items():
            # Import the module
            module = importlib.import_module(full_module_name)

            # Find all CommandBase subclasses
            for obj_name, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, CommandBase)
                    and obj != CommandBase
                    and obj.__module__ == full_module_name
                ):
                    command_name = obj.name()
                    if command_name not in cls._commands:
                        cls._commands[command_name] = obj
                        logger.debug(f"Lazy loaded command: {command_name}")

                    if command_name == name:
                        return

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
        cls._ensure_command_loaded(name)

        if name not in cls._commands:
            available = ", ".join(cls.get_all_commands().keys())
            raise KeyError(f"Command '{name}' not found. Available commands: {available}")

        return cls._commands[name]

    @classmethod
    def get_all_commands(cls) -> Dict[str, Type[CommandBase]]:
        """Get all registered commands.

        If lazy loading was used, this will load all commands.

        Returns:
            A dictionary mapping command names to command classes.
        """
        # Load all lazy commands if any
        for module_name in list(cls._lazy_commands.keys()):
            cls._load_module(module_name)

        # Clear lazy registry after loading all
        cls._lazy_commands.clear()

        return cls._commands

    @classmethod
    def reset(cls) -> None:
        """Reset the registry (mainly for testing purposes)."""
        cls._commands.clear()
        cls._lazy_commands.clear()
        cls._discovered = False
        logger.debug("Command registry reset")
