###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import logging
import sys
from abc import ABC, abstractmethod
from typing import List, Optional

logger = logging.getLogger(__name__)


class CommandBase(ABC):
    """Base class for all CLI commands.

    This class provides a framework for implementing CLI commands with
    argument registration, validation, and execution.

    Subclasses must implement:
    - name(): Return the command name
    - help(): Return the help text
    - register_arguments(): Register command-specific arguments
    - run(): Execute the command logic

    Subclasses may optionally override:
    - validate_args(): Perform custom argument validation
    - description(): Return detailed command description
    """

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """Return the name of the command as it should appear in CLI.

        Returns:
            Command name (e.g., 'train', 'benchmark').
        """

    @classmethod
    @abstractmethod
    def help(cls) -> str:
        """Return the help message for this command.

        Returns:
            Short help text displayed in command list.
        """

    @classmethod
    def description(cls) -> Optional[str]:
        """Return detailed description for this command.

        Returns:
            Long description displayed in command help, or None.
        """
        return None

    @classmethod
    @abstractmethod
    def register_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Register command-specific arguments to the parser.

        Args:
            parser: The argument parser for this command.
        """

    @classmethod
    def validate_args(cls, args: argparse.Namespace, unknown_args: Optional[List[str]] = None) -> bool:
        """Validate parsed arguments before execution.

        Override this method to implement custom validation logic.
        If validation fails, this method should print an error message
        and return False.

        Args:
            args: Parsed known arguments.
            unknown_args: List of unknown arguments.

        Returns:
            True if arguments are valid, False otherwise.
        """
        return True

    @classmethod
    @abstractmethod
    def run(cls, args: argparse.Namespace, unknown_args: Optional[List[str]] = None) -> None:
        """Execute the command.

        This method is called after argument validation. Subclasses should
        implement the main command logic here.

        Args:
            args: Parsed known arguments.
            unknown_args: List of unknown arguments.
        """

    @classmethod
    def execute(cls, args: argparse.Namespace, unknown_args: Optional[List[str]] = None) -> None:
        """Execute the command with validation.

        This is a convenience method that performs validation before
        calling run(). Use this instead of calling run() directly.

        Args:
            args: Parsed known arguments.
            unknown_args: List of unknown arguments.
        """
        logger.debug(f"Validating arguments for command: {cls.name()}")
        if not cls.validate_args(args, unknown_args):
            logger.error(f"Argument validation failed for command: {cls.name()}")
            sys.exit(1)

        logger.debug(f"Running command: {cls.name()}")
        cls.run(args, unknown_args)
