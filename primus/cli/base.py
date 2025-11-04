###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
from abc import ABC, abstractmethod
from typing import List, Optional


class CommandBase(ABC):
    """Base class for all CLI commands."""

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """Return the name of the command as it should appear in CLI."""

    @classmethod
    @abstractmethod
    def help(cls) -> str:
        """Return the help message for this command."""

    @classmethod
    @abstractmethod
    def register_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Register command-specific arguments to the parser.

        Args:
            parser: The argument parser for this command.
        """

    @classmethod
    @abstractmethod
    def run(cls, args: argparse.Namespace, unknown_args: Optional[List[str]] = None) -> None:
        """Execute the command.

        Args:
            args: Parsed known arguments.
            unknown_args: List of unknown arguments.
        """
