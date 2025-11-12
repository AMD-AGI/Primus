###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import logging
import sys
import time

from primus import __version__
from primus.cli.registry import CommandRegistry
from primus.cli.utils import (
    is_debug_mode,
    is_profile_mode,
    load_config_file,
    merge_config_with_args,
    print_error,
    setup_logging,
)

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the main argument parser.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="primus",
        description="Primus Unified CLI for Training & Utilities",
        epilog="""
Examples:
  primus train pretrain --config config.yaml
  primus benchmark gemm --batch-size 32
  primus preflight --check-all

Environment Variables:
  PRIMUS_DEBUG    Enable debug mode with full stack traces
  PRIMUS_PROFILE  Enable performance profiling and timing

For more information, visit: https://github.com/AMD/Primus
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Global arguments
    parser.add_argument(
        "--version",
        action="version",
        version=f"Primus {__version__}",
        help="Show version information and exit",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output (DEBUG level logging)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress non-error output (ERROR level logging only)",
    )
    parser.add_argument(
        "--config",
        type=str,
        metavar="PATH",
        help="Path to global configuration file (YAML)",
    )
    parser.add_argument(
        "--completion",
        choices=["bash", "zsh", "fish"],
        metavar="SHELL",
        help="Generate shell completion script and exit",
    )

    return parser


def register_commands(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register all available commands to the parser.

    Args:
        parser: Main argument parser.

    Returns:
        Parser with registered commands.
    """
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Auto-discover and register all commands
    CommandRegistry.discover_commands()
    for command_cls in CommandRegistry.get_all_commands().values():
        command_parser = subparsers.add_parser(
            command_cls.name(),
            help=command_cls.help(),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        command_cls.register_arguments(command_parser)
        command_parser.set_defaults(command_cls=command_cls)

    return parser


def generate_completion(shell: str) -> None:
    """Generate shell completion script.

    Args:
        shell: Shell type (bash, zsh, fish).
    """
    try:
        import argcomplete  # type: ignore  # Optional dependency

        print(f"# Primus shell completion for {shell}")
        print("# Add this to your shell configuration file:")
        print()
        if shell == "bash":
            print('eval "$(register-python-argcomplete primus)"')
        elif shell == "zsh":
            print("# For zsh, enable bashcompinit first:")
            print("# autoload -U bashcompinit && bashcompinit")
            print('eval "$(register-python-argcomplete primus)"')
        elif shell == "fish":
            print("# For fish, use:")
            print("register-python-argcomplete --shell fish primus | source")
    except ImportError:
        print_error("argcomplete is not installed. Install it with: pip install argcomplete")
        sys.exit(1)


def main() -> None:
    """
    Primus Unified CLI Entry

    Currently supported:
    - train: Launch Megatron / TorchTitan / Jax training.
    - benchmark: Run benchmarking tools for performance evaluation.
    - preflight: Environment and configuration checks.

    Reserved for future expansion:
    - evaluate: Model evaluation workflows.
    - export: Model export and conversion.
    """
    start_time = time.time()

    try:
        # Handle argcomplete if available (optional dependency for shell completion)
        try:
            import argcomplete  # type: ignore
        except ImportError:
            argcomplete = None  # type: ignore

        # Create parser and register commands
        parser = create_parser()

        # Parse global arguments first to set up logging
        args, remaining = parser.parse_known_args()

        # Handle completion generation
        if hasattr(args, "completion") and args.completion:
            generate_completion(args.completion)
            return

        # Setup logging based on verbosity
        setup_logging(
            verbose=getattr(args, "verbose", False),
            quiet=getattr(args, "quiet", False),
        )

        logger.debug(f"Primus CLI v{__version__} starting...")
        logger.debug(f"Command-line arguments: {sys.argv[1:]}")

        # Register commands after logging is configured
        parser = register_commands(parser)

        # Enable argcomplete if available
        if argcomplete:
            argcomplete.autocomplete(parser)

        # Re-parse with all commands registered
        args, unknown_args = parser.parse_known_args()

        # Load and merge configuration file if specified
        if hasattr(args, "config") and args.config:
            config = load_config_file(args.config)
            merge_config_with_args(config, args)
            logger.debug(f"Merged configuration: {config}")

        # Execute command
        if hasattr(args, "command_cls"):
            logger.info(f"Executing command: {args.command}")
            args.command_cls.run(args, unknown_args)
            logger.debug("Command execution completed successfully")
        else:
            parser.print_help()

        # Print execution time if profiling is enabled
        if is_profile_mode():
            elapsed = time.time() - start_time
            print(f"\n⏱️  Command completed in {elapsed:.2f}s", file=sys.stderr)

    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user.", file=sys.stderr)
        sys.exit(130)  # Standard SIGINT exit code

    except SystemExit:
        # Allow sys.exit() calls to propagate
        raise

    except Exception as e:
        if is_debug_mode():
            # In debug mode, show full stack trace
            logger.exception("Fatal error occurred")
            raise
        else:
            # In production mode, show clean error message
            print_error(str(e))
            logger.debug("Use PRIMUS_DEBUG=1 for full stack trace")
            sys.exit(1)


if __name__ == "__main__":
    # Support for -- separator in command line
    if len(sys.argv) > 1 and sys.argv[1] == "--":
        sys.argv.pop(1)
    main()
