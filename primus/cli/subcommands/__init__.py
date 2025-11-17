"""
Primus CLI subcommand package.

Every module in this package (excluding those starting with underscores) can
provide a `register_subcommand(subparsers)` function. The main CLI will scan
this package and automatically register each available command.
"""

__all__ = []
