"""
Primus CLI subcommand package.

Every module in this package (excluding those starting with underscores) must
provide a `register_subcommand(subparsers)` function, return the parser object,
and call `parser.set_defaults(func=run)` (or equivalent) so the main CLI knows
which handler to execute. The discovery logic in `primus.cli.main` will import
each module and register the associated command automatically.
"""

__all__ = []
