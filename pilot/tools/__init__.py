"""Pilot tools: business actions exposed at CLI / MCP boundary.

Each tool module is invokable via:

    python -m pilot.tools.<module> <subcommand> [args]

The Agent (Cursor / Claude / Codex / harness) calls these via shell or MCP,
not via direct Python import. Pilot core does not import any agent SDK.
"""
