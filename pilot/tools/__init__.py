"""Pilot tools: business actions exposed at CLI / MCP boundary.

Each tool module is invokable via either of two equivalent commands:

    python -m pilot.tools.<module> <subcommand> [args]   # legacy / direct
    python -m pilot          <module> <subcommand> [args] # unified front door

The unified form is implemented by ``pilot/cli/main.py`` and just re-emits
``sys.argv`` to the same ``_cli()``; there is no behavioural difference.

The Agent (Cursor / Claude / Codex / harness) calls these via shell or MCP,
not via direct Python import. Pilot core does not import any agent SDK.
"""
