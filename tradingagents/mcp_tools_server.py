"""MCP server exposing TradingAgents' data-fetch tools to the Claude Code CLI.

When the framework runs on the ``claude-cli`` provider, the tool-using analysts
need their Python data tools (yfinance / Alpha Vantage fetchers) callable from
inside the ``claude -p`` subprocess. This module wraps the existing LangChain
``@tool`` functions as MCP tools and serves them over stdio, so Claude Code can
call them natively during its internal tool loop.

It runs as a **separate process** spawned by ``claude -p`` (configured by
:class:`~tradingagents.llm_clients.claude_cli_client.ChatClaudeCLI`), so it
rebuilds the framework configuration from the JSON snapshot pointed to by the
``TRADINGAGENTS_MCP_CONFIG_PATH`` environment variable before registering any
tools — otherwise the vendor routing (``data_vendors``) and cache paths would
fall back to defaults.

Entry point: ``python -m tradingagents.mcp_tools_server``.
"""

from __future__ import annotations

import json
import os
import sys

from mcp.server.fastmcp import FastMCP

# The data tools the analysts bind. Imported from agent_utils, which is the
# single place the rest of the framework imports them from, so descriptions and
# schemas stay identical to what the API providers see.
from tradingagents.agents.utils.agent_utils import (
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_global_news,
    get_income_statement,
    get_indicators,
    get_insider_transactions,
    get_news,
    get_stock_data,
    get_verified_market_snapshot,
)
from tradingagents.dataflows.config import set_config

_DATA_TOOLS = (
    get_stock_data,
    get_indicators,
    get_verified_market_snapshot,
    get_news,
    get_global_news,
    get_insider_transactions,
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement,
)


def _load_config_from_env() -> None:
    """Restore the parent process's config snapshot, if one was provided."""
    path = os.environ.get("TRADINGAGENTS_MCP_CONFIG_PATH")
    if not path or not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
        set_config(config)
    except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - defensive
        print(f"[mcp_tools_server] could not load config from {path}: {exc}", file=sys.stderr)


def build_server() -> FastMCP:
    """Construct the FastMCP server with every data tool registered."""
    _load_config_from_env()
    server = FastMCP(name="tradingagents")
    for langchain_tool in _DATA_TOOLS:
        # ``langchain_tool.func`` is the underlying function with typed,
        # annotated parameters, which FastMCP introspects into an input schema.
        server.add_tool(
            langchain_tool.func,
            name=langchain_tool.name,
            description=langchain_tool.description,
        )
    return server


def main() -> None:
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
