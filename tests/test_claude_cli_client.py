"""Tests for the Claude Code CLI (`claude -p`) LLM provider.

All tests mock ``subprocess.run`` so they never invoke the real CLI, never hit
the network, and never consume the user's subscription. They lock in the
contract the rest of the framework relies on: text/structured/tool-loop output
shapes, error surfacing via ``is_error``, billing-safe env scrubbing, and that
``--bare`` is never used.
"""

from __future__ import annotations

import json
import types

import pytest

from tradingagents.agents.schemas import SentimentReport
from tradingagents.agents.utils.core_stock_tools import get_stock_data
from tradingagents.llm_clients import create_llm_client
from tradingagents.llm_clients.claude_cli_client import (
    ChatClaudeCLI,
    ClaudeCLIClient,
    _strip_code_fences,
)

CLI_MODULE = "tradingagents.llm_clients.claude_cli_client"


def _completed(stdout: str, returncode: int = 0, stderr: str = ""):
    """Build a stand-in for subprocess.CompletedProcess."""
    return types.SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


@pytest.fixture
def llm():
    # claude_bin is preset so the client never tries to locate a real binary.
    return ChatClaudeCLI(model="sonnet", claude_bin="/usr/bin/claude", timeout=30)


# ---- factory + registry ---------------------------------------------------


def test_factory_returns_claude_cli_client():
    client = create_llm_client(provider="claude-cli", model="sonnet")
    assert isinstance(client, ClaudeCLIClient)
    assert isinstance(client.get_llm(), ChatClaudeCLI)


def test_alias_provider_name_resolves():
    assert isinstance(create_llm_client(provider="claude_code", model="opus"), ClaudeCLIClient)


# ---- plain text path ------------------------------------------------------


def test_plain_text_parses_result_field(monkeypatch, llm):
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return _completed(json.dumps({"result": "BUY — strong momentum", "is_error": False}))

    monkeypatch.setattr(f"{CLI_MODULE}.subprocess.run", fake_run)
    out = llm.invoke("Analyze NVDA")
    assert out.content == "BUY — strong momentum"
    # Prompt is delivered via stdin, never as a positional argv element.
    assert captured["kwargs"]["input"] == "Analyze NVDA"
    assert "-p" in captured["argv"] and "--output-format" in captured["argv"]


def test_is_error_raises(monkeypatch, llm):
    monkeypatch.setattr(
        f"{CLI_MODULE}.subprocess.run",
        lambda argv, **kw: _completed(json.dumps({"is_error": True, "result": "rate limit"})),
    )
    with pytest.raises(RuntimeError, match="rate limit"):
        llm.invoke("hi")


def test_empty_stdout_raises(monkeypatch, llm):
    monkeypatch.setattr(
        f"{CLI_MODULE}.subprocess.run",
        lambda argv, **kw: _completed("", returncode=1, stderr="auth failed"),
    )
    with pytest.raises(RuntimeError, match="no output"):
        llm.invoke("hi")


def test_non_json_stdout_raises(monkeypatch, llm):
    monkeypatch.setattr(
        f"{CLI_MODULE}.subprocess.run",
        lambda argv, **kw: _completed("not json at all"),
    )
    with pytest.raises(RuntimeError, match="non-JSON"):
        llm.invoke("hi")


# ---- structured output path ----------------------------------------------


def test_with_structured_output_returns_pydantic(monkeypatch, llm):
    captured = {}
    payload = {
        "overall_band": "Bullish",
        "overall_score": 7.5,
        "confidence": "high",
        "narrative": "Positive chatter across Reddit and StockTwits.",
    }

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        return _completed(json.dumps({"structured_output": payload, "result": "ignored"}))

    monkeypatch.setattr(f"{CLI_MODULE}.subprocess.run", fake_run)
    result = llm.with_structured_output(SentimentReport).invoke("Score sentiment for AAPL")

    assert isinstance(result, SentimentReport)
    assert result.overall_band.value == "Bullish"
    assert result.overall_score == 7.5
    # The structured path passes the JSON schema to the CLI.
    assert "--json-schema" in captured["argv"]


# ---- tool-loop / MCP path -------------------------------------------------


def test_bind_tools_builds_mcp_argv_and_returns_empty_tool_calls(monkeypatch, llm):
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        return _completed(json.dumps({"result": "## Market report\nfinal text", "is_error": False}))

    monkeypatch.setattr(f"{CLI_MODULE}.subprocess.run", fake_run)
    result = llm.bind_tools([get_stock_data]).invoke("Analyze the chart")

    argv = captured["argv"]
    assert "--mcp-config" in argv
    assert "--strict-mcp-config" in argv
    # allowedTools restricts Claude to exactly the bound tool, MCP-prefixed.
    allowed_idx = argv.index("--allowedTools") + 1
    assert "mcp__tradingagents__get_stock_data" in argv[allowed_idx]
    # The whole tool loop ran inside the subprocess; nothing left for ToolNode.
    assert result.content.startswith("## Market report")
    assert result.tool_calls == []


# ---- billing safety -------------------------------------------------------


def test_env_is_scrubbed_of_api_key_and_no_bare(monkeypatch, llm):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://example.invalid")
    monkeypatch.setenv("PATH", "/usr/bin")  # an unrelated var must survive
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["env"] = kwargs["env"]
        return _completed(json.dumps({"result": "ok"}))

    monkeypatch.setattr(f"{CLI_MODULE}.subprocess.run", fake_run)
    llm.invoke("hi")

    env = captured["env"]
    assert "ANTHROPIC_API_KEY" not in env
    assert "ANTHROPIC_BASE_URL" not in env
    assert env.get("PATH") == "/usr/bin"  # non-billing vars pass through
    assert "--bare" not in captured["argv"]


# ---- helpers --------------------------------------------------------------


def test_strip_code_fences():
    assert _strip_code_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert _strip_code_fences('{"a": 1}') == '{"a": 1}'


def test_fenced_json_output_is_parsed(monkeypatch, llm):
    monkeypatch.setattr(
        f"{CLI_MODULE}.subprocess.run",
        lambda argv, **kw: _completed('```json\n{"result": "fenced ok"}\n```'),
    )
    assert llm.invoke("hi").content == "fenced ok"


def test_temperature_accepted_but_ignored():
    # The graph force-adds temperature for every provider; the client must not
    # choke on it (claude -p has no temperature knob).
    client = ClaudeCLIClient(model="sonnet", temperature=0.0)
    assert isinstance(client.get_llm(), ChatClaudeCLI)
