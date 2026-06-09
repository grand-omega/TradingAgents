"""LLM client that drives the Claude Code CLI (``claude -p``) via subprocess.

This provider lets TradingAgents run on a **Claude subscription (Pro/Max OAuth)**
instead of a per-token API key: every model call shells out to the locally
installed ``claude`` binary in print mode, which uses whatever auth Claude Code
is logged in with.

Three call shapes are supported, matching how the framework uses the LLM:

* **Plain text** (researchers, debaters, reflection): ``invoke(messages)`` →
  ``claude -p --output-format json`` → text from the JSON ``result`` field.
* **Structured output** (Research/Portfolio managers, Trader, Sentiment via
  ``with_structured_output``): ``claude -p --json-schema <schema>`` → validated
  object from the JSON ``structured_output`` field → Pydantic instance.
* **Tool-calling analysts** (market/news/fundamentals via ``bind_tools``): the
  bound Python data tools are exposed to Claude Code through an MCP server
  (``tradingagents.mcp_tools_server``); ``claude -p --mcp-config … --allowedTools
  mcp__tradingagents__*`` runs the *entire* tool loop inside the subprocess and
  returns the finished report. ``_generate`` returns an ``AIMessage`` with an
  empty ``tool_calls`` list, so the existing analyst nodes and LangGraph
  ``ToolNode`` work unchanged (the ToolNode simply never fires for this
  provider).

Billing safety: the subprocess environment is scrubbed of ``ANTHROPIC_API_KEY``
and related vars so Claude Code uses the OAuth subscription rather than silently
falling back to API/3P billing. ``--bare`` is never used (it forces API-key
auth).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages import convert_to_messages
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel

from .base_client import BaseLLMClient
from .validators import validate_model

logger = logging.getLogger(__name__)

# Env vars that, if present, flip Claude Code off the OAuth subscription onto
# API or third-party (Bedrock/Vertex) billing. Scrubbed from the subprocess env
# so this provider always bills the subscription.
_BILLING_ENV_SCRUB = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
)

# Prefix Claude Code uses for tools served by our MCP server "tradingagents".
_MCP_SERVER_NAME = "tradingagents"
_MCP_TOOL_PREFIX = f"mcp__{_MCP_SERVER_NAME}__"

_DEFAULT_TIMEOUT = 600

# System prompt that replaces Claude Code's default coding-agent persona. The
# real instructions arrive per-call as SystemMessages; this is only the floor
# used when a call carries none.
_FALLBACK_SYSTEM_PROMPT = (
    "You are a financial analysis assistant. Follow the user's instructions "
    "exactly and respond with the requested analysis only."
)


def _find_claude_binary() -> str:
    """Locate the ``claude`` executable, falling back to the default install path."""
    found = shutil.which("claude")
    if found:
        return found
    fallback = os.path.expanduser("~/.local/bin/claude")
    if os.path.exists(fallback):
        return fallback
    raise RuntimeError(
        "The 'claude' CLI was not found on PATH. Install Claude Code "
        "(https://docs.claude.com/en/docs/claude-code) and run `claude login` "
        "with your subscription account to use the 'claude-cli' provider."
    )


def _strip_code_fences(text: str) -> str:
    """Remove a leading/trailing ```...``` fence if the model wrapped its JSON."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    # drop the opening fence line (``` or ```json) and a trailing fence line
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _content_to_text(content: Any) -> str:
    """Coerce a LangChain message content (str or list of blocks) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(p for p in parts if p)
    return "" if content is None else str(content)


class ChatClaudeCLI(BaseChatModel):
    """LangChain chat model backed by the ``claude -p`` CLI (subscription billing)."""

    model: str
    effort: Optional[str] = None
    # Accepted for cross-provider compatibility but ignored: ``claude -p`` is an
    # agent, not a completion endpoint, and exposes no temperature control.
    temperature: Optional[float] = None
    timeout: Optional[int] = None
    claude_bin: Optional[str] = None

    @property
    def _llm_type(self) -> str:
        return "claude-cli"

    @property
    def _identifying_params(self) -> dict:
        return {"model": self.model, "effort": self.effort}

    # ------------------------------------------------------------------ tools
    def bind_tools(self, tools: List[Any], **kwargs: Any) -> Runnable:
        """Bind Python tools, exposed to Claude Code natively via an MCP server.

        We store the OpenAI-format specs (for their names) and forward them to
        ``_generate`` via ``self.bind``. The actual tools run inside the
        ``claude -p`` subprocess against ``tradingagents.mcp_tools_server``.
        """
        specs = [convert_to_openai_tool(t) for t in tools]
        return self.bind(claude_tools=specs, **kwargs)

    # ------------------------------------------------------- structured output
    def with_structured_output(
        self, schema: Any, *, include_raw: bool = False, **kwargs: Any
    ) -> Runnable:
        """Return a Runnable that uses the CLI's native ``--json-schema`` support."""
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            json_schema = schema.model_json_schema()
            validator = schema.model_validate
        elif isinstance(schema, dict):
            json_schema = schema
            validator = lambda obj: obj  # noqa: E731 — passthrough for raw dict schema
        else:
            raise NotImplementedError(
                f"ChatClaudeCLI.with_structured_output does not support schema type {type(schema)!r}"
            )

        def _invoke(prompt: Any) -> Any:
            messages = self._normalize_input(prompt)
            system_text, prompt_text = self._serialize(messages)
            data = self._invoke_cli(system_text, prompt_text, json_schema=json_schema)
            obj = data.get("structured_output")
            if obj is None:
                # Some responses place the object in ``result`` as a JSON string.
                raw = _strip_code_fences(data.get("result", ""))
                obj = json.loads(raw)
            return validator(obj)

        return RunnableLambda(_invoke)

    # ------------------------------------------------------------- generation
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        claude_tools = kwargs.get("claude_tools")
        json_schema = kwargs.get("json_schema")
        system_text, prompt_text = self._serialize(messages)
        data = self._invoke_cli(
            system_text, prompt_text, claude_tools=claude_tools, json_schema=json_schema
        )
        text = data.get("result", "") or ""
        message = AIMessage(content=text)
        return ChatResult(generations=[ChatGeneration(message=message)])

    # --------------------------------------------------------------- internals
    @staticmethod
    def _normalize_input(prompt: Any) -> List[BaseMessage]:
        """Coerce the various prompt shapes callers pass into a message list."""
        if isinstance(prompt, str):
            return [HumanMessage(content=prompt)]
        if isinstance(prompt, BaseMessage):
            return [prompt]
        if isinstance(prompt, PromptValue):
            return prompt.to_messages()
        return convert_to_messages(prompt)

    def _serialize(self, messages: List[BaseMessage]) -> tuple[str, str]:
        """Split messages into (system prompt, stdin prompt) text for the CLI."""
        name_by_id: dict[str, str] = {}
        for m in messages:
            if isinstance(m, AIMessage):
                for tc in m.tool_calls or []:
                    if tc.get("id"):
                        name_by_id[tc["id"]] = tc.get("name", "tool")

        system_parts = [
            _content_to_text(m.content)
            for m in messages
            if isinstance(m, SystemMessage)
        ]
        non_system = [m for m in messages if not isinstance(m, SystemMessage)]
        single_human = len(non_system) == 1 and isinstance(non_system[0], HumanMessage)

        convo: List[str] = []
        for m in non_system:
            text = _content_to_text(m.content)
            if isinstance(m, HumanMessage):
                convo.append(text if single_human else f"### User:\n{text}")
            elif isinstance(m, AIMessage):
                parts = []
                if text:
                    parts.append(f"### Assistant:\n{text}")
                for tc in m.tool_calls or []:
                    parts.append(
                        f"### Assistant requested tool {tc.get('name')}"
                        f"({json.dumps(tc.get('args', {}))})"
                    )
                if parts:
                    convo.append("\n".join(parts))
            elif isinstance(m, ToolMessage):
                name = name_by_id.get(getattr(m, "tool_call_id", ""), "tool")
                convo.append(f"### Tool result for {name}:\n{text}")
            elif text:
                convo.append(text)

        system_text = "\n\n".join(p for p in system_parts if p) or _FALLBACK_SYSTEM_PROMPT
        prompt_text = "\n\n".join(p for p in convo if p)
        return system_text, prompt_text

    def _invoke_cli(
        self,
        system_text: str,
        prompt_text: str,
        *,
        claude_tools: Optional[list] = None,
        json_schema: Optional[dict] = None,
    ) -> dict:
        """Run ``claude -p`` once and return the parsed JSON response object."""
        from tradingagents.dataflows.config import get_config

        config = get_config()
        claude_bin = self.claude_bin or _find_claude_binary()
        timeout = self.timeout or config.get("claude_cli_timeout", _DEFAULT_TIMEOUT)

        argv = [
            claude_bin,
            "-p",
            "--output-format",
            "json",
            "--model",
            self.model,
            "--system-prompt",
            system_text,
        ]
        if self.effort:
            argv += ["--effort", str(self.effort)]
        if json_schema is not None:
            argv += ["--json-schema", json.dumps(json_schema)]
        if claude_tools:
            mcp_config_path, allowed = self._prepare_mcp(claude_tools, config)
            argv += [
                "--mcp-config",
                mcp_config_path,
                "--strict-mcp-config",
                "--allowedTools",
                ",".join(allowed),
            ]

        env = {k: v for k, v in os.environ.items() if k not in _BILLING_ENV_SCRUB}
        cwd = config.get("data_cache_dir") or None
        if cwd:
            os.makedirs(cwd, exist_ok=True)

        try:
            proc = subprocess.run(
                argv,
                input=prompt_text,
                capture_output=True,
                text=True,
                env=env,
                cwd=cwd,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"claude CLI timed out after {timeout}s (model={self.model})"
            ) from exc

        if not proc.stdout.strip():
            raise RuntimeError(
                f"claude CLI returned no output (exit {proc.returncode}): "
                f"{proc.stderr.strip()[:500]}"
            )

        try:
            data = json.loads(_strip_code_fences(proc.stdout))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"claude CLI returned non-JSON output: {proc.stdout.strip()[:500]}"
            ) from exc

        if data.get("is_error"):
            raise RuntimeError(
                f"claude CLI error: {data.get('result') or data.get('api_error_status')}"
            )
        return data

    def _prepare_mcp(self, claude_tools: list, config: dict) -> tuple[str, List[str]]:
        """Write the MCP server config + a runtime config snapshot; return (path, allowed tools).

        The MCP server is a separate process spawned by ``claude -p``, so it must
        rebuild the framework config from a serialized snapshot before exposing
        the data tools. ``--allowedTools`` is restricted to exactly the bound
        tool subset, which also keeps Claude Code's own Bash/Read/etc. off.
        """
        cache_root = Path(config.get("data_cache_dir") or ".") / "claude_cli"
        cache_root.mkdir(parents=True, exist_ok=True)

        runtime_cfg = cache_root / "runtime_config.json"
        runtime_cfg.write_text(json.dumps(config, default=str), encoding="utf-8")

        mcp_cfg = cache_root / "mcp_config.json"
        mcp_cfg.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        _MCP_SERVER_NAME: {
                            "command": sys.executable,
                            "args": ["-m", "tradingagents.mcp_tools_server"],
                            "env": {
                                "TRADINGAGENTS_MCP_CONFIG_PATH": str(runtime_cfg)
                            },
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

        names = [
            spec["function"]["name"]
            for spec in claude_tools
            if isinstance(spec, dict) and "function" in spec
        ]
        allowed = [f"{_MCP_TOOL_PREFIX}{name}" for name in names]
        return str(mcp_cfg), allowed


class ClaudeCLIClient(BaseLLMClient):
    """Client that returns a :class:`ChatClaudeCLI` driving the local ``claude`` CLI."""

    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        # base_url is irrelevant for the CLI (auth + endpoint come from Claude
        # Code's own login), so it is accepted and ignored.
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        self.warn_if_unknown_model()
        llm_kwargs: dict[str, Any] = {"model": self.model}
        for key in ("effort", "temperature", "callbacks", "timeout"):
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]
        return ChatClaudeCLI(**llm_kwargs)

    def validate_model(self) -> bool:
        return validate_model("claude-cli", self.model)
