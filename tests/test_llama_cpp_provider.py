"""Tests for the llama.cpp (llama-server) local provider."""

from __future__ import annotations

import importlib

import pytest


def _reload_client():
    import tradingagents.llm_clients.openai_client as mod
    return importlib.reload(mod)


# ---- base URL resolution ---------------------------------------------------


def test_resolver_returns_default_when_env_unset(monkeypatch):
    monkeypatch.delenv("LLAMA_CPP_BASE_URL", raising=False)
    mod = _reload_client()
    assert mod._resolve_provider_base_url("llama-cpp") == "http://localhost:8080/v1"


def test_resolver_returns_env_when_set(monkeypatch):
    monkeypatch.setenv("LLAMA_CPP_BASE_URL", "http://gpu-box:8080/v1")
    mod = _reload_client()
    assert mod._resolve_provider_base_url("llama-cpp") == "http://gpu-box:8080/v1"


def test_resolver_evaluation_is_call_time(monkeypatch):
    """Setting the env AFTER module import must still take effect."""
    monkeypatch.delenv("LLAMA_CPP_BASE_URL", raising=False)
    mod = _reload_client()
    monkeypatch.setenv("LLAMA_CPP_BASE_URL", "http://late-set:8080/v1")
    assert mod._resolve_provider_base_url("llama-cpp") == "http://late-set:8080/v1"


def test_env_vars_do_not_cross_providers(monkeypatch):
    """LLAMA_CPP_BASE_URL must not leak into ollama, and vice versa."""
    monkeypatch.setenv("LLAMA_CPP_BASE_URL", "http://llamacpp-host:8080/v1")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama-host:11434/v1")
    mod = _reload_client()
    assert mod._resolve_provider_base_url("llama-cpp") == "http://llamacpp-host:8080/v1"
    assert mod._resolve_provider_base_url("ollama") == "http://ollama-host:11434/v1"


# ---- client construction ---------------------------------------------------


def test_client_get_llm_uses_default_endpoint(monkeypatch):
    monkeypatch.delenv("LLAMA_CPP_BASE_URL", raising=False)
    mod = _reload_client()
    client = mod.OpenAIClient(model="default", provider="llama-cpp")
    llm = client.get_llm()
    assert "localhost:8080" in str(llm.openai_api_base)


def test_client_get_llm_picks_up_env(monkeypatch):
    monkeypatch.setenv("LLAMA_CPP_BASE_URL", "http://my-llamacpp:8080/v1")
    mod = _reload_client()
    client = mod.OpenAIClient(model="default", provider="llama-cpp")
    llm = client.get_llm()
    assert "my-llamacpp" in str(llm.openai_api_base)


def test_explicit_base_url_overrides_env(monkeypatch):
    monkeypatch.setenv("LLAMA_CPP_BASE_URL", "http://env-set:8080/v1")
    mod = _reload_client()
    client = mod.OpenAIClient(
        model="default",
        provider="llama-cpp",
        base_url="http://explicit:8080/v1",
    )
    llm = client.get_llm()
    assert "explicit" in str(llm.openai_api_base)
    assert "env-set" not in str(llm.openai_api_base)


def test_no_api_key_required(monkeypatch):
    """llama-server does not authenticate; get_llm must not raise on a bare env."""
    monkeypatch.delenv("LLAMA_CPP_BASE_URL", raising=False)
    mod = _reload_client()
    client = mod.OpenAIClient(model="default", provider="llama-cpp")
    client.get_llm()  # must not raise ValueError about a missing key


def test_get_llm_returns_llamacpp_chat_class(monkeypatch):
    monkeypatch.delenv("LLAMA_CPP_BASE_URL", raising=False)
    mod = _reload_client()
    client = mod.OpenAIClient(model="default", provider="llama-cpp")
    assert isinstance(client.get_llm(), mod.LlamaCppChatOpenAI)


# ---- structured output method ----------------------------------------------


def test_structured_output_defaults_to_json_schema(monkeypatch):
    """llama.cpp enforces json_schema via grammar; it must be the default method."""
    mod = _reload_client()
    captured = {}

    def fake_super_wso(self, schema, *, method=None, **kwargs):
        captured["method"] = method
        return "bound"

    monkeypatch.setattr(
        mod.NormalizedChatOpenAI, "with_structured_output", fake_super_wso
    )
    llm = mod.LlamaCppChatOpenAI(model="default", api_key="none")
    llm.with_structured_output({"title": "X", "type": "object"})
    assert captured["method"] == "json_schema"


def test_structured_output_explicit_method_wins(monkeypatch):
    mod = _reload_client()
    captured = {}

    def fake_super_wso(self, schema, *, method=None, **kwargs):
        captured["method"] = method
        return "bound"

    monkeypatch.setattr(
        mod.NormalizedChatOpenAI, "with_structured_output", fake_super_wso
    )
    llm = mod.LlamaCppChatOpenAI(model="default", api_key="none")
    llm.with_structured_output(
        {"title": "X", "type": "object"}, method="function_calling"
    )
    assert captured["method"] == "function_calling"


# ---- factory dispatch and aliases -------------------------------------------


@pytest.mark.parametrize(
    "spelling", ["llama-cpp", "llama.cpp", "llama_cpp", "llamacpp", "LLAMA-CPP"]
)
def test_factory_accepts_all_spellings(spelling):
    from tradingagents.llm_clients.factory import create_llm_client
    from tradingagents.llm_clients.openai_client import OpenAIClient

    client = create_llm_client(spelling, "default")
    assert isinstance(client, OpenAIClient)
    assert client.provider == "llama-cpp"


# ---- validation / key mapping / catalog -------------------------------------


def test_any_model_name_is_valid():
    from tradingagents.llm_clients.validators import validate_model

    assert validate_model("llama-cpp", "default")
    assert validate_model("llama-cpp", "qwen3-32b-q4_k_m")
    assert validate_model("llama-cpp", "whatever.gguf")


def test_api_key_env_is_none():
    from tradingagents.llm_clients.api_key_env import get_api_key_env

    assert get_api_key_env("llama-cpp") is None


def test_model_catalog_offers_default_and_custom():
    from tradingagents.llm_clients.model_catalog import get_model_options

    for mode in ("quick", "deep"):
        values = [v for _, v in get_model_options("llama-cpp", mode)]
        assert "default" in values
        assert values[-1] == "custom", f"'custom' should be last entry: {values}"


# ---- CLI surface -------------------------------------------------------------


def test_cli_provider_table_lists_llama_cpp(monkeypatch):
    monkeypatch.delenv("LLAMA_CPP_BASE_URL", raising=False)
    import cli.utils as cli_utils

    table = cli_utils._llm_provider_table()
    entry = next((row for row in table if row[1] == "llama-cpp"), None)
    assert entry is not None, "llama-cpp missing from CLI provider table"
    assert entry[2] == "http://localhost:8080/v1"


def test_cli_provider_table_respects_env(monkeypatch):
    monkeypatch.setenv("LLAMA_CPP_BASE_URL", "http://cli-remote:8080/v1")
    import cli.utils as cli_utils

    table = cli_utils._llm_provider_table()
    entry = next(row for row in table if row[1] == "llama-cpp")
    assert entry[2] == "http://cli-remote:8080/v1"


def test_confirm_endpoint_shows_default(monkeypatch, capsys):
    monkeypatch.delenv("LLAMA_CPP_BASE_URL", raising=False)
    import cli.utils as cli_utils

    cli_utils.confirm_llama_cpp_endpoint("http://localhost:8080/v1")
    out = capsys.readouterr().out
    assert "http://localhost:8080/v1" in out
    assert "LLAMA_CPP_BASE_URL" not in out  # not from env
    assert "Note" not in out  # no warnings for the canonical default


def test_confirm_endpoint_marks_env_origin(monkeypatch, capsys):
    monkeypatch.setenv("LLAMA_CPP_BASE_URL", "http://remote-host:8080/v1")
    import cli.utils as cli_utils

    cli_utils.confirm_llama_cpp_endpoint("http://remote-host:8080/v1")
    out = capsys.readouterr().out
    assert "http://remote-host:8080/v1" in out
    assert "LLAMA_CPP_BASE_URL" in out


def test_confirm_endpoint_warns_on_missing_scheme(monkeypatch, capsys):
    import cli.utils as cli_utils

    cli_utils.confirm_llama_cpp_endpoint("0.0.0.128")
    out = capsys.readouterr().out
    assert "missing a scheme" in out
    assert "http://<host>:8080/v1" in out


def test_confirm_endpoint_warns_on_non_default_port_remote(monkeypatch, capsys):
    import cli.utils as cli_utils

    cli_utils.confirm_llama_cpp_endpoint("http://remote-host/v1")
    out = capsys.readouterr().out
    assert "port 8080" in out
