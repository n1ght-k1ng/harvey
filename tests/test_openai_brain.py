"""brain.provider: openai — an OpenAI-compatible chat endpoint as the brain."""

import asyncio
import json

import httpx
import pytest
from pydantic import ValidationError

from harvey import brain as brain_mod
from harvey.brain import Brain
from harvey.config import BrainConfig
from harvey.openai_compat import OpenAICompatClient, strip_thinking
from harvey.state import StateManager

BASE = "https://llm.example.test/v1"


def _config(**kw):
    return BrainConfig(provider="openai", base_url=BASE, model="Qwen/Qwen3-32B", **kw)


def _patch_transport(monkeypatch, handler):
    seen = []

    def wrapped(request):
        seen.append(request)
        return handler(request)

    def client(self, timeout):
        return httpx.AsyncClient(transport=httpx.MockTransport(wrapped), timeout=timeout)

    monkeypatch.setattr(OpenAICompatClient, "_client", client)
    monkeypatch.setattr(brain_mod, "RETRY_BASE_DELAY", 0)
    return seen


def _reply(text, finish="stop"):
    return httpx.Response(200, json={
        "model": "Qwen/Qwen3-32B",
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 5},
    })


@pytest.fixture
def state(tmp_path):
    s = StateManager(str(tmp_path / "harvey.db"))
    asyncio.run(s.init_db())
    return s


def test_config_validation():
    assert _config().model == "Qwen/Qwen3-32B"
    assert BrainConfig(provider="openai", base_url="http://localhost:8000/v1/", model="m").base_url \
        == "http://localhost:8000/v1"
    with pytest.raises(ValidationError):
        BrainConfig(provider="openai", model="m")                     # no endpoint
    with pytest.raises(ValidationError):
        BrainConfig(provider="openai", base_url=BASE)                 # no model
    with pytest.raises(ValidationError):
        BrainConfig(provider="openai", base_url="http://llm.example.test/v1", model="m")  # clear text
    with pytest.raises(ValidationError):
        BrainConfig(provider="openai", base_url="https://u:p@llm.example.test/v1", model="m")
    with pytest.raises(ValidationError):
        BrainConfig(provider="openai", base_url=BASE, model="m", ca_cert="no/such/file.pem")
    assert BrainConfig().provider == "claude"


def test_strip_thinking():
    assert strip_thinking("<think>plan it</think>\n\n{\"a\": 1}") == '{"a": 1}'
    assert strip_thinking("<think>never closed because max_tokens ran out") == ""
    assert strip_thinking("  plain  ") == "plain"


def test_think_sends_one_tool_less_request_and_records_usage(state, monkeypatch):
    monkeypatch.setenv("BRAIN_API_KEY", "test-key")
    seen = _patch_transport(monkeypatch, lambda r: _reply("<think>x</think>HARVEY_READY"))
    b = Brain(state, _config(max_tokens=2048))
    out = asyncio.run(b.think("Say it", session_id="harvey-writer-draft"))
    assert out == "HARVEY_READY"
    req = seen[0]
    assert str(req.url) == BASE + "/chat/completions"
    assert req.headers["authorization"] == "Bearer test-key"
    body = json.loads(req.content)
    assert body["model"] == "Qwen/Qwen3-32B" and body["max_tokens"] == 2048
    assert body["messages"] == [{"role": "user", "content": "Say it"}]
    assert "tools" not in body and body["chat_template_kwargs"] == {"enable_thinking": False}
    assert asyncio.run(state.get_usage_today(provider="openai")) == 1
    assert asyncio.run(state.get_usage_today(provider="claude")) == 0


def test_think_json_parses_reply(state, monkeypatch):
    monkeypatch.setenv("BRAIN_API_KEY", "test-key")
    _patch_transport(monkeypatch, lambda r: _reply('```json\n{"score": 7}\n```'))
    assert asyncio.run(Brain(state, _config()).think_json("Score it")) == {"score": 7}


def test_server_errors_retry_then_give_up(state, monkeypatch):
    monkeypatch.setenv("BRAIN_API_KEY", "test-key")
    seen = _patch_transport(monkeypatch, lambda r: httpx.Response(503))
    b = Brain(state, _config())
    assert asyncio.run(b.think("x", max_retries=2)) == ""
    assert len(seen) == 3 and not b.unavailable


def test_refused_key_pauses_until_fixed(state, monkeypatch):
    monkeypatch.setenv("BRAIN_API_KEY", "wrong")
    seen = _patch_transport(monkeypatch, lambda r: httpx.Response(401, json={"error": "nope"}))
    b = Brain(state, _config())
    assert asyncio.run(b.think("x")) == ""
    assert "refused the key" in b.unavailable and len(seen) == 1
    assert asyncio.run(b.think("x")) == "" and len(seen) == 1   # skipped, no request


def test_missing_key_is_a_setup_problem(state, monkeypatch, tmp_path):
    monkeypatch.delenv("BRAIN_API_KEY", raising=False)
    monkeypatch.setattr("harvey.paths.PROJECT_ROOT", tmp_path)   # no .env there
    seen = _patch_transport(monkeypatch, lambda r: _reply("x"))
    b = Brain(state, _config())
    assert "BRAIN_API_KEY" in asyncio.run(b.preflight()) and not seen


def test_preflight_unreachable_is_transient(state, monkeypatch):
    monkeypatch.setenv("BRAIN_API_KEY", "test-key")

    def down(request):
        raise httpx.ConnectTimeout("timed out", request=request)

    _patch_transport(monkeypatch, down)
    b = Brain(state, _config())
    assert "Can't reach" in asyncio.run(b.preflight()) and not b.unavailable

    _patch_transport(monkeypatch, lambda r: httpx.Response(200, json={"data": []}))
    assert asyncio.run(b.preflight()) == ""


def test_budget_falls_back_to_call_count(state, monkeypatch):
    monkeypatch.setenv("BRAIN_API_KEY", "test-key")
    b = Brain(state, _config())
    assert asyncio.run(b.is_within_budget(max_daily_calls=1, max_percent=80)) is True
    asyncio.run(state.increment_usage(provider="openai"))
    assert asyncio.run(b.is_within_budget(max_daily_calls=1, max_percent=80)) is False


def test_key_is_read_from_dotenv_when_not_in_environment(monkeypatch, tmp_path):
    monkeypatch.delenv("BRAIN_API_KEY", raising=False)
    monkeypatch.setattr("harvey.paths.PROJECT_ROOT", tmp_path)
    (tmp_path / ".env").write_text("BRAIN_API_KEY=from-file\n")
    assert OpenAICompatClient(_config()).api_key == "from-file"


def test_pinned_cert_builds_strict_context(tmp_path):
    import ssl
    import subprocess
    key, cert = tmp_path / "k.pem", tmp_path / "c.pem"
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", str(key),
                    "-out", str(cert), "-days", "1", "-subj", "/CN=llm.example.test"],
                   check=True, capture_output=True)
    ctx = OpenAICompatClient(_config(ca_cert=str(cert)))._verify()
    assert isinstance(ctx, ssl.SSLContext) and ctx.verify_mode == ssl.CERT_REQUIRED
    assert OpenAICompatClient(_config())._verify() is True
