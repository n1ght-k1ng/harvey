"""OpenAI-compatible chat endpoint as Harvey's brain (``brain.provider: openai``).

For a self-hosted model (vLLM, SGLang, llama.cpp server, Ollama, LM Studio)
or any hosted API that speaks ``POST /v1/chat/completions``. One request per
call, prompt in a single user message, text back: no tools are offered, so
the model can only return text, the same contract as the Claude and Codex
runtimes.

Settings (harvey.yaml ``brain:``): ``base_url`` (ends in /v1), ``model``,
optional ``max_tokens``, ``temperature``, ``ca_cert`` (a PEM file to pin for
a self-signed server) and ``tls_verify`` (false skips certificate checks:
anyone on the network path could then read prospects and the key). The key
is ``BRAIN_API_KEY`` in .env, never the config file.
"""

from __future__ import annotations

import logging
import os
import re
import ssl
import time
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger("harvey.openai_compat")

API_KEY_ENV = "BRAIN_API_KEY"
PREFLIGHT_TIMEOUT = 15.0

# Reasoning models (Qwen3, DeepSeek-R1) may put their thinking inline.
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


class OpenAICompatError(Exception):
    """kind: 'setup' (config or key missing), 'auth' (key refused),
    'transient' (network, 429, 5xx), 'bad_response'."""

    def __init__(self, message: str, kind: str = "transient"):
        super().__init__(message)
        self.kind = kind

    @property
    def retryable(self) -> bool:
        return self.kind == "transient"


@dataclass
class Completion:
    text: str
    model: str
    duration_ms: int
    usage: dict = field(default_factory=dict)


def strip_thinking(text: str) -> str:
    """Drop inline <think> blocks, and a dangling one cut off by max_tokens."""
    text = _THINK_RE.sub("", text or "")
    if re.match(r"\s*<think>", text, re.IGNORECASE):
        return ""
    return text.strip()


class OpenAICompatClient:
    def __init__(self, config, api_key: str | None = None):
        self.base_url = (config.base_url or "").rstrip("/")
        self.model = config.model
        self.max_tokens = config.max_tokens
        self.temperature = config.temperature
        self.disable_thinking = config.disable_thinking
        self.ca_cert = config.ca_cert
        self.tls_verify = config.tls_verify
        self._api_key = api_key

    @property
    def api_key(self) -> str:
        if self._api_key is not None:
            return self._api_key.strip()
        key = os.environ.get(API_KEY_ENV, "").strip()
        if not key:
            # Not every entry point loads .env first (`harvey status`).
            from dotenv import dotenv_values
            from harvey.paths import PROJECT_ROOT
            key = str(dotenv_values(PROJECT_ROOT / ".env").get(API_KEY_ENV) or "").strip()
        return key

    def fingerprint(self):
        return (self.base_url, self.model, bool(self.api_key))

    def _verify(self):
        if self.ca_cert:
            # Trust only this certificate (a self-signed server's own cert).
            # Many such certs name no host in subjectAltName, or the server
            # is reached by IP, so the name match is skipped; the chain
            # check against this single trusted cert still applies.
            ctx = ssl.create_default_context(cafile=self.ca_cert)
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_REQUIRED
            return ctx
        return self.tls_verify

    def _client(self, timeout: float) -> httpx.AsyncClient:
        if not self.tls_verify and not self.ca_cert:
            logger.warning("Brain endpoint TLS certificate is NOT verified (brain.tls_verify: false).")
        return httpx.AsyncClient(timeout=timeout, verify=self._verify(), follow_redirects=False)

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _check_setup(self):
        if not self.base_url:
            raise OpenAICompatError("brain.base_url is not set in harvey.yaml.", "setup")
        if not self.model:
            raise OpenAICompatError("brain.model is not set in harvey.yaml.", "setup")
        if not self.api_key:
            raise OpenAICompatError(f"{API_KEY_ENV} is not set in .env.", "setup")

    @staticmethod
    def _status_error(r: httpx.Response) -> OpenAICompatError:
        if r.status_code in (401, 403):
            return OpenAICompatError(
                f"The brain endpoint refused the key (HTTP {r.status_code}); check {API_KEY_ENV}.", "auth")
        if r.status_code == 404:
            return OpenAICompatError(
                "The brain endpoint returned 404; brain.base_url should end in /v1.", "setup")
        if r.status_code == 429 or r.status_code >= 500:
            return OpenAICompatError(f"The brain endpoint is busy or down (HTTP {r.status_code}).")
        # Never echo the body: it can quote the prompt back.
        return OpenAICompatError(f"The brain endpoint rejected the request (HTTP {r.status_code}).",
                                 "bad_response")

    async def preflight(self) -> None:
        """Key present, endpoint reachable, key accepted. No prompt sent."""
        self._check_setup()
        try:
            async with self._client(PREFLIGHT_TIMEOUT) as c:
                r = await c.get(self.base_url + "/models", headers=self._headers())
        except httpx.HTTPError as e:
            raise OpenAICompatError(
                f"Can't reach the brain endpoint {self.base_url} ({type(e).__name__}).")
        if r.status_code != 200:
            raise self._status_error(r)

    async def run(self, prompt: str, timeout: float) -> Completion:
        self._check_setup()
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.disable_thinking:
            # vLLM / SGLang pass this to the chat template (Qwen3 honours it);
            # servers that don't know it ignore it.
            body["chat_template_kwargs"] = {"enable_thinking": False}
        started = time.monotonic()
        try:
            async with self._client(timeout) as c:
                r = await c.post(self.base_url + "/chat/completions",
                                 headers=self._headers(), json=body)
        except httpx.TimeoutException:
            raise OpenAICompatError(f"The brain endpoint timed out after {timeout:.0f}s.")
        except httpx.HTTPError as e:
            raise OpenAICompatError(
                f"Can't reach the brain endpoint {self.base_url} ({type(e).__name__}).")
        if r.status_code != 200:
            raise self._status_error(r)
        try:
            data = r.json()
            choice = data["choices"][0]
            text = choice["message"].get("content") or ""
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            raise OpenAICompatError("The brain endpoint sent a reply Harvey can't read.", "bad_response")
        if choice.get("finish_reason") == "length":
            logger.warning(f"Brain reply hit max_tokens ({self.max_tokens}); it may be cut off. "
                           "Raise brain.max_tokens if replies look truncated.")
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        return Completion(
            text=strip_thinking(text),
            model=str(data.get("model") or self.model),
            duration_ms=int((time.monotonic() - started) * 1000),
            usage={
                "input_tokens": int(usage.get("prompt_tokens") or 0),
                "output_tokens": int(usage.get("completion_tokens") or 0),
            },
        )
