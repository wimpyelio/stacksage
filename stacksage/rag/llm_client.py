"""
Provider-agnostic LLM client for StackSage.

Auto-detects provider from environment:
  • LLM_API_KEY set   → any OpenAI-compatible API (Groq, OpenAI, Together, …)
  • LLM_API_KEY unset → Ollama (local, no key required)

The *judge* role independently resolves its provider via LLM_JUDGE_* vars,
so generation and evaluation can use different models or providers.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

_UNSET = {"", "your_api_key_here"}


class LLMClient:
    """
    Unified LLM client that works with any OpenAI-compatible REST API or Ollama.

    Args:
        role:        "generation" (main pipeline) or "judge" (evaluation stage).
                     Each role reads independent env-var sets so they can use
                     different providers / models.
        max_retries: Retry count on transient HTTP errors (429, 5xx).
        timeout:     Per-request wall-clock timeout in seconds.
    """

    def __init__(
        self,
        role: str = "generation",
        max_retries: int = 3,
        timeout: int = 30,
    ) -> None:
        self.role = role
        self.max_retries = max_retries
        self.timeout = timeout

        if role == "judge":
            self.api_key = os.getenv("LLM_JUDGE_API_KEY") or os.getenv("LLM_API_KEY", "")
            self.base_url = (
                os.getenv("LLM_JUDGE_BASE_URL")
                or os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
            )
            self.model = os.getenv("LLM_JUDGE_MODEL", "llama-3.1-70b-versatile")
            self.ollama_model = os.getenv("OLLAMA_JUDGE_MODEL", "llama3.1:8b")
        else:
            self.api_key = os.getenv("LLM_API_KEY", "")
            self.base_url = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
            self.model = os.getenv("LLM_GENERATION_MODEL", "llama-3.1-8b-instant")
            self.ollama_model = os.getenv("OLLAMA_GENERATION_MODEL", "llama3.1:8b")

        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        if self.api_key and self.api_key not in _UNSET:
            self._provider = "openai_compat"
            logger.info(
                "[LLMClient:%s] provider=openai_compat  base=%s  model=%s",
                role, self.base_url, self.model,
            )
        else:
            self._provider = "ollama"
            logger.info(
                "[LLMClient:%s] provider=ollama  base=%s  model=%s",
                role, self.ollama_base_url, self.ollama_model,
            )

    # ── Public API ─────────────────────────────────────────────────────────────

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        """
        Send a chat-completion request to the configured provider.

        Args:
            prompt:      User message / query.
            system:      Optional system prompt.
            temperature: Sampling temperature (0 = deterministic).
            max_tokens:  Response token budget.

        Returns:
            Model response text, whitespace-stripped.

        Raises:
            RuntimeError: All retries exhausted — caller should handle gracefully.
        """
        if self._provider == "openai_compat":
            return self._call_openai_compat(prompt, system, temperature, max_tokens)
        return self._call_ollama(prompt, system, temperature, max_tokens)

    @property
    def provider(self) -> str:
        """Active provider: "openai_compat" or "ollama"."""
        return self._provider

    @property
    def active_model(self) -> str:
        """Model name that will be used for calls."""
        return self.model if self._provider == "openai_compat" else self.ollama_model

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _build_messages(prompt: str, system: Optional[str]) -> list[dict]:
        msgs: list[dict] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def _call_openai_compat(
        self, prompt: str, system: Optional[str], temperature: float, max_tokens: int
    ) -> str:
        url = self.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.model,
            "messages": self._build_messages(prompt, system),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        return self._post_with_retry(url, payload, headers, _parse_openai_compat)

    def _call_ollama(
        self, prompt: str, system: Optional[str], temperature: float, max_tokens: int
    ) -> str:
        url = self.ollama_base_url.rstrip("/") + "/api/chat"
        payload = {
            "model": self.ollama_model,
            "messages": self._build_messages(prompt, system),
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        return self._post_with_retry(url, payload, {}, _parse_ollama)

    def _post_with_retry(self, url: str, payload: dict, headers: dict, parser) -> str:
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(url, json=payload, headers=headers)
                    resp.raise_for_status()
                    return parser(resp.json())
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                if exc.response.status_code in (429, 500, 502, 503, 504):
                    wait = 2 ** attempt
                    logger.warning(
                        "[LLMClient] HTTP %s — retry %d/%d in %ds",
                        exc.response.status_code, attempt, self.max_retries, wait,
                    )
                    time.sleep(wait)
                else:
                    break  # non-retryable (400, 401, 404…)
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_exc = exc
                wait = 2 ** attempt
                logger.warning(
                    "[LLMClient] network error — retry %d/%d in %ds: %s",
                    attempt, self.max_retries, wait, exc,
                )
                time.sleep(wait)
        raise RuntimeError(
            f"LLM call failed after {self.max_retries} retries: {last_exc}"
        )


# ── Response parsers ───────────────────────────────────────────────────────────

def _parse_openai_compat(data: dict) -> str:
    return data["choices"][0]["message"]["content"].strip()


def _parse_ollama(data: dict) -> str:
    return data["message"]["content"].strip()
