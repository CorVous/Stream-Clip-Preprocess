"""OpenRouter LLM backend."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import httpx

from stream_clip_preprocess.llm.base import (
    LLMAnalyzer,
    LLMError,
)

if TYPE_CHECKING:
    from stream_clip_preprocess.models import LLMConfig

_logger = logging.getLogger(__name__)

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
_FALLBACK_CONTEXT_WINDOW = 128_000


class OpenRouterBackend(LLMAnalyzer):
    """Calls the OpenRouter API to analyze transcripts."""

    def __init__(self, config: LLMConfig) -> None:
        """Initialize with LLM configuration.

        :param config: LLMConfig with backend=OPENROUTER and api_key set
        """
        self.config = config
        self._cached_context_window: int | None = None

    def _get_context_window(self) -> int:
        """Fetch the model's context window from the OpenRouter API.

        Queries ``/api/v1/models`` on first call, finds the matching model,
        and caches the result.  Falls back to 128k tokens on any error.

        :return: Context window size in tokens
        """
        if self._cached_context_window is not None:
            return self._cached_context_window

        try:
            resp = httpx.get(_OPENROUTER_MODELS_URL, timeout=15.0)
            resp.raise_for_status()
            models = resp.json().get("data", [])
            match = next(
                (m for m in models if m.get("id") == self.config.model_name),
                None,
            )
            if match is None:
                msg = f"model {self.config.model_name!r} not found in models list"
                raise LookupError(msg)  # noqa: TRY301
            ctx = int(match["context_length"])
            _logger.info(
                "Model %s context window: %d tokens",
                self.config.model_name,
                ctx,
            )
        except (httpx.HTTPError, KeyError, TypeError, ValueError, LookupError) as exc:
            _logger.warning(
                "Could not fetch context window for %s, using fallback (%d tokens): %s",
                self.config.model_name,
                _FALLBACK_CONTEXT_WINDOW,
                exc,
            )
            ctx = _FALLBACK_CONTEXT_WINDOW

        self._cached_context_window = ctx
        return ctx

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call the OpenRouter API and return the raw text response.

        :param system_prompt: System prompt string
        :param user_prompt: User prompt string
        :return: Raw LLM response text
        :raises LLMError: If the API call fails
        """
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        _logger.debug("Calling OpenRouter model=%s", self.config.model_name)

        try:
            resp = httpx.post(
                _OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=120.0,
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            msg = f"OpenRouter API call failed: {exc}"
            raise LLMError(msg) from exc

        try:
            return resp.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            msg = f"Unexpected OpenRouter response format: {exc}"
            raise LLMError(msg) from exc
