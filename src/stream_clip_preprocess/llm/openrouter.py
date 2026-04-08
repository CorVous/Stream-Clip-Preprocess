"""OpenRouter LLM backend."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import httpx

from stream_clip_preprocess.llm.base import (
    _SYSTEM_PROMPT,
    LLMAnalyzer,
    LLMError,
    parse_moments_from_response,
)

if TYPE_CHECKING:
    from stream_clip_preprocess.models import LLMConfig, Moment, TranscriptSegment

_logger = logging.getLogger(__name__)

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterBackend(LLMAnalyzer):
    """Calls the OpenRouter API to analyze transcripts."""

    def __init__(self, config: LLMConfig) -> None:
        """Initialize with LLM configuration.

        :param config: LLMConfig with backend=OPENROUTER and api_key set
        """
        self.config = config

    def analyze(
        self,
        segments: list[TranscriptSegment],
        stream_type: str,
        game_name: str,
        clip_description: str,
    ) -> list[Moment]:
        """Call OpenRouter API and return identified moments.

        :param segments: Transcript segments to analyze
        :param stream_type: Type of stream
        :param game_name: Game being played
        :param clip_description: What to clip
        :return: List of Moment objects
        :raises LLMError: If the API call fails
        """
        user_prompt = self._build_full_prompt(
            segments, stream_type, game_name, clip_description
        )

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.model_name,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
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
            content = resp.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            msg = f"Unexpected OpenRouter response format: {exc}"
            raise LLMError(msg) from exc

        try:
            return parse_moments_from_response(content)
        except ValueError as exc:
            raise LLMError(str(exc)) from exc
