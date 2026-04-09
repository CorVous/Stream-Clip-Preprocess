"""Local LLM backend using llama-cpp-python."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from stream_clip_preprocess.llm.base import (
    LLMAnalyzer,
    LLMError,
)

if TYPE_CHECKING:
    from stream_clip_preprocess.models import LLMConfig

_logger = logging.getLogger(__name__)


class LocalBackend(LLMAnalyzer):
    """Runs a local GGUF model via llama-cpp-python."""

    def __init__(self, config: LLMConfig) -> None:
        """Initialize with LLM configuration.

        :param config: LLMConfig with backend=LOCAL and model_path set
        """
        self.config = config
        self._llm: Any = None

    def _get_context_window(self) -> int:
        """Return the configured context window size."""
        return self.config.context_window

    def _load_model(self) -> Any:
        """Lazily load the GGUF model.

        :return: Llama model instance
        :raises LLMError: If llama-cpp-python is unavailable or model fails to load
        """
        if self._llm is not None:
            return self._llm

        try:
            from llama_cpp import (  # type: ignore[import-not-found]  # noqa: PLC0415
                Llama,
            )
        except ImportError as exc:
            msg = (
                "llama-cpp-python is not installed. "
                "Install it with: pip install llama-cpp-python"
            )
            raise LLMError(msg) from exc

        if self.config.model_path is None:
            msg = "model_path must be set for LOCAL backend"
            raise LLMError(msg)

        _logger.info("Loading local model from %s", self.config.model_path)
        try:
            self._llm = Llama(
                model_path=str(self.config.model_path),
                n_ctx=self.config.context_window,
                verbose=False,
            )
        except Exception as exc:
            msg = f"Failed to load model: {exc}"
            raise LLMError(msg) from exc

        return self._llm

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Run local inference and return the raw text response.

        :param system_prompt: System prompt string
        :param user_prompt: User prompt string
        :return: Raw LLM response text
        :raises LLMError: If inference fails
        """
        llm = self._load_model()

        _logger.debug("Running local inference")
        try:
            result = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=2048,
                temperature=0.1,
            )
            return result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            msg = f"Unexpected local LLM response format: {exc}"
            raise LLMError(msg) from exc
        except Exception as exc:
            msg = f"Local inference failed: {exc}"
            raise LLMError(msg) from exc
