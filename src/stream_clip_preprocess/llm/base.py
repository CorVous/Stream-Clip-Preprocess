"""Abstract LLM interface and shared utilities."""

from __future__ import annotations

import abc
import json
import logging
import re
from typing import TYPE_CHECKING

from stream_clip_preprocess.models import Moment
from stream_clip_preprocess.transcript import format_transcript_for_llm

if TYPE_CHECKING:
    from stream_clip_preprocess.models import TranscriptSegment

_logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an expert stream clip editor. Given a timestamped transcript of a "
    "live stream, identify the most notable, funny, or exciting moments worth "
    "clipping. Each moment should be at most 3 minutes long; shorter is preferred. "
    "Return your response as a JSON array and nothing else. Each element must have: "
    "start (float seconds), end (float seconds), summary (string), "
    "clip_name (short snake_case string)."
)

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]+?)\s*```")


class LLMError(Exception):
    """Raised when an LLM call fails."""


def build_prompt(
    stream_type: str,
    game_name: str,
    clip_description: str,
    transcript: str,
) -> str:
    """Build the user prompt for the LLM.

    :param stream_type: Type of stream, e.g. "gaming", "just chatting"
    :param game_name: Name of the game being played (empty string if none)
    :param clip_description: User description of what to clip
    :param transcript: Formatted transcript text
    :return: Full user prompt string
    """
    game_line = f"Game: {game_name}\n" if game_name else ""
    return (
        f"Stream type: {stream_type}\n"
        f"{game_line}"
        f"What to clip: {clip_description}\n\n"
        f"Transcript:\n{transcript}\n\n"
        "Return a JSON array of moments with fields: "
        "start, end, summary, clip_name."
    )


def parse_moments_from_response(response: str) -> list[Moment]:
    """Parse a JSON response from the LLM into Moment objects.

    Handles both bare JSON and JSON wrapped in markdown code fences.

    :param response: Raw LLM response string
    :return: List of Moment objects
    :raises ValueError: If JSON cannot be parsed
    """
    # Try to extract from code fence first
    m = _CODE_FENCE_RE.search(response)
    json_text = m.group(1) if m else response.strip()

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        msg = f"Failed to parse LLM response as JSON: {exc}"
        raise ValueError(msg) from exc

    try:
        return [
            Moment(
                start=float(item["start"]),
                end=float(item["end"]),
                summary=item["summary"],
                clip_name=item["clip_name"],
            )
            for item in data
        ]
    except (KeyError, TypeError, ValueError) as exc:
        msg = f"Malformed moment data in LLM response: {exc}"
        raise ValueError(msg) from exc


class LLMAnalyzer(abc.ABC):
    """Abstract base class for LLM backends."""

    @abc.abstractmethod
    def analyze(
        self,
        segments: list[TranscriptSegment],
        stream_type: str,
        game_name: str,
        clip_description: str,
    ) -> list[Moment]:
        """Analyze a transcript and return notable moments.

        :param segments: List of transcript segments
        :param stream_type: Type of stream
        :param game_name: Game being played (empty string if none)
        :param clip_description: What kind of clips to find
        :return: List of identified Moment objects
        :raises LLMError: If the LLM call fails
        """

    def _build_full_prompt(
        self,
        segments: list[TranscriptSegment],
        stream_type: str,
        game_name: str,
        clip_description: str,
    ) -> str:
        """Build the full prompt from segments and context."""
        transcript = format_transcript_for_llm(segments)
        return build_prompt(
            stream_type=stream_type,
            game_name=game_name,
            clip_description=clip_description,
            transcript=transcript,
        )
