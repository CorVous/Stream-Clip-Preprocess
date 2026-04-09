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
    from collections.abc import Callable

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

_PASS1_SYSTEM_PROMPT = (
    "You are an expert stream clip editor. Given a timestamped transcript of a "
    "live stream, identify ALL potentially interesting, funny, exciting, or notable "
    "moments. For each moment, provide a DETAILED description (3-5 sentences) "
    "explaining what makes it interesting and what is happening. Be thorough - "
    "it is better to include too many moments than too few. Each moment should be "
    "at most 3 minutes long; shorter is preferred. "
    "Return your response as a JSON array and nothing else. Each element must have: "
    "start (float seconds), end (float seconds), description (string - detailed "
    "multi-sentence description), clip_name (short snake_case string)."
)

_PASS2_SYSTEM_PROMPT = (
    "You are an expert stream clip curator. You will be given a list of candidate "
    "moments identified from a live stream, each with a detailed description. "
    "Your job is to select only the MOST interesting, funny, or exciting moments "
    "worth clipping. Be selective - pick only the best ones. For each selected "
    "moment, write a concise summary (1 sentence). "
    "IMPORTANT: Verify that the timestamps for each moment are accurate and "
    "match the description. If the description does not match what happens at "
    "the given timestamps, adjust the start/end times to correctly capture the "
    "described moment. "
    "Return your response as a JSON array and nothing else. Each element must have: "
    "start (float seconds), end (float seconds), summary (string - concise "
    "1-sentence summary), clip_name (short snake_case string)."
)

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]+?)\s*```")

# Chunking constants
# Transcript text with timestamps like "[1:23:45] words" tokenizes densely
# (~2.4 chars/token measured); we use 3 as a conservative estimate.
_CHARS_PER_TOKEN = 3
_MAX_RESPONSE_TOKENS = 2048
_OVERHEAD_TOKENS = 350  # system prompt + prompt template + safety margin
_OVERLAP_SECONDS = 120.0  # seconds of overlap between adjacent chunks


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


def build_pass1_prompt(
    stream_type: str,
    game_name: str,
    clip_description: str,
    transcript: str,
) -> str:
    """Build the user prompt for Pass 1 (describe all moments).

    :param stream_type: Type of stream
    :param game_name: Game being played (empty string if none)
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
        "Return a JSON array of ALL potentially interesting moments with fields: "
        "start, end, description, clip_name.\n"
        "Be thorough - include every moment that could be worth clipping."
    )


def build_pass2_prompt(
    stream_type: str,
    game_name: str,
    clip_description: str,
    candidates: list[Moment],
) -> str:
    """Build the user prompt for Pass 2 (select best moments).

    :param stream_type: Type of stream
    :param game_name: Game being played (empty string if none)
    :param clip_description: User description of what to clip
    :param candidates: Candidate moments from Pass 1 with descriptions
    :return: Full user prompt string
    """
    game_line = f"Game: {game_name}\n" if game_name else ""
    candidate_lines: list[str] = []
    for i, c in enumerate(candidates, 1):
        candidate_lines.append(
            f"{i}. [start={int(c.start)}s, end={int(c.end)}s] "
            f"(clip_name: {c.clip_name})\n"
            f"   {c.description}"
        )
    candidates_text = "\n\n".join(candidate_lines)
    return (
        f"Stream type: {stream_type}\n"
        f"{game_line}"
        f"What to clip: {clip_description}\n\n"
        f"Candidate moments:\n{candidates_text}\n\n"
        "Select only the MOST interesting moments from the candidates above. "
        "Verify that each moment's timestamps accurately match its description. "
        "If timestamps are off, correct them. "
        "Return a JSON array with fields: start, end, summary, clip_name."
    )


def parse_candidates_from_response(response: str) -> list[Moment]:
    """Parse a Pass 1 JSON response into Moment objects with descriptions.

    :param response: Raw LLM response string
    :return: List of Moment objects with description populated, summary empty
    :raises ValueError: If JSON cannot be parsed
    """
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
                summary="",
                clip_name=item["clip_name"],
                description=item["description"],
            )
            for item in data
        ]
    except (KeyError, TypeError, ValueError) as exc:
        msg = f"Malformed candidate data in LLM response: {exc}"
        raise ValueError(msg) from exc


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


def chunk_segments(
    segments: list[TranscriptSegment],
    max_transcript_chars: int,
    overlap_seconds: float = _OVERLAP_SECONDS,
) -> list[list[TranscriptSegment]]:
    """Split transcript segments into chunks that fit within a character budget.

    Adjacent chunks overlap by *overlap_seconds* so that moments near
    chunk boundaries are not missed.

    :param segments: Full list of transcript segments
    :param max_transcript_chars: Max characters of formatted transcript per chunk
    :param overlap_seconds: Seconds of overlap between adjacent chunks
    :return: List of segment lists (one per chunk)
    """
    if not segments:
        return [[]]

    chunks: list[list[TranscriptSegment]] = []
    start_idx = 0

    while start_idx < len(segments):
        chunk: list[TranscriptSegment] = []
        char_count = 0
        end_idx = start_idx

        while end_idx < len(segments):
            seg_chars = len(segments[end_idx].format_timestamp()) + 1  # +1 newline
            if char_count + seg_chars > max_transcript_chars and chunk:
                break
            chunk.append(segments[end_idx])
            char_count += seg_chars
            end_idx += 1

        chunks.append(chunk)

        if end_idx >= len(segments):
            break

        # Back up into the overlap region for the next chunk
        overlap_start_time = segments[end_idx].start - overlap_seconds
        next_start = end_idx
        while (
            next_start > start_idx
            and segments[next_start - 1].start >= overlap_start_time
        ):
            next_start -= 1

        # Guarantee forward progress (at least one segment advance)
        start_idx = max(next_start, start_idx + 1)

    return chunks


def deduplicate_moments(
    moments: list[Moment],
    overlap_ratio: float = 0.5,
) -> list[Moment]:
    """Remove duplicate moments whose time ranges overlap significantly.

    When two moments overlap by more than *overlap_ratio* of the shorter
    moment's duration, the longer one is kept.

    :param moments: List of moments (possibly with duplicates)
    :param overlap_ratio: Minimum overlap fraction to consider duplicates
    :return: Deduplicated list sorted by start time
    """
    if not moments:
        return []

    sorted_moments = sorted(moments, key=lambda m: m.start)
    result: list[Moment] = [sorted_moments[0]]

    for m in sorted_moments[1:]:
        prev = result[-1]
        overlap_start = max(prev.start, m.start)
        overlap_end = min(prev.end, m.end)
        overlap = max(0.0, overlap_end - overlap_start)
        shorter_duration = min(prev.duration, m.duration)

        if shorter_duration > 0 and overlap / shorter_duration > overlap_ratio:
            # Keep the longer one
            if m.duration > prev.duration:
                result[-1] = m
        else:
            result.append(m)

    return result


class LLMAnalyzer(abc.ABC):
    """Abstract base class for LLM backends."""

    @abc.abstractmethod
    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Send a prompt to the LLM and return the raw text response.

        :param system_prompt: System prompt string
        :param user_prompt: User prompt string
        :return: Raw LLM response text
        :raises LLMError: If the LLM call fails
        """

    @abc.abstractmethod
    def _get_context_window(self) -> int:
        """Return the model's context window size in tokens."""

    def analyze(
        self,
        segments: list[TranscriptSegment],
        stream_type: str,
        game_name: str,
        clip_description: str,
        on_progress: Callable[[int, int, str], None] | None = None,
    ) -> list[Moment]:
        """Analyze a transcript and return notable moments.

        Uses a two-pass approach:

        1. **Describe** -- chunk the transcript, send each chunk to the LLM
           asking for *all* potentially interesting moments with detailed
           descriptions, then deduplicate across chunks.
        2. **Select** -- send the deduplicated candidates to a second LLM call
           that curates the list and writes concise summaries.

        :param segments: List of transcript segments
        :param stream_type: Type of stream
        :param game_name: Game being played (empty string if none)
        :param clip_description: What kind of clips to find
        :param on_progress: ``(current_step, total_steps, phase)``
        :return: List of identified Moment objects
        :raises LLMError: If the LLM call fails
        """
        # Pass 1: describe all moments
        candidates = self._pass1_describe(
            segments, stream_type, game_name, clip_description, on_progress
        )

        if not candidates:
            return []

        # Pass 2: select the best moments
        return self._pass2_select(
            candidates, stream_type, game_name, clip_description, on_progress
        )

    # ------------------------------------------------------------------
    # Pass 1 — describe
    # ------------------------------------------------------------------

    def _pass1_describe(
        self,
        segments: list[TranscriptSegment],
        stream_type: str,
        game_name: str,
        clip_description: str,
        on_progress: Callable[[int, int, str], None] | None,
    ) -> list[Moment]:
        """Pass 1: find all candidate moments with detailed descriptions."""
        ctx = self._get_context_window()
        available_tokens = ctx - _OVERHEAD_TOKENS - _MAX_RESPONSE_TOKENS
        max_chars = max(available_tokens * _CHARS_PER_TOKEN, 1000)

        chunks = chunk_segments(segments, max_chars)
        total = len(chunks)
        _logger.info(
            "Pass 1: transcript split into %d chunk(s) (context_window=%d)",
            total,
            ctx,
        )

        all_candidates: list[Moment] = []
        for i, chunk in enumerate(chunks):
            current = i + 1
            _logger.info(
                "Pass 1: describing chunk %d/%d (%d segments)",
                current,
                total,
                len(chunk),
            )
            if on_progress is not None:
                on_progress(current, total, "describe")

            transcript = format_transcript_for_llm(chunk)
            prompt = build_pass1_prompt(
                stream_type=stream_type,
                game_name=game_name,
                clip_description=clip_description,
                transcript=transcript,
            )
            try:
                response = self._call_llm(_PASS1_SYSTEM_PROMPT, prompt)
            except LLMError:
                raise
            except Exception as exc:
                msg = f"LLM call failed on chunk {current}/{total}: {exc}"
                raise LLMError(msg) from exc

            try:
                all_candidates.extend(parse_candidates_from_response(response))
            except ValueError as exc:
                _logger.warning("Failed to parse chunk %d response: %s", current, exc)

        return deduplicate_moments(all_candidates)

    # ------------------------------------------------------------------
    # Pass 2 — select
    # ------------------------------------------------------------------

    def _pass2_select(
        self,
        candidates: list[Moment],
        stream_type: str,
        game_name: str,
        clip_description: str,
        on_progress: Callable[[int, int, str], None] | None,
    ) -> list[Moment]:
        """Pass 2: select the best moments and write concise summaries."""
        _logger.info("Pass 2: selecting from %d candidates", len(candidates))

        if on_progress is not None:
            on_progress(1, 1, "select")

        prompt = build_pass2_prompt(
            stream_type=stream_type,
            game_name=game_name,
            clip_description=clip_description,
            candidates=candidates,
        )
        try:
            response = self._call_llm(_PASS2_SYSTEM_PROMPT, prompt)
        except LLMError:
            raise
        except Exception as exc:
            msg = f"LLM call failed during selection: {exc}"
            raise LLMError(msg) from exc

        try:
            selected = parse_moments_from_response(response)
        except ValueError as exc:
            _logger.warning("Failed to parse selection response: %s", exc)
            # Fall back to returning candidates with empty summaries
            return candidates

        # Copy descriptions from candidates onto the selected moments
        desc_map: dict[str, str] = {c.clip_name: c.description for c in candidates}
        for moment in selected:
            if moment.clip_name in desc_map:
                moment.description = desc_map[moment.clip_name]

        return selected
