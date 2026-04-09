"""Tests for LLM analyzer."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from stream_clip_preprocess.llm.base import (
    LLMAnalyzer,
    LLMError,
    build_pass1_prompt,
    build_pass2_prompt,
    build_prompt,
    chunk_segments,
    deduplicate_moments,
    parse_candidates_from_response,
    parse_moments_from_response,
)
from stream_clip_preprocess.llm.openrouter import OpenRouterBackend
from stream_clip_preprocess.models import (
    LLMBackend,
    LLMConfig,
    Moment,
    TranscriptSegment,
)

# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    """Tests for build_prompt helper."""

    def test_includes_stream_type(self) -> None:
        """Test that built prompt includes stream type."""
        prompt = build_prompt(
            stream_type="gaming",
            game_name="Minecraft",
            clip_description="funny moments",
            transcript="[0:00] Hello",
        )
        assert "gaming" in prompt.lower() or "Minecraft" in prompt

    def test_includes_transcript(self) -> None:
        """Test that built prompt includes transcript content."""
        transcript = "[0:00] Hello world"
        prompt = build_prompt(
            stream_type="gaming",
            game_name="",
            clip_description="highlights",
            transcript=transcript,
        )
        assert transcript in prompt

    def test_includes_clip_description(self) -> None:
        """Test that built prompt includes clip description."""
        prompt = build_prompt(
            stream_type="just chatting",
            game_name="",
            clip_description="funny chat interactions",
            transcript="[0:00] Test",
        )
        assert "funny chat interactions" in prompt

    def test_requests_json_output(self) -> None:
        """Test that prompt instructs model to return JSON."""
        prompt = build_prompt(
            stream_type="gaming",
            game_name="Fortnite",
            clip_description="highlights",
            transcript="[0:00] Test",
        )
        assert "json" in prompt.lower() or "JSON" in prompt


# ---------------------------------------------------------------------------
# build_pass1_prompt / build_pass2_prompt
# ---------------------------------------------------------------------------


class TestBuildPass1Prompt:
    """Tests for Pass 1 (describe) prompt builder."""

    def test_includes_transcript(self) -> None:
        """Test that the prompt includes the transcript text."""
        prompt = build_pass1_prompt(
            stream_type="gaming",
            game_name="",
            clip_description="highlights",
            transcript="[0:00] Hello world",
        )
        assert "[0:00] Hello world" in prompt

    def test_requests_description_field(self) -> None:
        """Test that the prompt asks for a description field."""
        prompt = build_pass1_prompt(
            stream_type="gaming",
            game_name="",
            clip_description="highlights",
            transcript="[0:00] Test",
        )
        assert "description" in prompt.lower()

    def test_encourages_thoroughness(self) -> None:
        """Test that the prompt asks to find ALL moments."""
        prompt = build_pass1_prompt(
            stream_type="gaming",
            game_name="",
            clip_description="highlights",
            transcript="[0:00] Test",
        )
        lower = prompt.lower()
        assert "all" in lower or "thorough" in lower


class TestBuildPass2Prompt:
    """Tests for Pass 2 (select) prompt builder."""

    def test_includes_candidate_descriptions(self) -> None:
        """Test that the prompt contains candidate moment descriptions."""
        candidates = [
            Moment(
                start=10.0,
                end=30.0,
                summary="",
                clip_name="funny",
                description="The streamer fell off a cliff in a hilarious way.",
            ),
        ]
        prompt = build_pass2_prompt(
            stream_type="gaming",
            game_name="Minecraft",
            clip_description="funny moments",
            candidates=candidates,
        )
        assert "fell off a cliff" in prompt

    def test_includes_clip_description(self) -> None:
        """Test that the user's clip description is in the prompt."""
        candidates = [
            Moment(
                start=0.0,
                end=10.0,
                summary="",
                clip_name="m",
                description="A thing happened.",
            ),
        ]
        prompt = build_pass2_prompt(
            stream_type="gaming",
            game_name="",
            clip_description="epic fails",
            candidates=candidates,
        )
        assert "epic fails" in prompt

    def test_requests_summary_field(self) -> None:
        """Test that the prompt asks for a summary field in the output."""
        candidates = [
            Moment(
                start=0.0,
                end=10.0,
                summary="",
                clip_name="m",
                description="Desc.",
            ),
        ]
        prompt = build_pass2_prompt(
            stream_type="gaming",
            game_name="",
            clip_description="highlights",
            candidates=candidates,
        )
        assert "summary" in prompt.lower()

    def test_includes_timestamps(self) -> None:
        """Test that candidate timestamps use raw seconds."""
        candidates = [
            Moment(
                start=125.0,
                end=200.0,
                summary="",
                clip_name="play",
                description="Great play.",
            ),
        ]
        prompt = build_pass2_prompt(
            stream_type="gaming",
            game_name="",
            clip_description="highlights",
            candidates=candidates,
        )
        assert "start=125s" in prompt
        assert "end=200s" in prompt

    def test_instructs_timestamp_verification(self) -> None:
        """Test that the prompt asks the LLM to verify timestamps match."""
        candidates = [
            Moment(
                start=60.0,
                end=120.0,
                summary="",
                clip_name="moment",
                description="Something happened here.",
            ),
        ]
        prompt = build_pass2_prompt(
            stream_type="gaming",
            game_name="",
            clip_description="highlights",
            candidates=candidates,
        )
        # Prompt should instruct the LLM to verify timestamps
        prompt_lower = prompt.lower()
        assert "timestamp" in prompt_lower or "time" in prompt_lower
        has_verify = "verif" in prompt_lower or "match" in prompt_lower
        assert has_verify or "accurat" in prompt_lower


# ---------------------------------------------------------------------------
# parse_candidates_from_response
# ---------------------------------------------------------------------------


class TestParseCandidates:
    """Tests for parsing Pass 1 (describe) LLM responses."""

    def test_parse_valid_candidates(self) -> None:
        """Test parsing a valid candidate JSON array."""
        response = json.dumps([
            {
                "start": 10.0,
                "end": 30.0,
                "description": "The streamer made an incredible play.",
                "clip_name": "incredible_play",
            }
        ])
        moments = parse_candidates_from_response(response)
        assert len(moments) == 1
        assert moments[0].description == "The streamer made an incredible play."
        assert moments[0].clip_name == "incredible_play"

    def test_summary_is_empty(self) -> None:
        """Test that candidates have an empty summary (summary comes in Pass 2)."""
        response = json.dumps([
            {
                "start": 0.0,
                "end": 10.0,
                "description": "Something funny happened.",
                "clip_name": "funny",
            }
        ])
        moments = parse_candidates_from_response(response)
        assert not moments[0].summary

    def test_parse_code_fence_wrapped(self) -> None:
        """Test parsing JSON wrapped in markdown code fences."""
        inner = json.dumps([
            {
                "start": 0.0,
                "end": 5.0,
                "description": "Desc here.",
                "clip_name": "c",
            }
        ])
        response = f"```json\n{inner}\n```"
        moments = parse_candidates_from_response(response)
        assert len(moments) == 1

    def test_parse_empty_array(self) -> None:
        """Test that an empty array returns an empty list."""
        assert parse_candidates_from_response("[]") == []

    def test_parse_missing_description_raises(self) -> None:
        """Test that missing description key raises ValueError."""
        response = json.dumps([{"start": 0.0, "end": 10.0, "clip_name": "c"}])
        with pytest.raises(ValueError, match=r"[Mm]alformed"):
            parse_candidates_from_response(response)


# ---------------------------------------------------------------------------
# Two-pass analyze
# ---------------------------------------------------------------------------


class TestTwoPassAnalyze:
    """Tests for the two-pass (describe → select) analyze flow."""

    def test_analyze_calls_llm_twice_for_single_chunk(self) -> None:
        """Test that analyze makes 2 LLM calls: one describe, one select."""
        call_count = 0
        pass1_response = json.dumps([
            {
                "start": 10.0,
                "end": 30.0,
                "description": "Something cool happened here.",
                "clip_name": "cool_moment",
            }
        ])
        pass2_response = json.dumps([
            {
                "start": 10.0,
                "end": 30.0,
                "summary": "Cool moment",
                "clip_name": "cool_moment",
            }
        ])

        class StubBackend(LLMAnalyzer):
            def _call_llm(
                self,
                system_prompt: str,
                user_prompt: str,
            ) -> str:
                nonlocal call_count
                call_count += 1
                if "curator" not in system_prompt.lower():
                    return pass1_response
                return pass2_response

            def _get_context_window(self) -> int:
                return 8192

        backend = StubBackend()
        segments = [
            TranscriptSegment(text="Hello", start=0.0, duration=10.0),
        ]
        moments = backend.analyze(segments, "gaming", "", "highlights")
        assert call_count == 2
        assert len(moments) == 1
        assert moments[0].summary == "Cool moment"

    def test_analyze_preserves_description(self) -> None:
        """Test that final moments have both summary and description."""
        pass1_response = json.dumps([
            {
                "start": 10.0,
                "end": 30.0,
                "description": "Detailed explanation of what happened.",
                "clip_name": "moment_a",
            }
        ])
        pass2_response = json.dumps([
            {
                "start": 10.0,
                "end": 30.0,
                "summary": "Short summary",
                "clip_name": "moment_a",
            }
        ])

        class StubBackend(LLMAnalyzer):
            def _call_llm(
                self,
                system_prompt: str,
                user_prompt: str,
            ) -> str:
                if "curator" not in system_prompt.lower():
                    return pass1_response
                return pass2_response

            def _get_context_window(self) -> int:
                return 8192

        backend = StubBackend()
        segments = [
            TranscriptSegment(text="Hello", start=0.0, duration=10.0),
        ]
        moments = backend.analyze(segments, "gaming", "", "highlights")
        assert moments[0].summary == "Short summary"
        assert moments[0].description == "Detailed explanation of what happened."

    def test_analyze_pass2_selects_subset(self) -> None:
        """Test that Pass 2 can select a subset of Pass 1 candidates."""
        pass1_response = json.dumps([
            {
                "start": 10.0,
                "end": 30.0,
                "description": "Boring moment.",
                "clip_name": "boring",
            },
            {
                "start": 100.0,
                "end": 130.0,
                "description": "Amazing clutch play that won the game.",
                "clip_name": "clutch",
            },
            {
                "start": 200.0,
                "end": 220.0,
                "description": "Mildly interesting chat reaction.",
                "clip_name": "chat",
            },
        ])
        # Pass 2 only selects the clutch play
        pass2_response = json.dumps([
            {
                "start": 100.0,
                "end": 130.0,
                "summary": "Incredible clutch to win the game",
                "clip_name": "clutch",
            }
        ])

        class StubBackend(LLMAnalyzer):
            def _call_llm(
                self,
                system_prompt: str,
                user_prompt: str,
            ) -> str:
                if "curator" not in system_prompt.lower():
                    return pass1_response
                return pass2_response

            def _get_context_window(self) -> int:
                return 8192

        backend = StubBackend()
        segments = [
            TranscriptSegment(text="Hello", start=0.0, duration=10.0),
        ]
        moments = backend.analyze(segments, "gaming", "", "highlights")
        assert len(moments) == 1
        assert moments[0].clip_name == "clutch"
        assert moments[0].description == "Amazing clutch play that won the game."

    def test_analyze_empty_pass1_skips_pass2(self) -> None:
        """Test that an empty Pass 1 result skips Pass 2 entirely."""
        call_count = 0

        class StubBackend(LLMAnalyzer):
            def _call_llm(
                self,
                system_prompt: str,
                user_prompt: str,
            ) -> str:
                nonlocal call_count
                call_count += 1
                return "[]"

            def _get_context_window(self) -> int:
                return 8192

        backend = StubBackend()
        segments = [
            TranscriptSegment(text="Hello", start=0.0, duration=10.0),
        ]
        moments = backend.analyze(segments, "gaming", "", "highlights")
        assert moments == []
        assert call_count == 1  # Only Pass 1 called

    def test_analyze_progress_includes_both_phases(self) -> None:
        """Test that on_progress reports both describe and select phases."""
        pass1_response = json.dumps([
            {
                "start": 10.0,
                "end": 30.0,
                "description": "A moment.",
                "clip_name": "m",
            }
        ])
        pass2_response = json.dumps([
            {
                "start": 10.0,
                "end": 30.0,
                "summary": "A moment",
                "clip_name": "m",
            }
        ])
        progress_calls: list[tuple[int, int, str]] = []

        class StubBackend(LLMAnalyzer):
            def _call_llm(
                self,
                system_prompt: str,
                user_prompt: str,
            ) -> str:
                if "curator" not in system_prompt.lower():
                    return pass1_response
                return pass2_response

            def _get_context_window(self) -> int:
                return 8192

        backend = StubBackend()
        segments = [
            TranscriptSegment(text="Hello", start=0.0, duration=10.0),
        ]
        backend.analyze(
            segments,
            "gaming",
            "",
            "highlights",
            on_progress=lambda cur, total, phase: progress_calls.append(
                (cur, total, phase),
            ),
        )
        # Should have progress for both phases
        phases = [p[2] for p in progress_calls]
        assert "describe" in phases
        assert "select" in phases


# ---------------------------------------------------------------------------
# LLMAnalyzer (abstract interface)
# ---------------------------------------------------------------------------


class TestLLMAnalyzer:
    """Tests for LLMAnalyzer abstract interface."""

    def test_cannot_instantiate_directly(self) -> None:
        """Test that LLMAnalyzer cannot be instantiated directly."""
        with pytest.raises(TypeError):
            LLMAnalyzer()  # type: ignore[abstract]

    def test_subclass_must_implement_call_llm(self) -> None:
        """Test that subclasses must implement _call_llm and _get_context_window."""

        class BadBackend(LLMAnalyzer):
            pass  # Missing _call_llm() and _get_context_window()

        with pytest.raises(TypeError):
            BadBackend()  # type: ignore[abstract]

    def test_analyze_single_chunk(self) -> None:
        """Test that analyze works for a small transcript (single chunk)."""
        pass1_resp = json.dumps([
            {
                "start": 10.0,
                "end": 30.0,
                "description": "A funny bit where the streamer trips.",
                "clip_name": "funny",
            }
        ])
        pass2_resp = json.dumps([
            {
                "start": 10.0,
                "end": 30.0,
                "summary": "Funny bit",
                "clip_name": "funny",
            }
        ])

        class StubBackend(LLMAnalyzer):
            def _call_llm(
                self,
                system_prompt: str,
                user_prompt: str,
            ) -> str:
                if "curator" not in system_prompt.lower():
                    return pass1_resp
                return pass2_resp

            def _get_context_window(self) -> int:
                return 8192

        backend = StubBackend()
        segments = [
            TranscriptSegment(
                text="Hello world",
                start=0.0,
                duration=10.0,
            )
        ]
        moments = backend.analyze(segments, "gaming", "Minecraft", "funny moments")
        assert len(moments) == 1
        assert moments[0].summary == "Funny bit"

    def test_analyze_chunks_large_transcript(self) -> None:
        """Test that analyze splits a large transcript into multiple LLM calls."""
        pass1_call_count = 0
        pass1_response_a = json.dumps([
            {
                "start": 10.0,
                "end": 30.0,
                "description": "Moment A desc.",
                "clip_name": "a",
            }
        ])
        pass1_response_b = json.dumps([
            {
                "start": 500.0,
                "end": 520.0,
                "description": "Moment B desc.",
                "clip_name": "b",
            }
        ])
        pass2_response = json.dumps([
            {
                "start": 10.0,
                "end": 30.0,
                "summary": "Moment A",
                "clip_name": "a",
            },
            {
                "start": 500.0,
                "end": 520.0,
                "summary": "Moment B",
                "clip_name": "b",
            },
        ])

        class StubBackend(LLMAnalyzer):
            def _call_llm(
                self,
                system_prompt: str,
                user_prompt: str,
            ) -> str:
                nonlocal pass1_call_count
                if "curator" not in system_prompt.lower():
                    pass1_call_count += 1
                    if pass1_call_count == 1:
                        return pass1_response_a
                    return pass1_response_b
                return pass2_response

            def _get_context_window(self) -> int:
                return 300

        backend = StubBackend()
        segments = [
            TranscriptSegment(
                text=f"Segment number {i} with some words",
                start=float(i * 10),
                duration=10.0,
            )
            for i in range(200)
        ]
        moments = backend.analyze(segments, "gaming", "", "highlights")
        assert pass1_call_count >= 2
        assert len(moments) == 2

    def test_analyze_deduplicates_overlapping_moments(self) -> None:
        """Test that analyze deduplicates moments from overlapping chunks."""
        duplicate_candidate = json.dumps([
            {
                "start": 100.0,
                "end": 120.0,
                "description": "Same moment described.",
                "clip_name": "same",
            }
        ])
        pass2_response = json.dumps([
            {
                "start": 100.0,
                "end": 120.0,
                "summary": "Same moment",
                "clip_name": "same",
            }
        ])

        class StubBackend(LLMAnalyzer):
            def _call_llm(
                self,
                system_prompt: str,
                user_prompt: str,
            ) -> str:
                if "curator" not in system_prompt.lower():
                    return duplicate_candidate
                return pass2_response

            def _get_context_window(self) -> int:
                return 300

        backend = StubBackend()
        segments = [
            TranscriptSegment(
                text=f"Segment {i} words here",
                start=float(i * 10),
                duration=10.0,
            )
            for i in range(200)
        ]
        moments = backend.analyze(segments, "gaming", "", "highlights")
        same_moments = [m for m in moments if m.clip_name == "same"]
        assert len(same_moments) == 1

    def test_analyze_calls_on_progress(self) -> None:
        """Test that analyze invokes on_progress for each chunk and phase."""
        pass1_response = json.dumps([
            {
                "start": 10.0,
                "end": 30.0,
                "description": "Moment desc.",
                "clip_name": "m",
            }
        ])
        pass2_response = json.dumps([
            {
                "start": 10.0,
                "end": 30.0,
                "summary": "Moment",
                "clip_name": "m",
            }
        ])
        progress_calls: list[tuple[int, int, str]] = []

        class StubBackend(LLMAnalyzer):
            def _call_llm(
                self,
                system_prompt: str,
                user_prompt: str,
            ) -> str:
                if "curator" not in system_prompt.lower():
                    return pass1_response
                return pass2_response

            def _get_context_window(self) -> int:
                return 300

        backend = StubBackend()
        segments = [
            TranscriptSegment(
                text=f"Segment number {i} with some words",
                start=float(i * 10),
                duration=10.0,
            )
            for i in range(200)
        ]
        backend.analyze(
            segments,
            "gaming",
            "",
            "highlights",
            on_progress=lambda cur, total, phase: progress_calls.append(
                (cur, total, phase),
            ),
        )
        assert len(progress_calls) >= 2
        # First call should be (1, N, "describe")
        assert progress_calls[0][0] == 1
        assert progress_calls[0][2] == "describe"

    def test_analyze_no_progress_callback_ok(self) -> None:
        """Test that analyze works without on_progress callback."""
        pass1_response = json.dumps([
            {
                "start": 10.0,
                "end": 30.0,
                "description": "Moment desc.",
                "clip_name": "m",
            }
        ])
        pass2_response = json.dumps([
            {
                "start": 10.0,
                "end": 30.0,
                "summary": "Moment",
                "clip_name": "m",
            }
        ])

        class StubBackend(LLMAnalyzer):
            def _call_llm(
                self,
                system_prompt: str,
                user_prompt: str,
            ) -> str:
                if "curator" not in system_prompt.lower():
                    return pass1_response
                return pass2_response

            def _get_context_window(self) -> int:
                return 8192

        backend = StubBackend()
        segments = [
            TranscriptSegment(text="Hello", start=0.0, duration=10.0),
        ]
        moments = backend.analyze(segments, "gaming", "", "highlights")
        assert len(moments) == 1


# ---------------------------------------------------------------------------
# chunk_segments
# ---------------------------------------------------------------------------


class TestChunkSegments:
    """Tests for transcript chunking."""

    def test_empty_segments(self) -> None:
        """Test that empty segment list returns a single empty chunk."""
        chunks = chunk_segments([], max_transcript_chars=1000)
        assert chunks == [[]]

    def test_single_chunk_when_fits(self) -> None:
        """Test that small transcript stays in one chunk."""
        segments = [
            TranscriptSegment(text="Hello", start=0.0, duration=5.0),
            TranscriptSegment(text="World", start=5.0, duration=5.0),
        ]
        chunks = chunk_segments(segments, max_transcript_chars=10000)
        assert len(chunks) == 1
        assert chunks[0] == segments

    def test_splits_into_multiple_chunks(self) -> None:
        """Test that a large transcript splits into multiple chunks."""
        segments = [
            TranscriptSegment(
                text=f"This is segment number {i} with enough text",
                start=float(i * 10),
                duration=10.0,
            )
            for i in range(100)
        ]
        # Very small budget forces many chunks
        chunks = chunk_segments(segments, max_transcript_chars=200, overlap_seconds=0)
        assert len(chunks) > 1
        # All segments should appear in at least one chunk
        all_starts = {s.start for chunk in chunks for s in chunk}
        expected_starts = {float(i * 10) for i in range(100)}
        assert all_starts == expected_starts

    def test_overlap_includes_shared_segments(self) -> None:
        """Test that adjacent chunks share segments in the overlap region."""
        segments = [
            TranscriptSegment(
                text=f"Segment {i}",
                start=float(i * 10),
                duration=10.0,
            )
            for i in range(50)
        ]
        chunks = chunk_segments(
            segments,
            max_transcript_chars=200,
            overlap_seconds=30.0,
        )
        assert len(chunks) >= 2
        # The last segment(s) of chunk 0 should appear in chunk 1
        chunk0_end_time = chunks[0][-1].start
        chunk1_start_time = chunks[1][0].start
        assert chunk1_start_time < chunk0_end_time

    def test_always_makes_progress(self) -> None:
        """Test that chunking never gets stuck in an infinite loop."""
        # One very long segment that exceeds budget by itself
        segments = [
            TranscriptSegment(text="x" * 500, start=0.0, duration=10.0),
            TranscriptSegment(text="y" * 500, start=10.0, duration=10.0),
        ]
        chunks = chunk_segments(segments, max_transcript_chars=100)
        # Should still produce chunks (one segment per chunk) rather than loop
        assert len(chunks) == 2


# ---------------------------------------------------------------------------
# deduplicate_moments
# ---------------------------------------------------------------------------


class TestDeduplicateMoments:
    """Tests for moment deduplication."""

    def test_empty_list(self) -> None:
        """Test that empty input returns empty output."""
        assert deduplicate_moments([]) == []

    def test_no_overlap_keeps_all(self) -> None:
        """Test that non-overlapping moments are all kept."""
        moments = [
            Moment(start=0.0, end=30.0, summary="A", clip_name="a"),
            Moment(start=100.0, end=130.0, summary="B", clip_name="b"),
            Moment(start=200.0, end=230.0, summary="C", clip_name="c"),
        ]
        result = deduplicate_moments(moments)
        assert len(result) == 3

    def test_exact_duplicate_removed(self) -> None:
        """Test that identical moments are deduplicated to one."""
        m = Moment(start=10.0, end=30.0, summary="Same", clip_name="same")
        result = deduplicate_moments([m, m])
        assert len(result) == 1
        assert result[0].summary == "Same"

    def test_heavily_overlapping_deduplicated(self) -> None:
        """Test that moments overlapping by >50% are deduplicated."""
        m1 = Moment(start=10.0, end=40.0, summary="First", clip_name="first")
        m2 = Moment(start=15.0, end=45.0, summary="Second", clip_name="second")
        result = deduplicate_moments([m1, m2])
        assert len(result) == 1

    def test_keeps_longer_when_deduplicating(self) -> None:
        """Test that the longer moment is kept when deduplicating."""
        shorter = Moment(start=10.0, end=25.0, summary="Short", clip_name="short")
        longer = Moment(start=10.0, end=40.0, summary="Long", clip_name="long")
        result = deduplicate_moments([shorter, longer])
        assert len(result) == 1
        assert result[0].summary == "Long"

    def test_partial_overlap_kept(self) -> None:
        """Test that moments with minor overlap are both kept."""
        m1 = Moment(start=0.0, end=60.0, summary="A", clip_name="a")
        m2 = Moment(start=50.0, end=120.0, summary="B", clip_name="b")
        # Overlap is 10s out of 60s (16%) — should keep both
        result = deduplicate_moments([m1, m2])
        assert len(result) == 2


# ---------------------------------------------------------------------------
# parse_moments (JSON response parsing)
# ---------------------------------------------------------------------------


class TestParseMoments:
    """Tests for moment parsing from LLM JSON response."""

    def test_parse_valid_json_array(self) -> None:
        """Test parsing a valid JSON array of moments."""
        response = json.dumps([
            {
                "start": 120.0,
                "end": 180.0,
                "summary": "Epic fail moment",
                "clip_name": "epic_fail",
            },
            {
                "start": 300.0,
                "end": 360.0,
                "summary": "Funny chat interaction",
                "clip_name": "chat_funny",
            },
        ])
        moments = parse_moments_from_response(response)
        assert len(moments) == 2
        assert moments[0].start == pytest.approx(120.0)
        assert moments[0].summary == "Epic fail moment"
        assert moments[1].clip_name == "chat_funny"

    def test_parse_json_wrapped_in_markdown(self) -> None:
        """Test parsing JSON that's wrapped in markdown code blocks."""
        response = """Here are the moments I found:

```json
[{"start": 60.0, "end": 90.0, "summary": "Funny moment", "clip_name": "funny"}]
```

Those are the highlights."""
        moments = parse_moments_from_response(response)
        assert len(moments) == 1
        assert moments[0].summary == "Funny moment"

    def test_parse_empty_array(self) -> None:
        """Test parsing an empty array."""
        moments = parse_moments_from_response("[]")
        assert moments == []

    def test_parse_invalid_json_raises(self) -> None:
        """Test that invalid JSON raises ValueError."""
        with pytest.raises(ValueError, match="parse"):
            parse_moments_from_response("not json at all")

    def test_parse_missing_keys_raises(self) -> None:
        """Test that missing required keys in moment data raises ValueError."""
        # Missing summary and clip_name keys
        response = json.dumps([{"start": 10.0, "end": 20.0}])
        with pytest.raises(ValueError, match="Malformed moment data"):
            parse_moments_from_response(response)

    def test_parse_non_array_json_raises(self) -> None:
        """Test that a JSON object (not array) raises a meaningful error."""
        with pytest.raises((ValueError, TypeError)):
            parse_moments_from_response('{"start": 10.0}')

    def test_parse_string_number_coercion(self) -> None:
        """Test that string values where floats expected raises ValueError."""
        response = json.dumps([
            {"start": "not_a_number", "end": 20.0, "summary": "X", "clip_name": "x"}
        ])
        with pytest.raises(ValueError, match="Malformed moment data"):
            parse_moments_from_response(response)

    def test_moments_are_selected_by_default(self) -> None:
        """Test that parsed moments default to selected=True."""
        response = json.dumps([
            {"start": 10.0, "end": 20.0, "summary": "Test", "clip_name": "test"}
        ])
        moments = parse_moments_from_response(response)
        assert moments[0].selected is True


# ---------------------------------------------------------------------------
# OpenRouterBackend
# ---------------------------------------------------------------------------


class TestOpenRouterBackend:
    """Tests for OpenRouter LLM backend."""

    def test_analyze_returns_moments(self) -> None:
        """Test that analyze() returns a list of Moment objects."""
        pass1_response = json.dumps([
            {
                "start": 100.0,
                "end": 160.0,
                "description": "An incredible play during a crucial round.",
                "clip_name": "great_play",
            }
        ])
        pass2_response = json.dumps([
            {
                "start": 100.0,
                "end": 160.0,
                "summary": "Great play",
                "clip_name": "great_play",
            }
        ])
        config = LLMConfig(
            backend=LLMBackend.OPENROUTER,
            api_key="sk-or-test",
            model_name="meta-llama/llama-3.1-8b-instruct",
        )
        call_count = 0

        with patch("stream_clip_preprocess.llm.openrouter.httpx.post") as mock_post:

            def _fake_post(*_args: object, **_kwargs: object) -> MagicMock:
                nonlocal call_count
                call_count += 1
                resp = MagicMock()
                resp.raise_for_status.return_value = None
                content = pass1_response if call_count == 1 else pass2_response
                resp.json.return_value = {
                    "choices": [{"message": {"content": content}}]
                }
                return resp

            mock_post.side_effect = _fake_post

            backend = OpenRouterBackend(config)
            segments = [
                TranscriptSegment(text="Epic play!", start=100.0, duration=60.0)
            ]
            moments = backend.analyze(
                segments=segments,
                stream_type="gaming",
                game_name="Fortnite",
                clip_description="highlights",
            )

        assert len(moments) == 1
        assert isinstance(moments[0], Moment)
        assert moments[0].summary == "Great play"

    def test_analyze_raises_on_http_error(self) -> None:
        """Test that HTTP errors are wrapped as LLMError."""
        config = LLMConfig(
            backend=LLMBackend.OPENROUTER,
            api_key="sk-or-test",
            model_name="test-model",
        )

        with patch("stream_clip_preprocess.llm.openrouter.httpx.post") as mock_post:
            mock_post.side_effect = httpx.HTTPError("connection failed")

            backend = OpenRouterBackend(config)
            with pytest.raises(LLMError):
                backend.analyze(
                    segments=[],
                    stream_type="gaming",
                    game_name="",
                    clip_description="highlights",
                )

    def test_analyze_raises_on_malformed_response_missing_choices(self) -> None:
        """Test that malformed API response (missing choices) raises LLMError."""
        config = LLMConfig(
            backend=LLMBackend.OPENROUTER,
            api_key="sk-or-test",
            model_name="test-model",
        )

        with patch("stream_clip_preprocess.llm.openrouter.httpx.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.raise_for_status.return_value = None
            mock_resp.json.return_value = {"error": "something went wrong"}
            mock_post.return_value = mock_resp

            backend = OpenRouterBackend(config)
            with pytest.raises(LLMError):
                backend.analyze(
                    segments=[],
                    stream_type="gaming",
                    game_name="",
                    clip_description="highlights",
                )

    def test_analyze_raises_on_malformed_response_empty_choices(self) -> None:
        """Test that malformed API response (empty choices) raises LLMError."""
        config = LLMConfig(
            backend=LLMBackend.OPENROUTER,
            api_key="sk-or-test",
            model_name="test-model",
        )

        with patch("stream_clip_preprocess.llm.openrouter.httpx.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.raise_for_status.return_value = None
            mock_resp.json.return_value = {"choices": []}
            mock_post.return_value = mock_resp

            backend = OpenRouterBackend(config)
            with pytest.raises(LLMError):
                backend.analyze(
                    segments=[],
                    stream_type="gaming",
                    game_name="",
                    clip_description="highlights",
                )

    def test_builds_correct_api_payload(self) -> None:
        """Test that the API payload includes model and messages."""
        config = LLMConfig(
            backend=LLMBackend.OPENROUTER,
            api_key="sk-or-test",
            model_name="my-model",
        )

        with patch("stream_clip_preprocess.llm.openrouter.httpx.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.raise_for_status.return_value = None
            mock_resp.json.return_value = {"choices": [{"message": {"content": "[]"}}]}
            mock_post.return_value = mock_resp

            backend = OpenRouterBackend(config)
            backend.analyze(
                segments=[],
                stream_type="gaming",
                game_name="",
                clip_description="test",
            )

        _, kwargs = mock_post.call_args
        payload = kwargs.get("json", {})
        assert payload.get("model") == "my-model"
        assert "messages" in payload

    def test_fetches_context_window_from_api(self) -> None:
        """Test that _get_context_window queries the OpenRouter models API."""
        config = LLMConfig(
            backend=LLMBackend.OPENROUTER,
            api_key="sk-or-test",
            model_name="anthropic/claude-opus-4",
        )

        with patch("stream_clip_preprocess.llm.openrouter.httpx.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.raise_for_status.return_value = None
            mock_resp.json.return_value = {
                "data": [
                    {"id": "anthropic/claude-opus-4", "context_length": 200000},
                    {"id": "other/model", "context_length": 8192},
                ],
            }
            mock_get.return_value = mock_resp

            backend = OpenRouterBackend(config)
            ctx = backend._get_context_window()  # noqa: SLF001

        assert ctx == 200000

    def test_context_window_cached_after_first_fetch(self) -> None:
        """Test that the context window is only fetched once."""
        config = LLMConfig(
            backend=LLMBackend.OPENROUTER,
            api_key="sk-or-test",
            model_name="anthropic/claude-opus-4",
        )

        with patch("stream_clip_preprocess.llm.openrouter.httpx.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.raise_for_status.return_value = None
            mock_resp.json.return_value = {
                "data": [
                    {"id": "anthropic/claude-opus-4", "context_length": 200000},
                ],
            }
            mock_get.return_value = mock_resp

            backend = OpenRouterBackend(config)
            backend._get_context_window()  # noqa: SLF001
            backend._get_context_window()  # noqa: SLF001

        assert mock_get.call_count == 1

    def test_context_window_fallback_on_model_not_found(self) -> None:
        """Test that an unknown model falls back to the default."""
        config = LLMConfig(
            backend=LLMBackend.OPENROUTER,
            api_key="sk-or-test",
            model_name="some/unknown-model",
        )

        with patch("stream_clip_preprocess.llm.openrouter.httpx.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.raise_for_status.return_value = None
            mock_resp.json.return_value = {
                "data": [
                    {"id": "other/model", "context_length": 8192},
                ],
            }
            mock_get.return_value = mock_resp

            backend = OpenRouterBackend(config)
            ctx = backend._get_context_window()  # noqa: SLF001

        assert ctx == 128000

    def test_context_window_fallback_on_api_error(self) -> None:
        """Test that a failed API call falls back to a large default."""
        config = LLMConfig(
            backend=LLMBackend.OPENROUTER,
            api_key="sk-or-test",
            model_name="some/unknown-model",
        )

        with patch("stream_clip_preprocess.llm.openrouter.httpx.get") as mock_get:
            mock_get.side_effect = httpx.HTTPError("not found")

            backend = OpenRouterBackend(config)
            ctx = backend._get_context_window()  # noqa: SLF001

        assert ctx == 128000
