"""Tests for LLM analyzer."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from stream_clip_preprocess.llm.base import (
    LLMAnalyzer,
    LLMError,
    build_prompt,
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
# LLMAnalyzer (abstract interface)
# ---------------------------------------------------------------------------


class TestLLMAnalyzer:
    """Tests for LLMAnalyzer abstract interface."""

    def test_cannot_instantiate_directly(self) -> None:
        """Test that LLMAnalyzer cannot be instantiated directly."""
        with pytest.raises(TypeError):
            LLMAnalyzer()  # type: ignore[abstract]

    def test_subclass_must_implement_analyze(self) -> None:
        """Test that subclasses must implement analyze."""

        class BadBackend(LLMAnalyzer):
            pass  # Missing analyze()

        with pytest.raises(TypeError):
            BadBackend()  # type: ignore[abstract]


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
        fake_response = json.dumps([
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

        with patch("stream_clip_preprocess.llm.openrouter.httpx.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.raise_for_status.return_value = None
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": fake_response}}]
            }
            mock_post.return_value = mock_resp

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
