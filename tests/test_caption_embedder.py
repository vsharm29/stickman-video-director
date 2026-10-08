"""Unit and integration tests for Caption Embedder module."""

import json
from pathlib import Path
import pytest

from src.caption_embedder import (
    format_srt_timestamp,
    generate_srt_from_json,
    embed_captions,
    clean_caption_text,
)


def test_clean_caption_text():
    """Verify that clean_caption_text strips apostrophes, quotes, commas, and periods."""
    sample = "Every time you say 'I will start tomorrow,' you drop another heavy stone into the bag."
    cleaned = clean_caption_text(sample)
    assert cleaned == "Every time you say I will start tomorrow you drop another heavy stone into the bag"
    assert "'" not in cleaned
    assert "," not in cleaned
    assert "." not in cleaned

    # Test curly quotes
    curly = "Notice â€˜thisâ€™ and â€œthatâ€�, right now."
    assert clean_caption_text(curly) == "Notice this and that right now"


def test_format_srt_timestamp():
    """Verify conversion of seconds to standard SRT timestamp format."""
    assert format_srt_timestamp(0.0) == "00:00:00,000"
    assert format_srt_timestamp(10.0) == "00:00:10,000"
    assert format_srt_timestamp(65.5) == "00:01:05,500"
    assert format_srt_timestamp(3661.123) == "01:01:01,123"
    assert format_srt_timestamp(-5.0) == "00:00:00,000"


def test_generate_srt_phrase_level(tmp_path):
    """Verify accurate phrase-level SRT generation with 2-4 word speech chunks."""
    sample_json = Path("sample_data/phase_b_prompts.json")
    assert sample_json.exists(), "sample_data/phase_b_prompts.json must exist"

    test_srt = tmp_path / "test_dynamic_captions.srt"
    result = generate_srt_from_json(sample_json, output_srt_path=test_srt, phrase_level=True)

    assert result.exists()
    content = result.read_text(encoding="utf-8")

    # Dynamic phrases should produce many short blocks (around 20-30 blocks for 6 clips)
    assert "1\n" in content
    assert "You wake up" in content
    assert "every single morning" in content
    assert "carrying an invisible backpack" in content
    assert "full of unkept promises" in content
    assert "Every time you say" in content
    assert "I will start tomorrow" in content
    assert "Tell me" in content

    # Punctuation must be stripped from subtitle text lines
    text_lines = [l for l in content.splitlines() if not l.isdigit() and "-->" not in l and l.strip()]
    for line in text_lines:
        assert "'" not in line
        assert "," not in line
        assert "." not in line
        assert "?" not in line


def test_generate_srt_static_mode(tmp_path):
    """Verify legacy/static 1-block-per-clip SRT generation when phrase_level=False."""
    sample_json = Path("sample_data/phase_b_prompts.json")
    assert sample_json.exists(), "sample_data/phase_b_prompts.json must exist"

    test_srt = tmp_path / "test_static_captions.srt"
    result = generate_srt_from_json(sample_json, output_srt_path=test_srt, phrase_level=False)

    assert result.exists()
    content = result.read_text(encoding="utf-8")

    # Verify 10-second static block indices
    assert "1\n00:00:00,000 --> 00:00:10,000" in content
    assert "2\n00:00:10,000 --> 00:00:20,000" in content
    assert "6\n00:00:50,000 --> 00:01:00,000" in content

    # Verify text matching with cleaned punctuation
    assert "invisible backpack full of unkept promises" in content
    assert "Every time you say I will start tomorrow you drop another heavy stone into the bag" in content
    assert "Tell me in the comments" in content


def test_split_into_phrases():
    """Verify phrase chunking splits text into 2-4 word natural phrases without single-word orphans."""
    from src.caption_embedder import split_into_phrases

    text = "You wake up every single morning carrying an invisible backpack full of unkept promises."
    phrases = split_into_phrases(text, min_words=2, max_words=4)
    assert len(phrases) >= 3
    for p in phrases:
        words = p.split()
        assert 2 <= len(words) <= 4, f"Phrase '{p}' has invalid word count {len(words)}"


def test_get_audio_speech_intervals_missing_file():
    """Verify fallback to (0.0, duration) when media file does not exist."""
    from src.caption_embedder import get_audio_speech_intervals

    start, end = get_audio_speech_intervals("non_existent_clip.mp4", clip_duration=10.0)
    assert start == 0.0
    assert end == 10.0


def test_generate_srt_missing_json(tmp_path):
    """Verify FileNotFoundError when JSON file does not exist."""
    with pytest.raises(FileNotFoundError):
        generate_srt_from_json(tmp_path / "missing.json")


def test_generate_srt_empty_clips(tmp_path):
    """Verify ValueError when clips list is empty."""
    bad_json = tmp_path / "bad.json"
    bad_json.write_text(json.dumps({"clips": []}), encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        generate_srt_from_json(bad_json)
    assert "No clips found" in str(excinfo.value)


def test_embed_captions_missing_video(tmp_path):
    """Verify FileNotFoundError when input video is missing."""
    srt_file = tmp_path / "test.srt"
    srt_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        embed_captions(tmp_path / "missing.mp4", srt_file)


def test_embed_captions_missing_srt(tmp_path):
    """Verify FileNotFoundError when subtitle file is missing."""
    video_file = tmp_path / "test.mp4"
    video_file.write_text("fake video")

    with pytest.raises(FileNotFoundError):
        embed_captions(video_file, tmp_path / "missing.srt")
