"""Unit and integration tests for Agno Video Production Tools & ProductionManagerAgent."""

import json
from pathlib import Path
import pytest

from src.tools.flow_generator import generate_flow_clip
from src.tools.video_stitcher import stitch_clips
from src.tools.caption_engine import generate_srt_file, embed_captions_to_video
from src.agents import create_production_manager_agent


def test_production_manager_agent_creation():
    """Verify ProductionManagerAgent initializes with all required tools."""
    agent = create_production_manager_agent()
    assert agent.name == "ProductionManagerAgent"

    tool_names = [t.name for t in agent.tools]
    assert "generate_flow_clip" in tool_names
    assert "stitch_clips" in tool_names
    assert "generate_srt_file" in tool_names
    assert "embed_captions_to_video" in tool_names


def test_flow_generator_unauthenticated(tmp_path, monkeypatch):
    """Verify generate_flow_clip raises informative PermissionError when gflow is not logged in."""
    from unittest.mock import MagicMock
    import subprocess

    mock_proc = MagicMock()
    mock_proc.returncode = 1
    mock_proc.stdout.readline.side_effect = ["Error: No profiles found. Please run gflow auth login first.\n", ""]
    mock_proc.wait = MagicMock()

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: mock_proc)

    with pytest.raises(PermissionError) as excinfo:
        generate_flow_clip(
            prompt="Stickman walks",
            clip_index=1,
            output_dir=str(tmp_path),
        )
    assert "gflow auth login" in str(excinfo.value)


def test_stitch_clips_sample_data(tmp_path):
    """Verify stitch_clips concatenates sample clips via FFmpeg stream copy."""
    sample_dir = Path("sample_data")
    assert sample_dir.exists(), "sample_data directory must exist"

    out_stitched = tmp_path / "test_stitched.mp4"
    result = stitch_clips(sample_dir, str(out_stitched))

    assert Path(result).exists()
    assert Path(result).stat().st_size > 1000000  # Stitched file should be non-trivial


def test_stitch_clips_empty_folder(tmp_path):
    """Verify stitch_clips raises ValueError when no clips exist."""
    with pytest.raises(ValueError) as excinfo:
        stitch_clips(tmp_path, str(tmp_path / "out.mp4"))
    assert "No clips provided or found" in str(excinfo.value)


def test_stitch_clips_missing_file():
    """Verify stitch_clips raises FileNotFoundError for missing explicit clip paths."""
    with pytest.raises(FileNotFoundError):
        stitch_clips(["non_existent_clip.mp4"], "out.mp4")


def test_generate_srt_file_phrase_level(tmp_path):
    """Verify generate_srt_file creates dynamic phrase-level subtitle cues."""
    sample_dir = Path("sample_data")
    out_srt = tmp_path / "dynamic.srt"

    result = generate_srt_file(
        clips_meta=sample_dir,
        srt_path=out_srt,
        phrase_level=True,
        clean_punctuation=True,
    )

    assert Path(result).exists()
    content = Path(result).read_text(encoding="utf-8")

    # Should contain key phrases
    assert "You wake up" in content
    assert "invisible backpack" in content

    # Subtitle text lines should not contain commas, periods, quotes, or questions
    text_lines = [l for l in content.splitlines() if not l.isdigit() and "-->" not in l and l.strip()]
    for line in text_lines:
        assert "'" not in line
        assert "," not in line
        assert "." not in line
        assert "?" not in line


def test_generate_srt_file_static_mode(tmp_path):
    """Verify generate_srt_file creates 10s static block subtitles when phrase_level=False."""
    sample_dir = Path("sample_data")
    out_srt = tmp_path / "static.srt"

    result = generate_srt_file(
        clips_meta=sample_dir,
        srt_path=out_srt,
        phrase_level=False,
        clean_punctuation=True,
    )

    assert Path(result).exists()
    content = Path(result).read_text(encoding="utf-8")

    # Verify 10-second block intervals
    assert "1\n00:00:00,000 --> 00:00:10,000" in content
    assert "2\n00:00:10,000 --> 00:00:20,000" in content
    assert "6\n00:00:50,000 --> 00:01:00,000" in content


def test_embed_captions_missing_video(tmp_path):
    """Verify embed_captions_to_video raises FileNotFoundError when input video does not exist."""
    dummy_srt = tmp_path / "test.srt"
    dummy_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n")

    with pytest.raises(FileNotFoundError):
        embed_captions_to_video(
            video_input_path=tmp_path / "missing.mp4",
            srt_path=dummy_srt,
            video_output_path=tmp_path / "out.mp4",
        )


def test_embed_captions_missing_srt(tmp_path):
    """Verify embed_captions_to_video raises FileNotFoundError when subtitle file does not exist."""
    dummy_video = tmp_path / "test.mp4"
    dummy_video.write_text("dummy")

    with pytest.raises(FileNotFoundError):
        embed_captions_to_video(
            video_input_path=dummy_video,
            srt_path=tmp_path / "missing.srt",
            video_output_path=tmp_path / "out.mp4",
        )
