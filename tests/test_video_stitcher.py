"""Unit and integration tests for Video Stitcher module."""

from pathlib import Path
import pytest

from src.video_stitcher import find_sequential_clips, stitch_clips


def test_find_sequential_clips_sample_data():
    """Verify scanning and sorting of sequential clips in sample_data."""
    sample_dir = Path("sample_data")
    assert sample_dir.exists(), "sample_data/ directory must exist"

    clips = find_sequential_clips(sample_dir)
    assert len(clips) >= 6
    assert [c.name for c in clips[:6]] == [
        "clip_1.mp4",
        "clip_2.mp4",
        "clip_3.mp4",
        "clip_4.mp4",
        "clip_5.mp4",
        "clip_6.mp4",
    ]


def test_find_sequential_clips_gap_detection(tmp_path):
    """Verify that gaps in clip numbering raise ValueError."""
    (tmp_path / "clip_1.mp4").write_text("test")
    (tmp_path / "clip_2.mp4").write_text("test")
    (tmp_path / "clip_4.mp4").write_text("test")  # clip_3 missing!

    with pytest.raises(ValueError) as excinfo:
        find_sequential_clips(tmp_path)

    assert "Missing clip indices: [3]" in str(excinfo.value)


def test_find_sequential_clips_nonexistent_folder(tmp_path):
    """Verify FileNotFoundError on missing directory."""
    non_existent = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        find_sequential_clips(non_existent)


def test_find_sequential_clips_no_clips(tmp_path):
    """Verify FileNotFoundError when directory contains no clips."""
    (tmp_path / "random_file.txt").write_text("data")
    with pytest.raises(FileNotFoundError):
        find_sequential_clips(tmp_path)


def test_stitch_clips_creates_output(tmp_path):
    """Verify that stitch_clips creates stitched_raw.mp4 from sample clips."""
    sample_dir = Path("sample_data")
    if not (sample_dir / "clip_1.mp4").exists():
        pytest.skip("sample_data/clip_1.mp4 not found")

    output_file = tmp_path / "test_stitched.mp4"
    result = stitch_clips(sample_dir, output_path=output_file)

    assert result.exists()
    assert result.stat().st_size > 0
    assert result == output_file
