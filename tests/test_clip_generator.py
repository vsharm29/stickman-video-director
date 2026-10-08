"""Unit tests for automated clip generator module."""

import subprocess
from pathlib import Path
from PIL import Image, ImageDraw
import pytest

from src.clip_generator import draw_stickman, render_clip_frames


def test_draw_stickman_poses():
    """Verify that draw_stickman executes cleanly across all animated action poses."""
    im = Image.new("RGB", (1080, 1920), (255, 255, 255))
    draw = ImageDraw.Draw(im)

    poses = ["push", "slip", "crank", "celebrate", "idle", "stand"]
    for i, pose in enumerate(poses):
        draw_stickman(draw, x=200 + i * 100, y=1400, action=pose, t=1.5, facing=1)

    # Image should have non-white pixels drawn
    colors = im.getcolors(maxcolors=1000)
    assert len(colors) > 1, "Image must contain drawn stickman pixels"


def test_render_clip_frames_short(tmp_path):
    """Verify that render_clip_frames generates a valid, playable MP4 video file."""
    output_mp4 = tmp_path / "test_render.mp4"
    audio_wav = tmp_path / "test_silence.wav"

    # Create 1-second test audio with ffmpeg
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", "anullsrc=r=44100:cl=stereo",
        "-t", "1.0",
        str(audio_wav)
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    # Render a 1-second clip at 12 fps
    render_clip_frames(
        clip_index=1,
        output_mp4=output_mp4,
        audio_wav=audio_wav,
        duration_sec=1.0,
        fps=12,
    )

    assert output_mp4.exists()
    assert output_mp4.stat().st_size > 0

    # Probe duration
    dur_str = subprocess.check_output([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(output_mp4)
    ], text=True).strip()

    dur = float(dur_str)
    assert 0.8 <= dur <= 1.2
