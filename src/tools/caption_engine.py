"""SRT Generation & Subtitle Burning Engine Agno Tool.

Generates standard SRT subtitles and burns styled captions into vertical videos.
"""

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Union
from agno.tools import tool
from agno.tools.function import Function

if "__call__" not in Function.__dict__:
    Function.__call__ = lambda self, *args, **kwargs: self.entrypoint(*args, **kwargs)

from src.caption_embedder import (
    clean_caption_text,
    format_srt_timestamp,
    split_into_phrases,
    get_audio_speech_intervals,
)


@tool
def generate_srt_file(
    clips_meta: Union[List[Dict], str, Path],
    srt_path: Union[str, Path],
    phrase_level: bool = True,
    clean_punctuation: bool = True,
) -> str:
    """Generate an SRT subtitle file from clips metadata or prompts JSON.

    Args:
        clips_meta: List of clip dicts with 'spoken_words' and 'duration_sec', or path to JSON file.
        srt_path: Output destination path for the .srt file.
        phrase_level: If True, uses dynamic speech-synchronized phrasing. If False, uses 10s blocks.
        clean_punctuation: If True, strips quotes, commas, periods, etc. from subtitle text.

    Returns:
        Absolute string path to the created .srt file.
    """
    out_srt = Path(srt_path).resolve()
    out_srt.parent.mkdir(parents=True, exist_ok=True)

    # If clips_meta is a file path string or Path
    if isinstance(clips_meta, (str, Path)):
        p = Path(clips_meta).resolve()
        if p.is_dir():
            p = p / "phase_b_prompts.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        clips = sorted(data.get("clips", []), key=lambda c: c.get("index", 1))
        video_dir = p.parent
    else:
        clips = sorted(clips_meta, key=lambda c: c.get("index", 1))
        video_dir = out_srt.parent

    current_sec = 0.0
    srt_blocks = []
    block_idx = 1

    for clip in clips:
        idx = clip.get("index", block_idx)
        duration = float(clip.get("duration_sec", 10.0))
        spoken_words = clip.get("spoken_words", "").strip()

        if not spoken_words:
            current_sec += duration
            continue

        if not phrase_level:
            # Standard 10-second blocks (e.g. 00:00:00,000 --> 00:00:10,000)
            text = clean_caption_text(spoken_words) if clean_punctuation else spoken_words
            start_str = format_srt_timestamp(current_sec)
            end_str = format_srt_timestamp(current_sec + duration)
            srt_blocks.append(f"{block_idx}\n{start_str} --> {end_str}\n{text}\n")
            block_idx += 1
            current_sec += duration
            continue

        # Dynamic speech-paced phrasing
        clip_file = video_dir / f"clip_{idx}.mp4"
        speech_start, speech_end = get_audio_speech_intervals(clip_file, clip_duration=duration)
        speech_span = max(0.5, speech_end - speech_start)

        phrases = split_into_phrases(spoken_words, min_words=2, max_words=4)
        if not phrases:
            current_sec += duration
            continue

        word_counts = [len(p.split()) for p in phrases]
        total_words = max(1, sum(word_counts))

        curr_t = speech_start
        for i, phrase in enumerate(phrases):
            w_count = word_counts[i]
            p_dur = (w_count / total_words) * speech_span

            p_start_local = curr_t
            p_end_local = min(speech_end, curr_t + p_dur)
            curr_t = p_end_local

            start_time = current_sec + p_start_local
            end_time = current_sec + p_end_local

            if i < len(phrases) - 1:
                end_time = max(start_time + 0.3, end_time - 0.03)

            start_str = format_srt_timestamp(start_time)
            end_str = format_srt_timestamp(end_time)
            text = clean_caption_text(phrase) if clean_punctuation else phrase

            srt_blocks.append(f"{block_idx}\n{start_str} --> {end_str}\n{text}\n")
            block_idx += 1

        current_sec += duration

    srt_content = "\n".join(srt_blocks) + "\n"
    out_srt.write_text(srt_content, encoding="utf-8")
    print(f"📄 [CaptionEngine] Generated {len(srt_blocks)} subtitle cues -> {out_srt}")
    return str(out_srt.resolve())


@tool
def embed_captions_to_video(
    video_input_path: Union[str, Path],
    srt_path: Union[str, Path],
    video_output_path: Union[str, Path],
    font_name: str = "Arial",
    font_size: int = 18,
    margin_v: int = 60,
) -> str:
    """Burn styled subtitles into video using FFmpeg.

    Args:
        video_input_path: Path to the stitched input video.
        srt_path: Path to the .srt subtitles file.
        video_output_path: Path to write the final captioned video.
        font_name: Subtitle font name.
        font_size: Font size.
        margin_v: Vertical margin from bottom (middle-lower safe area).

    Returns:
        Absolute string path to the final captioned video.
    """
    input_v = Path(video_input_path).resolve()
    srt_f = Path(srt_path).resolve()
    out_v = Path(video_output_path).resolve()
    out_v.parent.mkdir(parents=True, exist_ok=True)

    if not input_v.exists():
        raise FileNotFoundError(f"Input video not found: {input_v}")
    if not srt_f.exists():
        raise FileNotFoundError(f"SRT file not found: {srt_f}")

    start_time = time.time()
    srt_filename = srt_f.name

    style_options = (
        f"Fontname={font_name},"
        f"FontSize={font_size},"
        f"PrimaryColour=&H00FFFFFF,"
        f"OutlineColour=&H00000000,"
        f"BorderStyle=3,"
        f"Outline=2,"
        f"Alignment=2,"
        f"MarginV={margin_v}"
    )

    subtitles_filter = f"subtitles='{srt_filename}':force_style='{style_options}'"

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_v),
        "-vf", subtitles_filter,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "192k",
        str(out_v),
    ]

    print(f"🎬 [CaptionEngine] Burning captions into video: {input_v.name} -> {out_v.name}")
    res = subprocess.run(
        cmd,
        cwd=str(srt_f.parent),
        capture_output=True,
        text=True,
        check=False,
    )

    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg caption embedding failed (code {res.returncode}):\n{res.stderr}")

    if not out_v.exists() or out_v.stat().st_size == 0:
        raise RuntimeError(f"Captioned video output was not created or is empty: {out_v}")

    elapsed = time.time() - start_time
    print(f"✅ [CaptionEngine] Captions burned in {elapsed:.2f}s -> {out_v.resolve()}")
    return str(out_v.resolve())
