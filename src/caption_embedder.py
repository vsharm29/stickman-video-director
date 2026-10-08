"""SRT Subtitle Generation & Caption Embedding Engine for Stickman Video Director.

Reads phase_b_prompts.json to build timed .srt subtitles and burns high-contrast
captions into stitched video clips using native FFmpeg (libass) filters.
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.video_stitcher import stitch_clips


def format_srt_timestamp(seconds: float) -> str:
    """Convert a timestamp in seconds to the standard SRT timestamp format: HH:MM:SS,mmm."""
    if seconds < 0:
        seconds = 0.0

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))

    if millis >= 1000:
        secs += 1
        millis -= 1000

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


HEAD_WORDS = {
    "every", "carrying", "full", "into", "by", "from", "with", "without",
    "because", "when", "while", "where", "after", "before", "tell", "empty",
    "take", "notice", "say", "drop", "what", "it", "that", "which", "and", "or"
}


def clean_caption_text(text: str) -> str:
    """Clean punctuation marks from subtitle text for clean, modern video captions.

    Removes commas, periods, single/double quotation marks, apostrophes, question marks,
    exclamation points, and colons so captions appear clean and uncluttered.
    """
    if not text:
        return ""
    # Strip quotes, periods, commas, questions, exclamations, etc.
    cleaned = re.sub(r"[\.,'\"`â€˜â€™â€œâ€�\?!;:â€”\-]", " ", text)
    # Collapse multiple whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def split_words_optimal(words: List[str], min_words: int = 2, max_words: int = 4) -> List[str]:
    """Partition a list of words into balanced 2-4 word phrases using dynamic programming.

    Prioritizes starting new chunks at natural clause/phrase boundaries (HEAD_WORDS).
    """
    n = len(words)
    if n <= max_words:
        return [" ".join(words)]

    def cost(i: int, j: int) -> float:
        l = j - i
        if l < min_words or l > max_words:
            return 1e6
        base = (l - 3.5) ** 2
        if j < n and words[j].lower() in HEAD_WORDS:
            base -= 2.0
        return base

    dp = [1e6] * (n + 1)
    prev = [-1] * (n + 1)
    dp[0] = 0

    for i in range(n):
        if dp[i] >= 1e6:
            continue
        for j in range(i + min_words, min(i + max_words + 1, n + 1)):
            c = dp[i] + cost(i, j)
            if c < dp[j]:
                dp[j] = c
                prev[j] = i

    # Fallback to balanced partitioning if no strict path found
    if dp[n] >= 1e5:
        k = math.ceil(n / max_words)
        while k > 1 and (n / k) < min_words:
            k -= 1
        base = n // k
        rem = n % k
        chunks = []
        idx = 0
        for i in range(k):
            size = base + (1 if i < rem else 0)
            chunks.append(" ".join(words[idx : idx + size]))
            idx += size
        return chunks

    splits = []
    curr = n
    while curr > 0:
        p = prev[curr]
        splits.append(" ".join(words[p:curr]))
        curr = p
    splits.reverse()
    return splits


def split_into_phrases(text: str, min_words: int = 2, max_words: int = 4) -> List[str]:
    """Split spoken dialogue into short, natural speech phrases of 2-4 words.

    First breaks on punctuation clauses, then partitions each clause smoothly.
    """
    # Clean quotes first so quotes attached to commas don't become isolated tokens
    clean_text = re.sub(r"['\"`â€˜â€™â€œâ€�]", "", text)
    clauses = re.split(r'[,.?!;:]+', clean_text)
    clauses = [c.strip() for c in clauses if c.strip()]
    phrases = []
    for clause in clauses:
        words = clause.split()
        if words:
            phrases.extend(split_words_optimal(words, min_words=min_words, max_words=max_words))
    return phrases


def get_audio_speech_intervals(clip_path: Path | str, clip_duration: float = 10.0) -> Tuple[float, float]:
    """Detect active speech boundaries in a video clip using FFmpeg silencedetect.

    Returns:
        (speech_start_sec, speech_end_sec)
    """
    path = Path(clip_path)
    if not path.exists():
        return 0.0, clip_duration

    try:
        cmd = [
            "ffmpeg",
            "-i", str(path),
            "-af", "silencedetect=noise=-25dB:d=0.2",
            "-f", "null", "-"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)

        silence_starts = []
        silence_ends = []
        for line in res.stderr.splitlines():
            if "silence_start:" in line:
                m = re.search(r"silence_start: ([\d\.]+)", line)
                if m:
                    silence_starts.append(float(m.group(1)))
            elif "silence_end:" in line:
                m = re.search(r"silence_end: ([\d\.]+)", line)
                if m:
                    silence_ends.append(float(m.group(1)))

        speech_start = 0.0
        speech_end = clip_duration

        # If silence is detected at beginning, speech starts when that silence ends
        if silence_starts and silence_starts[0] <= 0.05 and silence_ends:
            speech_start = min(silence_ends[0], clip_duration - 1.0)

        # If silence is detected at the end, speech ends when trailing silence begins
        if silence_ends and silence_ends[-1] >= clip_duration - 0.15 and silence_starts:
            speech_end = max(speech_start + 1.0, silence_starts[-1])

        speech_start = round(max(0.0, min(speech_start, 2.0)), 2)
        speech_end = round(min(clip_duration, max(speech_end, speech_start + 2.0)), 2)
        return speech_start, speech_end
    except Exception:
        return 0.0, clip_duration


def get_video_dimensions(video_path: Path | str) -> Tuple[int, int]:
    """Get (width, height) of a video file using ffprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=s=x:p=0",
            str(video_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        parts = res.stdout.strip().split("x")
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
    except Exception:
        pass
    return 1080, 1920


def generate_srt_from_json(
    json_path: Path | str,
    output_srt_path: Optional[Path | str] = None,
    clean_punctuation: bool = True,
    phrase_level: bool = True,
    min_words_per_phrase: int = 2,
    max_words_per_phrase: int = 4,
    clips_dir: Optional[Path | str] = None,
) -> Path:
    """Generate an SRT subtitle file from phase_b_prompts.json data.

    Supports dynamic, speech-paced phrase-level chunking so subtitles appear
    briefly and rhythmically as spoken in the audio, avoiding clumsy full-screen walls of text.

    Args:
        json_path: Path to phase_b_prompts.json or directory containing it.
        output_srt_path: Optional destination path for .srt file.
                         Defaults to <folder>/captions.srt.
        clean_punctuation: If True, removes commas, periods, quotes, etc. from subtitle text.
        phrase_level: If True, breaks speech into short 2-4 word phrases timed to the audio.
                      If False, creates 1 static block per clip.
        min_words_per_phrase: Minimum words per subtitle chunk (default 2).
        max_words_per_phrase: Maximum words per subtitle chunk (default 4).
        clips_dir: Optional directory containing clip_*.mp4 files for audio silence detection.
                   Defaults to the directory containing the JSON file.

    Returns:
        Path to the generated .srt file.
    """
    target = Path(json_path).resolve()
    if target.is_dir():
        json_file = target / "phase_b_prompts.json"
        video_dir = target
    else:
        json_file = target
        video_dir = target.parent

    if clips_dir is not None:
        video_dir = Path(clips_dir).resolve()

    if not json_file.exists():
        raise FileNotFoundError(f"Prompts JSON not found: {json_file}")

    data = json.loads(json_file.read_text(encoding="utf-8"))
    clips = data.get("clips", [])
    if not clips:
        raise ValueError(f"No clips found in prompts JSON: {json_file}")

    # Sort by index
    sorted_clips = sorted(clips, key=lambda c: c.get("index", 0))

    if output_srt_path:
        srt_file = Path(output_srt_path).resolve()
        srt_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        srt_file = json_file.parent / "captions.srt"

    current_sec = 0.0
    srt_blocks = []
    block_idx = 1

    for clip in sorted_clips:
        clip_idx = clip.get("index", block_idx)
        duration = float(clip.get("duration_sec", 10))
        spoken_words = clip.get("spoken_words", "").strip()

        if not spoken_words:
            current_sec += duration
            continue

        if not phrase_level:
            # Static 10s block
            text = clean_caption_text(spoken_words) if clean_punctuation else spoken_words
            start_time = current_sec
            end_time = current_sec + duration
            start_str = format_srt_timestamp(start_time)
            end_str = format_srt_timestamp(end_time)
            srt_blocks.append(f"{block_idx}\n{start_str} --> {end_str}\n{text}\n")
            block_idx += 1
            current_sec += duration
            continue

        # Dynamic phrase-level chunking
        clip_file = video_dir / f"clip_{clip_idx}.mp4"
        speech_start, speech_end = get_audio_speech_intervals(clip_file, clip_duration=duration)
        speech_span = max(0.5, speech_end - speech_start)

        phrases = split_into_phrases(
            spoken_words,
            min_words=min_words_per_phrase,
            max_words=max_words_per_phrase,
        )
        if not phrases:
            current_sec += duration
            continue

        word_counts = [len(p.split()) for p in phrases]
        total_words = max(1, sum(word_counts))

        curr_t = speech_start
        for i, phrase in enumerate(phrases):
            w_count = word_counts[i]
            p_dur = (w_count / total_words) * speech_span

            # Small 30ms gap before next cue to give subtitle engine crisp render
            p_start_local = curr_t
            p_end_local = min(speech_end, curr_t + p_dur)
            curr_t = p_end_local

            # Global timeline offsets
            start_time = current_sec + p_start_local
            end_time = current_sec + p_end_local

            # Leave 30ms inter-phrase gap if not the last phrase in clip
            if i < len(phrases) - 1:
                end_time = max(start_time + 0.3, end_time - 0.03)

            start_str = format_srt_timestamp(start_time)
            end_str = format_srt_timestamp(end_time)

            text = clean_caption_text(phrase) if clean_punctuation else phrase
            srt_blocks.append(f"{block_idx}\n{start_str} --> {end_str}\n{text}\n")
            block_idx += 1

        current_sec += duration

    srt_content = "\n".join(srt_blocks) + "\n"
    srt_file.write_text(srt_content, encoding="utf-8")
    mode_str = "dynamic phrases" if phrase_level else "static blocks"
    print(f"ðŸ“„ [CaptionEmbedder] Generated {len(srt_blocks)} subtitle blocks ({mode_str}) -> {srt_file.resolve()}")
    return srt_file


def embed_captions(
    video_path: Path | str,
    srt_path: Path | str,
    output_path: Optional[Path | str] = None,
    font_name: str = "Arial",
    font_size: Optional[int] = None,
    primary_color: str = "&H00FFFFFF",  # Opaque White (&HAABBGGRR)
    outline_color: str = "&H00000000",  # Black outline
    outline_width: Optional[float] = None,
    margin_v: Optional[int] = None,     # Lower safe zone margin for vertical reels/Shorts
) -> Path:
    """Burn high-contrast subtitles into video using FFmpeg subtitles filter.

    Args:
        video_path: Path to input stitched video (e.g. stitched_raw.mp4).
        srt_path: Path to captions.srt file.
        output_path: Path to final captioned video. Defaults to <dir>/final_published_short.mp4.
        font_name: Subtitle font name.
        font_size: Subtitle font size (auto-scaled by resolution if None).
        primary_color: Text color in ASS format (&HAABBGGRR).
        outline_color: Outline color in ASS format.
        outline_width: Outline stroke width (auto-scaled if None).
        margin_v: Vertical margin from bottom in pixels (auto-scaled if None).

    Returns:
        Path to the output captioned video.
    """
    start_time = time.time()
    input_video = Path(video_path).resolve()
    subtitle_file = Path(srt_path).resolve()

    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")
    if not subtitle_file.exists():
        raise FileNotFoundError(f"Subtitle file not found: {subtitle_file}")

    # Detect resolution for optimal font and margin scaling
    width, height = get_video_dimensions(input_video)
    # In FFmpeg subtitles filter on SRT, default virtual resolution is 288p height.
    # An effective font size of 10-11 and margin_v of 25 yields ideal lower-third placement.
    eff_font_size = font_size if font_size is not None else 10
    eff_outline = outline_width if outline_width is not None else 1.2
    eff_margin_v = margin_v if margin_v is not None else 25

    if output_path is not None:
        final_output = Path(output_path).resolve()
        final_output.parent.mkdir(parents=True, exist_ok=True)
    else:
        final_output = input_video.parent / "final_published_short.mp4"

    # To avoid Windows drive letter colon escaping bugs in FFmpeg's libass filter,
    # run FFmpeg with cwd set to subtitle_file's directory and pass relative filename
    srt_filename = subtitle_file.name

    style_options = (
        f"FontName={font_name},"
        f"FontSize={eff_font_size},"
        f"Bold=1,"
        f"PrimaryColour={primary_color},"
        f"OutlineColour={outline_color},"
        f"BorderStyle=1,"
        f"Outline={eff_outline},"
        f"Shadow=1,"
        f"Alignment=2,"
        f"MarginV={eff_margin_v}"
    )

    subtitles_filter = f"subtitles='{srt_filename}':force_style='{style_options}'"

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_video),
        "-vf", subtitles_filter,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "192k",
        str(final_output),
    ]

    print(f"[CaptionEmbedder] Burning subtitles into video: {input_video.name}")
    print(f"[CaptionEmbedder] Style: Font={font_name}, Size={eff_font_size}, Alignment=Center-Bottom, MarginV={eff_margin_v}")

    result = subprocess.run(
        cmd,
        cwd=str(subtitle_file.parent),
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg caption embedding failed with exit code {result.returncode}:\n{result.stderr}"
        )

    if not final_output.exists() or final_output.stat().st_size == 0:
        raise RuntimeError(f"Captioned video output was not created or is empty: {final_output}")

    elapsed = time.time() - start_time
    print(f"ðŸŽ¬ [CaptionEmbedder] Finished burning captions in {elapsed:.2f}s -> {final_output.resolve()}")
    return final_output


def process_folder(
    folder_path: Path | str,
    output_dir: Optional[Path | str] = None,
    output_filename: str = "final_published_short.mp4",
) -> Tuple[Path, Path, Path]:
    """Execute complete post-processing pipeline for a folder:
    1. Stitches clips (clip_1.mp4 ... clip_N.mp4) -> stitched_raw.mp4
    2. Generates captions.srt from phase_b_prompts.json
    3. Burns captions into final_published_short.mp4

    Args:
        folder_path: Path to target directory (e.g. sample_data/ or data/001_03_oct_26).
        output_dir: Optional destination directory for final video.
        output_filename: Name of the final captioned video.

    Returns:
        Tuple of (stitched_raw_path, srt_path, final_published_path).
    """
    target = Path(folder_path).resolve()
    if not target.exists() or not target.is_dir():
        raise FileNotFoundError(f"Target folder does not exist: {target}")

    print("\n" + "=" * 70)
    print(f"ðŸš€ VIDEO POST-PROCESSING PIPELINE: {target.name}")
    print("=" * 70)

    # 1. Stitch videos if stitched_raw.mp4 doesn't exist or need re-run
    stitched_video = target / "stitched_raw.mp4"
    if not stitched_video.exists() or stitched_video.stat().st_size == 0:
        stitched_video = stitch_clips(target, output_filename="stitched_raw.mp4")
    else:
        print(f"â„¹ï¸� [Pipeline] Reusing existing stitched video: {stitched_video.name}")

    # 2. Generate SRT
    json_path = target / "phase_b_prompts.json"
    srt_file = generate_srt_from_json(json_path)

    # 3. Determine final output destination
    if output_dir is not None:
        dest_dir = Path(output_dir).resolve()
        dest_dir.mkdir(parents=True, exist_ok=True)
        final_video_path = dest_dir / output_filename
    else:
        final_video_path = target / output_filename

    # 4. Embed Captions
    captioned_video = embed_captions(
        video_path=stitched_video,
        srt_path=srt_file,
        output_path=final_video_path,
    )

    print("\n" + "#" * 70)
    print("ðŸŽ‰ POST-PROCESSING PIPELINE COMPLETED SUCCESSFULLY")
    print(f"1. Stitched Video:   {stitched_video.resolve()}")
    print(f"2. Subtitles (.srt): {srt_file.resolve()}")
    print(f"3. Final Published:  {captioned_video.resolve()}")
    print("#" * 70 + "\n")

    return stitched_video, srt_file, captioned_video


def main():
    parser = argparse.ArgumentParser(
        description="Generate SRT subtitles and burn captions into stickman videos using FFmpeg.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "folder",
        type=str,
        help="Path to folder containing phase_b_prompts.json and clips (e.g. sample_data/)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Optional directory to write final_published_short.mp4",
    )
    parser.add_argument(
        "--srt-only",
        action="store_true",
        help="Only generate the captions.srt file without burning captions into video",
    )

    args = parser.parse_args()

    try:
        if args.srt_only:
            srt_path = generate_srt_from_json(args.folder)
            print(f"âœ… SRT generation complete: {srt_path}")
        else:
            process_folder(args.folder, output_dir=args.output_dir)
    except Exception as e:
        print(f"\nâ�Œ Caption embedding failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
