"""Video Stitcher Module for Stickman Video Director.

Scans sequential video clips (clip_1.mp4 ... clip_N.mp4) in a folder and concatenates
them using FFmpeg stream copy with zero re-encoding loss and ultra-fast execution (< 2s).
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def find_sequential_clips(folder_path: Path | str) -> List[Path]:
    """Scan and sort sequential clip_*.mp4 files in the target folder.

    Args:
        folder_path: Directory containing clip_1.mp4, clip_2.mp4, etc.

    Returns:
        Sorted list of Path objects in numerical order.

    Raises:
        FileNotFoundError: If the folder doesn't exist or contains no clips.
        ValueError: If there is a missing index in the sequence (e.g., 1, 2, 4).
    """
    target = Path(folder_path).resolve()
    if not target.exists() or not target.is_dir():
        raise FileNotFoundError(f"Target folder does not exist: {target}")

    clip_pattern = re.compile(r"^clip_(\d+)\.mp4$", re.IGNORECASE)
    clips_with_index = []

    for item in target.iterdir():
        if item.is_file():
            match = clip_pattern.match(item.name)
            if match:
                clips_with_index.append((int(match.group(1)), item))

    if not clips_with_index:
        raise FileNotFoundError(f"No clip_*.mp4 files found in {target}")

    # Sort numerically
    clips_with_index.sort(key=lambda x: x[0])

    # Validate sequential continuity
    indices = [idx for idx, _ in clips_with_index]
    expected = list(range(1, len(indices) + 1))
    if indices != expected:
        missing = set(expected) - set(indices)
        raise ValueError(
            f"Sequential clips gap detected in {target}. Missing clip indices: {sorted(missing)}. Found: {indices}"
        )

    return [path for _, path in clips_with_index]


def stitch_clips(
    folder_path: Path | str,
    output_path: Optional[Path | str] = None,
    output_filename: str = "stitched_raw.mp4",
    keep_manifest: bool = False,
) -> Path:
    """Concatenate sequential clips in folder_path using FFmpeg concat stream copy.

    Args:
        folder_path: Directory containing the clips (e.g. sample_data/ or data/001_03_oct_26).
        output_path: Optional exact output path. If None, saves output_filename inside folder_path.
        output_filename: Default output filename if output_path is not specified.
        keep_manifest: If True, preserves temporary clips.txt manifest file.

    Returns:
        Path to the stitched output video file.
    """
    start_time = time.time()
    target_dir = Path(folder_path).resolve()
    clips = find_sequential_clips(target_dir)

    if output_path is not None:
        final_output = Path(output_path).resolve()
        final_output.parent.mkdir(parents=True, exist_ok=True)
    else:
        final_output = target_dir / output_filename

    import json

    def _probe_clip(clip_p: Path) -> dict:
        p_cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-show_entries", "stream=codec_type,codec_name,time_base,sample_rate",
            "-of", "json",
            str(clip_p),
        ]
        r = subprocess.run(p_cmd, capture_output=True, text=True, check=False)
        if r.returncode == 0:
            try:
                return json.loads(r.stdout)
            except Exception:
                pass
        return {}

    # Check if timebases and sample rates match
    can_stream_copy = True
    v_timebase = None
    a_sample_rate = None
    expected_duration = 0.0

    for c in clips:
        meta = _probe_clip(c)
        try:
            expected_duration += float(meta.get("format", {}).get("duration", 0))
        except (ValueError, TypeError):
            pass

        for s in meta.get("streams", []):
            if s.get("codec_type") == "video":
                tb = s.get("time_base")
                if v_timebase is None:
                    v_timebase = tb
                elif v_timebase != tb:
                    can_stream_copy = False
            elif s.get("codec_type") == "audio":
                sr = s.get("sample_rate")
                if a_sample_rate is None:
                    a_sample_rate = sr
                elif a_sample_rate != sr:
                    can_stream_copy = False

    manifest_path = target_dir / "clips.txt"

    try:
        if can_stream_copy:
            # Write relative filenames to clips.txt for maximum cross-platform path safety
            manifest_content = "".join(f"file '{clip.name}'\n" for clip in clips)
            manifest_path.write_text(manifest_content, encoding="utf-8")

            cmd = [
                "ffmpeg",
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", "clips.txt",
                "-c", "copy",
                str(final_output),
            ]

            print(f"[VideoStitcher] Concatenating {len(clips)} clips via stream copy: {target_dir}")
            result = subprocess.run(
                cmd,
                cwd=str(target_dir),
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode == 0 and final_output.exists():
                out_meta = _probe_clip(final_output)
                try:
                    out_dur = float(out_meta.get("format", {}).get("duration", 0))
                    if abs(out_dur - expected_duration) > 2.0:
                        print(f"⚠️ [VideoStitcher] Stream copy duration mismatch ({out_dur:.1f}s vs expected {expected_duration:.1f}s). Switching to filter concat.")
                        can_stream_copy = False
                except Exception:
                    pass
            else:
                can_stream_copy = False

        if not can_stream_copy:
            print(f"[VideoStitcher] Normalizing {len(clips)} clips with mixed timebases/codecs via filter graph: {target_dir}")
            inputs = []
            for c in clips:
                inputs.extend(["-i", str(c)])
            filter_str = "".join(f"[{i}:v][{i}:a]" for i in range(len(clips))) + f"concat=n={len(clips)}:v=1:a=1[v][a]"
            cmd = [
                "ffmpeg", "-y",
                *inputs,
                "-filter_complex", filter_str,
                "-map", "[v]", "-map", "[a]",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
                "-c:a", "aac", "-b:a", "192k",
                str(final_output),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                raise RuntimeError(
                    f"FFmpeg concatenation failed with return code {result.returncode}:\n{result.stderr}"
                )

        if not final_output.exists() or final_output.stat().st_size == 0:
            raise RuntimeError(f"Stitched video output was not created or is empty: {final_output}")

        elapsed = time.time() - start_time
        print(f"✅ [VideoStitcher] Stitched {len(clips)} clips in {elapsed:.2f}s -> {final_output.resolve()}")
        return final_output

    finally:
        if not keep_manifest and manifest_path.exists():
            try:
                manifest_path.unlink()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(
        description="Stitch sequential stickman video clips (clip_1.mp4 ... clip_N.mp4) using FFmpeg stream copy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "folder",
        type=str,
        help="Path to folder containing clip_1.mp4, clip_2.mp4, etc. (e.g. sample_data/)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Optional explicit path for stitched output video (default: <folder>/stitched_raw.mp4)",
    )
    parser.add_argument(
        "--keep-manifest",
        action="store_true",
        help="Keep temporary clips.txt manifest file after concatenation",
    )

    args = parser.parse_args()

    try:
        out = stitch_clips(args.folder, output_path=args.output, keep_manifest=args.keep_manifest)
        print(f"\nðŸŽ‰ Successfully created stitched video: {out}")
    except Exception as e:
        print(f"\nâ�Œ Stitching failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
