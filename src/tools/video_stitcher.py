"""FFmpeg Stream-Copy Stitcher Agno Tool.

Concatenates sequential video clips using FFmpeg stream copy with zero transcoding loss.
"""

import os
import subprocess
import time
from pathlib import Path
from typing import List, Union
from agno.tools import tool
from agno.tools.function import Function

if "__call__" not in Function.__dict__:
    Function.__call__ = lambda self, *args, **kwargs: self.entrypoint(*args, **kwargs)


@tool
def stitch_clips(
    clip_paths: Union[List[Union[str, Path]], str, Path],
    output_path: Union[str, Path],
) -> str:
    """Concatenate video clips into a single video file using FFmpeg lossless stream copy.

    Args:
        clip_paths: List of file path strings/Paths to MP4 clips, or a folder containing clip_*.mp4.
        output_path: Destination path for the stitched master video.

    Returns:
        Absolute string path to the stitched video file.
    """
    start_time = time.time()
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Handle directory path vs list of clip paths
    if isinstance(clip_paths, (str, Path)):
        p = Path(clip_paths).resolve()
        if p.is_dir():
            files = sorted(
                list(p.glob("clip_*.mp4")),
                key=lambda f: int(f.stem.split("_")[-1]) if f.stem.split("_")[-1].isdigit() else 9999
            )
            clips = [f for f in files if f.name != out_file.name]
        else:
            clips = [p]
    else:
        clips = [Path(cp).resolve() for cp in clip_paths]

    if not clips:
        raise ValueError(f"No clips provided or found to stitch into: {output_path}")

    # Verify all input clips exist
    for c in clips:
        if not c.exists():
            raise FileNotFoundError(f"Clip file does not exist: {c}")

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

    concat_list_file = out_file.parent / f"concat_list_{int(time.time()*1000)}.txt"
    try:
        if can_stream_copy:
            # Format concat list entries (using forward slashes for cross-platform ffmpeg compatibility)
            lines = [f"file '{str(c.resolve()).replace(chr(92), '/')}'" for c in clips]
            concat_list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

            cmd = [
                "ffmpeg",
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_list_file),
                "-c", "copy",
                str(out_file),
            ]

            print(f"🎬 [VideoStitcher] Concatenating {len(clips)} clips via stream copy -> {out_file.name}")
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if res.returncode == 0 and out_file.exists():
                out_meta = _probe_clip(out_file)
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
            # Mixed timebases (e.g. 1/12288 vs 1/90000) or codec mismatch: use filter_complex concat
            print(f"🎬 [VideoStitcher] Normalizing {len(clips)} clips with mixed timebases/codecs via filter graph -> {out_file.name}")
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
                str(out_file),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if res.returncode != 0:
                raise RuntimeError(f"FFmpeg stitch failed (code {res.returncode}):\n{res.stderr}")

        if not out_file.exists() or out_file.stat().st_size == 0:
            raise RuntimeError(f"Stitched video output was not created or is empty: {out_file}")

        elapsed = time.time() - start_time
        print(f"✅ [VideoStitcher] Stitched {len(clips)} clips in {elapsed:.2f}s -> {out_file.resolve()}")
        return str(out_file.resolve())

    finally:
        concat_list_file.unlink(missing_ok=True)
