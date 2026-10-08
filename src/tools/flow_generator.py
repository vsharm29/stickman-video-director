"""Google Flow video generation bridge via gflow-cli.

Provides Agno @tool to generate individual clips via Google Flow Veo (Omni Flash / Veo 3.1)
using the open-source gflow-cli browser automation engine without paid proxies or API tokens.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Union

from agno.tools import tool
from agno.tools.function import Function

if "__call__" not in Function.__dict__:
    Function.__call__ = lambda self, *args, **kwargs: self.entrypoint(*args, **kwargs)


def get_gflow_command() -> List[str]:
    """Resolve the command prefix to execute gflow cleanly."""
    gflow_bin = shutil.which("gflow")
    if gflow_bin:
        return [gflow_bin]
    scripts_dir = Path(sys.executable).parent
    gflow_exe = scripts_dir / ("gflow.exe" if os.name == "nt" else "gflow")
    if gflow_exe.exists():
        return [str(gflow_exe)]
    return [sys.executable, "-m", "gflow_cli.cli"]


def check_gflow_auth() -> bool:
    """Check if gflow has an authenticated profile configured."""
    cmd = get_gflow_command() + ["auth", "status"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return res.returncode == 0


@tool
def generate_flow_clip(
    prompt: str,
    clip_index: int,
    output_dir: Union[str, Path],
    model: str = "omni-flash",
    duration: int = 10,
    aspect_ratio: str = "9:16",
    resolution: str = "720p",
) -> str:
    """Generate a single stickman video clip via Google Flow (gflow-cli) and save to disk.

    Args:
        prompt: Full generation prompt text for the clip.
        clip_index: 1-based index for naming (e.g. clip_1.mp4).
        output_dir: Directory where the downloaded MP4 clip will be saved.
        model: Target model (default: 'omni-flash', supporting 10s clips, or 'veo-lite').
        duration: Clip duration in seconds (default: 10).
        aspect_ratio: Video format (default: '9:16').
        resolution: Video resolution (default: '720p').

    Returns:
        Absolute string path to the saved local MP4 file.
    """
    dest_dir = Path(output_dir).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    target_file = dest_dir / f"clip_{clip_index}.mp4"

    if target_file.exists() and target_file.stat().st_size > 10000:
        print(f"🎬 [FlowGenerator] Clip {clip_index} already exists on disk ({target_file.stat().st_size} bytes). Reusing.")
        return str(target_file.resolve())

    cmd = get_gflow_command() + [
        "video", "t2v",
        prompt,
        "--model", model,
        "--duration", str(duration),
        "--aspect", aspect_ratio,
        "--resolution", resolution,
        "-o", str(target_file),
        "--json",
    ]

    print(f"🎬 [FlowGenerator] Submitting clip {clip_index} to Google Flow (model: {model}, {duration}s, {aspect_ratio})...")
    start_time = time.time()
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output_lines = []
    if process.stdout:
        for raw_line in iter(process.stdout.readline, ""):
            line_str = raw_line.strip()
            if line_str and not line_str.startswith("{"):
                print(f"   [Google Flow] {line_str}")
            output_lines.append(raw_line)
    process.wait()

    output_text = "".join(output_lines).strip()
    if process.returncode != 0:
        if "No profiles found" in output_text or "auth login" in output_text:
            raise PermissionError(
                "No authenticated Google Flow session found. "
                "Please run `gflow auth login` once in your terminal to sign into Google Flow."
            )
        raise RuntimeError(f"gflow video generation failed (code {process.returncode}):\n{output_text}")

    if not target_file.exists() or target_file.stat().st_size == 0:
        raise RuntimeError(f"Target video was not generated or is empty: {target_file}")

    elapsed = time.time() - start_time
    print(f"✅ [FlowGenerator] Clip {clip_index} generated successfully in {elapsed:.1f}s -> {target_file}")
    return str(target_file.resolve())
