"""Automated 2D Stickman Clip Synthesizer.

Generates high-resolution 1080x1920 24 FPS stickman video clips with neural voiceover
audio using Edge-TTS, Pillow vector rendering, and FFmpeg stream piping.
"""

import asyncio
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image, ImageDraw

import edge_tts

VOICE_ACTOR = "en-US-ChristopherNeural"


async def generate_clip_audio(spoken_words: str, output_wav: Path, target_duration: float = 10.0):
    """Generate clean neural voiceover audio padded to target_duration seconds."""
    temp_mp3 = output_wav.with_suffix(".mp3")
    comm = edge_tts.Communicate(spoken_words, VOICE_ACTOR, rate="-2%")
    await comm.save(str(temp_mp3))

    cmd = [
        "ffmpeg", "-y",
        "-i", str(temp_mp3),
        "-af", f"apad=whole_dur={target_duration},volume=1.2",
        "-t", str(target_duration),
        "-ar", "44100", "-ac", "2",
        str(output_wav)
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    temp_mp3.unlink(missing_ok=True)


def draw_stickman(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    action: str = "stand",
    t: float = 0.0,
    facing: int = 1,
):
    """Draw a minimalist 2D black stick figure in various animated poses."""
    head_r = 45
    head_y = y - 380
    body_color = (25, 25, 25)
    line_w = 9

    # Head
    draw.ellipse([x - head_r, head_y - head_r, x + head_r, head_y + head_r], outline=body_color, width=line_w)

    if action == "push":
        # Leaning forward
        torso_start = (x, head_y + head_r)
        torso_end = (x - 40 * facing, y - 180)
        draw.line([torso_start, torso_end], fill=body_color, width=line_w)

        shoulder = (x - 10 * facing, head_y + head_r + 35)
        hand1 = (x + 130 * facing, y - 270)
        hand2 = (x + 140 * facing, y - 200)
        draw.line([shoulder, hand1], fill=body_color, width=line_w)
        draw.line([shoulder, hand2], fill=body_color, width=line_w)

        hip = torso_end
        foot1 = (x - 110 * facing, y)
        foot2 = (x + 20 * facing, y)
        knee1 = (x - 70 * facing, y - 80)
        knee2 = (x - 10 * facing, y - 90)
        draw.line([hip, knee1], fill=body_color, width=line_w)
        draw.line([knee1, foot1], fill=body_color, width=line_w)
        draw.line([hip, knee2], fill=body_color, width=line_w)
        draw.line([knee2, foot2], fill=body_color, width=line_w)

    elif action == "slip":
        # Falling backwards on slick surface
        bob = math.sin(t * 10) * 10
        tilt = min(1.2, t * 0.4)
        torso_start = (x - int(tilt * 60), head_y + head_r)
        torso_end = (x - int(tilt * 140), y - 140 + int(bob))
        draw.line([torso_start, torso_end], fill=body_color, width=line_w)

        # Arms flailing
        shoulder = (torso_start[0], torso_start[1] + 30)
        hand1 = (shoulder[0] - 80, shoulder[1] - 80 + int(math.sin(t * 15) * 40))
        hand2 = (shoulder[0] + 90, shoulder[1] - 70 + int(math.cos(t * 15) * 40))
        draw.line([shoulder, hand1], fill=body_color, width=line_w)
        draw.line([shoulder, hand2], fill=body_color, width=line_w)

        # Legs sliding out forward
        hip = torso_end
        foot1 = (x + 80 + int(t * 20), y - 30)
        foot2 = (x + 140 + int(t * 25), y - 10)
        draw.line([hip, (x + 30, y - 90)], fill=body_color, width=line_w)
        draw.line([(x + 30, y - 90), foot1], fill=body_color, width=line_w)
        draw.line([hip, (x + 80, y - 70)], fill=body_color, width=line_w)
        draw.line([(x + 80, y - 70), foot2], fill=body_color, width=line_w)

    elif action == "crank":
        # Cranking mechanical winch
        torso_start = (x, head_y + head_r)
        torso_end = (x - 20, y - 180)
        draw.line([torso_start, torso_end], fill=body_color, width=line_w)

        # Cranking arm moving in a circle
        crank_angle = t * 6.0
        handle_x = x + 90 + int(math.cos(crank_angle) * 35)
        handle_y = y - 220 + int(math.sin(crank_angle) * 35)
        shoulder = (x + 5, head_y + head_r + 40)
        draw.line([shoulder, (handle_x, handle_y)], fill=body_color, width=line_w)

        hip = torso_end
        draw.line([hip, (x - 50, y - 90)], fill=body_color, width=line_w)
        draw.line([(x - 50, y - 90), (x - 70, y)], fill=body_color, width=line_w)
        draw.line([hip, (x + 30, y - 90)], fill=body_color, width=line_w)
        draw.line([(x + 30, y - 90), (x + 40, y)], fill=body_color, width=line_w)

    elif action == "celebrate":
        # Standing confidently, gesturing to viewer
        torso_start = (x, head_y + head_r)
        torso_end = (x, y - 180)
        draw.line([torso_start, torso_end], fill=body_color, width=line_w)

        shoulder = (x, head_y + head_r + 40)
        # One hand on hip, one gesturing to lower center (comments)
        hand1 = (x - 55, y - 210)
        hand2 = (x + 85, y - 140 + int(math.sin(t * 4) * 15))
        draw.line([shoulder, hand1], fill=body_color, width=line_w)
        draw.line([shoulder, hand2], fill=body_color, width=line_w)

        hip = torso_end
        draw.line([hip, (x - 45, y - 90)], fill=body_color, width=line_w)
        draw.line([(x - 45, y - 90), (x - 60, y)], fill=body_color, width=line_w)
        draw.line([hip, (x + 45, y - 90)], fill=body_color, width=line_w)
        draw.line([(x + 45, y - 90), (x + 60, y)], fill=body_color, width=line_w)

    else:
        # Default idle/watching
        torso_start = (x, head_y + head_r)
        torso_end = (x, y - 180)
        draw.line([torso_start, torso_end], fill=body_color, width=line_w)

        shoulder = (x, head_y + head_r + 40)
        # Crossed arms or hands down
        hand1 = (x - 45, y - 220)
        hand2 = (x + 45, y - 220)
        draw.line([shoulder, hand1], fill=body_color, width=line_w)
        draw.line([shoulder, hand2], fill=body_color, width=line_w)

        hip = torso_end
        draw.line([hip, (x - 40, y - 90)], fill=body_color, width=line_w)
        draw.line([(x - 40, y - 90), (x - 50, y)], fill=body_color, width=line_w)
        draw.line([hip, (x + 40, y - 90)], fill=body_color, width=line_w)
        draw.line([(x + 40, y - 90), (x + 50, y)], fill=body_color, width=line_w)


def render_clip_frames(
    clip_index: int,
    output_mp4: Path,
    audio_wav: Path,
    duration_sec: float = 10.0,
    fps: int = 24,
):
    """Render animated 1080x1920 frames for a specific clip and encode to MP4."""
    w, h = 1080, 1920
    total_frames = int(fps * duration_sec)

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{w}x{h}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "-",
        "-i", str(audio_wav),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(output_mp4)
    ]

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    ground_y = 1450

    for f in range(total_frames):
        t = f / fps
        im = Image.new("RGB", (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(im)

        # Baseline ground
        draw.line([(80, ground_y), (1000, ground_y)], fill=(35, 35, 35), width=8)

        # Scene choreography based on clip index
        if clip_index == 1:
            # CLIP 1: Straining against massive heavy flywheel
            wheel_center = (720, 1150)
            wheel_r = 280
            cx, cy = wheel_center

            if t < 3.0:
                angle = math.sin(t * 15) * 0.05
                push_jitter = int(math.sin(t * 18) * 10)
            elif t < 7.0:
                prog = (t - 3.0) / 4.0
                angle = (prog ** 2) * 22.0
                push_jitter = 0
            else:
                angle = 22.0 + (t - 7.0) * 14.0
                push_jitter = 0

            # Flywheel
            draw.ellipse([cx - wheel_r, cy - wheel_r, cx + wheel_r, cy + wheel_r], outline=(25, 25, 25), width=16)
            draw.ellipse([cx - 45, cy - 45, cx + 45, cy + 45], fill=(35, 35, 35), outline=(25, 25, 25), width=8)
            for sp in range(4):
                sp_ang = angle + sp * (math.pi / 2)
                draw.line([(cx, cy), (cx + int(math.cos(sp_ang) * (wheel_r - 10)), cy + int(math.sin(sp_ang) * (wheel_r - 10)))], fill=(25, 25, 25), width=10)

            # Crimson momentum lines
            if t > 3.0:
                for sp in range(8):
                    sp_ang = angle * 1.5 + sp * (math.pi / 4)
                    draw.line([
                        (cx + int(math.cos(sp_ang) * (wheel_r + 20)), cy + int(math.sin(sp_ang) * (wheel_r + 20))),
                        (cx + int(math.cos(sp_ang) * (wheel_r + 45)), cy + int(math.sin(sp_ang) * (wheel_r + 45)))
                    ], fill=(220, 38, 38), width=6)

            draw_stickman(draw, 340 + push_jitter, ground_y, action="push", t=t, facing=1)

            # Sweat drops [0-3s]
            if t < 3.0:
                for sw_i in range(3):
                    drop_x = 380 + sw_i * 20
                    drop_y = ground_y - 410 + int((t * 55 + sw_i * 15) % 40)
                    draw.ellipse([drop_x, drop_y, drop_x + 8, drop_y + 12], fill=(220, 38, 38))

        elif clip_index == 2:
            # CLIP 2: Frictionless track, slipping and falling
            # Cyan slick horizon track
            draw.line([(80, ground_y), (1000, ground_y)], fill=(6, 182, 212), width=14)
            # Slick sparkles
            for sp in range(5):
                sx = 200 + sp * 160 + int(math.sin(t * 5 + sp) * 20)
                draw.line([(sx - 15, ground_y - 10), (sx + 15, ground_y - 10)], fill=(6, 182, 212), width=4)

            # Red obstacle wall blocking path
            draw.rectangle([800, ground_y - 400, 850, ground_y], fill=(220, 38, 38), outline=(185, 28, 28), width=4)

            # Stickman slipping
            slip_x = 420 + int(min(2.5, t) * 60)
            draw_stickman(draw, slip_x, ground_y, action="slip", t=t, facing=1)

        elif clip_index == 3:
            # CLIP 3: Mechanical winch lifting heavy stone
            # Winch tower
            tower_x = 650
            draw.line([(tower_x, ground_y), (tower_x, ground_y - 500)], fill=(35, 35, 35), width=12)
            draw.line([(tower_x - 100, ground_y), (tower_x, ground_y - 500)], fill=(35, 35, 35), width=8)

            # Pulley wheel at top
            draw.ellipse([tower_x - 40, ground_y - 540, tower_x + 40, ground_y - 460], outline=(35, 35, 35), width=10)

            # Heavy stone lifted by cable
            cable_y = ground_y - 150 - int(min(1.0, t / 7.0) * 250)
            draw.line([(tower_x + 35, ground_y - 500), (tower_x + 180, ground_y - 500)], fill=(35, 35, 35), width=6)
            draw.line([(tower_x + 180, ground_y - 500), (tower_x + 180, cable_y)], fill=(35, 35, 35), width=6)
            # Heavy Stone block
            draw.rectangle([tower_x + 120, cable_y, tower_x + 240, cable_y + 120], fill=(200, 200, 200), outline=(35, 35, 35), width=8)
            draw.text((tower_x + 145, cable_y + 40), "HEAVY", fill=(35, 35, 35))

            # Cranking Stickman
            draw_stickman(draw, tower_x - 80, ground_y, action="crank", t=t, facing=1)

        elif clip_index == 4:
            # CLIP 4: Adding heavy blocks to flywheel, storing kinetic mass
            cx, cy = (540, 1100)
            wheel_r = 260
            angle = t * 8.0

            # Flywheel rim
            draw.ellipse([cx - wheel_r, cy - wheel_r, cx + wheel_r, cy + wheel_r], outline=(35, 35, 35), width=16)
            draw.ellipse([cx - 40, cy - 40, cx + 40, cy + 40], fill=(35, 35, 35), width=6)

            # 4 heavy blocks attached along the rim
            for b_i in range(4):
                b_ang = angle + b_i * (math.pi / 2)
                bx = cx + int(math.cos(b_ang) * wheel_r)
                by = cy + int(math.sin(b_ang) * wheel_r)
                draw.rectangle([bx - 30, by - 30, bx + 30, by + 30], fill=(40, 40, 40), outline=(234, 179, 8), width=6)

            # Golden energy pulse ripples radiating outward
            pulse_r = int((t * 120) % 220) + 120
            draw.ellipse([cx - pulse_r, cy - pulse_r, cx + pulse_r, cy + pulse_r], outline=(234, 179, 8), width=5)

            # Stickman standing beside wheel watching momentum build
            draw_stickman(draw, 180, ground_y, action="idle", t=t, facing=1)

        elif clip_index == 5:
            # CLIP 5: Flywheel self-sustaining momentum, smashing obstacles
            cx, cy = (450, 1150)
            wheel_r = 280
            angle = t * 24.0  # High-speed rotation

            # Flywheel
            draw.ellipse([cx - wheel_r, cy - wheel_r, cx + wheel_r, cy + wheel_r], outline=(35, 35, 35), width=18)
            draw.ellipse([cx - 45, cy - 45, cx + 45, cy + 45], fill=(35, 35, 35), width=8)
            for sp in range(6):
                sp_ang = angle + sp * (math.pi / 3)
                draw.line([(cx, cy), (cx + int(math.cos(sp_ang) * wheel_r), cy + int(math.sin(sp_ang) * wheel_r))], fill=(35, 35, 35), width=8)

            # Barrier block getting smashed away
            if t < 4.0:
                barr_x = cx + wheel_r + 40 - int(t * 10)
                barr_y = ground_y - 100
                draw.rectangle([barr_x, barr_y - 80, barr_x + 80, barr_y], fill=(220, 38, 38), outline=(35, 35, 35), width=6)
            else:
                # Smashed flying up and away
                fly_t = t - 4.0
                barr_x = cx + wheel_r + 40 + int(fly_t * 220)
                barr_y = ground_y - 100 - int(fly_t * 260)
                draw.rectangle([barr_x, barr_y - 80, barr_x + 80, barr_y], fill=(220, 38, 38), outline=(35, 35, 35), width=6)
                # Impact blast lines
                for imp in range(6):
                    imp_ang = imp * (math.pi / 3)
                    draw.line([
                        (cx + wheel_r + 20, ground_y - 120),
                        (cx + wheel_r + 20 + int(math.cos(imp_ang) * 50), ground_y - 120 + int(math.sin(imp_ang) * 50))
                    ], fill=(234, 179, 8), width=6)

            # Stickman standing triumphantly
            draw_stickman(draw, 140, ground_y, action="idle", t=t, facing=1)

        elif clip_index == 6:
            # CLIP 6: Conclusion & Call to Action (Gesturing to comments)
            cx, cy = (760, 1180)
            wheel_r = 200
            angle = t * 6.0
            # Humming balanced flywheel
            draw.ellipse([cx - wheel_r, cy - wheel_r, cx + wheel_r, cy + wheel_r], outline=(35, 35, 35), width=12)
            draw.ellipse([cx - 30, cy - 30, cx + 30, cy + 30], fill=(234, 179, 8), width=6)

            # Center stickman celebrating & pointing downward
            draw_stickman(draw, 420, ground_y, action="celebrate", t=t, facing=1)

            # Downward arrows / pulsing indicator pointing to comment section
            arrow_y = ground_y + 120 + int(math.sin(t * 8) * 15)
            draw.polygon([(420, arrow_y + 50), (390, arrow_y), (450, arrow_y)], fill=(234, 179, 8))

        proc.stdin.write(im.tobytes())

    proc.stdin.close()
    proc.wait()


async def generate_all_clips(folder_path: Path | str) -> List[Path]:
    """Generate all 6 clips for a folder containing phase_b_prompts.json."""
    target_dir = Path(folder_path).resolve()
    json_path = target_dir / "phase_b_prompts.json"
    if not json_path.exists():
        raise FileNotFoundError(f"phase_b_prompts.json not found in {target_dir}")

    data = json.loads(json_path.read_text(encoding="utf-8"))
    clips = sorted(data.get("clips", []), key=lambda c: c.get("index", 1))

    print("\n" + "=" * 70)
    print(f"ðŸŽ¬ AUTOMATED CLIP GENERATOR: {target_dir.name}")
    print(f"Synthesizing {len(clips)} clips with Neural Audio + 2D Stickman Engine...")
    print("=" * 70)

    generated_clips = []

    for clip in clips:
        idx = clip.get("index", 1)
        dur = float(clip.get("duration_sec", 10.0))
        spoken_words = clip.get("spoken_words", "")
        clip_mp4 = target_dir / f"clip_{idx}.mp4"
        temp_audio = target_dir / f"temp_audio_{idx}.wav"

        print(f"\n[Clip {idx}/{len(clips)}] Generating neural audio...")
        await generate_clip_audio(spoken_words, temp_audio, target_duration=dur)

        print(f"[Clip {idx}/{len(clips)}] Rendering 1080x1920 24FPS animation -> {clip_mp4.name}...")
        render_clip_frames(idx, clip_mp4, temp_audio, duration_sec=dur, fps=24)
        temp_audio.unlink(missing_ok=True)

        if not clip_mp4.exists() or clip_mp4.stat().st_size == 0:
            raise RuntimeError(f"Failed to generate clip: {clip_mp4}")

        print(f"âœ… Generated {clip_mp4.name} ({clip_mp4.stat().st_size / 1024:.1f} KB)")
        generated_clips.append(clip_mp4)

    print("\nðŸŽ‰ ALL 6 CLIPS SYNTHESIZED SUCCESSFULLY!")
    return generated_clips


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.clip_generator <folder_path>")
        sys.exit(1)
    asyncio.run(generate_all_clips(sys.argv[1]))


if __name__ == "__main__":
    main()
