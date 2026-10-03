import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Type, TypeVar
from pydantic import BaseModel

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.config import OUTPUT_DIR, normalize_style
from src.schemas import ReviewVerdict, PhaseBOutput
from src.agents import (
    create_writer_agent,
    create_reviewer_agent,
    create_compiler_agent,
)

T = TypeVar("T", bound=BaseModel)


def parse_structured_output(content: object, model_cls: Type[T]) -> T:
    """Safely parse Agent response content into a Pydantic model."""
    if isinstance(content, model_cls):
        return content

    if isinstance(content, dict):
        return model_cls.model_validate(content)

    if isinstance(content, str):
        cleaned = content.strip()
        # Remove markdown code blocks if wrapped
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            return model_cls.model_validate_json(cleaned)
        except Exception:
            # Try to extract the first JSON object using regex
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                return model_cls.model_validate_json(match.group(0))
            raise

    raise ValueError(f"Unable to parse response of type {type(content)} into {model_cls.__name__}: {content}")


def get_next_run_folder(output_dir: Optional[Path] = None, target_date: Optional[datetime] = None) -> Path:
    """Generate the next sequential run folder matching the pattern: NNN_DD_mon_YY.

    Examples:
        001_03_oct_26
        002_03_oct_26 (if generated same day manually)
        003_04_oct_26 (next day continues monotonic counter)

    Args:
        output_dir: Base directory where topic folders live (defaults to OUTPUT_DIR).
        target_date: Optional datetime override (defaults to datetime.now()).

    Returns:
        Path to the next folder (e.g. data/002_03_oct_26).
    """
    base_dir = output_dir or OUTPUT_DIR
    base_dir.mkdir(parents=True, exist_ok=True)

    current_time = target_date or datetime.now()
    date_str = current_time.strftime("%d_%b_%y").lower()

    max_seq = 0
    if base_dir.exists():
        for item in base_dir.iterdir():
            if item.is_dir():
                match = re.match(r"^(\d{3,})_", item.name)
                if match:
                    try:
                        seq = int(match.group(1))
                        # Ignore legacy 8-digit timestamps like 20261003
                        if seq < 100000:
                            if seq > max_seq:
                                max_seq = seq
                    except ValueError:
                        pass

    next_seq = max_seq + 1
    folder_name = f"{next_seq:03d}_{date_str}"
    return base_dir / folder_name


def run_director_pipeline(
    topic: str,
    aspect_ratio: str = "9:16",
    style: str = "1A",
    duration: int = 60,
    max_retries: int = 3,
    model_id: Optional[str] = None,
) -> Tuple[str, PhaseBOutput, Path]:
    """Execute the full Actor-Critic video director pipeline.

    Args:
        topic: Source topic or article for the video.
        aspect_ratio: Output ratio ("9:16" or "16:9").
        style: Visual style code ("1A", "1B", "2A", "2B").
        duration: Target video duration in seconds (must be multiple of 10).
        max_retries: Maximum revision iterations for Actor-Critic loop.
        model_id: Optional Gemini model override.

    Returns:
        Tuple of (approved_phase_a_markdown, phase_b_output, output_directory_path).
    """
    if duration % 10 != 0:
        duration = round(duration / 10) * 10
        print(f"[Pipeline] Duration rounded to nearest 10-second multiple: {duration}s")

    num_clips = max(1, duration // 10)
    normalized_style = normalize_style(style)

    print("\n" + "=" * 70)
    print(f"🎬 STICKMAN VIDEO DIRECTOR PIPELINE")
    print(f"Topic:       {topic}")
    print(f"Duration:    {duration}s ({num_clips} clips of 10s)")
    print(f"Ratio:       {aspect_ratio}")
    print(f"Style:       {normalized_style}")
    print(f"Max Retries: {max_retries}")
    print("=" * 70 + "\n")

    # Initialize agents
    writer_agent = create_writer_agent(model_id)
    reviewer_agent = create_reviewer_agent(model_id)
    compiler_agent = create_compiler_agent(model_id)

    # Actor-Critic Loop for Phase A
    phase_a_proposal: str = ""
    latest_verdict: Optional[ReviewVerdict] = None
    critique_history: list = []

    for attempt in range(1, max_retries + 1):
        print(f"--- [Iteration {attempt}/{max_retries}] Screenwriter generating Phase A draft ---")

        if attempt == 1:
            writer_prompt = f"""Produce Phase A (Director's Proposal) for the following video project:
- Topic / Source Material: "{topic}"
- Target Duration: {duration} seconds (exactly {num_clips} clips of 10s each)
- Aspect Ratio: {aspect_ratio}
- Visual Style: {normalized_style}

Follow the 5-Stage Heartbeat Engine:
1. Stage 1: Golden Hook (instant curiosity, counter-intuitive question or striking visual paradox in [0-3s] of Clip 1)
2. Stage 2: Disrupting Assumptions (shatter conventional wisdom in one sentence)
3. Stage 3: Unveiling Insider Secrets (pull back the curtain on hidden mechanics/friction)
4. Stage 4: Ultimate Truth Revelation (underlying truth with maximum cognitive payoff)
5. Stage 5: Elevation & High-Engagement Discussion (memorable punchline + open-ended comment question)

CRITICAL INSTRUCTIONS:
- Physical Metaphors: Use concrete physical character actions and props (e.g., carrying heavy boulders, running on accelerating treadmills, pressing glass controls, climbing ladders). DO NOT use abstract liquid morphing or melting shapes.
- Voiceover Pacing: Approximately 20-25 spoken English words per 10-second clip (~{num_clips * 22} words total across {num_clips} clips).
- Storyboard Table: Exactly {num_clips} rows. Specify timed visual beats ([0-3s], [3-7s], [7-10s]) in the motion column for every row.
- End each row with a transition element inherited by the next row.
"""
        else:
            assert latest_verdict is not None
            writer_prompt = f"""The Creative Director (Reviewer) evaluated your previous proposal with Score: {latest_verdict.score}/10 (Rejected).

CRITIQUE:
{latest_verdict.critique}

DEFICIENCIES TO FIX:
{json.dumps(latest_verdict.missing_elements, indent=2)}

Please thoroughly revise the Phase A proposal to address EVERY single critique point:
- Enforce strictly concrete, physical character actions (no abstract melting or idle stick figures).
- Ensure exactly {num_clips} storyboard rows with ~20-25 spoken English words per row.
- Ensure timed beats ([0-3s], [3-7s], [7-10s]) in every row.

PREVIOUS PROPOSAL DRAFT:
{phase_a_proposal}
"""

        writer_response = writer_agent.run(writer_prompt)
        phase_a_proposal = str(writer_response.content)

        print(f"--- [Iteration {attempt}/{max_retries}] DirectorCritic auditing Phase A ---")
        reviewer_prompt = f"""Audit the following Phase A Director's Proposal:

PROJECT PARAMETERS:
- Topic: "{topic}"
- Target Duration: {duration}s ({num_clips} clips)
- Ratio: {aspect_ratio}
- Style: {normalized_style}

PHASE A PROPOSAL:
{phase_a_proposal}

EVALUATION CHECKLIST:
1. Are all visual metaphors concrete and physical? (REJECT if abstract liquid morphing, floating blobs, or idle characters are present).
2. Is the 5-Stage narrative structure present and compelling?
3. Does each row have ~20-25 words of English VO?
4. Are timed beats ([0-3s], [3-7s], [7-10s]) present for each of the {num_clips} clips?
5. Is the hook sufficiently counter-intuitive and immediate?

Provide your structured ReviewVerdict with score (1-10), is_approved (True only if score >= 8), critique, and missing_elements.
"""
        reviewer_response = reviewer_agent.run(reviewer_prompt)
        latest_verdict = parse_structured_output(reviewer_response.content, ReviewVerdict)

        print(f"📊 Verdict: Score = {latest_verdict.score}/10 | Approved = {latest_verdict.is_approved}")
        print(f"📝 Critique: {latest_verdict.critique}")
        if latest_verdict.missing_elements:
            print(f"⚠️  Missing/Defective Elements: {latest_verdict.missing_elements}")

        critique_history.append({
            "iteration": attempt,
            "score": latest_verdict.score,
            "is_approved": latest_verdict.is_approved,
            "critique": latest_verdict.critique,
            "missing_elements": latest_verdict.missing_elements,
        })

        if latest_verdict.is_approved and latest_verdict.score >= 8:
            print(f"\n✅ Phase A APPROVED on iteration {attempt} with score {latest_verdict.score}/10!\n")
            break
        else:
            if attempt < max_retries:
                print(f"🔄 Re-invoking Writer with critique for iteration {attempt + 1}...\n")
            else:
                print(f"\n⚠️  Reached maximum retries ({max_retries}). Proceeding with current best draft.\n")

    # Phase B: Compilation
    print("--- ⚙️  OmniFlashCompiler generating Phase B production prompts ---")
    compiler_prompt = f"""Convert the following APPROVED Phase A Director's Proposal into a complete Phase B production package with structured PhaseBOutput.

PROJECT PARAMETERS:
- Topic: "{topic}"
- Aspect Ratio: {aspect_ratio}
- Style: {normalized_style}
- Total Duration: {duration} seconds ({num_clips} clips of 10s each)

APPROVED PHASE A PROPOSAL:
{phase_a_proposal}

INSTRUCTIONS:
Generate exactly {num_clips} ClipPrompt objects (indices 1 to {num_clips}).
Each ClipPrompt must include:
1. index: 1-based index (1 to {num_clips})
2. duration_sec: 10
3. prompt: Complete, self-contained Omni Flash generation prompt adhering to all contract rules:
   - Output spec: ~10 seconds, {aspect_ratio}, 720p, 24 FPS, synchronized audio
   - Background and environment matching {normalized_style}
   - Character visual lock (Style 1 monochrome or Style 2 Beanie Zeke)
   - Timed beats: [0-3s], [3-7s], [7-10s] with physical actions
   - Dialogue: Exact spoken English VO quoted in quotes with 'Audio voiceover only, strictly no speech bubbles, no dialogue boxes'
   - Narrator lock: Identical voice actor description across all prompts
   - BGM continuity lock: Seamless continuation from clip 1
   - Transitions: Matching opening/closing frame states
   - Negative constraints: No visible text, no captions, no speech bubbles, no abstract liquid morphing, no photorealism
4. spoken_words: The exact English VO for this clip
5. visual_metaphor: Concrete physical action or metaphor depicted
"""

    compiler_response = compiler_agent.run(compiler_prompt)
    phase_b_output = parse_structured_output(compiler_response.content, PhaseBOutput)

    # Save results to sequential output directory (e.g. 001_03_oct_26)
    run_dir = get_next_run_folder(OUTPUT_DIR)
    run_dir.mkdir(parents=True, exist_ok=True)

    proposal_file = run_dir / "phase_a_proposal.md"
    prompts_file = run_dir / "phase_b_prompts.json"
    package_md_file = run_dir / "phase_b_production_package.md"

    # Prepend critique history summary to Phase A proposal file
    proposal_content = f"""# Phase A: Director's Proposal
**Topic:** {topic}
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Aspect Ratio:** {aspect_ratio}
**Visual Style:** {normalized_style}
**Duration:** {duration}s ({num_clips} clips)
**Final Review Score:** {latest_verdict.score if latest_verdict else 'N/A'}/10 (Approved: {latest_verdict.is_approved if latest_verdict else False})

---

{phase_a_proposal}
"""
    proposal_file.write_text(proposal_content, encoding="utf-8")

    # Save JSON structured data
    prompts_file.write_text(
        phase_b_output.model_dump_json(indent=2),
        encoding="utf-8"
    )

    # Save Markdown production package for easy copying into AI video models (contains all clips in 1 MD file)
    package_md_content = format_phase_b_markdown(phase_b_output)
    package_md_file.write_text(package_md_content, encoding="utf-8")

    print(f"🎉 SUCCESS! Outputs saved to topic folder: {run_dir.resolve()}")
    print(f"   - Phase A Proposal:         {proposal_file.name}")
    print(f"   - Phase B Package Markdown: {package_md_file.name} (Contains all {len(phase_b_output.clips)} clips in 1 clean MD file)")
    print(f"   - Phase B Prompts JSON:     {prompts_file.name} (Key-value pairs for all clips)")
    print(f"   - Target Video Destination: clip_1.mp4 to clip_{len(phase_b_output.clips)}.mp4 (directly in this topic folder)")

    return phase_a_proposal, phase_b_output, run_dir


def format_phase_b_markdown(phase_b: PhaseBOutput) -> str:
    """Format PhaseBOutput into a clean, complete Markdown package containing all clips in one file."""
    lines = [
        f"# Phase B: Omni Flash Production Package — {phase_b.topic}",
        "",
        f"**Topic:** {phase_b.topic}",
        f"**Aspect Ratio:** `{phase_b.aspect_ratio}`",
        f"**Visual Style:** {phase_b.style}",
        f"**Total Duration:** {phase_b.total_duration_sec}s ({len(phase_b.clips)} clips of 10s each)",
        "",
        "---",
        "",
        "## Storyboard Overview",
        "",
        "| Clip | Duration | Visual Metaphor | Spoken Dialogue | Video Target |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]

    for clip in phase_b.clips:
        vo_escaped = clip.spoken_words.replace("|", "/")
        lines.append(f"| **Clip {clip.index}** | {clip.duration_sec}s | {clip.visual_metaphor} | *\"{vo_escaped}\"* | `clip_{clip.index}.mp4` |")

    lines.extend([
        "",
        "---",
        "",
        "## Omni Flash Production Prompts",
        "",
        "> **Note:** Each prompt below is a complete, self-contained text block. Copy the entire content inside the code fence and send it directly to the video generation model.",
        "",
    ])

    for clip in phase_b.clips:
        lines.append(f"### Clip {clip.index} (10s) — {clip.visual_metaphor}")
        lines.append(f"**Target Video Output:** `clip_{clip.index}.mp4`  ")
        lines.append(f"**VO Dialogue:** *\"{clip.spoken_words}\"*")
        lines.append("")
        lines.append("```text")
        lines.append(clip.prompt.strip())
        lines.append("```")
        lines.append("")

    if phase_b.stitching_guide:
        lines.append("---")
        lines.append("")
        lines.append("## Stitching Guide")
        lines.append(phase_b.stitching_guide)
        lines.append("")

    if phase_b.voice_and_music_note:
        lines.append("## Voice and Music Continuity Note")
        lines.append(phase_b.voice_and_music_note)
        lines.append("")

    if phase_b.post_production_overlays:
        lines.append("## Optional Post-Production Text Overlays")
        for overlay in phase_b.post_production_overlays:
            lines.append(f"- {overlay}")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## Video Generation Files",
        "When generating video files, save the rendered MP4 clips directly inside this topic directory:",
    ])
    for clip in phase_b.clips:
        lines.append(f"- `clip_{clip.index}.mp4`")
    lines.append("- `final_assembled_video.mp4` (full stitched master video)")

    lines.append("")
    return "\n".join(lines)
