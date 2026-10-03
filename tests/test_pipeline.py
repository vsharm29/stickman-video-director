"""Unit and integration tests for Stickman Video Director Pipeline."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from src.config import SKILL_MD_PATH, normalize_style, STYLE_MAP
from src.skill_loader import (
    load_skill_md,
    load_storyboard_template,
    load_omni_flash_contract,
    get_writer_system_prompt,
    get_reviewer_system_prompt,
    get_compiler_system_prompt,
)
from src.schemas import ReviewVerdict, ClipPrompt, PhaseBOutput
from src.pipeline import (
    parse_structured_output,
    run_director_pipeline,
    get_next_run_folder,
)


def test_skill_loader_loads_markdown_files():
    """Verify that SKILL.md and reference docs are loaded from pristine git directory."""
    assert SKILL_MD_PATH.exists(), "SKILL.md must exist in skills/directing-stickman-videos"
    skill_text = load_skill_md()
    assert "directing-stickman-videos" in skill_text
    assert "5-Stage High-Completion Heartbeat Engine" in skill_text

    storyboard_text = load_storyboard_template()
    assert "Golden Hook" in storyboard_text

    contract_text = load_omni_flash_contract()
    assert "Omni Flash" in contract_text


def test_system_prompts_contain_core_rules():
    """Verify that system prompts are properly composed with upstream rules."""
    writer_prompt = get_writer_system_prompt()
    assert "StickmanScreenwriter" in writer_prompt
    assert "Golden Hook" in writer_prompt
    assert "20-25" in writer_prompt

    reviewer_prompt = get_reviewer_system_prompt()
    assert "DirectorCritic" in reviewer_prompt
    assert "liquid morphing" in reviewer_prompt.lower()

    compiler_prompt = get_compiler_system_prompt()
    assert "OmniFlashCompiler" in compiler_prompt
    assert "Audio voiceover only" in compiler_prompt


def test_review_verdict_schema():
    """Verify ReviewVerdict validation and score boundaries."""
    valid_verdict = ReviewVerdict(
        is_approved=True,
        score=9,
        critique="Excellent tangible metaphor.",
        missing_elements=[],
    )
    assert valid_verdict.score == 9
    assert valid_verdict.is_approved is True

    # Test out of range score
    with pytest.raises(ValidationError):
        ReviewVerdict(
            is_approved=True,
            score=11,  # Must be <= 10
            critique="Invalid",
            missing_elements=[],
        )


def test_clip_prompt_schema():
    """Verify ClipPrompt schema."""
    clip = ClipPrompt(
        index=1,
        duration_sec=10,
        prompt="A minimalist 2D animated stick figure running...",
        spoken_words="Have you ever felt like you're climbing someone else's mountain?",
        visual_metaphor="Stick figure climbs an endless ladder while others hold identical ladders.",
    )
    assert clip.index == 1
    assert clip.duration_sec == 10
    assert "mountain" in clip.spoken_words


def test_phase_b_output_schema():
    """Verify PhaseBOutput schema and serialization."""
    clips = [
        ClipPrompt(
            index=i,
            duration_sec=10,
            prompt=f"Prompt for clip {i}",
            spoken_words=f"Dialogue {i}",
            visual_metaphor=f"Metaphor {i}",
        )
        for i in range(1, 7)
    ]
    phase_b = PhaseBOutput(
        topic="Chasing Goals",
        aspect_ratio="9:16",
        style="Style 1A",
        total_duration_sec=60,
        clips=clips,
    )
    assert len(phase_b.clips) == 6
    json_str = phase_b.model_dump_json()
    data = json.loads(json_str)
    assert data["total_duration_sec"] == 60
    assert len(data["clips"]) == 6


def test_parse_structured_output_from_json_and_markdown():
    """Test parse_structured_output with various agent response formats."""
    # Direct instance
    verdict = ReviewVerdict(is_approved=True, score=8, critique="Good", missing_elements=[])
    assert parse_structured_output(verdict, ReviewVerdict) == verdict

    # Raw JSON string
    raw_json = json.dumps({"is_approved": False, "score": 5, "critique": "Needs work", "missing_elements": ["metaphor"]})
    parsed = parse_structured_output(raw_json, ReviewVerdict)
    assert parsed.score == 5
    assert parsed.is_approved is False

    # Markdown fenced code block
    fenced_json = f"```json\n{raw_json}\n```"
    parsed_fenced = parse_structured_output(fenced_json, ReviewVerdict)
    assert parsed_fenced.score == 5


def test_style_normalization():
    """Verify style input normalizer."""
    assert "Style 1 (Classic Minimalist)" in normalize_style("1A")
    assert "Dark theme" in normalize_style("1B")
    assert "Beanie Zeke" in normalize_style("2A")
    assert "Cinematic Story" in normalize_style("2B")
    assert "Style 1" in normalize_style("unknown")


def test_get_next_run_folder_sequential_naming(tmp_path):
    """Verify that folders follow monotonic sequential numbering: 001_03_oct_26, 002_03_oct_26, 003_04_oct_26."""
    from datetime import datetime

    # Day 1: October 3, 2026
    day1 = datetime(2026, 10, 3, 14, 30, 0)
    
    # Run 1 on Day 1
    folder1 = get_next_run_folder(output_dir=tmp_path, target_date=day1)
    assert folder1.name == "001_03_oct_26"
    folder1.mkdir()

    # Run 2 on same day (manually or auto)
    folder2 = get_next_run_folder(output_dir=tmp_path, target_date=day1)
    assert folder2.name == "002_03_oct_26"
    folder2.mkdir()

    # Day 2: October 4, 2026 -> Monotonic increment to 003
    day2 = datetime(2026, 10, 4, 9, 15, 0)
    folder3 = get_next_run_folder(output_dir=tmp_path, target_date=day2)
    assert folder3.name == "003_04_oct_26"
    folder3.mkdir()

    # Verify that legacy timestamps like 20261003_... do not skew sequence numbers
    legacy_folder = tmp_path / "20261003_195738_some_topic"
    legacy_folder.mkdir()

    folder4 = get_next_run_folder(output_dir=tmp_path, target_date=day2)
    assert folder4.name == "004_04_oct_26"

