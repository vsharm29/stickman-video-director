"""Agno multi-agent definitions for Stickman Video Director Pipeline."""

import os
from typing import Optional
from agno.agent import Agent
from agno.models.google import Gemini

from src.config import DEFAULT_MODEL, GEMINI_API_KEY
from src.schemas import ReviewVerdict, PhaseBOutput
from src.skill_loader import (
    get_writer_system_prompt,
    get_reviewer_system_prompt,
    get_compiler_system_prompt,
)


def get_model(model_id: Optional[str] = None) -> Gemini:
    """Instantiate a Gemini model with configured API key."""
    chosen_id = model_id or DEFAULT_MODEL
    api_key = GEMINI_API_KEY or os.getenv("GOOGLE_API_KEY")
    return Gemini(id=chosen_id, api_key=api_key)


def get_fallback_models(chosen_id: Optional[str] = None) -> list:
    """Provide automatic fallback models if primary model is unavailable or rate-limited."""
    primary = chosen_id or DEFAULT_MODEL
    api_key = GEMINI_API_KEY or os.getenv("GOOGLE_API_KEY")
    candidates = ["gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-3.8-flash"]
    return [Gemini(id=m, api_key=api_key) for m in candidates if m != primary]


def create_writer_agent(model_id: Optional[str] = None) -> Agent:
    """Create the StickmanScreenwriter Agent responsible for Phase A Director's Proposal."""
    primary = model_id or DEFAULT_MODEL
    return Agent(
        name="StickmanScreenwriter",
        model=get_model(primary),
        fallback_models=get_fallback_models(primary),
        instructions=get_writer_system_prompt(),
        markdown=True,
    )


def create_reviewer_agent(model_id: Optional[str] = None) -> Agent:
    """Create the DirectorCritic Agent responsible for strict critique and scoring."""
    primary = model_id or DEFAULT_MODEL
    return Agent(
        name="DirectorCritic",
        model=get_model(primary),
        fallback_models=get_fallback_models(primary),
        instructions=get_reviewer_system_prompt(),
        output_schema=ReviewVerdict,
    )


def create_compiler_agent(model_id: Optional[str] = None) -> Agent:
    """Create the OmniFlashCompiler Agent responsible for Phase B generation prompts."""
    primary = model_id or DEFAULT_MODEL
    return Agent(
        name="OmniFlashCompiler",
        model=get_model(primary),
        fallback_models=get_fallback_models(primary),
        instructions=get_compiler_system_prompt(),
        output_schema=PhaseBOutput,
    )


def create_production_manager_agent(model_id: Optional[str] = None) -> Agent:
    """Create the ProductionManagerAgent responsible for orchestrating video generation, stitching, and captioning."""
    from src.tools import (
        generate_flow_clip,
        stitch_clips,
        generate_srt_file,
        embed_captions_to_video,
    )

    primary = model_id or DEFAULT_MODEL
    instructions = (
        "You are the ProductionManagerAgent for Stickman Video Director.\n"
        "Your mission is to take an approved Phase B production package and execute video production:\n"
        "1. Clip Generation:\n"
        "   - For each clip prompt in Phase B, call `generate_flow_clip` to produce `clip_1.mp4` through `clip_N.mp4`.\n"
        "2. Video Stitching:\n"
        "   - Once all clips are available in the folder, call `stitch_clips` using FFmpeg stream copy to create `stitched_raw.mp4`.\n"
        "3. Subtitle Generation:\n"
        "   - Call `generate_srt_file` to construct speech-synchronized subtitles (`captions.srt`) with clean punctuation.\n"
        "4. Caption Embedding:\n"
        "   - Call `embed_captions_to_video` to burn high-contrast vertical captions onto `stitched_raw.mp4` to output `final_published_short.mp4`.\n"
        "5. Final Review:\n"
        "   - Confirm the existence of all output files and provide a clear summary of paths and durations."
    )
    return Agent(
        name="ProductionManagerAgent",
        model=get_model(primary),
        fallback_models=get_fallback_models(primary),
        instructions=instructions,
        tools=[
            generate_flow_clip,
            stitch_clips,
            generate_srt_file,
            embed_captions_to_video,
        ],
        markdown=True,
    )
