"""Configuration module for Stickman Video Director Pipeline."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent
SKILLS_DIR = BASE_DIR / "skills" / "directing-stickman-videos"
SKILL_MD_PATH = SKILLS_DIR / "SKILL.md"
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR
TOPICS_HISTORY_PATH = DATA_DIR / "topics_history.json"
SUB_CATEGORIES_PATH = DATA_DIR / "sub_categories.json"

# Load environment variables
load_dotenv(BASE_DIR / ".env")

# API Keys & Models
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if GEMINI_API_KEY:
    # Ensure both environment variables are populated for Google GenAI / Agno
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

# Supported generation options
SUPPORTED_RATIOS = ["9:16", "16:9"]

STYLE_MAP = {
    "1A": "Style 1 (Classic Minimalist) - Light theme (pure white background, black stick figure)",
    "1B": "Style 1 (Classic Minimalist) - Dark theme (pitch black background, white stick figure)",
    "2A": "Style 2A (Modern Studio Tech) - Beanie Zeke, high-key white studio, light-gray perspective grid, glowing cyan/blue glass UI",
    "2B": "Style 2B (Cinematic Story) - Beanie Zeke, full-color cinematic environment, volumetric lighting and depth of field",
}

def normalize_style(style_input: str) -> str:
    """Normalize user style input to canonical key and description."""
    cleaned = style_input.strip().upper()
    for key, desc in STYLE_MAP.items():
        if key in cleaned:
            return desc
    # Default to 1A
    return STYLE_MAP.get(cleaned, STYLE_MAP["1A"])
