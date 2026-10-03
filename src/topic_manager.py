"""Topic Generation & Anti-Repetition Engine for Stickman Video Director.

Implements autonomous topic discovery, 70/30 niche ratio scheduling,
and sliding-window de-duplication without external vector databases.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from agno.agent import Agent

from src.config import (
    DATA_DIR,
    TOPICS_HISTORY_PATH,
    SUB_CATEGORIES_PATH,
    DEFAULT_MODEL,
)
from src.agents import get_model, get_fallback_models
from src.pipeline import parse_structured_output
from src.schemas import TopicIdea, TopicHistoryEntry


def load_history() -> List[dict]:
    """Read topics history from persistent storage. Return empty list if missing."""
    if not TOPICS_HISTORY_PATH.exists():
        return []
    try:
        content = TOPICS_HISTORY_PATH.read_text(encoding="utf-8").strip()
        if not content:
            return []
        data = json.loads(content)
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []


def get_target_niche(history: List[dict]) -> str:
    """Enforce 70/30 distribution between Mindset (70%) and Social Dynamics (30%).

    Inspects the last 10 entries:
    - If count of 'niche_2_social' is less than 3, returns 'niche_2_social'.
    - Otherwise, returns 'niche_1_mindset'.
    """
    recent = history[-10:] if len(history) >= 10 else history
    social_count = sum(1 for entry in recent if entry.get("niche") == "niche_2_social")
    if social_count < 3:
        return "niche_2_social"
    return "niche_1_mindset"


def get_recent_exclusions(history: List[dict], window_size: int = 25) -> Dict[str, List[str]]:
    """Extract exclusion lists from rolling window history to prevent repetition.

    Returns:
        dict containing:
        - 'excluded_titles': List of titles from the last window_size runs.
        - 'excluded_metaphors': List of physical metaphors from the last window_size runs.
        - 'recent_subcategories': List of sub-categories from the last 5 runs to prevent clustering.
    """
    window = history[-window_size:] if len(history) >= window_size else history
    excluded_titles = [entry.get("title", "") for entry in window if entry.get("title")]
    excluded_metaphors = [entry.get("physical_metaphor", "") for entry in window if entry.get("physical_metaphor")]

    recent_5 = history[-5:] if len(history) >= 5 else history
    recent_subcategories = [entry.get("sub_category", "") for entry in recent_5 if entry.get("sub_category")]

    return {
        "excluded_titles": excluded_titles,
        "excluded_metaphors": excluded_metaphors,
        "recent_subcategories": recent_subcategories,
    }


def record_topic_to_history(topic: TopicIdea, folder_path: Optional[str] = None) -> TopicHistoryEntry:
    """Append a newly generated topic into data/topics_history.json."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    history = load_history()

    entry = TopicHistoryEntry(
        date=datetime.now().isoformat(),
        niche=topic.niche,
        sub_category=topic.sub_category,
        title=topic.title,
        physical_metaphor=topic.physical_metaphor,
        folder_path=str(folder_path) if folder_path else None,
    )

    history.append(entry.model_dump())

    TOPICS_HISTORY_PATH.write_text(
        json.dumps(history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return entry


def update_topic_folder_in_history(title: str, folder_path: str) -> None:
    """Update an existing topic record with its generated folder path."""
    history = load_history()
    for entry in reversed(history):
        if entry.get("title") == title:
            entry["folder_path"] = str(folder_path)
            break
    TOPICS_HISTORY_PATH.write_text(
        json.dumps(history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def record_custom_topic_to_history(title: str, folder_path: Optional[str] = None) -> TopicHistoryEntry:
    """Record a user-provided custom topic into topics_history.json."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    history = load_history()
    entry = TopicHistoryEntry(
        date=datetime.now().isoformat(),
        niche="custom",
        sub_category="User Provided",
        title=title,
        physical_metaphor="custom",
        folder_path=str(folder_path) if folder_path else None,
    )
    history.append(entry.model_dump())
    TOPICS_HISTORY_PATH.write_text(
        json.dumps(history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return entry


def load_seed_taxonomy() -> dict:
    """Load the sub-categories taxonomy seed."""
    if not SUB_CATEGORIES_PATH.exists():
        raise FileNotFoundError(f"Taxonomy catalog not found at {SUB_CATEGORIES_PATH.resolve()}")
    return json.loads(SUB_CATEGORIES_PATH.read_text(encoding="utf-8"))


def create_topic_agent(model_id: Optional[str] = None) -> Agent:
    """Initialize the TopicStrategist Agno agent."""
    primary = model_id or DEFAULT_MODEL
    instructions = """You are the Lead Content Strategist and Viral Architect for viral stickman educational videos (TopicStrategist).
Your mission is to formulate breakthrough short-form video topics that achieve maximum viewer retention.

KEY RESPONSIBILITIES:
1. Target Niche & Sub-Category: Focus exclusively on the instructed niche and select an under-explored sub-category.
2. Concrete Physical Metaphor: Every concept MUST be anchored to an inanimate, tangible physical object or mechanism (e.g. leaky cup, treadmill, anchor, locked chest, balancing scale, drawbridge, severed rope).
   - STRICTLY FORBIDDEN: Abstract liquid morphing, floating misty energy, or shapeless blobs.
3. Psychological Counter-Intuitive Hook: The premise must challenge conventional wisdom or expose a hidden cognitive/social trap.
4. Anti-Repetition Guarantee: Strictly obey all negative exclusions for past titles and past metaphors.
"""
    return Agent(
        name="TopicStrategist",
        model=get_model(primary),
        fallback_models=get_fallback_models(primary),
        instructions=instructions,
        output_schema=TopicIdea,
    )


def generate_next_topic(model_id: Optional[str] = None) -> TopicIdea:
    """Autonomous topic generation with 70/30 ratio scheduling and anti-repetition constraints."""
    history = load_history()
    target_niche = get_target_niche(history)
    exclusions = get_recent_exclusions(history, window_size=25)
    taxonomy = load_seed_taxonomy()

    niche_data = taxonomy.get(target_niche, {})
    sub_categories = niche_data.get("sub_categories", [])
    niche_name = niche_data.get("name", target_niche)

    sub_category_names = [sc["name"] for sc in sub_categories]
    recent_subs = exclusions["recent_subcategories"]
    recommended_subs = [name for name in sub_category_names if name not in recent_subs] or sub_category_names

    prompt = f"""Generate the next video topic concept.

TARGET NICHE:
- Key: {target_niche}
- Name: {niche_name}

SUB-CATEGORY OPTIONS:
- Available: {json.dumps(sub_categories, indent=2)}
- Recommended for diversity (avoiding recent clustering): {recommended_subs}

STRICT EXCLUSIONS (DO NOT REUSE):
- Excluded Titles (Last 25 runs): {exclusions['excluded_titles']}
- Excluded Physical Metaphors (Last 25 runs): {exclusions['excluded_metaphors']}

MANDATORY RULES:
1. Niche must be exactly: "{target_niche}".
2. Choose one specific sub_category from the available options.
3. The physical_metaphor MUST be an inanimate, tangible physical object/mechanism that is NOT in the excluded list.
4. The title must be punchy, curiosity-inducing, and under 12 words.
5. hook_premise must highlight an unexpected contradiction or counter-intuitive truth.
"""

    topic_agent = create_topic_agent(model_id)
    response = topic_agent.run(prompt)
    topic_idea = parse_structured_output(response.content, TopicIdea)

    # Persist topic into history
    record_topic_to_history(topic_idea)

    print("\n" + "=" * 70)
    print("💡 AUTONOMOUS TOPIC GENERATED")
    print(f"Niche:        {topic_idea.niche} ({topic_idea.sub_category})")
    print(f"Title:        {topic_idea.title}")
    print(f"Metaphor:     {topic_idea.physical_metaphor}")
    print(f"Hook Premise: {topic_idea.hook_premise}")
    print("=" * 70 + "\n")

    return topic_idea
