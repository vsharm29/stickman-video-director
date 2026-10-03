"""Unit tests for Topic Generation & Anti-Repetition Engine (Amendment 01)."""

import json
from pathlib import Path
import pytest

from src.schemas import TopicIdea, TopicHistoryEntry
from src.topic_manager import (
    load_history,
    get_target_niche,
    get_recent_exclusions,
    record_topic_to_history,
    record_custom_topic_to_history,
    load_seed_taxonomy,
)


def test_seed_taxonomy_structure():
    """Verify that data/sub_categories.json contains required niches and subcategories."""
    taxonomy = load_seed_taxonomy()
    assert "niche_1_mindset" in taxonomy
    assert "niche_2_social" in taxonomy

    mindset_subs = [sc["name"] for sc in taxonomy["niche_1_mindset"]["sub_categories"]]
    social_subs = [sc["name"] for sc in taxonomy["niche_2_social"]["sub_categories"]]

    # Check required sub-categories for Niche 1
    assert "Mental Models" in mindset_subs
    assert "Friction Theory" in mindset_subs
    assert "Cognitive Biases" in mindset_subs
    assert "Sunk Cost Fallacy" in mindset_subs
    assert "Dopamine Regulation" in mindset_subs
    assert "Strategic Quitting" in mindset_subs
    assert "Compounding Habits" in mindset_subs

    # Check required sub-categories for Niche 2
    assert "Boundary Defense" in social_subs
    assert "The Power of Silence" in social_subs
    assert "Unspoken Leverage" in social_subs
    assert "Covert Contracts" in social_subs
    assert "Expectation Mismatches" in social_subs
    assert "Social Testing" in social_subs


def test_target_niche_70_30_ratio_simulation():
    """Verify that across 10 consecutive simulated runs from empty, exactly 7 mindset and 3 social are selected."""
    simulated_history = []
    selected_niches = []

    for i in range(10):
        target = get_target_niche(simulated_history)
        selected_niches.append(target)
        simulated_history.append({
            "date": f"2026-10-0{i+1}T00:00:00",
            "niche": target,
            "sub_category": "Test",
            "title": f"Title {i}",
            "physical_metaphor": f"Metaphor {i}",
        })

    social_count = selected_niches.count("niche_2_social")
    mindset_count = selected_niches.count("niche_1_mindset")

    assert social_count == 3, f"Expected exactly 3 niche_2_social, got {social_count}"
    assert mindset_count == 7, f"Expected exactly 7 niche_1_mindset, got {mindset_count}"
    assert len(selected_niches) == 10


def test_target_niche_rolling_window():
    """Verify that get_target_niche only considers the last 10 entries."""
    # 10 mindset entries followed by 2 social entries -> social_count in last 10 is 2 -> should return niche_2_social
    history = [{"niche": "niche_1_mindset"} for _ in range(8)] + [{"niche": "niche_2_social"} for _ in range(2)]
    assert get_target_niche(history) == "niche_2_social"

    # 7 mindset entries and 3 social entries in last 10 -> social_count is 3 -> should return niche_1_mindset
    history = [{"niche": "niche_1_mindset"} for _ in range(7)] + [{"niche": "niche_2_social"} for _ in range(3)]
    assert get_target_niche(history) == "niche_1_mindset"


def test_get_recent_exclusions_window_25():
    """Verify that exclusions extract the last 25 titles/metaphors and last 5 subcategories."""
    dummy_history = [
        {
            "date": f"2026-09-{i:02d}T00:00:00",
            "niche": "niche_1_mindset",
            "sub_category": f"SubCat_{i}",
            "title": f"Title_{i}",
            "physical_metaphor": f"Metaphor_{i}",
        }
        for i in range(1, 35)  # 34 entries
    ]

    exclusions = get_recent_exclusions(dummy_history, window_size=25)

    # Excluded titles and metaphors should be from last 25 (indices 10 to 34 -> i from 10 to 34)
    assert len(exclusions["excluded_titles"]) == 25
    assert len(exclusions["excluded_metaphors"]) == 25
    assert "Title_34" in exclusions["excluded_titles"]
    assert "Title_10" in exclusions["excluded_titles"]
    assert "Title_9" not in exclusions["excluded_titles"]  # Outside 25-window

    assert "Metaphor_34" in exclusions["excluded_metaphors"]
    assert "Metaphor_10" in exclusions["excluded_metaphors"]
    assert "Metaphor_9" not in exclusions["excluded_metaphors"]

    # Recent subcategories from last 5
    assert len(exclusions["recent_subcategories"]) == 5
    assert exclusions["recent_subcategories"] == [f"SubCat_{i}" for i in range(30, 35)]


def test_get_recent_exclusions_all_time():
    """Verify that when window_size is None, exclusions capture 100% of all-time historical titles and metaphors."""
    dummy_history = [
        {
            "date": f"2026-08-{i:02d}T00:00:00",
            "niche": "niche_1_mindset",
            "sub_category": f"SubCat_{i}",
            "title": f"Title_{i}",
            "physical_metaphor": f"Metaphor_{i}",
        }
        for i in range(1, 60)  # 59 entries
    ]

    exclusions = get_recent_exclusions(dummy_history, window_size=None)

    # All 59 titles and metaphors must be excluded, never forgotten after 25 days
    assert len(exclusions["excluded_titles"]) == 59
    assert len(exclusions["excluded_metaphors"]) == 59
    assert "Title_1" in exclusions["excluded_titles"]
    assert "Title_59" in exclusions["excluded_titles"]
    assert "Metaphor_1" in exclusions["excluded_metaphors"]
    assert "Metaphor_59" in exclusions["excluded_metaphors"]


def test_record_topic_to_history(tmp_path, monkeypatch):
    """Verify that record_topic_to_history persists new entries into JSON."""
    test_history_file = tmp_path / "test_topics_history.json"
    test_data_dir = tmp_path

    monkeypatch.setattr("src.topic_manager.TOPICS_HISTORY_PATH", test_history_file)
    monkeypatch.setattr("src.topic_manager.DATA_DIR", test_data_dir)

    topic = TopicIdea(
        niche="niche_1_mindset",
        sub_category="Cognitive Biases",
        title="Why Smart People Make Dumb Mistakes",
        hook_premise="The smarter you are, the faster you rationalize bad calls.",
        physical_metaphor="curved carnival mirror",
        narrative_angle="Looking into distorted mirrors showing false confidence.",
    )

    entry = record_topic_to_history(topic)
    assert entry.title == "Why Smart People Make Dumb Mistakes"
    assert entry.physical_metaphor == "curved carnival mirror"

    # Read back
    history = json.loads(test_history_file.read_text(encoding="utf-8"))
    assert len(history) == 1
    assert history[0]["title"] == "Why Smart People Make Dumb Mistakes"
    assert history[0]["physical_metaphor"] == "curved carnival mirror"


def test_record_custom_topic_to_history(tmp_path, monkeypatch):
    """Verify that user-provided custom topics are recorded into history."""
    test_history_file = tmp_path / "test_topics_history.json"
    test_data_dir = tmp_path

    monkeypatch.setattr("src.topic_manager.TOPICS_HISTORY_PATH", test_history_file)
    monkeypatch.setattr("src.topic_manager.DATA_DIR", test_data_dir)

    entry = record_custom_topic_to_history("My Custom Topic", "data/test_folder")
    assert entry.title == "My Custom Topic"
    assert entry.niche == "custom"
    assert entry.folder_path == "data/test_folder"

    history = json.loads(test_history_file.read_text(encoding="utf-8"))
    assert len(history) == 1
    assert history[0]["title"] == "My Custom Topic"
    assert history[0]["folder_path"] == "data/test_folder"


def test_deduplication_sliding_window_validation():
    """Verify that no metaphor appears twice within any 25-entry sliding window."""
    # Create 40 distinct items
    history = [
        {
            "date": f"2026-08-{i:02d}T10:00:00",
            "niche": "niche_1_mindset",
            "sub_category": "Compounding Habits",
            "title": f"Habit Key {i}",
            "physical_metaphor": f"distinct_object_{i}",
        }
        for i in range(1, 41)
    ]

    # Verify every 25-item sliding window has 25 unique metaphors
    window_size = 25
    for start_idx in range(len(history) - window_size + 1):
        window = history[start_idx : start_idx + window_size]
        metaphors = [item["physical_metaphor"] for item in window]
        assert len(metaphors) == len(set(metaphors)), f"Duplicate metaphor found in window starting at {start_idx}"

    # Verify exclusions correctly capture the last 25
    exclusions = get_recent_exclusions(history, window_size=25)
    assert len(set(exclusions["excluded_metaphors"])) == 25
    assert "distinct_object_40" in exclusions["excluded_metaphors"]
    assert "distinct_object_16" in exclusions["excluded_metaphors"]
    assert "distinct_object_15" not in exclusions["excluded_metaphors"]
