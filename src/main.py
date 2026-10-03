"""CLI entrypoint for Stickman Video Director Pipeline."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.pipeline import run_director_pipeline
from src.topic_manager import (
    generate_next_topic,
    update_topic_folder_in_history,
    record_custom_topic_to_history,
)
from src.config import SUPPORTED_RATIOS, STYLE_MAP


def main():
    parser = argparse.ArgumentParser(
        description="Stickman Video Director Pipeline (Phase A & B Automation)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--topic",
        "-t",
        type=str,
        default=None,
        help="Optional topic or concept. If omitted, TopicStrategist autonomously selects a unique non-repeating topic.",
    )
    parser.add_argument(
        "--ratio",
        "-r",
        choices=SUPPORTED_RATIOS,
        default="9:16",
        help="Video aspect ratio",
    )
    parser.add_argument(
        "--style",
        "-s",
        choices=list(STYLE_MAP.keys()),
        default="1A",
        help="Visual style: 1A (Classic Light), 1B (Classic Dark), 2A (Studio Tech), 2B (Cinematic Story)",
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=int,
        default=60,
        help="Target duration in seconds (must be a multiple of 10)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Max retries for review loop",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional Gemini model override (defaults to GEMINI_MODEL env var)",
    )

    args = parser.parse_args()

    try:
        topic_idea = None
        if args.topic:
            effective_topic = args.topic
            print(f"[CLI] Using user-provided topic: '{effective_topic}'")
        else:
            print("[CLI] No --topic provided. Invoking autonomous TopicStrategist...")
            topic_idea = generate_next_topic(model_id=args.model)
            effective_topic = f"{topic_idea.title} - Core Metaphor: {topic_idea.physical_metaphor}"
            print(f"[CLI] Autonomous topic selected: '{effective_topic}'")

        phase_a_proposal, phase_b_output, output_dir = run_director_pipeline(
            topic=effective_topic,
            aspect_ratio=args.ratio,
            style=args.style,
            duration=args.duration,
            max_retries=args.retries,
            model_id=args.model,
        )

        if topic_idea:
            update_topic_folder_in_history(topic_idea.title, str(output_dir))
        elif args.topic:
            record_custom_topic_to_history(args.topic, str(output_dir))

        print("\n" + "#" * 70)
        print("🎉 PIPELINE EXECUTION COMPLETED")
        print("#" * 70)
        print(f"Topic:        {phase_b_output.topic}")
        print(f"Total Clips:  {len(phase_b_output.clips)}")
        print(f"Total Length: {phase_b_output.total_duration_sec}s")
        print(f"Artifacts:    {output_dir}")
        print("-" * 70)
        for clip in phase_b_output.clips:
            print(f"Clip {clip.index} ({clip.duration_sec}s): Metaphor: {clip.visual_metaphor}")
            print(f"   VO: \"{clip.spoken_words}\"")
        print("#" * 70 + "\n")

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
