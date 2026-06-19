#!/usr/bin/env python3
"""
Analyze a CS:GO replay and print AI coaching recommendations.

Usage:
    python scripts/analyze.py --demo path/to/match.dem
    python scripts/analyze.py --demo match.dem --player 76561198012345678 --save report.json
"""

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.replay_analyzer import ReplayAnalyzer


def main():
    parser = argparse.ArgumentParser(description="Analyze a CS:GO replay")
    parser.add_argument("--demo",    required=True,          help="Path to .dem file")
    parser.add_argument("--model",   default="checkpoints/best_model.pt", help="Model checkpoint")
    parser.add_argument("--player",  default=None,           help="Steam ID to focus on")
    parser.add_argument("--side",    default="CT",           help="CT or T")
    parser.add_argument("--rounds",  type=int, default=None, help="Limit to N rounds")
    parser.add_argument("--save",    default=None,           help="Save report to JSON file")
    args = parser.parse_args()

    analyzer = ReplayAnalyzer(model_path=args.model)
    reports  = analyzer.analyze(
        demo_path=args.demo,
        player_steam_id=args.player,
        perspective=args.side,
        max_rounds=args.rounds,
    )

    analyzer.print_report(reports)

    if args.save:
        analyzer.save_report(reports, args.save)


if __name__ == "__main__":
    main()
