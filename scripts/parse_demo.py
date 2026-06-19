#!/usr/bin/env python3
"""
Parse a CS:GO .dem file and save structured JSON output.

Usage:
    python scripts/parse_demo.py --demo path/to/match.dem --output data/parsed/match.json
"""

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.demo_parser import DemoParser


def main():
    parser = argparse.ArgumentParser(description="Parse a CS:GO .dem file")
    parser.add_argument("--demo",   required=True,  help="Path to .dem file")
    parser.add_argument("--output", required=True,  help="Output JSON path")
    parser.add_argument("--sample", type=int, default=8,
                        help="Sample every N ticks (default: 8 = ~8 samples/sec at 64-tick)")
    args = parser.parse_args()

    p = DemoParser(args.demo)
    rounds = p.parse(sample_every_n_ticks=args.sample)
    p.save(rounds, args.output)
    print(f"✅ Parsed {len(rounds)} rounds → {args.output}")


if __name__ == "__main__":
    main()
