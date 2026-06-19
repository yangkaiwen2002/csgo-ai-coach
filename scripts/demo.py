#!/usr/bin/env python3
"""
End-to-End Demo
---------------
Runs the full CS:GO AI Coach pipeline with synthetic data.
No .dem file or pre-trained checkpoint needed.

Usage:
    python scripts/demo.py                        # synthetic data, full pipeline
    python scripts/demo.py --demo match.dem       # real .dem file
    python scripts/demo.py --rounds 8 --epochs 5  # quick run
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from src.data.fake_generator import (
    generate_fake_dataset,
    SCENARIO_NAMES,
    SCENARIO_FNS,
)
from src.features.feature_extractor import FeatureExtractor, ACTIONS
from src.models.decision_model import ModelTrainer, CSGODecisionTransformer, FEATURE_DIM, NUM_ACTIONS
from src.training.win_rate_labeler import WinRateLabeler


# ── Banner ────────────────────────────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════╗
║           CS:GO AI COACH  —  REPLAY DEMO            ║
╚══════════════════════════════════════════════════════╝
"""


# ── Pipeline steps ────────────────────────────────────────────────────────────

def step1_generate_data(n_rounds: int, seed: int) -> list[dict]:
    print(f"\n[1/4] Generating {n_rounds} synthetic rounds (seed={seed}) ...")
    rounds = generate_fake_dataset(n_rounds=n_rounds, seed=seed)
    scenario_map = {fn.__name__: name for fn, name in SCENARIO_NAMES.items()}
    for r in rounds[:4]:
        print(f"  Round {r['round_num']:2d}: {len(r['ticks'])} ticks  "
              f"| label = {r['ticks'][0].get('optimal_action', '?')}")
    print(f"  ... ({n_rounds} rounds total)")
    return rounds


def step2_label_with_win_rate(rounds: list[dict]) -> list[dict]:
    """
    Demonstrate the win-rate labeler on synthetic data.
    In production this runs on 1000+ pro matches; here it refines the
    pre-set scenario labels using aggregated win-rate evidence.
    """
    print("\n[2/4] Running win-rate labeler ...")
    labeler = WinRateLabeler(perspective="CT")
    labeled = labeler.fit_transform(rounds)

    table = labeler.win_rate_table()
    print(f"  State buckets discovered: {len(table)}")
    # Print the two most data-rich buckets as examples
    for bucket_str, rates in list(table.items())[:2]:
        best = max(rates, key=rates.get)
        print(f"  Bucket {bucket_str}")
        print(f"    best action → {best} (win rate {rates[best]:.0%})")

    return labeled


def step3_train(rounds: list[dict], epochs: int, config_path: str) -> ModelTrainer:
    print(f"\n[3/4] Extracting features and training for {epochs} epochs ...")

    extractor = FeatureExtractor(map_name="de_dust2")
    sequences = extractor.process_rounds(rounds, perspective="CT")
    X, y = extractor.to_arrays(sequences)

    print(f"  Dataset shape: X={X.shape}, y={y.shape}")
    unique, counts = np.unique(y, return_counts=True)
    print("  Class distribution:")
    for cls_idx, cnt in zip(unique, counts):
        print(f"    {ACTIONS[cls_idx]:12s} : {cnt:4d} samples")

    trainer = ModelTrainer(config_path=config_path)
    trainer.cfg["epochs"] = epochs
    losses = trainer.train(X, y)
    trainer.save("checkpoints/demo_model.pt")

    print(f"\n  Training complete. Final loss: {losses[-1]:.4f}")
    return trainer


def step4_analyze(rounds: list[dict], trainer: ModelTrainer) -> None:
    print("\n[4/4] Running AI coaching analysis ...\n")
    print("=" * 58)
    print("  CS:GO AI COACH  —  ROUND-BY-ROUND ANALYSIS")
    print("=" * 58)

    extractor = FeatureExtractor(map_name="de_dust2")
    sequences = extractor.process_rounds(rounds, perspective="CT")
    model     = trainer.model
    device    = trainer.device
    model.eval()

    # Map each scenario function name to scenario label
    scenario_fn_cycle = SCENARIO_FNS
    scenario_name_cycle = list(SCENARIO_NAMES.values())

    for seq, raw_round in zip(sequences, rounds):
        round_num = seq["round_num"]
        features  = torch.from_numpy(seq["features"]).float().unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(features)                                      # (1, T, 9)
            probs  = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()  # (T, 9)

        # Find the single most decisive moment (highest max probability)
        best_t     = int(probs.max(axis=1).argmax())
        best_probs = probs[best_t]
        best_idx   = int(best_probs.argmax())
        confidence = float(best_probs[best_idx])

        # Ground-truth label for this round (from fake generator)
        tick_labels = [t.get("optimal_action", "?") for t in raw_round["ticks"]]
        from collections import Counter
        gt_action = Counter(tick_labels).most_common(1)[0][0]

        # Situation description at best moment
        tick_state = raw_round["ticks"][min(best_t, len(raw_round["ticks"]) - 1)]
        players    = tick_state.get("players", [])
        ct_alive   = sum(1 for p in players if p["team"] == "CT" and p["is_alive"])
        t_alive    = sum(1 for p in players if p["team"] == "T"  and p["is_alive"])
        time_rem   = tick_state.get("time_remaining", 0)
        bomb_str   = "[BOMB PLANTED] " if tick_state.get("bomb_planted") else ""

        # Scenario name
        scenario_name = scenario_name_cycle[(round_num - 1) % len(scenario_name_cycle)]

        # Confidence bar
        bar = "█" * int(confidence * 12) + "░" * (12 - int(confidence * 12))

        print(f"\n  Round {round_num:2d}  [{scenario_name}]")
        print(f"  Situation : {bomb_str}{ct_alive}v{t_alive} | {time_rem:.0f}s remaining")
        print(f"  AI says   : {ACTIONS[best_idx].upper()}")
        print(f"  Confidence: [{bar}] {confidence:.0%}")

        # Show alternative actions
        top3 = sorted(enumerate(best_probs), key=lambda x: -x[1])[:3]
        alts = "  |  ".join(f"{ACTIONS[i]} {p:.0%}" for i, p in top3)
        print(f"  Top-3     : {alts}")

        correct = "✓" if ACTIONS[best_idx] == gt_action else "✗"
        print(f"  Ground truth: {gt_action}  {correct}")

    print("\n" + "=" * 58)
    print("  HARD PARTS — what needs work before production:")
    print("=" * 58)
    notes = [
        ("awpy column names",
         "awpy 1.x vs 2.x have different column names (steamID vs steam_id,\n"
         "     isAlive vs alive). Add a version-detecting adapter in demo_parser.py."),
        ("Action detection",
         "inferring 'pushed A' from raw positions is ambiguous: a player moving\n"
         "     toward A could be peeking, rotating, or falling back. The look-ahead\n"
         "     window in WinRateLabeler helps but needs tuning per map."),
        ("State bucketing",
         "the 5-feature discretization loses subtlety (e.g. which CORNER of A site?).\n"
         "     Replace with a learned state encoder (VAE or SimCLR) over larger datasets."),
        ("Round boundary leakage",
         "RoundSequenceDataset uses sliding windows that can span round ends.\n"
         "     Use per-round padding + masking instead."),
        ("Class imbalance",
         "hold_A/hold_B will dominate. Add class weights to cross_entropy loss\n"
         "     (computed from training set label frequency)."),
        ("Map support",
         "area bounds are hardcoded for de_dust2. Each map needs its own bounds\n"
         "     file, or use awpy's NavMesh to auto-derive them."),
        ("Semi-supervised",
         "WinRateLabeler generates labels from pro data. Unlabeled pub matches\n"
         "     can then be incorporated via pseudo-labeling or consistency loss."),
    ]
    for title, detail in notes:
        print(f"\n  [{title}]")
        print(f"     {detail}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CS:GO AI Coach end-to-end demo")
    parser.add_argument("--demo",   default=None,
                        help="Path to real .dem file (omit to use fake data)")
    parser.add_argument("--rounds", type=int, default=16,
                        help="Number of rounds to generate / analyze (default: 16)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Training epochs (default: 20, increase for better accuracy)")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--config", default="configs/model_config.yaml")
    args = parser.parse_args()

    print(BANNER)

    if args.demo:
        # Real .dem path — parse with awpy
        print(f"[Mode] REAL DEMO: {args.demo}")
        from src.data.demo_parser import DemoParser
        from src.training.win_rate_labeler import WinRateLabeler
        parser_obj = DemoParser(args.demo)
        raw_rounds = parser_obj.parse()
        labeler    = WinRateLabeler()
        rounds     = labeler.fit_transform(raw_rounds)
    else:
        print("[Mode] SYNTHETIC DATA (no .dem file needed)")
        rounds = step1_generate_data(args.rounds, args.seed)
        rounds = step2_label_with_win_rate(rounds)

    trainer = step3_train(rounds, args.epochs, args.config)
    step4_analyze(rounds, trainer)


if __name__ == "__main__":
    main()
