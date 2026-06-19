#!/usr/bin/env python3
"""
Train the CS:GO decision model on parsed demo data.

Usage:
    python scripts/train.py --data data/features/ --config configs/model_config.yaml
"""

import argparse
import json
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.feature_extractor import FeatureExtractor
from src.models.decision_model import ModelTrainer


def load_features(data_dir: str):
    """Load all parsed round JSON files and extract features."""
    data_path = Path(data_dir)
    all_X, all_y = [], []

    json_files = list(data_path.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No .json files found in {data_dir}")

    extractor = FeatureExtractor()
    for fpath in json_files:
        print(f"  Loading {fpath.name}...")
        with open(fpath) as f:
            rounds_data = json.load(f)
        sequences = extractor.process_rounds(rounds_data)
        X, y = extractor.to_arrays(sequences)
        all_X.append(X)
        all_y.append(y)

    return np.concatenate(all_X), np.concatenate(all_y)


def main():
    parser = argparse.ArgumentParser(description="Train CS:GO AI decision model")
    parser.add_argument("--data",   default="data/parsed/",         help="Dir of parsed .json demo files")
    parser.add_argument("--config", default="configs/model_config.yaml", help="Model config YAML")
    parser.add_argument("--output", default="checkpoints/best_model.pt", help="Save checkpoint path")
    args = parser.parse_args()

    print(f"[Train] Loading data from {args.data} ...")
    X, y = load_features(args.data)
    print(f"[Train] Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(set(y.tolist()))} action classes")

    trainer = ModelTrainer(config_path=args.config)
    losses = trainer.train(X, y)
    trainer.save(args.output)

    print(f"\n✅ Training complete. Final loss: {losses[-1]:.4f}")
    print(f"   Model saved → {args.output}")


if __name__ == "__main__":
    main()
