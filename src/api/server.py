"""
FastAPI Backend
---------------
Exposes the replay analysis pipeline as an HTTP API.

Endpoints:
  POST /analyze        — upload a .dem file, get back a per-round JSON report
  POST /analyze/fake   — run analysis on synthetic data (no .dem needed; for testing)
  GET  /health         — liveness check

Usage (after pip install -r requirements.txt):
  uvicorn src.api.server:app --reload --port 8000
"""

import json
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import JSONResponse

from src.analysis.replay_analyzer import ReplayAnalyzer
from src.data.fake_generator import generate_fake_dataset
from src.features.feature_extractor import FeatureExtractor
from src.models.decision_model import CSGODecisionTransformer, FEATURE_DIM, NUM_ACTIONS
import torch
import numpy as np

app = FastAPI(
    title="CS:GO AI Coach API",
    description="Upload a .dem replay and get per-round decision coaching.",
    version="0.1.0",
)

# Singleton analyzer — loaded once at startup so model weights aren't re-read on every request
_analyzer: ReplayAnalyzer = None


def get_analyzer() -> ReplayAnalyzer:
    global _analyzer
    if _analyzer is None:
        model_path = "checkpoints/best_model.pt"
        _analyzer = ReplayAnalyzer(model_path=model_path)
    return _analyzer


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": Path("checkpoints/best_model.pt").exists()}


@app.post("/analyze")
async def analyze_demo(
    file: UploadFile = File(..., description="CS:GO .dem replay file"),
    steam_id: str = Query(None, description="Focus on this player's Steam ID"),
    side: str    = Query("CT",  description="CT or T"),
    max_rounds: int = Query(None, description="Limit analysis to first N rounds"),
):
    """
    Upload a .dem file and receive per-round coaching recommendations.
    The file is written to a temp location, parsed, analyzed, then deleted.
    """
    if not file.filename.endswith(".dem"):
        raise HTTPException(status_code=400, detail="File must be a .dem file")

    with tempfile.NamedTemporaryFile(suffix=".dem", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        analyzer = get_analyzer()
        reports = analyzer.analyze(
            demo_path=tmp_path,
            player_steam_id=steam_id,
            perspective=side,
            max_rounds=max_rounds,
        )
        return _reports_to_json(reports)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/analyze/fake")
async def analyze_fake(
    n_rounds: int = Query(12, ge=1, le=50, description="Number of synthetic rounds"),
    side: str    = Query("CT", description="CT or T"),
):
    """
    Run analysis on synthetic data — no .dem file required.
    Useful for testing the API without a real match file.
    """
    rounds_data = generate_fake_dataset(n_rounds=n_rounds)
    extractor   = FeatureExtractor(map_name="de_dust2")
    sequences   = extractor.process_rounds(rounds_data, perspective=side)

    # Load model (uses random weights if no checkpoint)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model   = CSGODecisionTransformer(feature_dim=FEATURE_DIM, num_actions=NUM_ACTIONS).to(device)
    ckpt    = Path("checkpoints/best_model.pt")
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    results = []
    for seq, raw_round in zip(sequences, rounds_data):
        features = torch.from_numpy(seq["features"]).float().unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(features)
            probs  = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        best_tick_idx = int(probs.max(axis=1).argmax())
        best_action_idx = int(probs[best_tick_idx].argmax())
        from src.features.feature_extractor import ACTIONS
        results.append({
            "round_num": seq["round_num"],
            "n_ticks": len(seq["features"]),
            "recommended_action": ACTIONS[best_action_idx],
            "confidence": float(probs[best_tick_idx].max()),
            "ground_truth_label": _most_common_label(raw_round),
        })

    return JSONResponse(content={"rounds": results})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _reports_to_json(reports) -> dict:
    output = []
    for r in reports:
        output.append({
            "round_num":           r.round_num,
            "ct_won":              r.ct_won,
            "overall_assessment":  r.overall_assessment,
            "key_moments": [
                {
                    "tick":                m.tick,
                    "time_remaining":      m.time_remaining,
                    "recommended_action":  m.recommended_action,
                    "confidence":          round(m.confidence, 3),
                    "top_alternatives":    m.top_alternatives,
                    "situation_summary":   m.situation_summary,
                }
                for m in r.key_moments
            ],
        })
    return {"rounds": output}


def _most_common_label(round_data: dict) -> str:
    from collections import Counter
    labels = [t.get("optimal_action", "hold_A") for t in round_data.get("ticks", [])]
    if not labels:
        return "hold_A"
    return Counter(labels).most_common(1)[0][0]
