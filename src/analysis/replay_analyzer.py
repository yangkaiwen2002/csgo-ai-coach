"""
Replay Analyzer
---------------
User-facing module. Takes a .dem file and outputs a per-round
analysis: "At tick X, the optimal move was Y — you did Z."

This is the MVP feature the domain expert described:
  > "帮助普通玩家复盘" (help average players review their gameplay)
"""

import json
from pathlib import Path
from dataclasses import dataclass
import torch
import numpy as np

from src.data.demo_parser import DemoParser
from src.features.feature_extractor import FeatureExtractor, ACTIONS
from src.models.decision_model import CSGODecisionTransformer, FEATURE_DIM, NUM_ACTIONS


@dataclass
class DecisionMoment:
    """One moment in the replay where AI has a recommendation."""
    round_num: int
    tick: int
    time_remaining: float
    recommended_action: str
    confidence: float           # 0–1
    top_alternatives: list[tuple[str, float]]   # [(action, prob), ...]
    situation_summary: str      # human-readable description


@dataclass
class RoundReport:
    """Full AI analysis of one round."""
    round_num: int
    ct_won: bool
    key_moments: list[DecisionMoment]
    overall_assessment: str


class ReplayAnalyzer:
    """
    End-to-end replay analysis pipeline.

    Example
    -------
    >>> analyzer = ReplayAnalyzer(model_path="checkpoints/best_model.pt")
    >>> report = analyzer.analyze("my_match.dem", player_steam_id="76561198...")
    >>> analyzer.print_report(report)
    >>> analyzer.save_report(report, "analysis/my_match_report.json")
    """

    CONFIDENCE_THRESHOLD = 0.4   # Only flag moments where model is >40% confident
    TOP_K_MOMENTS = 5            # Show top 5 critical moments per round

    def __init__(
        self,
        model_path: str = "checkpoints/best_model.pt",
        map_name: str = "de_dust2",
        device: str = "auto",
    ):
        self.map_name = map_name
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )
        self.model = self._load_model(model_path)
        self.extractor = FeatureExtractor(map_name=map_name)

    # ── Public API ─────────────────────────────────────────────────────────

    def analyze(
        self,
        demo_path: str,
        player_steam_id: str = None,
        perspective: str = "CT",
        max_rounds: int = None,
    ) -> list[RoundReport]:
        """
        Analyze a .dem file and return per-round reports.

        Parameters
        ----------
        demo_path        : path to the .dem file
        player_steam_id  : focus on this player's perspective (None = team view)
        perspective      : "CT" or "T"
        max_rounds       : limit number of rounds analyzed
        """
        print(f"[Analyzer] Parsing demo: {demo_path}")
        parser = DemoParser(demo_path)
        rounds_data = parser.parse()

        if max_rounds:
            rounds_data = rounds_data[:max_rounds]

        print(f"[Analyzer] Extracting features for {len(rounds_data)} rounds...")
        sequences = self.extractor.process_rounds(
            rounds_data, perspective=perspective, focus_player_id=player_steam_id
        )

        print("[Analyzer] Running model inference...")
        reports = []
        for seq, raw_round in zip(sequences, rounds_data):
            report = self._analyze_round(seq, raw_round)
            reports.append(report)

        print(f"[Analyzer] Done. {len(reports)} rounds analyzed.")
        return reports

    def print_report(self, reports: list[RoundReport]) -> None:
        """Pretty-print the analysis to console."""
        print("\n" + "=" * 60)
        print("  CS:GO AI COACH — REPLAY ANALYSIS")
        print("=" * 60)

        for report in reports:
            outcome = "✅ CT WIN" if report.ct_won else "❌ CT LOSS"
            print(f"\n🔫 Round {report.round_num:2d}  {outcome}")
            print(f"   {report.overall_assessment}")

            if report.key_moments:
                print(f"   📍 Key Decision Moments:")
                for m in report.key_moments[:3]:
                    bar = "█" * int(m.confidence * 10) + "░" * (10 - int(m.confidence * 10))
                    print(f"      [{bar}] {m.confidence:.0%}  →  {m.recommended_action}")
                    print(f"      {m.situation_summary}")
                    alts = ", ".join(f"{a}({p:.0%})" for a, p in m.top_alternatives[:2])
                    print(f"      Alt: {alts}")

    def save_report(self, reports: list[RoundReport], output_path: str) -> None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        data = []
        for r in reports:
            data.append({
                "round_num": r.round_num,
                "ct_won": r.ct_won,
                "overall_assessment": r.overall_assessment,
                "key_moments": [
                    {
                        "tick": m.tick,
                        "time_remaining": m.time_remaining,
                        "recommended_action": m.recommended_action,
                        "confidence": m.confidence,
                        "top_alternatives": m.top_alternatives,
                        "situation_summary": m.situation_summary,
                    }
                    for m in r.key_moments
                ],
            })
        with open(out, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[Analyzer] Report saved → {out}")

    # ── Internal ───────────────────────────────────────────────────────────

    def _load_model(self, model_path: str) -> CSGODecisionTransformer:
        model = CSGODecisionTransformer(
            feature_dim=FEATURE_DIM,
            num_actions=NUM_ACTIONS,
        ).to(self.device)
        path = Path(model_path)
        if path.exists():
            model.load_state_dict(torch.load(path, map_location=self.device))
            print(f"[Analyzer] Model loaded from {model_path}")
        else:
            print(f"[Analyzer] ⚠️  No checkpoint at {model_path} — using random weights (train first!)")
        model.eval()
        return model

    def _analyze_round(self, seq: dict, raw_round: dict) -> RoundReport:
        features = torch.from_numpy(seq["features"]).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(features)   # (1, T, num_actions)
            probs  = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()  # (T, num_actions)

        key_moments = self._find_key_moments(probs, raw_round)
        ct_won = self._infer_ct_won(raw_round)

        assessment = self._generate_assessment(key_moments, ct_won)

        return RoundReport(
            round_num=seq["round_num"],
            ct_won=ct_won,
            key_moments=key_moments,
            overall_assessment=assessment,
        )

    def _find_key_moments(self, probs: np.ndarray, raw_round: dict) -> list[DecisionMoment]:
        """Find ticks where the model is highly confident about the best action."""
        moments = []
        ticks = raw_round.get("ticks", [])

        for t_idx, (tick_probs, tick_state) in enumerate(zip(probs, ticks)):
            best_action_idx = tick_probs.argmax()
            confidence = float(tick_probs[best_action_idx])

            if confidence < self.CONFIDENCE_THRESHOLD:
                continue

            top_k = sorted(enumerate(tick_probs), key=lambda x: -x[1])[:3]
            alternatives = [(ACTIONS[i], float(p)) for i, p in top_k[1:]]

            situation = self._describe_situation(tick_state)

            moments.append(DecisionMoment(
                round_num=raw_round["round_num"],
                tick=tick_state.get("tick", t_idx),
                time_remaining=float(tick_state.get("time_remaining", 0)),
                recommended_action=ACTIONS[best_action_idx],
                confidence=confidence,
                top_alternatives=alternatives,
                situation_summary=situation,
            ))

        # Return top-K moments by confidence
        moments.sort(key=lambda m: -m.confidence)
        return moments[:self.TOP_K_MOMENTS]

    def _describe_situation(self, tick_state: dict) -> str:
        """Generate a human-readable description of the game situation."""
        players = tick_state.get("players", [])
        ct_alive = sum(1 for p in players if p.get("team") == "CT" and p.get("is_alive"))
        t_alive  = sum(1 for p in players if p.get("team") == "T"  and p.get("is_alive"))
        t_rem = tick_state.get("time_remaining", 0)
        bomb = "💣 Bomb planted. " if tick_state.get("bomb_planted") else ""
        return f"{bomb}{ct_alive}v{t_alive} | {t_rem:.0f}s remaining"

    def _infer_ct_won(self, raw_round: dict) -> bool:
        """Infer round winner from the last tick state."""
        ticks = raw_round.get("ticks", [])
        if not ticks:
            return False
        last = ticks[-1]
        players = last.get("players", [])
        ct_alive = sum(1 for p in players if p.get("team") == "CT" and p.get("is_alive"))
        t_alive  = sum(1 for p in players if p.get("team") == "T"  and p.get("is_alive"))
        return ct_alive > 0 and t_alive == 0  # CTs win if Ts all dead

    def _generate_assessment(self, moments: list[DecisionMoment], ct_won: bool) -> str:
        if not moments:
            return "No clear decision points identified."
        avg_conf = sum(m.confidence for m in moments) / len(moments)
        outcome = "Round won." if ct_won else "Round lost."
        return f"{outcome} {len(moments)} key moments found. Avg model confidence: {avg_conf:.0%}."
