"""
Feature Extractor
-----------------
Converts raw parsed game states into model-ready feature vectors.

Key insight from domain expert:
  Features should represent ONLY what a player can actually observe —
  not the "God view". This mirrors real gameplay decision-making.

Core features:
  - Team movement direction (pushing A / B / mid)
  - Gunfire heard (which side, how many shots)
  - Player rotations (teammate moving from A → B = "A rotating")
  - Economic state (eco / half-buy / full-buy)
  - Time pressure (time remaining, bomb planted)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
import math


# ── Map areas for de_dust2 (extend for other maps) ──────────────────────────
MAP_AREAS = {
    "de_dust2": {
        "A_site":    (-900, 400, 300, 1500),   # (x_min, y_min, x_max, y_max)
        "A_long":    (-2000, -300, -900, 1000),
        "A_short":   (-500, 800, 300, 1800),
        "mid":       (-500, -500, 500, 400),
        "B_site":    (1000, -300, 2000, 600),
        "B_tunnels": (500, -1800, 1500, -300),
        "T_spawn":   (-2500, -2000, -1000, -800),
        "CT_spawn":  (500, 1000, 2000, 2500),
    }
}

ACTIONS = ["push_A", "push_B", "hold_A", "hold_B", "fall_back", "rotate_A", "rotate_B", "smoke_A", "smoke_B"]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}


@dataclass
class FeatureVector:
    """
    Fixed-size feature vector for one player at one tick.
    All values normalized to [0, 1] or one-hot encoded.
    """
    # --- Economy ---
    economy_ratio: float        # team money / max possible
    has_rifle: float            # 1.0 if has AK/M4/AWP
    team_avg_armor: float

    # --- Time / Round state ---
    time_ratio: float           # time_remaining / 115.0
    bomb_planted: float         # 0 or 1
    round_ratio: float          # round_num / 30

    # --- Team alive count ---
    ct_alive_ratio: float       # alive CTs / 5
    t_alive_ratio: float        # alive Ts / 5

    # --- Movement signals (where is the team going?) ---
    t_moving_to_A: float        # fraction of T players moving toward A
    t_moving_to_B: float
    ct_moving_to_A: float
    ct_moving_to_B: float

    # --- Acoustic signals (what can this player HEAR?) ---
    shots_heard_A: float        # shots in A area in last 2 seconds (normalized)
    shots_heard_B: float
    shots_heard_mid: float
    footsteps_heard: float      # nearby footsteps (0/1)

    # --- Rotation signals ---
    a_player_rotating_B: float  # CT on A rotating toward B
    b_player_rotating_A: float  # CT on B rotating toward A

    # --- Position of this player ---
    area_A_site: float          # one-hot: which area is this player in?
    area_B_site: float
    area_mid: float
    area_A_long: float
    area_other: float

    def to_numpy(self) -> np.ndarray:
        return np.array([
            self.economy_ratio, self.has_rifle, self.team_avg_armor,
            self.time_ratio, self.bomb_planted, self.round_ratio,
            self.ct_alive_ratio, self.t_alive_ratio,
            self.t_moving_to_A, self.t_moving_to_B,
            self.ct_moving_to_A, self.ct_moving_to_B,
            self.shots_heard_A, self.shots_heard_B, self.shots_heard_mid,
            self.footsteps_heard,
            self.a_player_rotating_B, self.b_player_rotating_A,
            self.area_A_site, self.area_B_site, self.area_mid,
            self.area_A_long, self.area_other,
        ], dtype=np.float32)

    @property
    def dim(self) -> int:
        return 23


class FeatureExtractor:
    """
    Transforms raw round state dicts (from DemoParser) into
    FeatureVector sequences ready for model training.

    Example
    -------
    >>> extractor = FeatureExtractor(map_name="de_dust2")
    >>> sequences = extractor.process_rounds(rounds_data, perspective="CT")
    >>> X, y = extractor.to_arrays(sequences)
    """

    HEARING_RANGE = 2000.0   # units; roughly how far gunfire is audible
    MAX_SHOTS_PER_WINDOW = 10

    def __init__(self, map_name: str = "de_dust2"):
        self.map_name = map_name
        self.areas = MAP_AREAS.get(map_name, {})

    # ── Public API ─────────────────────────────────────────────────────────

    def process_rounds(
        self,
        rounds_data: list[dict],
        perspective: str = "CT",         # "CT" or "T"
        focus_player_id: Optional[str] = None,
    ) -> list[dict]:
        """
        Process all rounds. Returns list of:
          {
            "round_num": int,
            "features": np.ndarray of shape (T, feature_dim),
            "labels":   np.ndarray of shape (T,)   — action indices
            "win_rate": float                        — round outcome label
          }
        """
        sequences = []
        for round_data in rounds_data:
            seq = self._process_round(round_data, perspective, focus_player_id)
            if seq is not None:
                sequences.append(seq)
        return sequences

    def to_arrays(self, sequences: list[dict]):
        """Flatten sequences into (X, y) numpy arrays for training."""
        X_list, y_list = [], []
        for seq in sequences:
            X_list.append(seq["features"])
            y_list.append(seq["labels"])
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        return X, y

    # ── Internal ───────────────────────────────────────────────────────────

    def _process_round(
        self, round_data: dict, perspective: str, focus_player_id: Optional[str]
    ) -> Optional[dict]:
        ticks = round_data.get("ticks", [])
        if not ticks:
            return None

        feature_list = []
        label_list = []

        prev_positions: dict[str, tuple] = {}

        for i, tick_state in enumerate(ticks):
            players = tick_state.get("players", [])
            events = tick_state.get("events_this_tick", [])

            ct_players = [p for p in players if p["team"] == "CT"]
            t_players  = [p for p in players if p["team"] == "T"]

            # Focus player defaults to first alive player of the perspective team
            focus = self._get_focus_player(
                players, perspective, focus_player_id
            )
            if focus is None:
                continue

            fv = self._build_feature_vector(
                tick_state, focus, ct_players, t_players,
                events, prev_positions, i, len(ticks)
            )
            label = self._infer_label(focus, ct_players, t_players, tick_state)

            feature_list.append(fv.to_numpy())
            label_list.append(label)

            # Update previous positions
            for p in players:
                prev_positions[p["steam_id"]] = (p["x"], p["y"])

        if not feature_list:
            return None

        return {
            "round_num": round_data["round_num"],
            "features": np.stack(feature_list),
            "labels":   np.array(label_list, dtype=np.int64),
        }

    def _get_focus_player(
        self, players: list, perspective: str, focus_id: Optional[str]
    ):
        team_players = [p for p in players if p["team"] == perspective and p["is_alive"]]
        if not team_players:
            return None
        if focus_id:
            match = [p for p in team_players if p["steam_id"] == focus_id]
            return match[0] if match else team_players[0]
        return team_players[0]

    def _build_feature_vector(
        self, tick_state, focus, ct_players, t_players,
        events, prev_positions, tick_idx, total_ticks
    ) -> FeatureVector:

        # Economy
        rifles = {"AK-47", "M4A1", "M4A4", "AWP", "SG 553", "AUG", "FAMAS", "Galil AR"}
        ct_money = sum(100 for p in ct_players)    # placeholder; real data has money field
        economy_ratio = min(ct_money / (5 * 16000), 1.0)
        has_rifle = 1.0 if focus.get("active_weapon", "") in rifles else 0.0
        team_avg_armor = sum(p.get("armor", 0) for p in ct_players) / max(len(ct_players), 1) / 100.0

        # Time
        time_rem = tick_state.get("time_remaining", 0)
        time_ratio = min(time_rem / 115.0, 1.0)
        bomb_planted = float(tick_state.get("bomb_planted", False))
        round_num = tick_state.get("round_num", 1)
        round_ratio = min(round_num / 30.0, 1.0)

        # Alive counts
        ct_alive = sum(1 for p in ct_players if p.get("is_alive", False))
        t_alive  = sum(1 for p in t_players  if p.get("is_alive", False))
        ct_alive_ratio = ct_alive / 5.0
        t_alive_ratio  = t_alive  / 5.0

        # Movement direction
        t_to_A, t_to_B  = self._team_movement_signal(t_players,  prev_positions, "A", "B")
        ct_to_A, ct_to_B = self._team_movement_signal(ct_players, prev_positions, "A", "B")

        # Acoustic signals
        fx, fy = focus["x"], focus["y"]
        shots_A   = self._shots_near_area(events, "A_site",   fx, fy)
        shots_B   = self._shots_near_area(events, "B_site",   fx, fy)
        shots_mid = self._shots_near_area(events, "mid",      fx, fy)
        footsteps = 0.0  # could be derived from player proximity

        # Rotation signals
        a_rot_B = self._detect_rotation(ct_players, prev_positions, from_area="A_site", to_area="B_site")
        b_rot_A = self._detect_rotation(ct_players, prev_positions, from_area="B_site", to_area="A_site")

        # Player area (one-hot)
        area = self._get_area(fx, fy)
        area_A = float(area == "A_site")
        area_B = float(area == "B_site")
        area_mid = float(area == "mid")
        area_long = float(area == "A_long")
        area_other = float(area not in {"A_site", "B_site", "mid", "A_long"})

        return FeatureVector(
            economy_ratio=economy_ratio,
            has_rifle=has_rifle,
            team_avg_armor=team_avg_armor,
            time_ratio=time_ratio,
            bomb_planted=bomb_planted,
            round_ratio=round_ratio,
            ct_alive_ratio=ct_alive_ratio,
            t_alive_ratio=t_alive_ratio,
            t_moving_to_A=t_to_A,
            t_moving_to_B=t_to_B,
            ct_moving_to_A=ct_to_A,
            ct_moving_to_B=ct_to_B,
            shots_heard_A=shots_A,
            shots_heard_B=shots_B,
            shots_heard_mid=shots_mid,
            footsteps_heard=footsteps,
            a_player_rotating_B=a_rot_B,
            b_player_rotating_A=b_rot_A,
            area_A_site=area_A,
            area_B_site=area_B,
            area_mid=area_mid,
            area_A_long=area_long,
            area_other=area_other,
        )

    def _infer_label(self, focus, ct_players, t_players, tick_state) -> int:
        """
        Returns the action label for this tick.

        Priority order:
          1. "optimal_action" key — set by WinRateLabeler on pro data, or by
             FakeDataGenerator on synthetic data. This is the primary path.
          2. Heuristic fallback based on player position (used when no label exists,
             e.g. raw unlabeled demo files before the labeler has been run).
        """
        action = tick_state.get("optimal_action")
        if action and action in ACTION_TO_IDX:
            return ACTION_TO_IDX[action]

        # Fallback heuristic: if majority of alive CTs are on A, hold A; else hold B
        a_count = sum(
            1 for p in ct_players
            if p.get("is_alive") and self._get_area(p["x"], p["y"]) in ("A_site", "A_long", "A_short")
        )
        b_count = sum(
            1 for p in ct_players
            if p.get("is_alive") and self._get_area(p["x"], p["y"]) in ("B_site", "B_tunnels")
        )
        return ACTION_TO_IDX["hold_A"] if a_count >= b_count else ACTION_TO_IDX["hold_B"]

    def _team_movement_signal(
        self, players: list, prev_pos: dict, area_A: str, area_B: str
    ) -> tuple[float, float]:
        """What fraction of the team is moving toward each site?"""
        toward_A = toward_B = 0
        for p in players:
            if not p.get("is_alive"):
                continue
            sid = p["steam_id"]
            if sid not in prev_pos:
                continue
            dx = p["x"] - prev_pos[sid][0]
            dy = p["y"] - prev_pos[sid][1]
            # Rough directional heuristic for dust2:
            # A site is roughly at negative X; B is positive X
            if dx < -10:
                toward_A += 1
            elif dx > 10:
                toward_B += 1
        n = max(len(players), 1)
        return toward_A / n, toward_B / n

    def _shots_near_area(
        self, events: list, area_name: str, px: float, py: float
    ) -> float:
        """Normalized count of gunfire events near a map area."""
        bounds = self.areas.get(area_name)
        if bounds is None:
            return 0.0
        x_min, y_min, x_max, y_max = bounds
        area_cx = (x_min + x_max) / 2
        area_cy = (y_min + y_max) / 2
        dist_to_area = math.hypot(px - area_cx, py - area_cy)
        if dist_to_area > self.HEARING_RANGE:
            return 0.0
        count = sum(
            1 for e in events
            if e["event_type"] == "kill" and
               x_min <= e["x"] <= x_max and y_min <= e["y"] <= y_max
        )
        return min(count / self.MAX_SHOTS_PER_WINDOW, 1.0)

    def _detect_rotation(
        self, players: list, prev_pos: dict, from_area: str, to_area: str
    ) -> float:
        """Detect if any player is moving FROM one area TOWARD another."""
        from_bounds = self.areas.get(from_area)
        if from_bounds is None:
            return 0.0
        x_min, y_min, x_max, y_max = from_bounds
        to_bounds = self.areas.get(to_area)
        if to_bounds is None:
            return 0.0
        tx_min, ty_min, tx_max, ty_max = to_bounds
        to_cx = (tx_min + tx_max) / 2
        to_cy = (ty_min + ty_max) / 2

        for p in players:
            sid = p["steam_id"]
            if sid not in prev_pos:
                continue
            px, py = prev_pos[sid]
            # Was previously in from_area?
            if x_min <= px <= x_max and y_min <= py <= y_max:
                # Now moving toward to_area?
                dx = p["x"] - px
                dy = p["y"] - py
                toward = (p["x"] - to_cx) * dx + (p["y"] - to_cy) * dy
                if toward < 0:   # dot product < 0 means moving closer
                    return 1.0
        return 0.0

    def _get_area(self, x: float, y: float) -> str:
        for area_name, (x_min, y_min, x_max, y_max) in self.areas.items():
            if x_min <= x <= x_max and y_min <= y <= y_max:
                return area_name
        return "unknown"
