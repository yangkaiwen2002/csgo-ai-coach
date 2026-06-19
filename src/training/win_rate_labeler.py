"""
Win-Rate Labeler
----------------
Assigns action labels to each game-state tick without human annotation.

Core idea (from domain design):
  - Parse a corpus of pro matches
  - At each tick, detect which action the team took (from position data)
  - Record (discretized_state, action, round_outcome)
  - optimal_action = argmax P(win | action, state_bucket)

This module is the central innovation that makes self-supervised training possible.
The resulting annotated data feeds directly into FeatureExtractor.

Hard problems that need work later:
  1. Action detection is ambiguous: a player moving toward A could be pushing,
     peeking, or just falling back. Currently we use a look-ahead window
     (where does the player end up N ticks later?) which only works in replay mode.
  2. State bucketing: too fine → no samples per bucket. Too coarse → loses signal.
     The current 5-feature discretization is a starting point; a learned embedding
     (e.g. a VAE over game states) would be better.
  3. Small-sample noise: for rare states, win rates are unreliable.
     We use Laplace smoothing (add 1 win + 1 loss per bucket) to avoid
     confident predictions from 1-2 data points.
"""

import math
from collections import defaultdict
from typing import Optional

from src.features.feature_extractor import ACTIONS, ACTION_TO_IDX


# ── Action inference from position data ──────────────────────────────────────

DUST2_AREAS = {
    "A_site":    (-900,  400,  300, 1500),
    "A_long":   (-2000, -300, -900, 1000),
    "A_short":   (-500,  800,  300, 1800),
    "mid":       (-500, -500,  500,  400),
    "B_site":   (1000,  -300, 2000,  600),
    "B_tunnels": (500, -1800, 1500, -300),
    "T_spawn":  (-2500, -2000, -1000, -800),
    "CT_spawn":  (500,  1000, 2000, 2500),
}


def _area_of(x: float, y: float) -> str:
    for name, (xmin, ymin, xmax, ymax) in DUST2_AREAS.items():
        if xmin <= x <= xmax and ymin <= y <= ymax:
            return name
    return "unknown"


def _detect_action(
    players: list,
    prev_players: list,
    future_players: list,
    events: list,
    perspective: str = "CT",
) -> str:
    """
    Infer the team-level action at this tick.

    Uses a look-ahead window: where does each player end up 10 ticks later?
    That trajectory determines the intent better than instantaneous velocity.

    Parameters
    ----------
    players        : current tick player list
    prev_players   : player list from ~5 ticks ago (for rotation detection)
    future_players : player list from ~10 ticks ahead (for push/hold distinction)
    perspective    : which team we're labeling ("CT" or "T")
    events         : kill/grenade events at this tick (for smoke detection)
    """
    team = [p for p in players if p["team"] == perspective and p["is_alive"]]
    future_team = {p["steam_id"]: p for p in future_players if p["team"] == perspective}

    if not team:
        return "fall_back"

    # Count players by current area
    current_areas = [_area_of(p["x"], p["y"]) for p in team]
    a_count = sum(1 for a in current_areas if "A" in a)
    b_count = sum(1 for a in current_areas if "B" in a)

    # Detect movement direction using look-ahead
    moving_to_a = moving_to_b = stationary = 0
    for p in team:
        future = future_team.get(p["steam_id"])
        if future is None or not future.get("is_alive", False):
            continue
        dx = future["x"] - p["x"]
        dy = future["y"] - p["y"]
        dist = math.hypot(dx, dy)
        if dist < 80:
            stationary += 1
            continue
        future_area = _area_of(future["x"], future["y"])
        if "A" in future_area:
            moving_to_a += 1
        elif "B" in future_area:
            moving_to_b += 1

    # Detect rotation: was previously on one site, now moving to another
    prev_team = {p["steam_id"]: p for p in prev_players if p["team"] == perspective}
    rotating_to_b = rotating_to_a = 0
    for p in team:
        prev = prev_team.get(p["steam_id"])
        if prev is None:
            continue
        prev_area = _area_of(prev["x"], prev["y"])
        curr_area = _area_of(p["x"], p["y"])
        if "A" in prev_area and "B" in curr_area:
            rotating_to_b += 1
        elif "B" in prev_area and "A" in curr_area:
            rotating_to_a += 1

    if rotating_to_b > 0:
        return "rotate_B"
    if rotating_to_a > 0:
        return "rotate_A"

    n = len(team)
    if moving_to_a > n * 0.5:
        return "push_A"
    if moving_to_b > n * 0.5:
        return "push_B"
    if stationary > n * 0.5:
        return "hold_A" if a_count >= b_count else "hold_B"

    return "hold_A" if a_count >= b_count else "hold_B"


# ── State discretization ──────────────────────────────────────────────────────

def _state_bucket(tick_state: dict, perspective: str = "CT") -> tuple:
    """
    Map a tick state to a discrete bucket for win-rate aggregation.

    Returns a tuple that can be used as a dict key.
    The 5 dimensions capture the most predictive features while keeping
    the bucket space small enough to accumulate samples.
    """
    players = tick_state.get("players", [])
    ct_alive = sum(1 for p in players if p["team"] == "CT" and p["is_alive"])
    t_alive  = sum(1 for p in players if p["team"] == "T"  and p["is_alive"])

    # Time bucket: 0 = late round (< 30s), 1 = mid (30-60s), 2 = early (> 60s)
    t_rem = tick_state.get("time_remaining", 0)
    time_bucket = 0 if t_rem < 30 else (1 if t_rem < 60 else 2)

    # Player count ratio bucket
    alive_key = f"{ct_alive}v{t_alive}"

    # Where are most Ts?
    t_players = [p for p in players if p["team"] == "T" and p["is_alive"]]
    t_areas = [_area_of(p["x"], p["y"]) for p in t_players]
    a_side = sum(1 for a in t_areas if "A" in a)
    b_side = sum(1 for a in t_areas if "B" in a)
    t_pressure = "A" if a_side > b_side else ("B" if b_side > a_side else "mid")

    bomb_planted = int(tick_state.get("bomb_planted", False))

    return (time_bucket, alive_key, t_pressure, bomb_planted)


# ── Main class ───────────────────────────────────────────────────────────────

class WinRateLabeler:
    """
    Fits on a corpus of parsed rounds (from DemoParser) and annotates each tick
    with the empirically optimal action (highest CT win rate in that state bucket).

    Example
    -------
    >>> labeler = WinRateLabeler()
    >>> annotated = labeler.fit_transform(rounds_data)
    # Now each tick in annotated[i]["ticks"][j] has "optimal_action" key.
    """

    def __init__(self, perspective: str = "CT", smoothing: float = 1.0):
        self.perspective = perspective
        self.smoothing = smoothing  # Laplace smoothing to avoid 0/1 win rates

        # (bucket, action) → [wins, total]
        self._table: dict[tuple, dict[str, list]] = defaultdict(
            lambda: {a: [self.smoothing, 2 * self.smoothing] for a in ACTIONS}
        )
        self._fitted = False

    def fit(self, rounds_data: list[dict]) -> "WinRateLabeler":
        """
        Build the win-rate table from a corpus of rounds.

        Parameters
        ----------
        rounds_data : output of DemoParser.parse() — list of round dicts
        """
        for round_data in rounds_data:
            ticks = round_data.get("ticks", [])
            if not ticks:
                continue

            ct_won = self._infer_ct_won(ticks[-1])

            for i, tick_state in enumerate(ticks):
                bucket = _state_bucket(tick_state, self.perspective)

                # Detect action using look-ahead (only available in replay)
                prev_tick = ticks[max(0, i - 5)]
                future_tick = ticks[min(len(ticks) - 1, i + 10)]
                action = _detect_action(
                    tick_state.get("players", []),
                    prev_tick.get("players", []),
                    future_tick.get("players", []),
                    tick_state.get("events_this_tick", []),
                    self.perspective,
                )

                entry = self._table[bucket][action]
                entry[1] += 1           # total
                if ct_won:
                    entry[0] += 1       # wins (for CT perspective)

        self._fitted = True
        return self

    def transform(self, rounds_data: list[dict]) -> list[dict]:
        """
        Annotate each tick with the optimal action from the win-rate table.
        Returns a new list; does not mutate the input.
        """
        annotated = []
        for round_data in rounds_data:
            new_ticks = []
            ticks = round_data.get("ticks", [])
            for tick_state in ticks:
                tick_copy = dict(tick_state)
                bucket = _state_bucket(tick_copy, self.perspective)
                tick_copy["optimal_action"] = self._best_action(bucket)
                new_ticks.append(tick_copy)
            new_round = dict(round_data)
            new_round["ticks"] = new_ticks
            annotated.append(new_round)
        return annotated

    def fit_transform(self, rounds_data: list[dict]) -> list[dict]:
        return self.fit(rounds_data).transform(rounds_data)

    def win_rate_table(self) -> dict:
        """Return human-readable win-rate table for debugging."""
        result = {}
        for bucket, actions in self._table.items():
            result[str(bucket)] = {
                a: round(v[0] / v[1], 3) for a, v in actions.items()
            }
        return result

    # ── Internal ─────────────────────────────────────────────────────────────

    def _best_action(self, bucket: tuple) -> str:
        if not self._fitted or bucket not in self._table:
            return "hold_A"
        rates = {a: v[0] / v[1] for a, v in self._table[bucket].items()}
        return max(rates, key=rates.get)

    def _infer_ct_won(self, last_tick: dict) -> bool:
        players = last_tick.get("players", [])
        ct_alive = sum(1 for p in players if p["team"] == "CT" and p["is_alive"])
        t_alive  = sum(1 for p in players if p["team"] == "T"  and p["is_alive"])
        # CT wins if all Ts are dead (even if bomb was planted — defuse counts)
        return ct_alive > 0 and t_alive == 0
