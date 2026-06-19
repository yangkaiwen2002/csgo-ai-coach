"""
Fake Data Generator
-------------------
Generates synthetic round sequences in the exact same format as DemoParser.parse().
Lets the full pipeline (features → train → analyze) run without a real .dem file.

Four pre-defined scenarios cover the most common tactical situations in CS:GO,
making the generated data useful for smoke-testing model behaviour.
"""

import copy
import math
import random
from typing import Optional
import numpy as np


# Approximate de_dust2 world coordinates
DUST2 = {
    "T_spawn":   (-2000.0, -1500.0),
    "A_long":    (-1500.0,   200.0),
    "A_short":   ( -300.0,  1000.0),
    "A_site":    ( -600.0,   800.0),
    "mid":       (    0.0,  -200.0),
    "B_tunnels": (  800.0, -1000.0),
    "B_site":    ( 1500.0,   100.0),
    "CT_spawn":  ( 1000.0,  1500.0),
}

RIFLES  = ["AK-47", "M4A4", "M4A1", "AWP", "SG 553", "AUG"]
PISTOLS = ["Glock-18", "USP-S", "P250", "Tec-9"]


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _make_player(
    team: str,
    idx: int,
    start_pos: str,
    weapon: str = "AK-47",
    has_rifle: bool = True,
) -> dict:
    x, y = DUST2[start_pos]
    return {
        "steam_id":    f"{team}_{idx}",
        "name":        f"Player_{team}_{idx}",
        "team":        team,
        "x":           x + random.gauss(0, 40),
        "y":           y + random.gauss(0, 40),
        "z":           0.0,
        "hp":          100,
        "armor":       100,
        "is_alive":    True,
        "has_helmet":  True,
        "has_defuser": team == "CT",
        "active_weapon": weapon,
        "heard_gunfire":  False,
        "heard_footsteps": False,
        "spotted_enemies": [],
    }


def _move_toward(player: dict, target_area: str, speed: float = 0.08) -> None:
    tx, ty = DUST2[target_area]
    player["x"] = _lerp(player["x"], tx, speed) + random.gauss(0, 12)
    player["y"] = _lerp(player["y"], ty, speed) + random.gauss(0, 12)


def _make_kill_event(tick: int, area: str, weapon: str) -> dict:
    x, y = DUST2[area]
    return {
        "tick":        tick,
        "event_type":  "kill",
        "actor_id":    "unknown",
        "target_id":   "unknown",
        "x":           x + random.gauss(0, 30),
        "y":           y + random.gauss(0, 30),
        "z":           0.0,
        "weapon":      weapon,
        "map_area":    area,
    }


def _make_tick_state(
    round_num: int,
    tick: int,
    time_remaining: float,
    ct_score: int,
    t_score: int,
    bomb_planted: bool,
    bomb_site: Optional[str],
    players: list,
    events: list,
    optimal_action: str,
) -> dict:
    return {
        "round_num":    round_num,
        "tick":         tick,
        "time_remaining": time_remaining,
        "phase":        "live",
        "ct_score":     ct_score,
        "t_score":      t_score,
        "bomb_planted": bomb_planted,
        "bomb_site":    bomb_site,
        "players":      [dict(p) for p in players],   # snapshot, not reference
        "events_this_tick": events,
        "optimal_action": optimal_action,             # used by FeatureExtractor._infer_label
    }


# ── Scenarios ─────────────────────────────────────────────────────────────────

def scenario_b_rush(round_num: int, ct_score: int, t_score: int) -> dict:
    """
    All 5 Ts rush B tunnels → B site.
    CTs have 2 on B, 3 on A.
    Optimal CT action: rotate_B (rotate the 3 from A to help B).
    Ts typically win if CTs don't rotate in time.
    """
    t_players  = [_make_player("T",  i, "T_spawn",   random.choice(RIFLES)) for i in range(5)]
    ct_players = [_make_player("CT", i, "CT_spawn",  random.choice(RIFLES)) for i in range(5)]
    # Pre-position 2 CTs at B, 3 at A
    for p in ct_players[:2]:
        p["x"], p["y"] = DUST2["B_site"]
        p["x"] += random.gauss(0, 40)
        p["y"] += random.gauss(0, 40)
    for p in ct_players[2:]:
        p["x"], p["y"] = DUST2["A_site"]
        p["x"] += random.gauss(0, 40)
        p["y"] += random.gauss(0, 40)

    ticks = []
    n_ticks = 64  # ~16s at 4 samples/sec after sample_every_8_ticks

    for i in range(n_ticks):
        tick_id = i * 8
        time_rem = max(115.0 - i * 1.8, 0.0)

        for p in t_players:
            if p["is_alive"]:
                _move_toward(p, "B_tunnels" if i < 20 else "B_site", speed=0.10)

        # CTs hold their positions — NOT rotating (this is the mistake)
        for p in ct_players[:2]:
            if p["is_alive"]:
                p["x"] += random.gauss(0, 5)
                p["y"] += random.gauss(0, 5)
        for p in ct_players[2:]:
            if p["is_alive"]:
                p["x"] += random.gauss(0, 5)
                p["y"] += random.gauss(0, 5)

        events = []
        if i > 45:
            # 5 Ts overwhelm 2 CTs at B
            for p in ct_players[:2]:
                p["is_alive"] = False
            events.append(_make_kill_event(tick_id, "B_site", "AK-47"))

        ticks.append(_make_tick_state(
            round_num, tick_id, time_rem, ct_score, t_score,
            False, None, ct_players + t_players, events,
            optimal_action="rotate_B",
        ))

    return {"round_num": round_num, "ticks": ticks}


def scenario_a_split(round_num: int, ct_score: int, t_score: int) -> dict:
    """
    3 Ts push A long + 2 Ts push A short simultaneously.
    CTs have 3 at A site, 2 at B.
    Optimal CT action: hold_A (don't let Ts take A unchallenged).
    """
    t_long  = [_make_player("T", i,   "A_long",  random.choice(RIFLES)) for i in range(3)]
    t_short = [_make_player("T", i+3, "A_short", random.choice(RIFLES)) for i in range(2)]
    ct_players = [_make_player("CT", i, "CT_spawn", random.choice(RIFLES)) for i in range(5)]
    for p in ct_players[:3]:
        p["x"], p["y"] = DUST2["A_site"]
        p["x"] += random.gauss(0, 40)
        p["y"] += random.gauss(0, 40)
    for p in ct_players[3:]:
        p["x"], p["y"] = DUST2["B_site"]
        p["x"] += random.gauss(0, 40)
        p["y"] += random.gauss(0, 40)

    ticks = []
    for i in range(60):
        tick_id = i * 8
        time_rem = max(115.0 - i * 1.9, 0.0)

        for p in t_long:
            if p["is_alive"]:
                _move_toward(p, "A_site", speed=0.07)
        for p in t_short:
            if p["is_alive"]:
                _move_toward(p, "A_site", speed=0.09)

        for p in ct_players:
            if p["is_alive"]:
                p["x"] += random.gauss(0, 4)
                p["y"] += random.gauss(0, 4)

        events = []
        if i > 42:
            ct_players[0]["is_alive"] = False
            events.append(_make_kill_event(tick_id, "A_site", "AK-47"))

        ticks.append(_make_tick_state(
            round_num, tick_id, time_rem, ct_score, t_score,
            False, None, ct_players + t_long + t_short, events,
            optimal_action="hold_A",
        ))

    return {"round_num": round_num, "ticks": ticks}


def scenario_ct_dominant(round_num: int, ct_score: int, t_score: int) -> dict:
    """
    CTs have 5v3 advantage (2 Ts died early in previous engagement).
    CTs can choose to retake/push or hold passively.
    Optimal action: hold_A or hold_B (passive win condition).
    """
    t_players  = [_make_player("T",  i, "T_spawn",  random.choice(RIFLES)) for i in range(3)]
    ct_players = [_make_player("CT", i, "CT_spawn", random.choice(RIFLES)) for i in range(5)]
    for i, p in enumerate(ct_players):
        area = "A_site" if i < 3 else "B_site"
        p["x"], p["y"] = DUST2[area]

    ticks = []
    for i in range(55):
        tick_id = i * 8
        time_rem = max(115.0 - i * 2.0, 0.0)

        for p in t_players:
            if p["is_alive"]:
                _move_toward(p, "mid", speed=0.04)

        for p in ct_players:
            if p["is_alive"]:
                p["x"] += random.gauss(0, 4)
                p["y"] += random.gauss(0, 4)

        events = []
        if i > 48:
            for p in t_players:
                p["is_alive"] = False
            events.append(_make_kill_event(tick_id, "mid", "M4A4"))

        optimal = "hold_A" if random.random() < 0.5 else "hold_B"
        ticks.append(_make_tick_state(
            round_num, tick_id, time_rem, ct_score, t_score,
            False, None, ct_players + t_players, events,
            optimal_action=optimal,
        ))

    return {"round_num": round_num, "ticks": ticks}


def scenario_eco_round(round_num: int, ct_score: int, t_score: int) -> dict:
    """
    Ts are on full eco (pistols). CTs have rifles + armor.
    Ts rush B hoping to surprise; CTs should hold B aggressively.
    Optimal CT action: hold_B.
    """
    t_players  = [_make_player("T",  i, "T_spawn",  random.choice(PISTOLS), has_rifle=False)
                  for i in range(5)]
    for p in t_players:
        p["armor"] = 0
        p["has_helmet"] = False
        p["active_weapon"] = random.choice(PISTOLS)
    ct_players = [_make_player("CT", i, "CT_spawn", random.choice(RIFLES)) for i in range(5)]
    for p in ct_players:
        p["x"], p["y"] = DUST2["B_site"]
        p["x"] += random.gauss(0, 50)
        p["y"] += random.gauss(0, 50)

    ticks = []
    for i in range(50):
        tick_id = i * 8
        time_rem = max(115.0 - i * 2.2, 0.0)

        for p in t_players:
            if p["is_alive"]:
                _move_toward(p, "B_site", speed=0.12)

        for p in ct_players:
            if p["is_alive"]:
                p["x"] += random.gauss(0, 5)
                p["y"] += random.gauss(0, 5)

        events = []
        if i > 35:
            for p in t_players:
                p["is_alive"] = False
            events.append(_make_kill_event(tick_id, "B_site", "M4A4"))

        ticks.append(_make_tick_state(
            round_num, tick_id, time_rem, ct_score, t_score,
            False, None, ct_players + t_players, events,
            optimal_action="hold_B",
        ))

    return {"round_num": round_num, "ticks": ticks}


# ── Public API ────────────────────────────────────────────────────────────────

SCENARIO_FNS = [
    scenario_b_rush,
    scenario_a_split,
    scenario_ct_dominant,
    scenario_eco_round,
]

SCENARIO_NAMES = {
    scenario_b_rush:      "B Site Rush",
    scenario_a_split:     "A Split",
    scenario_ct_dominant: "CT Dominant",
    scenario_eco_round:   "Eco Round",
}


def generate_fake_dataset(n_rounds: int = 20, seed: int = 42) -> list[dict]:
    """
    Generate n_rounds of synthetic CS:GO data.
    Each round is one of the four pre-defined scenarios, chosen cyclically
    so the model sees a balanced mix of all action types.

    Returns data in the exact same format as DemoParser.parse().
    """
    random.seed(seed)
    np.random.seed(seed)

    ct_score = t_score = 0
    rounds = []
    for i in range(n_rounds):
        fn = SCENARIO_FNS[i % len(SCENARIO_FNS)]
        round_data = fn(round_num=i + 1, ct_score=ct_score, t_score=t_score)
        rounds.append(round_data)
        # Determine winner by checking last tick's alive counts
        last_tick = round_data["ticks"][-1]
        ct_alive = sum(1 for p in last_tick["players"] if p["team"] == "CT" and p["is_alive"])
        t_alive  = sum(1 for p in last_tick["players"] if p["team"] == "T"  and p["is_alive"])
        if ct_alive > 0 and t_alive == 0:
            ct_score += 1
        else:
            t_score += 1

    return rounds
