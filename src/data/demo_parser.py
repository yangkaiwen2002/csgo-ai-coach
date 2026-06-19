"""
Demo Parser
-----------
Parses CS:GO .dem files using awpy and extracts structured game state data.
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import pandas as pd
from awpy import Demo


@dataclass
class PlayerState:
    """Snapshot of a single player at one tick."""
    tick: int
    steam_id: str
    name: str
    team: str          # CT or T
    x: float
    y: float
    z: float
    hp: int
    armor: int
    is_alive: bool
    has_helmet: bool
    has_defuser: bool
    active_weapon: str
    # What the player can actually observe (limited info)
    heard_gunfire: bool
    heard_footsteps: bool
    spotted_enemies: list[str]   # steam IDs of visible enemies


@dataclass
class RoundEvent:
    """A significant event in a round."""
    tick: int
    event_type: str    # kill, bomb_plant, smoke, flash, grenade, ...
    actor_id: str
    target_id: Optional[str]
    x: float
    y: float
    z: float
    weapon: Optional[str]
    map_area: str      # e.g. "A_site", "mid", "B_ramp"


@dataclass
class RoundState:
    """Full state of a round at a given tick."""
    round_num: int
    tick: int
    time_remaining: float
    phase: str              # buy, freezetime, live, over
    ct_score: int
    t_score: int
    bomb_planted: bool
    bomb_site: Optional[str]
    players: list[PlayerState]
    events_this_tick: list[RoundEvent]


class DemoParser:
    """
    Parses a .dem file into structured RoundState sequences.
    
    Example
    -------
    >>> parser = DemoParser("match.dem")
    >>> rounds = parser.parse()
    >>> parser.save(rounds, "data/parsed/match.json")
    """

    def __init__(self, demo_path: str, tick_rate: int = 64):
        self.demo_path = Path(demo_path)
        self.tick_rate = tick_rate
        self._demo: Optional[Demo] = None

    def parse(self, sample_every_n_ticks: int = 8) -> list[dict]:
        """
        Parse the demo and return a list of round state dicts.
        
        Parameters
        ----------
        sample_every_n_ticks : int
            Sample game state every N ticks to reduce data size.
            At 64-tick, every 8 ticks = ~8 samples/second.
        """
        print(f"[DemoParser] Loading {self.demo_path.name} ...")
        self._demo = Demo(str(self.demo_path))

        rounds_data = []

        # awpy exposes ticks as a DataFrame
        ticks_df: pd.DataFrame = self._demo.ticks
        kills_df: pd.DataFrame = self._demo.kills
        grenades_df: pd.DataFrame = self._demo.grenades

        for round_num, round_ticks in ticks_df.groupby("roundNum"):
            round_states = []
            sampled_ticks = round_ticks[
                round_ticks["tick"] % sample_every_n_ticks == 0
            ]

            for tick, tick_data in sampled_ticks.groupby("tick"):
                player_states = self._extract_player_states(tick, tick_data)
                events = self._extract_events(tick, kills_df, grenades_df)

                state = RoundState(
                    round_num=int(round_num),
                    tick=int(tick),
                    time_remaining=float(tick_data["timeRemaining"].iloc[0]),
                    phase=str(tick_data["phase"].iloc[0]),
                    ct_score=int(tick_data["ctScore"].iloc[0]),
                    t_score=int(tick_data["tScore"].iloc[0]),
                    bomb_planted=bool(tick_data["bombPlanted"].iloc[0]),
                    bomb_site=tick_data["bombSite"].iloc[0],
                    players=player_states,
                    events_this_tick=events,
                )
                round_states.append(asdict(state))

            rounds_data.append({
                "round_num": int(round_num),
                "ticks": round_states
            })

            print(f"  Round {round_num}: {len(round_states)} ticks parsed")

        print(f"[DemoParser] Done. {len(rounds_data)} rounds parsed.")
        return rounds_data

    def _extract_player_states(
        self, tick: int, tick_data: pd.DataFrame
    ) -> list[dict]:
        players = []
        for _, row in tick_data.iterrows():
            state = PlayerState(
                tick=int(tick),
                steam_id=str(row.get("steamID", "")),
                name=str(row.get("name", "")),
                team=str(row.get("side", "")),
                x=float(row.get("x", 0)),
                y=float(row.get("y", 0)),
                z=float(row.get("z", 0)),
                hp=int(row.get("hp", 0)),
                armor=int(row.get("armor", 0)),
                is_alive=bool(row.get("isAlive", False)),
                has_helmet=bool(row.get("hasHelmet", False)),
                has_defuser=bool(row.get("hasDefuser", False)),
                active_weapon=str(row.get("activeWeapon", "")),
                heard_gunfire=False,       # computed in feature extractor
                heard_footsteps=False,     # computed in feature extractor
                spotted_enemies=[],        # computed in feature extractor
            )
            players.append(asdict(state))
        return players

    def _extract_events(
        self,
        tick: int,
        kills_df: pd.DataFrame,
        grenades_df: pd.DataFrame,
        tick_window: int = 4
    ) -> list[dict]:
        """Collect events within ±tick_window of the current tick."""
        events = []

        # Kills
        nearby_kills = kills_df[
            (kills_df["tick"] >= tick - tick_window) &
            (kills_df["tick"] <= tick + tick_window)
        ]
        for _, kill in nearby_kills.iterrows():
            events.append(asdict(RoundEvent(
                tick=int(kill["tick"]),
                event_type="kill",
                actor_id=str(kill.get("attackerSteamID", "")),
                target_id=str(kill.get("victimSteamID", "")),
                x=float(kill.get("attackerX", 0)),
                y=float(kill.get("attackerY", 0)),
                z=float(kill.get("attackerZ", 0)),
                weapon=str(kill.get("weapon", "")),
                map_area=str(kill.get("mapArea", "")),
            )))

        return events

    def save(self, rounds_data: list[dict], output_path: str) -> None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(rounds_data, f, indent=2, ensure_ascii=False)
        print(f"[DemoParser] Saved to {out}")
