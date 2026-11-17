"""Loaders for tracking, metadata and Smart Tagging."""

import difflib
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

FPS = 25
HOME, AWAY = 0, 1

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHALLENGE_REPO = os.environ.get("MTAH_REPO") or os.path.join(
    os.path.dirname(REPO), "more-than-a-hack-2027")

MISSING = ["nan", "nan"]


@dataclass
class Tag:
    code: str     # lower case
    kind: str     # lower-case Type tag, "" if absent
    team: int     # HOME, AWAY or -1
    start: int    # first frame
    stop: int     # first frame after the interval


@dataclass
class Match:
    players: np.ndarray    # (F, S, 2) metres, NaN where untracked
    ball: np.ndarray       # (F, 2) metres, NaN where untracked
    slot_team: np.ndarray  # (S,) HOME or AWAY
    halves: list           # [(first_row, stop_row)], half-open
    pitch: tuple           # (length, width) metres
    tags: list             # Smart Tagging intervals, empty if there are none


def read_tracking(path, n_frames=None):
    """Normalised player (F, S, 2) and ball (F, 2) positions, row = frame_id - 1."""
    rows, frames = [], []
    with open(path) as handle:
        for line in handle:
            parts = line.rstrip().split(":")
            if len(parts) != 3:
                continue
            groups = parts[1].split(";")
            groups.append(parts[2])
            rows.append(np.array(
                [MISSING if g[0] == "N" else g.split(",")[:2] for g in groups], dtype=float))
            frames.append(int(parts[0]) - 1)

    frames = np.array(frames)
    data = np.full((n_frames or frames.max() + 1,) + rows[0].shape, np.nan)
    data[frames] = np.array(rows)
    return data[:, :-1], data[:, -1]


def slot_teams(root, n_slots):
    """HOME or AWAY for each slot of a tracking line. The first Team in the file is home."""
    home_id = next(root.iter("Team")).get("id")
    players = [(p.get("id"), p.get("teamId")) for p in root.iter("Player")]
    if len(players) == n_slots:
        teams = [team for _, team in players]
    else:
        # Sample games list every squad player; the slot order is in the format specification.
        team_of = dict(players)
        player_of = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
        spec = next(root.iter("DataFormatSpecification"))
        refs = [ref.get("playerChannelId") for ref in spec.iter("PlayerChannelRef")]
        teams = [team_of[player_of[ref]] for ref in refs[::2]]
    return np.array([HOME if team == home_id else AWAY for team in teams], dtype=np.int8)


def tag_team(value, names):
    """Map a Smart Tagging Team value to HOME or AWAY by name similarity, -1 for N/A."""
    if not value or value.strip().upper() == "N/A":
        return -1
    scores = [difflib.SequenceMatcher(None, value.lower(), name.lower()).ratio()
              for name in names]
    return HOME if scores[0] >= scores[1] else AWAY


def load_match():
    """The challenge match from MTAH_DATA_DIR, using the challenge repo's own helpers."""
    sys.path.insert(0, CHALLENGE_REPO)
    import games
    from starter import load_smart_tagging

    meta_path, raw_path = games.require(["atd_meta", "atd_raw"])
    pitch = games.GAMES[games.GAME]["pitch"]
    players, ball = read_tracking(raw_path, games.source_frames("tactical"))

    root = ET.parse(meta_path).getroot()
    names = [team.findtext("Name") for team in root.iter("Team")]
    tags = [Tag((t["code"] or "").strip().lower(),
                (t["tags"].get("Type") or "").strip().lower(),
                tag_team(t["tags"].get("Team"), names),
                t["start_frame"], t["end_frame"])
            for t in load_smart_tagging.load()]

    halves = [(first, last + 1) for _, first, last in games.half_video_blocks(games.GAME)]
    return Match(players * pitch, ball * pitch, slot_teams(root, players.shape[1]),
                 halves, pitch, tags)


def load_sample(folder):
    """Public sample game 3 (EPTS text), same structure as the match."""
    stem = os.path.join(folder, os.path.basename(os.path.normpath(folder)))
    root = ET.parse(stem + "_metadata.xml").getroot()
    params = {p.findtext("Name"): p.findtext("Value") for p in root.iter("ProviderParameter")}
    halves = [(int(params["first_half_start"]) - 1, int(params["first_half_end"])),
              (int(params["second_half_start"]) - 1, int(params["second_half_end"]))]
    pitch = (float(root.findtext(".//FieldSize/Width")),
             float(root.findtext(".//FieldSize/Height")))

    players, ball = read_tracking(stem + "_tracking.txt")
    return Match(players * pitch, ball * pitch, slot_teams(root, players.shape[1]),
                 halves, pitch, [])
