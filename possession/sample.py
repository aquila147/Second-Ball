"""Public Metrica sample games: loaders and rule-derived reference labels."""

import csv
import json
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from . import io
from .io import AWAY, HOME

CONTROL_EVENTS = {"PASS", "CARRY", "RECOVERY", "SET PIECE", "SHOT"}
PITCH = (105.0, 68.0)
H, A, D, X = 0, 1, 2, 3


def stem(folder):
    return os.path.join(folder, os.path.basename(os.path.normpath(folder)))


def load_csv_game(folder):
    """Sample games 1 and 2: one tracking CSV per team, ball in the last two columns."""
    tables = [pd.read_csv("{}_RawTrackingData_{}_Team.csv".format(stem(folder), side),
                          skiprows=3, header=None).to_numpy(dtype=float)
              for side in ("Home", "Away")]
    period = tables[0][:, 0]
    sides = [table[:, 3:-2].reshape(len(table), -1, 2) for table in tables]
    players = np.concatenate(sides, axis=1)
    slot_team = np.repeat([HOME, AWAY], [side.shape[1] for side in sides]).astype(np.int8)
    ball = tables[0][:, -2:]

    second = int(np.argmax(period == 2))
    halves = [(0, second), (second, len(period))]
    return io.Match(players * PITCH, ball * PITCH, slot_team, halves, PITCH, [])


def control_events(folder):
    """[(row, team)] for the events that show a team in control of the ball."""
    out = []
    if os.path.exists(stem(folder) + "_events.json"):
        home_id = next(ET.parse(stem(folder) + "_metadata.xml").getroot().iter("Team")).get("id")
        with open(stem(folder) + "_events.json") as handle:
            for event in json.load(handle)["data"]:
                if event["type"]["name"] in CONTROL_EVENTS:
                    team = HOME if event["team"]["id"] == home_id else AWAY
                    out.append((event["start"]["frame"] - 1, team))
    else:
        with open(stem(folder) + "_RawEventsData.csv") as handle:
            for event in csv.DictReader(handle):
                if event["Type"] in CONTROL_EVENTS:
                    team = HOME if event["Team"] == "Home" else AWAY
                    out.append((int(event["Start Frame"]) - 1, team))
    return out


def reference_labels(folder, match):
    """Codes per frame: D where the ball is untracked, else the team of the last control event."""
    n = len(match.ball)
    owner = np.full(n, -1, dtype=np.int8)
    for row, team in control_events(folder):
        owner[row] = team

    alive = ~np.isnan(match.ball[:, 0])
    labels = np.full(n, X, dtype=np.int8)
    for start, stop in match.halves:
        current = -1
        for frame in range(start, stop):
            if owner[frame] >= 0:
                current = owner[frame]
            labels[frame] = current if alive[frame] and current >= 0 else D
    return labels


def load(folder):
    """(match, reference labels) for any of the three sample games."""
    if os.path.exists(stem(folder) + "_tracking.txt"):
        match = io.load_sample(folder)
    else:
        match = load_csv_game(folder)
    return match, reference_labels(folder, match)
