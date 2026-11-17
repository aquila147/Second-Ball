"""Public Metrica sample games: loaders and rule-derived reference labels."""

import csv
import json
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from . import clean, io
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


def set_piece_rows(folder):
    """Rows of the SET PIECE events."""
    if os.path.exists(stem(folder) + "_events.json"):
        with open(stem(folder) + "_events.json") as handle:
            return np.array([e["start"]["frame"] - 1 for e in json.load(handle)["data"]
                             if e["type"]["name"] == "SET PIECE"])
    with open(stem(folder) + "_RawEventsData.csv") as handle:
        return np.array([int(e["Start Frame"]) - 1 for e in csv.DictReader(handle)
                         if e["Type"] == "SET PIECE"])


def reference_labels(folder, match, bridge=0):
    """Codes per frame: D where the ball is untracked, else the team of the last control event.

    Ball dropouts shorter than `bridge` frames that are not followed by a set piece within a
    second count as live play: almost none of them end in a restart.
    """
    n = len(match.ball)
    owner = np.full(n, -1, dtype=np.int8)
    for row, team in control_events(folder):
        owner[row] = team

    alive = ~np.isnan(match.ball[:, 0])
    if bridge:
        restarts = set_piece_rows(folder)
        for start, stop in clean.runs(~alive):
            if (start > 0 and stop < n and stop - start < bridge
                    and not np.any(np.abs(restarts - stop) <= 25)):
                alive[start:stop] = True
    labels = np.full(n, X, dtype=np.int8)
    for start, stop in match.halves:
        current = -1
        for frame in range(start, stop):
            if owner[frame] >= 0:
                current = owner[frame]
            labels[frame] = current if alive[frame] and current >= 0 else D
    return labels


def pass_events(folder):
    """[(row, team)] of the PASS events."""
    out = []
    if os.path.exists(stem(folder) + "_events.json"):
        home_id = next(ET.parse(stem(folder) + "_metadata.xml").getroot().iter("Team")).get("id")
        with open(stem(folder) + "_events.json") as handle:
            for event in json.load(handle)["data"]:
                if event["type"]["name"] == "PASS":
                    out.append((event["start"]["frame"] - 1,
                                HOME if event["team"]["id"] == home_id else AWAY))
    else:
        with open(stem(folder) + "_RawEventsData.csv") as handle:
            for event in csv.DictReader(handle):
                if event["Type"] == "PASS":
                    out.append((int(event["Start Frame"]) - 1,
                                HOME if event["Team"] == "Home" else AWAY))
    return out


def controlled(folder, labels, hold):
    """Labels where a change of possession needs control, following the challenge's proxies.

    A spell won by the other team counts only if that team keeps the ball for `hold` frames or
    completes more than two passes in it; otherwise possession stays with the previous team. The
    first spell after a stoppage always counts: a restart has a clear possessor.
    """
    events = pass_events(folder)
    rows = np.array([row for row, _ in events])
    teams = np.array([team for _, team in events])
    out = labels.copy()
    for start, stop in clean.runs(labels < 2):
        bounds = np.flatnonzero(np.diff(labels[start:stop]) != 0) + 1 + start
        edges = np.concatenate(([start], bounds, [stop]))
        current = labels[start]
        for a, b in zip(edges[1:-1], edges[2:]):
            team = labels[a]
            if team == current:
                continue
            passes = np.sum((rows >= a) & (rows < b) & (teams == team))
            if b - a >= hold or passes > 2:
                current = team
            else:
                out[a:b] = current
    return out


def load(folder, bridge=0):
    """(match, reference labels) for any of the three sample games."""
    if os.path.exists(stem(folder) + "_tracking.txt"):
        match = io.load_sample(folder)
    else:
        match = load_csv_game(folder)
    return match, reference_labels(folder, match, bridge)
