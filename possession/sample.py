"""Reference labels for sample game 3, derived from its tracking and events by rule."""

import json
import os
import xml.etree.ElementTree as ET

import numpy as np

from .io import AWAY, HOME

CONTROL_EVENTS = {"PASS", "CARRY", "RECOVERY", "SET PIECE", "SHOT"}
H, A, D, X = 0, 1, 2, 3


def reference_labels(folder, match):
    """Codes per frame: D where the ball is untracked, else the team of the last control event."""
    stem = os.path.join(folder, os.path.basename(os.path.normpath(folder)))
    with open(stem + "_events.json") as handle:
        events = json.load(handle)["data"]
    home_id = next(ET.parse(stem + "_metadata.xml").getroot().iter("Team")).get("id")

    n = len(match.ball)
    owner = np.full(n, -1, dtype=np.int8)
    for event in events:
        if event["type"]["name"] in CONTROL_EVENTS:
            team = HOME if event["team"]["id"] == home_id else AWAY
            owner[event["start"]["frame"] - 1] = team

    alive = ~np.isnan(match.ball[:, 0])
    labels = np.full(n, X, dtype=np.int8)
    for start, stop in match.halves:
        current = -1
        for frame in range(start, stop):
            if owner[frame] >= 0:
                current = owner[frame]
            if not alive[frame]:
                labels[frame] = D
            elif current >= 0:
                labels[frame] = current
            else:
                labels[frame] = D
    return labels
