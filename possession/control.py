"""Evidence for which team has the ball, as a log-likelihood ratio (positive = home)."""

import numpy as np
from scipy.special import expit

from . import clean
from .io import AWAY, HOME

NEAR = 2.5               # metres; a ball further than this from everyone is loose or in flight
MAX_CONTROL_SPEED = 12   # m/s; a faster ball passing a player is not under control
W_NEAR = 0.6             # per frame with the ball at one team's feet
W_TOUCH = 2.0            # per detected touch
W_DRIFT = 0.05           # per frame of both teams moving one way at 1.5 m/s or more
W_PHASE = 0.15           # per frame inside a Smart Tagging phase or set piece

ATTACKING_CODES = {"build up", "progression", "creating chances", "ball in final third",
                   "ball in the box", "players in the box", "long ball", "attacking transition",
                   "set pieces"}
DEFENDING_CODES = {"defending in defensive third", "defending in middle third",
                   "defending in attacking third", "defensive transition"}


def proximity(features):
    """Favour the team whose player is nearer while the ball is close to someone and not flying."""
    home, away = features.dist[:, 0], features.dist[:, 1]
    closeness = expit((NEAR - np.minimum(home, away)) / 0.5)
    with np.errstate(invalid="ignore"):
        advantage = np.nan_to_num(np.tanh(away - home))
    slow = expit((MAX_CONTROL_SPEED - np.nan_to_num(features.speed)) / 2.0)
    return np.where(features.valid, W_NEAR * closeness * advantage * slow, 0.0)


def drift(match, features):
    """Both teams moving towards one goal suggests the team attacking that goal has the ball."""
    out = np.zeros(len(features.drift))
    for half in match.halves:
        sign = clean.attack_sign(match.players, match.slot_team, half)
        d = np.nan_to_num(features.drift[half[0]:half[1]])
        out[half[0]:half[1]] = sign * np.sign(d) * np.clip(np.abs(d) - 0.5, 0.0, 1.0)
    return W_DRIFT * out


def phases(match, n):
    """Team named by the Smart Tagging intervals covering each frame."""
    votes = np.zeros(n)
    for tag in match.tags:
        if tag.team < 0:
            continue
        side = 1 if tag.team == HOME else -1
        if tag.code in ATTACKING_CODES:
            votes[tag.start:tag.stop] += side
        elif tag.code in DEFENDING_CODES:
            votes[tag.start:tag.stop] -= side
    return W_PHASE * np.clip(votes, -1, 1)


def team_evidence(match, features, touch_frames, touch_teams):
    evidence = proximity(features) + drift(match, features) + phases(match, len(features.valid))
    np.add.at(evidence, touch_frames, np.where(touch_teams == HOME, W_TOUCH, -W_TOUCH))
    return evidence
