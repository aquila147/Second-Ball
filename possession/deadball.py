"""Evidence that play is stopped, as a log-likelihood ratio (positive = dead ball)."""

import numpy as np
from scipy.special import expit

from .io import FPS

OUT_MARGIN = 0.5      # metres beyond the line before the ball counts as out
W_OUT = 3.0
MOVING_SPEED = 2.0    # m/s; a ball moving faster than this on the pitch is being played
W_MOVING = 0.3
STILL_RAMP = 2 * FPS  # frames at rest before the stillness evidence is at full weight
W_STILL = 0.6
WALKING_SPEED = 1.3   # m/s mean player speed; between the stopped and live medians in the samples
W_SLOW = 0.25
W_LEAD_IN = 0.4
W_FORCED = 50.0

KICKOFF = "kick off (start)"

# Seconds of dead ball assumed before a tagged restart: about half the usual delay for each kind.
LEAD_IN = {"throw-in": 5, "goal kick": 12, "corner kick": 12, "free kick": 12,
           "kick off (after goal)": 30}


def set_pieces(match, n):
    """Lead-in before each tagged restart, and everything before each half's kick-off."""
    evidence = np.zeros(n)
    restarts = [tag for tag in match.tags if tag.code == "set pieces"]
    for tag in restarts:
        if tag.kind in LEAD_IN:
            evidence[max(tag.start - LEAD_IN[tag.kind] * FPS, 0):tag.start] += W_LEAD_IN
    for start, stop in match.halves:
        kickoffs = [tag.start for tag in restarts
                    if tag.kind == KICKOFF and start <= tag.stop and tag.start < stop]
        if kickoffs:
            evidence[start:min(kickoffs)] = W_FORCED
    return evidence


def dead_evidence(match, features):
    out = W_OUT * expit((features.outside - OUT_MARGIN) / 0.15)
    out = np.where(features.valid, out, 0.0)

    speed = np.nan_to_num(features.speed)
    on_pitch = features.valid & (features.outside == 0)
    moving = np.where(on_pitch, W_MOVING * expit((speed - MOVING_SPEED) / 0.5), 0.0)
    still = W_STILL * np.minimum(features.still / STILL_RAMP, 1.0)
    slow = W_SLOW * np.nan_to_num(np.tanh((WALKING_SPEED - features.player_speed) / 0.4))

    return out - moving + still + slow + set_pieces(match, len(out))
